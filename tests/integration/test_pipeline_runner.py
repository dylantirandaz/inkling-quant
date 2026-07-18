"""CPU-only end-to-end coverage for governed experiment execution and resume."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from shlex import join as shell_join
from typing import Any

import pytest

from inkling_quant_lab.artifacts import sha256_file
from inkling_quant_lab.config import ExperimentConfig, load_config
from inkling_quant_lab.exceptions import ArtifactIntegrityError, CapabilityError, InklingQuantError
from inkling_quant_lab.manifests import RunStatus, StageStatus, load_manifest
from inkling_quant_lab.pipeline.resume import verify_successful_stage_outputs
from inkling_quant_lab.pipeline.runner import resume_experiment, run_experiment
from inkling_quant_lab.pipeline.stages import STAGE_ORDER, stage_descendants

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = PROJECT_ROOT / "configs" / "experiments"
FORCED_QUANTIZE_STAGES = stage_descendants("quantize", include_self=True)


def _fast_config(name: str, artifact_root: Path) -> ExperimentConfig:
    return load_config(
        EXPERIMENTS / name,
        (
            f"output.root={json.dumps(str(artifact_root))}",
            "benchmark.warmup_iterations=0",
            "benchmark.repetitions=1",
        ),
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_completed_run(source: Path, destination: Path) -> Path:
    return Path(shutil.copytree(source, destination))


@pytest.fixture(scope="module")
def completed_moe_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    artifact_root = tmp_path_factory.mktemp("pipeline-runner-moe") / "artifacts"
    config = _fast_config("tiny_moe_int8.yaml", artifact_root)
    return run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="completed-moe-int8",
    )


def test_tiny_moe_int8_run_satisfies_artifact_routing_and_report_contract(
    completed_moe_run: Path,
) -> None:
    manifest = load_manifest(completed_moe_run)
    resolved = load_config(completed_moe_run / "resolved_config.yaml")

    assert manifest.status is RunStatus.SUCCESS
    assert set(manifest.stages) == set(STAGE_ORDER)
    assert len(manifest.stages) == len(STAGE_ORDER)
    assert all(stage.status is StageStatus.SUCCESS for stage in manifest.stages.values())
    assert manifest.config_hash == resolved.config_hash()
    assert manifest.model.id == "local://fixtures/tiny-moe"
    assert manifest.model.revision == "tiny-moe-fixture-v1"
    assert manifest.model.resolved_class is not None
    assert manifest.model.checksum is not None
    assert manifest.environment["hardware"]["device"] == "cpu"
    assert manifest.environment["capabilities"]["model"]["is_moe"] is True
    assert manifest.environment["distributed"]["gloo"]["supports_model_sharding"] is False
    assert manifest.environment["distributed"]["gloo"]["status"] in {
        "available",
        "unavailable",
    }
    verify_successful_stage_outputs(completed_moe_run, manifest)

    required_files = (
        "manifest.json",
        "resolved_config.yaml",
        "environment.json",
        "events.jsonl",
        "status.json",
        "completion.json",
        "checkpoints/baseline/descriptor.json",
        "checkpoints/policy/resolved_policy.json",
        "checkpoints/candidate/quantization_manifest.json",
        "checkpoints/candidate/candidate/recipe.json",
        "checkpoints/candidate/candidate/weights.safetensors",
        "metrics/evaluation_baseline/results.json",
        "metrics/evaluation_candidate/results.json",
        "metrics/benchmark_baseline/benchmark.json",
        "metrics/benchmark_candidate/benchmark.json",
        "routing/baseline/aggregates.json",
        "routing/candidate/aggregates.json",
        "routing/comparison.json",
        "reports/baseline_summary.json",
        "reports/candidate_summary.json",
        "reports/comparison.json",
        "reports/pareto.json",
        "reports/report_data.json",
        "reports/report.md",
        "reports/tables/metrics.csv",
        "reports/tables/comparisons.csv",
        "reports/tables/pareto.csv",
        "reports/tables/routing_drift.csv",
    )
    assert all((completed_moe_run / relative).is_file() for relative in required_files)
    assert _read_json(completed_moe_run / "status.json")["status"] == "success"

    quantization = _read_json(
        completed_moe_run / "checkpoints/candidate/quantization_manifest.json"
    )
    precision_map = quantization["module_precision_map"]
    assert quantization["backend"] == "torch_dynamic_int8"
    assert any(
        ".experts." in module and precision == "int8" for module, precision in precision_map.items()
    )
    assert all(
        precision == "float32"
        for module, precision in precision_map.items()
        if module.endswith(".router")
    )

    for kind in ("baseline", "candidate"):
        benchmark = _read_json(completed_moe_run / f"metrics/benchmark_{kind}/benchmark.json")
        assert benchmark["warmup_iterations"] == 0
        assert benchmark["repetitions"] == 1
        assert len(benchmark["trials"]) == 1
        assert benchmark["model_load_time_ms"] > 0.0
        assert benchmark["model_load_time_kind"] == (
            "cold_model_load" if kind == "baseline" else "candidate_reconstruction"
        )
        assert benchmark["hardware_utilization"]["status"] == "available"
        assert benchmark["hardware_utilization"]["sampling_interval_seconds"] > 0.0
        workload = benchmark["workload"]
        assert workload["dataset_id"] == "local://fixtures/generation-prompts"
        assert workload["dataset_revision"] == "fixture-data-v1"
        assert workload["split"] == "evaluation"
        assert len(workload["dataset_sha256"]) == 64
        assert workload["sample_ids"] == ["eval-gen-001", "eval-gen-002", "eval-gen-003"]
        assert workload["seed"] == 17
        assert len(workload["prompt_template_hash"]) == 64
        assert workload["decode_config"]["max_new_tokens"] == 4
        assert workload["execution_mode"] == "sequential_samples"
        peak_memory = benchmark["peak_memory"]
        assert peak_memory["host_measurement_kind"] == "benchmark_interval_sampled_current_rss"
        assert "not a continuous or allocator-native peak" in peak_memory["host_scope"]
        assert peak_memory["host_baseline_bytes"] > 0
        assert peak_memory["host_max_observed_delta_bytes"] >= 0
        assert peak_memory["host_sample_count"] == 2
        assert peak_memory["measurement_interval_seconds"] > 0.0
        assert "peak_host_memory_bytes" not in benchmark["trials"][0]
        assert benchmark["trials"][0]["host_memory_bytes_at_post_trial_sample"] > 0

    exported_bundle = completed_moe_run / "checkpoints/candidate/candidate"
    assert quantization["serialized_size_bytes"] == sum(
        path.stat().st_size for path in exported_bundle.iterdir()
    )

    routing = _read_json(completed_moe_run / "routing/comparison.json")
    assert routing["macro"]["layer_count"] == 2
    assert routing["macro"]["token_weight"] > 0
    assert len(routing["per_layer"]) == 2
    assert set(routing["per_layer_drift_ranking"]) == {"moe_0", "moe_1"}
    for layer in routing["per_layer"]:
        assert layer["expert_count"] == 4
        assert layer["baseline_token_count"] > 0
        assert layer["candidate_token_count"] > 0

    candidate_summary = _read_json(completed_moe_run / "reports/candidate_summary.json")
    for metric in (
        "routing_js_divergence",
        "routing_load_imbalance",
        "routing_entropy",
    ):
        assert candidate_summary["metrics"][metric]["status"] == "available"
        assert candidate_summary["metrics"][metric]["value"] is not None
    for metric in ("route_agreement", "top_k_overlap"):
        assert candidate_summary["metrics"][metric]["status"] == "unavailable"
        assert candidate_summary["metrics"][metric]["value"] is None
        assert "aligned token traces" in candidate_summary["metrics"][metric]["reason"]

    assert candidate_summary["seed_set"] == [17]
    assert all(len(dataset["dataset_sha256"]) == 64 for dataset in candidate_summary["datasets"])
    assert len(candidate_summary["routing_dataset"]["dataset_sha256"]) == 64
    assert candidate_summary["routing_sample_ids"]
    capture = candidate_summary["routing_capture"]
    assert capture["configured_mode"] == "aggregate"
    assert capture["captured_mode"] == "aggregate"
    assert capture["recorded_event_count"] == 0
    assert capture["alignment_key_count"] == 0
    assert capture["alignment_key_sha256"] is None
    assert candidate_summary["hardware_environment"]["hardware"]["device"] == "cpu"
    scoped_peak = candidate_summary["metrics"]["peak_host_memory_bytes"]
    assert scoped_peak["status"] == "unavailable"
    assert scoped_peak["value"] is None
    assert "sampled only" in scoped_peak["reason"]
    lifetime_rss = candidate_summary["metrics"]["process_lifetime_peak_host_rss_bytes"]
    assert lifetime_rss["status"] == "unavailable"
    for metric in (
        "benchmark_interval_baseline_current_rss_bytes",
        "benchmark_interval_max_sampled_current_rss_bytes",
        "benchmark_interval_sampled_current_rss_delta_bytes",
        "benchmark_memory_sample_count",
        "benchmark_memory_interval_seconds",
    ):
        assert candidate_summary["metrics"][metric]["status"] == "available"
        assert candidate_summary["metrics"][metric]["value"] >= 0
        assert candidate_summary["metrics"][metric]["direction"] == "neutral"
    assert candidate_summary["benchmark_workload"]["sample_ids"] == [
        "eval-gen-001",
        "eval-gen-002",
        "eval-gen-003",
    ]
    assert (
        candidate_summary["benchmark_memory"]["host_measurement_kind"]
        == "benchmark_interval_sampled_current_rss"
    )
    assert (
        "not a continuous or allocator-native peak"
        in candidate_summary["benchmark_memory"]["host_scope"]
    )
    assert not (completed_moe_run / "routing/baseline/traces.jsonl").exists()
    assert not (completed_moe_run / "routing/candidate/traces.jsonl").exists()

    routing_table = (completed_moe_run / "reports/tables/routing_drift.csv").read_text(
        encoding="utf-8"
    )
    assert "layer_id,js_divergence,top_k_overlap,route_agreement" in routing_table
    assert "moe_0" in routing_table
    assert "moe_1" in routing_table
    assert "## Routing Layer Detail" in (completed_moe_run / "reports/report.md").read_text(
        encoding="utf-8"
    )
    report_markdown = (completed_moe_run / "reports/report.md").read_text(encoding="utf-8")
    assert "Benchmark workload" in report_markdown
    assert "eval-gen-001,eval-gen-002,eval-gen-003" in report_markdown
    assert "benchmark_interval_sampled_current_rss" in report_markdown
    assert "not a continuous or allocator-native peak" in report_markdown
    assert "benchmark_interval_sampled_current_rss_delta_bytes" in report_markdown

    for plot in (
        "quality_vs_memory",
        "quality_vs_latency",
        "quality_vs_throughput",
    ):
        assert (completed_moe_run / f"reports/plots/{plot}.svg").is_file()
        assert (completed_moe_run / f"reports/plots/{plot}.csv").is_file()


def test_full_trace_run_retains_exact_alignment_evidence_for_token_agreement(
    tmp_path: Path,
) -> None:
    """Token agreement is reportable only when its raw alignment evidence persists."""

    config = load_config(
        EXPERIMENTS / "tiny_moe_int8.yaml",
        (
            f"output.root={json.dumps(str(tmp_path / 'artifacts'))}",
            "benchmark.warmup_iterations=0",
            "benchmark.repetitions=1",
            "routing.mode=full_trace",
        ),
    )
    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="full-trace-evidence",
    )

    baseline = _read_json(run_directory / "reports/baseline_summary.json")
    candidate = _read_json(run_directory / "reports/candidate_summary.json")
    for summary in (baseline, candidate):
        capture = summary["routing_capture"]
        assert capture["configured_mode"] == "full_trace"
        assert capture["captured_mode"] == "full_trace"
        assert capture["recorded_event_count"] > 0
        assert capture["alignment_key_count"] == capture["recorded_event_count"]
        assert len(capture["alignment_key_sha256"]) == 64
    assert (
        baseline["routing_capture"]["alignment_key_sha256"]
        == candidate["routing_capture"]["alignment_key_sha256"]
    )
    assert candidate["metrics"]["route_agreement"]["status"] == "available"
    assert (run_directory / "routing/baseline/traces.jsonl").is_file()
    assert (run_directory / "routing/candidate/traces.jsonl").is_file()


def test_failure_after_quantization_resumes_without_rewriting_successes(tmp_path: Path) -> None:
    config = _fast_config("tiny_moe_int8.yaml", tmp_path / "artifacts")
    run_directory = tmp_path / "artifacts" / "failure-resume"

    with pytest.raises(InklingQuantError, match="evaluate_candidate"):
        run_experiment(
            config,
            project_root=PROJECT_ROOT,
            run_id="failure-resume",
            inject_failure_stage="evaluate_candidate",
        )

    failed = load_manifest(run_directory)
    assert failed.status is RunStatus.FAILED
    assert failed.stages["quantize"].status is StageStatus.SUCCESS
    assert failed.stages["evaluate_baseline"].status is StageStatus.SUCCESS
    assert failed.stages["evaluate_candidate"].status is StageStatus.FAILED
    assert failed.stages["evaluate_candidate"].attempt == 1
    assert not (run_directory / "metrics/evaluation_candidate").exists()
    assert (run_directory / "failures/evaluate_candidate-attempt-1.json").is_file()
    preserved = {
        name: record
        for name, record in failed.stages.items()
        if record.status is StageStatus.SUCCESS
    }

    assert resume_experiment(run_directory, project_root=PROJECT_ROOT) == run_directory.resolve()

    resumed = load_manifest(run_directory)
    assert resumed.status is RunStatus.SUCCESS
    assert resumed.stages["evaluate_candidate"].status is StageStatus.SUCCESS
    assert resumed.stages["evaluate_candidate"].attempt == 2
    for name, previous in preserved.items():
        current = resumed.stages[name]
        assert current.status is StageStatus.SUCCESS
        assert current.attempt == previous.attempt
        assert current.fingerprint == previous.fingerprint
        assert current.outputs == previous.outputs
        for output in current.outputs:
            assert sha256_file(run_directory / output.path) == output.sha256
    verify_successful_stage_outputs(run_directory, resumed)


def test_early_capability_failure_keeps_complete_failure_safe_topology(tmp_path: Path) -> None:
    config = _fast_config("tiny_moe_int8.yaml", tmp_path / "artifacts")
    config = config.model_copy(
        update={
            "quantization": config.quantization.model_copy(
                update={"backend": "awq", "method": "weight_only"}
            )
        }
    )
    run_directory = Path(config.output.root) / "early-capability-failure"

    with pytest.raises(CapabilityError, match="unavailable or unsupported"):
        run_experiment(
            config,
            project_root=PROJECT_ROOT,
            run_id=run_directory.name,
        )

    manifest = load_manifest(run_directory)
    environment = _read_json(run_directory / "environment.json")
    assert manifest.status is RunStatus.FAILED
    assert manifest.stages["probe_runtime"].status is StageStatus.FAILED
    assert load_config(run_directory / "resolved_config.yaml") == config
    assert environment["capability_probe"]["status"] == "pending"
    assert environment["hardware"]
    assert manifest.environment == environment
    assert manifest.environment["packages"]
    assert manifest.git.model_dump(mode="json") == environment["git"]
    assert (run_directory / "events.jsonl").stat().st_size > 0
    assert (run_directory / "status.json").is_file()
    assert all(
        (run_directory / category).is_dir()
        for category in ("metrics", "routing", "checkpoints", "reports")
    )


def test_force_quantize_archives_candidate_subtree_and_preserves_baseline_work(
    completed_moe_run: Path,
    tmp_path: Path,
) -> None:
    run_directory = _copy_completed_run(completed_moe_run, tmp_path / "forced-run")
    before = load_manifest(run_directory)
    baseline_names = ("evaluate_baseline", "benchmark_baseline")
    baseline_records = {name: before.stages[name] for name in baseline_names}
    prior_candidate_records = {name: before.stages[name] for name in FORCED_QUANTIZE_STAGES}

    assert (
        resume_experiment(
            run_directory,
            force_stage="quantize",
            project_root=PROJECT_ROOT,
        )
        == run_directory.resolve()
    )

    after = load_manifest(run_directory)
    assert after.status is RunStatus.SUCCESS
    assert FORCED_QUANTIZE_STAGES == (
        "quantize",
        "evaluate_candidate",
        "benchmark_candidate",
        "compare_routing",
        "generate_reports",
        "finalize_manifest",
    )
    for name, previous in baseline_records.items():
        current = after.stages[name]
        assert current.attempt == previous.attempt
        assert current.fingerprint == previous.fingerprint
        assert current.outputs == previous.outputs
        for output in current.outputs:
            assert sha256_file(run_directory / output.path) == output.sha256

    for name in FORCED_QUANTIZE_STAGES:
        assert after.stages[name].status is StageStatus.SUCCESS
        assert after.stages[name].attempt == prior_candidate_records[name].attempt + 1

    archive_roots = tuple((run_directory / "archive/forced").iterdir())
    assert len(archive_roots) == 1
    archive_root = archive_roots[0]
    for name, previous in prior_candidate_records.items():
        for output in previous.outputs:
            archived = archive_root / name / output.path
            assert archived.is_file()
            assert archived.stat().st_size == output.size_bytes
            assert sha256_file(archived) == output.sha256
    verify_successful_stage_outputs(run_directory, after)


def test_resume_detects_tampered_completed_output_before_mutation(
    completed_moe_run: Path,
    tmp_path: Path,
) -> None:
    run_directory = _copy_completed_run(completed_moe_run, tmp_path / "tampered-run")
    before = load_manifest(run_directory)
    target = run_directory / before.stages["evaluate_candidate"].outputs[0].path
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
        resume_experiment(run_directory, project_root=PROJECT_ROOT)

    assert load_manifest(run_directory) == before
    assert not (run_directory / "archive").exists()


def test_resume_rejects_missing_completed_output_without_recreating_it(
    completed_moe_run: Path,
    tmp_path: Path,
) -> None:
    run_directory = _copy_completed_run(completed_moe_run, tmp_path / "missing-output-run")
    before_manifest = (run_directory / "manifest.json").read_bytes()
    environment_path = run_directory / "environment.json"
    environment_path.unlink()

    with pytest.raises(ArtifactIntegrityError, match="Missing or unsafe output"):
        resume_experiment(run_directory, project_root=PROJECT_ROOT)

    assert not environment_path.exists()
    assert (run_directory / "manifest.json").read_bytes() == before_manifest
    assert not (run_directory / "archive").exists()


def test_dense_optional_routing_is_unsupported_without_failing_run(tmp_path: Path) -> None:
    config = _fast_config("tiny_dense_int8.yaml", tmp_path / "artifacts")
    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="dense-optional-routing",
    )

    manifest = load_manifest(run_directory)
    routing_stage = manifest.stages["compare_routing"]
    assert manifest.status is RunStatus.SUCCESS
    assert routing_stage.required is False
    assert routing_stage.status is StageStatus.UNSUPPORTED
    assert routing_stage.reason == (
        "routing capture is unsupported for model local://fixtures/tiny-dense"
    )
    assert routing_stage.outputs == ()
    assert not (run_directory / "routing/comparison.json").exists()
    assert manifest.stages["generate_reports"].status is StageStatus.SUCCESS
    assert manifest.stages["finalize_manifest"].status is StageStatus.SUCCESS

    candidate_summary = _read_json(run_directory / "reports/candidate_summary.json")
    for metric in ("routing_js_divergence", "route_agreement"):
        measurement = candidate_summary["metrics"][metric]
        assert measurement["status"] == "unsupported"
        assert measurement["value"] is None
        assert measurement["reason"] == "routing analysis is unsupported or disabled"
    report_data = _read_json(run_directory / "reports/report_data.json")
    assert report_data["metadata"]["routing_comparison"] is None
    assert _read_json(run_directory / "status.json")["status"] == "success"
    verify_successful_stage_outputs(run_directory, manifest)


def test_report_marks_benchmark_metrics_unavailable_when_benchmark_is_disabled(
    tmp_path: Path,
) -> None:
    configured = _fast_config("tiny_moe_baseline.yaml", tmp_path / "artifacts with spaces")
    config = configured.model_copy(
        update={"benchmark": configured.benchmark.model_copy(update={"enabled": False})}
    )

    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="benchmark-disabled",
    )

    manifest = load_manifest(run_directory)
    assert manifest.status is RunStatus.SUCCESS
    for stage_name in ("benchmark_baseline", "benchmark_candidate"):
        assert manifest.stages[stage_name].required is False
        assert manifest.stages[stage_name].status is StageStatus.SKIPPED_NOT_REQUIRED
    candidate = _read_json(run_directory / "reports/candidate_summary.json")
    for metric in ("latency_ms", "throughput_tokens_per_second", "serialized_size_bytes"):
        assert candidate["metrics"][metric]["status"] == "unavailable"
        assert candidate["metrics"][metric]["value"] is None
        assert "disabled" in candidate["metrics"][metric]["reason"]
    report_data = _read_json(run_directory / "reports/report_data.json")
    assert report_data["reproduction_commands"] == [
        shell_join(("uv", "run", "iql", "resume", str(run_directory))),
        shell_join(("uv", "run", "iql", "report", str(run_directory))),
    ]
