"""Opt-in public-MoE source-weight-free candidate artifact benchmark contract."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from inkling_quant_lab.config import ExperimentConfig, load_config
from inkling_quant_lab.manifests import RunStatus, StageStatus, load_manifest
from inkling_quant_lab.pipeline.resume import (
    verify_successful_stage_fingerprints,
    verify_successful_stage_outputs,
)
from inkling_quant_lab.pipeline.runner import run_experiment
from inkling_quant_lab.quantization.reference import load_export_recipe

pytestmark = [pytest.mark.integration, pytest.mark.model_public, pytest.mark.slow]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs/experiments/hf_stories15m_native_int8_source_weight_free_peak.yaml"
MODEL_ID = "ggml-org/stories15M_MOE"
MODEL_REVISION = "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
_RUN_PUBLIC_NATIVE_SOURCE_WEIGHT_FREE_PEAK = (
    os.environ.get("IQL_RUN_PUBLIC_NATIVE_SOURCE_WEIGHT_FREE_PEAK") == "1"
)
_OPT_IN = pytest.mark.skipif(
    not _RUN_PUBLIC_NATIVE_SOURCE_WEIGHT_FREE_PEAK,
    reason=(
        "set IQL_RUN_PUBLIC_NATIVE_SOURCE_WEIGHT_FREE_PEAK=1 after caching the exact "
        "pinned Stories15M revision"
    ),
)
_REQUIRED_CACHED_FILES = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_SOURCE_METADATA_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_SUBJECT_ARTIFACT_PEAK_KIND = "benchmark_subject_artifact_worker_process_peak_rss"
_SUBJECT_ARTIFACT_PEAK_SCOPE = (
    "OS process high-water RSS through the final governed benchmark boundary in a fresh "
    "spawned worker loading exactly one persisted benchmark subject; includes interpreter and "
    "import startup, artifact integrity validation, pinned architecture and tokenizer "
    "construction, baseline source-weight loading or candidate exported-weight loading, native "
    "prepacking, warm-up, and measured trials; the candidate path does not load float source "
    "weights, quantize, or create an export; excludes the parent pipeline, all prior stages, "
    "post-read runtime cleanup, result serialization/IPC, and process-exit work and is an "
    "artifact-load-plus-inference process peak rather than steady-state-only, tensor-attributable, "
    "or final-through-exit residency"
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _read_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return cast(list[dict[str, Any]], value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
    return digest.hexdigest()


def _require_exact_offline_cache(config: ExperimentConfig) -> Path:
    """Require every exact model/config/tokenizer input without permitting a download."""

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        pytest.fail("the opt-in public-model test requires the Hugging Face integration extra")

    cached_paths: list[Path] = []
    for filename in _REQUIRED_CACHED_FILES:
        cached = try_to_load_from_cache(
            config.model.model_id,
            filename,
            revision=config.model.revision,
        )
        if not isinstance(cached, str):
            pytest.fail(
                f"exact pinned cache is missing {filename}; populate it before enabling the test"
            )
        path = Path(cached)
        if not path.is_file():
            pytest.fail(f"cached pinned file is not readable: {filename}")
        cached_paths.append(path)

    snapshot_roots = {path.parent for path in cached_paths}
    assert len(snapshot_roots) == 1
    snapshot = snapshot_roots.pop()
    assert snapshot.name == MODEL_REVISION
    assert not any(
        path.suffix.lower() in {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
        for path in cached_paths
    )
    return snapshot


def _assert_evaluation_and_routing_outputs(run_directory: Path) -> None:
    baseline = _read_records(run_directory / "metrics/evaluation_baseline/results.json")
    candidate = _read_records(run_directory / "metrics/evaluation_candidate/results.json")
    assert len(baseline) == len(candidate) == 2
    identity_fields = (
        "evaluator_name",
        "evaluator_version",
        "dataset_id",
        "dataset_revision",
        "split",
        "dataset_sha256",
        "sample_ids",
        "sample_count",
        "seed",
        "prompt_template_hash",
        "decode_config",
    )
    for baseline_result, candidate_result in zip(baseline, candidate, strict=True):
        assert baseline_result["status"] == candidate_result["status"] == "success"
        assert baseline_result["failures"] == candidate_result["failures"] == []
        assert {field: baseline_result[field] for field in identity_fields} == {
            field: candidate_result[field] for field in identity_fields
        }

    routing = _read_object(run_directory / "routing/comparison.json")
    assert routing["macro"]["layer_count"] == 6
    assert routing["macro"]["token_weight"] > 0
    assert routing["macro"]["token_route_agreement"] is not None
    assert routing["macro"]["top_k_overlap"] is not None
    assert len(routing["per_layer"]) == 6
    assert (run_directory / "routing/candidate/traces.jsonl").is_file()


@_OPT_IN
def test_public_native_int8_source_weight_free_subject_artifact_peak_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute the exact cached public model and the governed exported candidate."""

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    config = load_config(
        CONFIG,
        (
            f"output.root={json.dumps(str(tmp_path / 'artifacts'))}",
            "benchmark.warmup_iterations=0",
            "benchmark.repetitions=1",
        ),
    )
    assert config.model.model_id == MODEL_ID
    assert config.model.revision == MODEL_REVISION
    assert config.model.local_files_only is True
    assert config.benchmark.host_memory_mode == "isolated_subject_artifact_peak_rss"
    snapshot = _require_exact_offline_cache(config)

    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="public-native-int8-source-weight-free-peak",
    )

    manifest = load_manifest(run_directory)
    assert manifest.status is RunStatus.SUCCESS
    assert len(manifest.stages) == 14
    assert all(stage.status is StageStatus.SUCCESS for stage in manifest.stages.values())
    assert manifest.stages["evaluate_candidate"].status is StageStatus.SUCCESS
    assert manifest.stages["compare_routing"].status is StageStatus.SUCCESS
    assert manifest.model.id == MODEL_ID
    assert manifest.model.revision == MODEL_REVISION
    verify_successful_stage_outputs(run_directory, manifest)
    verify_successful_stage_fingerprints(config, manifest)
    assert _read_object(run_directory / "status.json")["status"] == "success"

    quantization = _read_object(run_directory / "checkpoints/candidate/quantization_manifest.json")
    assert quantization["backend"] == "torch_native_dynamic_int8"
    assert quantization["method"] == "native_dynamic_w8a8"
    precision_map = cast(dict[str, str], quantization["module_precision_map"])
    selected = {name for name, precision in precision_map.items() if precision == "int8"}
    expected_selected = {
        f"model.layers.{layer}.self_attn.{projection}_proj"
        for layer in range(6)
        for projection in ("q", "k", "v", "o")
    }
    assert selected == expected_selected
    assert len(selected) == 24

    export = run_directory / "checkpoints/candidate/candidate"
    metadata_path = export / "metadata.json"
    tensor_path = export / "model.safetensors"
    assert set(path.name for path in export.iterdir()) == {
        metadata_path.name,
        tensor_path.name,
    }
    assert all(path.is_file() and not path.is_symlink() for path in export.iterdir())
    recipe = load_export_recipe(export)
    assert recipe["schema_version"] == "2.0"
    assert recipe["model"]["model_id"] == MODEL_ID
    assert recipe["model"]["revision"] == MODEL_REVISION
    assert recipe["quantization"] == quantization
    assert recipe["reload"] == {
        "adapter": "hf_causal_lm_source_weight_free_v1",
        "backend": "torch_native_dynamic_int8",
        "format": "safetensors",
        "metadata_file": "metadata.json",
        "tensor_file": "model.safetensors",
        "tensor_sha256": _sha256_file(tensor_path),
    }
    export_size = metadata_path.stat().st_size + tensor_path.stat().st_size
    assert quantization["serialized_size_bytes"] == export_size

    quantize_outputs = {output.path: output for output in manifest.stages["quantize"].outputs}
    for path in (metadata_path, tensor_path):
        relative = path.relative_to(run_directory).as_posix()
        assert quantize_outputs[relative].sha256 == _sha256_file(path)
        assert quantize_outputs[relative].size_bytes == path.stat().st_size

    benchmarks = {
        kind: _read_object(run_directory / f"metrics/benchmark_{kind}/benchmark.json")
        for kind in ("baseline", "candidate")
    }
    baseline_benchmark = benchmarks["baseline"]
    candidate_benchmark = benchmarks["candidate"]
    assert baseline_benchmark["model_load_time_kind"] == "cold_model_load"
    assert candidate_benchmark["model_load_time_kind"] == "candidate_source_weight_free_export_load"
    assert baseline_benchmark["source_weight_free_load_provenance"] is None

    provenance = cast(dict[str, Any], candidate_benchmark["source_weight_free_load_provenance"])
    assert provenance["schema_version"] == "source-weight-free-reload-v1"
    assert provenance["reload_adapter"] == "hf_causal_lm_source_weight_free_v1"
    assert provenance["backend"] == "torch_native_dynamic_int8"
    assert provenance["model_id"] == MODEL_ID
    assert provenance["revision"] == MODEL_REVISION
    assert provenance["source_model_checksum"] == quantization["source_model_checksum"]
    assert provenance["strict_load"] is True
    assert provenance["assign"] is True
    assert provenance["source_weights_loaded"] is False
    assert provenance["missing_keys"] == []
    assert provenance["unexpected_keys"] == []
    assert provenance["meta_tensor_names"] == []
    assert provenance["native_wrapper_count"] == 24
    assert set(provenance["quantized_module_names"]) == expected_selected
    assert provenance["metadata_file"] == metadata_path.name
    assert provenance["tensor_file"] == tensor_path.name
    assert provenance["metadata_sha256"] == _sha256_file(metadata_path)
    assert provenance["tensor_sha256"] == _sha256_file(tensor_path)
    assert provenance["bundle_sha256"] == _bundle_sha256((metadata_path, tensor_path))
    assert len(provenance["candidate_state_checksum"]) == 64
    assert int(provenance["candidate_state_checksum"], 16) >= 0
    expected_source_metadata = {
        filename: _sha256_file(snapshot / filename) for filename in _SOURCE_METADATA_FILES
    }
    assert provenance["source_metadata_file_sha256"] == expected_source_metadata
    assert "model.safetensors" not in provenance["source_metadata_file_sha256"]

    assert candidate_benchmark["serialized_size_bytes"] == export_size
    parent_pid = os.getpid()
    worker_pids: set[int] = set()
    for benchmark in benchmarks.values():
        assert benchmark["warmup_iterations"] == 0
        assert benchmark["repetitions"] == 1
        assert len(benchmark["trials"]) == 1
        memory = benchmark["peak_memory"]
        assert memory["host_measurement_kind"] == _SUBJECT_ARTIFACT_PEAK_KIND
        assert memory["host_scope"] == _SUBJECT_ARTIFACT_PEAK_SCOPE
        assert memory["host_process_isolated"] is True
        assert memory["host_available"] is True
        assert memory["host_bytes"] > 0
        assert memory["host_worker_pid"] != parent_pid
        assert (
            memory["host_bytes"] >= benchmark["trials"][0]["host_memory_bytes_at_post_trial_sample"]
        )
        worker_pids.add(memory["host_worker_pid"])
    assert len(worker_pids) == 2

    summaries = {
        kind: _read_object(run_directory / f"reports/{kind}_summary.json")
        for kind in ("baseline", "candidate")
    }
    assert summaries["baseline"]["benchmark_model_load_time_kind"] == "cold_model_load"
    assert summaries["baseline"]["benchmark_source_weight_free_load_provenance"] is None
    assert (
        summaries["candidate"]["benchmark_model_load_time_kind"]
        == "candidate_source_weight_free_export_load"
    )
    assert summaries["candidate"]["benchmark_source_weight_free_load_provenance"] == provenance
    for kind, benchmark in benchmarks.items():
        metrics = summaries[kind]["metrics"]
        assert metrics["peak_host_memory_bytes"]["value"] == benchmark["peak_memory"]["host_bytes"]
        assert (
            metrics["benchmark_subject_artifact_worker_process_peak_rss_bytes"]["value"]
            == benchmark["peak_memory"]["host_bytes"]
        )
        assert (
            int(metrics["benchmark_subject_artifact_worker_pid"]["value"])
            == benchmark["peak_memory"]["host_worker_pid"]
        )

    comparison = _read_object(run_directory / "reports/comparison.json")
    load_delta = next(delta for delta in comparison["deltas"] if delta["metric"] == "load_time_ms")
    assert load_delta["status"] == "unavailable"
    assert load_delta["absolute_delta"] is None
    assert load_delta["relative_delta"] is None
    assert load_delta["baseline_value"] == summaries["baseline"]["metrics"]["load_time_ms"]["value"]
    assert (
        load_delta["candidate_value"] == summaries["candidate"]["metrics"]["load_time_ms"]["value"]
    )
    assert load_delta["reason"] == (
        "model-load operations differ: baseline='cold_model_load', "
        "candidate='candidate_source_weight_free_export_load'"
    )

    _assert_evaluation_and_routing_outputs(run_directory)
