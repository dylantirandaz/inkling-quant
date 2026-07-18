"""Focused tests for the governed DAG, resume, force-stage, and atomic outputs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from inkling_quant_lab.artifacts import LocalArtifactStore, sha256_file
from inkling_quant_lab.config import ExperimentConfig, dump_resolved_config
from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.manifests import (
    ArtifactChecksum,
    ModelProvenance,
    RunManifest,
    RunStatus,
    StageError,
    StageStatus,
)
from inkling_quant_lab.pipeline.resume import (
    ForcedArchiveLedger,
    prepare_resume,
    recover_forced_archive_transactions,
    recover_published_stage_outputs,
    reopen_manifest_for_resume,
)
from inkling_quant_lab.pipeline.stages import (
    STAGE_DEPENDENCIES,
    STAGE_ORDER,
    StageName,
    commit_stage_result,
    initial_stage_records,
    record_stage_result,
    stage_definitions,
    stage_descendants,
    stage_fingerprint,
    stage_fingerprint_from_manifest,
)

pytestmark = pytest.mark.unit
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _checksum(path: Path, root: Path) -> ArtifactChecksum:
    return ArtifactChecksum(
        path=path.relative_to(root).as_posix(),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _manifest_with_stages(
    root: Path,
    config: ExperimentConfig,
    *,
    successful: Sequence[StageName],
    failed_stage: StageName | None = None,
    run_status: RunStatus = RunStatus.FAILED,
) -> RunManifest:
    store = LocalArtifactStore(root)
    records = initial_stage_records(config)
    successful_set = set(successful)
    for name in STAGE_ORDER:
        current = records[name]
        if name in successful_set:
            path = store.write_text(f"stages/{name}/result.json", f'{{"stage":"{name}"}}')
            outputs = (_checksum(path, store.root),)
            dependencies: dict[str, tuple[ArtifactChecksum, ...]] = {
                dependency: records[dependency].outputs for dependency in STAGE_DEPENDENCIES[name]
            }
            records[name] = current.model_copy(
                update={
                    "status": StageStatus.SUCCESS,
                    "fingerprint": stage_fingerprint(name, config, dependencies),
                    "started_at": NOW,
                    "completed_at": NOW,
                    "outputs": outputs,
                    "attempt": 1,
                }
            )
        elif name == failed_stage:
            records[name] = current.model_copy(
                update={
                    "status": StageStatus.FAILED,
                    "fingerprint": "f" * 64,
                    "started_at": NOW,
                    "completed_at": NOW,
                    "error": StageError(code="INJECTED", message="injected failure"),
                    "attempt": 1,
                }
            )

    return RunManifest(
        run_id="resume-test",
        config_hash=config.config_hash(),
        model=ModelProvenance(id=config.model.model_id, revision=config.model.revision),
        started_at=NOW,
        completed_at=NOW if run_status is not RunStatus.RUNNING else None,
        stages=records,
        status=run_status,
    )


def test_governed_dag_has_exact_objective_stage_names_and_dependencies() -> None:
    assert STAGE_ORDER == (
        "resolve_configuration",
        "probe_runtime",
        "load_baseline",
        "inventory_modules",
        "collect_statistics",
        "resolve_precision_policy",
        "quantize",
        "evaluate_baseline",
        "evaluate_candidate",
        "benchmark_baseline",
        "benchmark_candidate",
        "compare_routing",
        "generate_reports",
        "finalize_manifest",
    )
    assert len(STAGE_DEPENDENCIES) == 14
    assert STAGE_DEPENDENCIES["compare_routing"] == (
        "collect_statistics",
        "quantize",
    )
    assert STAGE_DEPENDENCIES["generate_reports"] == (
        "evaluate_baseline",
        "evaluate_candidate",
        "benchmark_baseline",
        "benchmark_candidate",
        "compare_routing",
    )


def test_stage_required_and_optional_decisions_follow_resolved_config(
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    default = config_factory()
    definitions = {item.name: item for item in stage_definitions(default)}

    assert definitions["evaluate_baseline"].required is True
    assert definitions["evaluate_candidate"].required is True
    assert definitions["collect_statistics"].enabled is True
    assert definitions["collect_statistics"].required is False
    assert definitions["compare_routing"].enabled is True
    assert definitions["compare_routing"].required is False
    assert definitions["benchmark_baseline"].required is True
    assert definitions["benchmark_candidate"].required is True
    assert definitions["finalize_manifest"].required is True

    raw = default.canonical_dict()
    for suite in raw["evaluation"]["suites"]:
        suite["optional"] = True
    raw["routing"] = {"mode": "off", "required": False}
    raw["benchmark"]["enabled"] = False
    raw["reporting"] = {
        "markdown": False,
        "html": False,
        "plots": False,
        "include_interpretation": False,
    }
    optional = ExperimentConfig.model_validate(raw)
    definitions = {item.name: item for item in stage_definitions(optional)}

    for name in (
        "collect_statistics",
        "evaluate_baseline",
        "evaluate_candidate",
        "benchmark_baseline",
        "benchmark_candidate",
        "compare_routing",
        "generate_reports",
    ):
        assert definitions[name].required is False
    assert definitions["collect_statistics"].enabled is False
    assert definitions["compare_routing"].enabled is False
    assert definitions["finalize_manifest"].required is True


def test_fingerprints_are_deterministic_from_config_and_direct_checksums(
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    config = config_factory()
    dependencies = STAGE_DEPENDENCIES["generate_reports"]
    values = {dependency: ("a" * 64, "b" * 64) for dependency in dependencies}
    reversed_values = {
        dependency: tuple(reversed(values[dependency])) for dependency in reversed(dependencies)
    }

    first = stage_fingerprint("generate_reports", config, values)
    second = stage_fingerprint("generate_reports", config, reversed_values)
    changed = dict(values)
    changed["compare_routing"] = ("a" * 64, "c" * 64)

    assert first == second
    assert first != stage_fingerprint("generate_reports", config, changed)
    assert len(first) == 64

    with pytest.raises(ArtifactIntegrityError, match="missing dependencies"):
        stage_fingerprint("generate_reports", config, {})

    with pytest.raises(ArtifactIntegrityError, match="invalid dependency checksum"):
        stage_fingerprint("probe_runtime", config, {"resolve_configuration": ("invalid",)})


def test_manifest_fingerprint_accepts_terminal_optional_dependency(
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    config = config_factory()
    stages = initial_stage_records(config)
    stages["collect_statistics"] = stages["collect_statistics"].model_copy(
        update={
            "status": StageStatus.SKIPPED_NOT_REQUIRED,
            "reason": "no required statistics",
            "completed_at": NOW,
        }
    )
    manifest = RunManifest(
        run_id="optional-dependency",
        config_hash=config.config_hash(),
        model=ModelProvenance(id=config.model.model_id, revision=config.model.revision),
        stages=stages,
    )

    fingerprint = stage_fingerprint_from_manifest("resolve_precision_policy", config, manifest)

    assert fingerprint == stage_fingerprint(
        "resolve_precision_policy", config, {"collect_statistics": ()}
    )

    stages["collect_statistics"] = stages["collect_statistics"].model_copy(
        update={"status": StageStatus.PENDING}
    )
    incomplete = manifest.model_copy(update={"stages": stages})
    with pytest.raises(ArtifactIntegrityError, match="is not complete"):
        stage_fingerprint_from_manifest("resolve_precision_policy", config, incomplete)


def test_quantize_descendants_exclude_independent_baseline_work() -> None:
    assert stage_descendants("quantize", include_self=True) == (
        "quantize",
        "evaluate_candidate",
        "benchmark_candidate",
        "compare_routing",
        "generate_reports",
        "finalize_manifest",
    )
    descendants = set(stage_descendants("quantize", include_self=True))
    assert "evaluate_baseline" not in descendants
    assert "benchmark_baseline" not in descendants


def test_resume_skips_valid_completed_stages_and_restarts_failure(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """TC-PIPELINE-003: successful checksums survive while the failed stage restarts."""

    config = config_factory()
    successful: tuple[StageName, ...] = (
        "resolve_configuration",
        "probe_runtime",
        "load_baseline",
        "inventory_modules",
        "collect_statistics",
        "resolve_precision_policy",
        "quantize",
        "evaluate_baseline",
        "benchmark_baseline",
    )
    manifest = _manifest_with_stages(
        tmp_path,
        config,
        successful=successful,
        failed_stage="evaluate_candidate",
    )
    preserved = manifest.stages["quantize"].outputs

    plan = prepare_resume(tmp_path, manifest, config, at=NOW)

    assert plan.manifest.status is RunStatus.RUNNING
    assert set(successful).issubset(plan.skip_stages)
    assert "evaluate_candidate" in plan.restart_stages
    assert plan.manifest.stages["evaluate_candidate"].status is StageStatus.FAILED
    assert plan.manifest.stages["quantize"].outputs == preserved
    assert manifest.status is RunStatus.FAILED
    for output in preserved:
        assert (tmp_path / output.path).is_file()

    fingerprint = stage_fingerprint(
        "evaluate_candidate",
        config,
        {"quantize": preserved},
    )
    restarted = plan.manifest.start_stage("evaluate_candidate", fingerprint, at=NOW)
    assert restarted.stages["evaluate_candidate"].status is StageStatus.RUNNING


def test_resume_rejects_any_mutated_successful_output(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    config = config_factory()
    manifest = _manifest_with_stages(
        tmp_path,
        config,
        successful=("resolve_configuration",),
    )
    output = manifest.stages["resolve_configuration"].outputs[0]
    (tmp_path / output.path).write_text("mutated", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
        prepare_resume(tmp_path, manifest, config)


def test_resume_rejects_successful_stage_with_stale_fingerprint(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    config = config_factory()
    manifest = _manifest_with_stages(
        tmp_path,
        config,
        successful=("resolve_configuration",),
    )
    stages = dict(manifest.stages)
    stages["resolve_configuration"] = stages["resolve_configuration"].model_copy(
        update={"fingerprint": "0" * 64}
    )
    manifest = manifest.model_copy(update={"stages": stages})

    with pytest.raises(ArtifactIntegrityError, match="fingerprint mismatch"):
        prepare_resume(tmp_path, manifest, config)


def test_resume_rejects_config_or_governed_stage_set_mismatch(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    config = config_factory()
    manifest = _manifest_with_stages(tmp_path, config, successful=())

    with pytest.raises(ArtifactIntegrityError, match="configuration hash"):
        prepare_resume(
            tmp_path,
            manifest.model_copy(update={"config_hash": "0" * 64}),
            config,
        )

    incomplete = manifest.model_copy(
        update={"stages": {"resolve_configuration": manifest.stages["resolve_configuration"]}}
    )
    with pytest.raises(ArtifactIntegrityError, match="stage set mismatch"):
        prepare_resume(tmp_path, incomplete, config)


def test_force_quantize_archives_dependents_and_retains_baseline_stages(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """TC-PIPELINE-004: forced candidate work cannot erase independent baselines."""

    config = config_factory()
    manifest = _manifest_with_stages(
        tmp_path,
        config,
        successful=STAGE_ORDER,
        run_status=RunStatus.SUCCESS,
    )
    expected_invalidated = stage_descendants("quantize", include_self=True)
    retained = (
        "resolve_configuration",
        "probe_runtime",
        "load_baseline",
        "inventory_modules",
        "collect_statistics",
        "resolve_precision_policy",
        "evaluate_baseline",
        "benchmark_baseline",
    )
    original_quantize = manifest.stages["quantize"].outputs[0]

    plan = prepare_resume(
        tmp_path,
        manifest,
        config,
        force_stage="quantize",
        archive_id="force-001",
        at=NOW,
    )

    assert plan.invalidated_stages == expected_invalidated
    assert set(plan.restart_stages) == set(expected_invalidated)
    assert set(retained).issubset(plan.skip_stages)
    assert plan.manifest.status is RunStatus.RUNNING
    for name in expected_invalidated:
        assert plan.manifest.stages[name].status is StageStatus.INVALIDATED
        assert plan.manifest.stages[name].outputs == ()
    for name in retained:
        assert plan.manifest.stages[name].status is StageStatus.SUCCESS
        for output in plan.manifest.stages[name].outputs:
            assert (tmp_path / output.path).is_file()

    quantize_archive = next(item for item in plan.archived_stages if item.name == "quantize")
    assert quantize_archive.original_outputs == (original_quantize,)
    archived_output = quantize_archive.archived_outputs[0]
    assert not (tmp_path / original_quantize.path).exists()
    assert (tmp_path / archived_output.path).is_file()
    assert sha256_file(tmp_path / archived_output.path) == original_quantize.sha256
    assert manifest.status is RunStatus.SUCCESS
    assert manifest.stages["quantize"].status is StageStatus.SUCCESS


def test_atomic_stage_result_records_only_committed_outputs(tmp_path: Path) -> None:
    """TC-PIPELINE-005: interrupted temporary output cannot resemble stage success."""

    store = LocalArtifactStore(tmp_path)

    def successful(directory: Path) -> None:
        (directory / "metrics.json").write_text('{"value":1}', encoding="utf-8")

    result = commit_stage_result(store, "evaluate_baseline", successful)

    assert result.name == "evaluate_baseline"
    assert tuple(output.path for output in result.outputs) == (
        "stages/evaluate_baseline/metrics.json",
    )
    assert len(result.outputs[0].sha256) == 64

    def interrupted(directory: Path) -> None:
        (directory / "partial.json").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected termination")

    with pytest.raises(RuntimeError, match="injected termination"):
        commit_stage_result(store, "benchmark_candidate", interrupted)

    assert not store.path("stages/benchmark_candidate").exists()
    assert not tuple(store.path("stages").glob(".benchmark_candidate.*.tmp"))

    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")

    def unsafe(directory: Path) -> None:
        (directory / "escape.json").symlink_to(outside)

    with pytest.raises(ArtifactIntegrityError, match="must not be a symlink"):
        commit_stage_result(store, "compare_routing", unsafe)
    assert not store.path("stages/compare_routing").exists()


def test_stage_result_rejects_missing_and_duplicate_outputs(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("result", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="duplicate output"):
        record_stage_result(
            tmp_path,
            "probe_runtime",
            (path, path),
        )
    with pytest.raises(ArtifactIntegrityError, match="missing or outside"):
        record_stage_result(
            tmp_path,
            "probe_runtime",
            (tmp_path / "missing.json",),
        )


@pytest.mark.parametrize("status", [RunStatus.FAILED, RunStatus.SUCCESS])
def test_terminal_manifest_can_be_reopened_without_editing_manifest_class(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
    status: RunStatus,
) -> None:
    manifest = _manifest_with_stages(
        tmp_path,
        config_factory(),
        successful=(),
        run_status=status,
    )

    reopened = reopen_manifest_for_resume(manifest, at=NOW)

    assert reopened.status is RunStatus.RUNNING
    assert reopened.completed_at is None
    assert manifest.status is status


def test_pending_manifest_starts_and_interrupted_stage_is_invalidated(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    config = config_factory()
    pending = _manifest_with_stages(
        tmp_path,
        config,
        successful=(),
        run_status=RunStatus.PENDING,
    )
    assert reopen_manifest_for_resume(pending, at=NOW).status is RunStatus.RUNNING

    stages = dict(pending.stages)
    stages["load_baseline"] = stages["load_baseline"].model_copy(
        update={
            "status": StageStatus.RUNNING,
            "fingerprint": "f" * 64,
            "started_at": NOW,
            "attempt": 1,
        }
    )
    interrupted = pending.model_copy(
        update={"status": RunStatus.RUNNING, "started_at": NOW, "stages": stages}
    )

    reopened = reopen_manifest_for_resume(interrupted, at=NOW)

    assert reopened.stages["load_baseline"].status is StageStatus.INVALIDATED
    assert reopened.stages["load_baseline"].fingerprint is None
    assert "interrupted" in (reopened.stages["load_baseline"].reason or "")


def test_resume_adopts_atomic_output_published_before_manifest_commit(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    config = config_factory()
    store = LocalArtifactStore(tmp_path)
    store.write_text("resolved_config.yaml", dump_resolved_config(config))
    stages = initial_stage_records(config)
    stages["resolve_configuration"] = stages["resolve_configuration"].model_copy(
        update={
            "status": StageStatus.RUNNING,
            "fingerprint": stage_fingerprint("resolve_configuration", config, {}),
            "started_at": NOW,
            "attempt": 1,
        }
    )
    interrupted = RunManifest(
        run_id="interrupted-publish",
        config_hash=config.config_hash(),
        model=ModelProvenance(id=config.model.model_id, revision=config.model.revision),
        started_at=NOW,
        stages=stages,
        status=RunStatus.RUNNING,
    )

    recovered, names = recover_published_stage_outputs(tmp_path, interrupted, config)

    assert names == ("resolve_configuration",)
    stage = recovered.stages["resolve_configuration"]
    assert stage.status is StageStatus.SUCCESS
    assert stage.attempt == 1
    assert tuple(output.path for output in stage.outputs) == ("resolved_config.yaml",)
    assert prepare_resume(tmp_path, recovered, config, at=NOW).skip_stages[0] == (
        "resolve_configuration"
    )


def test_forced_archive_ledger_recovers_move_before_manifest_commit(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    config = config_factory()
    original = _manifest_with_stages(
        tmp_path,
        config,
        successful=STAGE_ORDER,
        run_status=RunStatus.SUCCESS,
    )
    original_output = original.stages["generate_reports"].outputs[0]
    plan = prepare_resume(
        tmp_path,
        original,
        config,
        force_stage="generate_reports",
        archive_id="crash-window",
        at=NOW,
    )
    assert plan.forced_archive_ledger is not None
    assert not (tmp_path / original_output.path).exists()

    restored = recover_forced_archive_transactions(tmp_path, original)

    assert original_output.path in restored
    assert sha256_file(tmp_path / original_output.path) == original_output.sha256
    ledger = ForcedArchiveLedger.model_validate_json(
        (tmp_path / plan.forced_archive_ledger).read_text(encoding="utf-8")
    )
    assert ledger.state == "recovered"


def test_forced_archive_recovery_does_not_restore_after_invalidation_is_durable(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    config = config_factory()
    original = _manifest_with_stages(
        tmp_path,
        config,
        successful=STAGE_ORDER,
        run_status=RunStatus.SUCCESS,
    )
    original_output = original.stages["generate_reports"].outputs[0]
    plan = prepare_resume(
        tmp_path,
        original,
        config,
        force_stage="generate_reports",
        archive_id="durable-invalidation",
        at=NOW,
    )

    assert recover_forced_archive_transactions(tmp_path, plan.manifest) == ()
    assert not (tmp_path / original_output.path).exists()
    assert plan.forced_archive_ledger is not None
    ledger = ForcedArchiveLedger.model_validate_json(
        (tmp_path / plan.forced_archive_ledger).read_text(encoding="utf-8")
    )
    assert ledger.state == "manifest_committed"
