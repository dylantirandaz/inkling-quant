"""Abrupt-termination recovery at stage-publication and force-archive boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inkling_quant_lab.config import ExperimentConfig, load_config
from inkling_quant_lab.exceptions import ArtifactIntegrityError, ConfigurationError
from inkling_quant_lab.manifests import RunStatus, StageStatus, load_manifest
from inkling_quant_lab.pipeline import runner as runner_module
from inkling_quant_lab.pipeline.resume import ForcedArchiveLedger
from inkling_quant_lab.pipeline.runner import resume_experiment, run_experiment

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = PROJECT_ROOT / "configs/experiments/tiny_moe_int8.yaml"


def _fast_config(artifact_root: Path) -> ExperimentConfig:
    return load_config(
        EXPERIMENT,
        (
            f"output.root={json.dumps(str(artifact_root))}",
            "benchmark.warmup_iterations=0",
            "benchmark.repetitions=1",
        ),
    )


def _tree_snapshot(root: Path) -> tuple[tuple[str, ...], dict[str, bytes]]:
    """Capture names and bytes so a failed preflight can prove no mutation."""

    directories = tuple(
        path.relative_to(root).as_posix() for path in sorted(root.rglob("*")) if path.is_dir()
    )
    files = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    return directories, files


def test_resume_adopts_stage_directory_published_before_manifest_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fast_config(tmp_path / "artifacts")
    run_directory = config.output.root / Path("publish-crash")
    real_persist = runner_module.persist_manifest

    def crash_after_quantize_publish(store, manifest):
        if manifest.stages["quantize"].status is StageStatus.SUCCESS:
            raise KeyboardInterrupt("simulated process termination")
        return real_persist(store, manifest)

    monkeypatch.setattr(runner_module, "persist_manifest", crash_after_quantize_publish)
    with pytest.raises(KeyboardInterrupt, match="simulated process termination"):
        run_experiment(
            config,
            project_root=PROJECT_ROOT,
            run_id=run_directory.name,
        )

    interrupted = load_manifest(run_directory)
    assert interrupted.status is RunStatus.RUNNING
    assert interrupted.stages["quantize"].status is StageStatus.RUNNING
    assert (run_directory / "checkpoints/candidate/quantization_manifest.json").is_file()

    monkeypatch.setattr(runner_module, "persist_manifest", real_persist)
    assert resume_experiment(run_directory, project_root=PROJECT_ROOT) == run_directory

    completed = load_manifest(run_directory)
    assert completed.status is RunStatus.SUCCESS
    assert completed.stages["quantize"].status is StageStatus.SUCCESS
    assert completed.stages["quantize"].attempt == 1


def test_force_archive_move_before_manifest_commit_is_recovered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fast_config(tmp_path / "artifacts")
    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="force-archive-crash",
    )
    original = load_manifest(run_directory)
    real_persist = runner_module.persist_manifest

    def crash_before_invalidated_manifest(store, manifest):
        if manifest.stages["generate_reports"].status is StageStatus.INVALIDATED:
            raise KeyboardInterrupt("simulated archive transaction termination")
        return real_persist(store, manifest)

    monkeypatch.setattr(runner_module, "persist_manifest", crash_before_invalidated_manifest)
    with pytest.raises(KeyboardInterrupt, match="archive transaction"):
        resume_experiment(
            run_directory,
            force_stage="generate_reports",
            project_root=PROJECT_ROOT,
        )
    assert load_manifest(run_directory) == original
    assert not (run_directory / "reports/report.md").exists()

    ledger_path = next((run_directory / "archive/forced").glob("*/archive_manifest.json"))
    assert (
        ForcedArchiveLedger.model_validate_json(ledger_path.read_text(encoding="utf-8")).state
        == "outputs_archived"
    )

    monkeypatch.setattr(runner_module, "persist_manifest", real_persist)
    assert resume_experiment(run_directory, project_root=PROJECT_ROOT) == run_directory
    assert (run_directory / "reports/report.md").is_file()
    assert (
        ForcedArchiveLedger.model_validate_json(ledger_path.read_text(encoding="utf-8")).state
        == "recovered"
    )


def test_resume_preflight_uses_verified_archived_config_before_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fast_config(tmp_path / "artifacts")
    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="force-archive-config-preflight",
    )
    original = load_manifest(run_directory)
    real_persist = runner_module.persist_manifest

    def crash_before_invalidated_manifest(store, manifest):
        if manifest.stages["resolve_configuration"].status is StageStatus.INVALIDATED:
            raise KeyboardInterrupt("simulated archive transaction termination")
        return real_persist(store, manifest)

    monkeypatch.setattr(runner_module, "persist_manifest", crash_before_invalidated_manifest)
    with pytest.raises(KeyboardInterrupt, match="archive transaction"):
        resume_experiment(
            run_directory,
            force_stage="resolve_configuration",
            project_root=PROJECT_ROOT,
        )

    assert not (run_directory / "resolved_config.yaml").exists()
    ledger_path = next((run_directory / "archive/forced").glob("*/archive_manifest.json"))
    monkeypatch.setattr(runner_module, "persist_manifest", real_persist)

    assert resume_experiment(run_directory, project_root=PROJECT_ROOT) == run_directory
    assert load_manifest(run_directory) == original
    assert load_config(run_directory / "resolved_config.yaml") == config
    assert (
        ForcedArchiveLedger.model_validate_json(ledger_path.read_text(encoding="utf-8")).state
        == "recovered"
    )


def test_force_archive_recovery_preflight_rejects_unrelated_tamper_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fast_config(tmp_path / "artifacts")
    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="force-archive-unrelated-tamper",
    )
    original = load_manifest(run_directory)
    real_persist = runner_module.persist_manifest

    def crash_before_invalidated_manifest(store, manifest):
        if manifest.stages["generate_reports"].status is StageStatus.INVALIDATED:
            raise KeyboardInterrupt("simulated archive transaction termination")
        return real_persist(store, manifest)

    monkeypatch.setattr(runner_module, "persist_manifest", crash_before_invalidated_manifest)
    with pytest.raises(KeyboardInterrupt, match="archive transaction"):
        resume_experiment(
            run_directory,
            force_stage="generate_reports",
            project_root=PROJECT_ROOT,
        )

    unrelated = original.stages["evaluate_candidate"].outputs[0]
    unrelated_path = run_directory / unrelated.path
    unrelated_path.write_bytes(unrelated_path.read_bytes() + b"\n")
    ledger_path = next((run_directory / "archive/forced").glob("*/archive_manifest.json"))
    before = _tree_snapshot(run_directory)

    monkeypatch.setattr(runner_module, "persist_manifest", real_persist)
    with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
        resume_experiment(run_directory, project_root=PROJECT_ROOT)

    assert _tree_snapshot(run_directory) == before
    assert not (run_directory / "reports/report.md").exists()
    assert (
        ForcedArchiveLedger.model_validate_json(ledger_path.read_text(encoding="utf-8")).state
        == "outputs_archived"
    )


def test_force_archive_recovery_preflight_rejects_corrupt_archive_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fast_config(tmp_path / "artifacts")
    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="force-archive-corrupt-evidence",
    )
    real_persist = runner_module.persist_manifest

    def crash_before_invalidated_manifest(store, manifest):
        if manifest.stages["generate_reports"].status is StageStatus.INVALIDATED:
            raise KeyboardInterrupt("simulated archive transaction termination")
        return real_persist(store, manifest)

    monkeypatch.setattr(runner_module, "persist_manifest", crash_before_invalidated_manifest)
    with pytest.raises(KeyboardInterrupt, match="archive transaction"):
        resume_experiment(
            run_directory,
            force_stage="generate_reports",
            project_root=PROJECT_ROOT,
        )

    ledger_path = next((run_directory / "archive/forced").glob("*/archive_manifest.json"))
    ledger = ForcedArchiveLedger.model_validate_json(ledger_path.read_text(encoding="utf-8"))
    archived_output = tuple(output for stage in ledger.stages for output in stage.archived_outputs)[
        -1
    ]
    archived_path = run_directory / archived_output.path
    archived_path.write_bytes(archived_path.read_bytes() + b"corrupt")
    before = _tree_snapshot(run_directory)

    monkeypatch.setattr(runner_module, "persist_manifest", real_persist)
    with pytest.raises(ArtifactIntegrityError, match="archive copy is missing or corrupt"):
        resume_experiment(run_directory, project_root=PROJECT_ROOT)

    assert _tree_snapshot(run_directory) == before
    assert (
        ForcedArchiveLedger.model_validate_json(ledger_path.read_text(encoding="utf-8")).state
        == "outputs_archived"
    )


def test_force_archive_recovery_resolves_required_secrets_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "IQL_FORCE_ARCHIVE_RESUME_SECRET"
    monkeypatch.setenv(variable, "ephemeral-test-secret")
    raw = _fast_config(tmp_path / "artifacts").canonical_dict()
    raw["security"]["secrets"] = {"resume_token": {"env": variable, "required": True}}
    config = ExperimentConfig.model_validate(raw)
    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="force-archive-missing-secret",
    )
    real_persist = runner_module.persist_manifest

    def crash_before_invalidated_manifest(store, manifest):
        if manifest.stages["generate_reports"].status is StageStatus.INVALIDATED:
            raise KeyboardInterrupt("simulated archive transaction termination")
        return real_persist(store, manifest)

    monkeypatch.setattr(runner_module, "persist_manifest", crash_before_invalidated_manifest)
    with pytest.raises(KeyboardInterrupt, match="archive transaction"):
        resume_experiment(
            run_directory,
            force_stage="generate_reports",
            project_root=PROJECT_ROOT,
        )

    monkeypatch.delenv(variable)
    before = _tree_snapshot(run_directory)
    monkeypatch.setattr(runner_module, "persist_manifest", real_persist)

    with pytest.raises(ConfigurationError, match=variable):
        resume_experiment(run_directory, project_root=PROJECT_ROOT)

    assert _tree_snapshot(run_directory) == before


def test_resume_loads_config_from_committed_force_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _fast_config(tmp_path / "artifacts")
    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="force-config-crash",
    )
    real_execute = runner_module._execute_pipeline

    def crash_after_invalidation(*args, **kwargs):
        raise KeyboardInterrupt("simulated post-invalidation termination")

    monkeypatch.setattr(runner_module, "_execute_pipeline", crash_after_invalidation)
    with pytest.raises(KeyboardInterrupt, match="post-invalidation"):
        resume_experiment(
            run_directory,
            force_stage="resolve_configuration",
            project_root=PROJECT_ROOT,
        )
    # The old successful config is archived, then immediately re-materialized as
    # failure-safe topology before the restarted stage executes.
    assert load_config(run_directory / "resolved_config.yaml") == config
    assert (run_directory / "environment.json").is_file()
    assert all(
        (run_directory / category).is_dir()
        for category in ("metrics", "routing", "checkpoints", "reports")
    )
    assert load_manifest(run_directory).stages["resolve_configuration"].status is (
        StageStatus.INVALIDATED
    )

    monkeypatch.setattr(runner_module, "_execute_pipeline", real_execute)
    assert resume_experiment(run_directory, project_root=PROJECT_ROOT) == run_directory
    assert (run_directory / "resolved_config.yaml").is_file()
    assert load_manifest(run_directory).status is RunStatus.SUCCESS
