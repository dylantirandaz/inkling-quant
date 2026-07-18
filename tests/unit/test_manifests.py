from __future__ import annotations

from datetime import UTC, datetime

import pytest

from inkling_quant_lab.artifacts import LocalArtifactStore
from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.manifests import (
    ArtifactChecksum,
    ModelProvenance,
    RunManifest,
    RunStatus,
    StageError,
    StageRecord,
    StageStatus,
    verify_outputs,
)
from inkling_quant_lab.pipeline.runner import generate_run_id

pytestmark = pytest.mark.unit
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def manifest(required: bool = True) -> RunManifest:
    return RunManifest(
        run_id="run-test",
        config_hash="a" * 64,
        model=ModelProvenance(id="local://fixtures/tiny", revision="v1"),
        stages={"evaluate": StageRecord(name="evaluate", required=required)},
    )


def test_run_id_is_stable_for_declared_identity_inputs(config_factory) -> None:
    config = config_factory()

    first = generate_run_id(config, now=NOW, nonce="a1b2c3")
    second = generate_run_id(config, now=NOW, nonce="a1b2c3")

    assert first == second
    assert first.startswith("test-20260101T000000000000Z-")
    assert config.config_hash()[:8] in first
    assert first.endswith("-a1b2c3")
    assert generate_run_id(config, now=NOW, nonce="d4e5f6") != first


def test_valid_success_lifecycle():
    running = manifest().start(at=NOW)
    stage_running = running.start_stage("evaluate", "fingerprint", at=NOW)
    stage_success = stage_running.finish_stage("evaluate", (), at=NOW)
    complete = stage_success.succeed(at=NOW)

    assert complete.status is RunStatus.SUCCESS
    assert complete.stages["evaluate"].status is StageStatus.SUCCESS


def test_valid_failed_lifecycle():
    running = manifest().start(at=NOW).start_stage("evaluate", "fingerprint", at=NOW)
    failed_stage = running.fail_stage(
        "evaluate", StageError(code="EVALUATION_ERROR", message="broken"), at=NOW
    )
    failed = failed_stage.fail(at=NOW)
    assert failed.status is RunStatus.FAILED


def test_invalid_transitions_fail():
    with pytest.raises(ArtifactIntegrityError, match="Invalid run transition"):
        manifest().succeed(at=NOW)
    with pytest.raises(ArtifactIntegrityError, match="Invalid stage transition"):
        manifest().start(at=NOW).finish_stage("evaluate", (), at=NOW)


def test_required_unsupported_stage_prevents_success():
    running = manifest(required=True).start(at=NOW)
    unsupported = running.mark_stage(
        "evaluate", StageStatus.UNSUPPORTED, "dense routing unsupported", at=NOW
    )
    with pytest.raises(ArtifactIntegrityError, match="required stages"):
        unsupported.succeed(at=NOW)


def test_optional_unsupported_stage_allows_success():
    running = manifest(required=False).start(at=NOW)
    unsupported = running.mark_stage(
        "evaluate", StageStatus.UNSUPPORTED, "optional sensor unavailable", at=NOW
    )
    assert unsupported.succeed(at=NOW).status is RunStatus.SUCCESS


def test_mutated_completed_output_fails_integrity(tmp_path):
    store = LocalArtifactStore(tmp_path)
    path = store.write_text("metrics/value.json", "original")
    checksum = store.checksum("metrics/value.json")
    record = StageRecord(
        name="evaluate",
        status=StageStatus.SUCCESS,
        fingerprint="fingerprint",
        started_at=NOW,
        completed_at=NOW,
        outputs=(
            ArtifactChecksum(
                path="metrics/value.json", sha256=checksum, size_bytes=path.stat().st_size
            ),
        ),
    )
    path.write_text("mutated", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
        verify_outputs(tmp_path, record)
