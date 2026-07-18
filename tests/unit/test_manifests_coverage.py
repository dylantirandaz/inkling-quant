from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

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
    load_manifest,
    persist_manifest,
    utc_now,
    verify_outputs,
)

pytestmark = pytest.mark.unit
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _manifest(*, stage: StageRecord | None = None) -> RunManifest:
    record = stage or StageRecord(name="evaluate")
    return RunManifest(
        run_id="run-coverage",
        config_hash="a" * 64,
        model=ModelProvenance(id="local://fixtures/tiny", revision="v1"),
        stages={record.name: record},
    )


def _successful_stage(*, outputs: tuple[ArtifactChecksum, ...] = ()) -> StageRecord:
    return StageRecord(
        name="evaluate",
        status=StageStatus.SUCCESS,
        fingerprint="fingerprint",
        started_at=NOW,
        completed_at=NOW,
        outputs=outputs,
        attempt=1,
    )


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"status": StageStatus.SUCCESS}, "successful stage requires"),
        ({"status": StageStatus.FAILED}, "failed stage requires"),
        ({"status": StageStatus.UNSUPPORTED}, "unsupported stage requires"),
    ],
)
def test_stage_record_requires_state_specific_evidence(fields, message):
    with pytest.raises(ValidationError, match=message):
        StageRecord(name="evaluate", **fields)


def test_utc_now_is_timezone_aware_utc():
    timestamp = utc_now()

    assert timestamp.tzinfo is UTC
    assert timestamp.utcoffset() is not None


def test_run_cannot_be_started_twice():
    running = _manifest().start(at=NOW)

    with pytest.raises(ArtifactIntegrityError, match="running -> running"):
        running.start(at=NOW)


def test_stage_can_start_only_while_run_is_running():
    with pytest.raises(ArtifactIntegrityError, match="only while a run is running"):
        _manifest().start_stage("evaluate", "fingerprint", at=NOW)


@pytest.mark.parametrize("status", [StageStatus.RUNNING, StageStatus.SUCCESS])
def test_completed_or_active_stage_cannot_be_started(status):
    if status is StageStatus.SUCCESS:
        stage = _successful_stage()
    else:
        stage = StageRecord(
            name="evaluate",
            status=status,
            fingerprint="old-fingerprint",
            started_at=NOW,
            attempt=1,
        )
    running = _manifest(stage=stage).model_copy(update={"status": RunStatus.RUNNING})

    with pytest.raises(ArtifactIntegrityError, match="Invalid stage transition"):
        running.start_stage("evaluate", "new-fingerprint", at=NOW)


@pytest.mark.parametrize("status", [StageStatus.FAILED, StageStatus.INVALIDATED])
def test_failed_or_invalidated_stage_restarts_with_clean_attempt(status):
    stage = StageRecord(
        name="evaluate",
        status=status,
        fingerprint="old-fingerprint",
        started_at=NOW,
        completed_at=NOW if status is StageStatus.FAILED else None,
        outputs=(ArtifactChecksum(path="old.json", sha256="b" * 64, size_bytes=1),),
        error=(
            StageError(code="OLD_ERROR", message="old failure")
            if status is StageStatus.FAILED
            else None
        ),
        reason="old reason",
        attempt=2,
    )
    running = _manifest(stage=stage).model_copy(update={"status": RunStatus.RUNNING})

    restarted = running.start_stage("evaluate", "new-fingerprint", at=NOW).stages["evaluate"]

    assert restarted.status is StageStatus.RUNNING
    assert restarted.fingerprint == "new-fingerprint"
    assert restarted.attempt == 3
    assert restarted.outputs == ()
    assert restarted.completed_at is None
    assert restarted.error is None
    assert restarted.reason is None


def test_pending_stage_cannot_fail():
    running = _manifest().start(at=NOW)
    error = StageError(code="EVALUATION_ERROR", message="failure")

    with pytest.raises(ArtifactIntegrityError, match="Invalid stage transition"):
        running.fail_stage("evaluate", error, at=NOW)


def test_mark_stage_rejects_success_status():
    running = _manifest().start(at=NOW)

    with pytest.raises(ArtifactIntegrityError, match="does not support status"):
        running.mark_stage("evaluate", StageStatus.SUCCESS, "not supported", at=NOW)


def test_completed_stage_cannot_be_marked_again():
    stage = StageRecord(
        name="evaluate",
        required=False,
        status=StageStatus.UNSUPPORTED,
        reason="not available",
        completed_at=NOW,
    )
    running = _manifest(stage=stage).model_copy(update={"status": RunStatus.RUNNING})

    with pytest.raises(ArtifactIntegrityError, match="Cannot mark stage"):
        running.mark_stage(
            "evaluate", StageStatus.SKIPPED_NOT_REQUIRED, "still unavailable", at=NOW
        )


def test_invalidate_changes_only_terminal_stages():
    records = {
        "success": StageRecord(
            name="success",
            status=StageStatus.SUCCESS,
            fingerprint="fingerprint",
            started_at=NOW,
            completed_at=NOW,
        ),
        "failed": StageRecord(
            name="failed",
            status=StageStatus.FAILED,
            fingerprint="fingerprint",
            started_at=NOW,
            completed_at=NOW,
            error=StageError(code="ERROR", message="failed"),
        ),
        "running": StageRecord(name="running", status=StageStatus.RUNNING, started_at=NOW),
        "pending": StageRecord(name="pending"),
    }
    manifest = _manifest().model_copy(update={"stages": records})

    invalidated = manifest.invalidate(set(records))

    assert invalidated.stages["success"].status is StageStatus.INVALIDATED
    assert invalidated.stages["failed"].status is StageStatus.INVALIDATED
    assert invalidated.stages["running"].status is StageStatus.RUNNING
    assert invalidated.stages["pending"].status is StageStatus.PENDING
    assert invalidated.stages["success"].completed_at is None
    assert invalidated.stages["failed"].error is None
    assert invalidated.stages["success"].reason == "invalidated by forced stage execution"
    assert manifest.stages["success"].status is StageStatus.SUCCESS


def test_pending_run_cannot_fail():
    with pytest.raises(ArtifactIntegrityError, match="pending -> failed"):
        _manifest().fail(at=NOW)


def test_manifest_persistence_creates_then_replaces_ledger(tmp_path):
    store = LocalArtifactStore(tmp_path)
    pending = _manifest()

    path = persist_manifest(store, pending)
    running = pending.start(at=NOW)
    replaced_path = persist_manifest(store, running)

    assert path == replaced_path
    assert load_manifest(tmp_path) == running


def test_load_manifest_wraps_read_failures(tmp_path):
    with pytest.raises(ArtifactIntegrityError, match="Unable to read manifest"):
        load_manifest(tmp_path)


def test_verify_outputs_accepts_intact_artifact(tmp_path):
    store = LocalArtifactStore(tmp_path)
    path = store.write_text("metrics/value.json", "intact")
    output = ArtifactChecksum(
        path="metrics/value.json",
        sha256=store.checksum("metrics/value.json"),
        size_bytes=path.stat().st_size,
    )

    verify_outputs(tmp_path, _successful_stage(outputs=(output,)))


@pytest.mark.parametrize("path", ["metrics/missing.json", "../outside.json"])
def test_verify_outputs_rejects_missing_or_escaping_artifact(tmp_path, path):
    output = ArtifactChecksum(path=path, sha256="a" * 64, size_bytes=1)

    with pytest.raises(ArtifactIntegrityError, match="Missing or unsafe output"):
        verify_outputs(tmp_path, _successful_stage(outputs=(output,)))


def test_verify_outputs_rejects_size_mismatch_even_when_digest_matches(tmp_path):
    store = LocalArtifactStore(tmp_path)
    path = store.write_text("metrics/value.json", "intact")
    output = ArtifactChecksum(
        path="metrics/value.json",
        sha256=store.checksum("metrics/value.json"),
        size_bytes=path.stat().st_size + 1,
    )

    with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
        verify_outputs(tmp_path, _successful_stage(outputs=(output,)))
