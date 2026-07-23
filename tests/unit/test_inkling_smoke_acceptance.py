from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from inkling_quant_lab.gguf.inkling_smoke_acceptance import (
    SMOKE_POST_SPAWN_ACCEPTANCE_MAX_BYTES,
    SmokePostSpawnAcceptance,
    smoke_post_spawn_acceptance_path,
    validate_smoke_post_spawn_acceptance,
)
from inkling_quant_lab.gguf.inkling_smoke_execution import smoke_deployment_tag

ACCEPTED_AT = "2026-07-22T12:34:56.123456Z"
RUN_ID = "inkling-smoke-test"
LAUNCH_INTENT_SHA256 = "a" * 64
CALL_ID = "fc-01KXYZabc123"
CONTROL_PLANE_SHA256 = "b" * 64
APP_NAME = f"inkling-q3-smoke-{CONTROL_PLANE_SHA256[:12]}"
DEPLOYMENT_TAG = smoke_deployment_tag(CONTROL_PLANE_SHA256)
FUNCTION_ID = "fu-Abc123"
ATTEMPT_REGISTRY_NAME = "inkling-smoke-attempt-registry-v1"
ATTEMPT_REGISTRY_ID = "di-Attempt123"
ATTEMPT_REGISTRY_CREATED_AT_UTC = "2026-07-22T12:00:00.000000Z"
SMOKE_CONFIG_HASH = "c" * 64
EXPORT_REFERENCE_SHA256 = "d" * 64


def _record(**overrides: Any) -> SmokePostSpawnAcceptance:
    values: dict[str, Any] = {
        "accepted_at": ACCEPTED_AT,
        "run_id": RUN_ID,
        "launch_intent_sha256": LAUNCH_INTENT_SHA256,
        "call_id": CALL_ID,
        "app_name": APP_NAME,
        "deployment_version": 7,
        "deployment_tag": DEPLOYMENT_TAG,
        "function_id": FUNCTION_ID,
        "attempt_registry_name": ATTEMPT_REGISTRY_NAME,
        "attempt_registry_id": ATTEMPT_REGISTRY_ID,
        "attempt_registry_created_at_utc": ATTEMPT_REGISTRY_CREATED_AT_UTC,
        "smoke_config_hash": SMOKE_CONFIG_HASH,
        "verified_export_reference_sha256": EXPORT_REFERENCE_SHA256,
        "control_plane_sha256": CONTROL_PLANE_SHA256,
    }
    values.update(overrides)
    return SmokePostSpawnAcceptance(**values)


def _expected_arguments(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "evidence_path": smoke_post_spawn_acceptance_path(
            RUN_ID,
            LAUNCH_INTENT_SHA256,
        ),
        "accepted_at": ACCEPTED_AT,
        "run_id": RUN_ID,
        "launch_intent_sha256": LAUNCH_INTENT_SHA256,
        "call_id": CALL_ID,
        "app_name": APP_NAME,
        "environment_name": "inkling-quant",
        "deployment_version": 7,
        "deployment_tag": DEPLOYMENT_TAG,
        "function_id": FUNCTION_ID,
        "function_name": "smoke_test",
        "attempt_registry_name": ATTEMPT_REGISTRY_NAME,
        "attempt_registry_id": ATTEMPT_REGISTRY_ID,
        "attempt_registry_created_at_utc": ATTEMPT_REGISTRY_CREATED_AT_UTC,
        "smoke_config_hash": SMOKE_CONFIG_HASH,
        "verified_export_reference_sha256": EXPORT_REFERENCE_SHA256,
        "control_plane_sha256": CONTROL_PLANE_SHA256,
    }
    values.update(overrides)
    return values


def _validate(
    payload: bytes,
    *,
    acceptance_sha256: str | None = None,
    **overrides: Any,
) -> SmokePostSpawnAcceptance:
    expected = _expected_arguments(**overrides)
    return validate_smoke_post_spawn_acceptance(
        payload,
        acceptance_sha256=(
            hashlib.sha256(payload).hexdigest() if acceptance_sha256 is None else acceptance_sha256
        ),
        **expected,
    )


def _canonical_bytes(raw: dict[str, Any]) -> bytes:
    return (
        json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def test_acceptance_validates_exact_canonical_bytes() -> None:
    record = _record()
    payload = record.canonical_bytes()

    assert payload == _canonical_bytes(record.model_dump(mode="json"))
    assert payload.endswith(b"\n")
    assert record.acceptance_sha256() == hashlib.sha256(payload).hexdigest()
    assert _validate(payload) == record


def test_acceptance_is_frozen() -> None:
    record = _record()

    with pytest.raises(ValidationError):
        record.call_id = "fc-other"


def test_acceptance_path_is_fixed_by_run_and_launch_intent() -> None:
    assert smoke_post_spawn_acceptance_path(RUN_ID, LAUNCH_INTENT_SHA256) == (
        f"runs/{RUN_ID}/control/post-spawn-acceptances/{LAUNCH_INTENT_SHA256}.json"
    )
    with pytest.raises(ValueError, match="run ID"):
        smoke_post_spawn_acceptance_path("../escape", LAUNCH_INTENT_SHA256)
    with pytest.raises(ValueError, match="SHA-256"):
        smoke_post_spawn_acceptance_path(RUN_ID, "A" * 64)


@pytest.mark.parametrize(
    "payload_transform",
    (
        lambda payload: payload[:-1],
        lambda payload: b" " + payload,
        lambda payload: payload + b"\n",
        lambda payload: json.dumps(json.loads(payload), indent=2).encode("utf-8") + b"\n",
    ),
)
def test_acceptance_rejects_noncanonical_bytes(payload_transform: Any) -> None:
    payload = payload_transform(_record().canonical_bytes())

    with pytest.raises(ValueError, match="not canonical"):
        _validate(payload)


def test_acceptance_rejects_duplicate_keys() -> None:
    payload = _record().canonical_bytes()
    duplicate = payload[:-2] + b',"run_id":"duplicate"}\n'

    with pytest.raises(ValueError, match="duplicate key"):
        _validate(duplicate)


def test_acceptance_rejects_oversize_or_non_byte_payload() -> None:
    oversized = b"{" + b" " * SMOKE_POST_SPAWN_ACCEPTANCE_MAX_BYTES
    with pytest.raises(ValueError, match="size limit"):
        _validate(oversized)
    with pytest.raises(TypeError, match="must be bytes"):
        validate_smoke_post_spawn_acceptance(  # type: ignore[arg-type]
            _record().canonical_bytes().decode("utf-8"),
            acceptance_sha256="0" * 64,
            **_expected_arguments(),
        )


def test_acceptance_rejects_extra_fields() -> None:
    raw = _record().model_dump(mode="json") | {"unexpected": True}
    payload = _canonical_bytes(raw)

    with pytest.raises(ValueError, match="schema"):
        _validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "inkling-smoke-post-spawn-acceptance-v2"),
        ("status", "accepted"),
        ("accepted_at", "2026-07-22T12:34:56Z"),
        ("stage", "verify_export"),
        ("call_id", "fc-invalid-hyphen"),
        ("environment_name", "main"),
        ("deployment_version", "7"),
        ("function_name", "other"),
        ("attempt_registry_name", "other-registry"),
        ("attempt_registry_created_at_utc", "2026-07-22T12:00:00Z"),
        ("app_name", "inkling-q3-smoke-" + "0" * 12),
        ("deployment_tag", "iql-smoke-" + "0" * 64),
    ),
)
def test_acceptance_rejects_invalid_schema_values(field: str, value: object) -> None:
    raw = _record().model_dump(mode="json")
    raw[field] = value
    payload = _canonical_bytes(raw)

    with pytest.raises(ValueError, match="schema"):
        _validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("accepted_at", "2026-07-22T12:34:57.123456Z"),
        ("run_id", "different-run"),
        ("launch_intent_sha256", "e" * 64),
        ("call_id", "fc-Other123"),
        ("deployment_version", 8),
        ("function_id", "fu-Other123"),
        ("attempt_registry_id", "di-Other123"),
        (
            "attempt_registry_created_at_utc",
            "2026-07-22T12:00:01.000000Z",
        ),
        ("smoke_config_hash", "e" * 64),
        ("verified_export_reference_sha256", "e" * 64),
    ),
)
def test_acceptance_rejects_exact_binding_drift(field: str, value: object) -> None:
    raw = _record().model_dump(mode="json")
    raw[field] = value
    payload = _canonical_bytes(raw)

    with pytest.raises(ValueError, match="exact accepted call"):
        _validate(payload)


def test_acceptance_rejects_deployment_identity_drift() -> None:
    alternate_control = "e" * 64
    payload = _record(
        control_plane_sha256=alternate_control,
        app_name=f"inkling-q3-smoke-{alternate_control[:12]}",
        deployment_tag=smoke_deployment_tag(alternate_control),
    ).canonical_bytes()

    with pytest.raises(ValueError, match="exact accepted call"):
        _validate(payload)


def test_acceptance_rejects_wrong_path_or_hash() -> None:
    payload = _record().canonical_bytes()

    with pytest.raises(ValueError, match="fixed evidence path"):
        _validate(payload, evidence_path="runs/wrong/control/acceptance.json")
    with pytest.raises(ValueError, match="does not match"):
        _validate(payload, acceptance_sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256 is invalid"):
        _validate(payload, acceptance_sha256="A" * 64)


@pytest.mark.parametrize(
    ("accepted_at", "message"),
    (
        ("2026-07-22T12:34:56.12345Z", "canonical"),
        ("2026-07-22T12:34:56.123456+00:00", "canonical"),
        ("2026-02-30T12:34:56.123456Z", "real UTC"),
    ),
)
def test_acceptance_requires_real_canonical_utc_microseconds(
    accepted_at: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _record(accepted_at=accepted_at)
