from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from inkling_quant_lab.gguf.inkling_smoke_acceptance import (
    smoke_post_spawn_acceptance_path,
)
from inkling_quant_lab.gguf.inkling_smoke_attempt import (
    SMOKE_ATTEMPT_REGISTRY_MAX_BYTES,
    SMOKE_ATTEMPT_REGISTRY_NAME,
    SmokeAttemptRegistryClaim,
    claim_smoke_attempt,
    smoke_attempt_registry_key,
    validate_smoke_attempt_registry_claim,
)

RUN_ID = "inkling-smoke-test"
CALL_ID = "fc-01KXYZabc123"
INPUT_ID = "in-01KXYZabc123:0-0"
TASK_ID = "ta-01KXYZabc123"
LAUNCH_INTENT_SHA256 = "a" * 64
POST_SPAWN_ACCEPTANCE_SHA256 = "b" * 64
SMOKE_CONFIG_HASH = "c" * 64
CONTROL_PLANE_SHA256 = "d" * 64
REGISTRY_ID = "di-Attempt123"
REGISTRY_CREATED_AT_UTC = "2026-07-22T12:00:00.000000Z"


def _record(**overrides: Any) -> SmokeAttemptRegistryClaim:
    values: dict[str, Any] = {
        "registry_id": REGISTRY_ID,
        "registry_created_at_utc": REGISTRY_CREATED_AT_UTC,
        "registry_key": smoke_attempt_registry_key(RUN_ID),
        "run_id": RUN_ID,
        "call_id": CALL_ID,
        "input_id": INPUT_ID,
        "task_id": TASK_ID,
        "launch_intent_sha256": LAUNCH_INTENT_SHA256,
        "post_spawn_acceptance_path": smoke_post_spawn_acceptance_path(
            RUN_ID,
            LAUNCH_INTENT_SHA256,
        ),
        "post_spawn_acceptance_sha256": POST_SPAWN_ACCEPTANCE_SHA256,
        "smoke_config_hash": SMOKE_CONFIG_HASH,
        "control_plane_sha256": CONTROL_PLANE_SHA256,
    }
    values.update(overrides)
    return SmokeAttemptRegistryClaim(**values)


def _expected_arguments(**overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "registry_name": SMOKE_ATTEMPT_REGISTRY_NAME,
        "registry_id": REGISTRY_ID,
        "registry_created_at_utc": REGISTRY_CREATED_AT_UTC,
        "registry_key": smoke_attempt_registry_key(RUN_ID),
        "run_id": RUN_ID,
        "stage": "smoke_test",
        "call_id": CALL_ID,
        "input_id": INPUT_ID,
        "task_id": TASK_ID,
        "launch_intent_sha256": LAUNCH_INTENT_SHA256,
        "post_spawn_acceptance_path": smoke_post_spawn_acceptance_path(
            RUN_ID,
            LAUNCH_INTENT_SHA256,
        ),
        "post_spawn_acceptance_sha256": POST_SPAWN_ACCEPTANCE_SHA256,
        "smoke_config_hash": SMOKE_CONFIG_HASH,
        "control_plane_sha256": CONTROL_PLANE_SHA256,
    }
    values.update(overrides)
    return values


def _validate(
    payload: bytes,
    *,
    claim_sha256: str | None = None,
    **overrides: Any,
) -> SmokeAttemptRegistryClaim:
    expected = _expected_arguments(**overrides)
    return validate_smoke_attempt_registry_claim(
        payload,
        claim_sha256=(
            hashlib.sha256(payload).hexdigest() if claim_sha256 is None else claim_sha256
        ),
        **expected,
    )


def _canonical_bytes(raw: dict[str, Any]) -> bytes:
    return (
        json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


class _FakeAttemptRegistry:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.calls: list[tuple[str, bytes, bool]] = []

    def put(
        self,
        key: Any,
        value: Any,
        *,
        skip_if_exists: bool = False,
    ) -> bool:
        assert isinstance(key, str)
        assert isinstance(value, bytes)
        self.calls.append((key, value, skip_if_exists))
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True


def test_attempt_claim_validates_exact_canonical_bytes() -> None:
    record = _record()
    payload = record.canonical_bytes()

    assert payload == _canonical_bytes(record.model_dump(mode="json"))
    assert payload.endswith(b"\n")
    assert record.claim_sha256() == hashlib.sha256(payload).hexdigest()
    assert _validate(payload) == record


def test_attempt_registry_allows_only_the_first_atomic_claim() -> None:
    registry = _FakeAttemptRegistry()
    claim = _record()

    observed_sha256 = claim_smoke_attempt(registry, claim)

    assert observed_sha256 == claim.claim_sha256()
    assert registry.values == {claim.registry_key: claim.canonical_bytes()}
    assert registry.calls == [(claim.registry_key, claim.canonical_bytes(), True)]


@pytest.mark.parametrize(
    "competing_claim",
    (
        _record(),
        _record(task_id="ta-OtherTask123"),
        _record(
            call_id="fc-OtherCall123",
            input_id="in-OtherInput123:0-0",
            task_id="ta-OtherTask123",
        ),
        _record(
            launch_intent_sha256="e" * 64,
            post_spawn_acceptance_path=smoke_post_spawn_acceptance_path(
                RUN_ID,
                "e" * 64,
            ),
        ),
    ),
    ids=("replay", "reschedule", "competing-invocation", "different-launch-intent"),
)
def test_attempt_registry_rejects_every_replay_after_first_claim(
    competing_claim: SmokeAttemptRegistryClaim,
) -> None:
    registry = _FakeAttemptRegistry()
    owner = _record()
    claim_smoke_attempt(registry, owner)

    with pytest.raises(RuntimeError, match="already been consumed"):
        claim_smoke_attempt(registry, competing_claim)

    assert registry.values == {owner.registry_key: owner.canonical_bytes()}
    assert registry.calls[-1] == (
        competing_claim.registry_key,
        competing_claim.canonical_bytes(),
        True,
    )


def test_attempt_claim_is_frozen() -> None:
    record = _record()

    with pytest.raises(ValidationError):
        record.call_id = "fc-Other123"


def test_attempt_registry_key_depends_only_on_exact_run_and_stage() -> None:
    assert smoke_attempt_registry_key(RUN_ID) == f"{RUN_ID}:smoke_test"
    assert smoke_attempt_registry_key(RUN_ID) == smoke_attempt_registry_key(RUN_ID)
    assert smoke_attempt_registry_key("other-run") != smoke_attempt_registry_key(RUN_ID)
    other_launch = "e" * 64
    assert (
        _record().registry_key
        == _record(
            launch_intent_sha256=other_launch,
            post_spawn_acceptance_path=smoke_post_spawn_acceptance_path(
                RUN_ID,
                other_launch,
            ),
        ).registry_key
    )
    with pytest.raises(ValueError, match="run ID"):
        smoke_attempt_registry_key("../escape")
    with pytest.raises(ValueError, match="stage"):
        smoke_attempt_registry_key(RUN_ID, "verify_export")


@pytest.mark.parametrize(
    "payload_transform",
    (
        lambda payload: payload[:-1],
        lambda payload: b" " + payload,
        lambda payload: payload + b"\n",
        lambda payload: json.dumps(json.loads(payload), indent=2).encode("utf-8") + b"\n",
    ),
)
def test_attempt_claim_rejects_noncanonical_bytes(payload_transform: Any) -> None:
    payload = payload_transform(_record().canonical_bytes())

    with pytest.raises(ValueError, match="not canonical"):
        _validate(payload)


def test_attempt_claim_rejects_duplicate_keys() -> None:
    payload = _record().canonical_bytes()
    duplicate = payload[:-2] + b',"run_id":"duplicate"}\n'

    with pytest.raises(ValueError, match="duplicate key"):
        _validate(duplicate)


def test_attempt_claim_rejects_oversize_or_non_byte_payload() -> None:
    oversized = b"{" + b" " * SMOKE_ATTEMPT_REGISTRY_MAX_BYTES
    with pytest.raises(ValueError, match="size limit"):
        _validate(oversized)
    with pytest.raises(TypeError, match="must be bytes"):
        validate_smoke_attempt_registry_claim(  # type: ignore[arg-type]
            _record().canonical_bytes().decode("utf-8"),
            claim_sha256="0" * 64,
            **_expected_arguments(),
        )


def test_attempt_claim_rejects_extra_fields() -> None:
    raw = _record().model_dump(mode="json") | {"unexpected": True}

    with pytest.raises(ValueError, match="schema"):
        _validate(_canonical_bytes(raw))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "inkling-smoke-attempt-registry-claim-v1"),
        ("registry_name", "different-registry"),
        ("registry_id", "di-invalid-hyphen"),
        ("registry_created_at_utc", "2026-07-22T12:00:00Z"),
        ("stage", "verify_export"),
        ("call_id", "fc-invalid-hyphen"),
        ("input_id", "input-invalid"),
        ("task_id", "task-invalid"),
        ("launch_intent_sha256", "A" * 64),
        ("post_spawn_acceptance_sha256", "A" * 64),
        ("smoke_config_hash", "A" * 64),
        ("control_plane_sha256", "A" * 64),
    ),
)
def test_attempt_claim_rejects_invalid_schema_values(
    field: str,
    value: object,
) -> None:
    raw = _record().model_dump(mode="json")
    raw[field] = value

    with pytest.raises(ValueError, match="schema"):
        _validate(_canonical_bytes(raw))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("registry_id", "di-Other123"),
        ("registry_created_at_utc", "2026-07-22T12:00:01.000000Z"),
        ("call_id", "fc-Other123"),
        ("input_id", "in-Other123:0-0"),
        ("task_id", "ta-Other123"),
        ("post_spawn_acceptance_sha256", "e" * 64),
        ("smoke_config_hash", "e" * 64),
        ("control_plane_sha256", "e" * 64),
    ),
)
def test_attempt_claim_rejects_exact_invocation_binding_drift(
    field: str,
    value: str,
) -> None:
    raw = _record().model_dump(mode="json")
    raw[field] = value

    with pytest.raises(ValueError, match="exact invocation"):
        _validate(_canonical_bytes(raw))


def test_attempt_claim_rejects_run_launch_and_derived_path_drift() -> None:
    other_run = "other-run"
    other_launch = "e" * 64
    payload = _record(
        run_id=other_run,
        registry_key=smoke_attempt_registry_key(other_run),
        launch_intent_sha256=other_launch,
        post_spawn_acceptance_path=smoke_post_spawn_acceptance_path(
            other_run,
            other_launch,
        ),
    ).canonical_bytes()

    with pytest.raises(ValueError, match="exact invocation"):
        _validate(payload)


def test_attempt_claim_rejects_registry_key_or_acceptance_path_drift() -> None:
    raw = _record().model_dump(mode="json")
    raw["registry_key"] = smoke_attempt_registry_key("other-run")
    with pytest.raises(ValueError, match="schema"):
        _validate(_canonical_bytes(raw))

    raw = _record().model_dump(mode="json")
    raw["post_spawn_acceptance_path"] = smoke_post_spawn_acceptance_path(
        "other-run",
        LAUNCH_INTENT_SHA256,
    )
    with pytest.raises(ValueError, match="schema"):
        _validate(_canonical_bytes(raw))


def test_attempt_claim_rejects_wrong_external_hash() -> None:
    payload = _record().canonical_bytes()

    with pytest.raises(ValueError, match="does not match"):
        _validate(payload, claim_sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA-256 is invalid"):
        _validate(payload, claim_sha256="A" * 64)


def test_attempt_claim_rejects_wrong_expected_registry_identity() -> None:
    payload = _record().canonical_bytes()

    with pytest.raises(ValueError, match=r"expected.*bindings"):
        _validate(payload, registry_name="different-registry")
    with pytest.raises(ValueError, match="exact invocation"):
        _validate(payload, registry_id="di-Other123")
    with pytest.raises(ValueError, match="exact invocation"):
        _validate(
            payload,
            registry_created_at_utc="2026-07-22T12:00:01.000000Z",
        )
    with pytest.raises(ValueError, match=r"expected.*bindings"):
        _validate(payload, registry_key=smoke_attempt_registry_key("other-run"))
