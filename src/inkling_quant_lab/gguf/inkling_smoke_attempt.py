"""Strict server-side attempt claim for the Inkling Modal smoke stage."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any, Literal, Protocol

from pydantic import ConfigDict, Field, StrictStr, ValidationError, model_validator

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.gguf.inkling_smoke_acceptance import (
    smoke_post_spawn_acceptance_path,
)
from inkling_quant_lab.gguf.inkling_smoke_execution import (
    SMOKE_STAGE,
    strict_json_object,
    validate_smoke_attempt_registry_created_at_utc,
)

SMOKE_ATTEMPT_REGISTRY_SCHEMA = "inkling-smoke-attempt-registry-claim-v2"
SMOKE_ATTEMPT_REGISTRY_NAME = "inkling-smoke-attempt-registry-v1"
SMOKE_ATTEMPT_REGISTRY_MAX_BYTES = 64 * 1024

_RUN_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,95}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def smoke_attempt_registry_key(
    run_id: str,
    stage: str = SMOKE_STAGE,
) -> str:
    """Return the one registry key for an exact run and stage."""

    if re.fullmatch(_RUN_ID_PATTERN, run_id) is None:
        raise ValueError("smoke run ID is invalid")
    if stage != SMOKE_STAGE:
        raise ValueError("smoke attempt stage is invalid")
    return f"{run_id}:{stage}"


class SmokeAttemptRegistryClaim(StrictFrozenModel):
    """One immutable server-side claim of the smoke run's only attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    schema_version: Literal["inkling-smoke-attempt-registry-claim-v2"] = (
        "inkling-smoke-attempt-registry-claim-v2"
    )
    registry_name: Literal["inkling-smoke-attempt-registry-v1"] = (
        "inkling-smoke-attempt-registry-v1"
    )
    registry_id: StrictStr = Field(pattern=r"^di-[A-Za-z0-9]+$")
    registry_created_at_utc: StrictStr
    registry_key: StrictStr
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    stage: Literal["smoke_test"] = "smoke_test"
    call_id: StrictStr = Field(pattern=r"^fc-[A-Za-z0-9]+$")
    input_id: StrictStr = Field(pattern=r"^in-[A-Za-z0-9]+(?::[0-9]+-[0-9]+)?$")
    task_id: StrictStr = Field(pattern=r"^ta-[A-Za-z0-9]+$")
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    post_spawn_acceptance_path: StrictStr
    post_spawn_acceptance_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    smoke_config_hash: StrictStr = Field(pattern=_SHA256_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def derived_bindings_are_exact(self) -> SmokeAttemptRegistryClaim:
        validate_smoke_attempt_registry_created_at_utc(self.registry_created_at_utc)
        if self.registry_key != smoke_attempt_registry_key(self.run_id, self.stage):
            raise ValueError("smoke attempt registry key differs from its run and stage")
        expected_acceptance_path = smoke_post_spawn_acceptance_path(
            self.run_id,
            self.launch_intent_sha256,
        )
        if self.post_spawn_acceptance_path != expected_acceptance_path:
            raise ValueError("smoke attempt acceptance path differs from its run and launch intent")
        return self

    def canonical_bytes(self) -> bytes:
        """Return the only accepted byte encoding for this claim."""

        return _canonical_json_bytes(self.model_dump(mode="json"))

    def claim_sha256(self) -> str:
        """Return the SHA-256 of the exact canonical claim bytes."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class SmokeAttemptRegistryProtocol(Protocol):
    """Minimal atomic Modal Dict operation used by the attempt gate."""

    def put(
        self,
        key: Any,
        value: Any,
        *,
        skip_if_exists: bool = False,
    ) -> bool: ...


def claim_smoke_attempt(
    registry: SmokeAttemptRegistryProtocol,
    claim: SmokeAttemptRegistryClaim,
) -> str:
    """Atomically claim the only attempt and reject every existing key."""

    payload = claim.canonical_bytes()
    created = registry.put(
        claim.registry_key,
        payload,
        skip_if_exists=True,
    )
    if created is not True:
        raise RuntimeError("The one configured smoke attempt has already been consumed")
    return hashlib.sha256(payload).hexdigest()


def validate_smoke_attempt_registry_claim(
    payload: bytes,
    *,
    claim_sha256: str,
    registry_name: str,
    registry_id: str,
    registry_created_at_utc: str,
    registry_key: str,
    run_id: str,
    stage: str,
    call_id: str,
    input_id: str,
    task_id: str,
    launch_intent_sha256: str,
    post_spawn_acceptance_path: str,
    post_spawn_acceptance_sha256: str,
    smoke_config_hash: str,
    control_plane_sha256: str,
) -> SmokeAttemptRegistryClaim:
    """Validate canonical claim bytes against every expected attempt binding."""

    if not isinstance(payload, bytes):
        raise TypeError("smoke attempt registry claim must be bytes")
    if len(payload) > SMOKE_ATTEMPT_REGISTRY_MAX_BYTES:
        raise ValueError("smoke attempt registry claim exceeds its size limit")

    raw = strict_json_object(payload)
    try:
        claim = SmokeAttemptRegistryClaim.model_validate(raw)
    except ValidationError as error:
        raise ValueError("smoke attempt registry claim schema is invalid") from error

    canonical_bytes = claim.canonical_bytes()
    if payload != canonical_bytes:
        raise ValueError("smoke attempt registry claim bytes are not canonical")
    if re.fullmatch(_SHA256_PATTERN, claim_sha256) is None:
        raise ValueError("expected smoke attempt registry claim SHA-256 is invalid")
    if hashlib.sha256(canonical_bytes).hexdigest() != claim_sha256:
        raise ValueError("smoke attempt registry claim hash does not match its bytes")

    try:
        expected = SmokeAttemptRegistryClaim(
            registry_name=registry_name,
            registry_id=registry_id,
            registry_created_at_utc=registry_created_at_utc,
            registry_key=registry_key,
            run_id=run_id,
            stage=stage,
            call_id=call_id,
            input_id=input_id,
            task_id=task_id,
            launch_intent_sha256=launch_intent_sha256,
            post_spawn_acceptance_path=post_spawn_acceptance_path,
            post_spawn_acceptance_sha256=post_spawn_acceptance_sha256,
            smoke_config_hash=smoke_config_hash,
            control_plane_sha256=control_plane_sha256,
        )
    except ValidationError as error:
        raise ValueError("expected smoke attempt registry bindings are invalid") from error
    if expected.registry_name != SMOKE_ATTEMPT_REGISTRY_NAME:
        raise ValueError("smoke attempt registry name is inconsistent")
    if expected.stage != SMOKE_STAGE:
        raise ValueError("smoke attempt stage is inconsistent")
    if claim != expected:
        raise ValueError("smoke attempt registry claim does not bind the exact invocation")
    return claim


__all__ = [
    "SMOKE_ATTEMPT_REGISTRY_MAX_BYTES",
    "SMOKE_ATTEMPT_REGISTRY_NAME",
    "SMOKE_ATTEMPT_REGISTRY_SCHEMA",
    "SmokeAttemptRegistryClaim",
    "SmokeAttemptRegistryProtocol",
    "claim_smoke_attempt",
    "smoke_attempt_registry_key",
    "validate_smoke_attempt_registry_claim",
]
