"""Strict immutable evidence that Modal accepted one smoke call."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import (
    Field,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.gguf.inkling_smoke_execution import (
    SMOKE_DEPLOYMENT_TAG_HASH_PREFIX_LENGTH,
    SMOKE_ENVIRONMENT_NAME,
    SMOKE_STAGE,
    smoke_deployment_tag,
    strict_json_object,
    validate_smoke_attempt_registry_created_at_utc,
)

SMOKE_POST_SPAWN_ACCEPTANCE_SCHEMA = "inkling-smoke-post-spawn-acceptance-v3"
SMOKE_POST_SPAWN_ACCEPTANCE_STATUS = "accepted_after_spawn"
SMOKE_POST_SPAWN_ACCEPTANCE_MAX_BYTES = 64 * 1024

_RUN_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]{0,95}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


class SmokePostSpawnAcceptance(StrictFrozenModel):
    """One immutable record written only after a remote call is accepted."""

    schema_version: Literal["inkling-smoke-post-spawn-acceptance-v3"] = (
        "inkling-smoke-post-spawn-acceptance-v3"
    )
    status: Literal["accepted_after_spawn"] = "accepted_after_spawn"
    accepted_at: StrictStr
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    stage: Literal["smoke_test"] = "smoke_test"
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    call_id: StrictStr = Field(pattern=r"^fc-[A-Za-z0-9]+$")
    app_name: StrictStr = Field(pattern=r"^inkling-q3-smoke-[0-9a-f]{12}$")
    environment_name: Literal["inkling-quant"] = "inkling-quant"
    deployment_version: StrictInt = Field(gt=0)
    deployment_tag: StrictStr = Field(
        pattern=(
            rf"^iql-smoke-[0-9a-f]"
            rf"{{{SMOKE_DEPLOYMENT_TAG_HASH_PREFIX_LENGTH}}}$"
        )
    )
    function_id: StrictStr = Field(pattern=r"^fu-[A-Za-z0-9]+$")
    function_name: Literal["smoke_test"] = "smoke_test"
    attempt_registry_name: Literal["inkling-smoke-attempt-registry-v1"]
    attempt_registry_id: StrictStr = Field(pattern=r"^di-[A-Za-z0-9]+$")
    attempt_registry_created_at_utc: StrictStr
    smoke_config_hash: StrictStr = Field(pattern=_SHA256_PATTERN)
    verified_export_reference_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("accepted_at")
    @classmethod
    def accepted_at_is_canonical_utc(cls, value: str) -> str:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", value) is None:
            raise ValueError("acceptance time must use canonical UTC microsecond text")
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
        except ValueError as error:
            raise ValueError("acceptance time is not a real UTC time") from error
        if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:
            raise ValueError("acceptance time must use canonical UTC microsecond text")
        return value

    @field_validator("attempt_registry_created_at_utc")
    @classmethod
    def registry_creation_is_supported(cls, value: str) -> str:
        return validate_smoke_attempt_registry_created_at_utc(value)

    @model_validator(mode="after")
    def deployment_identity_matches_control_plane(self) -> SmokePostSpawnAcceptance:
        if self.app_name != f"inkling-q3-smoke-{self.control_plane_sha256[:12]}":
            raise ValueError("acceptance app name differs from its control-plane identity")
        if self.deployment_tag != smoke_deployment_tag(self.control_plane_sha256):
            raise ValueError("acceptance deployment tag differs from its control-plane identity")
        return self

    def canonical_bytes(self) -> bytes:
        """Return the only accepted byte encoding for this record."""

        return _canonical_json_bytes(self.model_dump(mode="json"))

    def acceptance_sha256(self) -> str:
        """Return the SHA-256 of the exact canonical record bytes."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def smoke_post_spawn_acceptance_path(run_id: str, launch_intent_sha256: str) -> str:
    """Return the only evidence path for one run and launch intent."""

    if re.fullmatch(_RUN_ID_PATTERN, run_id) is None:
        raise ValueError("smoke run ID is invalid")
    if re.fullmatch(_SHA256_PATTERN, launch_intent_sha256) is None:
        raise ValueError("smoke launch-intent SHA-256 is invalid")
    return PurePosixPath(
        "runs",
        run_id,
        "control",
        "post-spawn-acceptances",
        f"{launch_intent_sha256}.json",
    ).as_posix()


def validate_smoke_post_spawn_acceptance(
    payload: bytes,
    *,
    evidence_path: str,
    acceptance_sha256: str,
    accepted_at: str,
    run_id: str,
    launch_intent_sha256: str,
    call_id: str,
    app_name: str,
    environment_name: str,
    deployment_version: int,
    deployment_tag: str,
    function_id: str,
    function_name: str,
    attempt_registry_name: str,
    attempt_registry_id: str,
    attempt_registry_created_at_utc: str,
    smoke_config_hash: str,
    verified_export_reference_sha256: str,
    control_plane_sha256: str,
) -> SmokePostSpawnAcceptance:
    """Validate canonical bytes against the exact accepted remote invocation."""

    if not isinstance(payload, bytes):
        raise TypeError("smoke post-spawn acceptance must be bytes")
    if len(payload) > SMOKE_POST_SPAWN_ACCEPTANCE_MAX_BYTES:
        raise ValueError("smoke post-spawn acceptance exceeds its size limit")

    raw = strict_json_object(payload)
    try:
        acceptance = SmokePostSpawnAcceptance.model_validate(raw)
    except ValidationError as error:
        raise ValueError("smoke post-spawn acceptance schema is invalid") from error
    canonical_bytes = acceptance.canonical_bytes()
    if payload != canonical_bytes:
        raise ValueError("smoke post-spawn acceptance bytes are not canonical")

    if re.fullmatch(_SHA256_PATTERN, acceptance_sha256) is None:
        raise ValueError("expected acceptance SHA-256 is invalid")
    if hashlib.sha256(canonical_bytes).hexdigest() != acceptance_sha256:
        raise ValueError("smoke post-spawn acceptance hash does not match its bytes")

    try:
        expected = SmokePostSpawnAcceptance(
            accepted_at=accepted_at,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
            call_id=call_id,
            app_name=app_name,
            environment_name=environment_name,
            deployment_version=deployment_version,
            deployment_tag=deployment_tag,
            function_id=function_id,
            function_name=function_name,
            attempt_registry_name=attempt_registry_name,
            attempt_registry_id=attempt_registry_id,
            attempt_registry_created_at_utc=attempt_registry_created_at_utc,
            smoke_config_hash=smoke_config_hash,
            verified_export_reference_sha256=verified_export_reference_sha256,
            control_plane_sha256=control_plane_sha256,
        )
    except ValidationError as error:
        raise ValueError("expected post-spawn acceptance bindings are invalid") from error

    expected_path = smoke_post_spawn_acceptance_path(run_id, launch_intent_sha256)
    if not isinstance(evidence_path, str) or evidence_path != expected_path:
        raise ValueError("smoke post-spawn acceptance path is not the fixed evidence path")
    if expected.stage != SMOKE_STAGE or expected.environment_name != SMOKE_ENVIRONMENT_NAME:
        raise ValueError("smoke post-spawn acceptance constants are inconsistent")
    if acceptance != expected:
        raise ValueError("smoke post-spawn acceptance does not bind the exact accepted call")
    return acceptance


__all__ = [
    "SMOKE_POST_SPAWN_ACCEPTANCE_MAX_BYTES",
    "SMOKE_POST_SPAWN_ACCEPTANCE_SCHEMA",
    "SMOKE_POST_SPAWN_ACCEPTANCE_STATUS",
    "SmokePostSpawnAcceptance",
    "smoke_post_spawn_acceptance_path",
    "validate_smoke_post_spawn_acceptance",
]
