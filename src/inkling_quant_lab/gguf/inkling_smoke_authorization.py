"""Immutable remote launch authorization for the Inkling smoke stage."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field, StrictInt, StrictStr, ValidationError, field_validator, model_validator

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.gguf.inkling_smoke import (
    InklingSmokeConfig,
    InklingVerifiedExportReference,
)
from inkling_quant_lab.gguf.inkling_smoke_execution import (
    SMOKE_DEPLOYMENT_TAG_HASH_PREFIX_LENGTH,
    SMOKE_ENVIRONMENT_NAME,
    SMOKE_STAGE,
    SMOKE_WORKSPACE_BUDGET_USD,
    SmokeControlPlaneProvenance,
    SmokeLaunchDeploymentIdentity,
    smoke_deployment_tag,
    strict_json_object,
    validate_smoke_attempt_registry_created_at_utc,
)

SMOKE_LAUNCH_INTENT_SCHEMA = "inkling-smoke-modal-launch-intent-v4"
SMOKE_LAUNCH_AUTHORIZATION_SCOPE = "one_server_executing_attempt"
_MAX_LAUNCH_INTENT_BYTES = 64 * 1024


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


class SmokeLaunchIntent(StrictFrozenModel):
    """One sealed authorization that a remote smoke invocation must present."""

    schema_version: Literal["inkling-smoke-modal-launch-intent-v4"] = (
        "inkling-smoke-modal-launch-intent-v4"
    )
    status: Literal["prepared_before_spawn"] = "prepared_before_spawn"
    authorization_scope: Literal["one_server_executing_attempt"] = "one_server_executing_attempt"
    authorization_nonce: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    created_at_utc: StrictStr
    run_id: StrictStr = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    stage: Literal["smoke_test"] = "smoke_test"
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
    smoke_config_hash: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    verified_export_reference_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    control_plane_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    workspace_budget_usd: StrictStr
    billing_cycle_end_utc: StrictStr

    @field_validator("created_at_utc")
    @classmethod
    def created_at_is_canonical_utc(cls, value: str) -> str:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", value) is None:
            raise ValueError("launch intent creation time must use canonical UTC text")
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
        except ValueError as error:
            raise ValueError("launch intent creation time is not a real UTC time") from error
        if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:
            raise ValueError("launch intent creation time must use canonical UTC text")
        return value

    @field_validator("billing_cycle_end_utc")
    @classmethod
    def cycle_end_is_canonical_utc(cls, value: str) -> str:
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        except ValueError as error:
            raise ValueError(
                "billing cycle end must use a real YYYY-MM-DDTHH:MM:SSZ time"
            ) from error
        if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
            raise ValueError("billing cycle end must use canonical UTC text")
        return value

    @field_validator("attempt_registry_created_at_utc")
    @classmethod
    def registry_creation_is_supported(cls, value: str) -> str:
        return validate_smoke_attempt_registry_created_at_utc(value)

    @model_validator(mode="after")
    def exact_external_gate_and_derived_identity(self) -> SmokeLaunchIntent:
        try:
            workspace_budget_usd = Decimal(self.workspace_budget_usd)
        except InvalidOperation as error:
            raise ValueError("launch intent workspace cap is not a decimal") from error
        if workspace_budget_usd != SMOKE_WORKSPACE_BUDGET_USD or self.workspace_budget_usd != str(
            SMOKE_WORKSPACE_BUDGET_USD
        ):
            raise ValueError("launch intent has the wrong workspace cap")
        if self.app_name != f"inkling-q3-smoke-{self.control_plane_sha256[:12]}":
            raise ValueError("launch intent app name differs from its control-plane identity")
        if self.deployment_tag != smoke_deployment_tag(self.control_plane_sha256):
            raise ValueError("launch intent deployment tag differs from its control-plane identity")
        return self

    def canonical_bytes(self) -> bytes:
        """Return the exact byte representation stored on the evidence volume."""

        return _canonical_json_bytes(self.model_dump(mode="json"))

    def intent_sha256(self) -> str:
        """Return the SHA-256 of the exact stored launch-intent bytes."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def smoke_launch_intent_remote_path(run_id: str, launch_intent_sha256: str) -> str:
    """Return the only allowed remote path for one launch authorization."""

    if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,95}", run_id) is None:
        raise ValueError("smoke run ID is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", launch_intent_sha256) is None:
        raise ValueError("smoke launch-intent SHA-256 is invalid")
    return PurePosixPath(
        "runs",
        run_id,
        "control",
        "launch-authorizations",
        f"{launch_intent_sha256}.json",
    ).as_posix()


def validate_smoke_launch_intent(
    value: Mapping[str, Any] | str | bytes,
    *,
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    run_id: str,
    launch_intent_sha256: str,
    deployment: SmokeLaunchDeploymentIdentity | Mapping[str, Any],
) -> SmokeLaunchIntent:
    """Validate one immutable authorization against the exact sealed smoke run."""

    if isinstance(value, bytes):
        if len(value) > _MAX_LAUNCH_INTENT_BYTES:
            raise ValueError("smoke launch intent exceeds its size limit")
        raw = strict_json_object(value)
        supplied_bytes = value
    elif isinstance(value, str):
        supplied_bytes = value.encode("utf-8")
        if len(supplied_bytes) > _MAX_LAUNCH_INTENT_BYTES:
            raise ValueError("smoke launch intent exceeds its size limit")
        raw = strict_json_object(value)
    elif isinstance(value, Mapping):
        raw = dict(value)
        supplied_bytes = None
    else:
        raise TypeError("smoke launch intent must be JSON text, bytes, or a mapping")

    try:
        intent = SmokeLaunchIntent.model_validate(raw)
    except ValidationError as error:
        raise ValueError("smoke launch intent schema is invalid") from error
    canonical_bytes = intent.canonical_bytes()
    if supplied_bytes is not None and supplied_bytes != canonical_bytes:
        raise ValueError("smoke launch intent bytes are not canonical")
    if hashlib.sha256(canonical_bytes).hexdigest() != launch_intent_sha256:
        raise ValueError("smoke launch intent hash does not match its canonical bytes")
    if config.verified_export_reference_sha256 != reference.reference_sha256:
        raise ValueError("smoke launch-intent inputs use different export references")
    try:
        deployment_identity = SmokeLaunchDeploymentIdentity.model_validate(deployment)
    except ValidationError as error:
        raise ValueError("expected smoke deployment identity is invalid") from error
    observed = (
        intent.run_id,
        intent.stage,
        intent.environment_name,
        intent.smoke_config_hash,
        intent.verified_export_reference_sha256,
        intent.control_plane_sha256,
        intent.app_name,
        intent.deployment_version,
        intent.deployment_tag,
        intent.function_id,
        intent.function_name,
        intent.attempt_registry_name,
        intent.attempt_registry_id,
        intent.attempt_registry_created_at_utc,
    )
    expected = (
        run_id,
        SMOKE_STAGE,
        SMOKE_ENVIRONMENT_NAME,
        config.config_hash(),
        reference.reference_sha256,
        control_plane.tree_sha256,
        deployment_identity.app_name,
        deployment_identity.deployment_version,
        deployment_identity.deployment_tag,
        deployment_identity.function_id,
        deployment_identity.function_name,
        deployment_identity.attempt_registry_name,
        deployment_identity.attempt_registry_id,
        deployment_identity.attempt_registry_created_at_utc,
    )
    if observed != expected:
        raise ValueError("smoke launch intent does not bind the exact sealed run")
    return intent


__all__ = [
    "SMOKE_LAUNCH_AUTHORIZATION_SCOPE",
    "SMOKE_LAUNCH_INTENT_SCHEMA",
    "SmokeLaunchIntent",
    "smoke_launch_intent_remote_path",
    "validate_smoke_launch_intent",
]
