"""Pure control-plane contracts for the matched Inkling smoke run.

The models in this module authorize and record one BF16-then-Q3 smoke attempt.
They do not import Modal, inspect a remote service, or start compute.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Final, Literal, Protocol, TypeAlias

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from inkling_quant_lab.config import StrictFrozenModel

MATCHED_STAGE: Final = "matched_smoke"
MATCHED_ENVIRONMENT_NAME: Final = "inkling-quant"
MATCHED_FUNCTION_NAME: Final = "matched_smoke_test"
MATCHED_ATTEMPT_REGISTRY_NAME: Final = "inkling-matched-attempt-registry-v1"
MATCHED_EVIDENCE_VOLUME_NAME: Final = "inkling-matched-evidence-v1"

MATCHED_CONTROL_PLANE_HASH_DOMAIN: Final = b"inkling-matched-control-plane-v1\0"
MATCHED_DEPLOY_CHALLENGE_HASH_DOMAIN: Final = b"inkling-matched-deploy-challenge-v1\0"
MATCHED_LAUNCH_CHALLENGE_HASH_DOMAIN: Final = b"inkling-matched-launch-challenge-v1\0"
MATCHED_LAUNCH_INTENT_HASH_DOMAIN: Final = b"inkling-matched-launch-intent-v1\0"
MATCHED_POST_SPAWN_ACCEPTANCE_HASH_DOMAIN: Final = b"inkling-matched-post-spawn-acceptance-v1\0"
MATCHED_ATTEMPT_CLAIM_HASH_DOMAIN: Final = b"inkling-matched-attempt-claim-v1\0"
MATCHED_ATTEMPT_ACKNOWLEDGEMENT_HASH_DOMAIN: Final = b"inkling-matched-attempt-acknowledgement-v1\0"
MATCHED_SUCCESS_RECEIPT_HASH_DOMAIN: Final = b"inkling-matched-terminal-success-receipt-v1\0"
MATCHED_FAILURE_RECEIPT_HASH_DOMAIN: Final = b"inkling-matched-terminal-failure-receipt-v1\0"
MATCHED_PUBLICATION_STATE_HASH_DOMAIN: Final = b"inkling-matched-publication-state-v1\0"

MATCHED_CONTROL_RECORD_MAX_BYTES: Final = 512 * 1024
MATCHED_DEPLOY_CONFIRMATION_PREFIX: Final = "CONFIRM MATCHED DEPLOY"
MATCHED_LAUNCH_CONFIRMATION_PREFIX: Final = "CONFIRM MATCHED LAUNCH"

_RUN_ID_PATTERN: Final = r"^[a-z0-9][a-z0-9._-]{0,95}$"
_SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
_GIT_OBJECT_PATTERN: Final = r"^[0-9a-f]{40}$"
_CANONICAL_UTC_MICROSECOND_PATTERN: Final = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$"
_MINIMUM_REGISTRY_CREATED_AT: Final = datetime(2025, 5, 20, tzinfo=UTC)

MatchedSubject: TypeAlias = Literal["bf16", "q3"]
MatchedOutcome: TypeAlias = Literal["success", "failure"]


class _StrictControlModel(StrictFrozenModel):
    """Fail-closed base for immutable matched control records."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _canonical_json_bytes(value: object) -> bytes:
    """Return canonical UTF-8 JSON with exactly one final line feed."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_terminal_json_bytes(value: object) -> bytes:
    """Return the exact newline-free encoding used by terminal receipts."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _domain_hash(domain: bytes, value: bytes) -> str:
    return hashlib.sha256(domain + value).hexdigest()


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in matched control JSON: {key}")
        result[key] = value
    return result


def strict_matched_json_object(payload: bytes | str) -> dict[str, Any]:
    """Parse one bounded JSON object while rejecting duplicate keys."""

    if isinstance(payload, str):
        encoded = payload.encode("utf-8")
        text = payload
    elif isinstance(payload, bytes):
        encoded = payload
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("matched control JSON must be UTF-8") from error
    else:
        raise TypeError("matched control JSON must be bytes or text")
    if len(encoded) > MATCHED_CONTROL_RECORD_MAX_BYTES:
        raise ValueError("matched control JSON exceeds its size limit")
    try:
        value = json.loads(text, object_pairs_hook=_reject_duplicate_object_pairs)
    except json.JSONDecodeError as error:
        raise ValueError("matched control JSON is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("matched control JSON root must be an object")
    return value


def _canonical_utc_microseconds(value: str, *, label: str) -> str:
    if re.fullmatch(_CANONICAL_UTC_MICROSECOND_PATTERN, value) is None:
        raise ValueError(f"{label} must use canonical UTC microsecond text")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"{label} is not a real UTC time") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:
        raise ValueError(f"{label} must use canonical UTC microsecond text")
    return value


def _canonical_utc_seconds(value: str, *, label: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"{label} must use a real YYYY-MM-DDTHH:MM:SSZ time") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError(f"{label} must use canonical UTC text")
    return value


def _utc_microseconds_datetime(value: str, *, label: str) -> datetime:
    _canonical_utc_microseconds(value, label=label)
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def _utc_seconds_datetime(value: str, *, label: str) -> datetime:
    _canonical_utc_seconds(value, label=label)
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _validate_registry_created_at(value: str) -> str:
    _canonical_utc_microseconds(value, label="attempt registry creation time")
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    if parsed < _MINIMUM_REGISTRY_CREATED_AT:
        raise ValueError("attempt registry predates the supported Modal Dict boundary")
    return value


def _validate_repository_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
        or "\x00" in value
    ):
        raise ValueError("matched control file path must be canonical and repository-relative")
    return value


def _validate_run_id(run_id: str) -> str:
    if re.fullmatch(_RUN_ID_PATTERN, run_id) is None:
        raise ValueError("matched run ID is invalid")
    return run_id


def _validate_sha256(value: str, *, label: str) -> str:
    if re.fullmatch(_SHA256_PATTERN, value) is None:
        raise ValueError(f"{label} is invalid")
    return value


class MatchedControlPlaneFile(_StrictControlModel):
    """One exact deployed file in the matched implementation identity."""

    path: StrictStr
    size_bytes: StrictInt = Field(ge=0)
    sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return _validate_repository_relative_path(value)


def matched_control_plane_sha256(
    *,
    reviewed_commit_sha: str,
    reviewed_tree_sha: str,
    files: Sequence[MatchedControlPlaneFile],
) -> str:
    """Hash the reviewed Git identity and exact sorted deployed file manifest."""

    if re.fullmatch(_GIT_OBJECT_PATTERN, reviewed_commit_sha) is None:
        raise ValueError("reviewed Git commit SHA is invalid")
    if re.fullmatch(_GIT_OBJECT_PATTERN, reviewed_tree_sha) is None:
        raise ValueError("reviewed Git tree SHA is invalid")
    payload = {
        "schema_version": "inkling-matched-control-plane-v1",
        "reviewed_commit_sha": reviewed_commit_sha,
        "reviewed_tree_sha": reviewed_tree_sha,
        "files": [item.model_dump(mode="json") for item in files],
    }
    return _domain_hash(
        MATCHED_CONTROL_PLANE_HASH_DOMAIN,
        _canonical_json_bytes(payload),
    )


class MatchedControlPlaneProvenance(_StrictControlModel):
    """Content-addressed local and remote implementation manifest."""

    schema_version: Literal["inkling-matched-control-plane-v1"] = "inkling-matched-control-plane-v1"
    reviewed_commit_sha: StrictStr = Field(pattern=_GIT_OBJECT_PATTERN)
    reviewed_tree_sha: StrictStr = Field(pattern=_GIT_OBJECT_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    file_count: StrictInt = Field(gt=0)
    files: tuple[MatchedControlPlaneFile, ...]

    @model_validator(mode="after")
    def manifest_is_complete_and_self_hashed(self) -> MatchedControlPlaneProvenance:
        if self.file_count != len(self.files):
            raise ValueError("matched control-plane file count differs from its manifest")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("matched control-plane paths must be sorted and unique")
        expected = matched_control_plane_sha256(
            reviewed_commit_sha=self.reviewed_commit_sha,
            reviewed_tree_sha=self.reviewed_tree_sha,
            files=self.files,
        )
        if self.control_plane_sha256 != expected:
            raise ValueError("matched control-plane hash differs from its exact manifest")
        return self

    def canonical_bytes(self) -> bytes:
        """Return canonical bytes suitable for a mounted runtime comparison."""

        return _canonical_json_bytes(self.model_dump(mode="json"))


def build_matched_control_plane_provenance(
    *,
    reviewed_commit_sha: str,
    reviewed_tree_sha: str,
    files: Mapping[str, bytes],
    required_paths: Sequence[str],
) -> MatchedControlPlaneProvenance:
    """Build a provenance record from an explicitly closed deployed file set."""

    required = tuple(required_paths)
    if len(required) != len(set(required)):
        raise ValueError("matched control-plane required paths must be unique")
    for path in required:
        _validate_repository_relative_path(path)
    observed_paths = set(files)
    if observed_paths != set(required) or len(files) != len(required):
        raise ValueError("matched control-plane files must equal the exact required path set")
    manifest: list[MatchedControlPlaneFile] = []
    for path in sorted(required):
        payload = files[path]
        if not isinstance(payload, bytes):
            raise TypeError("matched control-plane file payloads must be bytes")
        manifest.append(
            MatchedControlPlaneFile(
                path=path,
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    tree_sha256 = matched_control_plane_sha256(
        reviewed_commit_sha=reviewed_commit_sha,
        reviewed_tree_sha=reviewed_tree_sha,
        files=manifest,
    )
    return MatchedControlPlaneProvenance(
        reviewed_commit_sha=reviewed_commit_sha,
        reviewed_tree_sha=reviewed_tree_sha,
        control_plane_sha256=tree_sha256,
        file_count=len(manifest),
        files=tuple(manifest),
    )


def validate_matched_control_plane_provenance(
    provenance: MatchedControlPlaneProvenance | Mapping[str, Any] | bytes,
    *,
    reviewed_commit_sha: str,
    reviewed_tree_sha: str,
    files: Mapping[str, bytes],
    required_paths: Sequence[str],
) -> MatchedControlPlaneProvenance:
    """Rebuild and compare provenance against exact local or mounted bytes."""

    if isinstance(provenance, bytes):
        raw = strict_matched_json_object(provenance)
        try:
            observed = MatchedControlPlaneProvenance.model_validate(raw)
        except ValidationError as error:
            raise ValueError("matched control-plane provenance schema is invalid") from error
        if provenance != observed.canonical_bytes():
            raise ValueError("matched control-plane provenance bytes are not canonical")
    elif isinstance(provenance, Mapping):
        try:
            observed = MatchedControlPlaneProvenance.model_validate(provenance)
        except ValidationError as error:
            raise ValueError("matched control-plane provenance schema is invalid") from error
    elif isinstance(provenance, MatchedControlPlaneProvenance):
        observed = provenance
    else:
        raise TypeError("matched control-plane provenance has an unsupported type")
    expected = build_matched_control_plane_provenance(
        reviewed_commit_sha=reviewed_commit_sha,
        reviewed_tree_sha=reviewed_tree_sha,
        files=files,
        required_paths=required_paths,
    )
    if observed != expected:
        raise ValueError("matched control-plane provenance differs from deployed bytes")
    return observed


class MatchedExecutionResources(_StrictControlModel):
    """Exact eight-B300 resource cell authorized by the launch."""

    provider: Literal["modal"] = "modal"
    gpu_type: Literal["B300"] = "B300"
    gpu_count: Literal[8] = 8
    compute_capability: Literal["10.3"] = "10.3"
    cpu_cores: Literal[16] = 16
    memory_mib: Literal[65536] = 65_536
    ephemeral_disk_mib: Literal[524288] = 524_288
    startup_timeout_seconds: Literal[1800] = 1_800
    function_timeout_seconds: Literal[14400]
    max_containers: Literal[1] = 1
    max_recovery_attempts: Literal[0] = 0


class MatchedReviewedInputs(_StrictControlModel):
    """All reviewed immutable inputs common to deploy and launch."""

    schema_version: Literal["inkling-matched-reviewed-inputs-v1"] = (
        "inkling-matched-reviewed-inputs-v1"
    )
    reviewed_commit_sha: StrictStr = Field(pattern=_GIT_OBJECT_PATTERN)
    reviewed_tree_sha: StrictStr = Field(pattern=_GIT_OBJECT_PATTERN)
    matched_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    matched_plan_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    bf16_subject_reference_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    q3_verified_export_reference_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    source_adoption_reference_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    subject_order: tuple[MatchedSubject, ...]
    resources: MatchedExecutionResources

    @model_validator(mode="after")
    def exact_subject_order(self) -> MatchedReviewedInputs:
        if self.subject_order != ("bf16", "q3"):
            raise ValueError("matched launch must use the exact BF16 then Q3 subject order")
        return self

    def validate_control_plane(
        self,
        provenance: MatchedControlPlaneProvenance,
    ) -> MatchedReviewedInputs:
        """Require the review record and deployed provenance to agree exactly."""

        observed = (
            self.reviewed_commit_sha,
            self.reviewed_tree_sha,
            self.control_plane_sha256,
        )
        expected = (
            provenance.reviewed_commit_sha,
            provenance.reviewed_tree_sha,
            provenance.control_plane_sha256,
        )
        if observed != expected:
            raise ValueError("matched reviewed inputs differ from control-plane provenance")
        return self


def matched_app_name(control_plane_sha256: str) -> str:
    """Derive the single App name from the complete control identity."""

    _validate_sha256(control_plane_sha256, label="matched control-plane SHA-256")
    return f"inkling-matched-smoke-{control_plane_sha256[:12]}"


def matched_deployment_tag(control_plane_sha256: str) -> str:
    """Derive the implementation tag while retaining the full hash elsewhere."""

    _validate_sha256(control_plane_sha256, label="matched control-plane SHA-256")
    return f"iql-matched-{control_plane_sha256[:40]}"


class MatchedDeploymentIdentity(_StrictControlModel):
    """Exact deployed objects authorized for one matched run."""

    schema_version: Literal["inkling-matched-deployment-identity-v1"] = (
        "inkling-matched-deployment-identity-v1"
    )
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    app_name: StrictStr = Field(pattern=r"^inkling-matched-smoke-[0-9a-f]{12}$")
    environment_name: Literal["inkling-quant"] = "inkling-quant"
    deployment_version: StrictInt = Field(gt=0)
    deployment_tag: StrictStr = Field(pattern=r"^iql-matched-[0-9a-f]{40}$")
    function_id: StrictStr = Field(pattern=r"^fu-[A-Za-z0-9]+$")
    function_name: Literal["matched_smoke_test"] = "matched_smoke_test"
    attempt_registry_name: Literal["inkling-matched-attempt-registry-v1"] = (
        "inkling-matched-attempt-registry-v1"
    )
    attempt_registry_id: StrictStr = Field(pattern=r"^di-[A-Za-z0-9]+$")
    attempt_registry_created_at_utc: StrictStr
    evidence_volume_name: Literal["inkling-matched-evidence-v1"] = "inkling-matched-evidence-v1"
    evidence_volume_id: StrictStr = Field(pattern=r"^vo-[A-Za-z0-9]+$")

    @field_validator("attempt_registry_created_at_utc")
    @classmethod
    def registry_creation_is_supported(cls, value: str) -> str:
        return _validate_registry_created_at(value)

    @model_validator(mode="after")
    def derived_names_are_exact(self) -> MatchedDeploymentIdentity:
        if self.app_name != matched_app_name(self.control_plane_sha256):
            raise ValueError("matched deployment app name differs from its control identity")
        if self.deployment_tag != matched_deployment_tag(self.control_plane_sha256):
            raise ValueError("matched deployment tag differs from its control identity")
        return self

    def canonical_bytes(self) -> bytes:
        """Return the exact stored deployment-identity bytes."""

        return _canonical_json_bytes(self.model_dump(mode="json"))


class MatchedDeployConfirmationChallenge(_StrictControlModel):
    """Content-addressed operator challenge shown before deployment."""

    schema_version: Literal["inkling-matched-deploy-confirmation-v1"] = (
        "inkling-matched-deploy-confirmation-v1"
    )
    status: Literal["prepared_before_deploy"] = "prepared_before_deploy"
    created_at_utc: StrictStr
    confirmation_nonce: StrictStr = Field(pattern=_SHA256_PATTERN)
    reviewed_inputs: MatchedReviewedInputs
    app_name: StrictStr = Field(pattern=r"^inkling-matched-smoke-[0-9a-f]{12}$")
    environment_name: Literal["inkling-quant"] = "inkling-quant"
    function_name: Literal["matched_smoke_test"] = "matched_smoke_test"
    attempt_registry_name: Literal["inkling-matched-attempt-registry-v1"] = (
        "inkling-matched-attempt-registry-v1"
    )
    evidence_volume_name: Literal["inkling-matched-evidence-v1"] = "inkling-matched-evidence-v1"

    @field_validator("created_at_utc")
    @classmethod
    def created_at_is_canonical(cls, value: str) -> str:
        return _canonical_utc_microseconds(value, label="deploy challenge creation time")

    @model_validator(mode="after")
    def app_is_derived(self) -> MatchedDeployConfirmationChallenge:
        if self.app_name != matched_app_name(self.reviewed_inputs.control_plane_sha256):
            raise ValueError("matched deploy challenge has the wrong app name")
        return self

    def canonical_bytes(self) -> bytes:
        """Return canonical challenge bytes."""

        return _canonical_json_bytes(self.model_dump(mode="json"))

    def challenge_sha256(self) -> str:
        """Return the domain-separated deploy challenge digest."""

        return _domain_hash(MATCHED_DEPLOY_CHALLENGE_HASH_DOMAIN, self.canonical_bytes())

    def confirmation_text(self) -> str:
        """Return the only accepted operator confirmation."""

        return f"{MATCHED_DEPLOY_CONFIRMATION_PREFIX}\n{self.challenge_sha256()}"

    def confirm(self, value: str) -> MatchedDeployConfirmationChallenge:
        """Validate an exact deploy confirmation without changing external state."""

        if not isinstance(value, str) or value != self.confirmation_text():
            raise ValueError("matched deploy confirmation does not match its challenge")
        return self


class MatchedLaunchConfirmationChallenge(_StrictControlModel):
    """Content-addressed operator challenge shown immediately before launch."""

    schema_version: Literal["inkling-matched-launch-confirmation-v1"] = (
        "inkling-matched-launch-confirmation-v1"
    )
    status: Literal["prepared_before_launch"] = "prepared_before_launch"
    created_at_utc: StrictStr
    authorization_nonce: StrictStr = Field(pattern=_SHA256_PATTERN)
    billing_cycle_end_utc: StrictStr
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    reviewed_inputs: MatchedReviewedInputs
    deployment: MatchedDeploymentIdentity

    @field_validator("created_at_utc")
    @classmethod
    def created_at_is_canonical(cls, value: str) -> str:
        return _canonical_utc_microseconds(value, label="launch challenge creation time")

    @field_validator("billing_cycle_end_utc")
    @classmethod
    def cycle_is_canonical(cls, value: str) -> str:
        return _canonical_utc_seconds(value, label="billing cycle end")

    @model_validator(mode="after")
    def identities_are_exact(self) -> MatchedLaunchConfirmationChallenge:
        if self.deployment.control_plane_sha256 != self.reviewed_inputs.control_plane_sha256:
            raise ValueError("matched launch deployment differs from reviewed control")
        created_at = _utc_microseconds_datetime(
            self.created_at_utc,
            label="launch challenge creation time",
        )
        cycle_end = _utc_seconds_datetime(
            self.billing_cycle_end_utc,
            label="billing cycle end",
        )
        registry_created_at = _utc_microseconds_datetime(
            self.deployment.attempt_registry_created_at_utc,
            label="attempt registry creation time",
        )
        if registry_created_at > created_at:
            raise ValueError("matched launch challenge predates its attempt registry")
        if created_at >= cycle_end:
            raise ValueError("matched launch challenge is outside its billing cycle")
        return self

    def canonical_bytes(self) -> bytes:
        """Return canonical launch challenge bytes."""

        return _canonical_json_bytes(self.model_dump(mode="json"))

    def challenge_sha256(self) -> str:
        """Return the domain-separated launch challenge digest."""

        return _domain_hash(MATCHED_LAUNCH_CHALLENGE_HASH_DOMAIN, self.canonical_bytes())

    def confirmation_text(self) -> str:
        """Return the only accepted launch confirmation."""

        return f"{MATCHED_LAUNCH_CONFIRMATION_PREFIX}\n{self.challenge_sha256()}"

    def confirm(self, value: str) -> MatchedLaunchConfirmationChallenge:
        """Validate an exact launch confirmation without starting compute."""

        if not isinstance(value, str) or value != self.confirmation_text():
            raise ValueError("matched launch confirmation does not match its challenge")
        return self


class MatchedLaunchIntent(_StrictControlModel):
    """One immutable authorization presented to the remote runner."""

    schema_version: Literal["inkling-matched-launch-intent-v1"] = "inkling-matched-launch-intent-v1"
    status: Literal["authorized_before_spawn"] = "authorized_before_spawn"
    authorization_scope: Literal["one_matched_smoke_attempt"] = "one_matched_smoke_attempt"
    authorized_at_utc: StrictStr
    launch_challenge_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    authorization_nonce: StrictStr = Field(pattern=_SHA256_PATTERN)
    billing_cycle_end_utc: StrictStr
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    reviewed_inputs: MatchedReviewedInputs
    subject_order: tuple[MatchedSubject, ...]
    resources: MatchedExecutionResources
    deployment: MatchedDeploymentIdentity
    one_atomic_attempt: Literal[True] = True
    fresh_process_per_subject: Literal[True] = True
    rehash_all_subject_files: Literal[True] = True
    measurement_execution_allowed: Literal[False] = False

    @field_validator("authorized_at_utc")
    @classmethod
    def authorized_at_is_canonical(cls, value: str) -> str:
        return _canonical_utc_microseconds(value, label="launch authorization time")

    @field_validator("billing_cycle_end_utc")
    @classmethod
    def cycle_is_canonical(cls, value: str) -> str:
        return _canonical_utc_seconds(value, label="billing cycle end")

    @model_validator(mode="after")
    def redundant_safety_bindings_are_exact(self) -> MatchedLaunchIntent:
        if self.subject_order != self.reviewed_inputs.subject_order:
            raise ValueError("matched launch subject order differs from reviewed inputs")
        if self.resources != self.reviewed_inputs.resources:
            raise ValueError("matched launch resources differ from reviewed inputs")
        if self.deployment.control_plane_sha256 != self.reviewed_inputs.control_plane_sha256:
            raise ValueError("matched launch deployment differs from reviewed control")
        authorized_at = _utc_microseconds_datetime(
            self.authorized_at_utc,
            label="launch authorization time",
        )
        cycle_end = _utc_seconds_datetime(
            self.billing_cycle_end_utc,
            label="billing cycle end",
        )
        if authorized_at >= cycle_end:
            raise ValueError("matched launch authorization time is outside its billing cycle")
        return self

    def canonical_bytes(self) -> bytes:
        """Return the only accepted stored launch-intent bytes."""

        return _canonical_json_bytes(self.model_dump(mode="json"))

    def intent_sha256(self) -> str:
        """Return the domain-separated launch-intent digest."""

        return _domain_hash(MATCHED_LAUNCH_INTENT_HASH_DOMAIN, self.canonical_bytes())


def build_matched_launch_intent(
    challenge: MatchedLaunchConfirmationChallenge,
    *,
    confirmation: str,
    authorized_at_utc: str,
) -> MatchedLaunchIntent:
    """Convert an exactly confirmed challenge into one remote authorization."""

    challenge.confirm(confirmation)
    authorized_at = _utc_microseconds_datetime(
        authorized_at_utc,
        label="launch authorization time",
    )
    challenge_created_at = _utc_microseconds_datetime(
        challenge.created_at_utc,
        label="launch challenge creation time",
    )
    cycle_end = _utc_seconds_datetime(
        challenge.billing_cycle_end_utc,
        label="billing cycle end",
    )
    if not challenge_created_at <= authorized_at < cycle_end:
        raise ValueError(
            "matched launch authorization time must follow its challenge "
            "and precede the billing-cycle end"
        )
    return MatchedLaunchIntent(
        authorized_at_utc=authorized_at_utc,
        launch_challenge_sha256=challenge.challenge_sha256(),
        authorization_nonce=challenge.authorization_nonce,
        billing_cycle_end_utc=challenge.billing_cycle_end_utc,
        run_id=challenge.run_id,
        reviewed_inputs=challenge.reviewed_inputs,
        subject_order=challenge.reviewed_inputs.subject_order,
        resources=challenge.reviewed_inputs.resources,
        deployment=challenge.deployment,
    )


def matched_launch_intent_path(run_id: str, launch_intent_sha256: str) -> str:
    """Return the content-addressed launch authorization path."""

    _validate_run_id(run_id)
    _validate_sha256(launch_intent_sha256, label="matched launch-intent SHA-256")
    return PurePosixPath(
        "runs",
        run_id,
        "control",
        "launch-intents",
        f"{launch_intent_sha256}.json",
    ).as_posix()


def validate_matched_launch_intent(
    payload: bytes,
    *,
    expected: MatchedLaunchIntent,
    launch_intent_sha256: str,
    evidence_path: str,
) -> MatchedLaunchIntent:
    """Validate canonical intent bytes against the complete expected launch."""

    if not isinstance(payload, bytes):
        raise TypeError("matched launch intent must be bytes")
    raw = strict_matched_json_object(payload)
    try:
        observed = MatchedLaunchIntent.model_validate(raw)
    except ValidationError as error:
        raise ValueError("matched launch intent schema is invalid") from error
    if payload != observed.canonical_bytes():
        raise ValueError("matched launch intent bytes are not canonical")
    if observed.intent_sha256() != launch_intent_sha256:
        raise ValueError("matched launch-intent hash differs from its canonical bytes")
    if evidence_path != matched_launch_intent_path(observed.run_id, launch_intent_sha256):
        raise ValueError("matched launch-intent path is not content addressed")
    if observed != expected:
        raise ValueError("matched launch intent differs from the exact expected launch")
    return observed


class MatchedPostSpawnAcceptance(_StrictControlModel):
    """Immutable evidence that the provider accepted one remote call."""

    schema_version: Literal["inkling-matched-post-spawn-acceptance-v1"] = (
        "inkling-matched-post-spawn-acceptance-v1"
    )
    status: Literal["accepted_after_spawn"] = "accepted_after_spawn"
    accepted_at_utc: StrictStr
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    call_id: StrictStr = Field(pattern=r"^fc-[A-Za-z0-9]+$")
    deployment: MatchedDeploymentIdentity
    matched_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("accepted_at_utc")
    @classmethod
    def accepted_at_is_canonical(cls, value: str) -> str:
        return _canonical_utc_microseconds(value, label="post-spawn acceptance time")

    @model_validator(mode="after")
    def deployment_is_exact(self) -> MatchedPostSpawnAcceptance:
        if self.deployment.control_plane_sha256 != self.control_plane_sha256:
            raise ValueError("matched acceptance deployment differs from its control")
        return self

    def canonical_bytes(self) -> bytes:
        """Return exact durable acceptance bytes."""

        return _canonical_json_bytes(self.model_dump(mode="json"))

    def acceptance_sha256(self) -> str:
        """Return the domain-separated acceptance digest."""

        return _domain_hash(
            MATCHED_POST_SPAWN_ACCEPTANCE_HASH_DOMAIN,
            self.canonical_bytes(),
        )


def matched_post_spawn_acceptance_path(run_id: str, launch_intent_sha256: str) -> str:
    """Return the fixed path for one launch's post-spawn acceptance."""

    _validate_run_id(run_id)
    _validate_sha256(launch_intent_sha256, label="matched launch-intent SHA-256")
    return PurePosixPath(
        "runs",
        run_id,
        "control",
        "post-spawn-acceptances",
        f"{launch_intent_sha256}.json",
    ).as_posix()


def validate_matched_post_spawn_acceptance(
    payload: bytes,
    *,
    expected: MatchedPostSpawnAcceptance,
    acceptance_sha256: str,
    evidence_path: str,
) -> MatchedPostSpawnAcceptance:
    """Validate exact canonical post-spawn acceptance evidence."""

    if not isinstance(payload, bytes):
        raise TypeError("matched post-spawn acceptance must be bytes")
    raw = strict_matched_json_object(payload)
    try:
        observed = MatchedPostSpawnAcceptance.model_validate(raw)
    except ValidationError as error:
        raise ValueError("matched post-spawn acceptance schema is invalid") from error
    if payload != observed.canonical_bytes():
        raise ValueError("matched post-spawn acceptance bytes are not canonical")
    if observed.acceptance_sha256() != acceptance_sha256:
        raise ValueError("matched post-spawn acceptance hash differs from its canonical bytes")
    expected_path = matched_post_spawn_acceptance_path(
        observed.run_id,
        observed.launch_intent_sha256,
    )
    if evidence_path != expected_path:
        raise ValueError("matched post-spawn acceptance path is not launch addressed")
    if observed != expected:
        raise ValueError("matched post-spawn acceptance differs from the exact expected acceptance")
    return observed


def matched_attempt_registry_key(
    run_id: str,
    stage: str = MATCHED_STAGE,
) -> str:
    """Return the only atomic Dict key for a matched run."""

    _validate_run_id(run_id)
    if stage != MATCHED_STAGE:
        raise ValueError("matched attempt stage is invalid")
    return f"{run_id}:{stage}"


class MatchedAttemptClaim(_StrictControlModel):
    """One immutable claim of the matched run's only remote attempt."""

    schema_version: Literal["inkling-matched-attempt-claim-v1"] = "inkling-matched-attempt-claim-v1"
    registry_name: Literal["inkling-matched-attempt-registry-v1"] = (
        "inkling-matched-attempt-registry-v1"
    )
    registry_id: StrictStr = Field(pattern=r"^di-[A-Za-z0-9]+$")
    registry_created_at_utc: StrictStr
    registry_key: StrictStr
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    stage: Literal["matched_smoke"] = "matched_smoke"
    call_id: StrictStr = Field(pattern=r"^fc-[A-Za-z0-9]+$")
    input_id: StrictStr = Field(pattern=r"^in-[A-Za-z0-9]+(?::[0-9]+-[0-9]+)?$")
    task_id: StrictStr = Field(pattern=r"^ta-[A-Za-z0-9]+$")
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    post_spawn_acceptance_path: StrictStr
    post_spawn_acceptance_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    matched_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("registry_created_at_utc")
    @classmethod
    def registry_creation_is_supported(cls, value: str) -> str:
        return _validate_registry_created_at(value)

    @model_validator(mode="after")
    def derived_paths_are_exact(self) -> MatchedAttemptClaim:
        if self.registry_key != matched_attempt_registry_key(self.run_id, self.stage):
            raise ValueError("matched attempt registry key differs from run and stage")
        expected_acceptance = matched_post_spawn_acceptance_path(
            self.run_id,
            self.launch_intent_sha256,
        )
        if self.post_spawn_acceptance_path != expected_acceptance:
            raise ValueError("matched attempt acceptance path differs from its launch")
        return self

    def canonical_bytes(self) -> bytes:
        """Return the exact bytes stored in the atomic Dict and Volume."""

        return _canonical_json_bytes(self.model_dump(mode="json"))

    def claim_sha256(self) -> str:
        """Return the domain-separated attempt-claim digest."""

        return _domain_hash(MATCHED_ATTEMPT_CLAIM_HASH_DOMAIN, self.canonical_bytes())


class MatchedAttemptRegistryProtocol(Protocol):
    """Minimal atomic operation required from the remote attempt registry."""

    def put(
        self,
        key: Any,
        value: Any,
        *,
        skip_if_exists: bool = False,
    ) -> bool: ...


def claim_matched_attempt(
    registry: MatchedAttemptRegistryProtocol,
    claim: MatchedAttemptClaim,
) -> str:
    """Atomically consume the one matched attempt."""

    created = registry.put(
        claim.registry_key,
        claim.canonical_bytes(),
        skip_if_exists=True,
    )
    if created is not True:
        raise RuntimeError("The one configured matched smoke attempt has already been consumed")
    return claim.claim_sha256()


def matched_attempt_claim_path(run_id: str, claim_sha256: str) -> str:
    """Return the durable content-addressed claim path."""

    _validate_run_id(run_id)
    _validate_sha256(claim_sha256, label="matched attempt-claim SHA-256")
    return PurePosixPath(
        "runs",
        run_id,
        "control",
        "attempt-claims",
        f"{claim_sha256}.json",
    ).as_posix()


def validate_matched_attempt_claim(
    payload: bytes,
    *,
    expected: MatchedAttemptClaim,
    claim_sha256: str,
    evidence_path: str,
) -> MatchedAttemptClaim:
    """Validate the exact canonical claim stored in the Dict and Volume."""

    if not isinstance(payload, bytes):
        raise TypeError("matched attempt claim must be bytes")
    raw = strict_matched_json_object(payload)
    try:
        observed = MatchedAttemptClaim.model_validate(raw)
    except ValidationError as error:
        raise ValueError("matched attempt claim schema is invalid") from error
    if payload != observed.canonical_bytes():
        raise ValueError("matched attempt claim bytes are not canonical")
    if observed.claim_sha256() != claim_sha256:
        raise ValueError("matched attempt claim hash differs from its canonical bytes")
    if evidence_path != matched_attempt_claim_path(observed.run_id, claim_sha256):
        raise ValueError("matched attempt claim path is not content addressed")
    if observed != expected:
        raise ValueError("matched attempt claim differs from the exact expected claim")
    return observed


class MatchedAttemptAcknowledgement(_StrictControlModel):
    """Durable acknowledgement written only after the atomic claim wins."""

    schema_version: Literal["inkling-matched-attempt-acknowledgement-v1"] = (
        "inkling-matched-attempt-acknowledgement-v1"
    )
    status: Literal["attempt_claim_acknowledged"] = "attempt_claim_acknowledged"
    acknowledged_at_utc: StrictStr
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    registry_key: StrictStr
    attempt_claim_path: StrictStr
    attempt_claim_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    call_id: StrictStr = Field(pattern=r"^fc-[A-Za-z0-9]+$")
    input_id: StrictStr = Field(pattern=r"^in-[A-Za-z0-9]+(?::[0-9]+-[0-9]+)?$")
    task_id: StrictStr = Field(pattern=r"^ta-[A-Za-z0-9]+$")
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    matched_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("acknowledged_at_utc")
    @classmethod
    def acknowledged_at_is_canonical(cls, value: str) -> str:
        return _canonical_utc_microseconds(value, label="attempt acknowledgement time")

    @model_validator(mode="after")
    def claim_bindings_are_exact(self) -> MatchedAttemptAcknowledgement:
        if self.registry_key != matched_attempt_registry_key(self.run_id):
            raise ValueError("matched acknowledgement has the wrong registry key")
        if self.attempt_claim_path != matched_attempt_claim_path(
            self.run_id,
            self.attempt_claim_sha256,
        ):
            raise ValueError("matched acknowledgement has the wrong claim path")
        return self

    def canonical_bytes(self) -> bytes:
        """Return exact durable acknowledgement bytes."""

        return _canonical_json_bytes(self.model_dump(mode="json"))

    def acknowledgement_sha256(self) -> str:
        """Return the domain-separated acknowledgement digest."""

        return _domain_hash(
            MATCHED_ATTEMPT_ACKNOWLEDGEMENT_HASH_DOMAIN,
            self.canonical_bytes(),
        )


def matched_attempt_acknowledgement_path(
    run_id: str,
    acknowledgement_sha256: str,
) -> str:
    """Return the content-addressed durable acknowledgement path."""

    _validate_run_id(run_id)
    _validate_sha256(
        acknowledgement_sha256,
        label="matched attempt-acknowledgement SHA-256",
    )
    return PurePosixPath(
        "runs",
        run_id,
        "control",
        "attempt-acknowledgements",
        f"{acknowledgement_sha256}.json",
    ).as_posix()


def validate_matched_attempt_acknowledgement(
    payload: bytes,
    *,
    expected: MatchedAttemptAcknowledgement,
    acknowledgement_sha256: str,
    evidence_path: str,
) -> MatchedAttemptAcknowledgement:
    """Validate exact canonical acknowledgement evidence."""

    if not isinstance(payload, bytes):
        raise TypeError("matched attempt acknowledgement must be bytes")
    raw = strict_matched_json_object(payload)
    try:
        observed = MatchedAttemptAcknowledgement.model_validate(raw)
    except ValidationError as error:
        raise ValueError("matched attempt acknowledgement schema is invalid") from error
    if payload != observed.canonical_bytes():
        raise ValueError("matched attempt acknowledgement bytes are not canonical")
    if observed.acknowledgement_sha256() != acknowledgement_sha256:
        raise ValueError("matched attempt acknowledgement hash differs from its canonical bytes")
    expected_path = matched_attempt_acknowledgement_path(
        observed.run_id,
        acknowledgement_sha256,
    )
    if evidence_path != expected_path:
        raise ValueError("matched attempt acknowledgement path is not content addressed")
    if observed != expected:
        raise ValueError(
            "matched attempt acknowledgement differs from the exact expected acknowledgement"
        )
    return observed


def matched_terminal_receipt_content_sha256(
    payload: bytes,
    *,
    run_id: str,
    outcome: MatchedOutcome,
) -> str:
    """Validate and content-hash one exact terminal receipt."""

    if not isinstance(payload, bytes):
        raise TypeError("matched terminal receipt must be bytes")
    _validate_run_id(run_id)
    raw = strict_matched_json_object(payload)
    if payload != _canonical_terminal_json_bytes(raw):
        raise ValueError("matched terminal receipt bytes are not canonical")
    expected_envelope = {
        "success": ("inkling-matched-rollup-v1", "passed"),
        "failure": ("inkling-matched-failure-v1", "failed"),
    }[outcome]
    if (
        raw.get("schema_version"),
        raw.get("status"),
    ) != expected_envelope:
        raise ValueError("matched terminal receipt does not match the requested outcome")
    if raw.get("stage") != MATCHED_STAGE:
        raise ValueError("matched terminal receipt has the wrong stage")
    observed_run_id = raw.get("run_id")
    if not isinstance(observed_run_id, str):
        raise ValueError("matched terminal receipt has no valid run ID")
    _validate_run_id(observed_run_id)
    if observed_run_id != run_id:
        raise ValueError("matched terminal receipt run ID differs from its expected run")
    if raw.get("prompt_text_recorded") is not False:
        raise ValueError("matched terminal receipt must not record raw prompts")
    if raw.get("output_text_recorded") is not False:
        raise ValueError("matched terminal receipt must not record raw outputs")
    receipt_sha256 = raw.get("receipt_sha256")
    if not isinstance(receipt_sha256, str):
        raise ValueError("matched terminal receipt has no valid self-hash")
    _validate_sha256(
        receipt_sha256,
        label="matched terminal receipt embedded SHA-256",
    )
    domain = (
        MATCHED_SUCCESS_RECEIPT_HASH_DOMAIN
        if outcome == "success"
        else MATCHED_FAILURE_RECEIPT_HASH_DOMAIN
    )
    return _domain_hash(domain, payload)


def matched_terminal_receipt_path(
    run_id: str,
    *,
    outcome: MatchedOutcome,
    content_sha256: str,
) -> str:
    """Return the only content-addressed terminal receipt path."""

    _validate_run_id(run_id)
    _validate_sha256(content_sha256, label="matched terminal receipt content SHA-256")
    return PurePosixPath(
        "runs",
        run_id,
        "terminal",
        outcome,
        f"{content_sha256}.json",
    ).as_posix()


class MatchedTerminalReceiptReference(_StrictControlModel):
    """Safe identity of one immutable terminal receipt."""

    schema_version: Literal["inkling-matched-terminal-reference-v1"] = (
        "inkling-matched-terminal-reference-v1"
    )
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    outcome: MatchedOutcome
    path: StrictStr
    embedded_receipt_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    content_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    size_bytes: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def path_is_content_addressed(self) -> MatchedTerminalReceiptReference:
        expected = matched_terminal_receipt_path(
            self.run_id,
            outcome=self.outcome,
            content_sha256=self.content_sha256,
        )
        if self.path != expected:
            raise ValueError("matched terminal receipt reference path is not content addressed")
        return self


def build_matched_terminal_receipt_reference(
    payload: bytes,
    *,
    run_id: str,
    outcome: MatchedOutcome,
) -> MatchedTerminalReceiptReference:
    """Build a safe reference from already validated canonical terminal bytes."""

    content_sha256 = matched_terminal_receipt_content_sha256(
        payload,
        run_id=run_id,
        outcome=outcome,
    )
    raw = strict_matched_json_object(payload)
    embedded_receipt_sha256 = raw["receipt_sha256"]
    if not isinstance(embedded_receipt_sha256, str):
        raise ValueError("matched terminal receipt has no valid self-hash")
    return MatchedTerminalReceiptReference(
        run_id=run_id,
        outcome=outcome,
        path=matched_terminal_receipt_path(
            run_id,
            outcome=outcome,
            content_sha256=content_sha256,
        ),
        embedded_receipt_sha256=embedded_receipt_sha256,
        content_sha256=content_sha256,
        size_bytes=len(payload),
    )


def validate_matched_terminal_receipt_reference(
    payload: bytes,
    *,
    expected: MatchedTerminalReceiptReference,
) -> MatchedTerminalReceiptReference:
    """Rebuild and compare one terminal reference against exact receipt bytes."""

    observed = build_matched_terminal_receipt_reference(
        payload,
        run_id=expected.run_id,
        outcome=expected.outcome,
    )
    if observed != expected:
        raise ValueError("matched terminal receipt reference differs from the exact receipt bytes")
    return observed


class MatchedPublicationSnapshot(_StrictControlModel):
    """Immutable snapshot of runner-side terminal publication."""

    schema_version: Literal["inkling-matched-publication-state-v1"] = (
        "inkling-matched-publication-state-v1"
    )
    publication_id: StrictStr = Field(pattern=_SHA256_PATTERN)
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    attempt_claim_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    status: Literal["not_started", "installing", "confirmed", "unknown"]
    cycle: StrictInt = Field(ge=0, le=2)
    terminal_receipt: MatchedTerminalReceiptReference | None = None
    mounted_reload_completed: StrictBool = False
    runner_network_blocked: Literal[True] = True
    runner_volume_read_method: Literal["mounted_after_reload"] = "mounted_after_reload"
    manager_cross_container_read_required: Literal[True] = True

    @model_validator(mode="after")
    def state_is_coherent(self) -> MatchedPublicationSnapshot:
        if self.status == "not_started":
            if self.cycle != 0 or self.terminal_receipt is not None:
                raise ValueError("not-started publication state cannot bind terminal work")
            if self.mounted_reload_completed:
                raise ValueError("not-started publication state cannot report a reload")
        elif self.status == "installing":
            if self.cycle not in {1, 2} or self.terminal_receipt is None:
                raise ValueError("installing publication state needs a bounded terminal cycle")
            if self.mounted_reload_completed:
                raise ValueError("installing publication state cannot claim completed reload proof")
        elif self.status == "confirmed":
            if self.cycle not in {1, 2} or self.terminal_receipt is None:
                raise ValueError("confirmed publication state needs a terminal receipt")
            if not self.mounted_reload_completed:
                raise ValueError("confirmed publication state needs mounted reload proof")
        elif self.status == "unknown":
            if self.cycle not in {1, 2}:
                raise ValueError("unknown publication state needs a started publication cycle")
        return self

    @property
    def failure_receipt_publication_allowed(self) -> bool:
        """State whether a new failure receipt can be installed safely."""

        return self.status == "not_started"

    def canonical_bytes(self) -> bytes:
        """Return exact immutable state bytes."""

        return _canonical_json_bytes(self.model_dump(mode="json"))

    def state_sha256(self) -> str:
        """Return the domain-separated publication-state digest."""

        return _domain_hash(MATCHED_PUBLICATION_STATE_HASH_DOMAIN, self.canonical_bytes())


def matched_publication_state_path(run_id: str, state_sha256: str) -> str:
    """Return the content-addressed immutable publication-state path."""

    _validate_run_id(run_id)
    _validate_sha256(state_sha256, label="matched publication-state SHA-256")
    return PurePosixPath(
        "runs",
        run_id,
        "control",
        "publication-states",
        f"{state_sha256}.json",
    ).as_posix()


def validate_matched_publication_state(
    payload: bytes,
    *,
    expected: MatchedPublicationSnapshot,
    state_sha256: str,
    evidence_path: str,
) -> MatchedPublicationSnapshot:
    """Validate one exact immutable publication-state snapshot."""

    if not isinstance(payload, bytes):
        raise TypeError("matched publication state must be bytes")
    raw = strict_matched_json_object(payload)
    try:
        observed = MatchedPublicationSnapshot.model_validate(raw)
    except ValidationError as error:
        raise ValueError("matched publication state schema is invalid") from error
    if payload != observed.canonical_bytes():
        raise ValueError("matched publication state bytes are not canonical")
    if observed.state_sha256() != state_sha256:
        raise ValueError("matched publication-state hash differs from its canonical bytes")
    if evidence_path != matched_publication_state_path(observed.run_id, state_sha256):
        raise ValueError("matched publication-state path is not content addressed")
    if observed != expected:
        raise ValueError("matched publication state differs from the exact expected state")
    return observed


def validate_matched_publication_transition(
    previous: MatchedPublicationSnapshot,
    current: MatchedPublicationSnapshot,
) -> MatchedPublicationSnapshot:
    """Validate one monotonic bounded publication transition."""

    if (
        previous.publication_id,
        previous.run_id,
        previous.attempt_claim_sha256,
    ) != (
        current.publication_id,
        current.run_id,
        current.attempt_claim_sha256,
    ):
        raise ValueError("matched publication transition changes immutable identity")
    allowed: dict[str, set[str]] = {
        "not_started": {"installing"},
        "installing": {"installing", "confirmed", "unknown"},
        "confirmed": set(),
        "unknown": set(),
    }
    if current.status not in allowed[previous.status]:
        raise ValueError("matched publication transition is not monotonic")
    if current.cycle < previous.cycle or current.cycle > previous.cycle + 1:
        raise ValueError("matched publication transition has an invalid cycle")
    if (
        previous.status == "installing"
        and current.status == "installing"
        and (previous.cycle != 1 or current.cycle != 2)
    ):
        raise ValueError("matched publication retry must be the one bounded second cycle")
    if previous.terminal_receipt is not None and (
        current.terminal_receipt != previous.terminal_receipt
    ):
        raise ValueError("matched publication transition changes the terminal receipt")
    return current


__all__ = [
    "MATCHED_ATTEMPT_ACKNOWLEDGEMENT_HASH_DOMAIN",
    "MATCHED_ATTEMPT_CLAIM_HASH_DOMAIN",
    "MATCHED_ATTEMPT_REGISTRY_NAME",
    "MATCHED_CONTROL_PLANE_HASH_DOMAIN",
    "MATCHED_CONTROL_RECORD_MAX_BYTES",
    "MATCHED_DEPLOY_CHALLENGE_HASH_DOMAIN",
    "MATCHED_DEPLOY_CONFIRMATION_PREFIX",
    "MATCHED_ENVIRONMENT_NAME",
    "MATCHED_EVIDENCE_VOLUME_NAME",
    "MATCHED_FAILURE_RECEIPT_HASH_DOMAIN",
    "MATCHED_FUNCTION_NAME",
    "MATCHED_LAUNCH_CHALLENGE_HASH_DOMAIN",
    "MATCHED_LAUNCH_CONFIRMATION_PREFIX",
    "MATCHED_LAUNCH_INTENT_HASH_DOMAIN",
    "MATCHED_POST_SPAWN_ACCEPTANCE_HASH_DOMAIN",
    "MATCHED_PUBLICATION_STATE_HASH_DOMAIN",
    "MATCHED_STAGE",
    "MATCHED_SUCCESS_RECEIPT_HASH_DOMAIN",
    "MatchedAttemptAcknowledgement",
    "MatchedAttemptClaim",
    "MatchedAttemptRegistryProtocol",
    "MatchedControlPlaneFile",
    "MatchedControlPlaneProvenance",
    "MatchedDeployConfirmationChallenge",
    "MatchedDeploymentIdentity",
    "MatchedExecutionResources",
    "MatchedLaunchConfirmationChallenge",
    "MatchedLaunchIntent",
    "MatchedPostSpawnAcceptance",
    "MatchedPublicationSnapshot",
    "MatchedReviewedInputs",
    "MatchedTerminalReceiptReference",
    "build_matched_control_plane_provenance",
    "build_matched_launch_intent",
    "build_matched_terminal_receipt_reference",
    "claim_matched_attempt",
    "matched_app_name",
    "matched_attempt_acknowledgement_path",
    "matched_attempt_claim_path",
    "matched_attempt_registry_key",
    "matched_control_plane_sha256",
    "matched_deployment_tag",
    "matched_launch_intent_path",
    "matched_post_spawn_acceptance_path",
    "matched_publication_state_path",
    "matched_terminal_receipt_content_sha256",
    "matched_terminal_receipt_path",
    "strict_matched_json_object",
    "validate_matched_attempt_acknowledgement",
    "validate_matched_attempt_claim",
    "validate_matched_control_plane_provenance",
    "validate_matched_launch_intent",
    "validate_matched_post_spawn_acceptance",
    "validate_matched_publication_state",
    "validate_matched_publication_transition",
    "validate_matched_terminal_receipt_reference",
]
