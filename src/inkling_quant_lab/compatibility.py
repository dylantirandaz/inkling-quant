"""Exact, fail-closed compatibility matrix records and read-only operations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from inkling_quant_lab.exceptions import ArtifactIntegrityError

ClaimStatus: TypeAlias = Literal["supported", "unsupported", "not_measured"]
EvidenceKind: TypeAlias = Literal[
    "structural_verification",
    "inference_smoke",
    "quality_evaluation",
    "performance_benchmark",
    "mtp_validation",
]
ArtifactKind: TypeAlias = Literal[
    "export_manifest",
    "model_shard",
    "multimodal_projector",
    "mtp_export",
]
CompatibilityScopeWarning: TypeAlias = Literal[
    "Read each machine-readable experiment record before using its result. Do not apply "
    "a result to a different model, dataset, runtime, software, hardware, or protocol."
]

COMPATIBILITY_SCOPE_WARNING: CompatibilityScopeWarning = (
    "Read each machine-readable experiment record before using its result. "
    "Do not apply a result to a different model, dataset, runtime, software, "
    "hardware, or protocol."
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_COMMIT_PATTERN = r"^[0-9a-f]{40}$"
_PINNED_IMAGE_PATTERN = r"^\S+@sha256:[0-9a-f]{64}$"
_FORBIDDEN_SCOPE_VALUES = frozenset(
    {
        "*",
        "any",
        "all",
        "example",
        "n/a",
        "na",
        "placeholder",
        "tbd",
        "todo",
        "unknown",
        "unspecified",
    }
)


class _ImmutableCompatibilityModel(BaseModel):
    """Strict immutable base for compatibility records."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _require_trimmed_text(value: str, *, label: str) -> str:
    if not value or value.strip() != value:
        raise ValueError(f"{label} must be non-empty and trimmed")
    return value


def _require_exact_text(value: str, *, label: str) -> str:
    _require_trimmed_text(value, label=label)
    if value.casefold() in _FORBIDDEN_SCOPE_VALUES or "*" in value or "?" in value:
        raise ValueError(f"{label} must identify one exact value; wildcards are forbidden")
    return value


class CompatibilityArtifact(_ImmutableCompatibilityModel):
    """One checksum-pinned artifact or artifact-set manifest."""

    artifact_id: str = Field(min_length=1)
    kind: ArtifactKind
    location: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("artifact_id", "location")
    @classmethod
    def identity_is_exact(cls, value: str) -> str:
        return _require_exact_text(value, label="artifact identity")


class CompatibilityEvidenceReference(_ImmutableCompatibilityModel):
    """Checksum-pinned evidence supporting one or more claims in a cell."""

    reference_id: str = Field(min_length=1)
    kind: EvidenceKind
    location: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("reference_id", "location")
    @classmethod
    def identity_is_exact(cls, value: str) -> str:
        return _require_exact_text(value, label="evidence reference identity")


class ProtocolIdentity(_ImmutableCompatibilityModel):
    """Exact workload or validation protocol used for one cell."""

    protocol_id: str = Field(min_length=1)
    protocol_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("protocol_id")
    @classmethod
    def protocol_id_is_exact(cls, value: str) -> str:
        return _require_exact_text(value, label="protocol ID")


class RuntimeBinaryIdentity(_ImmutableCompatibilityModel):
    """One executable byte identity in the deployment runtime."""

    name: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("name")
    @classmethod
    def name_is_exact(cls, value: str) -> str:
        return _require_exact_text(value, label="runtime binary name")


class GpuDeviceIdentity(_ImmutableCompatibilityModel):
    """One exact physical GPU, including allocation-relevant capacity."""

    uuid: str = Field(min_length=1)
    model: str = Field(min_length=1)
    memory_bytes: int = Field(gt=0)
    compute_capability: str = Field(pattern=r"^[0-9]+\.[0-9]+$")

    @field_validator("uuid", "model")
    @classmethod
    def identity_is_exact(cls, value: str) -> str:
        return _require_exact_text(value, label="GPU identity")


class CompatibilityClaim(_ImmutableCompatibilityModel):
    """One bounded claim whose positive or negative result names its evidence."""

    status: ClaimStatus
    evidence_refs: tuple[str, ...] = ()
    reason: str = Field(min_length=1)

    @field_validator("evidence_refs")
    @classmethod
    def references_are_unique_and_exact(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("claim evidence references must be unique")
        for reference in value:
            _require_exact_text(reference, label="claim evidence reference")
        if value != tuple(sorted(value)):
            raise ValueError("claim evidence references must be canonically sorted")
        return value

    @field_validator("reason")
    @classmethod
    def reason_is_stable(cls, value: str) -> str:
        return _require_trimmed_text(value, label="claim reason")

    @model_validator(mode="after")
    def evidence_matches_status(self) -> CompatibilityClaim:
        if self.status == "not_measured" and self.evidence_refs:
            raise ValueError("not_measured claims cannot cite result evidence")
        if self.status == "supported" and not self.evidence_refs:
            raise ValueError("supported claims require evidence references")
        return self


class CompatibilityClaims(_ImmutableCompatibilityModel):
    """The five claims every exact compatibility cell must state."""

    structurally_verified: CompatibilityClaim
    smoke_passed: CompatibilityClaim
    quality_measured: CompatibilityClaim
    performance_measured: CompatibilityClaim
    mtp_supported: CompatibilityClaim


class ModelIdentity(_ImmutableCompatibilityModel):
    """Exact model and immutable revision used by one cell."""

    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=_COMMIT_PATTERN)

    @field_validator("model_id")
    @classmethod
    def model_id_is_exact(cls, value: str) -> str:
        return _require_exact_text(value, label="model ID")


class RuntimeIdentity(_ImmutableCompatibilityModel):
    """Exact inference or verification runtime revision."""

    name: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    commit: str = Field(pattern=_COMMIT_PATTERN)

    @field_validator("name", "repository")
    @classmethod
    def runtime_text_is_exact(cls, value: str) -> str:
        return _require_exact_text(value, label="runtime identity")


class CudaIdentity(_ImmutableCompatibilityModel):
    """Explicit CUDA presence and version for one exact cell."""

    status: Literal["present", "absent"]
    version: str | None

    @model_validator(mode="after")
    def status_matches_version(self) -> CudaIdentity:
        if self.status == "present":
            if self.version is None:
                raise ValueError("present CUDA identity requires an exact version")
            _require_exact_text(self.version, label="CUDA version")
        elif self.version is not None:
            raise ValueError("absent CUDA identity cannot declare a version")
        return self


class SoftwareIdentity(_ImmutableCompatibilityModel):
    """Pinned image and complete software-manifest identity for one cell."""

    container_image: str = Field(pattern=_PINNED_IMAGE_PATTERN)
    identity_kind: Literal["control_plane_tree", "runtime_receipt"]
    software_identity_sha256: str = Field(pattern=_SHA256_PATTERN)
    cuda: CudaIdentity
    driver_version: str | None = None
    runtime_binaries: tuple[RuntimeBinaryIdentity, ...] = ()
    build_flags: tuple[str, ...] | None = None
    package_manifest_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @field_validator("driver_version")
    @classmethod
    def driver_is_exact(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_exact_text(value, label="driver version")

    @field_validator("runtime_binaries")
    @classmethod
    def binaries_are_canonical(
        cls,
        value: tuple[RuntimeBinaryIdentity, ...],
    ) -> tuple[RuntimeBinaryIdentity, ...]:
        names = tuple(item.name for item in value)
        if len(names) != len(set(names)):
            raise ValueError("runtime binary names must be unique")
        if value != tuple(sorted(value, key=lambda item: item.name)):
            raise ValueError("runtime binaries must be canonically sorted")
        return value

    @field_validator("build_flags")
    @classmethod
    def flags_are_canonical(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is None:
            return None
        if any(not flag or flag.strip() != flag for flag in value):
            raise ValueError("build flags must be non-empty and trimmed")
        if value != tuple(sorted(set(value))):
            raise ValueError("build flags must be sorted and unique")
        return value

    @model_validator(mode="after")
    def driver_matches_cuda(self) -> SoftwareIdentity:
        if self.cuda.status == "absent" and self.driver_version is not None:
            raise ValueError("CUDA-absent software cannot declare a GPU driver version")
        return self


class HardwareIdentity(_ImmutableCompatibilityModel):
    """Exact provider and GPU identity; CPU-only cells use none/zero."""

    provider: str = Field(min_length=1)
    gpu_model: str = Field(min_length=1)
    gpu_count: int = Field(ge=0, le=64)
    gpus: tuple[GpuDeviceIdentity, ...] = ()
    cpu_model: str | None = None
    logical_cpu_count: int | None = Field(default=None, gt=0)
    ram_bytes: int | None = Field(default=None, gt=0)
    topology_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @field_validator("provider", "gpu_model")
    @classmethod
    def hardware_text_is_exact(cls, value: str) -> str:
        return _require_exact_text(value, label="hardware identity")

    @field_validator("cpu_model")
    @classmethod
    def cpu_model_is_exact(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _require_exact_text(value, label="CPU model")

    @field_validator("gpus")
    @classmethod
    def gpus_are_canonical(
        cls,
        value: tuple[GpuDeviceIdentity, ...],
    ) -> tuple[GpuDeviceIdentity, ...]:
        uuids = tuple(item.uuid for item in value)
        if len(uuids) != len(set(uuids)):
            raise ValueError("GPU UUIDs must be unique")
        if value != tuple(sorted(value, key=lambda item: item.uuid)):
            raise ValueError("GPU devices must be canonically sorted by UUID")
        return value

    @model_validator(mode="after")
    def gpu_fields_are_consistent(self) -> HardwareIdentity:
        if self.gpu_count == 0 and self.gpu_model != "none":
            raise ValueError("zero-GPU cells must use gpu_model='none'")
        if self.gpu_count > 0 and self.gpu_model == "none":
            raise ValueError("GPU cells require an exact GPU model")
        if self.gpus and len(self.gpus) != self.gpu_count:
            raise ValueError("GPU device identities must equal gpu_count")
        if self.gpus and any(item.model != self.gpu_model for item in self.gpus):
            raise ValueError("GPU device models must equal gpu_model")
        return self


class CompatibilityCell(_ImmutableCompatibilityModel):
    """One non-generalizable model/runtime/software/hardware matrix cell."""

    cell_id: str = Field(min_length=1)
    execution_role: Literal["structural_verification", "deployment"]
    model: ModelIdentity
    artifacts: tuple[CompatibilityArtifact, ...]
    runtime: RuntimeIdentity
    protocol: ProtocolIdentity | None = None
    software: SoftwareIdentity
    hardware: HardwareIdentity
    evidence: tuple[CompatibilityEvidenceReference, ...]
    claims: CompatibilityClaims

    @field_validator("cell_id")
    @classmethod
    def cell_id_is_exact(cls, value: str) -> str:
        return _require_exact_text(value, label="cell ID")

    @field_validator("artifacts")
    @classmethod
    def artifacts_are_nonempty_unique_and_sorted(
        cls,
        value: tuple[CompatibilityArtifact, ...],
    ) -> tuple[CompatibilityArtifact, ...]:
        if not value:
            raise ValueError("compatibility cells require checksum-pinned artifacts")
        identifiers = tuple(item.artifact_id for item in value)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("compatibility artifact IDs must be unique")
        if value != tuple(sorted(value, key=lambda item: item.artifact_id)):
            raise ValueError("compatibility artifacts must be canonically sorted")
        if not any(item.kind == "export_manifest" for item in value):
            raise ValueError("compatibility cells require an export_manifest artifact")
        return value

    @field_validator("evidence")
    @classmethod
    def evidence_is_unique_and_sorted(
        cls,
        value: tuple[CompatibilityEvidenceReference, ...],
    ) -> tuple[CompatibilityEvidenceReference, ...]:
        identifiers = tuple(item.reference_id for item in value)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("compatibility evidence reference IDs must be unique")
        if value != tuple(sorted(value, key=lambda item: item.reference_id)):
            raise ValueError("compatibility evidence references must be canonically sorted")
        return value

    @model_validator(mode="after")
    def claims_are_consistent_and_bound(self) -> CompatibilityCell:
        evidence_by_id = {item.reference_id: item for item in self.evidence}
        claim_requirements: tuple[tuple[str, CompatibilityClaim, EvidenceKind], ...] = (
            (
                "structurally_verified",
                self.claims.structurally_verified,
                "structural_verification",
            ),
            ("smoke_passed", self.claims.smoke_passed, "inference_smoke"),
            ("quality_measured", self.claims.quality_measured, "quality_evaluation"),
            (
                "performance_measured",
                self.claims.performance_measured,
                "performance_benchmark",
            ),
            ("mtp_supported", self.claims.mtp_supported, "mtp_validation"),
        )
        for claim_name, claim, expected_kind in claim_requirements:
            missing = [item for item in claim.evidence_refs if item not in evidence_by_id]
            if missing:
                raise ValueError(
                    f"{claim_name} cites unknown evidence references: {', '.join(missing)}"
                )
            permitted_kinds = {expected_kind}
            if claim_name == "mtp_supported" and claim.status == "unsupported":
                permitted_kinds.add("structural_verification")
            if claim.evidence_refs and not any(
                evidence_by_id[item].kind in permitted_kinds for item in claim.evidence_refs
            ):
                raise ValueError(f"{claim_name} evidence must include the relevant result kind")

        structural = self.claims.structurally_verified.status == "supported"
        smoke = self.claims.smoke_passed.status == "supported"
        if smoke and not structural:
            raise ValueError("smoke_passed cannot be supported without structural verification")
        if self.claims.quality_measured.status == "supported" and not smoke:
            raise ValueError("quality_measured cannot be supported before smoke_passed")
        if self.claims.performance_measured.status == "supported" and not smoke:
            raise ValueError("performance_measured cannot be supported before smoke_passed")
        if self.claims.mtp_supported.status == "supported":
            if not smoke:
                raise ValueError("mtp_supported cannot be supported before smoke_passed")
            if not any(item.kind == "mtp_export" for item in self.artifacts):
                raise ValueError("mtp_supported requires a checksum-pinned mtp_export artifact")
        if self.execution_role == "structural_verification":
            if self.hardware.gpu_count != 0 or self.software.cuda.status != "absent":
                raise ValueError(
                    "structural_verification cells must record the exact CPU-only environment"
                )
            for name, claim, _ in claim_requirements[1:]:
                if claim.status == "supported":
                    raise ValueError(f"structural_verification cells cannot claim supported {name}")
        return self

    def exact_scope_document(self) -> dict[str, Any]:
        """Return every governed identity dimension for evidence binding."""

        return {
            "execution_role": self.execution_role,
            "model": self.model.model_dump(mode="json"),
            "artifacts": [item.model_dump(mode="json") for item in self.artifacts],
            "runtime": self.runtime.model_dump(mode="json"),
            "protocol": self.protocol.model_dump(mode="json")
            if self.protocol is not None
            else None,
            "software": self.software.model_dump(mode="json"),
            "hardware": self.hardware.model_dump(mode="json"),
        }

    def exact_scope_sha256(self) -> str:
        """Hash the complete exact-cell scope for terminal evidence records."""

        return hashlib.sha256(
            _canonical_json(self.exact_scope_document()).encode("utf-8")
        ).hexdigest()

    def exact_scope_key(self) -> str:
        """Return the canonical complete identity used to detect duplicate cells."""

        return _canonical_json(self.exact_scope_document())


class CompatibilityMatrixPayload(_ImmutableCompatibilityModel):
    """Exact compatibility cells; no row or column implies another cell."""

    matrix_id: str = Field(min_length=1)
    scope_policy: Literal["exact_cell_only"] = "exact_cell_only"
    scope_warning: CompatibilityScopeWarning = COMPATIBILITY_SCOPE_WARNING
    cells: tuple[CompatibilityCell, ...]

    @field_validator("matrix_id")
    @classmethod
    def matrix_id_is_exact(cls, value: str) -> str:
        return _require_exact_text(value, label="matrix ID")

    @field_validator("cells")
    @classmethod
    def cells_are_nonempty_unique_and_sorted(
        cls,
        value: tuple[CompatibilityCell, ...],
    ) -> tuple[CompatibilityCell, ...]:
        if not value:
            raise ValueError("compatibility matrices require at least one exact cell")
        identifiers = tuple(item.cell_id for item in value)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("compatibility cell IDs must be unique")
        if value != tuple(sorted(value, key=lambda item: item.cell_id)):
            raise ValueError("compatibility cells must be canonically sorted")
        scope_keys = tuple(item.exact_scope_key() for item in value)
        if len(scope_keys) != len(set(scope_keys)):
            raise ValueError("duplicate exact compatibility cell scope")
        return value


class CompatibilityMatrixRecord(_ImmutableCompatibilityModel):
    """Versioned matrix envelope whose payload digest detects mutation."""

    schema_version: Literal["iql-compatibility-matrix-v1"] = "iql-compatibility-matrix-v1"
    payload_sha256: str = Field(pattern=_SHA256_PATTERN)
    payload: CompatibilityMatrixPayload

    @model_validator(mode="after")
    def payload_digest_matches(self) -> CompatibilityMatrixRecord:
        expected = compatibility_payload_sha256(self.payload)
        if self.payload_sha256 != expected:
            raise ValueError(
                "compatibility payload SHA-256 mismatch: "
                f"expected {expected}, recorded {self.payload_sha256}"
            )
        return self


@dataclass(frozen=True)
class LoadedCompatibilityMatrix:
    """Validated matrix plus the exact file identity used by a command."""

    path: Path
    file_sha256: str
    record: CompatibilityMatrixRecord


class _DuplicateKeyError(ValueError):
    """A JSON object contained an ambiguous repeated key."""


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKeyError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def compatibility_payload_sha256(payload: CompatibilityMatrixPayload) -> str:
    """Hash the canonical matrix payload, excluding the envelope digest."""

    document = payload.model_dump(mode="json")
    # Schema-v1 records predate the full-scope fields. Keep their envelope hashes
    # readable while semantic validation still refuses to promote incomplete cells.
    for cell in document["cells"]:
        if cell["protocol"] is None:
            del cell["protocol"]
        software = cell["software"]
        for field in ("driver_version", "build_flags", "package_manifest_sha256"):
            if software[field] is None:
                del software[field]
        if not software["runtime_binaries"]:
            del software["runtime_binaries"]
        hardware = cell["hardware"]
        for field in ("cpu_model", "logical_cpu_count", "ram_bytes", "topology_sha256"):
            if hardware[field] is None:
                del hardware[field]
        if not hardware["gpus"]:
            del hardware["gpus"]
    encoded = _canonical_json(document).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_compatibility_matrix(
    payload: CompatibilityMatrixPayload,
) -> CompatibilityMatrixRecord:
    """Create a self-checking compatibility matrix envelope."""

    return CompatibilityMatrixRecord(
        payload_sha256=compatibility_payload_sha256(payload),
        payload=payload,
    )


def _validation_details(error: ValidationError) -> list[dict[str, str]]:
    return [
        {
            "field": ".".join(str(item) for item in issue["loc"]),
            "message": issue["msg"],
            "type": issue["type"],
        }
        for issue in error.errors(include_url=False)
    ]


def load_compatibility_matrix(path: str | Path) -> LoadedCompatibilityMatrix:
    """Load one exact supported schema and verify its canonical payload digest."""

    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise ArtifactIntegrityError(
            f"Compatibility matrix cannot be a symbolic link: {supplied}",
            component="compatibility",
            remediation="Pass the immutable compatibility JSON file directly.",
            details={"path": str(supplied)},
        )
    try:
        resolved = supplied.resolve(strict=True)
        if not resolved.is_file():
            raise ArtifactIntegrityError(
                f"Compatibility matrix is not a regular file: {resolved}",
                component="compatibility",
                details={"path": str(resolved)},
            )
        raw = resolved.read_bytes()
        text = raw.decode("utf-8")
    except ArtifactIntegrityError:
        raise
    except (OSError, UnicodeDecodeError) as error:
        raise ArtifactIntegrityError(
            f"Unable to read compatibility matrix {supplied}: {error}",
            component="compatibility",
            remediation="Verify the path, permissions, and UTF-8 encoding.",
            details={"path": str(supplied)},
        ) from error
    try:
        document = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, _DuplicateKeyError) as error:
        raise ArtifactIntegrityError(
            f"Unable to parse compatibility matrix {resolved}: {error}",
            component="compatibility",
            remediation="Provide one unambiguous iql-compatibility-matrix-v1 object.",
            details={"path": str(resolved)},
        ) from error
    try:
        record = CompatibilityMatrixRecord.model_validate(document)
    except ValidationError as error:
        details = _validation_details(error)
        raise ArtifactIntegrityError(
            "Compatibility matrix schema or payload integrity validation failed: "
            + "; ".join(f"{item['field']}: {item['message']}" for item in details),
            component="compatibility",
            remediation="Use an exact, checksum-valid iql-compatibility-matrix-v1 record.",
            details={"path": str(resolved), "errors": details},
        ) from error
    return LoadedCompatibilityMatrix(
        path=resolved,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        record=record,
    )


def _claim_payload(claim: CompatibilityClaim) -> dict[str, Any]:
    return claim.model_dump(mode="json")


def inspect_compatibility_matrix(matrix: LoadedCompatibilityMatrix) -> dict[str, Any]:
    """Return every exact cell without collapsing or generalizing support."""

    payload = matrix.record.payload
    cells = []
    for cell in payload.cells:
        cells.append(
            {
                "cell_id": cell.cell_id,
                "execution_role": cell.execution_role,
                "exact_scope": {
                    "model": cell.model.model_dump(mode="json"),
                    "artifacts": [item.model_dump(mode="json") for item in cell.artifacts],
                    "runtime": cell.runtime.model_dump(mode="json"),
                    "protocol": cell.protocol.model_dump(mode="json")
                    if cell.protocol is not None
                    else None,
                    "software": cell.software.model_dump(mode="json"),
                    "hardware": cell.hardware.model_dump(mode="json"),
                    "scope_sha256": cell.exact_scope_sha256(),
                },
                "claims": {
                    "structurally_verified": _claim_payload(cell.claims.structurally_verified),
                    "smoke_passed": _claim_payload(cell.claims.smoke_passed),
                    "quality_measured": _claim_payload(cell.claims.quality_measured),
                    "performance_measured": _claim_payload(cell.claims.performance_measured),
                    "mtp_supported": _claim_payload(cell.claims.mtp_supported),
                },
                "evidence": [item.model_dump(mode="json") for item in cell.evidence],
            }
        )
    return {
        "schema_version": "iql-compatibility-inspection-v1",
        "status": "inspected",
        "matrix_path": str(matrix.path),
        "matrix_file_sha256": matrix.file_sha256,
        "payload_sha256": matrix.record.payload_sha256,
        "matrix_id": payload.matrix_id,
        "scope_policy": payload.scope_policy,
        "cell_count": len(cells),
        "cells": cells,
        "warnings": [payload.scope_warning],
    }


def _repository_root(matrix_path: Path) -> Path:
    """Find the repository boundary used for safe relative evidence references."""

    for candidate in (matrix_path.parent, *matrix_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate.resolve()
    return matrix_path.parent.resolve()


def _is_external_location(location: str) -> bool:
    return "://" in location


def _resolve_local_reference(matrix_path: Path, location: str) -> Path:
    """Resolve one portable local reference without crossing the repository boundary."""

    relative = Path(location)
    if relative.is_absolute() or ".." in relative.parts:
        raise ArtifactIntegrityError(
            f"Compatibility evidence path escapes its repository: {location}",
            component="compatibility",
            remediation="Use a relative path contained by the matrix repository.",
            details={"location": location},
        )
    root = _repository_root(matrix_path)
    candidate = root.joinpath(relative)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ArtifactIntegrityError(
                f"Compatibility evidence path contains a symbolic link: {current}",
                component="compatibility",
                remediation="Reference the immutable regular evidence file directly.",
                details={"location": location, "path": str(current)},
            )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ArtifactIntegrityError(
            f"Unable to resolve compatibility evidence {location}: {error}",
            component="compatibility",
            remediation="Create the referenced local evidence record or use an external URI.",
            details={"location": location},
        ) from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ArtifactIntegrityError(
            f"Compatibility evidence is outside the repository or is not a file: {resolved}",
            component="compatibility",
            details={"location": location, "path": str(resolved)},
        )
    return resolved


def _scope_completeness_errors(cell: CompatibilityCell) -> list[str]:
    errors: list[str] = []
    if cell.protocol is None:
        errors.append("protocol identity is missing")
    if cell.software.cuda.status == "present" and cell.software.driver_version is None:
        errors.append("driver version is missing")
    if not cell.software.runtime_binaries:
        errors.append("runtime binary hashes are missing")
    if cell.software.build_flags is None:
        errors.append("build flags are missing")
    if cell.software.package_manifest_sha256 is None:
        errors.append("package manifest SHA-256 is missing")
    if cell.hardware.cpu_model is None:
        errors.append("CPU model is missing")
    if cell.hardware.logical_cpu_count is None:
        errors.append("logical CPU count is missing")
    if cell.hardware.ram_bytes is None:
        errors.append("RAM capacity is missing")
    if cell.hardware.topology_sha256 is None:
        errors.append("hardware topology SHA-256 is missing")
    if cell.hardware.gpu_count > 0 and len(cell.hardware.gpus) != cell.hardware.gpu_count:
        errors.append("GPU UUID, memory, and compute-capability identities are incomplete")
    return errors


def _verify_local_evidence(
    matrix: LoadedCompatibilityMatrix,
    cell: CompatibilityCell,
    reference: CompatibilityEvidenceReference,
) -> dict[str, Any]:
    from inkling_quant_lab.evidence import load_evidence_record

    path = _resolve_local_reference(matrix.path, reference.location)
    loaded = load_evidence_record(path)
    if loaded.file_sha256 != reference.sha256:
        raise ArtifactIntegrityError(
            f"Compatibility evidence SHA-256 mismatch for {reference.reference_id}",
            component="compatibility",
            remediation="Update the immutable reference only after reviewing the new record.",
            details={
                "reference_id": reference.reference_id,
                "expected_sha256": reference.sha256,
                "actual_sha256": loaded.file_sha256,
                "path": str(path),
            },
        )
    payload = loaded.record.payload
    mismatches: list[str] = []
    if payload.status != "success":
        mismatches.append(f"terminal outcome is {payload.status!r}, not 'success'")
    if payload.result_kind != reference.kind:
        mismatches.append(f"result kind is {payload.result_kind!r}, expected {reference.kind!r}")
    expected_scope_sha256 = cell.exact_scope_sha256()
    if payload.compatibility_scope_sha256 != expected_scope_sha256:
        mismatches.append("complete compatibility scope SHA-256 differs")
    if payload.subject.model_id != cell.model.model_id:
        mismatches.append("model ID differs")
    if payload.subject.model_revision != cell.model.revision:
        mismatches.append("model revision differs")
    toolchain = payload.subject.software_toolchain
    if toolchain is None or toolchain.runtime_revision != cell.runtime.commit:
        mismatches.append("runtime revision differs")
    expected_image_digest = cell.software.container_image.rsplit("@", maxsplit=1)[-1]
    if toolchain is None or toolchain.container_image_digest != expected_image_digest:
        mismatches.append("container image digest differs")
    evidence_artifact_hashes = {item.sha256 for item in payload.artifacts}
    missing_artifacts = sorted(
        item.artifact_id for item in cell.artifacts if item.sha256 not in evidence_artifact_hashes
    )
    if missing_artifacts:
        mismatches.append(
            "cell artifact hashes are absent from evidence: " + ", ".join(missing_artifacts)
        )
    if mismatches:
        raise ArtifactIntegrityError(
            f"Compatibility evidence does not match cell {cell.cell_id}",
            component="compatibility",
            remediation="Use successful terminal evidence produced for this exact matrix cell.",
            details={
                "cell_id": cell.cell_id,
                "reference_id": reference.reference_id,
                "mismatches": mismatches,
                "path": str(path),
            },
        )
    return {
        "reference_id": reference.reference_id,
        "kind": reference.kind,
        "location": reference.location,
        "resolution": "verified_local",
        "path": str(path),
        "file_sha256": loaded.file_sha256,
        "record_id": payload.record_id,
        "payload_sha256": loaded.record.payload_sha256,
        "terminal_outcome": payload.status,
        "scope_sha256": expected_scope_sha256,
    }


def _supported_reference_ids(cell: CompatibilityCell) -> set[str]:
    claims = cell.claims
    return {
        reference
        for claim in (
            claims.structurally_verified,
            claims.smoke_passed,
            claims.quality_measured,
            claims.performance_measured,
            claims.mtp_supported,
        )
        if claim.status == "supported"
        for reference in claim.evidence_refs
    }


def validate_compatibility_matrix(matrix: LoadedCompatibilityMatrix) -> dict[str, Any]:
    """Verify complete cell scope and all safely resolvable terminal evidence."""

    payload = matrix.record.payload
    reference_results: list[dict[str, Any]] = []
    for cell in payload.cells:
        completeness_errors = _scope_completeness_errors(cell)
        if completeness_errors:
            raise ArtifactIntegrityError(
                f"Compatibility cell {cell.cell_id} lacks complete exact scope",
                component="compatibility",
                remediation=(
                    "Record protocol, driver, runtime binaries, build flags, packages, "
                    "CPU/RAM/topology, and every GPU UUID/capacity/capability."
                ),
                details={"cell_id": cell.cell_id, "missing": completeness_errors},
            )
        supported_reference_ids = _supported_reference_ids(cell)
        for reference in cell.evidence:
            if _is_external_location(reference.location):
                result = {
                    "reference_id": reference.reference_id,
                    "kind": reference.kind,
                    "location": reference.location,
                    "resolution": "unresolved_external",
                }
                reference_results.append(result)
                if reference.reference_id in supported_reference_ids:
                    raise ArtifactIntegrityError(
                        f"Supported claim cites unresolved evidence {reference.reference_id}",
                        component="compatibility",
                        remediation=(
                            "Materialize and checksum a local terminal evidence record before "
                            "publishing support."
                        ),
                        details={"cell_id": cell.cell_id, **result},
                    )
                continue
            reference_results.append(_verify_local_evidence(matrix, cell, reference))

    return {
        "schema_version": "iql-compatibility-validation-v1",
        "status": "valid",
        "matrix_path": str(matrix.path),
        "matrix_file_sha256": matrix.file_sha256,
        "payload_sha256": matrix.record.payload_sha256,
        "matrix_id": payload.matrix_id,
        "scope_policy": payload.scope_policy,
        "cell_count": len(payload.cells),
        "validated_checks": [
            "supported_schema",
            "strict_fields",
            "canonical_payload_sha256",
            "exact_model_runtime_software_hardware_scope",
            "exact_protocol_scope",
            "driver_gpu_cpu_ram_topology_scope",
            "runtime_binary_build_and_package_scope",
            "checksum_pinned_artifacts_and_evidence",
            "local_evidence_bytes_and_sha256",
            "successful_terminal_evidence",
            "terminal_evidence_exact_cell_binding",
            "claim_dependencies",
            "no_duplicate_cells",
        ],
        "reference_results": reference_results,
        "referenced_bytes_verified": all(
            item["resolution"] == "verified_local" for item in reference_results
        ),
        "warnings": [payload.scope_warning],
    }


__all__ = [
    "COMPATIBILITY_SCOPE_WARNING",
    "CompatibilityArtifact",
    "CompatibilityCell",
    "CompatibilityClaim",
    "CompatibilityClaims",
    "CompatibilityEvidenceReference",
    "CompatibilityMatrixPayload",
    "CompatibilityMatrixRecord",
    "CudaIdentity",
    "GpuDeviceIdentity",
    "HardwareIdentity",
    "LoadedCompatibilityMatrix",
    "ModelIdentity",
    "ProtocolIdentity",
    "RuntimeBinaryIdentity",
    "RuntimeIdentity",
    "SoftwareIdentity",
    "compatibility_payload_sha256",
    "inspect_compatibility_matrix",
    "load_compatibility_matrix",
    "seal_compatibility_matrix",
    "validate_compatibility_matrix",
]
