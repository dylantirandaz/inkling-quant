"""Strict, self-checking experiment evidence records and read-only operations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from inkling_quant_lab.comparison import ComparisonResult, NormalizedRunSummary, compare_summaries
from inkling_quant_lab.exceptions import ArtifactIntegrityError, ComparisonCompatibilityError

EvidenceStatus: TypeAlias = Literal["success", "partial", "failed"]
EvidenceResultKind: TypeAlias = Literal[
    "structural_verification",
    "inference_smoke",
    "quality_evaluation",
    "performance_benchmark",
    "mtp_validation",
]
EvidenceComparisonRole: TypeAlias = Literal["baseline", "candidate", "standalone"]
EvidenceArtifactRole: TypeAlias = Literal[
    "model_export",
    "projector",
    "runtime_binary",
    "dataset",
    "auxiliary",
]
ComparisonArtifactRole: TypeAlias = Literal["model_export", "projector"]

_FORBIDDEN_IDENTITY_VALUES = frozenset(
    {
        "*",
        "all",
        "any",
        "example",
        "n/a",
        "na",
        "none",
        "placeholder",
        "tbd",
        "todo",
        "unknown",
        "unspecified",
    }
)

EVIDENCE_SCOPE_WARNING = (
    "Read each machine-readable experiment record before using its result. "
    "Do not apply a result to a different model, dataset, runtime, software, "
    "hardware, or protocol."
)


class _ImmutableEvidenceModel(BaseModel):
    """Strict immutable base for evidence records."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def _require_exact_identity(value: str, *, label: str) -> str:
    if not value or value.strip() != value:
        raise ValueError(f"{label} must be non-empty and trimmed")
    if value.casefold() in _FORBIDDEN_IDENTITY_VALUES or "*" in value or "?" in value:
        raise ValueError(f"{label} must identify one exact value")
    return value


class EvidenceArtifactIdentity(_ImmutableEvidenceModel):
    """Checksum-pinned artifact named by an evidence record."""

    artifact_id: str = Field(min_length=1)
    role: EvidenceArtifactRole
    location: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @field_validator("artifact_id", "location")
    @classmethod
    def text_has_no_surrounding_whitespace(cls, value: str) -> str:
        return _require_exact_identity(value, label="artifact identity")


class ComparisonArtifactIdentity(_ImmutableEvidenceModel):
    """Expected model-bearing artifact bound into a comparison protocol."""

    artifact_id: str = Field(min_length=1)
    role: ComparisonArtifactRole
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("artifact_id")
    @classmethod
    def artifact_id_is_exact(cls, value: str) -> str:
        return _require_exact_identity(value, label="comparison artifact ID")


class CandidateQuantizationIdentity(_ImmutableEvidenceModel):
    """Exact quantizer and policy expected for the candidate artifact set."""

    quantization_type: str = Field(min_length=1)
    quantizer_name: str = Field(min_length=1)
    quantizer_revision: str = Field(min_length=1)
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_artifact_ids: tuple[str, ...]
    output_artifact_ids: tuple[str, ...]

    @field_validator("quantization_type", "quantizer_name", "quantizer_revision")
    @classmethod
    def quantizer_identity_is_exact(cls, value: str) -> str:
        return _require_exact_identity(value, label="candidate quantization identity")

    @field_validator("input_artifact_ids", "output_artifact_ids")
    @classmethod
    def artifact_ids_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("candidate quantization requires non-empty artifact IDs")
        for artifact_id in value:
            _require_exact_identity(artifact_id, label="quantization artifact ID")
        if value != tuple(sorted(set(value))):
            raise ValueError("quantization artifact IDs must be sorted and unique")
        return value


class EvidenceComparisonProtocol(_ImmutableEvidenceModel):
    """Immutable BF16/candidate artifact and quantization contract."""

    protocol_id: str = Field(min_length=1)
    baseline_artifacts: tuple[ComparisonArtifactIdentity, ...]
    candidate_artifacts: tuple[ComparisonArtifactIdentity, ...]
    candidate_quantization: CandidateQuantizationIdentity

    @field_validator("protocol_id")
    @classmethod
    def protocol_id_is_exact(cls, value: str) -> str:
        return _require_exact_identity(value, label="comparison protocol ID")

    @field_validator("baseline_artifacts", "candidate_artifacts")
    @classmethod
    def artifact_sets_are_canonical(
        cls,
        value: tuple[ComparisonArtifactIdentity, ...],
    ) -> tuple[ComparisonArtifactIdentity, ...]:
        if not value:
            raise ValueError("comparison protocol requires non-empty artifact sets")
        ids = tuple(item.artifact_id for item in value)
        if len(ids) != len(set(ids)):
            raise ValueError("comparison protocol artifact IDs must be unique")
        if value != tuple(sorted(value, key=lambda item: (item.role, item.artifact_id))):
            raise ValueError("comparison protocol artifacts must be canonically sorted")
        if not any(item.role == "model_export" for item in value):
            raise ValueError("comparison protocol requires a model_export artifact")
        return value

    @model_validator(mode="after")
    def quantization_binds_baseline_and_candidate(self) -> EvidenceComparisonProtocol:
        baseline_ids = tuple(
            sorted(
                item.artifact_id for item in self.baseline_artifacts if item.role == "model_export"
            )
        )
        candidate_ids = tuple(
            sorted(
                item.artifact_id for item in self.candidate_artifacts if item.role == "model_export"
            )
        )
        if self.candidate_quantization.input_artifact_ids != baseline_ids:
            raise ValueError("candidate quantization input IDs must equal baseline model exports")
        if self.candidate_quantization.output_artifact_ids != candidate_ids:
            raise ValueError("candidate quantization output IDs must equal candidate model exports")
        return self


class ExperimentEvidencePayload(_ImmutableEvidenceModel):
    """Scientific scope and result contained in one evidence record."""

    record_kind: Literal["experiment_result"] = "experiment_result"
    record_id: str = Field(min_length=1)
    result_kind: EvidenceResultKind
    status: EvidenceStatus
    comparison_role: EvidenceComparisonRole
    comparison_protocol: EvidenceComparisonProtocol | None = None
    compatibility_scope_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    subject: NormalizedRunSummary
    artifacts: tuple[EvidenceArtifactIdentity, ...]
    claims: tuple[str, ...] = ()
    limitations: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @field_validator("record_id")
    @classmethod
    def record_id_has_no_surrounding_whitespace(cls, value: str) -> str:
        return _require_exact_identity(value, label="record ID")

    @field_validator("claims", "limitations", "warnings")
    @classmethod
    def statements_are_stable_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or item.strip() != item for item in value):
            raise ValueError("evidence statements must be non-empty and trimmed")
        if len(set(value)) != len(value):
            raise ValueError("evidence statements must be unique")
        return value

    @field_validator("artifacts")
    @classmethod
    def artifacts_are_complete_and_canonical(
        cls,
        value: tuple[EvidenceArtifactIdentity, ...],
    ) -> tuple[EvidenceArtifactIdentity, ...]:
        if not value:
            raise ValueError("evidence requires at least one checksum-pinned artifact")
        artifact_ids = tuple(item.artifact_id for item in value)
        locations = tuple(item.location for item in value)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("evidence artifact IDs must be unique")
        if len(set(locations)) != len(locations):
            raise ValueError("evidence artifact locations must be unique")
        if not any(item.role == "model_export" for item in value):
            raise ValueError("evidence requires at least one model_export artifact")
        ordering = tuple(
            sorted(value, key=lambda item: (item.role, item.artifact_id, item.location))
        )
        if value != ordering:
            raise ValueError("evidence artifacts must be canonically sorted")
        return value

    @model_validator(mode="after")
    def subject_is_evidence_ready(self) -> ExperimentEvidencePayload:
        if self.subject.schema_version != "1.2":
            raise ValueError("iql-evidence-v1 requires normalized summary schema 1.2")
        if not self.limitations:
            raise ValueError("evidence requires at least one explicit limitation")
        if self.status == "success" and self.subject.failures:
            raise ValueError("successful evidence cannot contain recorded failures")
        if self.status == "failed" and not self.subject.failures:
            raise ValueError("failed evidence must preserve at least one failure")
        if self.comparison_role == "standalone":
            if self.comparison_protocol is not None:
                raise ValueError("standalone evidence cannot declare a comparison protocol")
            return self
        if self.comparison_protocol is None:
            raise ValueError("baseline and candidate evidence require a comparison protocol")

        expected = (
            self.comparison_protocol.baseline_artifacts
            if self.comparison_role == "baseline"
            else self.comparison_protocol.candidate_artifacts
        )
        actual = tuple(
            ComparisonArtifactIdentity(
                artifact_id=item.artifact_id,
                role=item.role,
                sha256=item.sha256,
            )
            for item in self.artifacts
            if item.role in {"model_export", "projector"}
        )
        actual = tuple(sorted(actual, key=lambda item: (item.role, item.artifact_id)))
        if actual != expected:
            raise ValueError(
                f"{self.comparison_role} artifacts do not match the comparison protocol"
            )
        if self.comparison_role == "candidate":
            policy_digest = hashlib.sha256(
                _canonical_json(self.subject.quantization_policy).encode("utf-8")
            ).hexdigest()
            if policy_digest != self.comparison_protocol.candidate_quantization.policy_sha256:
                raise ValueError(
                    "candidate quantization policy does not match the comparison protocol"
                )
        return self


class ExperimentEvidenceRecord(_ImmutableEvidenceModel):
    """Versioned envelope whose payload digest detects accidental mutation."""

    schema_version: Literal["iql-evidence-v1"] = "iql-evidence-v1"
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: ExperimentEvidencePayload

    @model_validator(mode="after")
    def payload_digest_matches(self) -> ExperimentEvidenceRecord:
        expected = evidence_payload_sha256(self.payload)
        if self.payload_sha256 != expected:
            raise ValueError(
                "evidence payload SHA-256 mismatch: "
                f"expected {expected}, recorded {self.payload_sha256}"
            )
        return self


@dataclass(frozen=True)
class LoadedEvidenceRecord:
    """Validated evidence plus the exact file identity used by a command."""

    path: Path
    file_sha256: str
    record: ExperimentEvidenceRecord


@dataclass(frozen=True)
class EvidenceComparison:
    """Validated sources and their strict normalized comparison result."""

    baseline: LoadedEvidenceRecord
    candidate: LoadedEvidenceRecord
    result: ComparisonResult


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


def _sequence_sha256(values: tuple[str, ...]) -> str:
    return hashlib.sha256(_canonical_json(list(values)).encode("utf-8")).hexdigest()


def evidence_payload_sha256(payload: ExperimentEvidencePayload) -> str:
    """Hash the canonical evidence payload, excluding the envelope digest."""

    encoded = _canonical_json(payload.model_dump(mode="json")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_evidence_record(payload: ExperimentEvidencePayload) -> ExperimentEvidenceRecord:
    """Create a self-checking evidence envelope from a validated payload."""

    return ExperimentEvidenceRecord(
        payload_sha256=evidence_payload_sha256(payload),
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


def load_evidence_record(path: str | Path) -> LoadedEvidenceRecord:
    """Load one exact supported schema and verify its canonical payload digest."""

    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise ArtifactIntegrityError(
            f"Evidence record cannot be a symbolic link: {supplied}",
            component="evidence",
            remediation="Pass the immutable evidence JSON file directly.",
            details={"path": str(supplied)},
        )
    try:
        resolved = supplied.resolve(strict=True)
        if not resolved.is_file():
            raise ArtifactIntegrityError(
                f"Evidence record is not a regular file: {resolved}",
                component="evidence",
                details={"path": str(resolved)},
            )
        raw = resolved.read_bytes()
        text = raw.decode("utf-8")
    except ArtifactIntegrityError:
        raise
    except (OSError, UnicodeDecodeError) as error:
        raise ArtifactIntegrityError(
            f"Unable to read evidence record {supplied}: {error}",
            component="evidence",
            remediation="Verify the path, permissions, and UTF-8 encoding.",
            details={"path": str(supplied)},
        ) from error
    try:
        document = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, _DuplicateKeyError) as error:
        raise ArtifactIntegrityError(
            f"Unable to parse evidence record {resolved}: {error}",
            component="evidence",
            remediation="Provide one unambiguous iql-evidence-v1 JSON object.",
            details={"path": str(resolved)},
        ) from error
    try:
        record = ExperimentEvidenceRecord.model_validate(document)
    except ValidationError as error:
        details = _validation_details(error)
        raise ArtifactIntegrityError(
            "Evidence record schema or payload integrity validation failed: "
            + "; ".join(f"{item['field']}: {item['message']}" for item in details),
            component="evidence",
            remediation="Use an exact, checksum-valid iql-evidence-v1 record.",
            details={"path": str(resolved), "errors": details},
        ) from error
    return LoadedEvidenceRecord(
        path=resolved,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        record=record,
    )


def _scope_warnings(payload: ExperimentEvidencePayload) -> list[str]:
    return list(dict.fromkeys((EVIDENCE_SCOPE_WARNING, *payload.warnings)))


def inspect_evidence_record(record: LoadedEvidenceRecord) -> dict[str, Any]:
    """Return a bounded description of one already validated record."""

    payload = record.record.payload
    subject = payload.subject
    return {
        "schema_version": "iql-evidence-inspection-v1",
        "status": "inspected",
        "record_path": str(record.path),
        "record_file_sha256": record.file_sha256,
        "payload_sha256": record.record.payload_sha256,
        "record_id": payload.record_id,
        "record_kind": payload.record_kind,
        "result_kind": payload.result_kind,
        "experiment_status": payload.status,
        "comparison_ready": payload.status in {"success", "partial"},
        "comparison_governance": {
            "role": payload.comparison_role,
            "protocol": payload.comparison_protocol.model_dump(mode="json")
            if payload.comparison_protocol is not None
            else None,
            "compatibility_scope_sha256": payload.compatibility_scope_sha256,
        },
        "integrity": {
            "schema_validated": True,
            "payload_digest_verified": True,
            "artifact_bytes_verified": False,
        },
        "scope": {
            "run_id": subject.run_id,
            "model_id": subject.model_id,
            "model_revision": subject.model_revision,
            "source_model_sha256": subject.source_model_sha256,
            "software_toolchain": subject.software_toolchain.model_dump(mode="json")
            if subject.software_toolchain is not None
            else None,
            "datasets": [item.model_dump(mode="json") for item in subject.datasets],
            "evaluation_suites": [
                item.model_dump(mode="json") for item in subject.evaluation_suites
            ],
            "evaluation_suite_count": len(subject.evaluation_suites),
            "sample_count": len(subject.sample_ids),
            "sample_ids_sha256": _sequence_sha256(subject.sample_ids),
            "seed_set": list(subject.seed_set),
            "prompt_template_hash": subject.prompt_template_hash,
            "decode_config": subject.decode_config,
            "routing_dataset": subject.routing_dataset.model_dump(mode="json")
            if subject.routing_dataset is not None
            else None,
            "routing_sample_count": len(subject.routing_sample_ids),
            "routing_sample_ids_sha256": _sequence_sha256(subject.routing_sample_ids),
            "routing_capture": subject.routing_capture.model_dump(mode="json")
            if subject.routing_capture is not None
            else None,
            "benchmark_protocol_version": subject.benchmark_protocol_version,
            "benchmark_workload": subject.benchmark_workload.model_dump(mode="json")
            if subject.benchmark_workload is not None
            else None,
            "benchmark_memory": subject.benchmark_memory.model_dump(mode="json")
            if subject.benchmark_memory is not None
            else None,
            "hardware_environment": subject.hardware_environment,
            "environment": subject.environment,
            "resolved_config": subject.resolved_config,
            "quantization_policy": subject.quantization_policy,
        },
        "artifacts": [item.model_dump(mode="json") for item in payload.artifacts],
        "metrics": {
            name: metric.model_dump(mode="json") for name, metric in sorted(subject.metrics.items())
        },
        "claims": list(payload.claims),
        "limitations": list(payload.limitations),
        "failures": list(subject.failures),
        "unsupported_measurements": list(subject.unsupported_measurements),
        "warnings": _scope_warnings(payload),
    }


def validate_evidence_record(record: LoadedEvidenceRecord) -> dict[str, Any]:
    """Describe the checks completed by the strict standalone loader."""

    payload = record.record.payload
    return {
        "schema_version": "iql-evidence-validation-v1",
        "status": "valid",
        "record_path": str(record.path),
        "record_file_sha256": record.file_sha256,
        "payload_sha256": record.record.payload_sha256,
        "record_id": payload.record_id,
        "result_kind": payload.result_kind,
        "experiment_status": payload.status,
        "comparison_ready": payload.status in {"success", "partial"},
        "validated_checks": [
            "supported_schema",
            "strict_fields",
            "canonical_payload_sha256",
            "summary_scope",
            "artifact_identities",
            "comparison_artifact_binding",
            "candidate_quantization_identity",
            "failure_status_consistency",
        ],
        "artifact_bytes_verified": False,
        "warnings": _scope_warnings(payload),
    }


def compare_evidence_records(
    baseline: LoadedEvidenceRecord,
    candidate: LoadedEvidenceRecord,
    *,
    unsafe_overrides: set[str] | None = None,
) -> EvidenceComparison:
    """Compare two validated records under the normalized strict contract."""

    unusable = [
        loaded.record.payload.record_id
        for loaded in (baseline, candidate)
        if loaded.record.payload.status == "failed"
    ]
    if unusable:
        raise ComparisonCompatibilityError(
            "Failed experiment evidence cannot be used for metric comparison",
            component="evidence",
            remediation="Use successful or explicitly partial evidence records.",
            details={
                "mismatches": [
                    {
                        "dimension": "experiment_status",
                        "baseline": baseline.record.payload.status,
                        "candidate": candidate.record.payload.status,
                        "message": "failed experiment evidence is not comparison-ready",
                    }
                ],
                "failed_record_ids": unusable,
            },
        )
    baseline_payload = baseline.record.payload
    candidate_payload = candidate.record.payload
    protocol_mismatches: list[dict[str, JsonValue]] = []
    if baseline_payload.comparison_role != "baseline":
        protocol_mismatches.append(
            {
                "dimension": "baseline_comparison_role",
                "baseline": baseline_payload.comparison_role,
                "candidate": candidate_payload.comparison_role,
                "message": "the first record must declare comparison_role='baseline'",
            }
        )
    if candidate_payload.comparison_role != "candidate":
        protocol_mismatches.append(
            {
                "dimension": "candidate_comparison_role",
                "baseline": baseline_payload.comparison_role,
                "candidate": candidate_payload.comparison_role,
                "message": "the second record must declare comparison_role='candidate'",
            }
        )
    baseline_protocol = (
        baseline_payload.comparison_protocol.model_dump(mode="json")
        if baseline_payload.comparison_protocol is not None
        else None
    )
    candidate_protocol = (
        candidate_payload.comparison_protocol.model_dump(mode="json")
        if candidate_payload.comparison_protocol is not None
        else None
    )
    if baseline_protocol != candidate_protocol:
        protocol_mismatches.append(
            {
                "dimension": "comparison_protocol",
                "baseline": baseline_protocol,
                "candidate": candidate_protocol,
                "message": (
                    "baseline/candidate artifact hashes or candidate quantization identity differ"
                ),
            }
        )
    if protocol_mismatches:
        raise ComparisonCompatibilityError(
            "Evidence records do not share the required baseline/candidate protocol",
            component="evidence",
            remediation=(
                "Use records sealed against one protocol with the declared BF16 baseline "
                "and quantized candidate artifact identities."
            ),
            details={"mismatches": protocol_mismatches},
        )

    result = compare_summaries(
        baseline_payload.subject,
        candidate_payload.subject,
        unsafe_overrides=unsafe_overrides,
    )
    return EvidenceComparison(baseline=baseline, candidate=candidate, result=result)


def evidence_comparison_payload(comparison: EvidenceComparison) -> dict[str, Any]:
    """Return stable machine-readable output for a read-only evidence comparison."""

    baseline_payload = comparison.baseline.record.payload
    candidate_payload = comparison.candidate.record.payload
    warnings = list(
        dict.fromkeys(
            (
                EVIDENCE_SCOPE_WARNING,
                *baseline_payload.warnings,
                *candidate_payload.warnings,
                *comparison.result.warnings,
            )
        )
    )
    comparison_status = (
        "success"
        if baseline_payload.status == "success" and candidate_payload.status == "success"
        else "partial"
    )
    return {
        "schema_version": "iql-evidence-comparison-v1",
        "status": comparison_status,
        "baseline": {
            "record_id": baseline_payload.record_id,
            "record_path": str(comparison.baseline.path),
            "record_file_sha256": comparison.baseline.file_sha256,
            "payload_sha256": comparison.baseline.record.payload_sha256,
            "experiment_status": baseline_payload.status,
            "comparison_role": baseline_payload.comparison_role,
            "artifacts": [item.model_dump(mode="json") for item in baseline_payload.artifacts],
        },
        "candidate": {
            "record_id": candidate_payload.record_id,
            "record_path": str(comparison.candidate.path),
            "record_file_sha256": comparison.candidate.file_sha256,
            "payload_sha256": comparison.candidate.record.payload_sha256,
            "experiment_status": candidate_payload.status,
            "comparison_role": candidate_payload.comparison_role,
            "artifacts": [item.model_dump(mode="json") for item in candidate_payload.artifacts],
        },
        "comparison_protocol": baseline_payload.comparison_protocol.model_dump(mode="json")
        if baseline_payload.comparison_protocol is not None
        else None,
        "failures": {
            "baseline": list(baseline_payload.subject.failures),
            "candidate": list(candidate_payload.subject.failures),
        },
        "unsupported_measurements": {
            "baseline": list(baseline_payload.subject.unsupported_measurements),
            "candidate": list(candidate_payload.subject.unsupported_measurements),
        },
        "comparison": comparison.result.model_dump(mode="json"),
        "warnings": warnings,
    }
