"""Strict evidence contracts for the matched Inkling Modal measurement.

This module describes immutable evidence. It does not launch Modal, execute
llama.cpp, load a model, or provide a CPU substitute for CUDA validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import PurePosixPath
from typing import Any, Final, Literal, TypeAlias, cast

from pydantic import (
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.gguf.inkling_matched_execution import (
    ExactCudaPlacementPolicy,
    parse_exact_cuda_backend_audit,
)
from inkling_quant_lab.gguf.inkling_measurement_control import (
    MeasurementBenchCaseRollup,
    MeasurementLlamaBenchWorkloadIdentity,
    MeasurementPairedBytes,
    MeasurementPairedFiveBatchMetricSummary,
    MeasurementPairedFiveBatchNonnegativeMetricSummary,
    MeasurementPairedGpuBytes,
    MeasurementPairedGpuUtilization,
    MeasurementPairedNonnegativeValue,
    MeasurementPairedPositiveValue,
    MeasurementPairedRepeatedLoadDurations,
    MeasurementPerformanceRollup,
    MeasurementQualityRollup,
    MeasurementServerCellRollup,
    MeasurementServerWorkloadIdentity,
    MeasurementSuiteQuality,
    MeasurementSupportingRecordReference,
    build_measurement_supporting_record_reference,
    validate_absolute_evidence_path,
)
from inkling_quant_lab.gguf.inkling_measurement_raw_evidence import (
    MeasurementBackendAuditEvidence,
    MeasurementFiveBatchMetricSummary,
    MeasurementFiveBatchNonnegativeMetricSummary,
    MeasurementParsedRawEvidence,
    MeasurementRepeatedLoadDurations,
    MeasurementSubjectPerformanceSummary,
    MeasurementSubjectQualitySummary,
    parse_measurement_raw_evidence,
)

MeasurementEvidenceSubject: TypeAlias = Literal["bf16", "q3"]
MeasurementRawBlobKind: TypeAlias = Literal[
    "token_nll",
    "raw_trials",
    "resource_telemetry",
    "backend_audit",
]
MeasurementRawBlobFormat: TypeAlias = Literal["json", "jsonl"]
MeasurementPlacementWorkload: TypeAlias = Literal[
    "perplexity",
    "server_quality_and_performance",
    "llama_bench",
]

MEASUREMENT_RAW_BLOB_KIND_ORDER: Final = (
    "token_nll",
    "raw_trials",
    "resource_telemetry",
    "backend_audit",
)
MEASUREMENT_PLACEMENT_WORKLOAD_ORDER: Final = (
    "perplexity",
    "server_quality_and_performance",
    "llama_bench",
)
MEASUREMENT_CUDA_IDENTITY_ORDER: Final = tuple(
    (ordinal, f"CUDA{ordinal}", f"CUDA{ordinal}") for ordinal in range(8)
)
MEASUREMENT_SUBJECT_QUALITY_PROJECTION_HASH_DOMAIN: Final = (
    b"inkling-measurement-subject-quality-projection-v1\0"
)
MEASUREMENT_SUBJECT_PERFORMANCE_PROJECTION_HASH_DOMAIN: Final = (
    b"inkling-measurement-subject-performance-projection-v1\0"
)

_RUN_ID_PATTERN: Final = r"^[a-z0-9][a-z0-9._-]{0,95}$"
_SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
_MODAL_CALL_ID_PATTERN: Final = r"^fc-[A-Za-z0-9]+$"
_RAW_BLOB_FORMAT: Final[
    dict[
        MeasurementRawBlobKind,
        tuple[MeasurementRawBlobFormat, MeasurementRawBlobFormat],
    ]
] = {
    "token_nll": ("jsonl", "jsonl"),
    "raw_trials": ("json", "json"),
    "resource_telemetry": ("jsonl", "jsonl"),
    "backend_audit": ("json", "json"),
}
MEASUREMENT_RAW_BLOB_MAX_BYTES: Final[dict[str, int]] = {
    "token_nll": 16 * 1024 * 1024,
    "raw_trials": 64 * 1024 * 1024,
    "resource_telemetry": 256 * 1024 * 1024,
    "backend_audit": 64 * 1024 * 1024,
}
MEASUREMENT_RAW_BLOB_RECORD_COUNTS: Final[dict[str, int | None]] = {
    "token_nll": 16_320,
    "raw_trials": 1,
    "resource_telemetry": None,
    "backend_audit": 1,
}


class _StrictEvidenceModel(StrictFrozenModel):
    """Fail-closed base for finite immutable evidence records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


def canonical_measurement_evidence_json_bytes(value: object) -> bytes:
    """Encode one value as canonical UTF-8 JSON with one trailing line feed."""

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


def measurement_evidence_sha256(value: object) -> str:
    """Hash one canonical evidence value."""

    return hashlib.sha256(canonical_measurement_evidence_json_bytes(value)).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _require_finite_json(value: object) -> None:
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number is forbidden")
        return
    if isinstance(value, list):
        for item in value:
            _require_finite_json(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("JSON object keys must be strings")
            _require_finite_json(item)
        return
    raise ValueError("value is outside the JSON data model")


def _strict_json_value(payload: bytes) -> object:
    if type(payload) is not bytes:
        raise TypeError("measurement evidence payload must be bytes")
    if not payload:
        raise ValueError("measurement evidence payload must not be empty")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("measurement evidence JSON must be UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_finite_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("measurement evidence JSON is invalid") from error
    _require_finite_json(value)
    return value


def _strict_canonical_json_value(payload: bytes) -> object:
    value = _strict_json_value(payload)
    if payload != canonical_measurement_evidence_json_bytes(value):
        raise ValueError("measurement evidence JSON bytes are not canonical")
    return value


def _validate_run_id(value: str) -> str:
    if type(value) is not str or re.fullmatch(_RUN_ID_PATTERN, value) is None:
        raise ValueError("measurement run ID is invalid")
    return value


def _validate_sha256(value: str, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(_SHA256_PATTERN, value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _validate_relative_posix_path(value: str) -> str:
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or "\x00" in value
        or value.startswith("/")
    ):
        raise ValueError("evidence path must be canonical repository-relative POSIX")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("evidence path must be canonical repository-relative POSIX")
    return value


def measurement_raw_blob_path(
    run_id: str,
    *,
    subject: MeasurementEvidenceSubject,
    kind: MeasurementRawBlobKind,
    content_sha256: str,
) -> str:
    """Return the only content-addressed path for one raw evidence blob."""

    _validate_run_id(run_id)
    _validate_sha256(content_sha256, label="raw evidence content hash")
    _, suffix = _RAW_BLOB_FORMAT[kind]
    return PurePosixPath(
        "runs",
        run_id,
        "raw",
        subject,
        kind,
        f"{content_sha256}.{suffix}",
    ).as_posix()


def _canonical_raw_blob_record_count(
    payload: bytes,
    *,
    kind: MeasurementRawBlobKind,
) -> int:
    parsed: MeasurementParsedRawEvidence = parse_measurement_raw_evidence(
        payload,
        kind=kind,
    )
    if kind in ("raw_trials", "backend_audit"):
        return 1
    return len(cast("Any", parsed).rows)


class MeasurementRawBlobReference(_StrictEvidenceModel):
    """Content-addressed reference to one canonical raw evidence blob."""

    schema_version: Literal["inkling-measurement-raw-reference-v1"] = (
        "inkling-measurement-raw-reference-v1"
    )
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    subject: MeasurementEvidenceSubject
    kind: MeasurementRawBlobKind
    format: MeasurementRawBlobFormat
    relative_path: StrictStr
    content_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    size_bytes: StrictInt = Field(gt=0, le=max(MEASUREMENT_RAW_BLOB_MAX_BYTES.values()))
    record_count: StrictInt = Field(gt=0)

    @field_validator("relative_path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        return _validate_relative_posix_path(value)

    @model_validator(mode="after")
    def format_and_path_are_derived(self) -> MeasurementRawBlobReference:
        expected_format, _ = _RAW_BLOB_FORMAT[self.kind]
        if self.format != expected_format:
            raise ValueError("raw evidence format differs from its kind")
        if self.size_bytes > MEASUREMENT_RAW_BLOB_MAX_BYTES[self.kind]:
            raise ValueError("raw evidence exceeds the size limit for its kind")
        expected_count = MEASUREMENT_RAW_BLOB_RECORD_COUNTS[self.kind]
        if expected_count is not None and self.record_count != expected_count:
            raise ValueError("raw evidence record count differs from its exact kind")
        expected_path = measurement_raw_blob_path(
            self.run_id,
            subject=self.subject,
            kind=self.kind,
            content_sha256=self.content_sha256,
        )
        if self.relative_path != expected_path:
            raise ValueError("raw evidence path differs from its exact identity")
        return self


def build_measurement_raw_blob_reference(
    payload: bytes,
    *,
    run_id: str,
    subject: MeasurementEvidenceSubject,
    kind: MeasurementRawBlobKind,
) -> MeasurementRawBlobReference:
    """Build a reference after validating the exact canonical raw bytes."""

    record_count = _canonical_raw_blob_record_count(payload, kind=kind)
    if len(payload) > MEASUREMENT_RAW_BLOB_MAX_BYTES[kind]:
        raise ValueError("raw evidence exceeds the size limit for its kind")
    expected_count = MEASUREMENT_RAW_BLOB_RECORD_COUNTS[kind]
    if expected_count is not None and record_count != expected_count:
        raise ValueError("raw evidence record count differs from its exact kind")
    digest = hashlib.sha256(payload).hexdigest()
    expected_format, _ = _RAW_BLOB_FORMAT[kind]
    return MeasurementRawBlobReference(
        run_id=run_id,
        subject=subject,
        kind=kind,
        format=expected_format,
        relative_path=measurement_raw_blob_path(
            run_id,
            subject=subject,
            kind=kind,
            content_sha256=digest,
        ),
        content_sha256=digest,
        size_bytes=len(payload),
        record_count=record_count,
    )


def validate_measurement_raw_blob_reference(
    payload: bytes,
    *,
    expected: MeasurementRawBlobReference,
) -> MeasurementRawBlobReference:
    """Rebuild and compare one raw reference against the exact bytes."""

    observed = build_measurement_raw_blob_reference(
        payload,
        run_id=expected.run_id,
        subject=expected.subject,
        kind=expected.kind,
    )
    if observed != expected:
        raise ValueError("raw evidence reference differs from its exact bytes")
    return observed


class MeasurementExecutableArtifactIdentity(_StrictEvidenceModel):
    """One exact model artifact copied to ephemeral storage and measured."""

    ordinal: StrictInt = Field(ge=0, le=49)
    role: Literal["text_shard", "multimodal_projector"]
    source_path: StrictStr
    staged_path: StrictStr
    sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    size_bytes: StrictInt = Field(gt=0)

    @field_validator("source_path", "staged_path")
    @classmethod
    def paths_are_canonical_absolute_posix(cls, value: str) -> str:
        return validate_absolute_evidence_path(value)


class MeasurementCudaIdentitySummary(_StrictEvidenceModel):
    """Aggregated compute count for one exact llama.cpp CUDA identity."""

    ordinal: StrictInt = Field(ge=0, le=7)
    backend_name: StrictStr
    device_name: StrictStr
    device_type: Literal["gpu"] = "gpu"
    compute_operations: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def identity_matches_ordinal(self) -> MeasurementCudaIdentitySummary:
        expected = f"CUDA{self.ordinal}"
        if self.backend_name != expected or self.device_name != expected:
            raise ValueError("CUDA backend identity differs from its exact ordinal")
        return self


class MeasurementPlacementSummary(_StrictEvidenceModel):
    """Compact proof that one workload's complete compute graph stayed on CUDA."""

    schema_version: Literal["inkling-measurement-placement-summary-v1"] = (
        "inkling-measurement-placement-summary-v1"
    )
    workload: MeasurementPlacementWorkload
    backend_audit_content_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    audit_log_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    command_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    placement_policy_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    observed_graphs: StrictInt = Field(gt=0)
    compute_operations: StrictInt = Field(gt=0)
    cuda_operations: StrictInt = Field(gt=0)
    cpu_operations: StrictInt = Field(ge=0, le=0)
    accelerator_operations: StrictInt = Field(ge=0, le=0)
    other_operations: StrictInt = Field(ge=0, le=0)
    unassigned_operations: StrictInt = Field(ge=0, le=0)
    cuda_identities: tuple[
        MeasurementCudaIdentitySummary,
        MeasurementCudaIdentitySummary,
        MeasurementCudaIdentitySummary,
        MeasurementCudaIdentitySummary,
        MeasurementCudaIdentitySummary,
        MeasurementCudaIdentitySummary,
        MeasurementCudaIdentitySummary,
        MeasurementCudaIdentitySummary,
    ]
    exact_cuda_identity_inventory: Literal[True] = True
    full_cell_text_observed: Literal[True] = True
    all_compute_operations_cuda: Literal[True] = True
    no_cpu_model_graph_fallback: Literal[True] = True

    @model_validator(mode="after")
    def all_compute_is_exact_cuda(self) -> MeasurementPlacementSummary:
        observed_identities = tuple(
            (item.ordinal, item.backend_name, item.device_name) for item in self.cuda_identities
        )
        if observed_identities != MEASUREMENT_CUDA_IDENTITY_ORDER:
            raise ValueError("placement summary CUDA inventory is incomplete or out of order")
        if self.cuda_operations != self.compute_operations:
            raise ValueError("placement summary did not assign every operation to CUDA")
        if sum(item.compute_operations for item in self.cuda_identities) != self.cuda_operations:
            raise ValueError("CUDA identity operation counts differ from the aggregate")
        return self


def measurement_subject_quality_projection_sha256(
    summary: MeasurementSubjectQualitySummary,
) -> str:
    """Hash one recomputed subject-quality projection."""

    if not isinstance(summary, MeasurementSubjectQualitySummary):
        raise TypeError("quality projection hash requires a validated summary")
    return hashlib.sha256(
        MEASUREMENT_SUBJECT_QUALITY_PROJECTION_HASH_DOMAIN
        + canonical_measurement_evidence_json_bytes(summary.model_dump(mode="json"))
    ).hexdigest()


def measurement_subject_performance_projection_sha256(
    summary: MeasurementSubjectPerformanceSummary,
) -> str:
    """Hash one recomputed subject-performance projection."""

    if not isinstance(summary, MeasurementSubjectPerformanceSummary):
        raise TypeError("performance projection hash requires a validated summary")
    return hashlib.sha256(
        MEASUREMENT_SUBJECT_PERFORMANCE_PROJECTION_HASH_DOMAIN
        + canonical_measurement_evidence_json_bytes(summary.model_dump(mode="json"))
    ).hexdigest()


def build_measurement_quality_rollup(
    bf16: MeasurementSubjectQualitySummary,
    q3: MeasurementSubjectQualitySummary,
    *,
    paired_inputs_validated: Literal[True],
) -> MeasurementQualityRollup:
    """Build paired quality after the caller proves identical scored inputs."""

    if paired_inputs_validated is not True:
        raise ValueError("quality rollup requires validated paired inputs")
    if bf16.subject != "bf16" or q3.subject != "q3":
        raise ValueError("quality summaries must be exact BF16 then Q3")
    if tuple(item.suite for item in bf16.suites) != tuple(item.suite for item in q3.suites):
        raise ValueError("quality summaries use different suite order")

    suites = tuple(
        MeasurementSuiteQuality(
            suite=baseline.suite,
            item_count=8,
            bf16_accuracy=baseline.accuracy,
            q3_accuracy=candidate.accuracy,
            accuracy_loss=baseline.accuracy - candidate.accuracy,
            bf16_floor_passed=baseline.accuracy >= 0.5,
            q3_non_inferiority_passed=(baseline.accuracy - candidate.accuracy <= 0.125),
        )
        for baseline, candidate in zip(bf16.suites, q3.suites, strict=True)
    )
    mean_nll_delta = q3.token_nll.mean_nll - bf16.token_nll.mean_nll
    overall_accuracy_loss = bf16.overall_accuracy - q3.overall_accuracy
    mean_nll_gate_passed = mean_nll_delta < 0.1
    overall_accuracy_gate_passed = overall_accuracy_loss <= 0.05
    bf16_overall_floor_passed = bf16.overall_accuracy >= 0.75
    bf16_suite_floors_passed = all(item.bf16_floor_passed for item in suites)
    suite_accuracy_gates_passed = all(item.q3_non_inferiority_passed for item in suites)
    return MeasurementQualityRollup(
        paired_token_positions=16_320,
        bf16_diagnostic_items_scored=bf16.diagnostic_items,
        q3_diagnostic_items_scored=q3.diagnostic_items,
        bf16_mean_nll=bf16.token_nll.mean_nll,
        q3_mean_nll=q3.token_nll.mean_nll,
        mean_nll_delta=mean_nll_delta,
        bf16_perplexity=bf16.printed_perplexity,
        q3_perplexity=q3.printed_perplexity,
        printed_perplexity_absolute_tolerance=(bf16.printed_perplexity_absolute_tolerance),
        bf16_overall_accuracy=bf16.overall_accuracy,
        q3_overall_accuracy=q3.overall_accuracy,
        overall_accuracy_loss=overall_accuracy_loss,
        suites=cast(
            "tuple[MeasurementSuiteQuality, MeasurementSuiteQuality, "
            "MeasurementSuiteQuality, MeasurementSuiteQuality, "
            "MeasurementSuiteQuality, MeasurementSuiteQuality, "
            "MeasurementSuiteQuality, MeasurementSuiteQuality]",
            suites,
        ),
        all_items_scored=True,
        all_suites_interpretable=True,
        mean_nll_gate_passed=mean_nll_gate_passed,
        overall_accuracy_gate_passed=overall_accuracy_gate_passed,
        bf16_overall_floor_passed=bf16_overall_floor_passed,
        bf16_suite_floors_passed=bf16_suite_floors_passed,
        suite_accuracy_gates_passed=suite_accuracy_gates_passed,
        non_inferiority_passed=all(
            (
                mean_nll_gate_passed,
                overall_accuracy_gate_passed,
                bf16_overall_floor_passed,
                bf16_suite_floors_passed,
                suite_accuracy_gates_passed,
            )
        ),
    )


def _paired_positive(bf16: float, q3: float) -> MeasurementPairedPositiveValue:
    return MeasurementPairedPositiveValue(
        bf16=bf16,
        q3=q3,
        q3_to_bf16_ratio=q3 / bf16,
    )


def _paired_five_batch_metric(
    bf16: MeasurementFiveBatchMetricSummary,
    q3: MeasurementFiveBatchMetricSummary,
) -> MeasurementPairedFiveBatchMetricSummary:
    return MeasurementPairedFiveBatchMetricSummary(
        trial_count_per_subject=5,
        bf16_samples=bf16.samples,
        q3_samples=q3.samples,
        mean=_paired_positive(bf16.mean, q3.mean),
        median=_paired_positive(bf16.median, q3.median),
        sample_standard_deviation=MeasurementPairedNonnegativeValue(
            bf16=bf16.sample_standard_deviation,
            q3=q3.sample_standard_deviation,
        ),
    )


def _paired_five_batch_nonnegative_metric(
    bf16: MeasurementFiveBatchNonnegativeMetricSummary,
    q3: MeasurementFiveBatchNonnegativeMetricSummary,
) -> MeasurementPairedFiveBatchNonnegativeMetricSummary:
    return MeasurementPairedFiveBatchNonnegativeMetricSummary(
        trial_count_per_subject=5,
        bf16_samples=bf16.samples,
        q3_samples=q3.samples,
        mean=MeasurementPairedNonnegativeValue(
            bf16=bf16.mean,
            q3=q3.mean,
        ),
        median=MeasurementPairedNonnegativeValue(
            bf16=bf16.median,
            q3=q3.median,
        ),
        sample_standard_deviation=MeasurementPairedNonnegativeValue(
            bf16=bf16.sample_standard_deviation,
            q3=q3.sample_standard_deviation,
        ),
    )


def _paired_load_durations(
    bf16: MeasurementRepeatedLoadDurations,
    q3: MeasurementRepeatedLoadDurations,
) -> MeasurementPairedRepeatedLoadDurations:
    if bf16.trial_count != 3 or q3.trial_count != 3:
        raise ValueError("performance rollup requires exactly three load pairs per subject")
    return MeasurementPairedRepeatedLoadDurations(
        trial_count_per_subject=3,
        bf16_durations_seconds=cast(
            "tuple[float, float, float]",
            bf16.durations_seconds,
        ),
        q3_durations_seconds=cast(
            "tuple[float, float, float]",
            q3.durations_seconds,
        ),
        median_seconds=_paired_positive(
            bf16.median_seconds,
            q3.median_seconds,
        ),
        sample_standard_deviation_seconds=MeasurementPairedNonnegativeValue(
            bf16=bf16.sample_standard_deviation_seconds,
            q3=q3.sample_standard_deviation_seconds,
        ),
    )


def build_measurement_performance_rollup(
    bf16: MeasurementSubjectPerformanceSummary,
    q3: MeasurementSubjectPerformanceSummary,
    *,
    llama_bench_workload_identity: MeasurementLlamaBenchWorkloadIdentity,
    server_workload_identity: MeasurementServerWorkloadIdentity,
    equivalent_trials_validated: Literal[True],
) -> MeasurementPerformanceRollup:
    """Build paired performance after the caller proves equivalent trial scope."""

    if equivalent_trials_validated is not True:
        raise ValueError("performance rollup requires validated equivalent trials")
    if bf16.subject != "bf16" or q3.subject != "q3":
        raise ValueError("performance summaries must be exact BF16 then Q3")
    if tuple(item.case for item in bf16.bench_cases) != tuple(item.case for item in q3.bench_cases):
        raise ValueError("performance summaries use different bench case order")
    if tuple(item.concurrency for item in bf16.server_cells) != tuple(
        item.concurrency for item in q3.server_cells
    ):
        raise ValueError("performance summaries use different server cell order")
    if (
        bf16.load_pair_repetitions != q3.load_pair_repetitions
        or bf16.workload_load_pair_trial_index != q3.workload_load_pair_trial_index
    ):
        raise ValueError("performance summaries use different load-pair trials")
    if bf16.load_pair_repetitions != 3 or bf16.workload_load_pair_trial_index != 3:
        raise ValueError("performance rollup requires three pairs and the final warm workload")
    if any(
        baseline.measured_requests != candidate.measured_requests
        for baseline, candidate in zip(
            bf16.server_cells,
            q3.server_cells,
            strict=True,
        )
    ):
        raise ValueError("performance summaries use different request counts")

    bench_cases = tuple(
        MeasurementBenchCaseRollup(
            case=baseline.case,
            repetitions_per_subject=baseline.sample_count,
            average_tokens_per_second=_paired_positive(
                baseline.average_tokens_per_second,
                candidate.average_tokens_per_second,
            ),
            median_tokens_per_second=_paired_positive(
                baseline.median_tokens_per_second,
                candidate.median_tokens_per_second,
            ),
            standard_deviation_tokens_per_second=(
                MeasurementPairedNonnegativeValue(
                    bf16=baseline.standard_deviation_tokens_per_second,
                    q3=candidate.standard_deviation_tokens_per_second,
                )
            ),
        )
        for baseline, candidate in zip(
            bf16.bench_cases,
            q3.bench_cases,
            strict=True,
        )
    )
    server_cells = tuple(
        MeasurementServerCellRollup(
            concurrency=baseline.concurrency,
            measured_batches_per_subject=baseline.measured_batches,
            measured_requests_per_subject=baseline.measured_requests,
            request_end_to_end_latency_seconds=_paired_five_batch_metric(
                baseline.batch_metrics.mean_request_end_to_end_latency_seconds,
                candidate.batch_metrics.mean_request_end_to_end_latency_seconds,
            ),
            ttft_seconds=_paired_five_batch_metric(
                baseline.batch_metrics.mean_ttft_seconds,
                candidate.batch_metrics.mean_ttft_seconds,
            ),
            prompt_tokens_per_second=_paired_five_batch_metric(
                baseline.batch_metrics.mean_prompt_tokens_per_second,
                candidate.batch_metrics.mean_prompt_tokens_per_second,
            ),
            decode_tokens_per_second=_paired_five_batch_metric(
                baseline.batch_metrics.mean_decode_tokens_per_second,
                candidate.batch_metrics.mean_decode_tokens_per_second,
            ),
            aggregate_decode_tokens_per_second=_paired_five_batch_metric(
                baseline.batch_metrics.aggregate_decode_tokens_per_second,
                candidate.batch_metrics.aggregate_decode_tokens_per_second,
            ),
            inter_token_latency_p50_seconds=_paired_five_batch_nonnegative_metric(
                baseline.batch_metrics.inter_token_latency_p50_seconds,
                candidate.batch_metrics.inter_token_latency_p50_seconds,
            ),
            inter_token_latency_p95_seconds=_paired_five_batch_nonnegative_metric(
                baseline.batch_metrics.inter_token_latency_p95_seconds,
                candidate.batch_metrics.inter_token_latency_p95_seconds,
            ),
            inter_token_latency_p99_seconds=_paired_five_batch_nonnegative_metric(
                baseline.batch_metrics.inter_token_latency_p99_seconds,
                candidate.batch_metrics.inter_token_latency_p99_seconds,
            ),
            bf16_resource_sample_count=baseline.resource_sample_summary.sample_count,
            q3_resource_sample_count=candidate.resource_sample_summary.sample_count,
            max_sampled_host_rss_bytes=MeasurementPairedBytes(
                bf16=baseline.resource_sample_summary.max_sampled_host_rss_bytes,
                q3=candidate.resource_sample_summary.max_sampled_host_rss_bytes,
            ),
            max_sampled_per_gpu_memory_bytes=MeasurementPairedGpuBytes(
                bf16=baseline.resource_sample_summary.max_sampled_per_gpu_memory_bytes,
                q3=candidate.resource_sample_summary.max_sampled_per_gpu_memory_bytes,
            ),
            max_sampled_per_gpu_utilization_percent=MeasurementPairedGpuUtilization(
                bf16=(baseline.resource_sample_summary.max_sampled_per_gpu_utilization_percent),
                q3=(candidate.resource_sample_summary.max_sampled_per_gpu_utilization_percent),
            ),
        )
        for baseline, candidate in zip(
            bf16.server_cells,
            q3.server_cells,
            strict=True,
        )
    )
    return MeasurementPerformanceRollup(
        llama_bench_workload_identity=llama_bench_workload_identity,
        server_workload_identity=server_workload_identity,
        text_checkpoint_size_bytes=MeasurementPairedBytes(
            bf16=bf16.text_checkpoint_size_bytes,
            q3=q3.text_checkpoint_size_bytes,
        ),
        multimodal_projector_size_bytes=MeasurementPairedBytes(
            bf16=bf16.multimodal_projector_size_bytes,
            q3=q3.multimodal_projector_size_bytes,
        ),
        executable_gguf_bundle_size_bytes=MeasurementPairedBytes(
            bf16=bf16.executable_gguf_bundle_size_bytes,
            q3=q3.executable_gguf_bundle_size_bytes,
        ),
        load_pair_repetitions_per_subject=3,
        workload_load_pair_trial_index=3,
        cold_server_load_trials=_paired_load_durations(
            bf16.cold_server_load_trials,
            q3.cold_server_load_trials,
        ),
        warm_server_load_trials=_paired_load_durations(
            bf16.warm_server_load_trials,
            q3.warm_server_load_trials,
        ),
        warm_load_protocol=(
            "second_same_artifact_process_after_cold_termination_without_requested_cache_conditioning_or_eviction"
        ),
        requested_telemetry_sampling_interval_seconds=1.0,
        bench_cases=cast(
            "tuple[MeasurementBenchCaseRollup, MeasurementBenchCaseRollup, "
            "MeasurementBenchCaseRollup]",
            bench_cases,
        ),
        server_cells=cast(
            "tuple[MeasurementServerCellRollup, MeasurementServerCellRollup, "
            "MeasurementServerCellRollup]",
            server_cells,
        ),
        single_request_warmups_per_subject=2,
        concurrent_batch_warmups_per_cell_per_subject=1,
        bench_cases_share_one_model_load=True,
        server_quality_and_performance_share_one_model_load=True,
        warmups_excluded_from_measurement=True,
        raw_trials_recorded=True,
        matched_runtime_hardware_workload=True,
        all_metrics_complete=True,
        equivalent_trials_valid=True,
        comparison_complete=True,
        speedup_claim_allowed=True,
    )


def build_measurement_placement_summaries(
    backend_audit: MeasurementBackendAuditEvidence,
    *,
    backend_audit_content_sha256: str,
    policy: ExactCudaPlacementPolicy,
) -> tuple[
    MeasurementPlacementSummary,
    MeasurementPlacementSummary,
    MeasurementPlacementSummary,
]:
    """Replay every retained full log through the exact CUDA audit parser."""

    _validate_sha256(
        backend_audit_content_sha256,
        label="backend audit content hash",
    )
    policy_sha256 = measurement_evidence_sha256(policy.model_dump(mode="json"))
    summaries: list[MeasurementPlacementSummary] = []
    for workload in backend_audit.workloads:
        parsed = parse_exact_cuda_backend_audit(workload.log, policy=policy)
        if not parsed.exact_cuda_identity_inventory:
            raise ValueError("backend audit CUDA identity inventory is incomplete")
        if not parsed.text_full_cell_observed:
            raise ValueError("backend audit did not observe the full configured CUDA cell")
        if not parsed.all_compute_operations_accelerated:
            raise ValueError("backend audit observed a non-CUDA compute operation")
        if not parsed.no_cpu_model_graph_fallback:
            raise ValueError("backend audit observed CPU model graph fallback")
        identities = tuple(
            MeasurementCudaIdentitySummary(
                ordinal=ordinal,
                backend_name=f"CUDA{ordinal}",
                device_name=f"CUDA{ordinal}",
                device_type="gpu",
                compute_operations=sum(
                    row.compute for row in parsed.identities if row.backend_index == ordinal
                ),
            )
            for ordinal in range(8)
        )
        summaries.append(
            MeasurementPlacementSummary(
                workload=workload.workload,
                backend_audit_content_sha256=backend_audit_content_sha256,
                audit_log_sha256=workload.log_sha256,
                command_sha256=measurement_evidence_sha256(list(workload.command)),
                placement_policy_sha256=policy_sha256,
                observed_graphs=parsed.observed_graphs,
                compute_operations=parsed.compute_operations,
                cuda_operations=parsed.gpu_operations,
                cpu_operations=parsed.cpu_operations,
                accelerator_operations=parsed.accelerator_operations,
                other_operations=parsed.other_operations,
                unassigned_operations=parsed.unassigned_operations,
                cuda_identities=cast(
                    "tuple[MeasurementCudaIdentitySummary, "
                    "MeasurementCudaIdentitySummary, "
                    "MeasurementCudaIdentitySummary, "
                    "MeasurementCudaIdentitySummary, "
                    "MeasurementCudaIdentitySummary, "
                    "MeasurementCudaIdentitySummary, "
                    "MeasurementCudaIdentitySummary, "
                    "MeasurementCudaIdentitySummary]",
                    identities,
                ),
                exact_cuda_identity_inventory=True,
                full_cell_text_observed=True,
                all_compute_operations_cuda=True,
                no_cpu_model_graph_fallback=True,
            )
        )
    return cast(
        "tuple[MeasurementPlacementSummary, MeasurementPlacementSummary, "
        "MeasurementPlacementSummary]",
        tuple(summaries),
    )


class MeasurementSubjectCompactRecord(_StrictEvidenceModel):
    """Compact, run-bound result for one exact BF16 or Q3 subject."""

    schema_version: Literal["inkling-measurement-subject-v1"] = "inkling-measurement-subject-v1"
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    subject: MeasurementEvidenceSubject
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    reviewed_config_file_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    resolved_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    post_spawn_acceptance_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    call_id: StrictStr = Field(pattern=_MODAL_CALL_ID_PATTERN)
    attempt_claim_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    runtime_manifest_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    hardware_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    model_id: Literal["thinkingmachines/Inkling"]
    model_revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"]
    artifact_inventory: tuple[MeasurementExecutableArtifactIdentity, ...] = Field(
        min_length=50,
        max_length=50,
    )
    protocol_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    workload_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    raw_blobs: tuple[
        MeasurementRawBlobReference,
        MeasurementRawBlobReference,
        MeasurementRawBlobReference,
        MeasurementRawBlobReference,
    ]
    placement_summaries: tuple[
        MeasurementPlacementSummary,
        MeasurementPlacementSummary,
        MeasurementPlacementSummary,
    ]
    quality_projection_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    performance_projection_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    prompt_text_recorded: Literal[False] = False
    output_text_recorded: Literal[False] = False

    @model_validator(mode="after")
    def exact_subject_evidence(self) -> MeasurementSubjectCompactRecord:
        observed_artifacts = tuple((item.ordinal, item.role) for item in self.artifact_inventory)
        expected_artifacts = (
            *((ordinal, "text_shard") for ordinal in range(49)),
            (49, "multimodal_projector"),
        )
        if observed_artifacts != expected_artifacts:
            raise ValueError("executable artifact inventory must be 49 shards then one projector")
        source_paths = tuple(item.source_path for item in self.artifact_inventory)
        staged_paths = tuple(item.staged_path for item in self.artifact_inventory)
        if len(set(source_paths)) != 50 or len(set(staged_paths)) != 50:
            raise ValueError("executable artifact paths must be unique")
        expected_names = (
            *tuple(
                (
                    f"inkling-BF16-{ordinal:05d}-of-00049.gguf"
                    if self.subject == "bf16"
                    else f"inkling-Q3_K_M-{ordinal:05d}-of-00049.gguf"
                )
                for ordinal in range(1, 50)
            ),
            "mmproj-BF16.gguf",
        )
        observed_source_names = tuple(PurePosixPath(path).name for path in source_paths)
        observed_staged_names = tuple(PurePosixPath(path).name for path in staged_paths)
        if observed_source_names != expected_names or observed_staged_names != expected_names:
            raise ValueError("executable artifact names differ from the exact model bundle")
        staged_prefix = f"/cache/inkling-measurement-subject/{self.subject}/"
        if any(not path.startswith(staged_prefix) for path in staged_paths):
            raise ValueError("executable artifact is outside its subject staging root")
        if tuple(item.kind for item in self.raw_blobs) != MEASUREMENT_RAW_BLOB_KIND_ORDER:
            raise ValueError("subject raw evidence is incomplete or out of order")
        if any(
            item.run_id != self.run_id or item.subject != self.subject for item in self.raw_blobs
        ):
            raise ValueError("subject raw evidence belongs to another run or subject")
        if (
            tuple(item.workload for item in self.placement_summaries)
            != MEASUREMENT_PLACEMENT_WORKLOAD_ORDER
        ):
            raise ValueError("subject placement evidence is incomplete or out of order")
        backend_audit = self.raw_blobs[-1]
        if any(
            item.backend_audit_content_sha256 != backend_audit.content_sha256
            for item in self.placement_summaries
        ):
            raise ValueError("placement summary differs from the full backend-audit blob")
        return self

    def canonical_bytes(self) -> bytes:
        """Return the exact canonical compact-record bytes."""

        return canonical_measurement_evidence_json_bytes(self.model_dump(mode="json"))

    def content_sha256(self) -> str:
        """Return the compact record's plain content SHA-256."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()


class MeasurementComparisonCompactRecord(_StrictEvidenceModel):
    """Compact paired result linking both subject records and all raw evidence."""

    schema_version: Literal["inkling-measurement-comparison-v1"] = (
        "inkling-measurement-comparison-v1"
    )
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    reviewed_config_file_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    resolved_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    post_spawn_acceptance_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    call_id: StrictStr = Field(pattern=_MODAL_CALL_ID_PATTERN)
    attempt_claim_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    runtime_manifest_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    hardware_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    model_id: Literal["thinkingmachines/Inkling"]
    model_revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"]
    protocol_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    workload_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    subject_records: tuple[
        MeasurementSupportingRecordReference,
        MeasurementSupportingRecordReference,
    ]
    raw_blobs: tuple[
        MeasurementRawBlobReference,
        MeasurementRawBlobReference,
        MeasurementRawBlobReference,
        MeasurementRawBlobReference,
        MeasurementRawBlobReference,
        MeasurementRawBlobReference,
        MeasurementRawBlobReference,
        MeasurementRawBlobReference,
    ]
    token_nll_pairing_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    diagnostic_pairing_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    performance_pairing_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    quality_rollup_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    performance_rollup_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    prompt_text_recorded: Literal[False] = False
    output_text_recorded: Literal[False] = False

    @model_validator(mode="after")
    def exact_paired_evidence(self) -> MeasurementComparisonCompactRecord:
        if tuple(item.kind for item in self.subject_records) != (
            "bf16_subject",
            "q3_subject",
        ):
            raise ValueError("comparison subject references are incomplete or out of order")
        if any(item.run_id != self.run_id for item in self.subject_records):
            raise ValueError("comparison subject reference belongs to another run")
        expected_raw_order = tuple(
            (subject, kind)
            for subject in ("bf16", "q3")
            for kind in MEASUREMENT_RAW_BLOB_KIND_ORDER
        )
        observed_raw_order = tuple((item.subject, item.kind) for item in self.raw_blobs)
        if observed_raw_order != expected_raw_order:
            raise ValueError("comparison raw references are incomplete or out of order")
        if any(item.run_id != self.run_id for item in self.raw_blobs):
            raise ValueError("comparison raw reference belongs to another run")
        return self

    def canonical_bytes(self) -> bytes:
        """Return the exact canonical compact-record bytes."""

        return canonical_measurement_evidence_json_bytes(self.model_dump(mode="json"))

    def content_sha256(self) -> str:
        """Return the compact record's plain content SHA-256."""

        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def parse_measurement_subject_compact_record(
    payload: bytes,
    *,
    run_id: str,
    subject: MeasurementEvidenceSubject,
) -> MeasurementSubjectCompactRecord:
    """Parse one canonical compact subject record with exact scope bindings."""

    value = _strict_canonical_json_value(payload)
    if not isinstance(value, dict):
        raise ValueError("compact subject record root must be an object")
    record = MeasurementSubjectCompactRecord.model_validate_json(payload, strict=True)
    if record.run_id != run_id or record.subject != subject:
        raise ValueError("compact subject record has the wrong run or subject")
    return record


def parse_measurement_comparison_compact_record(
    payload: bytes,
    *,
    run_id: str,
) -> MeasurementComparisonCompactRecord:
    """Parse one canonical compact comparison record with an exact run binding."""

    value = _strict_canonical_json_value(payload)
    if not isinstance(value, dict):
        raise ValueError("compact comparison record root must be an object")
    record = MeasurementComparisonCompactRecord.model_validate_json(payload, strict=True)
    if record.run_id != run_id:
        raise ValueError("compact comparison record has the wrong run")
    return record


def validate_measurement_comparison_links(
    comparison: MeasurementComparisonCompactRecord,
    *,
    bf16: MeasurementSubjectCompactRecord,
    q3: MeasurementSubjectCompactRecord,
) -> MeasurementComparisonCompactRecord:
    """Validate comparison links against the exact parsed subject records."""

    if bf16.subject != "bf16" or q3.subject != "q3":
        raise ValueError("comparison subjects must be exact BF16 then Q3")
    shared_fields = (
        "run_id",
        "control_plane_sha256",
        "reviewed_config_file_sha256",
        "resolved_config_sha256",
        "launch_intent_sha256",
        "post_spawn_acceptance_sha256",
        "call_id",
        "attempt_claim_sha256",
        "runtime_manifest_sha256",
        "hardware_identity_sha256",
        "model_id",
        "model_revision",
        "protocol_sha256",
        "workload_sha256",
    )
    for field_name in shared_fields:
        expected = getattr(comparison, field_name)
        if getattr(bf16, field_name) != expected or getattr(q3, field_name) != expected:
            raise ValueError(f"comparison {field_name} differs from a subject record")
    expected_subject_records = (
        build_measurement_supporting_record_reference(
            bf16.canonical_bytes(),
            run_id=comparison.run_id,
            kind="bf16_subject",
        ),
        build_measurement_supporting_record_reference(
            q3.canonical_bytes(),
            run_id=comparison.run_id,
            kind="q3_subject",
        ),
    )
    if comparison.subject_records != expected_subject_records:
        raise ValueError("comparison subject references differ from exact compact records")
    if comparison.raw_blobs != (*bf16.raw_blobs, *q3.raw_blobs):
        raise ValueError("comparison raw references differ from exact subject records")
    return comparison


__all__ = [
    "MEASUREMENT_CUDA_IDENTITY_ORDER",
    "MEASUREMENT_PLACEMENT_WORKLOAD_ORDER",
    "MEASUREMENT_RAW_BLOB_KIND_ORDER",
    "MEASUREMENT_RAW_BLOB_MAX_BYTES",
    "MEASUREMENT_RAW_BLOB_RECORD_COUNTS",
    "MEASUREMENT_SUBJECT_PERFORMANCE_PROJECTION_HASH_DOMAIN",
    "MEASUREMENT_SUBJECT_QUALITY_PROJECTION_HASH_DOMAIN",
    "MeasurementComparisonCompactRecord",
    "MeasurementCudaIdentitySummary",
    "MeasurementEvidenceSubject",
    "MeasurementExecutableArtifactIdentity",
    "MeasurementPlacementSummary",
    "MeasurementPlacementWorkload",
    "MeasurementRawBlobFormat",
    "MeasurementRawBlobKind",
    "MeasurementRawBlobReference",
    "MeasurementSubjectCompactRecord",
    "build_measurement_performance_rollup",
    "build_measurement_placement_summaries",
    "build_measurement_quality_rollup",
    "build_measurement_raw_blob_reference",
    "canonical_measurement_evidence_json_bytes",
    "measurement_evidence_sha256",
    "measurement_raw_blob_path",
    "measurement_subject_performance_projection_sha256",
    "measurement_subject_quality_projection_sha256",
    "parse_measurement_comparison_compact_record",
    "parse_measurement_subject_compact_record",
    "validate_measurement_comparison_links",
    "validate_measurement_raw_blob_reference",
]
