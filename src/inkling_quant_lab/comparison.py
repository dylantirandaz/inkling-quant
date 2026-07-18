"""Strict run compatibility, metric deltas, and tolerance-aware Pareto analysis."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Set
from typing import Literal, TypeAlias, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from inkling_quant_lab.exceptions import ComparisonCompatibilityError

MetricCategory: TypeAlias = Literal["quality", "resource", "routing", "behavioral", "other"]
MetricDirection: TypeAlias = Literal["maximize", "minimize", "neutral"]
MetricStatus: TypeAlias = Literal["available", "unavailable", "unsupported", "failed"]
ParetoDirection: TypeAlias = Literal["maximize", "minimize"]


class ImmutableComparisonModel(BaseModel):
    """Strict immutable base for comparison artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DatasetIdentity(ImmutableComparisonModel):
    """Dataset provenance participating in a comparison contract."""

    dataset_id: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    split: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class BenchmarkWorkloadIdentity(ImmutableComparisonModel):
    """Exact benchmark workload participating in cross-run compatibility."""

    dataset_id: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: str = Field(min_length=1)
    sample_ids: tuple[str, ...]
    seed: int = Field(ge=0)
    prompt_template_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decode_config: dict[str, JsonValue]
    execution_mode: str = Field(min_length=1)

    @field_validator("sample_ids")
    @classmethod
    def sample_ids_are_exact_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not sample_id for sample_id in value):
            raise ValueError("benchmark workload requires non-empty stable sample IDs")
        if len(set(value)) != len(value):
            raise ValueError("benchmark workload sample IDs must be unique")
        return value


class BenchmarkMemoryIdentity(ImmutableComparisonModel):
    """Memory collector kinds and scopes retained for report interpretation."""

    host_measurement_kind: str | None = None
    host_scope: str | None = None
    host_process_isolated: bool = False
    device_measurement_kind: str | None = None
    device_scope: str | None = None

    @model_validator(mode="after")
    def kind_and_scope_are_paired(self) -> BenchmarkMemoryIdentity:
        if (self.host_measurement_kind is None) != (self.host_scope is None):
            raise ValueError("host memory collector kind and scope must be paired")
        if (self.device_measurement_kind is None) != (self.device_scope is None):
            raise ValueError("device memory collector kind and scope must be paired")
        if self.host_measurement_kind is None and self.device_measurement_kind is None:
            raise ValueError("benchmark memory identity requires an available collector")
        if self.host_process_isolated and self.host_measurement_kind is None:
            raise ValueError("isolated host memory identity requires a host collector")
        return self


class EvaluationSuiteIdentity(ImmutableComparisonModel):
    """Exact evaluator-scoped provenance used by the comparison contract."""

    evaluator_name: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    split: str = Field(min_length=1)
    dataset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sample_ids: tuple[str, ...]
    seed: int = Field(ge=0)
    prompt_template_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decode_config: dict[str, JsonValue]
    status: Literal["success", "partial", "unsupported", "failed"]

    @model_validator(mode="after")
    def validate_evidence(self) -> EvaluationSuiteIdentity:
        """Measured suites require bytes; setup failures retain an honest null digest."""

        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("evaluation suite sample IDs must be unique")
        if any(not sample_id for sample_id in self.sample_ids):
            raise ValueError("evaluation suite sample IDs must be non-empty strings")
        if self.status != "failed" and self.dataset_sha256 is None:
            raise ValueError("non-failed suites require an exact dataset SHA-256")
        return self

    def canonical_json(self) -> str:
        """Return a deterministic key without flattening evaluator associations."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def scientific_json(self) -> str:
        """Return the compared scientific identity without outcome status."""

        return json.dumps(
            self.model_dump(mode="json", exclude={"status"}),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


class RoutingCaptureIdentity(ImmutableComparisonModel):
    """Persisted evidence describing the routing decisions used by a summary."""

    configured_mode: Literal["off", "aggregate", "sampled_tokens", "full_trace"]
    captured_mode: Literal["off", "aggregate", "sampled_tokens", "full_trace"]
    observed_event_count: int = Field(ge=0)
    recorded_event_count: int = Field(ge=0)
    alignment_key_count: int = Field(ge=0)
    alignment_key_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_trace_evidence(self) -> RoutingCaptureIdentity:
        """Never claim token alignment without retained, digestible trace keys."""

        if self.recorded_event_count > self.observed_event_count:
            raise ValueError("recorded routing events cannot exceed observed events")
        if self.alignment_key_count != self.recorded_event_count:
            raise ValueError("routing alignment evidence count must equal retained trace count")
        has_alignment_digest = self.alignment_key_sha256 is not None
        if has_alignment_digest != (self.alignment_key_count > 0):
            raise ValueError(
                "routing alignment evidence requires both a positive key count and SHA-256"
            )
        if self.captured_mode in {"off", "aggregate"} and self.recorded_event_count:
            raise ValueError(f"{self.captured_mode} routing capture cannot retain raw traces")
        if self.captured_mode == "off" and self.observed_event_count:
            raise ValueError("off routing capture cannot contain observed events")
        return self


class MetricValue(ImmutableComparisonModel):
    """One normalized measurement with explicit availability semantics."""

    value: float | None = None
    status: MetricStatus = "available"
    unit: str | None = None
    category: MetricCategory = "other"
    direction: MetricDirection = "neutral"
    reason: str | None = None
    evaluation_suite: EvaluationSuiteIdentity | None = None

    @model_validator(mode="after")
    def validate_availability(self) -> MetricValue:
        """Keep unavailable values distinct from valid numeric zero."""

        if self.status == "available":
            if self.value is None:
                raise ValueError("available metric requires a numeric value")
        else:
            if self.value is not None:
                raise ValueError("unavailable, unsupported, and failed metrics cannot have a value")
            if not self.reason:
                raise ValueError(f"{self.status} metric requires a reason")
        if (
            self.evaluation_suite is not None
            and self.status == "available"
            and self.evaluation_suite.status not in {"success", "partial"}
        ):
            raise ValueError(
                "available evaluation metric requires successful or partial suite evidence"
            )
        return self


class NormalizedRunSummary(ImmutableComparisonModel):
    """Model-independent, report-ready summary of one immutable run."""

    schema_version: Literal["1.0", "1.1"] = "1.0"
    run_id: str = Field(min_length=1)
    artifact_path: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str | None
    datasets: tuple[DatasetIdentity, ...]
    seed_set: tuple[int, ...]
    sample_ids: tuple[str, ...]
    prompt_template_hash: str = Field(min_length=1)
    decode_config: dict[str, JsonValue]
    evaluation_suites: tuple[EvaluationSuiteIdentity, ...] = ()
    routing_dataset: DatasetIdentity | None = None
    routing_sample_ids: tuple[str, ...] = ()
    routing_capture: RoutingCaptureIdentity | None = None
    benchmark_protocol_version: str = Field(min_length=1)
    benchmark_workload: BenchmarkWorkloadIdentity | None = None
    benchmark_memory: BenchmarkMemoryIdentity | None = None
    benchmark_model_load_time_kind: str | None = None
    benchmark_source_weight_free_load_provenance: dict[str, JsonValue] | None = None
    hardware_environment: dict[str, JsonValue]
    metrics: dict[str, MetricValue]
    environment: dict[str, JsonValue] = Field(default_factory=dict)
    resolved_config: dict[str, JsonValue] = Field(default_factory=dict)
    quantization_policy: dict[str, JsonValue] = Field(default_factory=dict)
    failures: tuple[str, ...] = ()
    unsupported_measurements: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @field_validator("sample_ids")
    @classmethod
    def sample_ids_are_stable_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject ambiguous sample contracts."""

        if any(not sample_id for sample_id in value):
            raise ValueError("normalized summary sample IDs must be non-empty strings")
        if len(set(value)) != len(value):
            raise ValueError("sample IDs must be unique")
        return value

    @field_validator("seed_set")
    @classmethod
    def seeds_are_a_canonical_nonempty_set(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require the exact, deterministic seed set used by evaluation."""

        if not value:
            raise ValueError("normalized summary requires a non-empty seed set")
        if any(seed < 0 for seed in value):
            raise ValueError("seed set values must be non-negative")
        if tuple(sorted(set(value))) != value:
            raise ValueError("seed set must be sorted and unique")
        return value

    @field_validator("routing_sample_ids")
    @classmethod
    def routing_samples_are_stable_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject ambiguous routing sample alignment contracts."""

        if any(not sample_id for sample_id in value):
            raise ValueError("routing sample IDs must be non-empty")
        if len(set(value)) != len(value):
            raise ValueError("routing sample IDs must be unique")
        return value

    @field_validator("hardware_environment")
    @classmethod
    def hardware_contract_is_complete(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        """Require actual hardware plus the runtime placement contract."""

        if not value or "hardware" not in value or "runtime" not in value:
            raise ValueError("hardware_environment requires hardware and runtime provenance")
        return value

    @field_validator("datasets")
    @classmethod
    def datasets_are_present(
        cls, value: tuple[DatasetIdentity, ...]
    ) -> tuple[DatasetIdentity, ...]:
        """Require evaluation provenance for comparable run summaries."""

        identities = {(item.dataset_id, item.dataset_revision, item.split) for item in value}
        if len(identities) != len(value):
            raise ValueError("dataset identities must be unique")
        return value

    @field_validator("evaluation_suites")
    @classmethod
    def suites_are_canonical(
        cls, value: tuple[EvaluationSuiteIdentity, ...]
    ) -> tuple[EvaluationSuiteIdentity, ...]:
        """Require byte-stable full-suite ordering while preserving duplicate suites."""

        if tuple(sorted(value, key=lambda suite: suite.canonical_json())) != value:
            raise ValueError("evaluation suite identities must be canonically sorted")
        return value

    @field_validator("metrics")
    @classmethod
    def metric_names_are_valid(cls, value: dict[str, MetricValue]) -> dict[str, MetricValue]:
        """Require stable non-empty metric names."""

        if not value:
            raise ValueError("normalized summary requires at least one metric")
        if any(not name or name.strip() != name for name in value):
            raise ValueError(
                "metric names must be non-empty and cannot have surrounding whitespace"
            )
        return value

    @model_validator(mode="after")
    def validate_evaluation_provenance(self) -> NormalizedRunSummary:
        """Allow empty aggregates only for truthfully retained optional setup failures."""

        if self.schema_version == "1.0":
            if not self.datasets:
                raise ValueError("normalized summary requires at least one dataset identity")
            if not self.sample_ids:
                raise ValueError("normalized summary requires non-empty stable sample IDs")
            return self
        if not self.evaluation_suites:
            raise ValueError("summary schema 1.1 requires evaluator-scoped suite identities")
        measured = tuple(
            suite for suite in self.evaluation_suites if suite.status in {"success", "partial"}
        )
        if measured and not self.datasets:
            raise ValueError("measured evaluation suites require exact dataset identities")
        if measured and not self.sample_ids:
            raise ValueError("measured evaluation suites require stable sample IDs")
        suite_records = {suite.canonical_json() for suite in self.evaluation_suites}
        evaluation_aliases = {
            "quality",
            "mean_nll",
            "perplexity",
            "generation_exact_match",
            "behavioral_score",
            "behavioral_base_score",
            "behavioral_fine_tuned_score",
            "behavioral_retention",
            "multimodal_contract",
        }
        for name, metric in self.metrics.items():
            suite = metric.evaluation_suite
            if (name.startswith("evaluation.") or name in evaluation_aliases) and suite is None:
                raise ValueError(f"evaluation metric {name!r} requires suite provenance")
            if suite is not None and suite.canonical_json() not in suite_records:
                raise ValueError(
                    f"metric {name!r} references evaluation suite evidence absent from summary"
                )
        return self

    @model_validator(mode="after")
    def validate_routing_provenance(self) -> NormalizedRunSummary:
        """Keep routing data identity, samples, and capture evidence coupled."""

        if self.routing_dataset is None:
            if self.routing_sample_ids or self.routing_capture is not None:
                raise ValueError(
                    "routing samples or capture evidence require a routing dataset identity"
                )
        elif not self.routing_sample_ids:
            raise ValueError("routing dataset identity requires stable routing sample IDs")
        aligned_metrics = (
            self.metrics.get("route_agreement"),
            self.metrics.get("top_k_overlap"),
        )
        if any(
            metric is not None and metric.status == "available" for metric in aligned_metrics
        ) and (
            self.routing_capture is None
            or self.routing_capture.alignment_key_count == 0
            or self.routing_capture.alignment_key_sha256 is None
        ):
            raise ValueError(
                "available token routing agreement requires retained alignment evidence"
            )
        return self

    def canonical_json(self) -> str:
        """Return deterministic machine-readable summary JSON."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


class ComparisonContract(ImmutableComparisonModel):
    """Compatibility dimensions required before computing run deltas."""

    require_same_model_id: bool = True
    require_same_model_revision: bool = True
    require_same_datasets: bool = True
    require_same_seed_set: bool = True
    require_same_samples: bool = True
    require_same_prompt_template: bool = True
    require_same_decode_config: bool = True
    require_same_routing_dataset: bool = True
    require_same_routing_samples: bool = True
    require_same_routing_capture: bool = True
    require_same_benchmark_memory: bool = True
    require_same_benchmark_protocol: bool = True
    require_same_benchmark_workload: bool = True
    require_same_hardware_environment: bool = True


class CompatibilityMismatch(ImmutableComparisonModel):
    """One human- and machine-readable contract mismatch."""

    dimension: str
    baseline: JsonValue
    candidate: JsonValue
    message: str


class MetricDelta(ImmutableComparisonModel):
    """Candidate-minus-baseline metric change without availability coercion."""

    metric: str
    status: Literal["available", "unavailable"]
    category: MetricCategory
    direction: MetricDirection
    unit: str | None = None
    baseline_value: float | None = None
    candidate_value: float | None = None
    absolute_delta: float | None = None
    relative_delta: float | None = None
    relative_delta_reason: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_delta_state(self) -> MetricDelta:
        """Require numeric evidence only for available deltas."""

        if self.status == "available":
            if (
                self.baseline_value is None
                or self.candidate_value is None
                or self.absolute_delta is None
            ):
                raise ValueError(
                    "available delta requires baseline, candidate, and absolute values"
                )
            if self.reason is not None:
                raise ValueError("available delta cannot have an unavailability reason")
        elif self.absolute_delta is not None or self.relative_delta is not None:
            raise ValueError("unavailable delta cannot contain numeric deltas")
        return self


class ComparisonResult(ImmutableComparisonModel):
    """Normalized, serializable result of comparing two runs."""

    schema_version: Literal["1.0"] = "1.0"
    baseline_run_id: str
    candidate_run_id: str
    contract: ComparisonContract
    contract_compatible: bool
    unsafe_override: bool
    overridden_dimensions: tuple[str, ...] = ()
    mismatches: tuple[CompatibilityMismatch, ...] = ()
    warnings: tuple[str, ...] = ()
    deltas: tuple[MetricDelta, ...]

    def delta_for(self, metric: str) -> MetricDelta:
        """Return a named delta or raise a precise key error."""

        for delta in self.deltas:
            if delta.metric == metric:
                return delta
        raise KeyError(metric)


def _legacy_dataset_contract(summary: NormalizedRunSummary) -> list[dict[str, JsonValue]]:
    return [
        item.model_dump(mode="json")
        for item in sorted(
            summary.datasets,
            key=lambda value: (value.dataset_id, value.dataset_revision, value.split),
        )
    ]


def _suite_anchor(suite: EvaluationSuiteIdentity) -> dict[str, JsonValue]:
    """Identify one suite without folding its compared dimension into the anchor."""

    return {
        "evaluator_name": suite.evaluator_name,
        "evaluator_version": suite.evaluator_version,
    }


def _evaluation_seed_contract(summary: NormalizedRunSummary) -> JsonValue:
    if not summary.evaluation_suites:
        return cast(JsonValue, list(summary.seed_set))
    return cast(
        JsonValue,
        [
            {
                **_suite_anchor(suite),
                "dataset_id": suite.dataset_id,
                "dataset_revision": suite.dataset_revision,
                "split": suite.split,
                "seed": suite.seed,
            }
            for suite in summary.evaluation_suites
        ],
    )


def _evaluation_dataset_contract(summary: NormalizedRunSummary) -> JsonValue:
    if not summary.evaluation_suites:
        return cast(JsonValue, _legacy_dataset_contract(summary))
    return cast(
        JsonValue,
        [
            {
                **_suite_anchor(suite),
                "dataset_id": suite.dataset_id,
                "dataset_revision": suite.dataset_revision,
                "split": suite.split,
                "dataset_sha256": suite.dataset_sha256,
            }
            for suite in summary.evaluation_suites
        ],
    )


def _evaluation_sample_contract(summary: NormalizedRunSummary) -> JsonValue:
    if not summary.evaluation_suites:
        return cast(JsonValue, sorted(summary.sample_ids))
    return cast(
        JsonValue,
        [
            {
                **_suite_anchor(suite),
                "dataset_id": suite.dataset_id,
                "dataset_revision": suite.dataset_revision,
                "split": suite.split,
                "sample_ids": list(suite.sample_ids),
            }
            for suite in summary.evaluation_suites
        ],
    )


def _evaluation_prompt_contract(summary: NormalizedRunSummary) -> JsonValue:
    if not summary.evaluation_suites:
        return summary.prompt_template_hash
    return cast(
        JsonValue,
        [
            {
                **_suite_anchor(suite),
                "dataset_id": suite.dataset_id,
                "dataset_revision": suite.dataset_revision,
                "split": suite.split,
                "prompt_template_hash": suite.prompt_template_hash,
            }
            for suite in summary.evaluation_suites
        ],
    )


def _evaluation_decode_contract(summary: NormalizedRunSummary) -> JsonValue:
    if not summary.evaluation_suites:
        return cast(JsonValue, summary.decode_config)
    return cast(
        JsonValue,
        [
            {
                **_suite_anchor(suite),
                "dataset_id": suite.dataset_id,
                "dataset_revision": suite.dataset_revision,
                "split": suite.split,
                "decode_config": suite.decode_config,
            }
            for suite in summary.evaluation_suites
        ],
    )


def compatibility_mismatches(
    baseline: NormalizedRunSummary,
    candidate: NormalizedRunSummary,
    contract: ComparisonContract | None = None,
) -> tuple[CompatibilityMismatch, ...]:
    """Return all contract mismatches in a stable dimension order."""

    active = contract or ComparisonContract()
    checks: tuple[tuple[bool, str, JsonValue, JsonValue, str], ...] = (
        (
            active.require_same_model_id,
            "model_id",
            baseline.model_id,
            candidate.model_id,
            "base model identifiers differ",
        ),
        (
            active.require_same_model_revision,
            "model_revision",
            baseline.model_revision,
            candidate.model_revision,
            "model revisions differ",
        ),
        (
            active.require_same_datasets,
            "datasets",
            _evaluation_dataset_contract(baseline),
            _evaluation_dataset_contract(candidate),
            "evaluation dataset identities, revisions, splits, or exact SHA-256 digests differ",
        ),
        (
            active.require_same_seed_set,
            "seed_set",
            _evaluation_seed_contract(baseline),
            _evaluation_seed_contract(candidate),
            "evaluation seed sets differ",
        ),
        (
            active.require_same_prompt_template,
            "prompt_template_hash",
            _evaluation_prompt_contract(baseline),
            _evaluation_prompt_contract(candidate),
            "prompt templates differ",
        ),
        (
            active.require_same_decode_config,
            "decode_config",
            _evaluation_decode_contract(baseline),
            _evaluation_decode_contract(candidate),
            "decode configurations differ",
        ),
        (
            active.require_same_samples,
            "sample_ids",
            _evaluation_sample_contract(baseline),
            _evaluation_sample_contract(candidate),
            "stable evaluation sample IDs differ",
        ),
        (
            active.require_same_routing_dataset,
            "routing_dataset",
            cast(
                JsonValue,
                baseline.routing_dataset.model_dump(mode="json")
                if baseline.routing_dataset is not None
                else None,
            ),
            cast(
                JsonValue,
                candidate.routing_dataset.model_dump(mode="json")
                if candidate.routing_dataset is not None
                else None,
            ),
            "routing dataset identity, revision, split, or exact SHA-256 digest differs",
        ),
        (
            active.require_same_routing_samples,
            "routing_sample_ids",
            cast(JsonValue, list(baseline.routing_sample_ids)),
            cast(JsonValue, list(candidate.routing_sample_ids)),
            "stable routing sample alignment differs",
        ),
        (
            active.require_same_routing_capture,
            "routing_capture",
            cast(
                JsonValue,
                baseline.routing_capture.model_dump(mode="json")
                if baseline.routing_capture is not None
                else None,
            ),
            cast(
                JsonValue,
                candidate.routing_capture.model_dump(mode="json")
                if candidate.routing_capture is not None
                else None,
            ),
            "routing capture modes or retained token-alignment evidence differ",
        ),
        (
            active.require_same_benchmark_memory,
            "benchmark_memory",
            cast(
                JsonValue,
                baseline.benchmark_memory.model_dump(mode="json")
                if baseline.benchmark_memory is not None
                else None,
            ),
            cast(
                JsonValue,
                candidate.benchmark_memory.model_dump(mode="json")
                if candidate.benchmark_memory is not None
                else None,
            ),
            "benchmark memory collector kinds or measurement scopes differ",
        ),
        (
            active.require_same_benchmark_workload,
            "benchmark_workload",
            cast(
                JsonValue,
                baseline.benchmark_workload.model_dump(mode="json")
                if baseline.benchmark_workload is not None
                else None,
            ),
            cast(
                JsonValue,
                candidate.benchmark_workload.model_dump(mode="json")
                if candidate.benchmark_workload is not None
                else None,
            ),
            "exact benchmark dataset, samples, seed, prompt, decode, or execution mode differ",
        ),
        (
            active.require_same_benchmark_protocol,
            "benchmark_protocol_version",
            baseline.benchmark_protocol_version,
            candidate.benchmark_protocol_version,
            "benchmark protocol versions differ",
        ),
        (
            active.require_same_hardware_environment,
            "hardware_environment",
            cast(JsonValue, baseline.hardware_environment),
            cast(JsonValue, candidate.hardware_environment),
            "hardware or runtime placement environments differ",
        ),
    )
    return tuple(
        CompatibilityMismatch(
            dimension=dimension,
            baseline=baseline_value,
            candidate=candidate_value,
            message=message,
        )
        for required, dimension, baseline_value, candidate_value, message in checks
        if required and baseline_value != candidate_value
    )


_OVERRIDE_ALIASES = {
    "benchmark_memory_collector": "benchmark_memory",
    "benchmark_samples": "benchmark_workload",
    "benchmark_protocol": "benchmark_protocol_version",
    "evaluation_datasets": "datasets",
    "hardware": "hardware_environment",
    "prompt_template": "prompt_template_hash",
    "routing_mode": "routing_capture",
    "routing_samples": "routing_sample_ids",
    "seed": "seed_set",
    "seeds": "seed_set",
    "samples": "sample_ids",
}
_OVERRIDABLE_DIMENSIONS = {
    "model_id",
    "model_revision",
    "datasets",
    "seed_set",
    "prompt_template_hash",
    "decode_config",
    "sample_ids",
    "routing_dataset",
    "routing_sample_ids",
    "routing_capture",
    "benchmark_memory",
    "benchmark_workload",
    "benchmark_protocol_version",
    "hardware_environment",
}


def _normalize_overrides(values: Set[str] | None) -> set[str]:
    normalized = {_OVERRIDE_ALIASES.get(value, value) for value in values or set()}
    unknown = normalized.difference(_OVERRIDABLE_DIMENSIONS)
    if unknown:
        raise ValueError(
            "unknown unsafe comparison override dimensions: " + ", ".join(sorted(unknown))
        )
    return normalized


def _unavailable_delta(
    metric: str,
    baseline: MetricValue | None,
    candidate: MetricValue | None,
) -> MetricDelta:
    reasons: list[str] = []
    if baseline is None:
        reasons.append("baseline metric is missing")
    elif baseline.status != "available":
        reasons.append(f"baseline metric is {baseline.status}: {baseline.reason}")
    if candidate is None:
        reasons.append("candidate metric is missing")
    elif candidate.status != "available":
        reasons.append(f"candidate metric is {candidate.status}: {candidate.reason}")
    exemplar = candidate or baseline
    assert exemplar is not None
    return MetricDelta(
        metric=metric,
        status="unavailable",
        category=exemplar.category,
        direction=exemplar.direction,
        unit=exemplar.unit,
        baseline_value=baseline.value if baseline is not None else None,
        candidate_value=candidate.value if candidate is not None else None,
        reason="; ".join(reasons),
    )


def compute_metric_deltas(
    baseline: Mapping[str, MetricValue], candidate: Mapping[str, MetricValue]
) -> tuple[MetricDelta, ...]:
    """Compute stable candidate-minus-baseline absolute and relative changes."""

    deltas: list[MetricDelta] = []
    for metric in sorted(set(baseline).union(candidate)):
        baseline_metric = baseline.get(metric)
        candidate_metric = candidate.get(metric)
        if (
            baseline_metric is None
            or candidate_metric is None
            or baseline_metric.status != "available"
            or candidate_metric.status != "available"
        ):
            deltas.append(_unavailable_delta(metric, baseline_metric, candidate_metric))
            continue
        baseline_source = (
            baseline_metric.evaluation_suite.scientific_json()
            if baseline_metric.evaluation_suite is not None
            else None
        )
        candidate_source = (
            candidate_metric.evaluation_suite.scientific_json()
            if candidate_metric.evaluation_suite is not None
            else None
        )
        if baseline_source != candidate_source:
            deltas.append(
                MetricDelta(
                    metric=metric,
                    status="unavailable",
                    category=candidate_metric.category,
                    direction=candidate_metric.direction,
                    unit=candidate_metric.unit,
                    baseline_value=baseline_metric.value,
                    candidate_value=candidate_metric.value,
                    reason="evaluation metric suite provenance differs",
                )
            )
            continue
        if baseline_metric.unit != candidate_metric.unit:
            deltas.append(
                MetricDelta(
                    metric=metric,
                    status="unavailable",
                    category=candidate_metric.category,
                    direction=candidate_metric.direction,
                    unit=candidate_metric.unit,
                    baseline_value=baseline_metric.value,
                    candidate_value=candidate_metric.value,
                    reason=(
                        "metric units differ: "
                        f"baseline={baseline_metric.unit!r}, candidate={candidate_metric.unit!r}"
                    ),
                )
            )
            continue
        if baseline_metric.direction != candidate_metric.direction:
            deltas.append(
                MetricDelta(
                    metric=metric,
                    status="unavailable",
                    category=candidate_metric.category,
                    direction=candidate_metric.direction,
                    unit=candidate_metric.unit,
                    baseline_value=baseline_metric.value,
                    candidate_value=candidate_metric.value,
                    reason=(
                        "metric objective directions differ: "
                        f"baseline={baseline_metric.direction}, "
                        f"candidate={candidate_metric.direction}"
                    ),
                )
            )
            continue
        assert baseline_metric.value is not None
        assert candidate_metric.value is not None
        absolute = candidate_metric.value - baseline_metric.value
        if baseline_metric.value == 0.0:
            relative = None
            relative_reason = "baseline value is zero"
        else:
            relative = absolute / abs(baseline_metric.value)
            relative_reason = None
        deltas.append(
            MetricDelta(
                metric=metric,
                status="available",
                category=candidate_metric.category,
                direction=candidate_metric.direction,
                unit=candidate_metric.unit,
                baseline_value=baseline_metric.value,
                candidate_value=candidate_metric.value,
                absolute_delta=absolute,
                relative_delta=relative,
                relative_delta_reason=relative_reason,
            )
        )
    return tuple(deltas)


def compare_summaries(
    baseline: NormalizedRunSummary,
    candidate: NormalizedRunSummary,
    *,
    contract: ComparisonContract | None = None,
    unsafe: bool = False,
    unsafe_overrides: Set[str] | None = None,
) -> ComparisonResult:
    """Validate compatibility and compute deltas, requiring explicit unsafe overrides."""

    active_contract = contract or ComparisonContract()
    mismatches = compatibility_mismatches(baseline, candidate, active_contract)
    selected_overrides = _normalize_overrides(unsafe_overrides)
    overridden = {
        mismatch.dimension
        for mismatch in mismatches
        if unsafe or mismatch.dimension in selected_overrides
    }
    unresolved = tuple(mismatch for mismatch in mismatches if mismatch.dimension not in overridden)
    if unresolved:
        dimensions = ", ".join(mismatch.dimension for mismatch in unresolved)
        raise ComparisonCompatibilityError(
            f"Runs are incompatible; mismatches: {dimensions}",
            component="comparison",
            remediation=(
                "Use an explicit unsafe override only when the resulting comparison is intended"
            ),
            details={"mismatches": [mismatch.model_dump(mode="json") for mismatch in unresolved]},
        )
    overridden_dimensions = tuple(sorted(overridden))
    warnings: tuple[str, ...] = ()
    if overridden_dimensions:
        warnings = (
            "UNSAFE COMPARISON: compatibility mismatches were explicitly overridden for "
            + ", ".join(overridden_dimensions),
        )
    deltas = compute_metric_deltas(baseline.metrics, candidate.metrics)
    if baseline.benchmark_model_load_time_kind != candidate.benchmark_model_load_time_kind:
        baseline_load = baseline.metrics.get("load_time_ms")
        candidate_load = candidate.metrics.get("load_time_ms")
        deltas = tuple(
            MetricDelta(
                metric="load_time_ms",
                status="unavailable",
                category="resource",
                direction="minimize",
                unit="ms",
                baseline_value=baseline_load.value if baseline_load is not None else None,
                candidate_value=candidate_load.value if candidate_load is not None else None,
                reason=(
                    "model-load operations differ: "
                    f"baseline={baseline.benchmark_model_load_time_kind!r}, "
                    f"candidate={candidate.benchmark_model_load_time_kind!r}"
                ),
            )
            if delta.metric == "load_time_ms"
            else delta
            for delta in deltas
        )
    return ComparisonResult(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        contract=active_contract,
        contract_compatible=not mismatches,
        unsafe_override=bool(overridden_dimensions),
        overridden_dimensions=overridden_dimensions,
        mismatches=mismatches,
        warnings=warnings,
        deltas=deltas,
    )


class ParetoObjective(ImmutableComparisonModel):
    """One named Pareto objective with direction and comparison tolerance."""

    metric: str = Field(min_length=1)
    direction: ParetoDirection
    tolerance: float = Field(default=0.0, ge=0.0)
    relative_tolerance: float = Field(default=0.0, ge=0.0)


class ParetoPoint(ImmutableComparisonModel):
    """One candidate's complete finite objective vector."""

    point_id: str = Field(min_length=1)
    values: dict[str, float]

    @field_validator("values")
    @classmethod
    def values_are_finite(cls, value: dict[str, float]) -> dict[str, float]:
        """Reject values that cannot participate in a meaningful frontier."""

        if not value:
            raise ValueError("Pareto point requires at least one value")
        non_finite = sorted(name for name, number in value.items() if not math.isfinite(number))
        if non_finite:
            raise ValueError("Pareto values must be finite: " + ", ".join(non_finite))
        return value


class ParetoMembership(ImmutableComparisonModel):
    """Dominance result for one candidate."""

    point_id: str
    pareto_optimal: bool
    dominated_by: tuple[str, ...] = ()


class ParetoResult(ImmutableComparisonModel):
    """Deterministically ordered Pareto memberships and objective definitions."""

    objectives: tuple[ParetoObjective, ...]
    memberships: tuple[ParetoMembership, ...]

    @property
    def optimal_ids(self) -> tuple[str, ...]:
        """Return stable identifiers for non-dominated points."""

        return tuple(item.point_id for item in self.memberships if item.pareto_optimal)

    def membership_for(self, point_id: str) -> ParetoMembership:
        """Return membership for one point."""

        for membership in self.memberships:
            if membership.point_id == point_id:
                return membership
        raise KeyError(point_id)


def _effective_tolerance(left: float, right: float, objective: ParetoObjective) -> float:
    return objective.tolerance + objective.relative_tolerance * max(abs(left), abs(right))


def dominates(
    candidate: ParetoPoint,
    reference: ParetoPoint,
    objectives: tuple[ParetoObjective, ...],
) -> bool:
    """Return whether candidate is no worse everywhere and materially better somewhere."""

    strictly_better = False
    for objective in objectives:
        try:
            candidate_value = candidate.values[objective.metric]
            reference_value = reference.values[objective.metric]
        except KeyError as error:
            point = candidate if objective.metric not in candidate.values else reference
            raise ValueError(
                f"Pareto point {point.point_id!r} is missing objective {objective.metric!r}"
            ) from error
        tolerance = _effective_tolerance(candidate_value, reference_value, objective)
        if objective.direction == "maximize":
            if candidate_value < reference_value - tolerance:
                return False
            if candidate_value > reference_value + tolerance:
                strictly_better = True
        else:
            if candidate_value > reference_value + tolerance:
                return False
            if candidate_value < reference_value - tolerance:
                strictly_better = True
    return strictly_better


def pareto_frontier(
    points: Iterable[ParetoPoint], objectives: Iterable[ParetoObjective]
) -> ParetoResult:
    """Compute an O(n²) tolerance-aware frontier with deterministic ordering."""

    ordered_points = tuple(sorted(points, key=lambda point: point.point_id))
    ordered_objectives = tuple(objectives)
    if not ordered_objectives:
        raise ValueError("Pareto analysis requires at least one objective")
    objective_names = [objective.metric for objective in ordered_objectives]
    if len(set(objective_names)) != len(objective_names):
        raise ValueError("Pareto objective names must be unique")
    point_ids = [point.point_id for point in ordered_points]
    if len(set(point_ids)) != len(point_ids):
        raise ValueError("Pareto point IDs must be unique")
    for point in ordered_points:
        missing = [name for name in objective_names if name not in point.values]
        if missing:
            raise ValueError(
                f"Pareto point {point.point_id!r} is missing objectives: {', '.join(missing)}"
            )
    memberships = tuple(
        ParetoMembership(
            point_id=point.point_id,
            pareto_optimal=not dominators,
            dominated_by=dominators,
        )
        for point in ordered_points
        for dominators in [
            tuple(
                other.point_id
                for other in ordered_points
                if other.point_id != point.point_id and dominates(other, point, ordered_objectives)
            )
        ]
    )
    return ParetoResult(objectives=ordered_objectives, memberships=memberships)


def pareto_points_from_summaries(
    summaries: Iterable[NormalizedRunSummary], objectives: Iterable[ParetoObjective]
) -> tuple[ParetoPoint, ...]:
    """Extract available objective values without treating unsupported metrics as zero."""

    objective_tuple = tuple(objectives)
    summary_tuple = tuple(summaries)
    incompatible_sources: set[str] = set()
    for objective in objective_tuple:
        sources = {
            (
                metric.evaluation_suite.scientific_json()
                if metric.evaluation_suite is not None
                else None
            )
            for summary in summary_tuple
            for metric in [summary.metrics.get(objective.metric)]
            if metric is not None
        }
        if len(sources) > 1:
            incompatible_sources.add(objective.metric)
    points: list[ParetoPoint] = []
    for summary in summary_tuple:
        values: dict[str, float] = {}
        for objective in objective_tuple:
            if objective.metric in incompatible_sources:
                break
            metric = summary.metrics.get(objective.metric)
            if metric is None or metric.status != "available" or metric.value is None:
                break
            values[objective.metric] = metric.value
        else:
            points.append(ParetoPoint(point_id=summary.run_id, values=values))
    return tuple(points)


__all__ = [
    "BenchmarkMemoryIdentity",
    "BenchmarkWorkloadIdentity",
    "ComparisonContract",
    "ComparisonResult",
    "CompatibilityMismatch",
    "DatasetIdentity",
    "EvaluationSuiteIdentity",
    "MetricDelta",
    "MetricValue",
    "NormalizedRunSummary",
    "ParetoMembership",
    "ParetoObjective",
    "ParetoPoint",
    "ParetoResult",
    "RoutingCaptureIdentity",
    "compare_summaries",
    "compatibility_mismatches",
    "compute_metric_deltas",
    "dominates",
    "pareto_frontier",
    "pareto_points_from_summaries",
]
