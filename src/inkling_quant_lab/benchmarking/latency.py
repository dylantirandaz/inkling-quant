"""Deterministic generation benchmark orchestration and latency statistics."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from inkling_quant_lab.benchmarking.energy import (
    EnergyMeasurement,
    EnergySensor,
    begin_energy_measurement,
    finish_energy_measurement,
)
from inkling_quant_lab.benchmarking.memory import (
    PeakMemoryMeasurement,
    aggregate_peak_memory,
    validate_memory_snapshot,
)
from inkling_quant_lab.benchmarking.throughput import tokens_per_second
from inkling_quant_lab.benchmarking.utilization import (
    HardwareUtilizationMeasurement,
    UtilizationSensor,
    begin_hardware_utilization,
    default_process_cpu_sensor,
    finish_hardware_utilization,
)
from inkling_quant_lab.config import BenchmarkConfig
from inkling_quant_lab.exceptions import BenchmarkError
from inkling_quant_lab.models.base import LoadedModel, ModelAdapter
from inkling_quant_lab.runtimes.base import MemorySnapshot, RuntimeBackend

Clock = Callable[[], float]


class _ImmutableBenchmarkRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrialObservation(_ImmutableBenchmarkRecord):
    """Adapter/runtime observation returned from one instrumented generation."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=1)
    time_to_first_token_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    inter_token_latencies_seconds: tuple[float, ...] = ()

    @model_validator(mode="after")
    def validate_token_timeline(self) -> Self:
        """Require exactly one interval between every pair of output tokens."""

        expected = self.output_tokens - 1
        if len(self.inter_token_latencies_seconds) != expected:
            raise ValueError("inter_token_latencies_seconds must contain output_tokens - 1 values")
        if any(
            not math.isfinite(value) or value < 0.0 for value in self.inter_token_latencies_seconds
        ):
            raise ValueError("inter-token latencies must be finite and non-negative")
        return self


TrialCallable: TypeAlias = Callable[[ModelAdapter, LoadedModel], TrialObservation]


class BenchmarkTrial(_ImmutableBenchmarkRecord):
    """One retained measured trial; warm-up records are never stored here."""

    index: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=1)
    time_to_first_token_ms: float = Field(ge=0.0, allow_inf_nan=False)
    inter_token_latency_ms: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    end_to_end_ms: float = Field(gt=0.0, allow_inf_nan=False)
    tokens_per_second: float = Field(gt=0.0, allow_inf_nan=False)
    host_memory_bytes_at_post_trial_sample: int | None = Field(default=None, ge=0)
    device_memory_bytes_at_post_trial_sample: int | None = Field(default=None, ge=0)


class DistributionStatistics(_ImmutableBenchmarkRecord):
    """Population statistics and linearly interpolated percentiles."""

    sample_count: int = Field(ge=1)
    median: float = Field(allow_inf_nan=False)
    p10: float = Field(allow_inf_nan=False)
    p90: float = Field(allow_inf_nan=False)
    mean: float = Field(allow_inf_nan=False)
    stdev: float = Field(ge=0.0, allow_inf_nan=False)


class LatencyStatistics(_ImmutableBenchmarkRecord):
    """Statistics for latency metrics, each retaining its natural unit."""

    time_to_first_token_ms: DistributionStatistics
    inter_token_latency_ms: DistributionStatistics | None
    end_to_end_ms: DistributionStatistics


class BenchmarkWorkloadProvenance(_ImmutableBenchmarkRecord):
    """Exact immutable dataset, prompt, decode, and execution workload identity."""

    dataset_id: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: str = Field(min_length=1)
    sample_ids: tuple[str, ...]
    seed: int
    prompt_template_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decode_config: dict[str, JsonValue]
    execution_mode: str = "sequential_samples"

    @field_validator("sample_ids")
    @classmethod
    def sample_ids_are_nonempty_and_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require an exact, unambiguous benchmark sample sequence."""

        if not value or any(not sample_id for sample_id in value):
            raise ValueError("benchmark workload requires non-empty stable sample_ids")
        if len(set(value)) != len(value):
            raise ValueError("benchmark workload sample_ids must be unique")
        return value


class BenchmarkResult(_ImmutableBenchmarkRecord):
    """Complete steady-state benchmark result with separate model-load time."""

    protocol_version: str
    model_id: str
    model_revision: str | None
    model_checksum: str
    workload: BenchmarkWorkloadProvenance
    model_load_time_ms: float = Field(gt=0.0, allow_inf_nan=False)
    model_load_time_kind: str
    warmup_iterations: int = Field(ge=0)
    repetitions: int = Field(ge=1)
    trials: tuple[BenchmarkTrial, ...]
    latency: LatencyStatistics
    throughput_tokens_per_second: DistributionStatistics
    peak_memory: PeakMemoryMeasurement
    serialized_size_bytes: int = Field(ge=0)
    hardware_utilization: HardwareUtilizationMeasurement
    energy: EnergyMeasurement
    source_weight_free_load_provenance: dict[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class _TimedObservation:
    observation: TrialObservation
    elapsed_seconds: float


def run_generation_benchmark(
    adapter: ModelAdapter,
    model: LoadedModel,
    runtime: RuntimeBackend,
    config: BenchmarkConfig,
    trial: TrialCallable,
    *,
    serialized_size_bytes: int,
    workload: BenchmarkWorkloadProvenance,
    clock: Clock = time.perf_counter,
    energy_sensor: EnergySensor | None = None,
    energy_unavailable_reason: str | None = None,
    energy_clock: Clock = time.perf_counter,
    utilization_sensor: UtilizationSensor | None = None,
) -> BenchmarkResult:
    """Run warm-ups and measured generation trials under one runtime context.

    The trial callable supplies token-level timing facts because the generic
    model-adapter contract is intentionally not assumed to expose streaming.
    End-to-end duration is always measured by this runner around the callable.
    """

    if not config.enabled:
        raise BenchmarkError(
            "cannot execute a benchmark disabled by configuration",
            component="benchmark_latency",
        )
    if serialized_size_bytes < 0:
        raise BenchmarkError(
            "serialized checkpoint size must be non-negative",
            component="benchmark_latency",
        )
    if not math.isfinite(model.load_time_seconds) or model.load_time_seconds <= 0.0:
        raise BenchmarkError(
            "model load time must be finite and positive",
            component="benchmark_latency",
        )

    energy_session = begin_energy_measurement(
        enabled=config.measure_energy,
        sensor=energy_sensor,
        unavailable_reason=energy_unavailable_reason,
        clock=energy_clock,
    )
    utilization_session = begin_hardware_utilization(
        utilization_sensor or default_process_cpu_sensor()
    )
    memory_interval_started = time.perf_counter()
    memory_baseline = runtime.memory_snapshot()
    validate_memory_snapshot(memory_baseline)
    measured: list[tuple[_TimedObservation, MemorySnapshot]] = []
    memory_samples: list[MemorySnapshot] = []

    with runtime.execution_context():
        for warmup_index in range(config.warmup_iterations):
            timed = _run_timed_region(
                adapter,
                model,
                runtime,
                trial,
                clock,
                synchronize=config.synchronize,
                phase="warmup",
                index=warmup_index,
            )
            warmup_snapshot = runtime.memory_snapshot()
            validate_memory_snapshot(warmup_snapshot)
            memory_samples.append(warmup_snapshot)

        for trial_index in range(config.repetitions):
            timed = _run_timed_region(
                adapter,
                model,
                runtime,
                trial,
                clock,
                synchronize=config.synchronize,
                phase="trial",
                index=trial_index,
            )
            snapshot = runtime.memory_snapshot()
            validate_memory_snapshot(snapshot)
            memory_samples.append(snapshot)
            measured.append((timed, snapshot))
    memory_interval_seconds = time.perf_counter() - memory_interval_started

    hardware_utilization = finish_hardware_utilization(utilization_session)
    energy = finish_energy_measurement(
        energy_session,
        clock=energy_clock,
    )
    trials = tuple(
        _trial_record(index, timed, snapshot) for index, (timed, snapshot) in enumerate(measured)
    )
    peak_memory = aggregate_peak_memory(
        memory_samples,
        baseline=memory_baseline,
        measurement_interval_seconds=memory_interval_seconds,
    )
    latency = LatencyStatistics(
        time_to_first_token_ms=summarize_distribution(
            tuple(item.time_to_first_token_ms for item in trials)
        ),
        inter_token_latency_ms=_optional_distribution(
            tuple(
                item.inter_token_latency_ms
                for item in trials
                if item.inter_token_latency_ms is not None
            )
        ),
        end_to_end_ms=summarize_distribution(tuple(item.end_to_end_ms for item in trials)),
    )
    throughput = summarize_distribution(tuple(item.tokens_per_second for item in trials))

    return BenchmarkResult(
        protocol_version=config.protocol_version,
        model_id=model.descriptor.model_id,
        model_revision=model.descriptor.revision,
        model_checksum=model.descriptor.checksum,
        workload=workload,
        model_load_time_ms=model.load_time_seconds * 1_000.0,
        model_load_time_kind=model.load_time_kind,
        warmup_iterations=config.warmup_iterations,
        repetitions=config.repetitions,
        trials=trials,
        latency=latency,
        throughput_tokens_per_second=throughput,
        peak_memory=peak_memory,
        serialized_size_bytes=serialized_size_bytes,
        hardware_utilization=hardware_utilization,
        energy=energy,
    )


def summarize_distribution(values: Sequence[float]) -> DistributionStatistics:
    """Summarize non-empty finite values using population standard deviation."""

    if not values:
        raise BenchmarkError(
            "cannot summarize an empty benchmark distribution",
            component="benchmark_statistics",
        )
    measured = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in measured):
        raise BenchmarkError(
            "benchmark distributions must contain only finite values",
            component="benchmark_statistics",
        )
    ordered = tuple(sorted(measured))
    return DistributionStatistics(
        sample_count=len(ordered),
        median=statistics.median(ordered),
        p10=_percentile(ordered, 0.10),
        p90=_percentile(ordered, 0.90),
        mean=statistics.fmean(ordered),
        stdev=statistics.pstdev(ordered),
    )


def _run_timed_region(
    adapter: ModelAdapter,
    model: LoadedModel,
    runtime: RuntimeBackend,
    trial: TrialCallable,
    clock: Clock,
    *,
    synchronize: bool,
    phase: str,
    index: int,
) -> _TimedObservation:
    if synchronize:
        runtime.synchronize()
    started = float(clock())
    try:
        observation = trial(adapter, model)
    finally:
        if synchronize:
            runtime.synchronize()
    ended = float(clock())
    elapsed = ended - started
    if not math.isfinite(started) or not math.isfinite(ended) or elapsed <= 0.0:
        raise BenchmarkError(
            f"{phase} {index} produced a non-positive or non-finite duration",
            component="benchmark_latency",
            details={"phase": phase, "index": index, "started": started, "ended": ended},
        )
    latest_token = observation.time_to_first_token_seconds + math.fsum(
        observation.inter_token_latencies_seconds
    )
    tolerance = max(1e-12, elapsed * 1e-9)
    if latest_token > elapsed + tolerance:
        raise BenchmarkError(
            f"{phase} {index} token timeline exceeds measured end-to-end duration",
            component="benchmark_latency",
            details={
                "phase": phase,
                "index": index,
                "latest_token_seconds": latest_token,
                "elapsed_seconds": elapsed,
            },
        )
    return _TimedObservation(observation=observation, elapsed_seconds=elapsed)


def _trial_record(index: int, timed: _TimedObservation, snapshot: MemorySnapshot) -> BenchmarkTrial:
    observation = timed.observation
    intervals = observation.inter_token_latencies_seconds
    inter_token_ms = statistics.fmean(intervals) * 1_000.0 if intervals else None
    return BenchmarkTrial(
        index=index,
        input_tokens=observation.input_tokens,
        output_tokens=observation.output_tokens,
        time_to_first_token_ms=observation.time_to_first_token_seconds * 1_000.0,
        inter_token_latency_ms=inter_token_ms,
        end_to_end_ms=timed.elapsed_seconds * 1_000.0,
        tokens_per_second=tokens_per_second(observation.output_tokens, timed.elapsed_seconds),
        host_memory_bytes_at_post_trial_sample=(
            snapshot.host_bytes if snapshot.host_available else None
        ),
        device_memory_bytes_at_post_trial_sample=(
            snapshot.device_bytes if snapshot.device_available else None
        ),
    )


def _optional_distribution(values: Sequence[float]) -> DistributionStatistics | None:
    return summarize_distribution(values) if values else None


def _percentile(ordered: Sequence[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


__all__ = [
    "BenchmarkResult",
    "BenchmarkTrial",
    "BenchmarkWorkloadProvenance",
    "DistributionStatistics",
    "LatencyStatistics",
    "TrialCallable",
    "TrialObservation",
    "run_generation_benchmark",
    "summarize_distribution",
]
