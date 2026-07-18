"""Declared host CPU process-utilization sampling for benchmark intervals."""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inkling_quant_lab.exceptions import BenchmarkError

_SCOPE = "warm-up and measured trials, including runtime memory-sampling overhead"


class UtilizationSample(BaseModel):
    """One cumulative wall/process-CPU clock observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    wall_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    process_cpu_seconds: float = Field(ge=0.0, allow_inf_nan=False)


class UtilizationSensor(Protocol):
    """Cumulative sampler used to delimit one utilization interval."""

    name: str
    logical_cpu_count: int

    def sample(self) -> UtilizationSample: ...


class ProcessCPUTimeSensor:
    """Portable process CPU-time sensor normalized by logical CPU capacity."""

    name = "python_process_cpu_time"

    def __init__(self, logical_cpu_count: int) -> None:
        if logical_cpu_count <= 0:
            raise ValueError("logical_cpu_count must be positive")
        self.logical_cpu_count = logical_cpu_count

    def sample(self) -> UtilizationSample:
        """Read monotonic wall time and cumulative process CPU time."""

        return UtilizationSample(
            wall_seconds=time.perf_counter(),
            process_cpu_seconds=time.process_time(),
        )


class HardwareUtilizationMeasurement(BaseModel):
    """Available utilization fact or an explicit reason it was unavailable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["available", "unavailable"]
    metric: str = "normalized_process_cpu_percent"
    value_percent: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    process_cpu_seconds: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    sampling_interval_seconds: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    logical_cpu_count: int | None = Field(default=None, ge=1)
    sensor_name: str | None = None
    scope: str = _SCOPE
    reason: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> Self:
        """Prevent unavailable measurements from carrying fabricated numeric values."""

        facts = (
            self.value_percent,
            self.process_cpu_seconds,
            self.sampling_interval_seconds,
            self.logical_cpu_count,
            self.sensor_name,
        )
        if self.status == "available":
            if any(value is None for value in facts) or self.reason is not None:
                raise ValueError("available hardware utilization requires all measured facts")
        elif any(value is not None for value in facts) or not self.reason:
            raise ValueError("unavailable hardware utilization requires only a reason")
        return self


@dataclass(frozen=True, slots=True)
class UtilizationSession:
    """Start sample or a durable reason sampling did not begin."""

    sensor: UtilizationSensor | None
    start: UtilizationSample | None
    reason: str | None


def default_process_cpu_sensor() -> UtilizationSensor | None:
    """Return the dependency-free CPU sensor when logical capacity is known."""

    logical_cpu_count = os.cpu_count()
    if logical_cpu_count is None or logical_cpu_count <= 0:
        return None
    return ProcessCPUTimeSensor(logical_cpu_count)


def begin_hardware_utilization(sensor: UtilizationSensor | None) -> UtilizationSession:
    """Begin sampling, retaining explicit unavailability instead of a zero."""

    if sensor is None:
        return UtilizationSession(None, None, "logical CPU capacity is unavailable")
    try:
        start = sensor.sample()
    except (OSError, RuntimeError, ValueError) as error:
        return UtilizationSession(None, None, f"CPU utilization sensor failed to start: {error}")
    return UtilizationSession(sensor, start, None)


def finish_hardware_utilization(
    session: UtilizationSession,
) -> HardwareUtilizationMeasurement:
    """Finish the interval and compute normalized process CPU utilization."""

    if session.sensor is None or session.start is None:
        return HardwareUtilizationMeasurement(
            status="unavailable",
            reason=session.reason or "CPU utilization sensor unavailable",
        )
    try:
        end = session.sensor.sample()
    except (OSError, RuntimeError, ValueError) as error:
        return HardwareUtilizationMeasurement(
            status="unavailable",
            reason=f"CPU utilization sensor failed to finish: {error}",
        )
    wall_seconds = end.wall_seconds - session.start.wall_seconds
    process_cpu_seconds = end.process_cpu_seconds - session.start.process_cpu_seconds
    if (
        not math.isfinite(wall_seconds)
        or not math.isfinite(process_cpu_seconds)
        or wall_seconds <= 0.0
        or process_cpu_seconds < 0.0
    ):
        raise BenchmarkError(
            "CPU utilization sensor returned an invalid sampling interval",
            component="benchmark_utilization",
            details={
                "sampling_interval_seconds": wall_seconds,
                "process_cpu_seconds": process_cpu_seconds,
            },
        )
    logical_cpu_count = session.sensor.logical_cpu_count
    if logical_cpu_count <= 0:
        raise BenchmarkError(
            "CPU utilization sensor returned invalid logical CPU capacity",
            component="benchmark_utilization",
        )
    percent = 100.0 * process_cpu_seconds / (wall_seconds * logical_cpu_count)
    return HardwareUtilizationMeasurement(
        status="available",
        value_percent=percent,
        process_cpu_seconds=process_cpu_seconds,
        sampling_interval_seconds=wall_seconds,
        logical_cpu_count=logical_cpu_count,
        sensor_name=session.sensor.name,
    )


__all__ = [
    "HardwareUtilizationMeasurement",
    "ProcessCPUTimeSensor",
    "UtilizationSample",
    "UtilizationSensor",
    "begin_hardware_utilization",
    "default_process_cpu_sensor",
    "finish_hardware_utilization",
]
