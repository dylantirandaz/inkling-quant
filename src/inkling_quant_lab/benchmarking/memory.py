"""Portable peak-memory aggregation with explicit unavailable values."""

from __future__ import annotations

import ctypes
import os
import platform
from collections.abc import Iterable
from ctypes.util import find_library
from functools import lru_cache
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inkling_quant_lab.exceptions import BenchmarkError
from inkling_quant_lab.runtimes.base import MemorySnapshot


class _MachTimeValue(ctypes.Structure):
    _fields_ = (("seconds", ctypes.c_int32), ("microseconds", ctypes.c_int32))


class _MachTaskBasicInfo(ctypes.Structure):
    _fields_ = (
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time", _MachTimeValue),
        ("system_time", _MachTimeValue),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
    )


@lru_cache(maxsize=1)
def _mach_task_library() -> Any:
    """Bind the small Mach task-info surface once for repeated RSS reads."""

    library = ctypes.CDLL(find_library("System") or "/usr/lib/libSystem.B.dylib")
    library.mach_task_self.restype = ctypes.c_uint32
    library.task_info.argtypes = (
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    )
    library.task_info.restype = ctypes.c_int32
    return library


def _mach_task_basic_info() -> _MachTaskBasicInfo:
    """Return current and lifetime resident-size facts for this Mach task."""

    library = _mach_task_library()
    info = _MachTaskBasicInfo()
    count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
    status = int(
        library.task_info(
            library.mach_task_self(),
            20,  # MACH_TASK_BASIC_INFO
            ctypes.byref(info),
            ctypes.byref(count),
        )
    )
    if status != 0:
        raise OSError(f"macOS task_info RSS collector failed with status {status}")
    return info


def current_process_rss_bytes(system: str | None = None) -> int:
    """Read current process RSS on Linux or macOS without optional dependencies."""

    resolved_system = platform.system() if system is None else system
    if resolved_system == "Linux":
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        if len(fields) < 2:
            raise OSError("/proc/self/statm did not contain a resident-page field")
        resident_pages = int(fields[1])
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        if resident_pages < 0 or page_size <= 0:
            raise OSError("Linux current-RSS collector returned invalid page counts")
        return resident_pages * page_size
    if resolved_system == "Darwin":
        info = _mach_task_basic_info()
        if info.resident_size <= 0:
            raise OSError("macOS task_info current-RSS collector returned a non-positive value")
        return int(info.resident_size)
    raise OSError(f"current process RSS is unsupported on {resolved_system}")


def process_peak_rss_bytes(
    system: str | None = None,
    *,
    linux_status_path: str | Path = "/proc/self/status",
) -> int:
    """Read the OS lifetime high-water RSS for the current process.

    This value becomes stage-scoped only when the caller owns a fresh process
    dedicated to exactly one governed stage. It must not be relabeled as an
    interval peak inside a long-lived process.
    """

    resolved_system = platform.system() if system is None else system
    if resolved_system == "Linux":
        path = Path(linux_status_path)
        try:
            lines = path.read_text(encoding="ascii").splitlines()
        except OSError as error:
            raise OSError(f"unable to read Linux process high-water RSS from {path}") from error
        for line in lines:
            if not line.startswith("VmHWM:"):
                continue
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB":
                raise OSError("Linux VmHWM must contain one positive kB value")
            try:
                kibibytes = int(fields[1])
            except ValueError as error:
                raise OSError("Linux VmHWM was not an integer") from error
            if kibibytes <= 0:
                raise OSError("Linux VmHWM must be positive")
            return kibibytes * 1024
        raise OSError("Linux process status did not contain VmHWM")
    if resolved_system == "Darwin":
        peak = int(_mach_task_basic_info().resident_size_max)
        if peak <= 0:
            raise OSError("macOS task_info peak-RSS collector returned a non-positive value")
        return peak
    raise OSError(f"process peak RSS is unsupported on {resolved_system}")


class PeakMemoryMeasurement(BaseModel):
    """Maximum observed readings plus the exact collector kind and scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host_bytes: int | None = Field(default=None, ge=0)
    device_bytes: int | None = Field(default=None, ge=0)
    host_available: bool
    device_available: bool
    host_measurement_kind: str | None = None
    host_scope: str | None = None
    device_measurement_kind: str | None = None
    device_scope: str | None = None
    host_baseline_bytes: int | None = Field(default=None, ge=0)
    host_max_observed_delta_bytes: int | None = Field(default=None, ge=0)
    host_sample_count: int | None = Field(default=None, ge=1)
    measurement_interval_seconds: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    host_process_isolated: bool = False
    host_worker_pid: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_measurement_metadata(self) -> Self:
        """Require labels for values and forbid labels for unavailable readings."""

        _validate_metadata(
            "host",
            self.host_available,
            self.host_bytes,
            self.host_measurement_kind,
            self.host_scope,
        )
        _validate_metadata(
            "device",
            self.device_available,
            self.device_bytes,
            self.device_measurement_kind,
            self.device_scope,
        )
        interval_facts = (
            self.host_baseline_bytes,
            self.host_max_observed_delta_bytes,
            self.host_sample_count,
            self.measurement_interval_seconds,
        )
        if any(value is not None for value in interval_facts) and any(
            value is None for value in interval_facts
        ):
            raise ValueError("sampled host interval requires baseline, delta, count, and duration")
        if self.host_baseline_bytes is not None and not self.host_available:
            raise ValueError("unavailable host memory cannot carry sampled interval facts")
        if self.host_process_isolated:
            if not self.host_available or self.host_worker_pid is None:
                raise ValueError("isolated host memory requires an available value and worker PID")
            isolated_kinds = {
                "benchmark_stage_worker_process_peak_rss",
                "benchmark_subject_artifact_worker_process_peak_rss",
            }
            if self.host_measurement_kind not in isolated_kinds:
                raise ValueError("isolated host memory requires a governed worker peak-RSS kind")
        elif self.host_worker_pid is not None:
            raise ValueError("non-isolated host memory cannot carry a worker PID")
        return self


def validate_memory_snapshot(snapshot: MemorySnapshot) -> None:
    """Reject internally inconsistent or negative runtime memory readings."""

    _validate_reading(
        "host",
        snapshot.host_available,
        snapshot.host_bytes,
        snapshot.host_measurement_kind,
        snapshot.host_scope,
    )
    _validate_reading(
        "device",
        snapshot.device_available,
        snapshot.device_bytes,
        snapshot.device_measurement_kind,
        snapshot.device_scope,
    )


def aggregate_peak_memory(
    snapshots: Iterable[MemorySnapshot],
    *,
    baseline: MemorySnapshot | None = None,
    measurement_interval_seconds: float | None = None,
) -> PeakMemoryMeasurement:
    """Aggregate sampled maxima and a non-negative delta from an optional baseline."""

    host_values: list[int] = []
    device_values: list[int] = []
    host_metadata: set[tuple[str, str]] = set()
    device_metadata: set[tuple[str, str]] = set()
    host_baseline_bytes: int | None = None
    if baseline is not None:
        validate_memory_snapshot(baseline)
        if baseline.host_available and baseline.host_bytes is not None:
            host_baseline_bytes = baseline.host_bytes
            host_values.append(baseline.host_bytes)
            assert baseline.host_measurement_kind is not None
            assert baseline.host_scope is not None
            host_metadata.add((baseline.host_measurement_kind, baseline.host_scope))
    for snapshot in snapshots:
        validate_memory_snapshot(snapshot)
        if snapshot.host_available and snapshot.host_bytes is not None:
            host_values.append(snapshot.host_bytes)
            assert snapshot.host_measurement_kind is not None
            assert snapshot.host_scope is not None
            host_metadata.add((snapshot.host_measurement_kind, snapshot.host_scope))
        if snapshot.device_available and snapshot.device_bytes is not None:
            device_values.append(snapshot.device_bytes)
            assert snapshot.device_measurement_kind is not None
            assert snapshot.device_scope is not None
            device_metadata.add((snapshot.device_measurement_kind, snapshot.device_scope))

    if len(host_metadata) > 1 or len(device_metadata) > 1:
        raise BenchmarkError(
            "cannot aggregate memory snapshots with different collector kinds or scopes",
            component="benchmark_memory",
        )
    host_kind, host_scope = next(iter(host_metadata), (None, None))
    device_kind, device_scope = next(iter(device_metadata), (None, None))
    if host_kind == "process_current_rss":
        if len(host_values) < 2:
            raise BenchmarkError(
                "current-RSS interval requires both baseline and post-boundary samples",
                component="benchmark_memory",
            )
        host_kind = "benchmark_interval_sampled_current_rss"
        host_scope = (
            "Python process current RSS sampled immediately before warm-up and after each "
            "warm-up/measured trial; not a continuous or allocator-native peak"
        )
    host_max = max(host_values) if host_values else None
    host_delta = (
        max(0, host_max - host_baseline_bytes)
        if host_max is not None and host_baseline_bytes is not None
        else None
    )
    interval_seconds = measurement_interval_seconds if host_baseline_bytes is not None else None
    if host_baseline_bytes is not None and (interval_seconds is None or interval_seconds <= 0.0):
        raise BenchmarkError(
            "sampled host memory requires a positive measurement interval",
            component="benchmark_memory",
        )

    return PeakMemoryMeasurement(
        host_bytes=host_max,
        device_bytes=max(device_values) if device_values else None,
        host_available=bool(host_values),
        device_available=bool(device_values),
        host_measurement_kind=host_kind,
        host_scope=host_scope,
        device_measurement_kind=device_kind,
        device_scope=device_scope,
        host_baseline_bytes=host_baseline_bytes,
        host_max_observed_delta_bytes=host_delta,
        host_sample_count=len(host_values) if host_baseline_bytes is not None else None,
        measurement_interval_seconds=interval_seconds,
    )


def _validate_reading(
    name: str,
    available: bool,
    value: int | None,
    measurement_kind: str | None,
    scope: str | None,
) -> None:
    if available and value is None:
        raise BenchmarkError(
            f"runtime marks {name} memory available but returned no value",
            component="benchmark_memory",
        )
    if value is not None and value < 0:
        raise BenchmarkError(
            f"runtime returned negative {name} memory: {value}",
            component="benchmark_memory",
        )
    _validate_metadata(name, available, value, measurement_kind, scope)


def _validate_metadata(
    name: str,
    available: bool,
    value: int | None,
    measurement_kind: str | None,
    scope: str | None,
) -> None:
    metadata = (measurement_kind, scope)
    if available and value is not None and any(not item for item in metadata):
        raise BenchmarkError(
            f"runtime {name} memory reading requires measurement kind and scope",
            component="benchmark_memory",
        )
    if not available and (value is not None or any(item is not None for item in metadata)):
        raise BenchmarkError(
            f"unavailable {name} memory cannot carry a value or measurement metadata",
            component="benchmark_memory",
        )


__all__ = [
    "PeakMemoryMeasurement",
    "aggregate_peak_memory",
    "current_process_rss_bytes",
    "process_peak_rss_bytes",
    "validate_memory_snapshot",
]
