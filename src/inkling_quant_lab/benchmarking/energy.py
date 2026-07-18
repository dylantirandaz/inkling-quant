"""Capability-gated cumulative energy sampling with explicit measurement scope."""

from __future__ import annotations

import math
import os
import platform
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inkling_quant_lab.exceptions import BenchmarkError

Clock = Callable[[], float]

_RAPL_CONTROL_TYPE = "intel-rapl"
_RAPL_PACKAGE_ZONE = re.compile(r"^intel-rapl:[0-9]+$")
_DEFAULT_POWERCAP_ROOTS = (
    Path("/sys/class/powercap"),
    Path("/sys/devices/virtual/powercap"),
)
_DEFAULT_POWERMETRICS_PATH = Path("/usr/bin/powermetrics")
_RAPL_SCOPE = (
    "sum of top-level Intel RAPL CPU-package domains for the entire host from the sensor boundary "
    "before the pre-warm-up memory baseline through the boundary after all warm-up/measured "
    "trials, memory samples, and utilization sampling; includes concurrent processes and package "
    "components and is not process-attributable"
)
_RAPL_WRAPAROUND = (
    "each package counter delta is computed modulo its max_energy_range_uj; one rollover between "
    "adjacent reads is detectable"
)


class EnergyDomainProvenance(BaseModel):
    """One cumulative hardware-counter domain included in an energy measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    domain_id: str = Field(min_length=1)
    domain_name: str = Field(min_length=1)
    counter_path: str = Field(min_length=1)
    max_energy_range_uj: int = Field(gt=0)


class EnergySensorProvenance(BaseModel):
    """Stable interpretation metadata for a cumulative energy sensor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sensor_name: str = Field(min_length=1)
    measurement_kind: str = Field(min_length=1)
    scope: str = Field(min_length=1)
    unit: Literal["joules"] = "joules"
    domains: tuple[EnergyDomainProvenance, ...]
    counter_wraparound_handling: str = Field(min_length=1)

    @model_validator(mode="after")
    def require_domains(self) -> Self:
        """Available cumulative sensors must identify every aggregated domain."""

        if not self.domains:
            raise ValueError("energy sensor provenance requires at least one domain")
        if len({domain.domain_id for domain in self.domains}) != len(self.domains):
            raise ValueError("energy sensor domain IDs must be unique")
        return self


class EnergySensor(Protocol):
    """Cumulative-energy sensor used to delimit one benchmark interval."""

    name: str
    provenance: EnergySensorProvenance

    @property
    def observed_wraparounds(self) -> int: ...

    def read_joules(self) -> float: ...


class EnergySensorCapability(BaseModel):
    """Availability and interpretation of the default energy collector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["available", "unavailable"]
    provenance: EnergySensorProvenance | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def status_fields_are_consistent(self) -> Self:
        """Never attach a collector identity to an unavailable capability."""

        if self.status == "available":
            if self.provenance is None or self.reason is not None:
                raise ValueError("available energy capability requires provenance without a reason")
        elif self.provenance is not None or not self.reason:
            raise ValueError("unavailable energy capability requires only an explicit reason")
        return self


class EnergyMeasurement(BaseModel):
    """Available energy delta or an explicit unavailability reason."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["available", "unavailable"]
    joules: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    sensor_name: str | None = None
    sampling_interval_seconds: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)
    observed_counter_wraparounds: int | None = Field(default=None, ge=0)
    provenance: EnergySensorProvenance | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def status_fields_are_consistent(self) -> Self:
        """Require interpretation metadata for values and no numbers for unavailability."""

        facts = (
            self.joules,
            self.sensor_name,
            self.sampling_interval_seconds,
            self.observed_counter_wraparounds,
            self.provenance,
        )
        if self.status == "available":
            if any(value is None for value in facts) or self.reason is not None:
                raise ValueError("available energy measurement requires all measured facts")
            provenance = self.provenance
            assert provenance is not None
            if self.sensor_name != provenance.sensor_name:
                raise ValueError("energy sensor name must match its provenance")
        elif any(value is not None for value in facts) or not self.reason:
            raise ValueError("unavailable energy measurement requires only an explicit reason")
        return self


@dataclass(frozen=True, slots=True)
class EnergySession:
    """Internal start sample or a durable reason sampling did not begin."""

    sensor: EnergySensor | None
    start_joules: float | None
    unavailable_reason: str | None
    start_wraparounds: int = 0
    started_at_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class EnergySensorSelection:
    """A constructed sensor paired with its serializable capability result."""

    sensor: EnergySensor | None
    capability: EnergySensorCapability

    @property
    def available(self) -> bool:
        """Return whether a usable collector was constructed."""

        return self.sensor is not None and self.capability.status == "available"


@dataclass(frozen=True, slots=True)
class _RaplDomain:
    path: Path
    provenance: EnergyDomainProvenance


class LinuxPowercapRaplEnergySensor:
    """Aggregate Linux powercap counters for all top-level CPU package zones."""

    name = "linux_powercap_intel_rapl_packages"

    def __init__(self, domains: Sequence[_RaplDomain]) -> None:
        if not domains:
            raise ValueError("Linux powercap RAPL requires at least one package domain")
        self._domains = tuple(domains)
        self._last_energy_uj: tuple[int, ...] | None = None
        self._cumulative_energy_uj = 0
        self._observed_wraparounds = 0
        self.provenance = EnergySensorProvenance(
            sensor_name=self.name,
            measurement_kind="linux_powercap_rapl_package_energy_counter_delta",
            scope=_RAPL_SCOPE,
            domains=tuple(domain.provenance for domain in self._domains),
            counter_wraparound_handling=_RAPL_WRAPAROUND,
        )

    @classmethod
    def from_powercap_root(cls, powercap_root: Path) -> Self:
        """Discover readable top-level Intel RAPL package zones under one sysfs root."""

        root = Path(powercap_root)
        control_type = root if root.name == _RAPL_CONTROL_TYPE else root / _RAPL_CONTROL_TYPE
        if not control_type.is_dir():
            raise FileNotFoundError(f"Intel RAPL control type not found under {root}")
        zone_paths = tuple(
            sorted(
                (
                    path
                    for path in control_type.iterdir()
                    if _RAPL_PACKAGE_ZONE.fullmatch(path.name) and path.is_dir()
                ),
                key=lambda path: path.name,
            )
        )
        if not zone_paths:
            raise FileNotFoundError(
                f"no top-level Intel RAPL package zones found under {control_type}"
            )

        domains: list[_RaplDomain] = []
        for zone in zone_paths:
            domain_name = _read_text(zone / "name", "RAPL package name")
            if not domain_name.startswith("package-"):
                raise ValueError(
                    f"top-level RAPL zone {zone.name!r} has non-package name {domain_name!r}"
                )
            maximum = _read_nonnegative_integer(
                zone / "max_energy_range_uj", "RAPL maximum energy range"
            )
            if maximum <= 0:
                raise ValueError(f"RAPL package {domain_name!r} has a non-positive counter range")
            counter_path = zone / "energy_uj"
            current = _read_nonnegative_integer(counter_path, "RAPL energy counter")
            if current > maximum:
                raise ValueError(
                    f"RAPL package {domain_name!r} counter exceeds max_energy_range_uj"
                )
            domains.append(
                _RaplDomain(
                    path=counter_path,
                    provenance=EnergyDomainProvenance(
                        domain_id=zone.name,
                        domain_name=domain_name,
                        counter_path=str(counter_path),
                        max_energy_range_uj=maximum,
                    ),
                )
            )
        return cls(domains)

    @property
    def observed_wraparounds(self) -> int:
        """Return the number of per-domain rollovers observed by this sensor instance."""

        return self._observed_wraparounds

    def read_joules(self) -> float:
        """Return a virtual monotonic counter assembled from package-domain deltas."""

        current = tuple(self._read_domain(domain) for domain in self._domains)
        if self._last_energy_uj is None:
            self._last_energy_uj = current
            return 0.0

        delta_uj = 0
        for domain, previous, latest in zip(
            self._domains, self._last_energy_uj, current, strict=True
        ):
            maximum = domain.provenance.max_energy_range_uj
            if latest >= previous:
                delta_uj += latest - previous
            else:
                delta_uj += maximum - previous + latest
                self._observed_wraparounds += 1
        self._last_energy_uj = current
        self._cumulative_energy_uj += delta_uj
        return self._cumulative_energy_uj / 1_000_000.0

    @staticmethod
    def _read_domain(domain: _RaplDomain) -> int:
        value = _read_nonnegative_integer(domain.path, "RAPL energy counter")
        maximum = domain.provenance.max_energy_range_uj
        if value > maximum:
            raise ValueError(
                f"RAPL package {domain.provenance.domain_name!r} counter exceeds "
                "max_energy_range_uj"
            )
        return value


def select_default_energy_sensor(
    *,
    runtime_backend: str,
    system: str | None = None,
    powercap_roots: Sequence[Path] | None = None,
    powermetrics_path: Path = _DEFAULT_POWERMETRICS_PATH,
    effective_uid: int | None = None,
) -> EnergySensorSelection:
    """Construct the validated default collector for a runtime, or retain a precise reason."""

    if runtime_backend != "torch_eager_cpu":
        return _unavailable_selection(
            f"runtime {runtime_backend!r} has no validated cumulative-energy collector"
        )

    operating_system = platform.system() if system is None else system
    if operating_system == "Linux":
        roots = _DEFAULT_POWERCAP_ROOTS if powercap_roots is None else tuple(powercap_roots)
        failures: list[str] = []
        for root in roots:
            try:
                sensor = LinuxPowercapRaplEnergySensor.from_powercap_root(root)
            except (OSError, ValueError) as error:
                failures.append(str(error))
                continue
            return EnergySensorSelection(
                sensor=sensor,
                capability=EnergySensorCapability(status="available", provenance=sensor.provenance),
            )
        detail = "; ".join(failure for failure in failures if failure)
        reason = "no readable Linux powercap Intel RAPL package energy counters were found"
        if detail:
            reason = f"{reason}: {detail}"
        return _unavailable_selection(reason)

    if operating_system == "Darwin":
        path = Path(powermetrics_path)
        if not path.is_file() or not os.access(path, os.X_OK):
            return _unavailable_selection("Apple powermetrics is not installed or executable")
        uid = _effective_uid() if effective_uid is None else effective_uid
        if uid != 0:
            return _unavailable_selection(
                "Apple powermetrics requires superuser privileges; its per-process energy impact "
                "is a platform-specific proxy and is not accepted as joules"
            )
        return _unavailable_selection(
            "Apple powermetrics is present, but no validated privileged cumulative-joule collector "
            "is implemented; per-process energy impact is not a joule measurement"
        )

    return _unavailable_selection(
        f"no validated cumulative-energy collector exists for operating system {operating_system!r}"
    )


def probe_default_energy_sensor(
    *,
    runtime_backend: str = "torch_eager_cpu",
    system: str | None = None,
    powercap_roots: Sequence[Path] | None = None,
    powermetrics_path: Path = _DEFAULT_POWERMETRICS_PATH,
    effective_uid: int | None = None,
) -> EnergySensorCapability:
    """Return a serializable collector capability without claiming a benchmark sample."""

    return select_default_energy_sensor(
        runtime_backend=runtime_backend,
        system=system,
        powercap_roots=powercap_roots,
        powermetrics_path=powermetrics_path,
        effective_uid=effective_uid,
    ).capability


def begin_energy_measurement(
    *,
    enabled: bool,
    sensor: EnergySensor | None,
    unavailable_reason: str | None = None,
    clock: Clock = time.perf_counter,
) -> EnergySession:
    """Take the cumulative start sample when explicitly enabled and available."""

    if not enabled:
        return EnergySession(
            sensor=None,
            start_joules=None,
            unavailable_reason="energy measurement disabled by configuration",
        )
    if sensor is None:
        return EnergySession(
            sensor=None,
            start_joules=None,
            unavailable_reason=(unavailable_reason or "energy sensor unavailable for this runtime"),
        )
    started_at = float(clock())
    if not math.isfinite(started_at):
        raise BenchmarkError(
            "energy interval clock returned a non-finite start",
            component="benchmark_energy",
        )
    try:
        start = _valid_sensor_reading(sensor.read_joules(), sensor.name)
    except (OSError, RuntimeError, ValueError) as error:
        raise BenchmarkError(
            f"energy sensor {sensor.name!r} failed to start: {error}",
            component="benchmark_energy",
        ) from error
    return EnergySession(
        sensor=sensor,
        start_joules=start,
        unavailable_reason=None,
        start_wraparounds=sensor.observed_wraparounds,
        started_at_seconds=started_at,
    )


def finish_energy_measurement(
    session: EnergySession,
    *,
    sampling_interval_seconds: float | None = None,
    clock: Clock = time.perf_counter,
) -> EnergyMeasurement:
    """Finish a cumulative energy sample or preserve explicit unavailability."""

    if sampling_interval_seconds is not None and (
        not math.isfinite(sampling_interval_seconds) or sampling_interval_seconds < 0.0
    ):
        raise BenchmarkError(
            "energy sampling interval must be finite and non-negative",
            component="benchmark_energy",
        )
    if session.sensor is None or session.start_joules is None:
        return EnergyMeasurement(
            status="unavailable",
            reason=session.unavailable_reason or "energy sensor unavailable",
        )

    try:
        end = _valid_sensor_reading(session.sensor.read_joules(), session.sensor.name)
    except (OSError, RuntimeError, ValueError) as error:
        raise BenchmarkError(
            f"energy sensor {session.sensor.name!r} failed to finish: {error}",
            component="benchmark_energy",
        ) from error
    if sampling_interval_seconds is None:
        if session.started_at_seconds is None:
            raise BenchmarkError(
                "energy sampling session has no interval start",
                component="benchmark_energy",
            )
        finished_at = float(clock())
        sampling_interval_seconds = finished_at - session.started_at_seconds
        if not math.isfinite(finished_at) or sampling_interval_seconds < 0.0:
            raise BenchmarkError(
                "energy interval clock returned an invalid sampling interval",
                component="benchmark_energy",
            )
    delta = end - session.start_joules
    if delta < 0.0:
        raise BenchmarkError(
            f"energy sensor {session.sensor.name!r} decreased during sampling",
            component="benchmark_energy",
        )
    return EnergyMeasurement(
        status="available",
        joules=delta,
        sensor_name=session.sensor.name,
        sampling_interval_seconds=sampling_interval_seconds,
        observed_counter_wraparounds=(
            session.sensor.observed_wraparounds - session.start_wraparounds
        ),
        provenance=session.sensor.provenance,
    )


def _read_text(path: Path, label: str) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"{label} at {path} is empty")
    return value


def _read_nonnegative_integer(path: Path, label: str) -> int:
    raw = _read_text(path, label)
    try:
        value = int(raw, 10)
    except ValueError as error:
        raise ValueError(f"{label} at {path} is not a base-10 integer") from error
    if value < 0:
        raise ValueError(f"{label} at {path} is negative")
    return value


def _valid_sensor_reading(value: float, sensor_name: str) -> float:
    measured = float(value)
    if not math.isfinite(measured) or measured < 0.0:
        raise BenchmarkError(
            f"energy sensor {sensor_name!r} returned an invalid cumulative reading",
            component="benchmark_energy",
        )
    return measured


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    return int(getter()) if getter is not None else -1


def _unavailable_selection(reason: str) -> EnergySensorSelection:
    return EnergySensorSelection(
        sensor=None,
        capability=EnergySensorCapability(status="unavailable", reason=reason),
    )


__all__ = [
    "EnergyDomainProvenance",
    "EnergyMeasurement",
    "EnergySensor",
    "EnergySensorCapability",
    "EnergySensorProvenance",
    "EnergySensorSelection",
    "EnergySession",
    "LinuxPowercapRaplEnergySensor",
    "begin_energy_measurement",
    "finish_energy_measurement",
    "probe_default_energy_sensor",
    "select_default_energy_sensor",
]
