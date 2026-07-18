"""Linux powercap energy discovery, rollover, provenance, and wiring contracts."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from inkling_quant_lab.benchmarking.energy import (
    EnergyMeasurement,
    EnergySensorSelection,
    LinuxPowercapRaplEnergySensor,
    probe_default_energy_sensor,
    select_default_energy_sensor,
)
from inkling_quant_lab.models.local import LocalFixtureAdapter
from inkling_quant_lab.pipeline.operations import Components, benchmark_model
from inkling_quant_lab.quantization.reference import NoopQuantizer
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.unit


def _write(path: Path, value: str | int) -> None:
    path.write_text(f"{value}\n", encoding="utf-8")


def _package_zone(
    root: Path,
    index: int,
    *,
    energy_uj: int,
    maximum_uj: int = 1_000,
) -> Path:
    zone = root / "intel-rapl" / f"intel-rapl:{index}"
    zone.mkdir(parents=True)
    _write(zone / "name", f"package-{index}")
    _write(zone / "energy_uj", energy_uj)
    _write(zone / "max_energy_range_uj", maximum_uj)
    return zone


def test_linux_powercap_aggregates_only_top_level_packages_with_provenance(
    tmp_path: Path,
) -> None:
    first = _package_zone(tmp_path, 0, energy_uj=100)
    second = _package_zone(tmp_path, 1, energy_uj=400)
    subzone = first / "intel-rapl:0:0"
    subzone.mkdir()
    _write(subzone / "name", "core")
    _write(subzone / "energy_uj", 900)
    _write(subzone / "max_energy_range_uj", 1_000)

    sensor = LinuxPowercapRaplEnergySensor.from_powercap_root(tmp_path)

    assert sensor.read_joules() == 0.0
    _write(first / "energy_uj", 150)
    _write(second / "energy_uj", 450)
    _write(subzone / "energy_uj", 999)
    assert sensor.read_joules() == pytest.approx(100 / 1_000_000)
    assert tuple(domain.domain_name for domain in sensor.provenance.domains) == (
        "package-0",
        "package-1",
    )
    assert all("intel-rapl:0:0" not in domain.counter_path for domain in sensor.provenance.domains)
    assert "entire host" in sensor.provenance.scope
    assert "not process-attributable" in sensor.provenance.scope


def test_linux_powercap_handles_independent_package_counter_rollover(tmp_path: Path) -> None:
    first = _package_zone(tmp_path, 0, energy_uj=990)
    second = _package_zone(tmp_path, 1, energy_uj=800)
    sensor = LinuxPowercapRaplEnergySensor.from_powercap_root(tmp_path)

    assert sensor.read_joules() == 0.0
    _write(first / "energy_uj", 10)
    _write(second / "energy_uj", 900)

    assert sensor.read_joules() == pytest.approx(120 / 1_000_000)
    assert sensor.observed_wraparounds == 1
    assert "max_energy_range_uj" in sensor.provenance.counter_wraparound_handling


@pytest.mark.parametrize(
    ("filename", "value", "message"),
    (
        ("name", "", "empty"),
        ("max_energy_range_uj", "not-an-integer", "not a base-10 integer"),
        ("energy_uj", "1001", "exceeds max_energy_range_uj"),
    ),
)
def test_linux_powercap_probe_fails_closed_for_malformed_package_domains(
    tmp_path: Path,
    filename: str,
    value: str,
    message: str,
) -> None:
    zone = _package_zone(tmp_path, 0, energy_uj=100)
    _write(zone / filename, value)

    selection = select_default_energy_sensor(
        runtime_backend="torch_eager_cpu",
        system="Linux",
        powercap_roots=(tmp_path,),
    )

    assert selection.available is False
    assert selection.sensor is None
    assert selection.capability.status == "unavailable"
    assert message in (selection.capability.reason or "")


def test_linux_powercap_probe_requires_package_domains(tmp_path: Path) -> None:
    control = tmp_path / "intel-rapl"
    control.mkdir()

    capability = probe_default_energy_sensor(
        system="Linux",
        powercap_roots=(tmp_path,),
    )

    assert capability.status == "unavailable"
    assert capability.provenance is None
    assert "no top-level" in (capability.reason or "")


def test_darwin_powermetrics_privilege_blocker_never_claims_joules(tmp_path: Path) -> None:
    executable = tmp_path / "powermetrics"
    _write(executable, "#!/bin/sh")
    executable.chmod(0o755)

    capability = probe_default_energy_sensor(
        system="Darwin",
        powermetrics_path=executable,
        effective_uid=501,
    )

    assert capability.status == "unavailable"
    assert capability.provenance is None
    assert "superuser" in (capability.reason or "")
    assert "not accepted as joules" in (capability.reason or "")


@pytest.mark.skipif(platform.system() != "Darwin", reason="installed Apple interface probe")
def test_installed_apple_energy_interface_is_truthfully_unavailable() -> None:
    capability = probe_default_energy_sensor(system="Darwin")

    assert capability.status == "unavailable"
    assert capability.provenance is None
    assert "joule" in (capability.reason or "")
    if getattr(os, "geteuid", lambda: -1)() != 0:
        assert "superuser" in (capability.reason or "")


def test_available_energy_measurement_requires_full_sensor_provenance() -> None:
    with pytest.raises(ValidationError, match="all measured facts"):
        EnergyMeasurement(
            status="available",
            joules=1.0,
            sensor_name="incomplete",
            sampling_interval_seconds=1.0,
            observed_counter_wraparounds=0,
        )


def test_pipeline_benchmark_selects_supported_default_energy_sensor(
    tmp_path: Path,
    config_factory: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _package_zone(tmp_path, 0, energy_uj=100)
    selection = select_default_energy_sensor(
        runtime_backend="torch_eager_cpu",
        system="Linux",
        powercap_roots=(tmp_path,),
    )
    assert selection.available
    captured: dict[str, Any] = {}

    def select(*, runtime_backend: str) -> EnergySensorSelection:
        assert runtime_backend == "torch_eager_cpu"
        return selection

    def run(*args: Any, **kwargs: Any) -> object:
        del args
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "inkling_quant_lab.pipeline.operations.select_default_energy_sensor", select
    )
    monkeypatch.setattr("inkling_quant_lab.pipeline.operations.run_generation_benchmark", run)
    config = config_factory(routing_mode="off")
    config = config.model_copy(
        update={
            "benchmark": config.benchmark.model_copy(update={"measure_energy": True}),
        }
    )
    adapter = LocalFixtureAdapter()
    runtime = TorchEagerCPURuntime()
    loaded = adapter.load(config, runtime)
    components = Components(adapter=adapter, runtime=runtime, quantizer=NoopQuantizer())

    result = benchmark_model(config, components, loaded, serialized_size_bytes=1)

    assert result is not None
    assert captured["energy_sensor"] is selection.sensor
    assert captured["energy_unavailable_reason"] is None
    assert cast(LinuxPowercapRaplEnergySensor, captured["energy_sensor"]).provenance.scope
