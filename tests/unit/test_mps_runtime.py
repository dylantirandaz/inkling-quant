"""Contract tests for the eager Apple MPS runtime."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from inkling_quant_lab.config import RuntimeConfig
from inkling_quant_lab.runtimes.torch_mps import TorchEagerMPSRuntime

pytestmark = pytest.mark.unit


def _config(*, device: str = "mps") -> RuntimeConfig:
    return RuntimeConfig.model_validate(
        {
            "backend": "torch_eager_mps",
            "device": device,
            "dtype": "float32",
            "device_map": "single",
        }
    )


def test_probe_reports_hardware_and_configuration_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    runtime = TorchEagerMPSRuntime()

    capabilities = runtime.probe(_config(device="cpu"))

    assert capabilities.available is False
    assert capabilities.devices == ()
    assert capabilities.supports_memory_measurement is False
    assert any("not built" in reason for reason in capabilities.reasons)
    assert any("runtime.device=mps" in reason for reason in capabilities.reasons)


def test_probe_reports_available_mps_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    runtime = TorchEagerMPSRuntime()

    capabilities = runtime.probe(_config())

    assert capabilities.available is True
    assert capabilities.devices == ("mps",)
    assert capabilities.supported_dtypes == ("float32", "float16")
    assert capabilities.supports_routing_hooks is True
    assert capabilities.supports_memory_measurement is True
    assert capabilities.supports_energy_measurement is False
    assert capabilities.supports_sharding is False
    assert capabilities.reasons == ()


def test_sampled_mps_memory_has_explicit_non_peak_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 12_345)
    runtime = TorchEagerMPSRuntime()

    snapshot = runtime.memory_snapshot()

    assert snapshot.device_bytes == 12_345
    assert snapshot.device_available is True
    assert snapshot.device_measurement_kind == "mps_driver_allocated_memory_at_sample"
    assert snapshot.device_scope is not None
    assert "not an interval peak" in snapshot.device_scope


def test_synchronize_and_cleanup_delegate_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: events.append("empty_cache"))
    runtime = TorchEagerMPSRuntime()

    runtime.synchronize()
    runtime.cleanup()

    assert events == ["synchronize", "synchronize", "empty_cache"]


def test_place_module_uses_mps_device() -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class RecordingModule(nn.Module):
        def to(self, *args: Any, **kwargs: Any) -> RecordingModule:
            calls.append((args, kwargs))
            return self

    runtime = TorchEagerMPSRuntime()

    placed = runtime.place_module(RecordingModule())

    assert isinstance(placed, RecordingModule)
    assert calls == [((), {"device": torch.device("mps")})]
