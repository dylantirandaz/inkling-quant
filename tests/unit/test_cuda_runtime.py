"""Contract tests for the eager CUDA runtime without requiring a GPU."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from inkling_quant_lab.config import RuntimeConfig
from inkling_quant_lab.runtimes.torch_cuda import TorchEagerCUDARuntime

pytestmark = pytest.mark.unit


def _config(*, device: str = "cuda", dtype: str = "float32") -> RuntimeConfig:
    return RuntimeConfig.model_validate(
        {
            "backend": "torch_eager_cuda",
            "device": device,
            "dtype": dtype,
            "device_map": "single",
        }
    )


def test_probe_reports_unavailable_cuda_without_allocating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    capabilities = TorchEagerCUDARuntime().probe(_config())

    assert capabilities.available is False
    assert capabilities.devices == ()
    assert capabilities.supported_dtypes == ("float32", "float16")
    assert any("not built" in reason for reason in capabilities.reasons)


def test_probe_reports_device_bf16_and_incompatible_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    capabilities = TorchEagerCUDARuntime().probe(_config(device="cpu", dtype="bfloat16"))

    assert capabilities.available is True
    assert capabilities.devices == ("cuda",)
    assert capabilities.supported_dtypes == ("float32", "float16", "bfloat16")
    assert any("runtime.device=cuda" in reason for reason in capabilities.reasons)


def test_memory_snapshot_reads_and_resets_cuda_allocator_peak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 98_765)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: events.append("reset"))

    snapshot = TorchEagerCUDARuntime().memory_snapshot()

    assert snapshot.device_bytes == 98_765
    assert snapshot.device_measurement_kind == "cuda_allocator_peak_since_previous_sample"
    assert snapshot.device_scope is not None and "excludes" in snapshot.device_scope
    assert events == ["reset"]


def test_synchronize_cleanup_and_placement_delegate_to_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class RecordingModule(nn.Module):
        def to(self, *args: Any, **kwargs: Any) -> RecordingModule:
            calls.append((args, kwargs))
            return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    runtime = TorchEagerCUDARuntime()

    placed = runtime.place_module(RecordingModule())
    runtime.synchronize()
    runtime.cleanup()

    assert isinstance(placed, RecordingModule)
    assert calls == [((), {"device": torch.device("cuda")})]
    assert events == ["synchronize", "synchronize", "empty_cache"]
