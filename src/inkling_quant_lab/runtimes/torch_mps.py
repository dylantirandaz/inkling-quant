"""Eager PyTorch runtime for Apple Metal Performance Shaders."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, TypeVar

import torch
from torch import Tensor, nn

from inkling_quant_lab.benchmarking.memory import current_process_rss_bytes
from inkling_quant_lab.config import RuntimeConfig
from inkling_quant_lab.runtimes.base import MemorySnapshot, RuntimeCapabilities

_ModuleT = TypeVar("_ModuleT", bound=nn.Module)


class TorchEagerMPSRuntime:
    """Single-device eager inference on an available Apple MPS accelerator."""

    name = "torch_eager_mps"
    device = torch.device("mps")

    def probe(self, config: RuntimeConfig) -> RuntimeCapabilities:
        """Report build, driver, dtype, and placement support without allocating a model."""

        built = bool(torch.backends.mps.is_built())
        available = bool(torch.backends.mps.is_available())
        reasons: list[str] = []
        if not built:
            reasons.append("this PyTorch build was not built with MPS support")
        elif not available:
            reasons.append("PyTorch MPS is built but no working MPS device is available")
        if config.device != "mps":
            reasons.append("torch_eager_mps requires runtime.device=mps")
        if config.dtype not in {"float32", "float16"}:
            reasons.append("torch_eager_mps supports runtime.dtype float32 or float16")
        if config.device_map != "single":
            reasons.append("torch_eager_mps requires runtime.device_map=single")
        if config.sharding is not None:
            reasons.append("torch_eager_mps does not support sharding")
        return RuntimeCapabilities(
            backend=self.name,
            available=available,
            devices=("mps",) if available else (),
            supported_dtypes=("float32", "float16"),
            supports_routing_hooks=True,
            supports_forward_loss=True,
            supports_memory_measurement=available,
            supports_energy_measurement=False,
            supports_sharding=False,
            version=torch.__version__,
            reasons=tuple(reasons),
        )

    def execution_context(self) -> AbstractContextManager[Any]:
        """Return an inference-only execution context."""

        return torch.inference_mode()

    def place_module(self, module: _ModuleT) -> _ModuleT:
        """Move a model to the accelerator selected by this runtime."""

        return module.to(device=self.device)

    def place_tensor(self, tensor: Tensor) -> Tensor:
        """Move an input tensor to the accelerator selected by this runtime."""

        return tensor.to(device=self.device)

    def synchronize(self) -> None:
        """Wait for all queued MPS kernels before a timing boundary."""

        torch.mps.synchronize()

    def memory_snapshot(self) -> MemorySnapshot:
        """Sample MPS process allocation with explicit non-peak semantics."""

        try:
            host_bytes = current_process_rss_bytes()
        except OSError:
            host_bytes = None
        if not torch.backends.mps.is_available():
            return MemorySnapshot(
                host_bytes=host_bytes,
                device_bytes=None,
                host_available=host_bytes is not None,
                device_available=False,
                host_measurement_kind=("process_current_rss" if host_bytes is not None else None),
                host_scope=(
                    "current Python-process resident set at this sampling boundary"
                    if host_bytes is not None
                    else None
                ),
            )
        return MemorySnapshot(
            host_bytes=host_bytes,
            device_bytes=int(torch.mps.driver_allocated_memory()),
            host_available=host_bytes is not None,
            device_available=True,
            host_measurement_kind=("process_current_rss" if host_bytes is not None else None),
            host_scope=(
                "current Python-process resident set at this sampling boundary"
                if host_bytes is not None
                else None
            ),
            device_measurement_kind="mps_driver_allocated_memory_at_sample",
            device_scope=(
                "current Metal-driver allocation for this process at the sampling boundary; "
                "includes allocator caches and framework allocations and is not an interval peak"
            ),
        )

    def cleanup(self) -> None:
        """Synchronize outstanding kernels and release unused MPS allocator cache blocks."""

        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()


def create_runtime() -> TorchEagerMPSRuntime:
    """Registry factory for the Apple MPS runtime."""

    return TorchEagerMPSRuntime()
