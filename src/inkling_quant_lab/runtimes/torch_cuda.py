"""Eager single-device PyTorch CUDA runtime."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, TypeVar

import torch
from torch import Tensor, nn

from inkling_quant_lab.benchmarking.memory import current_process_rss_bytes
from inkling_quant_lab.config import RuntimeConfig
from inkling_quant_lab.runtimes.base import MemorySnapshot, RuntimeCapabilities

_ModuleT = TypeVar("_ModuleT", bound=nn.Module)


class TorchEagerCUDARuntime:
    """Single-device eager inference on the default CUDA device."""

    name = "torch_eager_cuda"
    device = torch.device("cuda")

    def probe(self, config: RuntimeConfig) -> RuntimeCapabilities:
        """Report CUDA build, device, dtype, and placement support without model loading."""

        built = torch.version.cuda is not None
        available = bool(torch.cuda.is_available())
        bf16 = available and bool(torch.cuda.is_bf16_supported())
        supported_dtypes = ("float32", "float16", *(("bfloat16",) if bf16 else ()))
        reasons: list[str] = []
        if not built:
            reasons.append("this PyTorch build was not built with CUDA support")
        elif not available:
            reasons.append("PyTorch CUDA is built but no working CUDA device is available")
        if config.device != "cuda":
            reasons.append("torch_eager_cuda requires runtime.device=cuda")
        if config.dtype not in supported_dtypes:
            reasons.append(
                "torch_eager_cuda does not support the requested runtime dtype on this device"
            )
        if config.device_map != "single":
            reasons.append("torch_eager_cuda requires runtime.device_map=single")
        if config.sharding is not None:
            reasons.append("torch_eager_cuda does not support sharding")
        return RuntimeCapabilities(
            backend=self.name,
            available=available,
            devices=("cuda",) if available else (),
            supported_dtypes=supported_dtypes,
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
        """Move a model to the default CUDA device."""

        return module.to(device=self.device)

    def place_tensor(self, tensor: Tensor) -> Tensor:
        """Move an input tensor to the default CUDA device."""

        return tensor.to(device=self.device)

    def synchronize(self) -> None:
        """Wait for queued CUDA kernels at a benchmark boundary."""

        torch.cuda.synchronize()

    def memory_snapshot(self) -> MemorySnapshot:
        """Read current host RSS and the CUDA allocator peak since the prior sample."""

        try:
            host_bytes = current_process_rss_bytes()
        except OSError:
            host_bytes = None
        if not torch.cuda.is_available():
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
        device_bytes = int(torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        return MemorySnapshot(
            host_bytes=host_bytes,
            device_bytes=device_bytes,
            host_available=host_bytes is not None,
            device_available=True,
            host_measurement_kind=("process_current_rss" if host_bytes is not None else None),
            host_scope=(
                "current Python-process resident set at this sampling boundary"
                if host_bytes is not None
                else None
            ),
            device_measurement_kind="cuda_allocator_peak_since_previous_sample",
            device_scope=(
                "maximum live tensor bytes reported by the CUDA caching allocator since the "
                "previous sampling boundary; excludes non-allocator driver and library memory"
            ),
        )

    def cleanup(self) -> None:
        """Synchronize outstanding work and release unused CUDA cache blocks."""

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def create_runtime() -> TorchEagerCUDARuntime:
    """Registry factory for eager CUDA execution."""

    return TorchEagerCUDARuntime()
