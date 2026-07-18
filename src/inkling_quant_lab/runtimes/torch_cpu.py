"""Portable eager PyTorch CPU runtime."""

from __future__ import annotations

import platform
from contextlib import AbstractContextManager
from typing import Any

import torch

from inkling_quant_lab.benchmarking.energy import select_default_energy_sensor
from inkling_quant_lab.benchmarking.memory import current_process_rss_bytes
from inkling_quant_lab.config import RuntimeConfig
from inkling_quant_lab.runtimes.base import MemorySnapshot, RuntimeCapabilities

_CURRENT_RSS_KIND = "process_current_rss"
_CURRENT_RSS_SCOPE = "instantaneous Python process resident set size at the sampling boundary"


class TorchEagerCPURuntime:
    """Local eager execution with explicit CPU-only capabilities."""

    name = "torch_eager_cpu"

    def probe(self, config: RuntimeConfig) -> RuntimeCapabilities:
        """Report support without allocating a model."""

        supported = config.device == "cpu"
        reasons = () if supported else ("torch_eager_cpu requires runtime.device=cpu",)
        energy = select_default_energy_sensor(runtime_backend=self.name)
        return RuntimeCapabilities(
            backend=self.name,
            available=True,
            devices=("cpu",),
            supported_dtypes=("float32", "float16", "bfloat16"),
            supports_routing_hooks=True,
            supports_forward_loss=True,
            supports_memory_measurement=platform.system() in {"Darwin", "Linux"},
            supports_energy_measurement=energy.available,
            supports_sharding=False,
            version=torch.__version__,
            reasons=reasons,
        )

    def execution_context(self) -> AbstractContextManager[Any]:
        """Return an inference-only execution context."""

        return torch.inference_mode()

    def synchronize(self) -> None:
        """CPU eager operations are synchronous; this is a documented no-op."""

    def memory_snapshot(self) -> MemorySnapshot:
        """Return dependency-free current RSS for an explicitly sampled interval."""

        try:
            rss = current_process_rss_bytes()
        except (OSError, ValueError):
            return MemorySnapshot(
                host_bytes=None,
                device_bytes=None,
                host_available=False,
                device_available=False,
            )
        return MemorySnapshot(
            host_bytes=rss,
            device_bytes=None,
            host_available=True,
            device_available=False,
            host_measurement_kind=_CURRENT_RSS_KIND,
            host_scope=_CURRENT_RSS_SCOPE,
        )

    def cleanup(self) -> None:
        """Release runtime-owned caches; CPU eager owns none."""


def create_runtime() -> TorchEagerCPURuntime:
    """Registry factory for the CPU runtime."""

    return TorchEagerCPURuntime()


__all__ = ["TorchEagerCPURuntime", "create_runtime"]
