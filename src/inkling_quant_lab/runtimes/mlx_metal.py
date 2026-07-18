"""Capability-gated MLX runtime for single-device Apple Metal execution."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

from inkling_quant_lab.benchmarking.memory import current_process_rss_bytes
from inkling_quant_lab.config import RuntimeConfig
from inkling_quant_lab.mlx_contract import mlx_environment_status
from inkling_quant_lab.runtimes.base import MemorySnapshot, RuntimeCapabilities


class MLXMetalRuntime:
    """Single-process MLX execution on the Apple unified-memory Metal device."""

    name = "mlx_metal"

    def __init__(self) -> None:
        self._mx: Any | None = None

    def probe(self, config: RuntimeConfig) -> RuntimeCapabilities:
        """Probe exact dependencies and a real Metal device without loading weights."""

        status = mlx_environment_status()
        reasons = list(status.reasons)
        available = status.available
        if config.device != "mps":
            reasons.append("mlx_metal requires runtime.device=mps")
        if config.dtype != "float32":
            reasons.append("the validated MLX matrix requires runtime.dtype=float32")
        if config.device_map != "single":
            reasons.append("mlx_metal requires runtime.device_map=single")
        if config.sharding is not None:
            reasons.append("the registered mlx_metal runtime does not support sharding")
        if status.available:
            try:
                import mlx.core as mx

                self._mx = mx
                available = bool(mx.metal.is_available())
                if not available:
                    reasons.append("MLX imports but reports no available Metal device")
            except (ImportError, RuntimeError) as error:
                available = False
                reasons.append(f"MLX Metal probe failed: {error}")
        version = ";".join(f"{name}={value}" for name, value in sorted(status.versions.items()))
        return RuntimeCapabilities(
            backend=self.name,
            available=available,
            devices=("mps",) if available else (),
            supported_dtypes=("float32",),
            supports_routing_hooks=True,
            supports_forward_loss=True,
            supports_memory_measurement=available,
            supports_energy_measurement=False,
            supports_sharding=False,
            version=version or "unavailable",
            reasons=tuple(reasons),
        )

    def execution_context(self) -> AbstractContextManager[Any]:
        """MLX inference has no torch-style context manager requirement."""

        return nullcontext()

    def synchronize(self) -> None:
        """Wait for every queued Metal kernel."""

        if self._mx is None:
            import mlx.core as mx

            self._mx = mx
        self._mx.synchronize()

    def memory_snapshot(self) -> MemorySnapshot:
        """Sample current host RSS and current MLX allocator residency."""

        try:
            host_bytes = current_process_rss_bytes()
        except OSError:
            host_bytes = None
        if self._mx is None:
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
            device_bytes=int(self._mx.get_active_memory()),
            host_available=host_bytes is not None,
            device_available=True,
            host_measurement_kind=("process_current_rss" if host_bytes is not None else None),
            host_scope=(
                "current Python-process resident set at this sampling boundary"
                if host_bytes is not None
                else None
            ),
            device_measurement_kind="mlx_allocator_active_bytes_at_sample",
            device_scope=(
                "current process MLX allocator active bytes in Apple unified memory at this "
                "sampling boundary; not whole-device use and not an interval peak"
            ),
        )

    def cleanup(self) -> None:
        """Drain Metal work and release unused MLX allocator cache blocks."""

        if self._mx is not None:
            self._mx.synchronize()
            self._mx.clear_cache()


def create_runtime() -> MLXMetalRuntime:
    """Lazy registry factory."""

    return MLXMetalRuntime()


__all__ = ["MLXMetalRuntime", "create_runtime"]
