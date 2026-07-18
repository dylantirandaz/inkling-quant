"""Deterministic fake runtime for contract, ordering, and benchmark tests."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

from inkling_quant_lab.config import RuntimeConfig
from inkling_quant_lab.runtimes.base import MemorySnapshot, RuntimeCapabilities


@dataclass(slots=True)
class FakeRuntime:
    """Configurable runtime that records every synchronization and cleanup."""

    available: bool = True
    supports_energy: bool = False
    host_memory_bytes: int = 1024
    device_memory_bytes: int | None = None
    host_memory_measurement_kind: str = "synthetic_test_reading"
    host_memory_scope: str = "configured fake-runtime sampling boundary"
    events: list[str] = field(default_factory=list)
    timing_values: list[float] = field(default_factory=list)

    def probe(self, config: RuntimeConfig) -> RuntimeCapabilities:
        """Return configured fake capabilities."""

        self.events.append("probe")
        return RuntimeCapabilities(
            backend="fake",
            available=self.available,
            devices=(config.device,) if self.available else (),
            supported_dtypes=("float32", "int8", "int4"),
            supports_routing_hooks=True,
            supports_forward_loss=True,
            supports_memory_measurement=True,
            supports_energy_measurement=self.supports_energy,
            supports_sharding=True,
            version="fake-v1",
            reasons=() if self.available else ("fake runtime configured unavailable",),
        )

    def execution_context(self) -> AbstractContextManager[Any]:
        """Return a no-op execution context while recording entry intent."""

        self.events.append("execution_context")
        return nullcontext()

    def synchronize(self) -> None:
        """Record a synchronization boundary."""

        self.events.append("synchronize")

    def memory_snapshot(self) -> MemorySnapshot:
        """Return deterministic configured memory values."""

        self.events.append("memory_snapshot")
        return MemorySnapshot(
            host_bytes=self.host_memory_bytes,
            device_bytes=self.device_memory_bytes,
            host_available=True,
            device_available=self.device_memory_bytes is not None,
            host_measurement_kind=self.host_memory_measurement_kind,
            host_scope=self.host_memory_scope,
            device_measurement_kind=(
                "synthetic_test_reading" if self.device_memory_bytes is not None else None
            ),
            device_scope=(
                "configured fake-runtime post-trial sample"
                if self.device_memory_bytes is not None
                else None
            ),
        )

    def cleanup(self) -> None:
        """Record cleanup."""

        self.events.append("cleanup")

    def clock(self) -> float:
        """Pop a deterministic clock value for benchmark tests."""

        if not self.timing_values:
            raise RuntimeError("fake timing sequence exhausted")
        return self.timing_values.pop(0)
