"""Runtime protocol and capability records."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from inkling_quant_lab.config import RuntimeConfig


@dataclass(frozen=True, slots=True)
class RuntimeCapabilities:
    """Factual runtime support report."""

    backend: str
    available: bool
    devices: tuple[str, ...]
    supported_dtypes: tuple[str, ...]
    supports_routing_hooks: bool
    supports_forward_loss: bool
    supports_memory_measurement: bool
    supports_energy_measurement: bool
    supports_sharding: bool
    version: str
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Host/device memory readings with explicit collector semantics."""

    host_bytes: int | None
    device_bytes: int | None
    host_available: bool
    device_available: bool
    host_measurement_kind: str | None = None
    host_scope: str | None = None
    device_measurement_kind: str | None = None
    device_scope: str | None = None


class RuntimeBackend(Protocol):
    """Device placement and timing synchronization boundary."""

    def probe(self, config: RuntimeConfig) -> RuntimeCapabilities: ...

    def execution_context(self) -> AbstractContextManager[Any]: ...

    def synchronize(self) -> None: ...

    def memory_snapshot(self) -> MemorySnapshot: ...

    def cleanup(self) -> None: ...
