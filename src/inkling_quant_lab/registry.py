"""Thread-safe lazy registries that do not eagerly import optional backends."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from inkling_quant_lab.exceptions import CapabilityError

if TYPE_CHECKING:
    from inkling_quant_lab.evaluation.base import Evaluator
    from inkling_quant_lab.models.base import ModelAdapter
    from inkling_quant_lab.quantization.base import Quantizer
    from inkling_quant_lab.reporting.report import Reporter
    from inkling_quant_lab.runtimes.base import RuntimeBackend

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RegistryDescriptor:
    """Safe metadata returned without loading a component."""

    name: str
    description: str
    optional_extra: str | None
    available: bool | None


@dataclass(frozen=True, slots=True)
class _Entry(Generic[T]):
    name: str
    description: str
    optional_extra: str | None
    available: bool | None
    factory: Callable[..., T] | None = None
    module: str | None = None
    attribute: str | None = None


class Registry(Generic[T]):
    """Map stable names to direct or lazily imported factories."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, _Entry[T]] = {}
        self._lock = RLock()

    def register(
        self,
        name: str,
        factory: Callable[..., T],
        *,
        description: str,
        optional_extra: str | None = None,
        available: bool | None = True,
    ) -> None:
        """Register an in-process factory."""

        self._add(
            _Entry(
                name=name,
                factory=factory,
                description=description,
                optional_extra=optional_extra,
                available=available,
            )
        )

    def register_lazy(
        self,
        name: str,
        module: str,
        attribute: str,
        *,
        description: str,
        optional_extra: str | None = None,
        available: bool | None = None,
    ) -> None:
        """Register an import target without importing it."""

        self._add(
            _Entry(
                name=name,
                module=module,
                attribute=attribute,
                description=description,
                optional_extra=optional_extra,
                available=available,
            )
        )

    def _add(self, entry: _Entry[T]) -> None:
        if not entry.name or any(character.isspace() for character in entry.name):
            raise ValueError("registry names must be non-empty and contain no whitespace")
        with self._lock:
            if entry.name in self._entries:
                raise ValueError(f"Duplicate {self.kind} registration: {entry.name}")
            self._entries[entry.name] = entry

    def descriptors(self) -> tuple[RegistryDescriptor, ...]:
        """List sorted component metadata without resolving lazy imports."""

        with self._lock:
            return tuple(
                RegistryDescriptor(
                    name=entry.name,
                    description=entry.description,
                    optional_extra=entry.optional_extra,
                    available=entry.available,
                )
                for entry in sorted(self._entries.values(), key=lambda item: item.name)
            )

    def create(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Resolve one factory on demand and instantiate the component."""

        with self._lock:
            try:
                entry = self._entries[name]
            except KeyError as error:
                choices = ", ".join(sorted(self._entries)) or "none"
                raise CapabilityError(
                    f"Unknown {self.kind} '{name}'. Registered choices: {choices}",
                    component=self.kind,
                ) from error
        factory = entry.factory
        if factory is None:
            if entry.module is None or entry.attribute is None:
                raise RuntimeError(f"Invalid lazy registry entry: {name}")
            try:
                module = importlib.import_module(entry.module)
                factory = cast(Callable[..., T], getattr(module, entry.attribute))
            except (ImportError, AttributeError) as error:
                hint = (
                    f" Install with `uv sync --extra {entry.optional_extra}`."
                    if entry.optional_extra
                    else ""
                )
                raise CapabilityError(
                    f"{self.kind} '{name}' is unavailable.{hint}",
                    component=name,
                    remediation=hint.strip() or None,
                ) from error
        return factory(*args, **kwargs)

    def __contains__(self, name: object) -> bool:
        """Return whether a stable name is registered without loading it."""

        with self._lock:
            return name in self._entries


MODEL_ADAPTERS: Registry[ModelAdapter] = Registry("model adapter")
QUANTIZERS: Registry[Quantizer] = Registry("quantizer")
EVALUATORS: Registry[Evaluator] = Registry("evaluator")
RUNTIMES: Registry[RuntimeBackend] = Registry("runtime")
REPORTERS: Registry[Reporter] = Registry("reporter")
