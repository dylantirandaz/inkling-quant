"""Generic routing hooks for modules exposing normalized routing snapshots."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from inkling_quant_lab.exceptions import RoutingInstrumentationError
from inkling_quant_lab.routing.traces import BatchMeta, RoutingEvent, RoutingSink


def _as_nested_list(value: Any, *, field_name: str) -> list[Any]:
    """Convert tensor-like snapshot data to nested Python lists without NumPy."""

    if value is None:
        raise RoutingInstrumentationError(
            f"routing snapshot is missing {field_name}", component="routing_hooks"
        )
    detached = value.detach() if callable(getattr(value, "detach", None)) else value
    on_cpu = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
    converted = on_cpu.tolist() if callable(getattr(on_cpu, "tolist", None)) else on_cpu
    if not isinstance(converted, (list, tuple)):
        raise RoutingInstrumentationError(
            f"routing snapshot field {field_name} must be tensor-like",
            component="routing_hooks",
        )
    return list(converted)


def _validate_batch_sequence_shape(
    values: list[Any],
    *,
    field_name: str,
    batch_size: int,
    sequence_lengths: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    if len(values) != batch_size:
        raise RoutingInstrumentationError(
            f"{field_name} batch dimension {len(values)} does not match {batch_size} sample IDs",
            component="routing_hooks",
        )
    observed_lengths: list[int] = []
    for sequence in values:
        if not isinstance(sequence, (list, tuple)):
            raise RoutingInstrumentationError(
                f"{field_name} must have shape [batch, sequence, width]",
                component="routing_hooks",
            )
        observed_lengths.append(len(sequence))
    result = tuple(observed_lengths)
    if sequence_lengths is not None and result != sequence_lengths:
        raise RoutingInstrumentationError(
            f"{field_name} sequence dimensions do not match selected_expert_ids",
            component="routing_hooks",
        )
    return result


def _row(
    values: list[Any], batch_index: int, token_position: int, *, field_name: str
) -> tuple[Any, ...]:
    result = values[batch_index][token_position]
    if not isinstance(result, (list, tuple)):
        raise RoutingInstrumentationError(
            f"{field_name} must have shape [batch, sequence, width]",
            component="routing_hooks",
        )
    return tuple(result)


def events_from_routing_snapshot(
    snapshot: Mapping[str, Any],
    *,
    sample_ids: tuple[str, ...],
    capture_router_logits: bool = False,
    fallback_layer_id: str | None = None,
) -> tuple[RoutingEvent, ...]:
    """Normalize a module routing snapshot into token-level routing events."""

    if not sample_ids:
        raise RoutingInstrumentationError(
            "routing capture requires stable sample IDs; start a sink batch first",
            component="routing_hooks",
            remediation="call sink.start_batch(BatchMeta(sample_ids=...)) before inference",
        )
    layer_value = snapshot.get("layer_id", fallback_layer_id)
    if not isinstance(layer_value, str) or not layer_value:
        raise RoutingInstrumentationError(
            "routing snapshot is missing a non-empty layer_id",
            component="routing_hooks",
        )
    expert_ids = _as_nested_list(
        snapshot.get("selected_expert_ids"), field_name="selected_expert_ids"
    )
    weights = _as_nested_list(snapshot.get("selected_weights"), field_name="selected_weights")
    sequence_lengths = _validate_batch_sequence_shape(
        expert_ids, field_name="selected_expert_ids", batch_size=len(sample_ids)
    )
    _validate_batch_sequence_shape(
        weights,
        field_name="selected_weights",
        batch_size=len(sample_ids),
        sequence_lengths=sequence_lengths,
    )

    probabilities_value = snapshot.get("router_probabilities", snapshot.get("router_probs"))
    probabilities = (
        None
        if probabilities_value is None
        else _as_nested_list(probabilities_value, field_name="router_probabilities")
    )
    if probabilities is not None:
        _validate_batch_sequence_shape(
            probabilities,
            field_name="router_probabilities",
            batch_size=len(sample_ids),
            sequence_lengths=sequence_lengths,
        )

    logits_value = snapshot.get("router_logits") if capture_router_logits else None
    logits = (
        None if logits_value is None else _as_nested_list(logits_value, field_name="router_logits")
    )
    if logits is not None:
        _validate_batch_sequence_shape(
            logits,
            field_name="router_logits",
            batch_size=len(sample_ids),
            sequence_lengths=sequence_lengths,
        )

    events: list[RoutingEvent] = []
    for sequence_index, sample_id in enumerate(sample_ids):
        for token_position in range(sequence_lengths[sequence_index]):
            selected_expert_ids = _row(
                expert_ids,
                sequence_index,
                token_position,
                field_name="selected_expert_ids",
            )
            selected_weights = _row(
                weights,
                sequence_index,
                token_position,
                field_name="selected_weights",
            )
            router_probabilities = (
                None
                if probabilities is None
                else _row(
                    probabilities,
                    sequence_index,
                    token_position,
                    field_name="router_probabilities",
                )
            )
            router_logits = (
                None
                if logits is None
                else _row(
                    logits,
                    sequence_index,
                    token_position,
                    field_name="router_logits",
                )
            )
            try:
                events.append(
                    RoutingEvent(
                        sample_id=sample_id,
                        sequence_index=sequence_index,
                        token_position=token_position,
                        layer_id=layer_value,
                        selected_expert_ids=tuple(
                            int(expert_id) for expert_id in selected_expert_ids
                        ),
                        selected_weights=tuple(float(weight) for weight in selected_weights),
                        router_probabilities=(
                            None
                            if router_probabilities is None
                            else tuple(float(value) for value in router_probabilities)
                        ),
                        router_logits=(
                            None
                            if router_logits is None
                            else tuple(float(value) for value in router_logits)
                        ),
                    )
                )
            except (TypeError, ValueError) as error:
                raise RoutingInstrumentationError(
                    f"invalid routing snapshot for layer {layer_value}: {error}",
                    component="routing_hooks",
                ) from error
    return tuple(events)


def _tensor_field(snapshot: Mapping[str, Any], field_name: str) -> Tensor:
    value = snapshot.get(field_name)
    if not isinstance(value, Tensor):
        raise RoutingInstrumentationError(
            f"aggregate fast path requires tensor field {field_name}",
            component="routing_hooks",
        )
    # Instrumentation is intentionally reduced on CPU. This avoids retaining
    # accelerator storage and avoids requiring device support for float64
    # accumulation, while leaving model execution and routing on the device.
    return value.detach().cpu()


def _record_aggregate_snapshot(
    snapshot: Mapping[str, Any],
    *,
    sample_ids: tuple[str, ...],
    fallback_layer_id: str | None,
    sink: RoutingSink,
) -> None:
    """Reduce a tensor snapshot by batch, avoiding per-token Python events."""

    if not sample_ids:
        raise RoutingInstrumentationError(
            "routing capture requires stable sample IDs; start a sink batch first",
            component="routing_hooks",
            remediation="call sink.start_batch(BatchMeta(sample_ids=...)) before inference",
        )
    layer_value = snapshot.get("layer_id", fallback_layer_id)
    if not isinstance(layer_value, str) or not layer_value:
        raise RoutingInstrumentationError(
            "routing snapshot is missing a non-empty layer_id",
            component="routing_hooks",
        )
    expert_ids = _tensor_field(snapshot, "selected_expert_ids")
    selected_weights = _tensor_field(snapshot, "selected_weights")
    if expert_ids.ndim != 3 or tuple(expert_ids.shape) != tuple(selected_weights.shape):
        raise RoutingInstrumentationError(
            "aggregate routing tensors must share shape [batch, sequence, top_k]",
            component="routing_hooks",
        )
    if expert_ids.shape[0] != len(sample_ids) or expert_ids.shape[2] <= 0:
        raise RoutingInstrumentationError(
            "aggregate routing tensor dimensions do not match the active batch",
            component="routing_hooks",
        )
    if expert_ids.dtype == torch.bool or expert_ids.is_floating_point():
        raise RoutingInstrumentationError(
            "selected_expert_ids must use an integer tensor dtype",
            component="routing_hooks",
        )
    if not selected_weights.is_floating_point():
        selected_weights = selected_weights.to(dtype=torch.float64)
    flat_ids = expert_ids.reshape(-1).to(dtype=torch.long)
    flat_weights = selected_weights.reshape(-1).to(dtype=torch.float64)
    if flat_ids.numel() and bool(torch.any(flat_ids < 0).item()):
        raise RoutingInstrumentationError(
            "selected expert IDs must be non-negative", component="routing_hooks"
        )
    if not bool(torch.isfinite(flat_weights).all().item()) or bool(
        torch.any(flat_weights < 0).item()
    ):
        raise RoutingInstrumentationError(
            "selected_weights must be finite and non-negative", component="routing_hooks"
        )

    probabilities = snapshot.get("router_probabilities", snapshot.get("router_probs"))
    if isinstance(probabilities, Tensor) and probabilities.ndim == 3:
        expert_count = int(probabilities.shape[-1])
    else:
        expert_count = int(flat_ids.max().item()) + 1 if flat_ids.numel() else 0
    if flat_ids.numel() and int(flat_ids.max().item()) >= expert_count:
        raise RoutingInstrumentationError(
            "selected expert ID is outside router probabilities", component="routing_hooks"
        )

    counts = torch.bincount(flat_ids, minlength=expert_count)
    weight_sums = torch.zeros(
        expert_count, dtype=torch.float64, device=flat_weights.device
    ).scatter_add_(0, flat_ids, flat_weights)
    sink.record_aggregate_batch(
        layer_id=layer_value,
        sample_ids=sample_ids,
        tokens_per_sample=int(expert_ids.shape[1]),
        expert_selection_counts=tuple(int(value) for value in counts.cpu().tolist()),
        expert_weight_sums=tuple(float(value) for value in weight_sums.cpu().tolist()),
    )


def discover_routing_modules(model: Any) -> tuple[tuple[str, Any], ...]:
    """Find hookable modules that expose ``routing_snapshot()``."""

    named_modules = getattr(model, "named_modules", None)
    candidates: list[tuple[str, Any]] = []
    if callable(named_modules):
        candidates.extend((str(name), module) for name, module in named_modules())
    else:
        candidates.append(("", model))

    discovered: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for name, module in candidates:
        if id(module) in seen:
            continue
        snapshot = getattr(module, "routing_snapshot", None)
        register_hook = getattr(module, "register_forward_hook", None)
        if callable(snapshot) and callable(register_hook):
            seen.add(id(module))
            discovered.append((name, module))
    return tuple(discovered)


class RoutingHookHandle:
    """Removable generic hook collection with optional batch lifecycle helpers."""

    def __init__(
        self,
        modules: Sequence[tuple[str, Any]],
        sink: RoutingSink,
        *,
        sample_ids: tuple[str, ...] = (),
        capture_router_logits: bool = False,
    ) -> None:
        self._sink = sink
        self._sample_ids = BatchMeta(sample_ids=tuple(sample_ids)).sample_ids if sample_ids else ()
        self._capture_router_logits = capture_router_logits
        self._removed = False
        self._owns_batch = False
        self._module_handles: list[Any] = []
        try:
            for name, module in modules:
                callback = self._callback(name)
                self._module_handles.append(module.register_forward_hook(callback))
        except (AttributeError, RuntimeError, TypeError) as error:
            self.remove()
            raise RoutingInstrumentationError(
                f"failed to attach routing hook: {error}", component="routing_hooks"
            ) from error

    def _callback(self, module_name: str) -> Callable[[Any, Any, Any], None]:
        def capture(module: Any, _inputs: Any, _output: Any) -> None:
            if self._removed:
                return
            try:
                snapshot = module.routing_snapshot()
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError) as error:
                raise RoutingInstrumentationError(
                    f"routing_snapshot() failed on {module_name or '<root>'}: {error}",
                    component="routing_hooks",
                ) from error
            if snapshot is None:
                return
            if not isinstance(snapshot, Mapping):
                raise RoutingInstrumentationError(
                    f"routing_snapshot() on {module_name or '<root>'} must return a mapping",
                    component="routing_hooks",
                )
            active_batch = getattr(self._sink, "active_batch", None)
            sample_ids = (
                tuple(active_batch.sample_ids)
                if isinstance(active_batch, BatchMeta)
                else self._sample_ids
            )
            fallback_layer_id = getattr(module, "layer_id", None) or module_name or None
            if (
                getattr(self._sink, "mode", None) == "aggregate"
                and isinstance(snapshot.get("selected_expert_ids"), Tensor)
                and isinstance(snapshot.get("selected_weights"), Tensor)
            ):
                _record_aggregate_snapshot(
                    snapshot,
                    sample_ids=sample_ids,
                    fallback_layer_id=fallback_layer_id,
                    sink=self._sink,
                )
                return
            for event in events_from_routing_snapshot(
                snapshot,
                sample_ids=sample_ids,
                capture_router_logits=self._capture_router_logits,
                fallback_layer_id=fallback_layer_id,
            ):
                self._sink.record(event)

        return capture

    def start_batch(self, sample_ids: tuple[str, ...], *, batch_id: str | None = None) -> None:
        """Start a sink-owned batch and use its stable IDs for subsequent callbacks."""

        if self._removed:
            raise RoutingInstrumentationError(
                "cannot start a batch after routing hooks are removed",
                component="routing_hooks",
            )
        if self._owns_batch:
            raise RoutingInstrumentationError(
                "routing hook handle already owns an active batch",
                component="routing_hooks",
            )
        self._sample_ids = tuple(sample_ids)
        self._sink.start_batch(BatchMeta(sample_ids=self._sample_ids, batch_id=batch_id))
        self._owns_batch = True

    def end_batch(self) -> None:
        """End a batch opened through this handle."""

        if not self._owns_batch:
            raise RoutingInstrumentationError(
                "routing hook handle does not own an active batch",
                component="routing_hooks",
            )
        self._sink.end_batch()
        self._owns_batch = False

    def set_sample_ids(self, sample_ids: tuple[str, ...]) -> None:
        """Update fallback sample IDs when another component owns sink batching."""

        if self._removed:
            raise RoutingInstrumentationError(
                "cannot update sample IDs after routing hooks are removed",
                component="routing_hooks",
            )
        self._sample_ids = BatchMeta(sample_ids=tuple(sample_ids)).sample_ids

    def remove(self) -> None:
        """Remove all hooks idempotently so later inference records no events."""

        if self._removed:
            return
        for handle in self._module_handles:
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()
        self._module_handles.clear()
        if self._owns_batch:
            self._sink.end_batch()
            self._owns_batch = False
        self._removed = True

    def __enter__(self) -> RoutingHookHandle:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.remove()


def attach_routing_hooks(
    model: Any,
    sink: RoutingSink,
    *,
    sample_ids: tuple[str, ...] = (),
    capture_router_logits: bool = False,
) -> RoutingHookHandle:
    """Attach hooks to all snapshot-capable modules in ``model``.

    Sample IDs may be supplied here, set later on the handle, or provided by an
    active batch on :class:`~inkling_quant_lab.routing.traces.InMemoryRoutingSink`.
    """

    modules = discover_routing_modules(model)
    if not modules:
        raise RoutingInstrumentationError(
            "model exposes no hookable routing_snapshot() modules",
            component="routing_hooks",
            remediation="use an architecture adapter or mark routing unsupported",
        )
    return RoutingHookHandle(
        modules,
        sink,
        sample_ids=sample_ids,
        capture_router_logits=capture_router_logits,
    )


attach_routing_snapshot_hooks = attach_routing_hooks
