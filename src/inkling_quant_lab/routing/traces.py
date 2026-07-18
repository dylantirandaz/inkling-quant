"""Routing event validation, in-memory aggregation, and trace persistence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from inkling_quant_lab.exceptions import RoutingInstrumentationError

RoutingMode = Literal["off", "aggregate", "sampled_tokens", "full_trace"]


def _finite_tuple(values: Sequence[float], *, field_name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{field_name} must contain only finite values")
    return result


@dataclass(frozen=True, slots=True)
class BatchMeta:
    """Stable identity for the samples participating in one inference batch."""

    sample_ids: tuple[str, ...]
    batch_id: str | None = None

    def __post_init__(self) -> None:
        sample_ids = tuple(str(sample_id) for sample_id in self.sample_ids)
        if not sample_ids:
            raise ValueError("a routing batch must contain at least one sample ID")
        if any(not sample_id for sample_id in sample_ids):
            raise ValueError("routing sample IDs must be non-empty")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("routing sample IDs must be unique within a batch")
        object.__setattr__(self, "sample_ids", sample_ids)


@dataclass(frozen=True, slots=True)
class RoutingEvent:
    """One token's normalized routing decision at one MoE layer.

    The schema intentionally carries both ``sample_id`` and ``sequence_index``.
    Stable sample IDs align runs, while the sequence index records the event's
    position in the batch that produced it.
    """

    sample_id: str
    sequence_index: int
    token_position: int
    layer_id: str
    selected_expert_ids: tuple[int, ...]
    selected_weights: tuple[float, ...]
    router_probabilities: tuple[float, ...] | None = None
    router_logits: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        expert_ids = tuple(int(expert_id) for expert_id in self.selected_expert_ids)
        weights = _finite_tuple(self.selected_weights, field_name="selected_weights")
        probabilities = (
            None
            if self.router_probabilities is None
            else _finite_tuple(self.router_probabilities, field_name="router_probabilities")
        )
        logits = (
            None
            if self.router_logits is None
            else _finite_tuple(self.router_logits, field_name="router_logits")
        )

        if not self.sample_id:
            raise ValueError("sample_id must be non-empty")
        if not self.layer_id:
            raise ValueError("layer_id must be non-empty")
        if self.sequence_index < 0:
            raise ValueError("sequence_index must be non-negative")
        if self.token_position < 0:
            raise ValueError("token_position must be non-negative")
        if not expert_ids:
            raise ValueError("selected_expert_ids must contain at least one expert")
        if any(expert_id < 0 for expert_id in expert_ids):
            raise ValueError("selected expert IDs must be non-negative")
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError("selected expert IDs must be unique for one token")
        if len(expert_ids) != len(weights):
            raise ValueError("selected expert IDs and weights must have equal lengths")
        if any(weight < 0.0 for weight in weights):
            raise ValueError("selected_weights must be non-negative")
        if probabilities is not None and any(value < 0.0 for value in probabilities):
            raise ValueError("router_probabilities must be non-negative")
        if probabilities is not None and expert_ids and max(expert_ids) >= len(probabilities):
            raise ValueError("selected expert ID is outside router_probabilities")
        if logits is not None and expert_ids and max(expert_ids) >= len(logits):
            raise ValueError("selected expert ID is outside router_logits")
        if probabilities is not None and logits is not None and len(probabilities) != len(logits):
            raise ValueError("router probabilities and logits must describe the same experts")

        object.__setattr__(self, "selected_expert_ids", expert_ids)
        object.__setattr__(self, "selected_weights", weights)
        object.__setattr__(self, "router_probabilities", probabilities)
        object.__setattr__(self, "router_logits", logits)

    @property
    def alignment_key(self) -> tuple[str, int, str]:
        """Return the stable cross-run token alignment key."""

        return (self.sample_id, self.token_position, self.layer_id)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable trace record."""

        return {
            "sample_id": self.sample_id,
            "sequence_index": self.sequence_index,
            "token_position": self.token_position,
            "layer_id": self.layer_id,
            "selected_expert_ids": list(self.selected_expert_ids),
            "selected_weights": list(self.selected_weights),
            "router_probabilities": (
                None if self.router_probabilities is None else list(self.router_probabilities)
            ),
            "router_logits": None if self.router_logits is None else list(self.router_logits),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoutingEvent:
        """Restore a routing event from its serialized representation."""

        return cls(
            sample_id=str(data["sample_id"]),
            sequence_index=int(data["sequence_index"]),
            token_position=int(data["token_position"]),
            layer_id=str(data["layer_id"]),
            selected_expert_ids=tuple(int(value) for value in data["selected_expert_ids"]),
            selected_weights=tuple(float(value) for value in data["selected_weights"]),
            router_probabilities=(
                None
                if data.get("router_probabilities") is None
                else tuple(float(value) for value in data["router_probabilities"])
            ),
            router_logits=(
                None
                if data.get("router_logits") is None
                else tuple(float(value) for value in data["router_logits"])
            ),
        )


@dataclass(frozen=True, slots=True)
class LayerRoutingAggregate:
    """Full-denominator online aggregate for one routed layer."""

    layer_id: str
    token_count: int
    assignment_count: int
    expert_count: int
    expert_selection_counts: dict[int, int]
    expert_weight_sums: dict[int, float]
    sample_token_counts: dict[str, int]
    selected_weight_sum: float

    def __post_init__(self) -> None:
        if not self.layer_id:
            raise ValueError("layer_id must be non-empty")
        if self.token_count < 0 or self.assignment_count < 0 or self.expert_count < 0:
            raise ValueError("routing aggregate denominators must be non-negative")
        if any(
            expert_id < 0 or expert_id >= self.expert_count
            for expert_id in self.expert_selection_counts
        ):
            raise ValueError("selection count contains an out-of-range expert ID")
        if any(count < 0 for count in self.expert_selection_counts.values()):
            raise ValueError("expert selection counts must be non-negative")
        if sum(self.expert_selection_counts.values()) != self.assignment_count:
            raise ValueError("expert selection counts must sum to assignment_count")
        if any(
            expert_id < 0 or expert_id >= self.expert_count for expert_id in self.expert_weight_sums
        ):
            raise ValueError("weight sum contains an out-of-range expert ID")
        if any(
            not math.isfinite(weight) or weight < 0.0 for weight in self.expert_weight_sums.values()
        ):
            raise ValueError("expert weight sums must be finite and non-negative")
        if any(count < 0 for count in self.sample_token_counts.values()):
            raise ValueError("sample token counts must be non-negative")
        if sum(self.sample_token_counts.values()) != self.token_count:
            raise ValueError("sample token counts must sum to token_count")
        if not math.isfinite(self.selected_weight_sum) or self.selected_weight_sum < 0.0:
            raise ValueError("selected_weight_sum must be finite and non-negative")

    @property
    def selection_counts(self) -> dict[int, int]:
        """Compatibility alias for the expert selection count vector."""

        return dict(self.expert_selection_counts)

    @property
    def expert_frequencies(self) -> dict[int, float]:
        """Return assignment-normalized expert selection frequencies."""

        if self.assignment_count == 0:
            return {expert_id: 0.0 for expert_id in range(self.expert_count)}
        return {
            expert_id: self.expert_selection_counts.get(expert_id, 0) / self.assignment_count
            for expert_id in range(self.expert_count)
        }

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable aggregate record."""

        return {
            "layer_id": self.layer_id,
            "token_count": self.token_count,
            "assignment_count": self.assignment_count,
            "expert_count": self.expert_count,
            "expert_selection_counts": {
                str(key): value for key, value in self.expert_selection_counts.items()
            },
            "expert_weight_sums": {
                str(key): value for key, value in self.expert_weight_sums.items()
            },
            "sample_token_counts": dict(self.sample_token_counts),
            "selected_weight_sum": self.selected_weight_sum,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LayerRoutingAggregate:
        """Restore a layer aggregate from its serialized representation."""

        return cls(
            layer_id=str(data["layer_id"]),
            token_count=int(data["token_count"]),
            assignment_count=int(data["assignment_count"]),
            expert_count=int(data["expert_count"]),
            expert_selection_counts={
                int(key): int(value) for key, value in dict(data["expert_selection_counts"]).items()
            },
            expert_weight_sums={
                int(key): float(value) for key, value in dict(data["expert_weight_sums"]).items()
            },
            sample_token_counts={
                str(key): int(value) for key, value in dict(data["sample_token_counts"]).items()
            },
            selected_weight_sum=float(data["selected_weight_sum"]),
        )


@dataclass(slots=True)
class _MutableLayerAggregate:
    layer_id: str
    expert_count: int = 0
    token_count: int = 0
    assignment_count: int = 0
    expert_selection_counts: dict[int, int] = field(default_factory=dict)
    expert_weight_sums: dict[int, float] = field(default_factory=dict)
    sample_token_counts: dict[str, int] = field(default_factory=dict)
    selected_weight_sum: float = 0.0

    def record(self, event: RoutingEvent) -> None:
        observed_expert_count = max(event.selected_expert_ids) + 1
        if event.router_probabilities is not None:
            observed_expert_count = max(observed_expert_count, len(event.router_probabilities))
        if event.router_logits is not None:
            observed_expert_count = max(observed_expert_count, len(event.router_logits))
        self.expert_count = max(self.expert_count, observed_expert_count)
        self.token_count += 1
        self.assignment_count += len(event.selected_expert_ids)
        self.sample_token_counts[event.sample_id] = (
            self.sample_token_counts.get(event.sample_id, 0) + 1
        )
        for expert_id, weight in zip(
            event.selected_expert_ids, event.selected_weights, strict=True
        ):
            self.expert_selection_counts[expert_id] = (
                self.expert_selection_counts.get(expert_id, 0) + 1
            )
            self.expert_weight_sums[expert_id] = (
                self.expert_weight_sums.get(expert_id, 0.0) + weight
            )
            self.selected_weight_sum += weight

    def record_batch(
        self,
        *,
        sample_token_counts: Mapping[str, int],
        expert_selection_counts: Sequence[int],
        expert_weight_sums: Sequence[float],
    ) -> None:
        """Merge one vectorized routing batch without materializing token events."""

        if len(expert_selection_counts) != len(expert_weight_sums):
            raise ValueError("batch routing count and weight vectors must have equal lengths")
        if any(count < 0 for count in sample_token_counts.values()):
            raise ValueError("batch routing sample token counts must be non-negative")
        if any(count < 0 for count in expert_selection_counts):
            raise ValueError("batch routing expert counts must be non-negative")
        if any(not math.isfinite(weight) or weight < 0.0 for weight in expert_weight_sums):
            raise ValueError("batch routing expert weight sums must be finite and non-negative")

        self.expert_count = max(self.expert_count, len(expert_selection_counts))
        batch_token_count = sum(sample_token_counts.values())
        self.token_count += batch_token_count
        self.assignment_count += sum(expert_selection_counts)
        for sample_id, count in sample_token_counts.items():
            self.sample_token_counts[sample_id] = self.sample_token_counts.get(sample_id, 0) + count
        for expert_id, count in enumerate(expert_selection_counts):
            self.expert_selection_counts[expert_id] = (
                self.expert_selection_counts.get(expert_id, 0) + count
            )
        for expert_id, weight in enumerate(expert_weight_sums):
            self.expert_weight_sums[expert_id] = (
                self.expert_weight_sums.get(expert_id, 0.0) + weight
            )
            self.selected_weight_sum += weight

    def freeze(self) -> LayerRoutingAggregate:
        counts = {
            expert_id: self.expert_selection_counts.get(expert_id, 0)
            for expert_id in range(self.expert_count)
        }
        weights = {
            expert_id: self.expert_weight_sums.get(expert_id, 0.0)
            for expert_id in range(self.expert_count)
        }
        return LayerRoutingAggregate(
            layer_id=self.layer_id,
            token_count=self.token_count,
            assignment_count=self.assignment_count,
            expert_count=self.expert_count,
            expert_selection_counts=counts,
            expert_weight_sums=weights,
            sample_token_counts=dict(sorted(self.sample_token_counts.items())),
            selected_weight_sum=self.selected_weight_sum,
        )


@dataclass(frozen=True, slots=True)
class RoutingArtifact:
    """Closed routing capture with aggregates and separately retained raw traces."""

    mode: RoutingMode
    aggregates: dict[str, LayerRoutingAggregate]
    raw_traces: tuple[RoutingEvent, ...]
    observed_event_count: int
    recorded_event_count: int
    batch_count: int

    def __post_init__(self) -> None:
        if self.observed_event_count < 0 or self.recorded_event_count < 0:
            raise ValueError("routing artifact event counts must be non-negative")
        if self.batch_count < 0:
            raise ValueError("routing artifact batch_count must be non-negative")
        if self.recorded_event_count != len(self.raw_traces):
            raise ValueError("recorded_event_count must equal the retained raw trace count")
        if self.recorded_event_count > self.observed_event_count:
            raise ValueError("recorded events cannot exceed observed events")
        aggregate_event_count = sum(aggregate.token_count for aggregate in self.aggregates.values())
        if aggregate_event_count != self.observed_event_count:
            raise ValueError("aggregate token denominators must equal observed_event_count")

    @property
    def events(self) -> tuple[RoutingEvent, ...]:
        """Compatibility alias for retained raw traces."""

        return self.raw_traces

    @property
    def traces(self) -> tuple[RoutingEvent, ...]:
        """Compatibility alias for retained raw traces."""

        return self.raw_traces

    def as_dict(self, *, include_raw_traces: bool = False) -> dict[str, Any]:
        """Return artifact metadata and aggregates, optionally embedding traces."""

        result: dict[str, Any] = {
            "schema_version": "1.0",
            "mode": self.mode,
            "observed_event_count": self.observed_event_count,
            "recorded_event_count": self.recorded_event_count,
            "batch_count": self.batch_count,
            "aggregates": {
                layer_id: aggregate.as_dict()
                for layer_id, aggregate in sorted(self.aggregates.items())
            },
        }
        if include_raw_traces:
            result["raw_traces"] = [event.as_dict() for event in self.raw_traces]
        return result

    def write(self, destination: Path) -> dict[str, Path]:
        """Persist metadata/aggregates and raw traces as separate JSON files."""

        destination.mkdir(parents=True, exist_ok=True)
        aggregates_path = destination / "aggregates.json"
        traces_path = destination / "traces.jsonl"
        aggregates_path.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        paths = {"aggregates": aggregates_path}
        if self.raw_traces:
            traces_path.write_text(
                "".join(
                    json.dumps(event.as_dict(), sort_keys=True) + "\n" for event in self.raw_traces
                ),
                encoding="utf-8",
            )
            paths["traces"] = traces_path
        return paths

    @classmethod
    def read(cls, destination: Path) -> RoutingArtifact:
        """Load an artifact written by :meth:`write`."""

        aggregate_data = json.loads((destination / "aggregates.json").read_text("utf-8"))
        traces_path = destination / "traces.jsonl"
        raw_traces = (
            tuple(
                RoutingEvent.from_dict(json.loads(line))
                for line in traces_path.read_text("utf-8").splitlines()
                if line
            )
            if traces_path.exists()
            else ()
        )
        return cls(
            mode=aggregate_data["mode"],
            aggregates={
                str(layer_id): LayerRoutingAggregate.from_dict(aggregate)
                for layer_id, aggregate in aggregate_data["aggregates"].items()
            },
            raw_traces=raw_traces,
            observed_event_count=int(aggregate_data["observed_event_count"]),
            recorded_event_count=int(aggregate_data["recorded_event_count"]),
            batch_count=int(aggregate_data["batch_count"]),
        )


@runtime_checkable
class RoutingSink(Protocol):
    """Architecture-independent routing capture target."""

    def start_batch(self, batch_meta: BatchMeta) -> None: ...

    def record(self, event: RoutingEvent) -> None: ...

    def record_aggregate_batch(
        self,
        *,
        layer_id: str,
        sample_ids: tuple[str, ...],
        tokens_per_sample: int,
        expert_selection_counts: Sequence[int],
        expert_weight_sums: Sequence[float],
    ) -> None: ...

    def end_batch(self) -> None: ...

    def close(self) -> RoutingArtifact: ...


class InMemoryRoutingSink:
    """Collect routing aggregates online and retain raw events per storage mode."""

    def __init__(
        self,
        mode: RoutingMode = "aggregate",
        *,
        sampled_token_positions: Sequence[int] = (),
        expert_counts: Mapping[str, int] | None = None,
    ) -> None:
        if mode not in {"off", "aggregate", "sampled_tokens", "full_trace"}:
            raise ValueError(f"unsupported routing mode: {mode}")
        sampled_positions = frozenset(int(position) for position in sampled_token_positions)
        if any(position < 0 for position in sampled_positions):
            raise ValueError("sampled token positions must be non-negative")
        if mode == "sampled_tokens" and not sampled_positions:
            raise ValueError("sampled_tokens mode requires sampled_token_positions")

        self.mode = mode
        self.sampled_token_positions = sampled_positions
        self._layers = {
            layer_id: _MutableLayerAggregate(layer_id=layer_id, expert_count=int(expert_count))
            for layer_id, expert_count in (expert_counts or {}).items()
        }
        if any(layer.expert_count < 0 for layer in self._layers.values()):
            raise ValueError("expert counts must be non-negative")
        self._raw_traces: list[RoutingEvent] = []
        self._observed_event_count = 0
        self._batch_count = 0
        self._active_batch: BatchMeta | None = None
        self._closed_artifact: RoutingArtifact | None = None

    @property
    def active_batch(self) -> BatchMeta | None:
        """Return the current batch metadata, if explicit batch tracking is active."""

        return self._active_batch

    def start_batch(self, batch_meta: BatchMeta) -> None:
        """Begin a batch and validate its stable sample identity."""

        self._ensure_open()
        if self._active_batch is not None:
            raise RoutingInstrumentationError(
                "cannot start a routing batch before ending the current batch",
                component="routing_sink",
            )
        self._active_batch = batch_meta
        self._batch_count += 1

    def record(self, event: RoutingEvent) -> None:
        """Record one event, updating full aggregates before raw-trace filtering."""

        self._ensure_open()
        if self.mode == "off":
            return
        if self._active_batch is not None:
            if event.sample_id not in self._active_batch.sample_ids:
                raise RoutingInstrumentationError(
                    f"routing event sample {event.sample_id!r} is not in the active batch",
                    component="routing_sink",
                )
            if event.sequence_index >= len(self._active_batch.sample_ids):
                raise RoutingInstrumentationError(
                    "routing event sequence index is outside the active batch",
                    component="routing_sink",
                )
            expected_sample_id = self._active_batch.sample_ids[event.sequence_index]
            if event.sample_id != expected_sample_id:
                raise RoutingInstrumentationError(
                    "routing event sequence index does not match its stable sample ID",
                    component="routing_sink",
                )

        aggregate = self._layers.setdefault(
            event.layer_id, _MutableLayerAggregate(layer_id=event.layer_id)
        )
        aggregate.record(event)
        self._observed_event_count += 1

        retain_raw = self.mode == "full_trace" or (
            self.mode == "sampled_tokens" and event.token_position in self.sampled_token_positions
        )
        if retain_raw:
            self._raw_traces.append(event)

    def record_aggregate_batch(
        self,
        *,
        layer_id: str,
        sample_ids: tuple[str, ...],
        tokens_per_sample: int,
        expert_selection_counts: Sequence[int],
        expert_weight_sums: Sequence[float],
    ) -> None:
        """Merge vectorized aggregate-mode counts from one routed-layer callback.

        This path deliberately accepts only already-reduced vectors. It keeps
        aggregate capture free of per-token Python event objects while retaining
        the same denominators and per-sample accounting as :meth:`record`.
        """

        self._ensure_open()
        if self.mode != "aggregate":
            raise RoutingInstrumentationError(
                "vectorized batch recording is only valid in aggregate mode",
                component="routing_sink",
            )
        if not layer_id:
            raise RoutingInstrumentationError(
                "vectorized routing batches require a layer ID", component="routing_sink"
            )
        if tokens_per_sample < 0:
            raise RoutingInstrumentationError(
                "vectorized routing token counts must be non-negative",
                component="routing_sink",
            )
        if not sample_ids:
            raise RoutingInstrumentationError(
                "vectorized routing batches require stable sample IDs",
                component="routing_sink",
            )
        if self._active_batch is not None and sample_ids != self._active_batch.sample_ids:
            raise RoutingInstrumentationError(
                "vectorized routing sample IDs do not match the active batch",
                component="routing_sink",
            )

        aggregate = self._layers.setdefault(
            layer_id,
            _MutableLayerAggregate(layer_id=layer_id, expert_count=len(expert_selection_counts)),
        )
        try:
            aggregate.record_batch(
                sample_token_counts={sample_id: tokens_per_sample for sample_id in sample_ids},
                expert_selection_counts=expert_selection_counts,
                expert_weight_sums=expert_weight_sums,
            )
        except ValueError as error:
            raise RoutingInstrumentationError(
                f"invalid vectorized routing batch for layer {layer_id}: {error}",
                component="routing_sink",
            ) from error
        self._observed_event_count += len(sample_ids) * tokens_per_sample

    def end_batch(self) -> None:
        """Finish the active batch."""

        self._ensure_open()
        if self._active_batch is None:
            raise RoutingInstrumentationError(
                "cannot end a routing batch when none is active",
                component="routing_sink",
            )
        self._active_batch = None

    def close(self) -> RoutingArtifact:
        """Freeze capture state into an immutable artifact; repeated calls are safe."""

        if self._closed_artifact is not None:
            return self._closed_artifact
        self._active_batch = None
        artifact = RoutingArtifact(
            mode=self.mode,
            aggregates={
                layer_id: layer.freeze()
                for layer_id, layer in sorted(self._layers.items())
                if self.mode != "off"
            },
            raw_traces=tuple(self._raw_traces),
            observed_event_count=self._observed_event_count,
            recorded_event_count=len(self._raw_traces),
            batch_count=self._batch_count,
        )
        self._closed_artifact = artifact
        return artifact

    def _ensure_open(self) -> None:
        if self._closed_artifact is not None:
            raise RoutingInstrumentationError(
                "routing sink is already closed", component="routing_sink"
            )


RoutingTraceSink = InMemoryRoutingSink
