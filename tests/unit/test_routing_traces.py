"""Routing capture contract tests (TC-ROUTE-001 through TC-ROUTE-004)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from inkling_quant_lab.exceptions import RoutingInstrumentationError
from inkling_quant_lab.routing import (
    BatchMeta,
    InMemoryRoutingSink,
    RoutingArtifact,
    RoutingEvent,
    attach_routing_hooks,
    events_from_routing_snapshot,
)

pytestmark = pytest.mark.unit


class SnapshotMoELayer(nn.Module):
    """Small hookable layer exposing the normalized snapshot contract."""

    layer_id = "moe.0"

    def __init__(self) -> None:
        super().__init__()
        self._snapshot: dict[str, Any] | None = None

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        primary = token_ids.remainder(4)
        secondary = primary.add(1).remainder(4)
        selected = torch.stack((primary, secondary), dim=-1)
        weights = torch.tensor([0.75, 0.25], dtype=torch.float32, device=token_ids.device).expand(
            *token_ids.shape, 2
        )
        probabilities = torch.zeros(
            *token_ids.shape, 4, dtype=torch.float32, device=token_ids.device
        )
        probabilities.scatter_(-1, selected, weights)
        self._snapshot = {
            "layer_id": self.layer_id,
            "selected_expert_ids": selected,
            "selected_weights": weights,
            "router_probabilities": probabilities,
            "router_logits": probabilities.add(1e-6).log(),
        }
        return token_ids

    def routing_snapshot(self) -> dict[str, Any] | None:
        return self._snapshot


class SnapshotModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.moe = SnapshotMoELayer()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.moe(token_ids)


def _capture(mode: str, *, sampled_positions: tuple[int, ...] = ()) -> RoutingArtifact:
    model = SnapshotModel()
    sink = InMemoryRoutingSink(  # type: ignore[arg-type]
        mode=mode, sampled_token_positions=sampled_positions
    )
    handle = attach_routing_hooks(
        model,
        sink,
        sample_ids=("sample-0",),
        capture_router_logits=True,
    )
    model(torch.tensor([[0, 1, 2, 3]], dtype=torch.long))
    handle.remove()
    return sink.close()


def test_event_capture_contains_complete_normalized_schema() -> None:
    """TC-ROUTE-001: hook snapshots retain identity, top-k, and weights."""

    model = SnapshotModel()
    sink = InMemoryRoutingSink(mode="full_trace")
    sink.start_batch(BatchMeta(sample_ids=("alpha", "beta")))
    handle = attach_routing_hooks(model, sink, capture_router_logits=True)

    model(torch.tensor([[0, 1, 2], [3, 2, 1]], dtype=torch.long))
    sink.end_batch()
    handle.remove()
    artifact = sink.close()

    assert artifact.observed_event_count == 6
    assert artifact.recorded_event_count == 6
    first = artifact.raw_traces[0]
    assert first.sample_id == "alpha"
    assert first.sequence_index == 0
    assert first.token_position == 0
    assert first.layer_id == "moe.0"
    assert first.selected_expert_ids == (0, 1)
    assert first.selected_weights == pytest.approx((0.75, 0.25))
    assert first.router_probabilities == pytest.approx((0.75, 0.25, 0.0, 0.0))
    assert first.router_logits is not None


def test_aggregate_mode_matches_full_trace_counts() -> None:
    """TC-ROUTE-002: online aggregates equal full-trace aggregates."""

    aggregate_only = _capture("aggregate")
    full_trace = _capture("full_trace")

    assert aggregate_only.raw_traces == ()
    assert aggregate_only.observed_event_count == full_trace.observed_event_count == 4
    assert aggregate_only.aggregates == full_trace.aggregates
    layer = aggregate_only.aggregates["moe.0"]
    assert layer.token_count == 4
    assert layer.assignment_count == 8
    assert sum(layer.expert_selection_counts.values()) == 8


def test_sampled_tokens_filters_raw_only_and_preserves_denominators() -> None:
    """TC-ROUTE-003: token sampling never changes aggregate denominators."""

    sampled = _capture("sampled_tokens", sampled_positions=(1, 3))
    full_trace = _capture("full_trace")

    assert [event.token_position for event in sampled.raw_traces] == [1, 3]
    assert sampled.recorded_event_count == 2
    assert sampled.observed_event_count == 4
    assert sampled.aggregates == full_trace.aggregates
    assert sampled.aggregates["moe.0"].token_count == 4
    assert sampled.aggregates["moe.0"].assignment_count == 8


def test_hook_cleanup_stops_subsequent_capture() -> None:
    """TC-ROUTE-004: removed hooks emit no later routing events."""

    model = SnapshotModel()
    sink = InMemoryRoutingSink(mode="full_trace")
    handle = attach_routing_hooks(
        model, sink, sample_ids=("sample-0",), capture_router_logits=False
    )
    model(torch.tensor([[0, 1]], dtype=torch.long))
    handle.remove()
    model(torch.tensor([[2, 3]], dtype=torch.long))

    artifact = sink.close()
    assert artifact.observed_event_count == 2
    assert [event.token_position for event in artifact.raw_traces] == [0, 1]
    assert all(event.router_logits is None for event in artifact.raw_traces)


def test_off_mode_retains_neither_aggregates_nor_raw_traces() -> None:
    artifact = _capture("off")

    assert artifact.observed_event_count == 0
    assert artifact.recorded_event_count == 0
    assert artifact.aggregates == {}
    assert artifact.raw_traces == ()


def test_routing_artifact_round_trips_separate_trace_storage(tmp_path: Path) -> None:
    artifact = _capture("full_trace")

    paths = artifact.write(tmp_path)
    restored = RoutingArtifact.read(tmp_path)

    assert set(paths) == {"aggregates", "traces"}
    assert restored == artifact


def test_aggregate_artifact_round_trip_omits_raw_trace_file(tmp_path: Path) -> None:
    artifact = _capture("aggregate")

    paths = artifact.write(tmp_path)
    restored = RoutingArtifact.read(tmp_path)

    assert set(paths) == {"aggregates"}
    assert restored == artifact
    assert artifact.events == artifact.traces == ()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"sample_id": ""}, "sample_id"),
        ({"layer_id": ""}, "layer_id"),
        ({"sequence_index": -1}, "sequence_index"),
        ({"token_position": -1}, "token_position"),
        ({"selected_expert_ids": ()}, "at least one"),
        ({"selected_expert_ids": (-1,)}, "non-negative"),
        ({"selected_expert_ids": (0, 0), "selected_weights": (0.5, 0.5)}, "unique"),
        ({"selected_weights": ()}, "equal lengths"),
        ({"selected_weights": (-1.0,)}, "non-negative"),
        ({"selected_weights": (math.nan,)}, "finite"),
        ({"router_probabilities": (-1.0, 2.0)}, "non-negative"),
        ({"router_probabilities": (1.0,)}, "outside router_probabilities"),
        ({"router_logits": (0.0,)}, "outside router_logits"),
        (
            {"router_probabilities": (0.5, 0.5), "router_logits": (0.0, 0.0, 0.0)},
            "same experts",
        ),
    ],
)
def test_routing_event_rejects_invalid_schema(updates: dict[str, Any], message: str) -> None:
    values: dict[str, Any] = {
        "sample_id": "sample-0",
        "sequence_index": 0,
        "token_position": 0,
        "layer_id": "layer.0",
        "selected_expert_ids": (1,),
        "selected_weights": (1.0,),
        "router_probabilities": None,
        "router_logits": None,
    }
    values.update(updates)

    with pytest.raises(ValueError, match=message):
        RoutingEvent(**values)


def test_batch_metadata_and_sink_lifecycle_are_validated() -> None:
    with pytest.raises(ValueError, match="at least one"):
        BatchMeta(())
    with pytest.raises(ValueError, match="non-empty"):
        BatchMeta(("",))
    with pytest.raises(ValueError, match="unique"):
        BatchMeta(("same", "same"))
    with pytest.raises(ValueError, match="unsupported routing mode"):
        InMemoryRoutingSink("invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-negative"):
        InMemoryRoutingSink("sampled_tokens", sampled_token_positions=(-1,))
    with pytest.raises(ValueError, match="requires"):
        InMemoryRoutingSink("sampled_tokens")
    with pytest.raises(ValueError, match="expert counts"):
        InMemoryRoutingSink("aggregate", expert_counts={"layer.0": -1})

    sink = InMemoryRoutingSink("full_trace")
    sink.start_batch(BatchMeta(("alpha", "beta")))
    with pytest.raises(RoutingInstrumentationError, match="before ending"):
        sink.start_batch(BatchMeta(("gamma",)))
    with pytest.raises(RoutingInstrumentationError, match="not in the active batch"):
        sink.record(RoutingEvent("gamma", 0, 0, "layer.0", (0,), (1.0,)))
    with pytest.raises(RoutingInstrumentationError, match="outside the active batch"):
        sink.record(RoutingEvent("alpha", 2, 0, "layer.0", (0,), (1.0,)))
    with pytest.raises(RoutingInstrumentationError, match="does not match"):
        sink.record(RoutingEvent("alpha", 1, 0, "layer.0", (0,), (1.0,)))
    sink.end_batch()
    with pytest.raises(RoutingInstrumentationError, match="none is active"):
        sink.end_batch()
    first_close = sink.close()
    assert sink.close() is first_close
    with pytest.raises(RoutingInstrumentationError, match="already closed"):
        sink.record(RoutingEvent("alpha", 0, 0, "layer.0", (0,), (1.0,)))


def test_seeded_zero_traffic_aggregate_preserves_explicit_bins() -> None:
    aggregate = (
        InMemoryRoutingSink("aggregate", expert_counts={"layer.0": 4}).close().aggregates["layer.0"]
    )

    assert aggregate.selection_counts == {0: 0, 1: 0, 2: 0, 3: 0}
    assert aggregate.expert_frequencies == {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    assert aggregate.as_dict()["assignment_count"] == 0


def test_snapshot_conversion_requires_identity_and_tensor_shapes() -> None:
    valid = {
        "selected_expert_ids": [[[0, 1]]],
        "selected_weights": [[[0.75, 0.25]]],
    }
    events = events_from_routing_snapshot(
        valid, sample_ids=("sample-0",), fallback_layer_id="layer.0"
    )
    assert events[0].layer_id == "layer.0"

    with pytest.raises(RoutingInstrumentationError, match="stable sample IDs"):
        events_from_routing_snapshot(valid, sample_ids=(), fallback_layer_id="layer.0")
    with pytest.raises(RoutingInstrumentationError, match="layer_id"):
        events_from_routing_snapshot(valid, sample_ids=("sample-0",))
    with pytest.raises(RoutingInstrumentationError, match="missing selected_weights"):
        events_from_routing_snapshot(
            {"layer_id": "layer.0", "selected_expert_ids": [[[0]]]},
            sample_ids=("sample-0",),
        )
    with pytest.raises(RoutingInstrumentationError, match="tensor-like"):
        events_from_routing_snapshot(
            {
                "layer_id": "layer.0",
                "selected_expert_ids": 0,
                "selected_weights": [[[1.0]]],
            },
            sample_ids=("sample-0",),
        )
    with pytest.raises(RoutingInstrumentationError, match="batch dimension"):
        events_from_routing_snapshot(
            {**valid, "layer_id": "layer.0"},
            sample_ids=("sample-0", "sample-1"),
        )


def test_handle_batch_helpers_context_cleanup_and_unsupported_model() -> None:
    model = SnapshotModel()
    sink = InMemoryRoutingSink("full_trace")
    handle = attach_routing_hooks(model, sink)
    handle.start_batch(("sample-0",))
    with pytest.raises(RoutingInstrumentationError, match="already owns"):
        handle.start_batch(("sample-0",))
    model(torch.tensor([[0]], dtype=torch.long))
    handle.end_batch()
    with pytest.raises(RoutingInstrumentationError, match="does not own"):
        handle.end_batch()
    handle.set_sample_ids(("sample-0",))
    handle.remove()
    handle.remove()
    with pytest.raises(RoutingInstrumentationError, match="after routing hooks"):
        handle.start_batch(("sample-0",))
    with pytest.raises(RoutingInstrumentationError, match="after routing hooks"):
        handle.set_sample_ids(("sample-0",))
    assert sink.close().observed_event_count == 1

    owned_sink = InMemoryRoutingSink("aggregate")
    with attach_routing_hooks(model, owned_sink) as owned_handle:
        owned_handle.start_batch(("sample-0",))
        model(torch.tensor([[0]], dtype=torch.long))
    assert owned_sink.active_batch is None

    with pytest.raises(RoutingInstrumentationError, match="no hookable"):
        attach_routing_hooks(nn.Linear(2, 2), InMemoryRoutingSink("aggregate"))
