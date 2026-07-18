"""Routing metric contract tests (TC-ROUTE-METRIC-001 through 004)."""

from __future__ import annotations

import math

import pytest

from inkling_quant_lab.routing import (
    InMemoryRoutingSink,
    RoutingEvent,
    compare_routing,
    expert_selection_frequency,
    jensen_shannon_divergence,
    load_imbalance,
    routing_entropy,
    top_k_overlap,
)

pytestmark = pytest.mark.unit


def _event(
    *,
    layer_id: str,
    token_position: int,
    experts: tuple[int, ...],
    expert_count: int = 2,
    sample_id: str = "sample-0",
) -> RoutingEvent:
    weights = tuple(1.0 / len(experts) for _ in experts)
    probabilities = tuple(
        weights[experts.index(expert_id)] if expert_id in experts else 0.0
        for expert_id in range(expert_count)
    )
    return RoutingEvent(
        sample_id=sample_id,
        sequence_index=0,
        token_position=token_position,
        layer_id=layer_id,
        selected_expert_ids=experts,
        selected_weights=weights,
        router_probabilities=probabilities,
    )


def test_identical_routes_have_zero_drift_and_full_agreement() -> None:
    """TC-ROUTE-METRIC-001: identical traces produce the identity values."""

    events = (
        _event(layer_id="layer.0", token_position=0, experts=(0, 1)),
        _event(layer_id="layer.0", token_position=1, experts=(1, 0)),
    )

    result = compare_routing(events, events)
    layer = result.per_layer[0]

    assert layer.js_divergence == pytest.approx(0.0)
    assert layer.top_k_overlap == pytest.approx(1.0)
    assert layer.token_route_agreement == pytest.approx(1.0)
    assert result.macro.js_divergence == pytest.approx(0.0)
    assert result.weighted.top_k_overlap == pytest.approx(1.0)


def test_disjoint_routes_have_zero_overlap_and_expected_js_divergence() -> None:
    """TC-ROUTE-METRIC-002: disjoint one-hot routes reach maximal JS drift."""

    baseline = (_event(layer_id="layer.0", token_position=0, experts=(0,)),)
    candidate = (_event(layer_id="layer.0", token_position=0, experts=(1,)),)

    result = compare_routing(baseline, candidate)
    layer = result.per_layer[0]

    assert layer.top_k_overlap == pytest.approx(0.0)
    assert layer.token_route_agreement == pytest.approx(0.0)
    assert layer.js_divergence == pytest.approx(math.log(2.0))
    assert jensen_shannon_divergence((1.0, 0.0), (0.0, 1.0)) == pytest.approx(math.log(2.0))


def test_zero_traffic_bins_remain_finite() -> None:
    """TC-ROUTE-METRIC-003: zero bins never yield NaN or infinity."""

    baseline = (_event(layer_id="layer.0", token_position=0, experts=(0,), expert_count=4),)
    candidate = (_event(layer_id="layer.0", token_position=0, experts=(1,), expert_count=4),)

    result = compare_routing(baseline, candidate)
    layer = result.per_layer[0]
    values = (
        layer.js_divergence,
        layer.baseline_load_imbalance,
        layer.candidate_load_imbalance,
        layer.baseline_entropy,
        layer.candidate_entropy,
    )

    assert all(math.isfinite(value) for value in values)
    assert layer.baseline_selection_frequency == pytest.approx((1.0, 0.0, 0.0, 0.0))
    assert expert_selection_frequency((0, 0, 0, 0)) == (0.0, 0.0, 0.0, 0.0)
    assert jensen_shannon_divergence((0, 0), (0, 0)) == 0.0
    assert load_imbalance((0, 0, 0, 0)) == 0.0
    assert routing_entropy((0, 0, 0, 0)) == 0.0


def test_macro_and_token_weighted_aggregates_are_distinct() -> None:
    """TC-ROUTE-METRIC-004: layer and token weighting use different denominators."""

    baseline = (
        _event(layer_id="sparse", token_position=0, experts=(0,)),
        _event(layer_id="busy", token_position=0, experts=(0,)),
        _event(layer_id="busy", token_position=1, experts=(1,)),
        _event(layer_id="busy", token_position=2, experts=(0,)),
    )
    candidate = (
        _event(layer_id="sparse", token_position=0, experts=(1,)),
        _event(layer_id="busy", token_position=0, experts=(0,)),
        _event(layer_id="busy", token_position=1, experts=(1,)),
        _event(layer_id="busy", token_position=2, experts=(0,)),
    )

    result = compare_routing(baseline, candidate)

    assert result.macro.js_divergence == pytest.approx(math.log(2.0) / 2.0)
    assert result.weighted.js_divergence == pytest.approx(math.log(2.0) / 4.0)
    assert result.macro.top_k_overlap == pytest.approx(0.5)
    assert result.weighted.top_k_overlap == pytest.approx(0.75)
    assert result.macro.token_route_agreement == pytest.approx(0.5)
    assert result.weighted.token_route_agreement == pytest.approx(0.75)
    assert result.per_layer_drift_ranking == ("sparse", "busy")


def test_frequency_load_balance_and_entropy_have_documented_values() -> None:
    frequencies = expert_selection_frequency((2, 2, 0, 0))

    assert frequencies == pytest.approx((0.5, 0.5, 0.0, 0.0))
    assert load_imbalance(frequencies) == pytest.approx(1.0)
    assert routing_entropy(frequencies) == pytest.approx(math.log(2.0))
    assert routing_entropy(frequencies, normalized=True) == pytest.approx(0.5)


def test_metric_edge_cases_preserve_unavailable_and_empty_semantics() -> None:
    assert load_imbalance(()) == 0.0
    assert routing_entropy((1,), normalized=True) == 0.0
    assert top_k_overlap((), ()) == 1.0
    assert jensen_shannon_divergence((0, 0), (1, 0)) == pytest.approx(math.log(2.0))
    with pytest.raises(ValueError, match="logarithm base"):
        jensen_shannon_divergence((1, 0), (0, 1), base=1.0)

    empty = compare_routing((), ())
    assert empty.per_layer == ()
    assert empty.macro.layer_count == 0
    assert empty.weighted.token_weight == 0
    assert empty.macro.top_k_overlap is None

    only_baseline = compare_routing(
        (_event(layer_id="layer.0", token_position=0, experts=(0,)),), ()
    )
    assert only_baseline.per_layer[0].candidate_token_count == 0
    assert only_baseline.per_layer[0].top_k_overlap is None
    assert only_baseline.per_layer[0].js_divergence == pytest.approx(math.log(2.0))


def test_aggregate_only_inputs_report_trace_metrics_unavailable() -> None:
    sink = InMemoryRoutingSink(mode="aggregate", expert_counts={"layer.0": 4})
    artifact = sink.close()

    result = compare_routing(artifact, artifact.aggregates)

    assert result.layers == result.per_layer
    assert result.per_layer[0].expert_count == 4
    assert result.per_layer[0].top_k_overlap is None
    assert result.weighted.js_divergence == 0.0
    assert result.as_dict()["per_layer_drift_ranking"] == ["layer.0"]


def test_metric_inputs_are_validated() -> None:
    assert expert_selection_frequency({1: 2}, expert_count=3) == (0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="indexes"):
        expert_selection_frequency({-1: 2})
    with pytest.raises(ValueError, match="excludes"):
        expert_selection_frequency({2: 1}, expert_count=2)
    with pytest.raises(ValueError, match="excludes"):
        expert_selection_frequency((1, 2), expert_count=1)
    with pytest.raises(ValueError, match="finite"):
        expert_selection_frequency((1.0, math.nan))
    with pytest.raises(TypeError, match="LayerRoutingAggregate"):
        compare_routing({"layer.0": object()}, {})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="RoutingEvent"):
        compare_routing((object(),), ())  # type: ignore[arg-type]


def test_duplicate_or_differently_aligned_traces_are_not_silently_compared() -> None:
    first = _event(layer_id="layer.0", token_position=0, experts=(0,))
    second_sample = _event(
        layer_id="layer.0",
        token_position=0,
        experts=(0,),
        sample_id="sample-1",
    )

    with pytest.raises(ValueError, match="duplicate routing trace alignment key"):
        compare_routing((first, first), (first,))

    unaligned = compare_routing((first,), (second_sample,))
    assert unaligned.per_layer[0].top_k_overlap is None
    assert unaligned.per_layer[0].token_route_agreement is None
