"""Numerically stable MoE routing statistics and baseline/candidate comparison."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any, TypeAlias

from inkling_quant_lab.routing.traces import (
    InMemoryRoutingSink,
    LayerRoutingAggregate,
    RoutingArtifact,
    RoutingEvent,
)

CountVector: TypeAlias = Mapping[int, int | float] | Sequence[int | float]
RoutingMetricInput: TypeAlias = (
    RoutingArtifact | Mapping[str, LayerRoutingAggregate] | Iterable[RoutingEvent]
)


def _values(vector: CountVector, *, size: int | None = None) -> tuple[float, ...]:
    if isinstance(vector, Mapping):
        if any(index < 0 for index in vector):
            raise ValueError("expert vector indexes must be non-negative")
        inferred_size = max(vector, default=-1) + 1
        vector_size = inferred_size if size is None else size
        if vector_size < inferred_size:
            raise ValueError("requested expert count excludes observed experts")
        result = tuple(float(vector.get(index, 0.0)) for index in range(vector_size))
    else:
        result = tuple(float(value) for value in vector)
        if size is not None:
            if size < len(result):
                raise ValueError("requested expert count excludes observed experts")
            result += (0.0,) * (size - len(result))
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError("routing count/probability vectors must be finite and non-negative")
    return result


def expert_selection_frequency(
    counts: CountVector | LayerRoutingAggregate,
    *,
    expert_count: int | None = None,
) -> tuple[float, ...]:
    """Normalize expert assignment counts, preserving explicit zero bins."""

    if isinstance(counts, LayerRoutingAggregate):
        expert_count = max(expert_count or 0, counts.expert_count)
        values = _values(counts.expert_selection_counts, size=expert_count)
    else:
        values = _values(counts, size=expert_count)
    total = math.fsum(values)
    if total == 0.0:
        return (0.0,) * len(values)
    return tuple(value / total for value in values)


selection_frequency = expert_selection_frequency


def load_imbalance(values: CountVector) -> float:
    """Return coefficient of variation across experts (zero means balanced).

    The population coefficient of variation is scale independent, is one for a
    two-expert one-hot load, and remains finite for an all-zero vector.
    """

    vector = _values(values)
    if not vector:
        return 0.0
    mean = fmean(vector)
    if mean == 0.0:
        return 0.0
    variance = math.fsum((value - mean) ** 2 for value in vector) / len(vector)
    return math.sqrt(max(variance, 0.0)) / mean


def routing_entropy(values: CountVector, *, normalized: bool = False) -> float:
    """Return Shannon entropy in nats, optionally normalized to ``[0, 1]``."""

    probabilities = expert_selection_frequency(values)
    entropy = -math.fsum(
        probability * math.log(probability) for probability in probabilities if probability > 0.0
    )
    if not normalized:
        return entropy
    if len(probabilities) <= 1:
        return 0.0
    return entropy / math.log(len(probabilities))


def jensen_shannon_divergence(
    baseline: CountVector, candidate: CountVector, *, base: float = math.e
) -> float:
    """Return finite Jensen-Shannon divergence after count normalization.

    Empty bins contribute zero. If exactly one complete vector is empty, the
    absence/presence mismatch is treated as maximal divergence (``log(2)`` in
    the requested base); two empty vectors have zero divergence.
    """

    if base <= 0.0 or base == 1.0:
        raise ValueError("logarithm base must be positive and different from one")
    size = max(
        max(baseline, default=-1) + 1 if isinstance(baseline, Mapping) else len(baseline),
        max(candidate, default=-1) + 1 if isinstance(candidate, Mapping) else len(candidate),
    )
    baseline_values = _values(baseline, size=size)
    candidate_values = _values(candidate, size=size)
    baseline_total = math.fsum(baseline_values)
    candidate_total = math.fsum(candidate_values)
    if baseline_total == 0.0 and candidate_total == 0.0:
        return 0.0
    if baseline_total == 0.0 or candidate_total == 0.0:
        return math.log(2.0, base)

    p = tuple(value / baseline_total for value in baseline_values)
    q = tuple(value / candidate_total for value in candidate_values)
    midpoint = tuple((left + right) / 2.0 for left, right in zip(p, q, strict=True))

    def divergence(distribution: tuple[float, ...]) -> float:
        return math.fsum(
            probability * math.log(probability / mixture, base)
            for probability, mixture in zip(distribution, midpoint, strict=True)
            if probability > 0.0 and mixture > 0.0
        )

    result = (divergence(p) + divergence(q)) / 2.0
    maximum = math.log(2.0, base)
    return min(max(result, 0.0), maximum)


js_divergence = jensen_shannon_divergence


def top_k_overlap(baseline_experts: Sequence[int], candidate_experts: Sequence[int]) -> float:
    """Return selected-set intersection divided by the larger top-k width."""

    baseline = frozenset(int(expert_id) for expert_id in baseline_experts)
    candidate = frozenset(int(expert_id) for expert_id in candidate_experts)
    denominator = max(len(baseline), len(candidate))
    if denominator == 0:
        return 1.0
    return len(baseline.intersection(candidate)) / denominator


def token_route_agreement(
    baseline_experts: Sequence[int], candidate_experts: Sequence[int]
) -> float:
    """Return one for exact selected-set agreement and zero otherwise."""

    return float(
        frozenset(int(expert_id) for expert_id in baseline_experts)
        == frozenset(int(expert_id) for expert_id in candidate_experts)
    )


@dataclass(frozen=True, slots=True)
class LayerRoutingMetrics:
    """Routing quality and drift metrics for one layer."""

    layer_id: str
    expert_count: int
    baseline_token_count: int
    candidate_token_count: int
    baseline_assignment_count: int
    candidate_assignment_count: int
    baseline_selection_frequency: tuple[float, ...]
    candidate_selection_frequency: tuple[float, ...]
    baseline_load_imbalance: float
    candidate_load_imbalance: float
    baseline_entropy: float
    candidate_entropy: float
    js_divergence: float
    top_k_overlap: float | None
    token_route_agreement: float | None
    aligned_token_count: int
    drift_score: float

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable values for reporting."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoutingMetricSummary:
    """Macro- or token-weighted aggregate across routed layers."""

    layer_count: int
    token_weight: int
    js_divergence: float
    top_k_overlap: float | None
    token_route_agreement: float | None
    baseline_load_imbalance: float
    candidate_load_imbalance: float
    baseline_entropy: float
    candidate_entropy: float

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable values for reporting."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoutingComparison:
    """Per-layer routing drift plus macro and weighted summaries."""

    per_layer: tuple[LayerRoutingMetrics, ...]
    per_layer_drift_ranking: tuple[str, ...]
    macro: RoutingMetricSummary
    weighted: RoutingMetricSummary

    @property
    def layers(self) -> tuple[LayerRoutingMetrics, ...]:
        """Compatibility alias for per-layer metrics."""

        return self.per_layer

    def as_dict(self) -> dict[str, Any]:
        """Return a normalized report-ready representation."""

        return {
            "per_layer": [layer.as_dict() for layer in self.per_layer],
            "per_layer_drift_ranking": list(self.per_layer_drift_ranking),
            "macro": self.macro.as_dict(),
            "weighted": self.weighted.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class _RoutingData:
    aggregates: dict[str, LayerRoutingAggregate]
    traces: tuple[RoutingEvent, ...]


def _coerce_routing_data(source: RoutingMetricInput) -> _RoutingData:
    if isinstance(source, RoutingArtifact):
        return _RoutingData(dict(source.aggregates), source.raw_traces)
    if isinstance(source, Mapping):
        aggregates = dict(source)
        if any(not isinstance(value, LayerRoutingAggregate) for value in aggregates.values()):
            raise TypeError("routing aggregate mappings must contain LayerRoutingAggregate values")
        return _RoutingData(aggregates, ())

    events = tuple(source)
    if any(not isinstance(event, RoutingEvent) for event in events):
        raise TypeError("routing event iterables must contain RoutingEvent values")
    sink = InMemoryRoutingSink(mode="full_trace")
    for event in events:
        sink.record(event)
    artifact = sink.close()
    return _RoutingData(dict(artifact.aggregates), artifact.raw_traces)


def _empty_aggregate(layer_id: str, *, expert_count: int) -> LayerRoutingAggregate:
    return LayerRoutingAggregate(
        layer_id=layer_id,
        token_count=0,
        assignment_count=0,
        expert_count=expert_count,
        expert_selection_counts={expert_id: 0 for expert_id in range(expert_count)},
        expert_weight_sums={expert_id: 0.0 for expert_id in range(expert_count)},
        sample_token_counts={},
        selected_weight_sum=0.0,
    )


def _trace_index(
    events: tuple[RoutingEvent, ...], layer_id: str
) -> dict[tuple[str, int], RoutingEvent]:
    result: dict[tuple[str, int], RoutingEvent] = {}
    for event in events:
        if event.layer_id != layer_id:
            continue
        key = (event.sample_id, event.token_position)
        if key in result:
            raise ValueError(f"duplicate routing trace alignment key for layer {layer_id}: {key}")
        result[key] = event
    return result


def _aligned_trace_metrics(
    baseline: tuple[RoutingEvent, ...],
    candidate: tuple[RoutingEvent, ...],
    layer_id: str,
) -> tuple[float | None, float | None, int]:
    baseline_index = _trace_index(baseline, layer_id)
    candidate_index = _trace_index(candidate, layer_id)
    if not baseline_index or not candidate_index:
        return None, None, 0
    if baseline_index.keys() != candidate_index.keys():
        return None, None, 0
    overlaps: list[float] = []
    agreements: list[float] = []
    for key in sorted(baseline_index):
        baseline_event = baseline_index[key]
        candidate_event = candidate_index[key]
        overlaps.append(
            top_k_overlap(
                baseline_event.selected_expert_ids,
                candidate_event.selected_expert_ids,
            )
        )
        agreements.append(
            token_route_agreement(
                baseline_event.selected_expert_ids,
                candidate_event.selected_expert_ids,
            )
        )
    return fmean(overlaps), fmean(agreements), len(overlaps)


def _weighted_mean(values: Sequence[tuple[float, int]]) -> float:
    denominator = sum(weight for _, weight in values)
    if denominator == 0:
        return 0.0
    return math.fsum(value * weight for value, weight in values) / denominator


def _optional_mean(values: Sequence[tuple[float | None, int]]) -> float | None:
    available = [(value, weight) for value, weight in values if value is not None]
    if not available:
        return None
    return _weighted_mean([(value, weight) for value, weight in available])


def _summarize(layers: Sequence[LayerRoutingMetrics], *, weighted: bool) -> RoutingMetricSummary:
    if not layers:
        return RoutingMetricSummary(
            layer_count=0,
            token_weight=0,
            js_divergence=0.0,
            top_k_overlap=None,
            token_route_agreement=None,
            baseline_load_imbalance=0.0,
            candidate_load_imbalance=0.0,
            baseline_entropy=0.0,
            candidate_entropy=0.0,
        )
    distribution_weights = [
        max(layer.baseline_token_count, layer.candidate_token_count) if weighted else 1
        for layer in layers
    ]
    trace_weights = [layer.aligned_token_count if weighted else 1 for layer in layers]
    return RoutingMetricSummary(
        layer_count=len(layers),
        token_weight=sum(
            max(layer.baseline_token_count, layer.candidate_token_count) for layer in layers
        ),
        js_divergence=_weighted_mean(
            [
                (layer.js_divergence, weight)
                for layer, weight in zip(layers, distribution_weights, strict=True)
            ]
        ),
        top_k_overlap=_optional_mean(
            [
                (layer.top_k_overlap, weight)
                for layer, weight in zip(layers, trace_weights, strict=True)
            ]
        ),
        token_route_agreement=_optional_mean(
            [
                (layer.token_route_agreement, weight)
                for layer, weight in zip(layers, trace_weights, strict=True)
            ]
        ),
        baseline_load_imbalance=_weighted_mean(
            [
                (layer.baseline_load_imbalance, weight)
                for layer, weight in zip(layers, distribution_weights, strict=True)
            ]
        ),
        candidate_load_imbalance=_weighted_mean(
            [
                (layer.candidate_load_imbalance, weight)
                for layer, weight in zip(layers, distribution_weights, strict=True)
            ]
        ),
        baseline_entropy=_weighted_mean(
            [
                (layer.baseline_entropy, weight)
                for layer, weight in zip(layers, distribution_weights, strict=True)
            ]
        ),
        candidate_entropy=_weighted_mean(
            [
                (layer.candidate_entropy, weight)
                for layer, weight in zip(layers, distribution_weights, strict=True)
            ]
        ),
    )


def compare_routing(
    baseline: RoutingMetricInput, candidate: RoutingMetricInput
) -> RoutingComparison:
    """Compare routing aggregates and aligned traces without requiring raw data."""

    baseline_data = _coerce_routing_data(baseline)
    candidate_data = _coerce_routing_data(candidate)
    layer_ids = sorted(baseline_data.aggregates.keys() | candidate_data.aggregates.keys())
    layers: list[LayerRoutingMetrics] = []
    for layer_id in layer_ids:
        baseline_present = baseline_data.aggregates.get(layer_id)
        candidate_present = candidate_data.aggregates.get(layer_id)
        expert_count = max(
            baseline_present.expert_count if baseline_present is not None else 0,
            candidate_present.expert_count if candidate_present is not None else 0,
        )
        baseline_layer = baseline_present or _empty_aggregate(layer_id, expert_count=expert_count)
        candidate_layer = candidate_present or _empty_aggregate(layer_id, expert_count=expert_count)
        baseline_frequency = expert_selection_frequency(baseline_layer, expert_count=expert_count)
        candidate_frequency = expert_selection_frequency(candidate_layer, expert_count=expert_count)
        overlap, agreement, aligned_count = _aligned_trace_metrics(
            baseline_data.traces, candidate_data.traces, layer_id
        )
        divergence = jensen_shannon_divergence(baseline_frequency, candidate_frequency)
        layers.append(
            LayerRoutingMetrics(
                layer_id=layer_id,
                expert_count=expert_count,
                baseline_token_count=baseline_layer.token_count,
                candidate_token_count=candidate_layer.token_count,
                baseline_assignment_count=baseline_layer.assignment_count,
                candidate_assignment_count=candidate_layer.assignment_count,
                baseline_selection_frequency=baseline_frequency,
                candidate_selection_frequency=candidate_frequency,
                baseline_load_imbalance=load_imbalance(baseline_frequency),
                candidate_load_imbalance=load_imbalance(candidate_frequency),
                baseline_entropy=routing_entropy(baseline_frequency),
                candidate_entropy=routing_entropy(candidate_frequency),
                js_divergence=divergence,
                top_k_overlap=overlap,
                token_route_agreement=agreement,
                aligned_token_count=aligned_count,
                drift_score=divergence,
            )
        )

    ranking = tuple(
        layer.layer_id
        for layer in sorted(layers, key=lambda layer: (-layer.drift_score, layer.layer_id))
    )
    return RoutingComparison(
        per_layer=tuple(layers),
        per_layer_drift_ranking=ranking,
        macro=_summarize(layers, weighted=False),
        weighted=_summarize(layers, weighted=True),
    )


compute_routing_metrics = compare_routing
