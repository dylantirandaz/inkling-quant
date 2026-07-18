"""Dependency-free deterministic SVG plots with adjacent CSV source data."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from html import escape
from typing import Final

from inkling_quant_lab.comparison import NormalizedRunSummary, ParetoResult
from inkling_quant_lab.reporting.tables import format_number


@dataclass(frozen=True, slots=True)
class PlotSpec:
    """Two-dimensional plot definition over normalized metric names."""

    slug: str
    title: str
    x_metric: str
    y_metric: str
    x_label: str
    y_label: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", self.slug):
            raise ValueError("plot slug must contain lowercase words separated by underscores")
        if not all((self.title, self.x_metric, self.y_metric, self.x_label, self.y_label)):
            raise ValueError("plot fields must be non-empty")


@dataclass(frozen=True, slots=True)
class PlotObservation:
    """One run's available or unavailable source row."""

    run_id: str
    x: float | None
    y: float | None
    status: str
    reason: str


DEFAULT_PLOT_SPECS: Final[tuple[PlotSpec, ...]] = (
    PlotSpec(
        slug="quality_vs_memory",
        title="Quality versus serialized size",
        x_metric="serialized_size_bytes",
        y_metric="quality",
        x_label="Serialized size (bytes)",
        y_label="Quality",
    ),
    PlotSpec(
        slug="quality_vs_latency",
        title="Quality versus latency",
        x_metric="latency_ms",
        y_metric="quality",
        x_label="Latency (ms)",
        y_label="Quality",
    ),
    PlotSpec(
        slug="quality_vs_throughput",
        title="Quality versus throughput",
        x_metric="throughput_tokens_per_second",
        y_metric="quality",
        x_label="Throughput (tokens/s)",
        y_label="Quality",
    ),
)


def plot_observations(
    summaries: tuple[NormalizedRunSummary, ...], spec: PlotSpec
) -> tuple[PlotObservation, ...]:
    """Extract plot inputs without replacing absent measurements with zero."""

    observations: list[PlotObservation] = []
    for summary in sorted(summaries, key=lambda item: item.run_id):
        x_metric = summary.metrics.get(spec.x_metric)
        y_metric = summary.metrics.get(spec.y_metric)
        reasons: list[str] = []
        if x_metric is None:
            reasons.append(f"{spec.x_metric} is missing")
        elif x_metric.status != "available":
            reasons.append(f"{spec.x_metric} is {x_metric.status}: {x_metric.reason}")
        if y_metric is None:
            reasons.append(f"{spec.y_metric} is missing")
        elif y_metric.status != "available":
            reasons.append(f"{spec.y_metric} is {y_metric.status}: {y_metric.reason}")
        available = not reasons
        observations.append(
            PlotObservation(
                run_id=summary.run_id,
                x=x_metric.value if available and x_metric is not None else None,
                y=y_metric.value if available and y_metric is not None else None,
                status="available" if available else "unavailable",
                reason="; ".join(reasons),
            )
        )
    return tuple(observations)


def plot_source_csv(summaries: tuple[NormalizedRunSummary, ...], spec: PlotSpec) -> str:
    """Serialize all source rows, including unavailable candidates."""

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=(
            "run_id",
            "x_metric",
            "x_value",
            "y_metric",
            "y_value",
            "status",
            "reason",
        ),
        lineterminator="\n",
    )
    writer.writeheader()
    for observation in plot_observations(summaries, spec):
        writer.writerow(
            {
                "run_id": observation.run_id,
                "x_metric": spec.x_metric,
                "x_value": format_number(observation.x),
                "y_metric": spec.y_metric,
                "y_value": format_number(observation.y),
                "status": observation.status,
                "reason": observation.reason,
            }
        )
    return buffer.getvalue()


def _domain(values: tuple[float, ...]) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    padding = max(abs(lower) * 0.05, 1.0) if lower == upper else (upper - lower) * 0.08
    return lower - padding, upper + padding


def render_svg_plot(
    summaries: tuple[NormalizedRunSummary, ...],
    spec: PlotSpec,
    *,
    pareto: ParetoResult | None = None,
) -> str:
    """Render a fixed-layout SVG scatter plot using only the standard library."""

    width = 720
    height = 460
    left = 92
    right = 28
    top = 54
    bottom = 74
    chart_width = width - left - right
    chart_height = height - top - bottom
    observations = plot_observations(summaries, spec)
    available = tuple(
        observation
        for observation in observations
        if observation.x is not None and observation.y is not None
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">'
        ),
        f"  <title>{escape(spec.title)}</title>",
        '  <rect width="100%" height="100%" fill="white"/>',
        (
            f'  <text x="{width / 2:.1f}" y="30" text-anchor="middle" '
            f'font-family="sans-serif" font-size="18">{escape(spec.title)}</text>'
        ),
        (
            f'  <line x1="{left}" y1="{top + chart_height}" x2="{left + chart_width}" '
            f'y2="{top + chart_height}" stroke="#333"/>'
        ),
        (f'  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="#333"/>'),
        (
            f'  <text x="{left + chart_width / 2:.1f}" y="{height - 18}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="13">{escape(spec.x_label)}</text>'
        ),
        (
            f'  <text x="20" y="{top + chart_height / 2:.1f}" text-anchor="middle" '
            f'transform="rotate(-90 20 {top + chart_height / 2:.1f})" '
            f'font-family="sans-serif" font-size="13">{escape(spec.y_label)}</text>'
        ),
    ]
    if available:
        x_values = tuple(observation.x for observation in available if observation.x is not None)
        y_values = tuple(observation.y for observation in available if observation.y is not None)
        x_min, x_max = _domain(x_values)
        y_min, y_max = _domain(y_values)
        lines.extend(
            (
                (
                    f'  <text x="{left}" y="{top + chart_height + 22}" text-anchor="start" '
                    f'font-family="monospace" font-size="11">{escape(format_number(x_min))}</text>'
                ),
                (
                    f'  <text x="{left + chart_width}" y="{top + chart_height + 22}" '
                    f'text-anchor="end" font-family="monospace" font-size="11">'
                    f"{escape(format_number(x_max))}</text>"
                ),
                (
                    f'  <text x="{left - 8}" y="{top + chart_height}" text-anchor="end" '
                    f'font-family="monospace" font-size="11">{escape(format_number(y_min))}</text>'
                ),
                (
                    f'  <text x="{left - 8}" y="{top + 5}" text-anchor="end" '
                    f'font-family="monospace" font-size="11">{escape(format_number(y_max))}</text>'
                ),
            )
        )
        optimal_ids = set(pareto.optimal_ids) if pareto is not None else set()
        palette = ("#2563eb", "#9333ea", "#0891b2", "#dc2626", "#ca8a04", "#16a34a")
        for index, observation in enumerate(available):
            assert observation.x is not None
            assert observation.y is not None
            x = left + (observation.x - x_min) / (x_max - x_min) * chart_width
            y = top + chart_height - (observation.y - y_min) / (y_max - y_min) * chart_height
            color = (
                "#15803d" if observation.run_id in optimal_ids else palette[index % len(palette)]
            )
            lines.extend(
                (
                    (
                        f'  <circle cx="{x:.3f}" cy="{y:.3f}" r="6" fill="{color}" '
                        'stroke="white" stroke-width="1.5"/>'
                    ),
                    (
                        f'  <text x="{x + 9:.3f}" y="{y - 8:.3f}" font-family="sans-serif" '
                        f'font-size="11">{escape(observation.run_id)}</text>'
                    ),
                )
            )
    else:
        lines.append(
            f'  <text x="{left + chart_width / 2:.1f}" y="{top + chart_height / 2:.1f}" '
            'text-anchor="middle" font-family="sans-serif" font-size="14" fill="#666">'
            "No available measurements</text>"
        )
    unavailable_count = len(observations) - len(available)
    if unavailable_count:
        lines.append(
            f'  <text x="{width - right}" y="{height - 18}" text-anchor="end" '
            'font-family="sans-serif" font-size="11" fill="#666">'
            f"{unavailable_count} unavailable</text>"
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_PLOT_SPECS",
    "PlotObservation",
    "PlotSpec",
    "plot_observations",
    "plot_source_csv",
    "render_svg_plot",
]
