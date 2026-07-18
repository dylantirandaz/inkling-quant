"""Stable CSV and Markdown tables built from normalized summaries only."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Mapping, Sequence

from inkling_quant_lab.comparison import (
    ComparisonResult,
    MetricCategory,
    NormalizedRunSummary,
    ParetoResult,
)


def format_number(value: float | None) -> str:
    """Format a measured number stably without inventing unavailable values."""

    if value is None:
        return ""
    if value == 0.0:
        return "0"
    return format(value, ".12g")


def _csv_text(fieldnames: Sequence[str], rows: Iterable[Mapping[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def metrics_csv(summaries: Iterable[NormalizedRunSummary]) -> str:
    """Return one stable long-form row per run and metric."""

    rows = []
    for summary in sorted(summaries, key=lambda item: item.run_id):
        for name, metric in sorted(summary.metrics.items()):
            suite = metric.evaluation_suite
            rows.append(
                {
                    "run_id": summary.run_id,
                    "metric": name,
                    "category": metric.category,
                    "status": metric.status,
                    "value": format_number(metric.value),
                    "unit": metric.unit or "",
                    "direction": metric.direction,
                    "reason": metric.reason or "",
                    "evaluator_name": suite.evaluator_name if suite is not None else "",
                    "evaluator_version": suite.evaluator_version if suite is not None else "",
                    "dataset_id": suite.dataset_id if suite is not None else "",
                    "dataset_revision": suite.dataset_revision if suite is not None else "",
                    "split": suite.split if suite is not None else "",
                    "dataset_sha256": suite.dataset_sha256 or "" if suite is not None else "",
                    "sample_count": str(len(suite.sample_ids)) if suite is not None else "",
                    "sample_ids": ";".join(suite.sample_ids) if suite is not None else "",
                    "seed": str(suite.seed) if suite is not None else "",
                    "prompt_template_hash": (
                        suite.prompt_template_hash if suite is not None else ""
                    ),
                    "decode_config": (
                        json.dumps(suite.decode_config, sort_keys=True, separators=(",", ":"))
                        if suite is not None
                        else ""
                    ),
                }
            )
    return _csv_text(
        (
            "run_id",
            "metric",
            "category",
            "status",
            "value",
            "unit",
            "direction",
            "reason",
            "evaluator_name",
            "evaluator_version",
            "dataset_id",
            "dataset_revision",
            "split",
            "dataset_sha256",
            "sample_count",
            "sample_ids",
            "seed",
            "prompt_template_hash",
            "decode_config",
        ),
        rows,
    )


def comparisons_csv(comparisons: Iterable[ComparisonResult]) -> str:
    """Return stable long-form candidate-minus-baseline deltas."""

    rows = []
    for comparison in sorted(
        comparisons, key=lambda item: (item.baseline_run_id, item.candidate_run_id)
    ):
        for delta in sorted(comparison.deltas, key=lambda item: item.metric):
            rows.append(
                {
                    "baseline_run_id": comparison.baseline_run_id,
                    "candidate_run_id": comparison.candidate_run_id,
                    "metric": delta.metric,
                    "category": delta.category,
                    "status": delta.status,
                    "baseline_value": format_number(delta.baseline_value),
                    "candidate_value": format_number(delta.candidate_value),
                    "absolute_delta": format_number(delta.absolute_delta),
                    "relative_delta": format_number(delta.relative_delta),
                    "unit": delta.unit or "",
                    "direction": delta.direction,
                    "reason": delta.reason or delta.relative_delta_reason or "",
                }
            )
    return _csv_text(
        (
            "baseline_run_id",
            "candidate_run_id",
            "metric",
            "category",
            "status",
            "baseline_value",
            "candidate_value",
            "absolute_delta",
            "relative_delta",
            "unit",
            "direction",
            "reason",
        ),
        rows,
    )


def pareto_csv(result: ParetoResult | None) -> str:
    """Return stable Pareto membership rows, including dominated candidates."""

    rows = []
    if result is not None:
        rows = [
            {
                "point_id": membership.point_id,
                "pareto_optimal": "true" if membership.pareto_optimal else "false",
                "dominated_by": ";".join(membership.dominated_by),
            }
            for membership in result.memberships
        ]
    return _csv_text(("point_id", "pareto_optimal", "dominated_by"), rows)


def _escape_markdown(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """Render a deterministic CommonMark table or an explicit unavailable marker."""

    materialized = [tuple(_escape_markdown(cell) for cell in row) for row in rows]
    if not materialized:
        return "_unavailable_"
    header = "| " + " | ".join(_escape_markdown(item) for item in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in materialized]
    return "\n".join((header, separator, *body))


def metric_markdown(summaries: Iterable[NormalizedRunSummary], category: MetricCategory) -> str:
    """Render measured values of one category, preserving unavailability."""

    rows: list[tuple[str, ...]] = []
    for summary in sorted(summaries, key=lambda item: item.run_id):
        for name, metric in sorted(summary.metrics.items()):
            if metric.category != category:
                continue
            display = format_number(metric.value) if metric.status == "available" else "unavailable"
            suite = metric.evaluation_suite
            source = (
                "—"
                if suite is None
                else (
                    f"{suite.evaluator_name}@{suite.evaluator_version}; "
                    f"{suite.dataset_id}@{suite.dataset_revision}:{suite.split}; "
                    f"sha256={suite.dataset_sha256 or 'unavailable'}; "
                    f"samples={len(suite.sample_ids)}; seed={suite.seed}"
                )
            )
            rows.append(
                (
                    summary.run_id,
                    name,
                    display,
                    metric.unit or "",
                    metric.status,
                    metric.reason or "",
                    source,
                )
            )
    return markdown_table(
        ("Run", "Metric", "Value", "Unit", "Status", "Notes", "Evaluation source"),
        rows,
    )


def pareto_markdown(result: ParetoResult | None) -> str:
    """Render all memberships so dominated points are not discarded."""

    if result is None:
        return "_unavailable_"
    return markdown_table(
        ("Configuration", "Pareto optimal", "Dominated by"),
        (
            (
                membership.point_id,
                "yes" if membership.pareto_optimal else "no",
                ", ".join(membership.dominated_by) or "—",
            )
            for membership in result.memberships
        ),
    )


__all__ = [
    "comparisons_csv",
    "format_number",
    "markdown_table",
    "metric_markdown",
    "metrics_csv",
    "pareto_csv",
    "pareto_markdown",
]
