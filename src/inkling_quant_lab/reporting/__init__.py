"""Deterministic tables, plots, and human-readable reports."""

from inkling_quant_lab.reporting.plots import DEFAULT_PLOT_SPECS, PlotSpec
from inkling_quant_lab.reporting.report import (
    MarkdownReporter,
    ReportArtifacts,
    ReportData,
    Reporter,
    generate_report,
    render_markdown_report,
    resolve_reporter,
)
from inkling_quant_lab.reporting.tables import (
    comparisons_csv,
    metrics_csv,
    pareto_csv,
)

__all__ = [
    "DEFAULT_PLOT_SPECS",
    "MarkdownReporter",
    "PlotSpec",
    "ReportArtifacts",
    "ReportData",
    "Reporter",
    "comparisons_csv",
    "generate_report",
    "metrics_csv",
    "pareto_csv",
    "render_markdown_report",
    "resolve_reporter",
]
