"""Deterministic machine-readable and Markdown comparison reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, JsonValue, field_validator, model_validator

from inkling_quant_lab.artifacts import LocalArtifactStore
from inkling_quant_lab.comparison import (
    ComparisonResult,
    ImmutableComparisonModel,
    NormalizedRunSummary,
    ParetoResult,
)
from inkling_quant_lab.reporting.plots import (
    DEFAULT_PLOT_SPECS,
    PlotSpec,
    plot_source_csv,
    render_svg_plot,
)
from inkling_quant_lab.reporting.tables import (
    comparisons_csv,
    markdown_table,
    metric_markdown,
    metrics_csv,
    pareto_csv,
    pareto_markdown,
)


class ReportData(ImmutableComparisonModel):
    """All normalized inputs needed for report generation; no model handles allowed."""

    schema_version: Literal["1.0"] = "1.0"
    title: str = Field(default="Inkling Quant Lab Comparison Report", min_length=1)
    runs: tuple[NormalizedRunSummary, ...]
    comparisons: tuple[ComparisonResult, ...] = ()
    pareto: ParetoResult | None = None
    interpretations: tuple[str, ...] = ()
    reproduction_commands: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("runs")
    @classmethod
    def runs_are_present_and_unique(
        cls, value: tuple[NormalizedRunSummary, ...]
    ) -> tuple[NormalizedRunSummary, ...]:
        """Require reportable runs with unambiguous identifiers."""

        if not value:
            raise ValueError("report requires at least one normalized run summary")
        identifiers = [summary.run_id for summary in value]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("report run IDs must be unique")
        return value

    @model_validator(mode="after")
    def references_known_runs(self) -> ReportData:
        """Prevent reports from linking comparison rows to absent provenance."""

        run_ids = {summary.run_id for summary in self.runs}
        unknown_comparison_ids = {
            run_id
            for comparison in self.comparisons
            for run_id in (comparison.baseline_run_id, comparison.candidate_run_id)
            if run_id not in run_ids
        }
        if unknown_comparison_ids:
            raise ValueError(
                "comparison references unknown report runs: "
                + ", ".join(sorted(unknown_comparison_ids))
            )
        if self.pareto is not None:
            unknown_points = {
                membership.point_id
                for membership in self.pareto.memberships
                if membership.point_id not in run_ids
            }
            if unknown_points:
                raise ValueError(
                    "Pareto result references unknown report runs: "
                    + ", ".join(sorted(unknown_points))
                )
        return self


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    """Paths emitted by one report generation call."""

    markdown_path: Path
    machine_readable_path: Path
    table_paths: tuple[Path, ...]
    plot_paths: tuple[Path, ...]
    plot_source_paths: tuple[Path, ...]
    html_path: Path | None = None


class Reporter(Protocol):
    """Typed report-generation surface resolved through the lazy registry."""

    name: str

    def generate(
        self,
        data: ReportData,
        destination: str | Path,
        *,
        include_plots: bool = True,
        plot_specs: tuple[PlotSpec, ...] = DEFAULT_PLOT_SPECS,
    ) -> ReportArtifacts: ...


class MarkdownReporter:
    """Registry-facing dependency-free reporter implementation."""

    name = "markdown"

    def render(
        self,
        data: ReportData,
        *,
        plot_specs: tuple[PlotSpec, ...] = DEFAULT_PLOT_SPECS,
    ) -> str:
        """Render Markdown without writing artifacts."""

        return render_markdown_report(data, plot_specs=plot_specs)

    def generate(
        self,
        data: ReportData,
        destination: str | Path,
        *,
        include_plots: bool = True,
        plot_specs: tuple[PlotSpec, ...] = DEFAULT_PLOT_SPECS,
    ) -> ReportArtifacts:
        """Generate the complete deterministic local report bundle."""

        return generate_report(
            data,
            destination,
            include_plots=include_plots,
            plot_specs=plot_specs,
        )


def resolve_reporter(name: str = "markdown") -> Reporter:
    """Instantiate only the selected reporter through its lazy registry entry."""

    from inkling_quant_lab.bootstrap import register_builtins
    from inkling_quant_lab.registry import REPORTERS

    register_builtins()
    return REPORTERS.create(name)


def _json_block(value: dict[str, JsonValue]) -> str:
    if not value:
        return "_unavailable_"
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False)
    return f"```json\n{payload}\n```"


def _artifact_link(summary: NormalizedRunSummary) -> str:
    path = summary.artifact_path.replace(">", "%3E")
    return f"[{summary.run_id}](<{path}>)"


def _executive_summary(data: ReportData) -> str:
    facts = [
        f"- Runs included: {len(data.runs)}.",
        f"- Pairwise comparisons included: {len(data.comparisons)}.",
    ]
    if data.pareto is None:
        facts.append("- Pareto membership: unavailable.")
    else:
        optimal = ", ".join(data.pareto.optimal_ids) or "none"
        facts.append(f"- Pareto-optimal configurations: {optimal}.")
    warnings = [*data.warnings]
    warnings.extend(warning for summary in data.runs for warning in summary.warnings)
    warnings.extend(warning for comparison in data.comparisons for warning in comparison.warnings)
    warning_text = "\n".join(f"- **{warning}**" for warning in warnings) or "- None recorded."
    interpretations = (
        "\n".join(f"- {interpretation}" for interpretation in data.interpretations)
        if data.interpretations
        else "- No interpretations were supplied."
    )
    return (
        "Measured facts are derived from normalized run artifacts. Interpretations are supplied "
        "separately and are not presented as measurements.\n\n"
        "### Measured facts\n\n"
        + "\n".join(facts)
        + "\n\n### Warnings\n\n"
        + warning_text
        + "\n\n### Interpretations\n\n"
        + interpretations
    )


def _experimental_setup(data: ReportData) -> str:
    rows = []
    for summary in sorted(data.runs, key=lambda item: item.run_id):
        datasets = "; ".join(
            (f"{item.dataset_id}@{item.dataset_revision}:{item.split} sha256={item.dataset_sha256}")
            for item in sorted(
                summary.datasets,
                key=lambda value: (
                    value.dataset_id,
                    value.dataset_revision,
                    value.split,
                    value.dataset_sha256,
                ),
            )
        )
        routing_dataset = (
            "disabled"
            if summary.routing_dataset is None
            else (
                f"{summary.routing_dataset.dataset_id}@"
                f"{summary.routing_dataset.dataset_revision}:"
                f"{summary.routing_dataset.split} "
                f"sha256={summary.routing_dataset.dataset_sha256}"
            )
        )
        routing_capture = (
            "unsupported or not captured"
            if summary.routing_capture is None
            else (
                f"{summary.routing_capture.captured_mode}; "
                f"retained={summary.routing_capture.recorded_event_count}; "
                "alignment_sha256="
                f"{summary.routing_capture.alignment_key_sha256 or 'unavailable'}"
            )
        )
        benchmark_workload = (
            "disabled or unavailable"
            if summary.benchmark_workload is None
            else (
                f"{summary.benchmark_workload.dataset_id}@"
                f"{summary.benchmark_workload.dataset_revision}:"
                f"{summary.benchmark_workload.split} "
                f"sha256={summary.benchmark_workload.dataset_sha256}; "
                f"samples={','.join(summary.benchmark_workload.sample_ids)}; "
                f"seed={summary.benchmark_workload.seed}; "
                f"prompt_sha256={summary.benchmark_workload.prompt_template_hash}; "
                "decode="
                f"{json.dumps(summary.benchmark_workload.decode_config, sort_keys=True)}; "
                f"mode={summary.benchmark_workload.execution_mode}"
            )
        )
        benchmark_memory = (
            "unavailable"
            if summary.benchmark_memory is None
            else (
                "host_kind="
                f"{summary.benchmark_memory.host_measurement_kind or 'unavailable'}; "
                f"host_scope={summary.benchmark_memory.host_scope or 'unavailable'}; "
                "device_kind="
                f"{summary.benchmark_memory.device_measurement_kind or 'unavailable'}; "
                f"device_scope={summary.benchmark_memory.device_scope or 'unavailable'}"
            )
        )
        rows.append(
            (
                summary.run_id,
                summary.model_id,
                summary.model_revision or "unavailable",
                datasets,
                ", ".join(str(seed) for seed in summary.seed_set),
                str(len(summary.sample_ids)),
                routing_dataset,
                routing_capture,
                benchmark_workload,
                benchmark_memory,
                summary.benchmark_protocol_version,
                _artifact_link(summary),
            )
        )
    sections = [
        "### Measured facts",
        "",
        markdown_table(
            (
                "Run",
                "Model",
                "Revision",
                "Evaluation data",
                "Seed set",
                "Samples",
                "Routing data",
                "Routing capture evidence",
                "Benchmark workload",
                "Memory collector evidence",
                "Benchmark protocol",
                "Artifacts",
            ),
            rows,
        ),
    ]
    for summary in sorted(data.runs, key=lambda item: item.run_id):
        sections.extend(
            (
                "",
                f"### {summary.run_id} environment",
                "",
                _json_block(summary.environment),
                "",
                f"### {summary.run_id} resolved configuration",
                "",
                _json_block(summary.resolved_config),
                "",
                f"### {summary.run_id} quantization policy",
                "",
                _json_block(summary.quantization_policy),
            )
        )
    return "\n".join(sections)


def _compatibility(data: ReportData) -> str:
    rows: list[tuple[str, ...]] = []
    for comparison in sorted(
        data.comparisons,
        key=lambda item: (item.baseline_run_id, item.candidate_run_id),
    ):
        if comparison.mismatches:
            for mismatch in comparison.mismatches:
                rows.append(
                    (
                        comparison.baseline_run_id,
                        comparison.candidate_run_id,
                        mismatch.dimension,
                        "overridden"
                        if mismatch.dimension in comparison.overridden_dimensions
                        else "failed",
                        mismatch.message,
                    )
                )
        else:
            rows.append(
                (
                    comparison.baseline_run_id,
                    comparison.candidate_run_id,
                    "all required dimensions",
                    "compatible",
                    "",
                )
            )
    warning_lines = [warning for comparison in data.comparisons for warning in comparison.warnings]
    caveats = "\n".join(f"- **{warning}**" for warning in warning_lines) or "- None recorded."
    return (
        "### Measured facts\n\n"
        + markdown_table(("Baseline", "Candidate", "Dimension", "Status", "Notes"), rows)
        + "\n\n### Caveats\n\n"
        + caveats
    )


def _metric_section(
    data: ReportData,
    category: Literal["quality", "resource", "routing", "behavioral"],
) -> str:
    return "### Measured facts\n\n" + metric_markdown(data.runs, category)


def _pareto_section(data: ReportData) -> str:
    if data.pareto is None:
        objective_text = "_unavailable_"
    else:
        objective_text = markdown_table(
            ("Metric", "Direction", "Absolute tolerance", "Relative tolerance"),
            (
                (
                    objective.metric,
                    objective.direction,
                    format(objective.tolerance, ".12g"),
                    format(objective.relative_tolerance, ".12g"),
                )
                for objective in data.pareto.objectives
            ),
        )
    return "### Measured facts\n\n" + objective_text + "\n\n" + pareto_markdown(data.pareto)


def _failures_and_unsupported(data: ReportData) -> str:
    rows: list[tuple[str, ...]] = []
    for summary in sorted(data.runs, key=lambda item: item.run_id):
        rows.extend((summary.run_id, "failure", failure) for failure in summary.failures)
        rows.extend(
            (summary.run_id, "unavailable", measurement)
            for measurement in summary.unsupported_measurements
        )
        for name, metric in sorted(summary.metrics.items()):
            if metric.status != "available":
                rows.append(
                    (
                        summary.run_id,
                        "unavailable",
                        f"{name} | unavailable ({metric.status}): {metric.reason}",
                    )
                )
    for comparison in data.comparisons:
        for delta in comparison.deltas:
            if delta.status != "available":
                rows.append(
                    (
                        comparison.candidate_run_id,
                        "unavailable comparison",
                        f"{delta.metric}: {delta.reason}",
                    )
                )
    return "### Measured facts\n\n" + markdown_table(("Run", "Kind", "Details"), rows)


def _reproduction(data: ReportData) -> str:
    if not data.reproduction_commands:
        return "_unavailable_"
    return "\n\n".join(f"```console\n{command}\n```" for command in data.reproduction_commands)


def _plot_links(plot_specs: tuple[PlotSpec, ...]) -> str:
    if not plot_specs:
        return "_unavailable_"
    return "\n".join(
        (f"- [{spec.title}](plots/{spec.slug}.svg) ([source data](plots/{spec.slug}.csv))")
        for spec in plot_specs
    )


def render_markdown_report(
    data: ReportData, *, plot_specs: tuple[PlotSpec, ...] = DEFAULT_PLOT_SPECS
) -> str:
    """Render all SDD report sections without importing plotting or HTML libraries."""

    resource_section = _metric_section(data, "resource")
    if plot_specs:
        resource_section += "\n\n### Plot artifacts\n\n" + _plot_links(plot_specs)
    sections = (
        ("1. Executive Summary", _executive_summary(data)),
        ("2. Experimental Setup", _experimental_setup(data)),
        ("3. Compatibility and Caveats", _compatibility(data)),
        ("4. Quality Results", _metric_section(data, "quality")),
        ("5. Resource Results", resource_section),
        ("6. Routing Results", _metric_section(data, "routing")),
        ("7. Behavioral Retention", _metric_section(data, "behavioral")),
        ("8. Pareto Frontier", _pareto_section(data)),
        ("9. Failures and Unsupported Measurements", _failures_and_unsupported(data)),
        ("10. Reproduction Commands", _reproduction(data)),
    )
    body = "\n\n".join(f"## {heading}\n\n{content}" for heading, content in sections)
    return f"# {data.title}\n\n{body}\n"


def _machine_readable_json(data: ReportData) -> str:
    return (
        json.dumps(
            data.model_dump(mode="json"),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def generate_report(
    data: ReportData,
    destination: str | Path,
    *,
    include_plots: bool = True,
    plot_specs: tuple[PlotSpec, ...] = DEFAULT_PLOT_SPECS,
) -> ReportArtifacts:
    """Write immutable Markdown, JSON, CSV tables, and dependency-free SVG plots."""

    active_specs = plot_specs if include_plots else ()
    store = LocalArtifactStore(destination)
    machine_readable_path = store.write_text("report_data.json", _machine_readable_json(data))
    table_paths = (
        store.write_text("tables/metrics.csv", metrics_csv(data.runs)),
        store.write_text("tables/comparisons.csv", comparisons_csv(data.comparisons)),
        store.write_text("tables/pareto.csv", pareto_csv(data.pareto)),
    )
    plot_paths: list[Path] = []
    source_paths: list[Path] = []
    for spec in active_specs:
        source_paths.append(
            store.write_text(f"plots/{spec.slug}.csv", plot_source_csv(data.runs, spec))
        )
        plot_paths.append(
            store.write_text(
                f"plots/{spec.slug}.svg",
                render_svg_plot(data.runs, spec, pareto=data.pareto),
            )
        )
    markdown_path = store.write_text(
        "report.md", render_markdown_report(data, plot_specs=active_specs)
    )
    return ReportArtifacts(
        markdown_path=markdown_path,
        machine_readable_path=machine_readable_path,
        table_paths=table_paths,
        plot_paths=tuple(plot_paths),
        plot_source_paths=tuple(source_paths),
    )


__all__ = [
    "MarkdownReporter",
    "ReportArtifacts",
    "ReportData",
    "Reporter",
    "generate_report",
    "render_markdown_report",
    "resolve_reporter",
]
