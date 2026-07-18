"""Deterministic dependency-free report requirements."""

from __future__ import annotations

from pathlib import Path

import pytest

from inkling_quant_lab.comparison import (
    DatasetIdentity,
    EvaluationSuiteIdentity,
    MetricValue,
    NormalizedRunSummary,
    ParetoObjective,
    ParetoPoint,
    RoutingCaptureIdentity,
    compare_summaries,
    pareto_frontier,
)
from inkling_quant_lab.reporting.plots import PlotSpec
from inkling_quant_lab.reporting.report import MarkdownReporter, ReportData, generate_report

pytestmark = pytest.mark.unit


def report_summary(run_id: str, *, unsupported_energy: bool = False) -> NormalizedRunSummary:
    """Create report data spanning quality, resources, routing, and behavior."""

    energy = (
        MetricValue(
            status="unsupported",
            unit="joules",
            category="resource",
            direction="minimize",
            reason="CPU energy sensor unavailable",
        )
        if unsupported_energy
        else MetricValue(
            value=5.0,
            unit="joules",
            category="resource",
            direction="minimize",
        )
    )
    return NormalizedRunSummary(
        run_id=run_id,
        artifact_path=f"artifacts/{run_id}",
        model_id="fixture://tiny-moe",
        model_revision="fixture-v1",
        datasets=(
            DatasetIdentity(
                dataset_id="fixture://eval",
                dataset_revision="fixture-v1",
                split="test",
                dataset_sha256="a" * 64,
            ),
        ),
        seed_set=(17,),
        sample_ids=("s1", "s2"),
        prompt_template_hash="b" * 64,
        decode_config={"do_sample": False},
        routing_dataset=DatasetIdentity(
            dataset_id="fixture://routing",
            dataset_revision="fixture-v1",
            split="routing",
            dataset_sha256="c" * 64,
        ),
        routing_sample_ids=("route-1", "route-2"),
        routing_capture=RoutingCaptureIdentity(
            configured_mode="full_trace",
            captured_mode="full_trace",
            observed_event_count=4,
            recorded_event_count=4,
            alignment_key_count=4,
            alignment_key_sha256="d" * 64,
        ),
        benchmark_protocol_version="cpu-v1",
        hardware_environment={
            "hardware": {"device": "cpu"},
            "runtime": {"backend": "torch_eager_cpu", "device": "cpu"},
        },
        metrics={
            "quality": MetricValue(
                value=0.9,
                category="quality",
                direction="maximize",
            ),
            "serialized_size_bytes": MetricValue(
                value=1000.0,
                unit="bytes",
                category="resource",
                direction="minimize",
            ),
            "latency_ms": MetricValue(
                value=10.0,
                unit="ms",
                category="resource",
                direction="minimize",
            ),
            "throughput_tokens_per_second": MetricValue(
                value=20.0,
                unit="tokens/s",
                category="resource",
                direction="maximize",
            ),
            "energy_joules": energy,
            "route_agreement": MetricValue(
                value=0.95,
                category="routing",
                direction="maximize",
            ),
            "behavioral_retention": MetricValue(
                value=0.98,
                category="behavioral",
                direction="maximize",
            ),
        },
        environment={"python": "3.11", "hardware": {"device": "cpu"}},
        resolved_config={"seed": 17, "runtime": {"backend": "torch_eager_cpu"}},
        quantization_policy={"default_precision": "int8"},
    )


def build_report_data() -> ReportData:
    """Create a complete two-run report input."""

    baseline = report_summary("baseline")
    candidate = report_summary("candidate", unsupported_energy=True)
    comparison = compare_summaries(baseline, candidate)
    pareto = pareto_frontier(
        (
            ParetoPoint(
                point_id=summary.run_id,
                values={
                    "quality": summary.metrics["quality"].value or 0.0,
                    "memory": summary.metrics["serialized_size_bytes"].value or 0.0,
                },
            )
            for summary in (baseline, candidate)
        ),
        (
            ParetoObjective(metric="quality", direction="maximize"),
            ParetoObjective(metric="memory", direction="minimize"),
        ),
    )
    return ReportData(
        runs=(baseline, candidate),
        comparisons=(comparison,),
        pareto=pareto,
        interpretations=("The candidate preserves the measured fixture quality.",),
        reproduction_commands=("uv run iql run configs/experiments/tiny_moe_int8.yaml",),
    )


def test_markdown_fallback_has_all_sdd_sections_without_optional_dependencies(
    tmp_path: Path,
) -> None:
    """TC-REPORT-001: Markdown and SVG generation require only core dependencies."""

    artifacts = generate_report(build_report_data(), tmp_path)
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")

    assert artifacts.html_path is None
    assert "## 1. Executive Summary" in markdown
    assert "## 2. Experimental Setup" in markdown
    assert "## 3. Compatibility and Caveats" in markdown
    assert "## 4. Quality Results" in markdown
    assert "## 5. Resource Results" in markdown
    assert "## 6. Routing Results" in markdown
    assert "## 7. Behavioral Retention" in markdown
    assert "## 8. Pareto Frontier" in markdown
    assert "## 9. Failures and Unsupported Measurements" in markdown
    assert "## 10. Reproduction Commands" in markdown
    assert "### Measured facts" in markdown
    assert "### Interpretations" in markdown


def test_every_svg_plot_has_adjacent_stable_source_csv(tmp_path: Path) -> None:
    """TC-REPORT-002: each plot has source data and output is deterministic."""

    first = generate_report(build_report_data(), tmp_path / "first")
    generate_report(build_report_data(), tmp_path / "second")

    assert len(first.plot_paths) == 3
    for plot in first.plot_paths:
        source = plot.with_suffix(".csv")
        assert source in first.plot_source_paths
        assert source.is_file()
        counterpart = tmp_path / "second" / plot.relative_to(tmp_path / "first")
        assert plot.read_bytes() == counterpart.read_bytes()
        assert source.read_bytes() == counterpart.with_suffix(".csv").read_bytes()


def test_unsupported_measurement_is_unavailable_never_zero(tmp_path: Path) -> None:
    """TC-REPORT-003: unsupported metrics render as unavailable, not zero."""

    artifacts = generate_report(build_report_data(), tmp_path)
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    metrics_csv = (tmp_path / "tables" / "metrics.csv").read_text(encoding="utf-8")

    assert "energy_joules | unavailable" in markdown
    assert "CPU energy sensor unavailable" in markdown
    energy_row = next(
        line for line in metrics_csv.splitlines() if line.startswith("candidate,energy_joules,")
    )
    assert ",unsupported,," in energy_row
    assert ",0," not in energy_row


def test_evaluation_metric_provenance_is_explicit_in_csv_and_markdown(tmp_path: Path) -> None:
    suite = EvaluationSuiteIdentity(
        evaluator_name="perplexity",
        evaluator_version="fixture-v1",
        dataset_id="fixture://eval",
        dataset_revision="fixture-data-v1",
        split="evaluation",
        dataset_sha256="a" * 64,
        sample_ids=("s1", "s2"),
        seed=17,
        prompt_template_hash="b" * 64,
        decode_config={"do_sample": False},
        status="success",
    )
    measured = report_summary("measured").model_copy(
        update={
            "metrics": {
                "quality": MetricValue(
                    value=0.9,
                    category="quality",
                    direction="maximize",
                    evaluation_suite=suite,
                )
            }
        }
    )

    artifacts = generate_report(ReportData(runs=(measured,)), tmp_path)
    csv_text = (tmp_path / "tables/metrics.csv").read_text(encoding="utf-8")
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")

    assert "evaluator_name,evaluator_version,dataset_id,dataset_revision" in csv_text
    assert "perplexity,fixture-v1,fixture://eval,fixture-data-v1,evaluation" in csv_text
    assert "perplexity@fixture-v1" in markdown
    assert "samples=2; seed=17" in markdown


def test_registry_reporter_supports_markdown_only_fallback(tmp_path: Path) -> None:
    """The lazy registry target provides a no-plot core-dependency path."""

    summary = report_summary("only").model_copy(
        update={
            "environment": {},
            "resolved_config": {},
            "quantization_policy": {},
            "failures": ("one retained sample failure",),
        }
    )
    data = ReportData(runs=(summary,))

    artifacts = MarkdownReporter().generate(data, tmp_path, include_plots=False)
    markdown = artifacts.markdown_path.read_text(encoding="utf-8")

    assert artifacts.plot_paths == ()
    assert "Pareto membership: unavailable" in markdown
    assert "one retained sample failure" in markdown
    assert "## 10. Reproduction Commands\n\n_unavailable_" in markdown
    assert MarkdownReporter().render(data, plot_specs=()).startswith("# Inkling Quant Lab")


def test_unavailable_plot_inputs_stay_blank_in_source_and_svg(tmp_path: Path) -> None:
    """A missing axis measurement yields an unavailable row and explanatory SVG."""

    summary = report_summary("quality-only").model_copy(
        update={"metrics": {"quality": MetricValue(value=0.9, category="quality")}}
    )

    artifacts = generate_report(ReportData(runs=(summary,)), tmp_path)

    assert all("No available measurements" in path.read_text() for path in artifacts.plot_paths)
    assert all(",unavailable," in path.read_text() for path in artifacts.plot_source_paths)


@pytest.mark.parametrize(
    "spec",
    (
        {
            "slug": "Not Safe",
            "title": "x",
            "x_metric": "x",
            "y_metric": "y",
            "x_label": "x",
            "y_label": "y",
        },
        {
            "slug": "safe",
            "title": "",
            "x_metric": "x",
            "y_metric": "y",
            "x_label": "x",
            "y_label": "y",
        },
    ),
)
def test_plot_specs_reject_unsafe_or_empty_fields(spec: dict[str, str]) -> None:
    """Plot file stems and labels are validated before artifact creation."""

    with pytest.raises(ValueError):
        PlotSpec(**spec)
