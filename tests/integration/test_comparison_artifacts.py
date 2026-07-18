"""Cross-run comparison artifact integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from shlex import join as shell_join

import pytest

from inkling_quant_lab.config import load_config
from inkling_quant_lab.data import load_local_dataset
from inkling_quant_lab.exceptions import (
    ArtifactIntegrityError,
    ComparisonCompatibilityError,
    ConfigurationError,
)
from inkling_quant_lab.pipeline.comparison_artifacts import compare_runs, report_artifact
from inkling_quant_lab.pipeline.runner import run_experiment
from inkling_quant_lab.reporting.report import ReportData

pytestmark = pytest.mark.integration
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLE_MARKER = ("sample-1", "sample-2")


def _completed_run(
    root: Path,
    run_id: str,
    *,
    quality: float,
    sample_ids: tuple[str, ...] = DEFAULT_SAMPLE_MARKER,
) -> Path:
    del quality  # The governed run supplies measured metrics; callers only distinguish fixtures.
    config = load_config(
        PROJECT_ROOT / "configs/experiments/tiny_moe_int8.yaml",
        (
            f"output.root={json.dumps(str(root))}",
            "benchmark.warmup_iterations=0",
            "benchmark.repetitions=1",
        ),
    )
    if sample_ids != DEFAULT_SAMPLE_MARKER:
        suites = tuple(
            suite.model_copy(
                update={
                    "sample_ids": load_local_dataset(
                        suite.dataset,
                        suite.revision,
                        suite.split,
                    ).sample_ids[:1]
                }
            )
            for suite in config.evaluation.suites
        )
        config = config.model_copy(
            update={"evaluation": config.evaluation.model_copy(update={"suites": suites})}
        )
    return run_experiment(config, project_root=PROJECT_ROOT, run_id=run_id)


def test_cross_run_comparison_and_normalized_report_regeneration(tmp_path: Path) -> None:
    run_root = tmp_path / "runs with spaces"
    baseline = _completed_run(run_root, "baseline", quality=0.9)
    candidate = _completed_run(run_root, "candidate", quality=0.85)

    comparison = compare_runs(
        (baseline, candidate),
        output_root=tmp_path / "comparison-artifacts",
    )

    assert comparison.name.startswith("comparison-")
    assert (comparison / "comparison_manifest.json").is_file()
    assert (comparison / "report.md").is_file()
    assert (comparison / "tables/comparisons.csv").is_file()
    assert len(tuple((comparison / "plots").glob("*.svg"))) == 3
    data = ReportData.model_validate_json(
        (comparison / "report_data.json").read_text(encoding="utf-8")
    )
    assert len(data.runs) == 2
    assert len(data.comparisons) == 1
    assert data.pareto is not None
    assert data.reproduction_commands == (
        shell_join(("uv", "run", "iql", "compare", str(baseline), str(candidate))),
    )
    assert report_artifact(comparison) == comparison / "report.md"
    assert report_artifact(baseline / "reports") == baseline / "reports/report.md"

    baseline_report = baseline / "reports/report.md"
    baseline_report_bytes = baseline_report.read_bytes()
    baseline_report.write_bytes(baseline_report_bytes + b"\ntampered\n")
    with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
        report_artifact(baseline / "reports")
    baseline_report.write_bytes(baseline_report_bytes)

    unexpected = comparison / "untracked.txt"
    unexpected.write_text("not governed\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="file set"):
        report_artifact(comparison)
    unexpected.unlink()

    regenerated = tmp_path / "regenerated"
    assert report_artifact(comparison, destination=regenerated) == regenerated / "report.md"
    assert (regenerated / "report_data.json").is_file()
    with pytest.raises(ArtifactIntegrityError, match="overwrite"):
        report_artifact(comparison, destination=regenerated)

    standalone_data = tmp_path / "standalone" / "report_data.json"
    standalone_data.parent.mkdir()
    standalone_data.write_bytes((comparison / "report_data.json").read_bytes())
    standalone_report = tmp_path / "standalone-regenerated"
    assert (
        report_artifact(standalone_data, destination=standalone_report)
        == standalone_report / "report.md"
    )
    standalone_reports_data = tmp_path / "not-a-run" / "reports" / "report_data.json"
    standalone_reports_data.parent.mkdir(parents=True)
    standalone_reports_data.write_bytes((comparison / "report_data.json").read_bytes())
    assert report_artifact(
        standalone_reports_data,
        destination=tmp_path / "standalone-reports-regenerated",
    ).is_file()

    report_path = comparison / "report.md"
    report_path.write_text(
        report_path.read_text(encoding="utf-8") + "\ntampered\n", encoding="utf-8"
    )
    with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
        report_artifact(comparison)

    (comparison / "comparison_manifest.json").unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing its governing manifest"):
        report_artifact(comparison)

    (baseline / "manifest.json").unlink()
    with pytest.raises(ArtifactIntegrityError, match="missing its governing manifest"):
        report_artifact(baseline / "reports")


def test_comparison_requires_compatible_contract_or_recorded_override(tmp_path: Path) -> None:
    baseline = _completed_run(tmp_path, "baseline", quality=0.9)
    candidate = _completed_run(
        tmp_path,
        "candidate",
        quality=0.8,
        sample_ids=("different",),
    )

    with pytest.raises(ComparisonCompatibilityError, match="sample_ids"):
        compare_runs((baseline, candidate), output_root=tmp_path / "rejected")

    comparison = compare_runs(
        (baseline, candidate),
        output_root=tmp_path / "accepted",
        unsafe_overrides={"sample_ids", "benchmark_workload"},
    )
    report_data = json.loads((comparison / "report_data.json").read_text(encoding="utf-8"))
    result = report_data["comparisons"][0]
    assert result["unsafe_override"] is True
    assert result["overridden_dimensions"] == ["benchmark_workload", "sample_ids"]
    assert "UNSAFE COMPARISON" in result["warnings"][0]

    summary_path = candidate / "reports/candidate_summary.json"
    summary_path.write_text(
        summary_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
        compare_runs((baseline, candidate), output_root=tmp_path / "tampered")

    manifest_path = baseline / "manifest.json"
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_data["status"] = "failed"
    manifest_path.write_text(
        json.dumps(manifest_data, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    with pytest.raises(ArtifactIntegrityError, match="not successful"):
        compare_runs((baseline, candidate), output_root=tmp_path / "failed")


def test_comparison_and_report_reject_missing_inputs(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="at least two"):
        compare_runs((tmp_path / "only",))
    with pytest.raises(ArtifactIntegrityError, match=r"report_data\.json"):
        report_artifact(tmp_path / "missing")
