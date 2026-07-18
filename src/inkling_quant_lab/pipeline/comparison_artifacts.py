"""Immutable cross-run comparison and normalized report regeneration."""

from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Sequence, Set
from datetime import UTC, datetime
from pathlib import Path
from shlex import join as shell_join

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from inkling_quant_lab.artifacts import LocalArtifactStore, sha256_file
from inkling_quant_lab.comparison import (
    ComparisonResult,
    NormalizedRunSummary,
    ParetoObjective,
    compare_summaries,
    pareto_frontier,
    pareto_points_from_summaries,
)
from inkling_quant_lab.exceptions import ArtifactIntegrityError, ConfigurationError
from inkling_quant_lab.manifests import ArtifactChecksum
from inkling_quant_lab.pipeline.runner import load_normalized_candidate, validate_completed_run
from inkling_quant_lab.reporting.report import ReportData, resolve_reporter

_COMPARISON_ID_PATTERN = re.compile(r"comparison-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{8}-[0-9a-f]{6}")


class _ImmutableBundleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComparisonSourceRecord(_ImmutableBundleRecord):
    """Checksum-pinned source evidence used to build a comparison."""

    run_id: str
    run_directory: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ComparisonBundleManifest(_ImmutableBundleRecord):
    """Self-verifying immutable comparison bundle ledger."""

    schema_version: str = "1.0"
    comparison_id: str
    source_runs: tuple[ComparisonSourceRecord, ...]
    unsafe_overrides: tuple[str, ...] = ()
    outputs: tuple[ArtifactChecksum, ...]


def _comparison_id(summaries: Sequence[NormalizedRunSummary]) -> str:
    digest = hashlib.sha256()
    for summary in summaries:
        digest.update(summary.canonical_json().encode("utf-8"))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"comparison-{timestamp}-{digest.hexdigest()[:8]}-{secrets.token_hex(3)}"


def _objectives() -> tuple[ParetoObjective, ...]:
    return (
        ParetoObjective(metric="quality", direction="maximize", tolerance=1e-12),
        ParetoObjective(metric="serialized_size_bytes", direction="minimize"),
        ParetoObjective(metric="latency_ms", direction="minimize"),
        ParetoObjective(
            metric="throughput_tokens_per_second",
            direction="maximize",
        ),
    )


def compare_runs(
    run_directories: Sequence[str | Path],
    *,
    output_root: str | Path | None = None,
    unsafe_overrides: Set[str] | None = None,
) -> Path:
    """Compare candidate summaries against the first run under a strict contract."""

    if len(run_directories) < 2:
        raise ConfigurationError(
            "compare requires at least two completed run directories",
            component="comparison",
            remediation="Pass the baseline run first, followed by one or more candidates.",
        )
    directories = tuple(Path(value).expanduser().resolve() for value in run_directories)
    summaries = tuple(load_normalized_candidate(directory) for directory in directories)
    if len({summary.run_id for summary in summaries}) != len(summaries):
        raise ArtifactIntegrityError(
            "comparison inputs contain duplicate run IDs",
            component="comparison",
        )
    baseline = summaries[0]
    comparisons: tuple[ComparisonResult, ...] = tuple(
        compare_summaries(
            baseline,
            candidate,
            unsafe_overrides=unsafe_overrides,
        )
        for candidate in summaries[1:]
    )
    objectives = _objectives()
    pareto = pareto_frontier(
        pareto_points_from_summaries(summaries, objectives),
        objectives,
    )
    warnings = tuple(warning for comparison in comparisons for warning in comparison.warnings)
    report_data = ReportData(
        title="Inkling Quant Lab — Cross-run Comparison",
        runs=summaries,
        comparisons=comparisons,
        pareto=pareto,
        interpretations=(
            "Observed differences describe the recorded runs and do not establish causation.",
        ),
        reproduction_commands=(
            shell_join(("uv", "run", "iql", "compare", *(str(path) for path in directories))),
        ),
        warnings=warnings,
        metadata={
            "source_run_directories": [str(path) for path in directories],
            "unsafe_overrides": sorted(unsafe_overrides or set()),
        },
    )
    root = (
        directories[0].parent / "comparisons"
        if output_root is None
        else Path(output_root).expanduser().resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    store = LocalArtifactStore(root)
    identifier = _comparison_id(summaries)
    reporter = resolve_reporter()
    with store.staged_directory(identifier) as temporary:
        reporter.generate(report_data, temporary, include_plots=True)
        source_records = tuple(
            ComparisonSourceRecord(
                run_id=summary.run_id,
                run_directory=str(directory),
                manifest_sha256=sha256_file(directory / "manifest.json"),
                candidate_summary_sha256=sha256_file(directory / "reports/candidate_summary.json"),
            )
            for directory, summary in zip(directories, summaries, strict=True)
        )
        outputs = tuple(
            ArtifactChecksum(
                path=path.relative_to(temporary).as_posix(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
            for path in sorted(temporary.rglob("*"))
            if path.is_file() and not path.is_symlink()
        )
        bundle_manifest = ComparisonBundleManifest(
            comparison_id=identifier,
            source_runs=source_records,
            unsafe_overrides=tuple(sorted(unsafe_overrides or set())),
            outputs=outputs,
        )
        (temporary / "comparison_manifest.json").write_text(
            bundle_manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
    return store.path(identifier)


def _report_data_path(source: Path) -> Path:
    candidates = (
        source if source.is_file() else source / "report_data.json",
        source / "reports/report_data.json",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "report_data.json":
            return candidate
    raise ArtifactIntegrityError(
        f"No normalized report_data.json found under {source}",
        component="reporting",
        remediation="Run or resume the experiment through generate_reports first.",
    )


def _verify_comparison_bundle(source: Path) -> ComparisonBundleManifest:
    manifest_path = source / "comparison_manifest.json"
    try:
        manifest = ComparisonBundleManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise ArtifactIntegrityError(
            f"Unable to validate comparison bundle manifest {manifest_path}: {error}",
            component="comparison",
        ) from error
    if manifest.comparison_id != source.name:
        raise ArtifactIntegrityError(
            "Comparison bundle directory does not match its manifest ID",
            component="comparison",
        )
    output_paths = tuple(output.path for output in manifest.outputs)
    if len(set(output_paths)) != len(output_paths):
        raise ArtifactIntegrityError(
            "Comparison bundle manifest contains duplicate output paths",
            component="comparison",
        )
    expected_files = {"comparison_manifest.json", *output_paths}
    actual_files = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise ArtifactIntegrityError(
            "Comparison bundle file set does not match its manifest; "
            f"missing={sorted(expected_files - actual_files)}, "
            f"unexpected={sorted(actual_files - expected_files)}",
            component="comparison",
        )
    for output in manifest.outputs:
        path = (source / output.path).resolve(strict=False)
        if not path.is_relative_to(source) or not path.is_file() or path.is_symlink():
            raise ArtifactIntegrityError(
                f"Missing or unsafe comparison output: {output.path}",
                component="comparison",
            )
        if path.stat().st_size != output.size_bytes or sha256_file(path) != output.sha256:
            raise ArtifactIntegrityError(
                f"Checksum mismatch for comparison output: {output.path}",
                component="comparison",
            )
    return manifest


def _governed_report_source(source: Path) -> tuple[str, Path] | None:
    """Recognize governed layouts even when their required manifest was deleted."""

    def looks_like_run_root(root: Path) -> bool:
        run_markers = sum(
            (root / name).is_file()
            for name in (
                "status.json",
                "resolved_config.yaml",
                "environment.json",
                "events.jsonl",
            )
        )
        return (root / "manifest.json").is_file() or run_markers >= 2

    root = source
    if source.is_file():
        if source.name != "report_data.json":
            return None
        root = source.parent
        if root.name == "reports":
            run_root = root.parent
            if looks_like_run_root(run_root):
                return "run", run_root
    elif source.is_dir() and source.name == "reports":
        run_root = source.parent
        if (source / "report_data.json").is_file() and looks_like_run_root(run_root):
            return "run", run_root
    if (root / "reports/report_data.json").is_file():
        return "run", root
    if (root / "manifest.json").is_file() and looks_like_run_root(root):
        return "run", root
    if (root / "comparison_manifest.json").is_file() or (
        (root / "report_data.json").is_file()
        and _COMPARISON_ID_PATTERN.fullmatch(root.name) is not None
    ):
        return "comparison", root
    return None


def report_artifact(
    source: str | Path,
    *,
    destination: str | Path | None = None,
) -> Path:
    """Return an existing report or regenerate one only from normalized artifacts."""

    source_path = Path(source).expanduser().resolve()
    governed = _governed_report_source(source_path)
    if governed is not None and governed[0] == "run":
        run_root = governed[1]
        if not (run_root / "manifest.json").is_file():
            raise ArtifactIntegrityError(
                "Run report source is missing its governing manifest: "
                f"{run_root / 'manifest.json'}",
                component="reporting",
            )
        _, manifest, _ = validate_completed_run(run_root)
        report_outputs = {output.path for output in manifest.stages["generate_reports"].outputs}
        if "reports/report_data.json" not in report_outputs:
            raise ArtifactIntegrityError(
                "Successful run does not govern reports/report_data.json",
                component="reporting",
            )
    elif governed is not None:
        comparison_root = governed[1]
        if not (comparison_root / "comparison_manifest.json").is_file():
            raise ArtifactIntegrityError(
                "Comparison report source is missing its governing manifest: "
                f"{comparison_root / 'comparison_manifest.json'}",
                component="reporting",
            )
        _verify_comparison_bundle(comparison_root)
    data_path = _report_data_path(source_path)
    if governed is not None:
        expected_data_path = (
            governed[1] / "reports/report_data.json"
            if governed[0] == "run"
            else governed[1] / "report_data.json"
        )
        if data_path != expected_data_path:
            raise ArtifactIntegrityError(
                "Report data path does not match its governing artifact layout",
                component="reporting",
            )
    existing_report = data_path.parent / "report.md"
    if destination is None:
        if not existing_report.is_file():
            raise ArtifactIntegrityError(
                f"Normalized report data exists but report.md is missing: {existing_report}",
                component="reporting",
                remediation="Supply --destination to regenerate an immutable report bundle.",
            )
        return existing_report

    target = Path(destination).expanduser().resolve()
    if target.exists():
        raise ArtifactIntegrityError(
            f"Refusing to overwrite report destination: {target}",
            component="reporting",
        )
    data = ReportData.model_validate_json(data_path.read_text(encoding="utf-8"))
    resolve_reporter().generate(data, target, include_plots=True)
    return target / "report.md"


__all__ = [
    "ComparisonBundleManifest",
    "ComparisonSourceRecord",
    "compare_runs",
    "report_artifact",
]
