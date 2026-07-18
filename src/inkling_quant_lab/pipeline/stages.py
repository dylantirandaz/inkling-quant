"""Governed pipeline DAG, fingerprints, and atomic stage-result recording."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

from inkling_quant_lab.artifacts import LocalArtifactStore, sha256_file
from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.manifests import (
    ArtifactChecksum,
    RunManifest,
    StageRecord,
    StageStatus,
)

StageName: TypeAlias = Literal[
    "resolve_configuration",
    "probe_runtime",
    "load_baseline",
    "inventory_modules",
    "collect_statistics",
    "resolve_precision_policy",
    "quantize",
    "evaluate_baseline",
    "evaluate_candidate",
    "benchmark_baseline",
    "benchmark_candidate",
    "compare_routing",
    "generate_reports",
    "finalize_manifest",
]
DependencyChecksum: TypeAlias = str | ArtifactChecksum
StageProducer: TypeAlias = Callable[[Path], None]

STAGE_ORDER: tuple[StageName, ...] = (
    "resolve_configuration",
    "probe_runtime",
    "load_baseline",
    "inventory_modules",
    "collect_statistics",
    "resolve_precision_policy",
    "quantize",
    "evaluate_baseline",
    "evaluate_candidate",
    "benchmark_baseline",
    "benchmark_candidate",
    "compare_routing",
    "generate_reports",
    "finalize_manifest",
)

STAGE_DEPENDENCIES: dict[StageName, tuple[StageName, ...]] = {
    "resolve_configuration": (),
    "probe_runtime": ("resolve_configuration",),
    "load_baseline": ("probe_runtime",),
    "inventory_modules": ("load_baseline",),
    "collect_statistics": ("inventory_modules",),
    "resolve_precision_policy": ("collect_statistics",),
    "quantize": ("resolve_precision_policy",),
    "evaluate_baseline": ("load_baseline",),
    "evaluate_candidate": ("quantize",),
    "benchmark_baseline": ("load_baseline",),
    "benchmark_candidate": ("quantize",),
    "compare_routing": ("collect_statistics", "quantize"),
    "generate_reports": (
        "evaluate_baseline",
        "evaluate_candidate",
        "benchmark_baseline",
        "benchmark_candidate",
        "compare_routing",
    ),
    "finalize_manifest": ("generate_reports",),
}

# Each governed stage owns a disjoint canonical file or directory. The mapping
# lets resume distinguish a fully published-but-not-ledgered output from an
# in-progress temporary directory after an abrupt process termination.
STAGE_OUTPUT_LOCATIONS: dict[StageName, tuple[str, ...]] = {
    "resolve_configuration": ("resolved_config.yaml",),
    "probe_runtime": ("environment.json",),
    "load_baseline": ("checkpoints/baseline",),
    "inventory_modules": ("metrics/inventory",),
    "collect_statistics": ("metrics/statistics",),
    "resolve_precision_policy": ("checkpoints/policy",),
    "quantize": ("checkpoints/candidate",),
    "evaluate_baseline": ("metrics/evaluation_baseline",),
    "evaluate_candidate": ("metrics/evaluation_candidate",),
    "benchmark_baseline": ("metrics/benchmark_baseline",),
    "benchmark_candidate": ("metrics/benchmark_candidate",),
    "compare_routing": ("routing",),
    "generate_reports": ("reports",),
    "finalize_manifest": ("completion.json",),
}


@dataclass(frozen=True, slots=True)
class StageDefinition:
    """One stage's stable graph position and config-derived participation."""

    name: StageName
    dependencies: tuple[StageName, ...]
    enabled: bool
    required: bool


class StageResult(BaseModel):
    """Checksummed files committed atomically for one successful stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: StageName
    outputs: tuple[ArtifactChecksum, ...]


def stage_definitions(config: ExperimentConfig) -> tuple[StageDefinition, ...]:
    """Return all 14 stages with deterministic enabled/required decisions."""

    evaluation_enabled = bool(config.evaluation.suites)
    evaluation_required = any(not suite.optional for suite in config.evaluation.suites)
    routing_enabled = config.routing.mode != "off"
    routing_required = routing_enabled and config.routing.required
    statistics_enabled = (
        config.quantization.calibration is not None
        or routing_enabled
        or config.quantization.policy.type != "uniform"
    )
    statistics_required = (
        config.quantization.calibration is not None
        or routing_required
        or config.quantization.policy.type != "uniform"
    )
    benchmark_enabled = config.benchmark.enabled
    report_enabled = config.reporting.markdown or config.reporting.html or config.reporting.plots

    enabled: dict[StageName, bool] = {
        "resolve_configuration": True,
        "probe_runtime": True,
        "load_baseline": True,
        "inventory_modules": True,
        "collect_statistics": statistics_enabled,
        "resolve_precision_policy": True,
        "quantize": True,
        "evaluate_baseline": evaluation_enabled,
        "evaluate_candidate": evaluation_enabled,
        "benchmark_baseline": benchmark_enabled,
        "benchmark_candidate": benchmark_enabled,
        "compare_routing": routing_enabled,
        "generate_reports": report_enabled,
        "finalize_manifest": True,
    }
    required: dict[StageName, bool] = {
        "resolve_configuration": True,
        "probe_runtime": True,
        "load_baseline": True,
        "inventory_modules": True,
        "collect_statistics": statistics_required,
        "resolve_precision_policy": True,
        "quantize": True,
        "evaluate_baseline": evaluation_required,
        "evaluate_candidate": evaluation_required,
        "benchmark_baseline": benchmark_enabled,
        "benchmark_candidate": benchmark_enabled,
        "compare_routing": routing_required,
        "generate_reports": report_enabled,
        "finalize_manifest": True,
    }
    return tuple(
        StageDefinition(
            name=name,
            dependencies=STAGE_DEPENDENCIES[name],
            enabled=enabled[name],
            required=required[name],
        )
        for name in STAGE_ORDER
    )


def initial_stage_records(config: ExperimentConfig) -> dict[str, StageRecord]:
    """Build manifest records without executing or silently skipping any stage."""

    return {
        definition.name: StageRecord(
            name=definition.name,
            required=definition.required,
        )
        for definition in stage_definitions(config)
    }


def stage_descendants(name: StageName, *, include_self: bool = False) -> tuple[StageName, ...]:
    """Return transitive dependents in topological execution order."""

    affected: set[StageName] = {name}
    changed = True
    while changed:
        changed = False
        for candidate in STAGE_ORDER:
            if candidate in affected:
                continue
            if any(dependency in affected for dependency in STAGE_DEPENDENCIES[candidate]):
                affected.add(candidate)
                changed = True
    if not include_self:
        affected.remove(name)
    return tuple(candidate for candidate in STAGE_ORDER if candidate in affected)


def stage_fingerprint(
    name: StageName,
    config: ExperimentConfig,
    dependency_checksums: Mapping[str, Sequence[DependencyChecksum]],
) -> str:
    """Hash the resolved config and direct dependency output checksums."""

    expected = set(STAGE_DEPENDENCIES[name])
    provided = set(dependency_checksums)
    if expected != provided:
        missing = sorted(expected.difference(provided))
        extra = sorted(provided.difference(expected))
        raise ArtifactIntegrityError(
            f"fingerprint inputs for {name} have missing dependencies {missing} "
            f"and unexpected dependencies {extra}",
            component="pipeline_fingerprint",
        )

    dependencies: list[dict[str, object]] = []
    for dependency in STAGE_DEPENDENCIES[name]:
        checksums = sorted(
            _normalized_checksum(value) for value in dependency_checksums[dependency]
        )
        dependencies.append({"name": dependency, "output_checksums": checksums})
    payload = {
        "schema_version": "stage-fingerprint-v1",
        "stage": name,
        "config_hash": config.config_hash(),
        "dependencies": dependencies,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def stage_fingerprint_from_manifest(
    name: StageName, config: ExperimentConfig, manifest: RunManifest
) -> str:
    """Derive a fingerprint only from terminal direct dependency records."""

    dependency_checksums: dict[str, tuple[ArtifactChecksum, ...]] = {}
    for dependency in STAGE_DEPENDENCIES[name]:
        try:
            record = manifest.stages[dependency]
        except KeyError as error:
            raise ArtifactIntegrityError(
                f"manifest is missing dependency stage {dependency!r} for {name!r}",
                component="pipeline_fingerprint",
            ) from error
        if record.status is StageStatus.SUCCESS:
            dependency_checksums[dependency] = record.outputs
            continue
        if not record.required and record.status in {
            StageStatus.SKIPPED_NOT_REQUIRED,
            StageStatus.UNSUPPORTED,
        }:
            dependency_checksums[dependency] = ()
            continue
        raise ArtifactIntegrityError(
            f"dependency stage {dependency!r} is not complete for {name!r}: {record.status}",
            component="pipeline_fingerprint",
        )
    return stage_fingerprint(name, config, dependency_checksums)


def record_stage_result(
    run_directory: str | Path,
    name: StageName,
    output_paths: Sequence[str | Path],
) -> StageResult:
    """Validate and checksum already committed files under a run directory."""

    root = Path(run_directory).resolve()
    outputs: dict[str, ArtifactChecksum] = {}
    for value in output_paths:
        candidate = Path(value)
        path = candidate if candidate.is_absolute() else root / candidate
        if path.is_symlink():
            raise ArtifactIntegrityError(
                f"stage {name} output must not be a symlink: {value}",
                component="pipeline_stage_result",
            )
        resolved = path.resolve(strict=False)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ArtifactIntegrityError(
                f"stage {name} output is missing or outside the run directory: {value}",
                component="pipeline_stage_result",
            )
        relative = resolved.relative_to(root).as_posix()
        if relative in outputs:
            raise ArtifactIntegrityError(
                f"stage {name} recorded duplicate output: {relative}",
                component="pipeline_stage_result",
            )
        outputs[relative] = ArtifactChecksum(
            path=relative,
            sha256=sha256_file(resolved),
            size_bytes=resolved.stat().st_size,
        )
    return StageResult(
        name=name,
        outputs=tuple(outputs[path] for path in sorted(outputs)),
    )


def commit_stage_result(
    store: LocalArtifactStore,
    name: StageName,
    producer: StageProducer,
    *,
    relative_directory: str | Path | None = None,
    replace_empty_placeholder: bool = False,
) -> StageResult:
    """Validate/checksum temporary outputs, then publish the directory atomically."""

    relative = Path(relative_directory) if relative_directory is not None else Path("stages") / name
    with store.staged_directory(
        relative,
        replace_empty_placeholder=replace_empty_placeholder,
    ) as temporary:
        producer(temporary)
        paths = tuple(
            path for path in sorted(temporary.rglob("*")) if path.is_file() or path.is_symlink()
        )
        staged = record_stage_result(temporary, name, paths)
    return StageResult(
        name=name,
        outputs=tuple(
            output.model_copy(update={"path": (relative / output.path).as_posix()})
            for output in staged.outputs
        ),
    )


def _normalized_checksum(value: DependencyChecksum) -> str:
    checksum = value.sha256 if isinstance(value, ArtifactChecksum) else value
    if len(checksum) != 64 or any(character not in "0123456789abcdef" for character in checksum):
        raise ArtifactIntegrityError(
            f"invalid dependency checksum: {checksum!r}",
            component="pipeline_fingerprint",
        )
    return checksum


__all__ = [
    "STAGE_DEPENDENCIES",
    "STAGE_ORDER",
    "StageDefinition",
    "StageName",
    "StageResult",
    "commit_stage_result",
    "initial_stage_records",
    "record_stage_result",
    "stage_definitions",
    "stage_descendants",
    "stage_fingerprint",
    "stage_fingerprint_from_manifest",
]
