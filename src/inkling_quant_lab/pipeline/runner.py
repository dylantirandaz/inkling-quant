"""Governed 14-stage CPU experiment runner with immutable resume semantics."""

from __future__ import annotations

import csv
import html
import io
import json
import re
import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from shlex import join as shell_join
from typing import Any, cast

import torch
from pydantic import ValidationError

from inkling_quant_lab.artifacts import LocalArtifactStore
from inkling_quant_lab.benchmarking.latency import BenchmarkResult
from inkling_quant_lab.comparison import (
    ComparisonResult,
    NormalizedRunSummary,
    ParetoObjective,
    ParetoResult,
    compare_summaries,
    pareto_frontier,
    pareto_points_from_summaries,
)
from inkling_quant_lab.config import ExperimentConfig, dump_resolved_config, load_config
from inkling_quant_lab.evaluation.base import EvaluationResult, RequiredEvaluationFailure
from inkling_quant_lab.exceptions import (
    ArtifactIntegrityError,
    CapabilityError,
    ConfigurationError,
    InklingQuantError,
    RoutingInstrumentationError,
)
from inkling_quant_lab.hardware import probe_environment
from inkling_quant_lab.logging import EventLogger
from inkling_quant_lab.manifests import (
    GitProvenance,
    ModelProvenance,
    RunManifest,
    RunStatus,
    StageError,
    StageStatus,
    load_manifest,
    persist_manifest,
    utc_now,
)
from inkling_quant_lab.models.base import ModelDescriptor, SourceWeightFreeModelAdapter
from inkling_quant_lab.pipeline.benchmark_worker import run_isolated_benchmark
from inkling_quant_lab.pipeline.candidate_artifact import (
    load_governed_candidate_artifact,
)
from inkling_quant_lab.pipeline.operations import (
    Components,
    StatisticsBundle,
    benchmark_model,
    build_candidate,
    collect_statistics,
    compare_model_routing,
    create_components,
    evaluate_model,
    load_baseline,
    probe_capabilities,
    resolve_policy,
)
from inkling_quant_lab.pipeline.resume import (
    load_resume_config,
    mark_forced_archive_committed,
    preflight_resume_integrity,
    prepare_resume,
    recover_forced_archive_transactions,
    recover_published_stage_outputs,
    verify_successful_stage_fingerprints,
    verify_successful_stage_outputs,
)
from inkling_quant_lab.pipeline.routing_capture import RoutingUnsupportedError
from inkling_quant_lab.pipeline.stages import (
    STAGE_ORDER,
    StageDefinition,
    StageName,
    StageResult,
    commit_stage_result,
    initial_stage_records,
    record_stage_result,
    stage_definitions,
    stage_fingerprint_from_manifest,
)
from inkling_quant_lab.pipeline.summaries import build_run_summaries
from inkling_quant_lab.quantization.base import QuantizationManifest, QuantizedModel
from inkling_quant_lab.quantization.optional import reload_gptqmodel_cpu_export
from inkling_quant_lab.quantization.policies import ResolvedPrecisionPolicy
from inkling_quant_lab.quantization.reference import safe_model_serialized_size_bytes
from inkling_quant_lab.reporting.report import ReportData, resolve_reporter
from inkling_quant_lab.routing.metrics import (
    LayerRoutingMetrics,
    RoutingComparison,
    RoutingMetricSummary,
)
from inkling_quant_lab.routing.traces import RoutingArtifact
from inkling_quant_lab.security import Redactor, safe_path


@dataclass(slots=True)
class RunContext:
    """Mutable process-local context; persisted state remains immutable models/files."""

    config: ExperimentConfig
    run_directory: Path
    store: LocalArtifactStore
    logger: EventLogger
    redactor: Redactor
    project_root: Path
    components: Components | None = None


@dataclass(frozen=True, slots=True)
class StageExecution:
    """Committed outputs plus optional manifest provenance updates."""

    result: StageResult
    environment: dict[str, Any] | None = None
    model: ModelProvenance | None = None


def generate_run_id(
    config: ExperimentConfig,
    *,
    now: datetime | None = None,
    nonce: str | None = None,
) -> str:
    """Build a unique, sortable ID with deterministic injected inputs for tests."""

    timestamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    slug = re.sub(r"[^a-z0-9]+", "-", config.output.run_prefix.lower()).strip("-")
    slug = slug or "run"
    suffix = nonce or secrets.token_hex(3)
    return f"{slug}-{timestamp}-{config.config_hash()[:8]}-{suffix}"


def _provisional_environment(project_root: Path) -> dict[str, Any]:
    """Capture failure-safe host provenance before capability probing succeeds."""

    environment = probe_environment(project_root)
    environment["capability_probe"] = {
        "status": "pending",
        "reason": "the governed probe_runtime stage has not completed",
    }
    return environment


def _ensure_run_topology(
    store: LocalArtifactStore,
    config: ExperimentConfig,
    project_root: Path,
    *,
    environment: dict[str, Any] | None = None,
) -> None:
    """Materialize the mandatory failure-safe run contract without replacing evidence."""

    resolved_path = store.path("resolved_config.yaml")
    if not resolved_path.exists():
        store.write_text("resolved_config.yaml", dump_resolved_config(config))
    environment_path = store.path("environment.json")
    if not environment_path.exists():
        store.write_json(
            "environment.json",
            environment if environment is not None else _provisional_environment(project_root),
        )
    events_path = store.path("events.jsonl")
    if not events_path.exists():
        store.write_text("events.jsonl", "")
    for directory in ("metrics", "routing", "checkpoints", "reports"):
        store.ensure_directory(directory)


def run_experiment(
    config: ExperimentConfig,
    *,
    project_root: str | Path | None = None,
    run_id: str | None = None,
    inject_failure_stage: StageName | None = None,
) -> Path:
    """Create a new run directory and execute every enabled governed stage."""

    root = Path.cwd() if project_root is None else Path(project_root).resolve()
    secrets_in_memory = config.resolve_secrets()
    initial_environment = _provisional_environment(root)
    artifact_root = Path(config.output.root).expanduser()
    if not artifact_root.is_absolute():
        artifact_root = root / artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)
    identifier = generate_run_id(config) if run_id is None else run_id
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}", identifier) is None:
        raise ArtifactIntegrityError(
            "run_id must be a single safe path component containing only letters, digits, "
            "'.', '_', or '-'",
            component="pipeline",
        )
    run_directory = safe_path(artifact_root, identifier)
    if run_directory.exists():
        raise ValueError(f"Run directory already exists: {run_directory}")
    run_directory.mkdir(parents=False)
    store = LocalArtifactStore(run_directory)
    _ensure_run_topology(
        store,
        config,
        root,
        environment=initial_environment,
    )
    redactor = Redactor(tuple(secrets_in_memory.values()))
    logger = EventLogger(
        run_directory / "events.jsonl",
        run_id=identifier,
        model_id=config.model.model_id,
        config_hash=config.config_hash(),
        redactor=redactor,
    )
    manifest = RunManifest(
        run_id=identifier,
        config_hash=config.config_hash(),
        git=GitProvenance(
            commit=cast(str | None, cast(dict[str, Any], initial_environment["git"]).get("commit")),
            dirty=cast(bool | None, cast(dict[str, Any], initial_environment["git"]).get("dirty")),
        ),
        model=ModelProvenance(id=config.model.model_id, revision=config.model.revision),
        environment=initial_environment,
        stages=initial_stage_records(config),
    ).start()
    context = RunContext(
        config=config,
        run_directory=run_directory,
        store=store,
        logger=logger,
        redactor=redactor,
        project_root=root,
    )
    persist_manifest(store, manifest)
    _persist_status(store, manifest, current_stage=None)
    logger.emit(stage="run", event="run_started", message=f"Run created at {run_directory}")
    return _execute_pipeline(
        context,
        manifest,
        inject_failure_stage=inject_failure_stage,
    )


def run_experiment_file(
    config_path: str | Path,
    *,
    overrides: tuple[str, ...] = (),
    project_root: str | Path | None = None,
    inject_failure_stage: StageName | None = None,
) -> Path:
    """Load, resolve, and execute a checked-in YAML experiment."""

    config = load_config(config_path, overrides)
    return run_experiment(
        config,
        project_root=project_root,
        inject_failure_stage=inject_failure_stage,
    )


def resume_experiment(
    run_directory: str | Path,
    *,
    force_stage: StageName | None = None,
    inject_failure_stage: StageName | None = None,
    project_root: str | Path | None = None,
    allow_remote_code: bool = False,
) -> Path:
    """Verify successful evidence, optionally archive a subtree, and continue."""

    directory = Path(run_directory).expanduser().resolve()
    manifest = load_manifest(directory)
    config = load_resume_config(directory, manifest)
    if config.model.trust_remote_code and not allow_remote_code:
        raise ConfigurationError(
            "Resuming a run with remote model code requires explicit allow_remote_code=true",
            component="security",
        )
    secret_values = config.resolve_secrets()
    root = Path.cwd() if project_root is None else Path(project_root).expanduser().resolve()
    store = LocalArtifactStore(directory)
    preflight_resume_integrity(directory, manifest, config)
    recover_forced_archive_transactions(directory, manifest)
    manifest, recovered_stages = recover_published_stage_outputs(directory, manifest, config)
    if recovered_stages:
        recovery_store = LocalArtifactStore(directory)
        persist_manifest(recovery_store, manifest)
        _persist_status(recovery_store, manifest, current_stage=None)
    if manifest.status is RunStatus.SUCCESS and force_stage is None:
        return directory
    preparation = prepare_resume(
        directory,
        manifest,
        config,
        force_stage=force_stage,
    )
    redactor = Redactor(tuple(secret_values.values()))
    logger = EventLogger(
        directory / "events.jsonl",
        run_id=manifest.run_id,
        model_id=config.model.model_id,
        config_hash=config.config_hash(),
        redactor=redactor,
    )
    context = RunContext(
        config=config,
        run_directory=directory,
        store=store,
        logger=logger,
        redactor=redactor,
        project_root=root,
    )
    persist_manifest(store, preparation.manifest)
    if preparation.forced_archive_ledger is not None:
        mark_forced_archive_committed(directory, preparation.forced_archive_ledger)
    _ensure_run_topology(store, config, root)
    _persist_status(store, preparation.manifest, current_stage=None)
    logger.emit(
        stage="run",
        event="run_resumed",
        message=(
            f"Resuming run; preserved {len(preparation.skip_stages)} stages"
            + (f", forced {force_stage}" if force_stage else "")
        ),
        data={
            "skipped_stages": preparation.skip_stages,
            "restarted_stages": preparation.restart_stages,
            "invalidated_stages": preparation.invalidated_stages,
            "recovered_published_stages": recovered_stages,
        },
    )
    return _execute_pipeline(
        context,
        preparation.manifest,
        inject_failure_stage=inject_failure_stage,
    )


def _components(context: RunContext) -> Components:
    if context.components is None:
        context.components = create_components(context.config)
    return context.components


def _execute_pipeline(
    context: RunContext,
    manifest: RunManifest,
    *,
    inject_failure_stage: StageName | None,
) -> Path:
    definitions = {definition.name: definition for definition in stage_definitions(context.config)}
    torch.manual_seed(context.config.seed)
    try:
        for name in STAGE_ORDER:
            definition = definitions[name]
            record = manifest.stages[name]
            if record.status is StageStatus.SUCCESS:
                continue
            if (
                record.status
                in {
                    StageStatus.SKIPPED_NOT_REQUIRED,
                    StageStatus.UNSUPPORTED,
                }
                and not record.required
            ):
                continue
            if not definition.enabled:
                manifest = _ensure_pending(manifest, name)
                manifest = manifest.mark_stage(
                    name,
                    StageStatus.SKIPPED_NOT_REQUIRED,
                    "stage disabled by resolved configuration",
                )
                persist_manifest(context.store, manifest)
                _persist_status(context.store, manifest, current_stage=name)
                context.logger.emit(
                    stage=name,
                    event="stage_skipped",
                    message="Stage is not required by the resolved configuration",
                )
                continue
            fingerprint = stage_fingerprint_from_manifest(name, context.config, manifest)
            manifest = manifest.start_stage(name, fingerprint)
            persist_manifest(context.store, manifest)
            _persist_status(context.store, manifest, current_stage=name)
            context.logger.emit(
                stage=name,
                event="stage_started",
                message=f"Starting stage {name}",
                data={"fingerprint": fingerprint, "attempt": manifest.stages[name].attempt},
            )
            try:
                if inject_failure_stage == name:
                    raise RuntimeError(f"injected failure at stage {name}")
                execution = _execute_stage(context, definition)
            except (CapabilityError, RoutingUnsupportedError) as error:
                if not definition.required:
                    reason = context.redactor.text(error.message)
                    manifest = manifest.mark_stage(name, StageStatus.UNSUPPORTED, reason)
                    persist_manifest(context.store, manifest)
                    _persist_status(context.store, manifest, current_stage=name)
                    context.logger.emit(
                        stage=name,
                        event="stage_unsupported",
                        level="warning",
                        message=reason,
                        data=error.as_dict(),
                    )
                    continue
                manifest = _record_failure(context, manifest, name, error)
                raise
            except Exception as error:
                manifest = _record_failure(context, manifest, name, error)
                if isinstance(error, InklingQuantError):
                    raise
                raise InklingQuantError(
                    f"Stage {name} failed: {context.redactor.text(str(error))}",
                    stage=name,
                    component="pipeline",
                    details={"run_directory": str(context.run_directory)},
                ) from error
            manifest = manifest.finish_stage(name, execution.result.outputs)
            if execution.environment is not None:
                environment = execution.environment
                git_data = cast(dict[str, Any], environment.get("git", {}))
                manifest = manifest.model_copy(
                    update={
                        "environment": environment,
                        "git": GitProvenance(
                            commit=cast(str | None, git_data.get("commit")),
                            dirty=cast(bool | None, git_data.get("dirty")),
                        ),
                    }
                )
            if execution.model is not None:
                manifest = manifest.model_copy(update={"model": execution.model})
            persist_manifest(context.store, manifest)
            _persist_status(context.store, manifest, current_stage=name)
            context.logger.emit(
                stage=name,
                event="stage_succeeded",
                message=f"Completed stage {name}",
                data={"outputs": [item.path for item in execution.result.outputs]},
            )
        manifest = manifest.succeed()
        persist_manifest(context.store, manifest)
        _persist_status(context.store, manifest, current_stage=None)
        context.logger.emit(
            stage="run",
            event="run_succeeded",
            message="All required stages completed successfully",
        )
        return context.run_directory
    finally:
        if context.components is not None:
            context.components.runtime.cleanup()


def _ensure_pending(manifest: RunManifest, name: StageName) -> RunManifest:
    record = manifest.stages[name]
    if record.status is StageStatus.PENDING:
        return manifest
    stages = dict(manifest.stages)
    stages[name] = record.model_copy(
        update={
            "status": StageStatus.PENDING,
            "fingerprint": None,
            "started_at": None,
            "completed_at": None,
            "outputs": (),
            "error": None,
            "reason": None,
        }
    )
    return manifest.model_copy(update={"stages": stages})


def _record_failure(
    context: RunContext,
    manifest: RunManifest,
    name: StageName,
    error: Exception,
) -> RunManifest:
    if isinstance(error, InklingQuantError):
        error.details = {
            **error.details,
            "run_directory": str(context.run_directory),
        }
        payload = error.as_dict()
        code = error.code
        component = error.component
        remediation = error.remediation
    else:
        payload = {
            "code": "UNEXPECTED_STAGE_ERROR",
            "message": str(error),
            "type": f"{error.__class__.__module__}.{error.__class__.__qualname__}",
        }
        code = "UNEXPECTED_STAGE_ERROR"
        component = "pipeline"
        remediation = None
    redacted = cast(dict[str, Any], context.redactor.value(payload))
    failure_path = f"failures/{name}-attempt-{manifest.stages[name].attempt}.json"
    context.store.write_json(failure_path, redacted)
    stage_error = StageError(
        code=code,
        message=context.redactor.text(str(error)),
        component=component,
        remediation=(context.redactor.text(remediation) if remediation else None),
        details_path=failure_path,
    )
    failed = manifest.fail_stage(name, stage_error)
    failed = failed.fail()
    persist_manifest(context.store, failed)
    _persist_status(context.store, failed, current_stage=name, failure=redacted)
    context.logger.emit(
        stage=name,
        event="stage_failed",
        level="error",
        message=stage_error.message,
        data=redacted,
    )
    return failed


def _persist_status(
    store: LocalArtifactStore,
    manifest: RunManifest,
    *,
    current_stage: str | None,
    failure: dict[str, Any] | None = None,
) -> None:
    payload = {
        "schema_version": "1.0",
        "run_id": manifest.run_id,
        "status": manifest.status,
        "current_stage": current_stage,
        "updated_at": utc_now().isoformat(),
        "stages": {name: stage.status for name, stage in sorted(manifest.stages.items())},
        "failure": failure,
    }
    path = store.path("status.json")
    store.write_json("status.json", payload, replace=path.exists())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _commit_json(
    context: RunContext,
    name: StageName,
    directory: str,
    files: dict[str, Any],
) -> StageResult:
    def producer(temporary: Path) -> None:
        for filename, value in sorted(files.items()):
            _write_json(temporary / filename, value)

    return commit_stage_result(
        context.store,
        name,
        producer,
        relative_directory=directory,
    )


def _persist_failed_evaluation_report(
    context: RunContext,
    name: StageName,
    error: RequiredEvaluationFailure,
) -> None:
    """Persist typed required-suite evidence and an explicit failed-run report."""

    manifest = load_manifest(context.run_directory)
    attempt = manifest.stages[name].attempt
    kind = "candidate" if name == "evaluate_candidate" else "baseline"
    results_relative = f"metrics/evaluation_failures/{kind}/attempt-{attempt}/results.json"
    results_payload = [result.model_dump(mode="json") for result in error.results]
    redacted_results = cast(list[dict[str, Any]], context.redactor.value(results_payload))
    context.store.write_json(results_relative, redacted_results)
    report_stem = f"failure_reports/evaluation/{name}-attempt-{attempt}"
    report_json_relative = report_stem + ".json"
    report_markdown_relative = report_stem + ".md"
    report_payload = {
        "schema_version": "1.0",
        "run_id": context.logger.run_id,
        "config_hash": context.config.config_hash(),
        "run_status": "failed",
        "failed_stage": name,
        "failed_suites": list(error.failed_suites),
        "evaluation_results_path": results_relative,
        "evaluation_results": redacted_results,
        "completion_claim": "required evaluation failed; this is a failure report",
    }
    redacted_payload = cast(dict[str, Any], context.redactor.value(report_payload))
    context.store.write_json(report_json_relative, redacted_payload)
    rows = []
    for result in error.results:
        messages = "; ".join(failure.message for failure in result.failures) or "none"
        rows.append(
            "| "
            + " | ".join(
                value.replace("|", "\\|").replace("\n", " ")
                for value in (
                    result.evaluator_name,
                    result.status,
                    context.redactor.text(result.dataset_id),
                    str(result.sample_count),
                    context.redactor.text(messages),
                )
            )
            + " |"
        )
    markdown = (
        "# Inkling Quant Lab Failed-Run Evaluation Report\n\n"
        "This report records measured evaluation evidence from a run that is **failed**. "
        "It is not a successful completion claim.\n\n"
        f"- Run ID: `{context.logger.run_id}`\n"
        f"- Failed stage: `{name}`\n"
        f"- Required suites that failed: `{', '.join(error.failed_suites)}`\n"
        f"- Typed results: [`{results_relative}`](../../{results_relative})\n\n"
        "## Retained evaluation results\n\n"
        "| Evaluator | Status | Dataset | Samples | Failure |\n"
        "|---|---|---|---:|---|\n" + "\n".join(rows) + "\n"
    )
    context.store.write_text(report_markdown_relative, markdown)
    error.details = {
        **error.details,
        "evaluation_results_path": results_relative,
        "failure_report_json": report_json_relative,
        "failure_report_markdown": report_markdown_relative,
    }


def _execute_stage(context: RunContext, definition: StageDefinition) -> StageExecution:
    name = definition.name
    if name == "resolve_configuration":
        path = context.store.path("resolved_config.yaml")
        if path.exists():
            persisted = load_config(path)
            if persisted.config_hash() != context.config.config_hash():
                raise ArtifactIntegrityError(
                    "provisional resolved configuration does not match the active run",
                    component="pipeline",
                )
        else:
            path = context.store.write_text(
                "resolved_config.yaml", dump_resolved_config(context.config)
            )
        return StageExecution(record_stage_result(context.run_directory, name, (path,)))
    if name == "probe_runtime":
        components = _components(context)
        environment = probe_environment(context.project_root)
        capabilities = probe_capabilities(context.config, components)
        environment["capabilities"] = {
            "runtime": asdict(capabilities.runtime),
            "model": asdict(capabilities.model),
            "quantizer": capabilities.quantizer.model_dump(mode="json"),
        }
        environment["capability_probe"] = {"status": "success"}
        path = context.store.write_json("environment.json", environment, replace=True)
        return StageExecution(
            record_stage_result(context.run_directory, name, (path,)),
            environment=environment,
        )
    if name == "load_baseline":
        model = load_baseline(context.config, _components(context))
        descriptor = _descriptor_dict(model.descriptor)
        if context.config.benchmark.host_memory_mode == "isolated_subject_artifact_peak_rss":
            descriptor["serialized_size_bytes"] = safe_model_serialized_size_bytes(model)
        result = _commit_json(
            context, name, "checkpoints/baseline", {"descriptor.json": descriptor}
        )
        return StageExecution(
            result,
            model=ModelProvenance(
                id=model.descriptor.model_id,
                revision=model.descriptor.revision,
                resolved_class=model.descriptor.resolved_class,
                architecture=model.descriptor.architecture,
                checksum=model.descriptor.checksum,
            ),
        )
    if name == "inventory_modules":
        components = _components(context)
        model = load_baseline(context.config, components)
        inventory = components.adapter.enumerate_modules(model)
        moe = components.adapter.discover_moe(model)
        return StageExecution(
            _commit_json(
                context,
                name,
                "metrics/inventory",
                {
                    "modules.json": [asdict(module) for module in inventory],
                    "moe_descriptor.json": None if moe is None else asdict(moe),
                },
            )
        )
    if name == "collect_statistics":
        components = _components(context)
        model = load_baseline(context.config, components)
        inventory = components.adapter.enumerate_modules(model)
        statistics = collect_statistics(context.config, components, model, inventory)
        return StageExecution(
            _commit_json(
                context,
                name,
                "metrics/statistics",
                {"statistics.json": statistics.as_dict()},
            )
        )
    if name == "resolve_precision_policy":
        components = _components(context)
        model = load_baseline(context.config, components)
        inventory = components.adapter.enumerate_modules(model)
        statistics = _load_statistics(context)
        policy = resolve_policy(context.config, components, model, inventory, statistics)
        return StageExecution(
            _commit_json(
                context,
                name,
                "checkpoints/policy",
                {"resolved_policy.json": policy.model_dump(mode="json")},
            )
        )
    if name == "quantize":
        return StageExecution(_quantize_stage(context, name))
    if name in {"evaluate_baseline", "evaluate_candidate"}:
        candidate = name == "evaluate_candidate"
        model = (
            _candidate(context).loaded
            if candidate
            else load_baseline(context.config, _components(context))
        )
        try:
            results = evaluate_model(context.config, _components(context), model)
        except RequiredEvaluationFailure as error:
            _persist_failed_evaluation_report(context, name, error)
            raise
        directory = "metrics/evaluation_candidate" if candidate else "metrics/evaluation_baseline"
        serialized_results = cast(
            list[dict[str, Any]],
            context.redactor.value([result.model_dump(mode="json") for result in results]),
        )
        return StageExecution(
            _commit_json(
                context,
                name,
                directory,
                {"results.json": serialized_results},
            )
        )
    if name in {"benchmark_baseline", "benchmark_candidate"}:
        candidate = name == "benchmark_candidate"
        if context.config.benchmark.host_memory_mode in {
            "isolated_stage_worker_peak_rss",
            "isolated_subject_artifact_peak_rss",
        }:
            benchmark_result = run_isolated_benchmark(
                context.config,
                context.run_directory,
                candidate=candidate,
                project_root=context.project_root,
            )
        else:
            quantized = _candidate(context) if candidate else None
            model = (
                quantized.loaded
                if quantized is not None
                else load_baseline(context.config, _components(context))
            )
            size = quantized.manifest.serialized_size_bytes if quantized is not None else None
            benchmark_result = benchmark_model(
                context.config,
                _components(context),
                model,
                serialized_size_bytes=size,
            )
        directory = "metrics/benchmark_candidate" if candidate else "metrics/benchmark_baseline"
        return StageExecution(
            _commit_json(
                context,
                name,
                directory,
                {"benchmark.json": benchmark_result.model_dump(mode="json")},
            )
        )
    if name == "compare_routing":
        return StageExecution(_routing_stage(context, name))
    if name == "generate_reports":
        return StageExecution(_report_stage(context, name))
    if name == "finalize_manifest":
        for directory in ("metrics", "routing", "checkpoints", "reports"):
            context.store.ensure_directory(directory)
        path = context.store.write_json(
            "completion.json",
            {
                "schema_version": "1.0",
                "run_id": context.logger.run_id,
                "config_hash": context.config.config_hash(),
                "completed_required_stages": [
                    stage.name for stage in stage_definitions(context.config) if stage.required
                ],
                "completion_claim": "all required stages succeeded before final run transition",
            },
        )
        return StageExecution(record_stage_result(context.run_directory, name, (path,)))
    raise AssertionError(f"No executor for governed stage {name}")


def _descriptor_dict(descriptor: ModelDescriptor) -> dict[str, Any]:
    return asdict(descriptor)


def _load_statistics(context: RunContext) -> StatisticsBundle:
    path = context.run_directory / "metrics/statistics/statistics.json"
    if not path.exists():
        return StatisticsBundle(
            usage={},
            sensitivity={},
            sensitivity_details={},
            calibration_dataset_sha256=None,
            calibration_sample_ids=(),
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return StatisticsBundle(
        usage={str(key): float(value) for key, value in data["usage"].items()},
        sensitivity={str(key): float(value) for key, value in data["sensitivity"].items()},
        sensitivity_details=cast(
            dict[str, dict[str, float | int | str]], data["sensitivity_details"]
        ),
        calibration_dataset_sha256=cast(str | None, data["calibration_dataset_sha256"]),
        calibration_sample_ids=tuple(str(item) for item in data["calibration_sample_ids"]),
    )


def _load_policy(context: RunContext) -> ResolvedPrecisionPolicy:
    return ResolvedPrecisionPolicy.model_validate_json(
        (context.run_directory / "checkpoints/policy/resolved_policy.json").read_text(
            encoding="utf-8"
        )
    )


def _candidate(context: RunContext) -> QuantizedModel:
    components = _components(context)
    if context.config.benchmark.host_memory_mode == "isolated_subject_artifact_peak_rss":
        if not isinstance(components.adapter, SourceWeightFreeModelAdapter):
            raise CapabilityError(
                "configured adapter cannot construct a source-weight-free export shell",
                component="pipeline",
            )
        candidate, _ = load_governed_candidate_artifact(
            context.run_directory,
            context.config,
            components.adapter,
            components.runtime,
        )
        return candidate
    baseline = load_baseline(context.config, components)
    if (
        context.config.quantization.backend == "gptq"
        and context.config.quantization.method == "gptq"
        and context.config.quantization.parameters.get("device") == "cpu"
        and context.config.quantization.export.enabled
    ):
        persisted_manifest = _load_quantization_manifest(context)
        export_path = (
            context.run_directory
            / "checkpoints/candidate"
            / context.config.quantization.export.destination
        )
        return reload_gptqmodel_cpu_export(
            export_path,
            baseline,
            persisted_manifest,
            context.config.quantization,
        )
    candidate = build_candidate(
        context.config,
        components,
        baseline,
        _load_policy(context),
        _load_statistics(context),
    )
    persisted_manifest = _load_quantization_manifest(context)
    reconstructed_manifest = candidate.manifest.model_copy(
        update={"serialized_size_bytes": persisted_manifest.serialized_size_bytes}
    )
    if reconstructed_manifest != persisted_manifest:
        raise ArtifactIntegrityError(
            "candidate reconstruction differs from the persisted quantization manifest "
            "outside the export-derived serialized size",
            component="pipeline",
        )
    return candidate


def _quantize_stage(context: RunContext, name: StageName) -> StageResult:
    components = _components(context)
    baseline = load_baseline(context.config, components)
    policy = _load_policy(context)
    candidate = build_candidate(
        context.config,
        components,
        baseline,
        policy,
        _load_statistics(context),
    )

    def producer(temporary: Path) -> None:
        if context.config.quantization.export.enabled:
            destination = temporary / context.config.quantization.export.destination
            destination.parent.mkdir(parents=True, exist_ok=True)
            components.quantizer.export(candidate, destination, context.config.quantization)
        _write_json(
            temporary / "quantization_manifest.json",
            candidate.manifest.model_dump(mode="json"),
        )

    return commit_stage_result(
        context.store,
        name,
        producer,
        relative_directory="checkpoints/candidate",
    )


def _persistable_routing(artifact: RoutingArtifact, context: RunContext) -> RoutingArtifact:
    if artifact.mode != context.config.routing.mode:
        raise RoutingInstrumentationError(
            "routing capture mode differs from the resolved configuration; refusing to hide "
            f"capture promotion ({context.config.routing.mode} -> {artifact.mode})",
            component="routing_capture",
        )
    return artifact


def _routing_stage(context: RunContext, name: StageName) -> StageResult:
    components = _components(context)
    baseline = load_baseline(context.config, components)
    candidate = _candidate(context).loaded
    baseline_routes, candidate_routes, comparison = compare_model_routing(
        context.config, components, baseline, candidate
    )

    def producer(temporary: Path) -> None:
        _persistable_routing(baseline_routes, context).write(temporary / "baseline")
        _persistable_routing(candidate_routes, context).write(temporary / "candidate")
        _write_json(temporary / "comparison.json", comparison.as_dict())

    return commit_stage_result(
        context.store,
        name,
        producer,
        relative_directory="routing",
        replace_empty_placeholder=True,
    )


def _load_evaluations(context: RunContext, candidate: bool) -> tuple[EvaluationResult, ...]:
    kind = "candidate" if candidate else "baseline"
    data = json.loads(
        (context.run_directory / f"metrics/evaluation_{kind}/results.json").read_text(
            encoding="utf-8"
        )
    )
    return tuple(EvaluationResult.model_validate(item) for item in data)


def _load_benchmark(context: RunContext, candidate: bool) -> BenchmarkResult | None:
    kind = "candidate" if candidate else "baseline"
    path = context.run_directory / f"metrics/benchmark_{kind}/benchmark.json"
    if not path.exists():
        return None
    return BenchmarkResult.model_validate_json(path.read_text(encoding="utf-8"))


def _load_routing_comparison(context: RunContext) -> RoutingComparison | None:
    path = context.run_directory / "routing/comparison.json"
    if not path.exists():
        return None
    data = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    per_layer: list[LayerRoutingMetrics] = []
    for raw_layer in cast(list[dict[str, Any]], data["per_layer"]):
        layer = dict(raw_layer)
        layer["baseline_selection_frequency"] = tuple(
            cast(list[float], layer["baseline_selection_frequency"])
        )
        layer["candidate_selection_frequency"] = tuple(
            cast(list[float], layer["candidate_selection_frequency"])
        )
        per_layer.append(LayerRoutingMetrics(**layer))
    return RoutingComparison(
        per_layer=tuple(per_layer),
        per_layer_drift_ranking=tuple(cast(list[str], data["per_layer_drift_ranking"])),
        macro=RoutingMetricSummary(**cast(dict[str, Any], data["macro"])),
        weighted=RoutingMetricSummary(**cast(dict[str, Any], data["weighted"])),
    )


def _load_routing_artifact(context: RunContext, candidate: bool) -> RoutingArtifact | None:
    """Load the exact persisted routing evidence for one side of a run."""

    kind = "candidate" if candidate else "baseline"
    path = context.run_directory / f"routing/{kind}/aggregates.json"
    if not path.exists():
        return None
    return RoutingArtifact.read(path.parent)


def _load_quantization_manifest(context: RunContext) -> QuantizationManifest:
    path = context.run_directory / "checkpoints/candidate/quantization_manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ArtifactIntegrityError(
            "persisted candidate quantization manifest is missing or unsafe",
            component="pipeline",
        )
    try:
        return QuantizationManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValidationError) as error:
        raise ArtifactIntegrityError(
            f"persisted candidate quantization manifest is invalid: {error}",
            component="pipeline",
        ) from error


def _routing_detail_csv(comparison: RoutingComparison | None) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        (
            "layer_id",
            "js_divergence",
            "top_k_overlap",
            "route_agreement",
            "baseline_load_imbalance",
            "candidate_load_imbalance",
        )
    )
    if comparison is not None:
        for layer in comparison.per_layer:
            writer.writerow(
                (
                    layer.layer_id,
                    format(layer.js_divergence, ".12g"),
                    "unavailable"
                    if layer.top_k_overlap is None
                    else format(layer.top_k_overlap, ".12g"),
                    "unavailable"
                    if layer.token_route_agreement is None
                    else format(layer.token_route_agreement, ".12g"),
                    format(layer.baseline_load_imbalance, ".12g"),
                    format(layer.candidate_load_imbalance, ".12g"),
                )
            )
    return output.getvalue()


def _prior_evaluation_failure_artifacts(context: RunContext) -> tuple[str, ...]:
    """List immutable evidence retained from required evaluator attempts."""

    artifacts: list[str] = []
    for relative_root in ("failure_reports/evaluation", "metrics/evaluation_failures"):
        root = context.run_directory / relative_root
        if not root.is_dir():
            continue
        artifacts.extend(
            path.relative_to(context.run_directory).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
    return tuple(sorted(artifacts))


def _report_stage(context: RunContext, name: StageName) -> StageResult:
    components = _components(context)
    baseline = load_baseline(context.config, components)
    policy = _load_policy(context)
    quantization_manifest = _load_quantization_manifest(context)
    routing = _load_routing_comparison(context)
    environment = cast(
        dict[str, Any],
        json.loads((context.run_directory / "environment.json").read_text(encoding="utf-8")),
    )
    baseline_summary, candidate_summary = build_run_summaries(
        run_id=context.logger.run_id,
        run_directory=context.run_directory,
        config=context.config,
        descriptor=baseline.descriptor,
        policy=policy,
        quantization_manifest=quantization_manifest,
        baseline_evaluations=_load_evaluations(context, False),
        candidate_evaluations=_load_evaluations(context, True),
        baseline_benchmark=_load_benchmark(context, False),
        candidate_benchmark=_load_benchmark(context, True),
        routing=routing,
        baseline_routing_artifact=_load_routing_artifact(context, False),
        candidate_routing_artifact=_load_routing_artifact(context, True),
        environment=environment,
    )
    comparison = compare_summaries(baseline_summary, candidate_summary)
    objectives = (
        ParetoObjective(metric="quality", direction="maximize", tolerance=1e-12),
        ParetoObjective(metric="serialized_size_bytes", direction="minimize"),
        ParetoObjective(metric="latency_ms", direction="minimize"),
        ParetoObjective(metric="throughput_tokens_per_second", direction="maximize"),
    )
    summaries = (baseline_summary, candidate_summary)
    pareto = pareto_frontier(pareto_points_from_summaries(summaries, objectives), objectives)
    interpretation = (
        "Observed differences describe this declared fixture, seed, software, and hardware; "
        "they do not establish causation."
    )
    interpretations = (interpretation,) if context.config.reporting.include_interpretation else ()
    prior_failure_artifacts = _prior_evaluation_failure_artifacts(context)
    prior_failure_reports = tuple(
        path
        for path in prior_failure_artifacts
        if path.startswith("failure_reports/evaluation/") and path.endswith(".md")
    )
    prior_failure_warning = (
        (
            "This run succeeded after one or more required evaluation attempts failed; "
            "the prior failure evidence remains linked in Prior Failed Evaluation Attempts."
        )
        if prior_failure_reports
        else None
    )
    data = ReportData(
        title=f"Inkling Quant Lab — {context.config.name}",
        runs=summaries,
        comparisons=(comparison,),
        pareto=pareto,
        interpretations=interpretations,
        reproduction_commands=(
            shell_join(("uv", "run", "iql", "resume", str(context.run_directory))),
            shell_join(("uv", "run", "iql", "report", str(context.run_directory))),
        ),
        warnings=quantization_manifest.warnings
        + (() if prior_failure_warning is None else (prior_failure_warning,)),
        metadata={
            "run_id": context.logger.run_id,
            "config_hash": context.config.config_hash(),
            "routing_comparison": (
                None if routing is None else cast(Any, json.loads(json.dumps(routing.as_dict())))
            ),
            "prior_evaluation_failure_artifacts": list(prior_failure_artifacts),
        },
    )
    reporter = resolve_reporter()

    def producer(temporary: Path) -> None:
        reporter.generate(
            data,
            temporary,
            include_plots=context.config.reporting.plots,
        )
        _write_json(temporary / "baseline_summary.json", baseline_summary.model_dump(mode="json"))
        _write_json(temporary / "candidate_summary.json", candidate_summary.model_dump(mode="json"))
        _write_json(temporary / "comparison.json", comparison.model_dump(mode="json"))
        _write_json(temporary / "pareto.json", pareto.model_dump(mode="json"))
        detail_csv = _routing_detail_csv(routing)
        (temporary / "tables/routing_drift.csv").write_text(detail_csv, encoding="utf-8")
        report_path = temporary / "report.md"
        report_path.write_text(
            report_path.read_text(encoding="utf-8")
            + "\n## Routing Layer Detail\n\n"
            + "Measured per-layer source data: [routing_drift.csv](tables/routing_drift.csv).\n"
            + (
                "\n## Prior Failed Evaluation Attempts\n\n"
                "This completed run succeeded only after earlier required evaluation attempts "
                "failed. Their typed evidence and failure reports remain immutable and linked "
                "below; they were not discarded on resume.\n\n"
                + "\n".join(f"- [`{path}`](../{path})" for path in prior_failure_artifacts)
                + "\n"
                if prior_failure_artifacts
                else ""
            ),
            encoding="utf-8",
        )
        if context.config.reporting.html:
            markdown = report_path.read_text(encoding="utf-8")
            (temporary / "report.html").write_text(
                '<!doctype html><meta charset="utf-8"><title>'
                + html.escape(data.title)
                + "</title><pre>"
                + html.escape(markdown)
                + "</pre>\n",
                encoding="utf-8",
            )

    return commit_stage_result(
        context.store,
        name,
        producer,
        relative_directory="reports",
        replace_empty_placeholder=True,
    )


def load_normalized_candidate(run_directory: str | Path) -> NormalizedRunSummary:
    """Load a checksum-verified candidate summary from a successful governed run."""

    directory, manifest, _ = validate_completed_run(run_directory)
    relative = "reports/candidate_summary.json"
    report_stage = manifest.stages["generate_reports"]
    if relative not in {output.path for output in report_stage.outputs}:
        raise ArtifactIntegrityError(
            f"Successful generate_reports stage does not govern {relative}",
            component="comparison",
        )
    path = directory / relative
    try:
        summary = NormalizedRunSummary.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ArtifactIntegrityError(
            f"Unable to validate normalized candidate summary {path}: {error}",
            component="comparison",
        ) from error
    if summary.run_id != f"{manifest.run_id}:candidate":
        raise ArtifactIntegrityError(
            f"Candidate summary run ID {summary.run_id!r} does not match manifest "
            f"{manifest.run_id!r}",
            component="comparison",
        )
    if Path(summary.artifact_path).expanduser().resolve() != directory:
        raise ArtifactIntegrityError(
            "Candidate summary artifact path does not match its source run directory",
            component="comparison",
        )
    if summary.model_id != manifest.model.id or summary.model_revision != manifest.model.revision:
        raise ArtifactIntegrityError(
            "Candidate summary model identity does not match its source manifest",
            component="comparison",
        )
    return summary


def validate_completed_run(
    run_directory: str | Path,
) -> tuple[Path, RunManifest, ExperimentConfig]:
    """Prove a run is complete and every governed output/fingerprint is intact."""

    directory = Path(run_directory).expanduser().resolve()
    manifest = load_manifest(directory)
    if manifest.status is not RunStatus.SUCCESS:
        raise ArtifactIntegrityError(
            f"Comparison/report source run is not successful: {manifest.status}",
            component="comparison",
        )
    if set(manifest.stages) != set(STAGE_ORDER):
        raise ArtifactIntegrityError(
            "Completed run manifest does not contain the exact governed stage set",
            component="comparison",
        )
    incomplete = [
        name
        for name, stage in manifest.stages.items()
        if stage.required and stage.status is not StageStatus.SUCCESS
    ]
    if incomplete:
        raise ArtifactIntegrityError(
            "Completed run has non-successful required stages: " + ", ".join(sorted(incomplete)),
            component="comparison",
        )
    config = load_resume_config(directory, manifest)
    verify_successful_stage_outputs(directory, manifest)
    verify_successful_stage_fingerprints(config, manifest)
    for name in ("generate_reports", "finalize_manifest"):
        if manifest.stages[name].status is not StageStatus.SUCCESS:
            raise ArtifactIntegrityError(
                f"Completed comparison source requires successful stage {name}",
                component="comparison",
            )
    return directory, manifest, config


def load_internal_comparison(run_directory: str | Path) -> ComparisonResult:
    """Load the baseline-versus-candidate comparison generated by a run."""

    path = Path(run_directory).expanduser().resolve() / "reports/comparison.json"
    return ComparisonResult.model_validate_json(path.read_text(encoding="utf-8"))


def load_pareto(run_directory: str | Path) -> ParetoResult:
    """Load persisted within-run Pareto membership."""

    path = Path(run_directory).expanduser().resolve() / "reports/pareto.json"
    return ParetoResult.model_validate_json(path.read_text(encoding="utf-8"))
