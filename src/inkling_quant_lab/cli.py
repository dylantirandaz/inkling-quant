"""Typer command surface for reproducible local experiment workflows."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, NoReturn, cast

import typer
from pydantic import ValidationError

from inkling_quant_lab.bootstrap import register_builtins
from inkling_quant_lab.config import (
    ExperimentConfig,
    load_config,
    load_inspection_config,
)
from inkling_quant_lab.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
    InklingQuantError,
)
from inkling_quant_lab.hardware import probe_environment
from inkling_quant_lab.registry import (
    EVALUATORS,
    MODEL_ADAPTERS,
    QUANTIZERS,
    REPORTERS,
    RUNTIMES,
    Registry,
)
from inkling_quant_lab.security import Redactor, safe_path

app = typer.Typer(
    name="iql",
    help=(
        "Run reproducible model experiments. "
        "Quantize models. "
        "Evaluate models. "
        "Compare runs. "
        "Create reports."
    ),
    add_completion=False,
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

_EXIT_CODES = {
    "CONFIGURATION_ERROR": 2,
    "CAPABILITY_ERROR": 3,
    "MODEL_LOAD_ERROR": 4,
    "QUANTIZATION_ERROR": 4,
    "ROUTING_INSTRUMENTATION_ERROR": 4,
    "EVALUATION_ERROR": 4,
    "BENCHMARK_ERROR": 4,
    "ARTIFACT_INTEGRITY_ERROR": 5,
    "COMPARISON_COMPATIBILITY_ERROR": 6,
}


def _json_document(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _emit_success(payload: dict[str, Any], human: str, *, json_output: bool) -> None:
    if json_output:
        typer.echo(_json_document(payload))
    else:
        typer.echo(human)


def _normalize_error(error: Exception) -> InklingQuantError:
    if isinstance(error, InklingQuantError):
        return error
    if isinstance(error, ValidationError):
        details = [
            {
                "field": ".".join(str(item) for item in issue["loc"]),
                "message": issue["msg"],
            }
            for issue in error.errors(include_url=False)
        ]
        return ConfigurationError(
            "Artifact validation failed: "
            + "; ".join(f"{item['field']}: {item['message']}" for item in details),
            component="cli",
            details={"errors": details},
        )
    if isinstance(error, FileNotFoundError):
        filename = error.filename or str(error)
        return ArtifactIntegrityError(
            f"Required file or directory does not exist: {filename}",
            component="cli",
            remediation="Verify the supplied path and that its producing stage completed.",
            details={"path": filename},
        )
    if isinstance(error, OSError):
        return ArtifactIntegrityError(
            f"Unable to access a required artifact: {error}",
            component="cli",
            remediation="Check the path, permissions, and available storage.",
        )
    if isinstance(error, ValueError):
        return ConfigurationError(str(error), component="cli")
    return InklingQuantError(
        f"Unexpected {error.__class__.__name__}: {error}",
        component="cli",
        remediation="Inspect the run status and event log, then retry or resume the run.",
        details={"exception_type": f"{error.__class__.__module__}.{error.__class__.__qualname__}"},
    )


def _abort(
    error: Exception,
    *,
    json_output: bool,
    run_directory: Path | None = None,
    redactor: Redactor | None = None,
) -> NoReturn:
    normalized = _normalize_error(error)
    error_payload = normalized.as_dict()
    details = cast(dict[str, Any], error_payload["details"])

    effective_directory = run_directory
    if effective_directory is None:
        recorded = details.get("run_directory")
        if isinstance(recorded, str):
            effective_directory = Path(recorded).expanduser().resolve()
    if effective_directory is not None:
        effective_directory = effective_directory.expanduser().resolve()
        details.setdefault("run_directory", str(effective_directory))
        details.setdefault("status_path", str(effective_directory / "status.json"))

    raw_payload: dict[str, Any] = {"status": "error", "error": error_payload}
    if effective_directory is not None:
        raw_payload["run_directory"] = str(effective_directory)
        raw_payload["status_path"] = str(effective_directory / "status.json")

    active_redactor = redactor or Redactor()
    payload = cast(dict[str, Any], active_redactor.value(raw_payload))
    safe_error = cast(dict[str, Any], payload["error"])

    if json_output:
        typer.echo(_json_document(payload))
    else:
        lines = [f"Error [{normalized.code}]: {safe_error['message']}"]
        if safe_error.get("stage"):
            lines.append(f"Stage: {safe_error['stage']}")
        if safe_error.get("component"):
            lines.append(f"Component: {safe_error['component']}")
        if effective_directory is not None:
            lines.append(f"Run directory: {payload['run_directory']}")
            lines.append(f"Status artifact: {payload['status_path']}")
        if safe_error.get("remediation"):
            lines.append(f"Remediation: {safe_error['remediation']}")
        typer.echo("\n".join(lines), err=True)
    raise typer.Exit(code=_EXIT_CODES.get(normalized.code, 1))


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _redactor_for_config(config: ExperimentConfig) -> Redactor:
    values = tuple(
        value
        for reference in config.security.secrets.values()
        if (value := os.environ.get(reference.env))
    )
    return Redactor(values)


def _redactor_for_run(run_directory: Path) -> Redactor:
    try:
        config = load_config(run_directory / "resolved_config.yaml")
    except InklingQuantError:
        return Redactor()
    return _redactor_for_config(config)


def _inspection_target_label(target: str) -> str:
    path = Path(target).expanduser()
    if path.exists() or path.suffix.lower() in {".yaml", ".yml"}:
        return str(path.resolve())
    return target


def _generate_run_id(config: ExperimentConfig) -> str:
    from inkling_quant_lab.pipeline.runner import generate_run_id

    return generate_run_id(config)


def _run_pipeline(
    config: ExperimentConfig,
    *,
    run_id: str,
    project_root: Path,
) -> Path:
    from inkling_quant_lab.pipeline.runner import run_experiment

    return run_experiment(config, run_id=run_id, project_root=project_root)


def _resume_pipeline(
    run_directory: Path,
    *,
    force_stage: str | None,
    project_root: Path,
    allow_remote_code: bool,
) -> Path:
    from inkling_quant_lab.pipeline.runner import resume_experiment
    from inkling_quant_lab.pipeline.stages import StageName

    return resume_experiment(
        run_directory,
        force_stage=cast(StageName | None, force_stage),
        project_root=project_root,
        allow_remote_code=allow_remote_code,
    )


def _compare_artifacts(
    run_directories: tuple[Path, ...],
    *,
    output_root: Path | None,
    unsafe_overrides: set[str] | None,
) -> Path:
    from inkling_quant_lab.pipeline.comparison_artifacts import compare_runs

    return compare_runs(
        run_directories,
        output_root=output_root,
        unsafe_overrides=unsafe_overrides,
    )


def _report_artifact(source: Path, *, destination: Path | None) -> Path:
    from inkling_quant_lab.pipeline.comparison_artifacts import report_artifact

    return report_artifact(source, destination=destination)


def _planned_run_directory(config: ExperimentConfig, run_id: str, project_root: Path) -> Path:
    artifact_root = Path(config.output.root).expanduser()
    if not artifact_root.is_absolute():
        artifact_root = project_root / artifact_root
    return safe_path(artifact_root, run_id)


def _execution_overrides(
    set_values: list[str] | None,
    *,
    allow_remote_code: bool,
) -> tuple[str, ...]:
    values = list(set_values or ())
    if allow_remote_code:
        values.extend(
            (
                "model.trust_remote_code=true",
                "security.allow_remote_code=true",
            )
        )
    return tuple(values)


def _require_remote_code_cli_opt_in(
    config: ExperimentConfig,
    *,
    allow_remote_code: bool,
) -> None:
    if config.model.trust_remote_code and not allow_remote_code:
        raise ConfigurationError(
            "Remote model code is enabled in configuration but the execution command did not "
            "include --allow-remote-code",
            component="security",
            remediation=(
                "Audit and pin the model repository, then pass --allow-remote-code explicitly."
            ),
        )


def _backend_payload() -> dict[str, list[dict[str, Any]]]:
    register_builtins()
    registries: tuple[tuple[str, Registry[Any]], ...] = (
        ("model_adapters", MODEL_ADAPTERS),
        ("quantizers", QUANTIZERS),
        ("runtimes", RUNTIMES),
        ("evaluators", EVALUATORS),
        ("reporters", REPORTERS),
    )
    return {
        name: [asdict(descriptor) for descriptor in registry.descriptors()]
        for name, registry in registries
    }


def _backend_lines(backends: dict[str, list[dict[str, Any]]]) -> list[str]:
    lines: list[str] = []
    for category, descriptors in backends.items():
        lines.append(category.replace("_", " ").title() + ":")
        for descriptor in descriptors:
            available = descriptor["available"]
            status = (
                "available"
                if available is True
                else "unavailable"
                if available is False
                else "not probed"
            )
            extra = descriptor["optional_extra"]
            suffix = f" (extra: {extra})" if extra else ""
            lines.append(f"  {descriptor['name']}: {status}{suffix}")
    return lines


@app.command("validate")
def validate_command(
    config_path: Annotated[
        Path,
        typer.Argument(help="Specify the experiment YAML file to validate."),
    ],
    set_values: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            metavar="KEY=VALUE",
            help=(
                "Override one configuration value with a dotted key. "
                "Repeat this option to override another value."
            ),
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write exactly one machine-readable JSON document to standard output.",
        ),
    ] = False,
) -> None:
    """Resolve the experiment configuration.

    Validate the configuration without loading model weights.
    """

    try:
        config = load_config(config_path, tuple(set_values or ()))
    except Exception as error:
        _abort(error, json_output=json_output)
    payload = {
        "schema_version": "1.0",
        "status": "valid",
        "validation_scope": "configuration_only",
        "capabilities_probed": False,
        "config_path": str(_resolved(config_path)),
        "config_hash": config.config_hash(),
        "name": config.name,
        "resolved_config": config.canonical_dict(),
    }
    _emit_success(
        payload,
        (
            f"Configuration is valid: {_resolved(config_path)}\n"
            f"Config hash: {config.config_hash()}\n"
            "Runtime, model, and backend capabilities were not probed."
        ),
        json_output=json_output,
    )


@app.command("run")
def run_command(
    config_path: Annotated[
        Path,
        typer.Argument(help="Specify the experiment YAML file to run."),
    ],
    set_values: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            metavar="KEY=VALUE",
            help=(
                "Override one configuration value with a dotted key. "
                "Repeat this option to override another value."
            ),
        ),
    ] = None,
    allow_remote_code: Annotated[
        bool,
        typer.Option(
            "--allow-remote-code",
            help=(
                "Allow remote model code for this run. "
                "Use this option only after you audit and pin the code."
            ),
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write exactly one machine-readable JSON document to standard output.",
        ),
    ] = False,
) -> None:
    """Create an immutable run. Execute all governed stages."""

    run_directory: Path | None = None
    redactor = Redactor()
    try:
        config = load_config(
            config_path,
            _execution_overrides(set_values, allow_remote_code=allow_remote_code),
        )
        _require_remote_code_cli_opt_in(config, allow_remote_code=allow_remote_code)
        redactor = _redactor_for_config(config)
        project_root = Path.cwd().resolve()
        run_id = _generate_run_id(config)
        run_directory = _planned_run_directory(config, run_id, project_root)
        completed = _run_pipeline(config, run_id=run_id, project_root=project_root).resolve()
    except Exception as error:
        _abort(
            error,
            json_output=json_output,
            run_directory=run_directory,
            redactor=redactor,
        )
    payload = {
        "schema_version": "1.0",
        "status": "success",
        "run_id": completed.name,
        "run_directory": str(completed),
        "manifest_path": str(completed / "manifest.json"),
        "status_path": str(completed / "status.json"),
    }
    _emit_success(
        payload,
        f"Run completed successfully.\nRun directory: {completed}",
        json_output=json_output,
    )


@app.command("resume")
def resume_command(
    run_directory: Annotated[
        Path,
        typer.Argument(help="Specify the existing run directory."),
    ],
    force_stage: Annotated[
        str | None,
        typer.Option(
            "--force-stage",
            metavar="STAGE",
            help="Invalidate STAGE. Rerun STAGE.",
        ),
    ] = None,
    allow_remote_code: Annotated[
        bool,
        typer.Option(
            "--allow-remote-code",
            help=(
                "Allow remote model code for this run. "
                "Use this option only after you audit and pin the code."
            ),
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write exactly one machine-readable JSON document to standard output.",
        ),
    ] = False,
) -> None:
    """Verify the stored run evidence. Continue an incomplete run. Rerun a forced stage."""

    directory = _resolved(run_directory)
    redactor = _redactor_for_run(directory)
    try:
        if force_stage is not None:
            from inkling_quant_lab.pipeline.stages import STAGE_ORDER

            if force_stage not in STAGE_ORDER:
                raise ConfigurationError(
                    f"Unknown force stage '{force_stage}'",
                    component="resume",
                    remediation="Choose one of: " + ", ".join(STAGE_ORDER),
                )
        completed = _resume_pipeline(
            directory,
            force_stage=force_stage,
            project_root=Path.cwd().resolve(),
            allow_remote_code=allow_remote_code,
        ).resolve()
    except Exception as error:
        _abort(
            error,
            json_output=json_output,
            run_directory=directory,
            redactor=redactor,
        )
    payload = {
        "schema_version": "1.0",
        "status": "success",
        "run_id": completed.name,
        "run_directory": str(completed),
        "manifest_path": str(completed / "manifest.json"),
        "status_path": str(completed / "status.json"),
        "forced_stage": force_stage,
    }
    _emit_success(
        payload,
        f"Run is complete.\nRun directory: {completed}",
        json_output=json_output,
    )


@app.command("inspect-model")
@app.command("inspect")
def inspect_model_command(
    model_config: Annotated[
        str,
        typer.Argument(
            help=(
                "Specify a model identifier, a model fragment, or a complete experiment YAML file."
            )
        ),
    ],
    set_values: Annotated[
        list[str] | None,
        typer.Option(
            "--set",
            metavar="KEY=VALUE",
            help=(
                "Override one configuration value with a dotted key. "
                "Repeat this option to override another value."
            ),
        ),
    ] = None,
    allow_remote_code: Annotated[
        bool,
        typer.Option(
            "--allow-remote-code",
            help=(
                "Allow remote model code for this inspection. "
                "Use this option only after you audit and pin the code."
            ),
        ),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write exactly one machine-readable JSON document to standard output.",
        ),
    ] = False,
) -> None:
    """Load one model safely.

    Report the model capabilities.
    Report the module inventory.
    Discover the mixture-of-experts (MoE) structure.
    """

    runtime: Any | None = None
    try:
        config = load_inspection_config(
            model_config,
            _execution_overrides(set_values, allow_remote_code=allow_remote_code),
        )
        _require_remote_code_cli_opt_in(config, allow_remote_code=allow_remote_code)
        register_builtins()
        adapter = MODEL_ADAPTERS.create(config.model.adapter)
        runtime = RUNTIMES.create(config.runtime.backend)
        runtime_capabilities = runtime.probe(config.runtime)
        if not runtime_capabilities.available or runtime_capabilities.reasons:
            from inkling_quant_lab.exceptions import CapabilityError

            raise CapabilityError(
                "Selected runtime is unavailable or incompatible: "
                + "; ".join(runtime_capabilities.reasons),
                component=config.runtime.backend,
            )
        declared_capabilities = adapter.capabilities(config)
        if config.model.dtype not in declared_capabilities.supported_dtypes:
            from inkling_quant_lab.exceptions import CapabilityError

            raise CapabilityError(
                f"Model adapter does not support dtype {config.model.dtype}",
                component=config.model.adapter,
            )
        if config.runtime.device_map not in declared_capabilities.supported_device_maps:
            from inkling_quant_lab.exceptions import CapabilityError

            raise CapabilityError(
                f"Model adapter does not support device map {config.runtime.device_map}",
                component=config.model.adapter,
            )
        loaded = adapter.load(config, runtime)
        modules = adapter.enumerate_modules(loaded)
        moe = adapter.discover_moe(loaded)
        module_rows = [asdict(module) for module in modules]
        payload = {
            "schema_version": "1.0",
            "status": "ok",
            "inspection_target": _inspection_target_label(model_config),
            "config_hash": config.config_hash(),
            "model": asdict(loaded.descriptor),
            "runtime": asdict(runtime_capabilities),
            "load_time_seconds": loaded.load_time_seconds,
            "inventory": {
                "module_count": len(modules),
                "parameter_count": sum(module.parameter_count for module in modules),
                "size_bytes": sum(module.size_bytes for module in modules),
                "router_count": sum(module.is_router for module in modules),
                "expert_module_count": sum(module.is_expert for module in modules),
                "modules": module_rows,
            },
            "moe": None if moe is None else asdict(moe),
        }
    except Exception as error:
        _abort(error, json_output=json_output)
    finally:
        if runtime is not None:
            runtime.cleanup()
    moe_summary = "not discovered" if payload["moe"] is None else "discovered"
    inventory = cast(dict[str, Any], payload["inventory"])
    model = cast(dict[str, Any], payload["model"])
    _emit_success(
        payload,
        (
            f"Model: {model['model_id']}\n"
            f"Architecture: {model['architecture']}\n"
            f"Modules: {inventory['module_count']}\n"
            f"Parameters: {inventory['parameter_count']}\n"
            f"MoE: {moe_summary}"
        ),
        json_output=json_output,
    )


@app.command("compare")
def compare_command(
    run_directories: Annotated[
        list[Path],
        typer.Argument(
            help="Specify the baseline run first. Then specify one or more candidate runs."
        ),
    ],
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="Create the comparison under this directory."),
    ] = None,
    unsafe_overrides: Annotated[
        list[str] | None,
        typer.Option(
            "--unsafe-override",
            metavar="DIMENSION",
            help=(
                "Waive one comparison compatibility dimension. "
                "Repeat this option to waive another dimension."
            ),
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write exactly one machine-readable JSON document to standard output.",
        ),
    ] = False,
) -> None:
    """Compare completed runs. Enforce the compatibility contract."""

    try:
        if len(run_directories) < 2:
            raise ConfigurationError(
                "compare requires at least two completed run directories",
                component="comparison",
                remediation="Pass the baseline run first, followed by one or more candidates.",
            )
        directories = tuple(_resolved(path) for path in run_directories)
        destination = None if output_root is None else _resolved(output_root)
        comparison = _compare_artifacts(
            directories,
            output_root=destination,
            unsafe_overrides=set(unsafe_overrides) if unsafe_overrides else None,
        ).resolve()
    except Exception as error:
        _abort(error, json_output=json_output)
    payload = {
        "schema_version": "1.0",
        "status": "success",
        "comparison_directory": str(comparison),
        "report_path": str(comparison / "report.md"),
        "source_run_directories": [str(path) for path in directories],
        "unsafe_overrides": sorted(set(unsafe_overrides or ())),
    }
    _emit_success(
        payload,
        f"Comparison completed successfully.\nComparison directory: {comparison}",
        json_output=json_output,
    )


@app.command("report")
def report_command(
    source: Annotated[
        Path,
        typer.Argument(
            help=("Specify a completed run, a comparison directory, or a report_data.json file.")
        ),
    ],
    destination: Annotated[
        Path | None,
        typer.Option(
            "--destination",
            help="Create the regenerated report at a new immutable destination.",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write exactly one machine-readable JSON document to standard output.",
        ),
    ] = False,
) -> None:
    """Locate an existing report. Regenerate a report only from normalized artifacts."""

    source_path = _resolved(source)
    destination_path = None if destination is None else _resolved(destination)
    try:
        report_path = _report_artifact(
            source_path,
            destination=destination_path,
        ).resolve()
    except Exception as error:
        _abort(error, json_output=json_output)
    payload = {
        "schema_version": "1.0",
        "status": "success",
        "source": str(source_path),
        "report_path": str(report_path),
    }
    _emit_success(
        payload,
        f"Report: {report_path}",
        json_output=json_output,
    )


@app.command("list-backends")
def list_backends_command(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write exactly one machine-readable JSON document to standard output.",
        ),
    ] = False,
) -> None:
    """List declared components. Report declared availability. Leave lazy imports unresolved."""

    try:
        backends = _backend_payload()
    except Exception as error:
        _abort(error, json_output=json_output)
    payload = {"schema_version": "1.0", "status": "ok", "backends": backends}
    _emit_success(
        payload,
        "\n".join(_backend_lines(backends)),
        json_output=json_output,
    )


@app.command("doctor")
def doctor_command(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Write exactly one machine-readable JSON document to standard output.",
        ),
    ] = False,
) -> None:
    """Report the local environment.

    Report available devices.
    List declared backends.
    Do not test backend capabilities.
    """

    try:
        environment = probe_environment(Path.cwd().resolve())
        backends = _backend_payload()
    except Exception as error:
        _abort(error, json_output=json_output)
    hardware = cast(dict[str, Any], environment["hardware"])
    declared_devices = hardware.get("available_devices")
    if isinstance(declared_devices, (list, tuple)) and declared_devices:
        available_devices = [str(device) for device in declared_devices]
    else:
        available_devices = [str(hardware.get("device", "cpu"))]
        accelerator = hardware.get("accelerator")
        if accelerator:
            available_devices.append(str(accelerator))
    payload = {
        "schema_version": "1.0",
        "status": "ok",
        "backend_scope": "declared_registry_only",
        "backend_capabilities_probed": False,
        "available_devices": available_devices,
        "environment": environment,
        "backends": backends,
    }
    python = cast(dict[str, Any], environment["python"])
    platform = cast(dict[str, Any], environment["platform"])
    human_lines = [
        (
            f"Python: {python.get('version', 'unavailable')} "
            f"({python.get('implementation', 'unknown')})"
        ),
        f"Platform: {platform.get('system', 'unknown')} {platform.get('machine', '')}".rstrip(),
        f"Available devices: {', '.join(available_devices)}",
        f"Logical CPUs: {hardware.get('logical_cpu_count', 'unavailable')}",
        "Backend capability checks: not performed (registry declarations only)",
        *_backend_lines(backends),
    ]
    _emit_success(payload, "\n".join(human_lines), json_output=json_output)


__all__ = ["app"]
