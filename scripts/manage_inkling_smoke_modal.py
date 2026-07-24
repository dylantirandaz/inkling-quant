"""Deploy, launch, and inspect the sealed Inkling Modal smoke stage.

This manager keeps the inference smoke test separate from the completed export
App.  It records a deployment version, a concrete Function identity, a launch
intent written before spawn, and the accepted Function call ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import modal  # noqa: E402

EXPECTED_MODAL_VERSION: Final = "1.5.0"
if modal.__version__ != EXPECTED_MODAL_VERSION:
    raise RuntimeError(
        f"This launcher requires Modal {EXPECTED_MODAL_VERSION}, got {modal.__version__}"
    )

from inkling_quant_lab.gguf.inkling import (  # noqa: E402
    InklingGGUFConfig,
    load_inkling_gguf_config,
    modal_stage_resources,
    require_stage_billing_window,
)
from inkling_quant_lab.gguf.inkling_smoke import (  # noqa: E402
    InklingSmokeConfig,
    InklingVerifiedExportReference,
    load_inkling_smoke_config,
    load_verified_export_reference,
)
from inkling_quant_lab.gguf.inkling_smoke_acceptance import (  # noqa: E402
    SmokePostSpawnAcceptance,
    smoke_post_spawn_acceptance_path,
    validate_smoke_post_spawn_acceptance,
)
from inkling_quant_lab.gguf.inkling_smoke_attempt import (  # noqa: E402
    SmokeAttemptRegistryClaim,
    smoke_attempt_registry_key,
    validate_smoke_attempt_registry_claim,
)
from inkling_quant_lab.gguf.inkling_smoke_authorization import (  # noqa: E402
    SMOKE_LAUNCH_INTENT_SCHEMA,
    SmokeLaunchIntent,
    smoke_launch_intent_remote_path,
    validate_smoke_launch_intent,
)
from inkling_quant_lab.gguf.inkling_smoke_execution import (  # noqa: E402
    SMOKE_ENVIRONMENT_NAME,
    SMOKE_STAGE,
    SMOKE_WORKSPACE_BUDGET_USD,
    SmokeControlPlaneProvenance,
    SmokeLaunchAcknowledgement,
    SmokeLaunchDeploymentIdentity,
    canonical_smoke_attempt_registry_created_at_utc,
    smoke_control_plane_provenance,
    smoke_deployment_tag,
    smoke_run_id,
    strict_json_object,
    validate_smoke_failure_receipt,
    validate_smoke_terminal_receipt,
)

CONFIG_PATH: Final = PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_smoke_modal.yaml"
REFERENCE_PATH: Final = PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_verified_export.json"
EXPORT_CONFIG_PATH: Final = PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_modal.yaml"
APP_PATH: Final = PROJECT_ROOT / "scripts/smoke_inkling_modal.py"
DEPLOYMENT_SCHEMA: Final = "inkling-smoke-modal-deployment-v3"
CALL_SCHEMA: Final = "inkling-smoke-modal-call-v4"
MAX_REMOTE_RECEIPT_BYTES: Final = 16 * 1024 * 1024


class _VolumeUploadStateUnknownError(RuntimeError):
    """The manager cannot prove whether an immutable Volume upload committed."""


def _load_checked_context() -> tuple[
    InklingSmokeConfig,
    InklingVerifiedExportReference,
    InklingGGUFConfig,
    SmokeControlPlaneProvenance,
    str,
]:
    config = load_inkling_smoke_config(CONFIG_PATH)
    reference = load_verified_export_reference(REFERENCE_PATH)
    if config.verified_export_reference_sha256 != reference.reference_sha256:
        raise RuntimeError("Smoke config does not bind the checked export reference")
    export_config = load_inkling_gguf_config(EXPORT_CONFIG_PATH)
    if export_config.config_hash() != InklingGGUFConfig().config_hash():
        raise RuntimeError("Checked export YAML differs from the frozen deployment schema")
    resources = modal_stage_resources(export_config, SMOKE_STAGE)
    smoke_resources = config.resources
    expected_resources = {
        "cpu_cores": resources.cpu_cores,
        "memory_gib": resources.memory_gib,
        "gpu_type": resources.gpu_type,
        "gpu_count": resources.gpu_count,
        "ephemeral_disk_mib": resources.ephemeral_disk_mib,
        "startup_timeout_seconds": resources.startup_timeout_seconds,
        "max_hours": int(resources.max_hours),
        "max_attempts": resources.max_attempts,
        "max_recovery_attempts": resources.max_recovery_attempts,
    }
    observed_resources = {name: getattr(smoke_resources, name) for name in expected_resources}
    if observed_resources != expected_resources:
        raise RuntimeError("Smoke resources differ from the reserved export configuration")
    control_plane = smoke_control_plane_provenance(PROJECT_ROOT)
    return (
        config,
        reference,
        export_config,
        control_plane,
        smoke_run_id(config, control_plane.tree_sha256),
    )


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    expected = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_text() != expected:
            raise RuntimeError(f"Refusing to replace immutable control receipt: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(16)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file() or path.read_text() != expected:
                raise RuntimeError(
                    f"Refusing to replace immutable control receipt: {path}"
                ) from None
        else:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Required control receipt is missing or unsafe: {path}")
    try:
        value = strict_json_object(path.read_bytes())
    except (OSError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid control receipt: {path}") from error
    return value


def _run_root(run_id: str) -> Path:
    return PROJECT_ROOT / "artifacts" / "inkling-smoke-modal" / run_id


def _deployment_path(run_id: str) -> Path:
    return _run_root(run_id) / "deployment.json"


@contextmanager
def _exclusive_operation(run_id: str, operation: str) -> Iterator[None]:
    lock_path = _run_root(run_id) / "control" / "manager.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = {
        "schema_version": "inkling-smoke-manager-lock-v1",
        "run_id": run_id,
        "operation": operation,
        "pid": os.getpid(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "nonce": secrets.token_hex(16),
    }
    expected = _canonical_json(lock)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"Another operation or unresolved local crash holds {lock_path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        if lock_path.is_symlink() or not lock_path.is_file() or lock_path.read_text() != expected:
            raise RuntimeError("Smoke manager lock disappeared or changed")
        lock_path.unlink()


def _require_paid_gate(
    export_config: InklingGGUFConfig,
    workspace_budget_usd: Decimal,
) -> str:
    confirmed = os.environ.get("IQL_MODAL_WORKSPACE_BUDGET_CONFIRMED")
    cycle_end = os.environ.get("IQL_MODAL_BILLING_CYCLE_END_CONFIRMED")
    if (
        workspace_budget_usd != SMOKE_WORKSPACE_BUDGET_USD
        or confirmed != "800"
        or cycle_end is None
    ):
        raise RuntimeError(
            "Confirm the active Modal workspace hard budget with "
            "IQL_MODAL_WORKSPACE_BUDGET_CONFIRMED=800, provide the exact cycle end in "
            "IQL_MODAL_BILLING_CYCLE_END_CONFIRMED, and pass --workspace-budget-usd 800"
        )
    require_stage_billing_window(
        export_config,
        cycle_end,
        SMOKE_STAGE,
        include_startup=True,
        invocations=1,
    )
    return cycle_end


def _deployment_tag(control_plane: SmokeControlPlaneProvenance) -> str:
    return smoke_deployment_tag(control_plane.tree_sha256)


def _deployment_name(control_plane: SmokeControlPlaneProvenance) -> str:
    return f"inkling-q3-smoke-{control_plane.tree_sha256[:12]}"


def _modal_history(*, app_name: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "app",
            "history",
            app_name,
            "-e",
            SMOKE_ENVIRONMENT_NAME,
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise RuntimeError("Modal App history returned an unexpected JSON shape")
    return value


def _modal_history_or_empty(*, app_name: str) -> list[dict[str, Any]]:
    try:
        return _modal_history(app_name=app_name)
    except subprocess.CalledProcessError as error:
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        if "not found" in stderr.lower() or "no app" in stderr.lower():
            return []
        raise


def _deployment_row(
    history: Sequence[Mapping[str, Any]],
    *,
    tag: str,
) -> tuple[int, Mapping[str, Any]]:
    matches = [row for row in history if row.get("tag") == tag]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one Modal deployment tagged {tag!r}")
    version_text = matches[0].get("version")
    if not isinstance(version_text, str) or re.fullmatch(r"v[1-9][0-9]*", version_text) is None:
        raise RuntimeError("Modal deployment history has an invalid version")
    return int(version_text[1:]), matches[0]


def _require_newest(history: Sequence[Mapping[str, Any]], *, version: int) -> None:
    if not history or history[0].get("version") != f"v{version}":
        raise RuntimeError("Implementation-addressed smoke App has a newer deployment")


def _function_binding(function: Any) -> dict[str, str]:
    object_id = function.object_id
    metadata = function._get_metadata()
    function_name = getattr(metadata, "function_name", None)
    if not isinstance(object_id, str) or not object_id.startswith("fu-"):
        raise RuntimeError("Modal returned an invalid smoke Function ID")
    if function_name != SMOKE_STAGE:
        raise RuntimeError("Modal returned the wrong smoke Function name")
    return {"function_id": object_id, "function_name": function_name}


def _attempt_registry(
    config: InklingSmokeConfig,
    *,
    create: bool,
) -> tuple[Any, str, str]:
    if config.storage.attempt_registry_append_only is not True:
        raise RuntimeError("Smoke attempt registry must be append-only")
    registry = modal.Dict.from_name(
        config.storage.attempt_registry,
        environment_name=SMOKE_ENVIRONMENT_NAME,
        create_if_missing=create,
    )
    registry.hydrate()
    object_id = registry.object_id
    if not isinstance(object_id, str) or re.fullmatch(r"di-[A-Za-z0-9]+", object_id) is None:
        raise RuntimeError("Modal returned an invalid smoke attempt Dict ID")
    info = registry.info()
    if info.name != config.storage.attempt_registry:
        raise RuntimeError("Modal returned the wrong smoke attempt Dict name")
    try:
        created_at_utc = canonical_smoke_attempt_registry_created_at_utc(info.created_at)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Modal returned an unsupported smoke attempt Dict") from error
    return registry, object_id, created_at_utc


def _sealed_attempt_registry(
    config: InklingSmokeConfig,
    deployment: SmokeLaunchDeploymentIdentity,
) -> Any:
    """Resolve and verify the exact Dict sealed into the deployment receipt."""

    if (
        config.storage.attempt_registry_append_only is not True
        or deployment.attempt_registry_name != config.storage.attempt_registry
    ):
        raise RuntimeError("Sealed smoke attempt Dict configuration differs")
    registry = modal.Dict.from_id(deployment.attempt_registry_id)
    registry.hydrate()
    if registry.object_id != deployment.attempt_registry_id:
        raise RuntimeError("Modal resolved the wrong sealed smoke attempt Dict ID")
    info = registry.info()
    if info.name != deployment.attempt_registry_name:
        raise RuntimeError("Modal resolved the wrong sealed smoke attempt Dict name")
    try:
        created_at_utc = canonical_smoke_attempt_registry_created_at_utc(info.created_at)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Modal returned an unsupported sealed smoke attempt Dict") from error
    if created_at_utc != deployment.attempt_registry_created_at_utc:
        raise RuntimeError("Sealed smoke attempt Dict creation time changed")
    return registry


def _read_attempt_registry_claim(
    config: InklingSmokeConfig,
    control_plane: SmokeControlPlaneProvenance,
    deployment: SmokeLaunchDeploymentIdentity,
    run_id: str,
) -> tuple[SmokeAttemptRegistryClaim, bytes, str] | None:
    """Read a run's claim from the exact Dict and validate it fail closed."""

    registry = _sealed_attempt_registry(config, deployment)
    registry_key = smoke_attempt_registry_key(run_id)
    try:
        present = registry.contains(registry_key)
    except Exception as error:
        raise RuntimeError(
            "The sealed smoke attempt Dict is unreadable; treating the attempt as consumed"
        ) from error
    if type(present) is not bool:
        raise RuntimeError(
            "The sealed smoke attempt Dict returned an invalid presence result; "
            "treating the attempt as consumed"
        )
    if present is False:
        return None
    try:
        payload = registry.get(registry_key)
    except Exception as error:
        raise RuntimeError(
            "The sealed smoke attempt claim is unreadable; treating the attempt as consumed"
        ) from error
    if not isinstance(payload, bytes):
        raise RuntimeError(
            "The sealed smoke attempt claim is malformed; treating the attempt as consumed"
        )
    claim_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        raw = strict_json_object(payload)
        candidate = SmokeAttemptRegistryClaim.model_validate(raw)
        claim = validate_smoke_attempt_registry_claim(
            payload,
            claim_sha256=claim_sha256,
            registry_name=deployment.attempt_registry_name,
            registry_id=deployment.attempt_registry_id,
            registry_created_at_utc=deployment.attempt_registry_created_at_utc,
            registry_key=registry_key,
            run_id=run_id,
            stage=SMOKE_STAGE,
            call_id=candidate.call_id,
            input_id=candidate.input_id,
            task_id=candidate.task_id,
            launch_intent_sha256=candidate.launch_intent_sha256,
            post_spawn_acceptance_path=candidate.post_spawn_acceptance_path,
            post_spawn_acceptance_sha256=(candidate.post_spawn_acceptance_sha256),
            smoke_config_hash=config.config_hash(),
            control_plane_sha256=control_plane.tree_sha256,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "The sealed smoke attempt claim is invalid; treating the attempt as consumed"
        ) from error
    return claim, payload, claim_sha256


def _validate_attempt_registry_claim_for_invocation(
    payload: bytes,
    claim_sha256: str,
    *,
    config: InklingSmokeConfig,
    control_plane: SmokeControlPlaneProvenance,
    deployment: SmokeLaunchDeploymentIdentity,
    run_id: str,
    invocation: Any,
) -> SmokeAttemptRegistryClaim:
    """Validate one persisted claim against a terminal invocation."""

    try:
        claim = validate_smoke_attempt_registry_claim(
            payload,
            claim_sha256=claim_sha256,
            registry_name=deployment.attempt_registry_name,
            registry_id=deployment.attempt_registry_id,
            registry_created_at_utc=deployment.attempt_registry_created_at_utc,
            registry_key=smoke_attempt_registry_key(run_id),
            run_id=run_id,
            stage=SMOKE_STAGE,
            call_id=invocation.call_id,
            input_id=invocation.input_id,
            task_id=invocation.task_id,
            launch_intent_sha256=invocation.launch_intent_sha256,
            post_spawn_acceptance_path=(f"runs/{run_id}/{invocation.post_spawn_acceptance_path}"),
            post_spawn_acceptance_sha256=(invocation.post_spawn_acceptance_sha256),
            smoke_config_hash=config.config_hash(),
            control_plane_sha256=control_plane.tree_sha256,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "The atomic Dict attempt claim differs from the terminal invocation"
        ) from error
    if (
        invocation.attempt_registry_name != deployment.attempt_registry_name
        or invocation.attempt_registry_id != deployment.attempt_registry_id
        or invocation.attempt_registry_created_at_utc != deployment.attempt_registry_created_at_utc
        or invocation.attempt_registry_key != claim.registry_key
        or invocation.attempt_registry_claim_sha256 != claim_sha256
        or invocation.attempt_claim_sha256 != claim_sha256
    ):
        raise RuntimeError("Terminal invocation has the wrong atomic Dict claim binding")
    return claim


def _deployment_identity(deployment: Mapping[str, Any]) -> SmokeLaunchDeploymentIdentity:
    binding = deployment.get("function_binding")
    if not isinstance(binding, Mapping):
        raise RuntimeError("Smoke deployment has no Function binding")
    try:
        return SmokeLaunchDeploymentIdentity(
            app_name=deployment.get("app_name"),
            environment_name=deployment.get("environment_name"),
            deployment_version=deployment.get("version"),
            deployment_tag=deployment.get("tag"),
            function_id=binding.get("function_id"),
            function_name=binding.get("function_name"),
            attempt_registry_name=deployment.get("attempt_registry_name"),
            attempt_registry_id=deployment.get("attempt_registry_id"),
            attempt_registry_created_at_utc=deployment.get("attempt_registry_created_at_utc"),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("Smoke deployment identity is invalid") from error


def _call_deployment_identity(receipt: Mapping[str, Any]) -> SmokeLaunchDeploymentIdentity:
    try:
        return SmokeLaunchDeploymentIdentity(
            app_name=receipt.get("app_name"),
            environment_name=receipt.get("environment_name"),
            deployment_version=receipt.get("deployment_version"),
            deployment_tag=receipt.get("deployment_tag"),
            function_id=receipt.get("function_id"),
            function_name=receipt.get("function_name"),
            attempt_registry_name=receipt.get("attempt_registry_name"),
            attempt_registry_id=receipt.get("attempt_registry_id"),
            attempt_registry_created_at_utc=receipt.get("attempt_registry_created_at_utc"),
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("Smoke call deployment identity is invalid") from error


def _hydrate_binding(*, app_name: str) -> tuple[Any, dict[str, str]]:
    function = modal.Function.from_name(
        app_name,
        SMOKE_STAGE,
        environment_name=SMOKE_ENVIRONMENT_NAME,
    )
    function.hydrate()
    return function, _function_binding(function)


def _deploy_locked(
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    export_config: InklingGGUFConfig,
    control_plane: SmokeControlPlaneProvenance,
    run_id: str,
    workspace_budget_usd: Decimal,
) -> None:
    cycle_end = _require_paid_gate(export_config, workspace_budget_usd)
    receipt_path = _deployment_path(run_id)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise RuntimeError("The smoke deployment is already sealed; do not redeploy it")
    app_name = _deployment_name(control_plane)
    tag = _deployment_tag(control_plane)
    (
        _registry,
        attempt_registry_id,
        attempt_registry_created_at_utc,
    ) = _attempt_registry(config, create=True)
    history = _modal_history_or_empty(app_name=app_name)
    existing = [row for row in history if row.get("tag") == tag]
    if len(existing) > 1:
        raise RuntimeError("More than one deployment has the smoke implementation tag")
    if not existing:
        deployment_environment = os.environ.copy()
        deployment_environment["IQL_MODAL_SMOKE_CONTROL_PLANE_SHA256"] = control_plane.tree_sha256
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "deploy",
                "-e",
                SMOKE_ENVIRONMENT_NAME,
                "--name",
                app_name,
                "--tag",
                tag,
                str(APP_PATH),
            ],
            check=True,
            env=deployment_environment,
            shell=False,
        )
        history = _modal_history(app_name=app_name)
    version, row = _deployment_row(history, tag=tag)
    _require_newest(history, version=version)
    _function, binding = _hydrate_binding(app_name=app_name)
    history_after = _modal_history(app_name=app_name)
    observed_version, observed_row = _deployment_row(history_after, tag=tag)
    _require_newest(history_after, version=observed_version)
    if observed_version != version or observed_row != row:
        raise RuntimeError("Smoke deployment changed while its Function was being sealed")
    (
        _registry_after,
        observed_attempt_registry_id,
        observed_attempt_registry_created_at_utc,
    ) = _attempt_registry(
        config,
        create=False,
    )
    if (
        observed_attempt_registry_id,
        observed_attempt_registry_created_at_utc,
    ) != (
        attempt_registry_id,
        attempt_registry_created_at_utc,
    ):
        raise RuntimeError("Smoke attempt Dict changed while the deployment was sealed")
    receipt = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "app_name": app_name,
        "environment_name": SMOKE_ENVIRONMENT_NAME,
        "version": version,
        "tag": tag,
        "time_deployed": row.get("time_deployed"),
        "modal_client": row.get("client"),
        "function_binding": binding,
        "attempt_registry_name": config.storage.attempt_registry,
        "attempt_registry_id": attempt_registry_id,
        "attempt_registry_created_at_utc": attempt_registry_created_at_utc,
        "run_id": run_id,
        "smoke_config_hash": config.config_hash(),
        "verified_export_reference_sha256": reference.reference_sha256,
        "control_plane_sha256": control_plane.tree_sha256,
        "billing_cycle_end_utc": cycle_end,
        "workspace_budget_usd": str(SMOKE_WORKSPACE_BUDGET_USD),
    }
    _write_immutable_json(receipt_path, receipt)
    print(json.dumps({"status": "deployed_and_sealed", **receipt}, indent=2, sort_keys=True))


def deploy(workspace_budget_usd: Decimal) -> None:
    context = _load_checked_context()
    config, reference, export_config, control_plane, run_id = context
    with _exclusive_operation(run_id, "deploy"):
        _deploy_locked(
            config,
            reference,
            export_config,
            control_plane,
            run_id,
            workspace_budget_usd,
        )


def _validated_deployment(
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    run_id: str,
) -> dict[str, Any]:
    receipt = _read_object(_deployment_path(run_id))
    expected = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "app_name": _deployment_name(control_plane),
        "environment_name": SMOKE_ENVIRONMENT_NAME,
        "tag": _deployment_tag(control_plane),
        "run_id": run_id,
        "smoke_config_hash": config.config_hash(),
        "verified_export_reference_sha256": reference.reference_sha256,
        "control_plane_sha256": control_plane.tree_sha256,
        "workspace_budget_usd": str(SMOKE_WORKSPACE_BUDGET_USD),
        "attempt_registry_name": config.storage.attempt_registry,
    }
    mismatches = [name for name, value in expected.items() if receipt.get(name) != value]
    version = receipt.get("version")
    binding = receipt.get("function_binding")
    attempt_registry_id = receipt.get("attempt_registry_id")
    attempt_registry_created_at_utc = receipt.get("attempt_registry_created_at_utc")
    if type(version) is not int or version < 1:
        mismatches.append("version")
    if (
        not isinstance(binding, dict)
        or set(binding) != {"function_id", "function_name"}
        or not isinstance(binding.get("function_id"), str)
        or not str(binding["function_id"]).startswith("fu-")
        or binding.get("function_name") != SMOKE_STAGE
    ):
        mismatches.append("function_binding")
    if (
        not isinstance(attempt_registry_id, str)
        or re.fullmatch(r"di-[A-Za-z0-9]+", attempt_registry_id) is None
    ):
        mismatches.append("attempt_registry_id")
    try:
        SmokeLaunchDeploymentIdentity(
            app_name=receipt.get("app_name"),
            environment_name=receipt.get("environment_name"),
            deployment_version=receipt.get("version"),
            deployment_tag=receipt.get("tag"),
            function_id=binding.get("function_id") if isinstance(binding, dict) else None,
            function_name=binding.get("function_name") if isinstance(binding, dict) else None,
            attempt_registry_name=receipt.get("attempt_registry_name"),
            attempt_registry_id=attempt_registry_id,
            attempt_registry_created_at_utc=attempt_registry_created_at_utc,
        )
    except (TypeError, ValueError):
        mismatches.append("attempt_registry_created_at_utc")
    if mismatches:
        raise RuntimeError("Smoke deployment receipt drifted: " + ", ".join(sorted(mismatches)))
    (
        _registry,
        current_attempt_registry_id,
        current_attempt_registry_created_at_utc,
    ) = _attempt_registry(config, create=False)
    if (
        current_attempt_registry_id,
        current_attempt_registry_created_at_utc,
    ) != (
        attempt_registry_id,
        attempt_registry_created_at_utc,
    ):
        raise RuntimeError("Sealed smoke attempt Dict identity changed")
    history = _modal_history(app_name=str(receipt["app_name"]))
    history_version, _ = _deployment_row(history, tag=str(receipt["tag"]))
    if history_version != version:
        raise RuntimeError("Sealed smoke deployment version no longer matches its tag")
    _require_newest(history, version=history_version)
    return receipt


def _revalidate_history(deployment: Mapping[str, Any]) -> None:
    history = _modal_history(app_name=str(deployment["app_name"]))
    version, _ = _deployment_row(history, tag=str(deployment["tag"]))
    if version != deployment["version"]:
        raise RuntimeError("Smoke deployment tag moved after the local seal")
    _require_newest(history, version=version)


def _evidence_volume(config: InklingSmokeConfig, *, create: bool) -> Any:
    volume = modal.Volume.from_name(
        config.storage.evidence_volume,
        environment_name=SMOKE_ENVIRONMENT_NAME,
        create_if_missing=create,
        version=1,
    )
    volume.hydrate()
    return volume


def _read_volume_bytes(volume: Any, remote_path: str) -> bytes | None:
    payload = bytearray()
    try:
        for chunk in volume.read_file(remote_path):
            if not isinstance(chunk, bytes):
                raise RuntimeError("Modal returned non-byte smoke evidence")
            payload.extend(chunk)
            if len(payload) > MAX_REMOTE_RECEIPT_BYTES:
                raise RuntimeError("Remote smoke receipt exceeds its size limit")
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
    return bytes(payload)


def _read_volume_json(volume: Any, remote_path: str) -> dict[str, Any] | None:
    payload = _read_volume_bytes(volume, remote_path)
    if payload is None:
        return None
    try:
        value = strict_json_object(payload)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"Remote smoke evidence is invalid JSON: {remote_path}") from error
    return value


def _list_volume(volume: Any, remote_path: str) -> list[Any]:
    try:
        return list(volume.listdir(remote_path, recursive=False))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return []


def _reconcile_uploaded_volume_bytes(
    *,
    config: InklingSmokeConfig,
    remote_path: str,
    expected_payload: bytes,
    label: str,
) -> bytes:
    """Prove an upload through an independently hydrated Volume handle."""

    try:
        fresh_volume = _evidence_volume(config, create=False)
        reconciled = _read_volume_bytes(fresh_volume, remote_path)
    except Exception as error:
        raise _VolumeUploadStateUnknownError(
            f"Could not reconcile the remote {label} after its upload"
        ) from error
    if reconciled != expected_payload:
        raise RuntimeError(f"Remote {label} differs after upload reconciliation")
    return reconciled


def _validate_remote_post_spawn_acceptance(
    *,
    volume: Any,
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    deployment: SmokeLaunchDeploymentIdentity,
    run_id: str,
    invocation: Any,
) -> None:
    remote_path = smoke_post_spawn_acceptance_path(
        run_id,
        invocation.launch_intent_sha256,
    )
    run_prefix = f"runs/{run_id}/"
    relative_path = remote_path.removeprefix(run_prefix)
    if (
        not remote_path.startswith(run_prefix)
        or invocation.post_spawn_acceptance_path != relative_path
    ):
        raise RuntimeError("Remote smoke invocation has the wrong acceptance path")
    payload = _read_volume_bytes(volume, remote_path)
    if payload is None:
        raise RuntimeError("Remote smoke invocation has no post-spawn acceptance")
    acceptance_sha256 = hashlib.sha256(payload).hexdigest()
    if acceptance_sha256 != invocation.post_spawn_acceptance_sha256:
        raise RuntimeError("Remote smoke acceptance hash differs from the receipt")
    try:
        raw = strict_json_object(payload)
        accepted_at = raw.get("accepted_at")
        if not isinstance(accepted_at, str):
            raise ValueError("post-spawn acceptance has no canonical time")
        validate_smoke_post_spawn_acceptance(
            payload,
            evidence_path=remote_path,
            acceptance_sha256=acceptance_sha256,
            accepted_at=accepted_at,
            run_id=run_id,
            launch_intent_sha256=invocation.launch_intent_sha256,
            call_id=invocation.call_id,
            app_name=deployment.app_name,
            environment_name=deployment.environment_name,
            deployment_version=deployment.deployment_version,
            deployment_tag=deployment.deployment_tag,
            function_id=deployment.function_id,
            function_name=deployment.function_name,
            attempt_registry_name=deployment.attempt_registry_name,
            attempt_registry_id=deployment.attempt_registry_id,
            attempt_registry_created_at_utc=(deployment.attempt_registry_created_at_utc),
            smoke_config_hash=config.config_hash(),
            verified_export_reference_sha256=reference.reference_sha256,
            control_plane_sha256=control_plane.tree_sha256,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("Remote smoke post-spawn acceptance is invalid") from error


def _validate_persisted_invocation_records(
    *,
    volume: Any,
    config: InklingSmokeConfig,
    control_plane: SmokeControlPlaneProvenance,
    deployment: SmokeLaunchDeploymentIdentity,
    run_id: str,
    invocation: Any,
) -> None:
    if invocation.run_id != run_id:
        raise RuntimeError("Remote smoke invocation differs from its run")
    run_root = f"runs/{run_id}"
    claim_path = f"{run_root}/{invocation.attempt_claim_path}"
    persisted_claim = _read_volume_bytes(volume, claim_path)
    if persisted_claim is None:
        raise RuntimeError("Remote smoke attempt claim is missing")
    persisted_claim_sha256 = hashlib.sha256(persisted_claim).hexdigest()
    _validate_attempt_registry_claim_for_invocation(
        persisted_claim,
        persisted_claim_sha256,
        config=config,
        control_plane=control_plane,
        deployment=deployment,
        run_id=run_id,
        invocation=invocation,
    )
    if (
        persisted_claim_sha256 != invocation.attempt_claim_sha256
        or persisted_claim_sha256 != invocation.attempt_registry_claim_sha256
    ):
        raise RuntimeError("Remote smoke attempt claim differs from the receipt")
    registry_claim = _read_attempt_registry_claim(
        config,
        control_plane,
        deployment,
        run_id,
    )
    if registry_claim is not None and (
        registry_claim[1] != persisted_claim or registry_claim[2] != persisted_claim_sha256
    ):
        raise RuntimeError("Remote smoke attempt claim differs from its atomic Dict claim")

    persisted_event = invocation.model_dump(
        mode="json",
        exclude={
            "attempt_claim_path",
            "attempt_claim_sha256",
            "invocation_history_path",
            "invocation_history_sha256",
        },
    )
    expected_history_payload = _canonical_json(persisted_event).encode("utf-8")
    history_path = f"{run_root}/{invocation.invocation_history_path}"
    history_payload = _read_volume_bytes(volume, history_path)
    if history_payload is None:
        raise RuntimeError("Remote smoke invocation history is missing")
    if (
        history_payload != expected_history_payload
        or hashlib.sha256(history_payload).hexdigest() != invocation.invocation_history_sha256
    ):
        raise RuntimeError("Remote smoke invocation history differs from the receipt")

    history_root = f"{run_root}/control/history"
    observed_histories = sorted(str(item.path) for item in _list_volume(volume, history_root))
    if observed_histories != [history_path]:
        raise RuntimeError("Remote smoke invocation history is not unique")

    expected_ledger = {
        "schema_version": "inkling-smoke-attempt-ledger-v1",
        "stage": SMOKE_STAGE,
        "attempts": 1,
        "limit": config.resources.max_attempts,
        "last_history_path": invocation.invocation_history_path,
        "last_history_sha256": invocation.invocation_history_sha256,
        "last_call_id": invocation.call_id,
        "last_input_id": invocation.input_id,
        "last_task_id": invocation.task_id,
        "launch_intent_sha256": invocation.launch_intent_sha256,
        "smoke_config_hash": invocation.smoke_config_hash,
        "control_plane_sha256": invocation.control_plane_sha256,
    }
    ledger_path = f"{run_root}/control/smoke_test.attempts.json"
    ledger_payload = _read_volume_bytes(volume, ledger_path)
    if ledger_payload is None:
        raise RuntimeError("Remote smoke attempt ledger is missing")
    expected_ledger_payload = _canonical_json(expected_ledger).encode("utf-8")
    if ledger_payload != expected_ledger_payload:
        raise RuntimeError("Remote smoke attempt ledger differs from the invocation")


def _validate_terminal_receipt(
    value: Mapping[str, Any],
    *,
    volume: Any,
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    deployment: SmokeLaunchDeploymentIdentity,
    run_id: str,
) -> None:
    try:
        receipt = validate_smoke_terminal_receipt(
            value,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("Remote smoke receipt is invalid") from error

    authorization_path = smoke_launch_intent_remote_path(
        run_id,
        receipt.launch_intent_sha256,
    )
    authorization = _read_volume_bytes(volume, authorization_path)
    if authorization is None:
        raise RuntimeError("Remote smoke success has no launch authorization")
    try:
        validate_smoke_launch_intent(
            authorization,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=receipt.launch_intent_sha256,
            deployment=deployment,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("Remote smoke success launch authorization is invalid") from error

    _validate_remote_post_spawn_acceptance(
        volume=volume,
        config=config,
        reference=reference,
        control_plane=control_plane,
        deployment=deployment,
        run_id=run_id,
        invocation=receipt.invocation,
    )
    _validate_persisted_invocation_records(
        volume=volume,
        config=config,
        control_plane=control_plane,
        deployment=deployment,
        run_id=run_id,
        invocation=receipt.invocation,
    )


def _require_fresh_attempt(
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    deployment: SmokeLaunchDeploymentIdentity,
    run_id: str,
) -> None:
    registry_claim = _read_attempt_registry_claim(
        config,
        control_plane,
        deployment,
        run_id,
    )
    if registry_claim is not None:
        raise RuntimeError(
            "The atomic smoke attempt claim already exists; the one attempt is consumed"
        )
    run_root = f"runs/{run_id}"
    volume = _evidence_volume(config, create=True)
    success = _read_volume_json(volume, f"{run_root}/smoke_test.success.json")
    outcomes = _list_volume(volume, f"{run_root}/control/outcomes")
    failures = _validated_remote_failure_receipts(
        volume,
        outcomes,
        config=config,
        reference=reference,
        control_plane=control_plane,
        deployment=deployment,
        run_id=run_id,
    )
    if success is not None and failures:
        raise RuntimeError("Smoke evidence contains conflicting success and failure outcomes")
    if success is not None:
        _validate_terminal_receipt(
            success,
            volume=volume,
            config=config,
            reference=reference,
            control_plane=control_plane,
            deployment=deployment,
            run_id=run_id,
        )
        raise RuntimeError("The smoke stage is already successful; refusing another launch")
    claim = _read_volume_json(
        volume,
        f"{run_root}/control/smoke_test.attempt.claim.json",
    )
    ledger = _read_volume_json(volume, f"{run_root}/control/smoke_test.attempts.json")
    history = _list_volume(volume, f"{run_root}/control/history")
    acceptances = _list_volume(
        volume,
        f"{run_root}/control/post-spawn-acceptances",
    )
    if claim is not None or ledger is not None or history or acceptances or failures:
        raise RuntimeError("The one smoke attempt has already been consumed")


def _safe_subprocess_failure_record(receipt: Any) -> dict[str, Any] | None:
    """Return the allowlisted subprocess diagnostic without raw process data."""

    evidence = receipt.safe_subprocess_failure
    if evidence is None:
        return None
    return {
        "schema_version": evidence.schema_version,
        "command_id": evidence.command_id,
        "return_code": evidence.return_code,
        "stdout_size_bytes": evidence.stdout_size_bytes,
        "stdout_sha256": evidence.stdout_sha256,
        "stderr_size_bytes": evidence.stderr_size_bytes,
        "stderr_sha256": evidence.stderr_sha256,
        "stdout_recorded": evidence.stdout_recorded,
        "stderr_recorded": evidence.stderr_recorded,
    }


def _safe_server_log_failure_record(receipt: Any) -> dict[str, Any]:
    """Project only the validated structural facts from a version 6 server log."""

    evidence = receipt.server_log_evidence
    signals = evidence.safe_failure_signals
    backend = evidence.backend_diagnostic
    return {
        "schema_version": evidence.schema_version,
        "present": evidence.present,
        "size_bytes": evidence.size_bytes,
        "sha256": evidence.sha256,
        "raw_log_recorded": evidence.raw_log_recorded,
        "scan_integrity": evidence.scan_integrity,
        "safe_failure_signals": {
            "out_of_memory_observed": signals.out_of_memory_observed,
            "no_usable_gpu_observed": signals.no_usable_gpu_observed,
            "model_load_failure_observed": signals.model_load_failure_observed,
            "projector_load_failure_observed": signals.projector_load_failure_observed,
            "unsupported_architecture_observed": signals.unsupported_architecture_observed,
        },
        "backend_diagnostic": {
            "schema_version": backend.schema_version,
            "cpu_model_graph_fallback_observed": backend.cpu_model_graph_fallback_observed,
            "graph_marker_count": backend.graph_marker_count,
            "affected_graph_marker_count": backend.affected_graph_marker_count,
            "cpu_node_marker_count": backend.cpu_node_marker_count,
            "affected_graphs": [
                {
                    "graph_uid": graph.graph_uid,
                    "phase": graph.phase,
                    "scope": graph.scope,
                    "compute": graph.compute,
                    "gpu": graph.gpu,
                    "cpu": graph.cpu,
                    "accel": graph.accel,
                    "other": graph.other,
                    "unassigned": graph.unassigned,
                }
                for graph in backend.affected_graphs
            ],
            "cpu_node_samples": [
                {
                    "graph_uid": sample.graph_uid,
                    "ordinal": sample.ordinal,
                    "op": sample.op,
                    "node_name_size_bytes": sample.node_name_size_bytes,
                    "node_name_sha256": sample.node_name_sha256,
                    "node_name_recorded": sample.node_name_recorded,
                }
                for sample in backend.cpu_node_samples
            ],
            "records_truncated": backend.records_truncated,
            "raw_marker_lines_recorded": backend.raw_marker_lines_recorded,
            "raw_node_names_recorded": backend.raw_node_names_recorded,
        },
    }


def _validated_remote_failure_receipts(
    volume: Any,
    outcomes: Sequence[Any],
    *,
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    deployment: SmokeLaunchDeploymentIdentity,
    run_id: str,
) -> list[dict[str, Any]]:
    run_root = f"runs/{run_id}"
    expected_prefix = f"{run_root}/control/outcomes/"
    records: list[dict[str, Any]] = []
    for item in sorted(outcomes, key=lambda value: str(value.path)):
        outcome_path = item.path
        if (
            not isinstance(outcome_path, str)
            or not outcome_path.startswith(expected_prefix)
            or re.fullmatch(
                rf"{re.escape(expected_prefix)}smoke_test\.failed\.[0-9a-f]{{64}}\.json",
                outcome_path,
            )
            is None
        ):
            raise RuntimeError("Remote smoke outcome path is not a canonical failure receipt")
        payload = _read_volume_bytes(volume, outcome_path)
        if payload is None:
            raise RuntimeError("Remote smoke failure receipt disappeared during validation")
        try:
            raw = strict_json_object(payload)
        except (TypeError, ValueError) as error:
            raise RuntimeError("Remote smoke failure receipt is invalid JSON") from error
        launch_intent_sha256 = raw.get("launch_intent_sha256")
        if (
            not isinstance(launch_intent_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", launch_intent_sha256) is None
        ):
            raise RuntimeError("Remote smoke failure receipt has no valid launch binding")
        authorization_path = smoke_launch_intent_remote_path(
            run_id,
            launch_intent_sha256,
        )
        authorization = _read_volume_bytes(volume, authorization_path)
        if authorization is None:
            raise RuntimeError("Remote smoke failure has no launch authorization")
        try:
            validate_smoke_launch_intent(
                authorization,
                config=config,
                reference=reference,
                control_plane=control_plane,
                run_id=run_id,
                launch_intent_sha256=launch_intent_sha256,
                deployment=deployment,
            )
            receipt = validate_smoke_failure_receipt(
                payload,
                config=config,
                reference=reference,
                control_plane=control_plane,
                run_id=run_id,
                launch_intent_sha256=launch_intent_sha256,
                outcome_path=outcome_path,
            )
            if receipt.schema_version in {
                "inkling-smoke-terminal-v3",
                "inkling-smoke-terminal-v4",
                "inkling-smoke-terminal-v5",
                "inkling-smoke-terminal-v6",
            }:
                invocation = getattr(receipt, "invocation", None)
                if invocation is None:
                    raise RuntimeError("Current smoke failure receipt lacks invocation evidence")
                _validate_remote_post_spawn_acceptance(
                    volume=volume,
                    config=config,
                    reference=reference,
                    control_plane=control_plane,
                    deployment=deployment,
                    run_id=run_id,
                    invocation=invocation,
                )
                _validate_persisted_invocation_records(
                    volume=volume,
                    config=config,
                    control_plane=control_plane,
                    deployment=deployment,
                    run_id=run_id,
                    invocation=invocation,
                )
        except (TypeError, ValueError) as error:
            raise RuntimeError("Remote smoke failure receipt is invalid") from error
        record: dict[str, Any] = {
            "path": outcome_path,
            "receipt_sha256": receipt.receipt_sha256,
            "launch_intent_sha256": receipt.launch_intent_sha256,
            "authorization_path": authorization_path,
        }
        if receipt.schema_version in {
            "inkling-smoke-terminal-v4",
            "inkling-smoke-terminal-v5",
            "inkling-smoke-terminal-v6",
        }:
            safe_subprocess_failure = _safe_subprocess_failure_record(receipt)
            if safe_subprocess_failure is not None:
                record["safe_subprocess_failure"] = safe_subprocess_failure
        if receipt.schema_version == "inkling-smoke-terminal-v6":
            record["server_log_evidence"] = _safe_server_log_failure_record(receipt)
        records.append(record)
    return records


def _validated_call_status(receipt: Mapping[str, Any], path: Path) -> str:
    status = receipt.get("status")
    if not isinstance(status, str) or status not in {
        "accepted",
        "acceptance_state_unknown",
        "cancellation_requested",
        "cancellation_failed",
    }:
        raise RuntimeError(f"Smoke call receipt has an invalid status: {path}")

    has_post_spawn_error = "post_spawn_error" in receipt
    has_cancel_error = "cancel_error" in receipt
    post_spawn_error = receipt.get("post_spawn_error")
    cancel_error = receipt.get("cancel_error")
    if status == "accepted":
        if has_post_spawn_error or has_cancel_error:
            raise RuntimeError(
                f"Accepted smoke call receipt contains cancellation evidence: {path}"
            )
    elif not has_post_spawn_error or not isinstance(post_spawn_error, str) or not post_spawn_error:
        raise RuntimeError(
            f"Cancelled smoke call receipt lacks post-spawn failure evidence: {path}"
        )
    elif status == "acceptance_state_unknown" and (
        not has_cancel_error or cancel_error is not None
    ):
        raise RuntimeError(f"Ambiguous smoke acceptance receipt claims a cancellation: {path}")
    elif status == "cancellation_requested" and (not has_cancel_error or cancel_error is not None):
        raise RuntimeError(
            f"Smoke cancellation receipt has unexpected cancellation failure: {path}"
        )
    elif status == "cancellation_failed" and (
        not has_cancel_error or not isinstance(cancel_error, str) or not cancel_error
    ):
        raise RuntimeError(f"Failed smoke cancellation receipt lacks its error: {path}")
    return status


def _validate_local_post_spawn_acceptance(
    receipt: Mapping[str, Any],
    *,
    call_receipt_path: Path,
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    deployment: SmokeLaunchDeploymentIdentity,
    run_id: str,
) -> None:
    launch_intent_sha256 = receipt.get("launch_intent_sha256")
    call_id = receipt.get("call_id")
    accepted_at = receipt.get("accepted_at_utc")
    acceptance_sha256 = receipt.get("post_spawn_acceptance_sha256")
    local_relative = receipt.get("post_spawn_acceptance")
    remote_path = receipt.get("remote_post_spawn_acceptance")
    if not all(
        isinstance(value, str)
        for value in (
            launch_intent_sha256,
            call_id,
            accepted_at,
            acceptance_sha256,
            local_relative,
            remote_path,
        )
    ):
        raise RuntimeError(
            f"Accepted smoke call has incomplete post-spawn evidence: {call_receipt_path}"
        )
    assert isinstance(launch_intent_sha256, str)
    assert isinstance(call_id, str)
    assert isinstance(accepted_at, str)
    assert isinstance(acceptance_sha256, str)
    assert isinstance(local_relative, str)
    assert isinstance(remote_path, str)
    expected_remote = smoke_post_spawn_acceptance_path(run_id, launch_intent_sha256)
    run_prefix = f"runs/{run_id}/"
    expected_local = expected_remote.removeprefix(run_prefix)
    if not expected_remote.startswith(run_prefix) or (
        remote_path,
        local_relative,
    ) != (
        expected_remote,
        expected_local,
    ):
        raise RuntimeError(
            f"Accepted smoke call has the wrong acceptance path: {call_receipt_path}"
        )
    local_path = _run_root(run_id) / expected_local
    if (
        Path(expected_local).is_absolute()
        or ".." in Path(expected_local).parts
        or not local_path.resolve().is_relative_to(_run_root(run_id).resolve())
        or local_path.is_symlink()
        or not local_path.is_file()
    ):
        raise RuntimeError(
            f"Accepted smoke call has unsafe acceptance evidence: {call_receipt_path}"
        )
    payload = local_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != acceptance_sha256:
        raise RuntimeError(f"Accepted smoke call acceptance hash differs: {call_receipt_path}")
    try:
        validate_smoke_post_spawn_acceptance(
            payload,
            evidence_path=expected_remote,
            acceptance_sha256=acceptance_sha256,
            accepted_at=accepted_at,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
            call_id=call_id,
            app_name=deployment.app_name,
            environment_name=deployment.environment_name,
            deployment_version=deployment.deployment_version,
            deployment_tag=deployment.deployment_tag,
            function_id=deployment.function_id,
            function_name=deployment.function_name,
            attempt_registry_name=deployment.attempt_registry_name,
            attempt_registry_id=deployment.attempt_registry_id,
            attempt_registry_created_at_utc=(deployment.attempt_registry_created_at_utc),
            smoke_config_hash=config.config_hash(),
            verified_export_reference_sha256=reference.reference_sha256,
            control_plane_sha256=control_plane.tree_sha256,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Accepted smoke call acceptance is invalid: {call_receipt_path}"
        ) from error


def _validated_local_calls(
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    run_id: str,
) -> list[dict[str, Any]]:
    run_root = _run_root(run_id)
    calls_root = run_root / "calls"
    if not calls_root.exists():
        return []
    if calls_root.is_symlink() or not calls_root.is_dir():
        raise RuntimeError("Smoke call receipt directory is unsafe")
    receipts: list[dict[str, Any]] = []
    for path in sorted(calls_root.glob("*.json")):
        receipt = _read_object(path)
        relative_intent = receipt.get("launch_intent")
        if not isinstance(relative_intent, str):
            raise RuntimeError("Smoke call receipt has no launch-intent path")
        intent_path = run_root / relative_intent
        if (
            Path(relative_intent).is_absolute()
            or ".." in Path(relative_intent).parts
            or not intent_path.resolve().is_relative_to(run_root.resolve())
        ):
            raise RuntimeError("Smoke call receipt has an unsafe launch-intent path")
        intent_sha256 = receipt.get("launch_intent_sha256")
        remote_intent = receipt.get("remote_launch_intent")
        expected = {
            "schema_version": CALL_SCHEMA,
            "run_id": run_id,
            "stage": SMOKE_STAGE,
            "app_name": _deployment_name(control_plane),
            "environment_name": SMOKE_ENVIRONMENT_NAME,
            "deployment_tag": _deployment_tag(control_plane),
            "attempt_registry_name": config.storage.attempt_registry,
            "smoke_config_hash": config.config_hash(),
            "verified_export_reference_sha256": reference.reference_sha256,
            "control_plane_sha256": control_plane.tree_sha256,
        }
        if any(receipt.get(name) != value for name, value in expected.items()):
            raise RuntimeError(f"Smoke call receipt drifted: {path}")
        status = _validated_call_status(receipt, path)
        deployment = _call_deployment_identity(receipt)
        if not isinstance(intent_sha256, str) or _sha256(intent_path) != intent_sha256:
            raise RuntimeError(f"Smoke launch intent drifted: {intent_path}")
        expected_remote_intent = smoke_launch_intent_remote_path(run_id, intent_sha256)
        if remote_intent != expected_remote_intent:
            raise RuntimeError("Smoke call receipt has the wrong remote launch-intent path")
        try:
            validate_smoke_launch_intent(
                intent_path.read_bytes(),
                config=config,
                reference=reference,
                control_plane=control_plane,
                run_id=run_id,
                launch_intent_sha256=intent_sha256,
                deployment=deployment,
            )
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError(f"Smoke launch intent is invalid: {intent_path}") from error
        acceptance_fields = {
            "post_spawn_acceptance",
            "remote_post_spawn_acceptance",
            "post_spawn_acceptance_sha256",
        }
        if status == "accepted":
            _validate_local_post_spawn_acceptance(
                receipt,
                call_receipt_path=path,
                config=config,
                reference=reference,
                control_plane=control_plane,
                deployment=deployment,
                run_id=run_id,
            )
        elif acceptance_fields & receipt.keys():
            raise RuntimeError(f"Cancelled smoke call claims completed acceptance: {path}")
        receipts.append(receipt)
    return receipts


def _require_no_unresolved_intent(
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    deployment: SmokeLaunchDeploymentIdentity,
    run_id: str,
) -> None:
    run_root = _run_root(run_id)
    resolved = {
        str(receipt["launch_intent_sha256"])
        for receipt in _validated_local_calls(
            config,
            reference,
            control_plane,
            run_id,
        )
    }
    intents_root = run_root / "launch-intents"
    if not intents_root.exists():
        return
    if intents_root.is_symlink() or not intents_root.is_dir():
        raise RuntimeError("Smoke launch-intent directory is unsafe")
    for path in sorted(intents_root.glob("*.json")):
        launch_intent_sha256 = _sha256(path)
        try:
            validate_smoke_launch_intent(
                path.read_bytes(),
                config=config,
                reference=reference,
                control_plane=control_plane,
                run_id=run_id,
                launch_intent_sha256=launch_intent_sha256,
                deployment=deployment,
            )
        except (OSError, TypeError, ValueError) as error:
            raise RuntimeError(f"Smoke launch intent drifted: {path}") from error
        if launch_intent_sha256 not in resolved:
            raise RuntimeError(
                f"Unresolved pre-spawn intent {path}; inspect Modal before any retry"
            )


def _create_launch_intent(
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    run_id: str,
    deployment: Mapping[str, Any],
    deployment_identity: SmokeLaunchDeploymentIdentity,
) -> tuple[str, str]:
    created_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    filename = (
        f"{created_at.replace(':', '').replace('+', '')}-smoke_test-{secrets.token_hex(16)}.json"
    )
    relative = (Path("launch-intents") / filename).as_posix()
    binding = deployment["function_binding"]
    intent = SmokeLaunchIntent(
        schema_version=SMOKE_LAUNCH_INTENT_SCHEMA,
        authorization_nonce=secrets.token_hex(32),
        created_at_utc=created_at,
        run_id=run_id,
        app_name=deployment["app_name"],
        deployment_version=deployment["version"],
        deployment_tag=deployment["tag"],
        function_id=binding["function_id"],
        function_name=binding["function_name"],
        attempt_registry_name=deployment["attempt_registry_name"],
        attempt_registry_id=deployment["attempt_registry_id"],
        attempt_registry_created_at_utc=deployment["attempt_registry_created_at_utc"],
        smoke_config_hash=config.config_hash(),
        verified_export_reference_sha256=reference.reference_sha256,
        control_plane_sha256=control_plane.tree_sha256,
        workspace_budget_usd=str(SMOKE_WORKSPACE_BUDGET_USD),
        billing_cycle_end_utc=deployment["billing_cycle_end_utc"],
    )
    path = _run_root(run_id) / relative
    _write_immutable_json(path, intent.model_dump(mode="json"))
    launch_intent_sha256 = _sha256(path)
    validate_smoke_launch_intent(
        path.read_bytes(),
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
        launch_intent_sha256=launch_intent_sha256,
        deployment=deployment_identity,
    )
    return relative, launch_intent_sha256


def _publish_remote_launch_intent(
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    run_id: str,
    relative_intent: str,
    launch_intent_sha256: str,
    deployment: SmokeLaunchDeploymentIdentity,
) -> str:
    local_path = _run_root(run_id) / relative_intent
    remote_path = smoke_launch_intent_remote_path(run_id, launch_intent_sha256)
    payload = local_path.read_bytes()
    validate_smoke_launch_intent(
        payload,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
        launch_intent_sha256=launch_intent_sha256,
        deployment=deployment,
    )
    volume = _evidence_volume(config, create=True)
    if _read_volume_bytes(volume, remote_path) is not None:
        raise RuntimeError("Remote smoke launch authorization already exists")
    with volume.batch_upload(force=False) as batch:
        batch.put_file(local_path, remote_path)
    persisted = _read_volume_bytes(volume, remote_path)
    if persisted != payload:
        raise RuntimeError("Remote smoke launch authorization differs after upload")
    validate_smoke_launch_intent(
        persisted,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
        launch_intent_sha256=launch_intent_sha256,
        deployment=deployment,
    )
    return remote_path


def _publish_post_spawn_acceptance(
    *,
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    deployment: SmokeLaunchDeploymentIdentity,
    run_id: str,
    launch_intent_sha256: str,
    call_id: str,
    accepted_at: str,
) -> tuple[str, str, str]:
    """Publish the exact accepted Modal call with overwrite disabled."""

    acceptance = SmokePostSpawnAcceptance(
        accepted_at=accepted_at,
        run_id=run_id,
        launch_intent_sha256=launch_intent_sha256,
        call_id=call_id,
        app_name=deployment.app_name,
        environment_name=deployment.environment_name,
        deployment_version=deployment.deployment_version,
        deployment_tag=deployment.deployment_tag,
        function_id=deployment.function_id,
        function_name=deployment.function_name,
        attempt_registry_name=deployment.attempt_registry_name,
        attempt_registry_id=deployment.attempt_registry_id,
        attempt_registry_created_at_utc=deployment.attempt_registry_created_at_utc,
        smoke_config_hash=config.config_hash(),
        verified_export_reference_sha256=reference.reference_sha256,
        control_plane_sha256=control_plane.tree_sha256,
    )
    remote_path = smoke_post_spawn_acceptance_path(run_id, launch_intent_sha256)
    run_prefix = f"runs/{run_id}/"
    if not remote_path.startswith(run_prefix):
        raise RuntimeError("Post-spawn acceptance path differs from its run")
    local_relative = remote_path.removeprefix(run_prefix)
    local_path = _run_root(run_id) / local_relative
    _write_immutable_json(local_path, acceptance.model_dump(mode="json"))
    payload = local_path.read_bytes()
    acceptance_sha256 = acceptance.acceptance_sha256()
    if payload != acceptance.canonical_bytes() or _sha256(local_path) != acceptance_sha256:
        raise RuntimeError("Local post-spawn acceptance differs after publication")
    try:
        validate_smoke_post_spawn_acceptance(
            payload,
            evidence_path=remote_path,
            acceptance_sha256=acceptance_sha256,
            accepted_at=accepted_at,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
            call_id=call_id,
            app_name=deployment.app_name,
            environment_name=deployment.environment_name,
            deployment_version=deployment.deployment_version,
            deployment_tag=deployment.deployment_tag,
            function_id=deployment.function_id,
            function_name=deployment.function_name,
            attempt_registry_name=deployment.attempt_registry_name,
            attempt_registry_id=deployment.attempt_registry_id,
            attempt_registry_created_at_utc=(deployment.attempt_registry_created_at_utc),
            smoke_config_hash=config.config_hash(),
            verified_export_reference_sha256=reference.reference_sha256,
            control_plane_sha256=control_plane.tree_sha256,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("Local post-spawn acceptance is invalid") from error

    volume = _evidence_volume(config, create=True)
    if _read_volume_bytes(volume, remote_path) is not None:
        raise RuntimeError("Remote post-spawn acceptance already exists")
    upload_error: Exception | None = None
    try:
        with volume.batch_upload(force=False) as batch:
            batch.put_file(local_path, remote_path)
    except Exception as error:
        upload_error = error
    persisted = _reconcile_uploaded_volume_bytes(
        config=config,
        remote_path=remote_path,
        expected_payload=payload,
        label="post-spawn acceptance",
    )
    if upload_error is not None:
        print(
            json.dumps(
                {
                    "event": "post_spawn_acceptance_upload_response_reconciled",
                    "run_id": run_id,
                    "stage": SMOKE_STAGE,
                    "call_id": call_id,
                    "error_type": type(upload_error).__name__,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
    try:
        validate_smoke_post_spawn_acceptance(
            persisted,
            evidence_path=remote_path,
            acceptance_sha256=acceptance_sha256,
            accepted_at=accepted_at,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
            call_id=call_id,
            app_name=deployment.app_name,
            environment_name=deployment.environment_name,
            deployment_version=deployment.deployment_version,
            deployment_tag=deployment.deployment_tag,
            function_id=deployment.function_id,
            function_name=deployment.function_name,
            attempt_registry_name=deployment.attempt_registry_name,
            attempt_registry_id=deployment.attempt_registry_id,
            attempt_registry_created_at_utc=(deployment.attempt_registry_created_at_utc),
            smoke_config_hash=config.config_hash(),
            verified_export_reference_sha256=reference.reference_sha256,
            control_plane_sha256=control_plane.tree_sha256,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError("Remote post-spawn acceptance is invalid") from error
    return local_relative, remote_path, acceptance_sha256


def _launch_locked(
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    export_config: InklingGGUFConfig,
    control_plane: SmokeControlPlaneProvenance,
    run_id: str,
    workspace_budget_usd: Decimal,
) -> None:
    cycle_end = _require_paid_gate(export_config, workspace_budget_usd)
    deployment = _validated_deployment(config, reference, control_plane, run_id)
    deployment_identity = _deployment_identity(deployment)
    _require_no_unresolved_intent(
        config,
        reference,
        control_plane,
        deployment_identity,
        run_id,
    )
    if _validated_local_calls(config, reference, control_plane, run_id):
        raise RuntimeError("A smoke call is already recorded; refusing a duplicate")
    _require_fresh_attempt(
        config,
        reference,
        control_plane,
        deployment_identity,
        run_id,
    )
    if deployment.get("billing_cycle_end_utc") != cycle_end:
        raise RuntimeError("The sealed smoke deployment belongs to another billing cycle")
    function, binding = _hydrate_binding(app_name=str(deployment["app_name"]))
    if binding != deployment["function_binding"]:
        raise RuntimeError("Smoke Function binding changed after it was sealed")
    _revalidate_history(deployment)
    _require_fresh_attempt(
        config,
        reference,
        control_plane,
        deployment_identity,
        run_id,
    )
    relative_intent, launch_intent_sha256 = _create_launch_intent(
        config,
        reference,
        control_plane,
        run_id,
        deployment,
        deployment_identity,
    )
    remote_launch_intent = _publish_remote_launch_intent(
        config,
        reference,
        control_plane,
        run_id,
        relative_intent,
        launch_intent_sha256,
        deployment_identity,
    )
    _revalidate_history(deployment)
    if _function_binding(function) != deployment["function_binding"]:
        raise RuntimeError("Smoke Function binding changed after launch authorization")
    _require_fresh_attempt(
        config,
        reference,
        control_plane,
        deployment_identity,
        run_id,
    )
    acknowledgement = SmokeLaunchAcknowledgement(
        smoke_config_hash=config.config_hash(),
        verified_export_reference_sha256=reference.reference_sha256,
        control_plane_sha256=control_plane.tree_sha256,
        launch_intent_sha256=launch_intent_sha256,
        run_id=run_id,
        deployment=deployment_identity,
        workspace_budget_usd=SMOKE_WORKSPACE_BUDGET_USD,
        billing_cycle_end_utc=cycle_end,
    )
    call = function.spawn(
        config.canonical_json(),
        reference.canonical_json(),
        run_id,
        launch_intent_sha256,
        acknowledgement.canonical_json(),
        control_plane.canonical_json(),
    )
    accepted_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    call_id = call.object_id
    if not isinstance(call_id, str) or re.fullmatch(r"fc-[A-Za-z0-9]+", call_id) is None:
        raise RuntimeError("Modal returned an invalid smoke Function call ID")
    post_acceptance_error: Exception | None = None
    cancel_error: Exception | None = None
    acceptance_state_unknown = False
    post_spawn_acceptance: tuple[str, str, str] | None = None
    try:
        _revalidate_history(deployment)
        _fresh_function, fresh_binding = _hydrate_binding(app_name=str(deployment["app_name"]))
        if fresh_binding != deployment["function_binding"]:
            raise RuntimeError("Smoke Function binding changed after spawn")
        post_spawn_acceptance = _publish_post_spawn_acceptance(
            config=config,
            reference=reference,
            control_plane=control_plane,
            deployment=deployment_identity,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
            call_id=call_id,
            accepted_at=accepted_at,
        )
    except _VolumeUploadStateUnknownError as error:
        post_acceptance_error = error
        acceptance_state_unknown = True
    except Exception as error:
        post_acceptance_error = error
        try:
            call.cancel(terminate_containers=True)
        except Exception as error_during_cancel:
            cancel_error = error_during_cancel
    status = "accepted" if post_acceptance_error is None else "cancellation_requested"
    if acceptance_state_unknown:
        status = "acceptance_state_unknown"
    if post_acceptance_error is not None and cancel_error is not None:
        status = "cancellation_failed"
    call_receipt: dict[str, Any] = {
        "schema_version": CALL_SCHEMA,
        "status": status,
        "accepted_at_utc": accepted_at,
        "run_id": run_id,
        "stage": SMOKE_STAGE,
        "call_id": call_id,
        "app_name": deployment["app_name"],
        "environment_name": SMOKE_ENVIRONMENT_NAME,
        "deployment_version": deployment["version"],
        "deployment_tag": deployment["tag"],
        "attempt_registry_name": deployment["attempt_registry_name"],
        "attempt_registry_id": deployment["attempt_registry_id"],
        "attempt_registry_created_at_utc": deployment["attempt_registry_created_at_utc"],
        "launch_intent": relative_intent,
        "remote_launch_intent": remote_launch_intent,
        "launch_intent_sha256": launch_intent_sha256,
        **binding,
        "smoke_config_hash": config.config_hash(),
        "verified_export_reference_sha256": reference.reference_sha256,
        "control_plane_sha256": control_plane.tree_sha256,
        "billing_cycle_end_utc": cycle_end,
        "workspace_budget_usd": str(SMOKE_WORKSPACE_BUDGET_USD),
    }
    if post_spawn_acceptance is not None:
        (
            call_receipt["post_spawn_acceptance"],
            call_receipt["remote_post_spawn_acceptance"],
            call_receipt["post_spawn_acceptance_sha256"],
        ) = post_spawn_acceptance
    if post_acceptance_error is not None:
        call_receipt["post_spawn_error"] = str(post_acceptance_error)
        call_receipt["cancel_error"] = None if cancel_error is None else str(cancel_error)
    call_path = _run_root(run_id) / "calls" / (f"{accepted_at.replace(':', '')}-smoke_test.json")
    _write_immutable_json(call_path, call_receipt)
    if post_acceptance_error is not None:
        raise RuntimeError(
            f"Post-spawn acceptance failed; retained call {call_id} with status {status}"
        ) from post_acceptance_error
    print(json.dumps(call_receipt, indent=2, sort_keys=True))


def launch(workspace_budget_usd: Decimal) -> None:
    context = _load_checked_context()
    config, reference, export_config, control_plane, run_id = context
    with _exclusive_operation(run_id, "launch:smoke_test"):
        _launch_locked(
            config,
            reference,
            export_config,
            control_plane,
            run_id,
            workspace_budget_usd,
        )


def _remote_status(
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    deployment: SmokeLaunchDeploymentIdentity,
    run_id: str,
) -> dict[str, Any]:
    registry_key = smoke_attempt_registry_key(run_id)
    registry_claim = _read_attempt_registry_claim(
        config,
        control_plane,
        deployment,
        run_id,
    )
    registry_status: dict[str, Any] = {
        "attempt_registry_key": registry_key,
        "attempt_registry_key_present": registry_claim is not None,
        "attempt_registry_live_claim_sha256": (
            None if registry_claim is None else registry_claim[2]
        ),
    }
    volume = _evidence_volume(config, create=False)
    run_root = f"runs/{run_id}"
    success = _read_volume_json(volume, f"{run_root}/smoke_test.success.json")
    outcomes = _list_volume(volume, f"{run_root}/control/outcomes")
    failures = _validated_remote_failure_receipts(
        volume,
        outcomes,
        config=config,
        reference=reference,
        control_plane=control_plane,
        deployment=deployment,
        run_id=run_id,
    )
    if success is not None and failures:
        raise RuntimeError("Smoke evidence contains conflicting success and failure outcomes")
    if success is not None:
        _validate_terminal_receipt(
            success,
            volume=volume,
            config=config,
            reference=reference,
            control_plane=control_plane,
            deployment=deployment,
            run_id=run_id,
        )
        return {
            "evidence_status": "passed",
            "attempt_consumed": True,
            "receipt_sha256": success["receipt_sha256"],
            "receipt_path": f"{run_root}/smoke_test.success.json",
            **registry_status,
        }
    histories = _list_volume(volume, f"{run_root}/control/history")
    attempt_paths = sorted(str(item.path) for item in histories)
    history_pattern = re.compile(
        rf"^{re.escape(run_root)}/control/history/"
        r"smoke_test\.attempt\.1\.[0-9a-f]{64}\.json$"
    )
    if len(attempt_paths) > 1 or any(
        history_pattern.fullmatch(path) is None for path in attempt_paths
    ):
        raise RuntimeError("Remote smoke invocation history contains an unknown entry")
    volume_claim_path = f"{run_root}/control/smoke_test.attempt.claim.json"
    volume_claim = _read_volume_bytes(volume, volume_claim_path)
    if (
        registry_claim is not None
        and volume_claim is not None
        and (
            volume_claim != registry_claim[1]
            or hashlib.sha256(volume_claim).hexdigest() != registry_claim[2]
        )
    ):
        raise RuntimeError("Remote smoke Volume claim differs from its atomic Dict claim")
    ledger = _read_volume_bytes(
        volume,
        f"{run_root}/control/smoke_test.attempts.json",
    )
    acceptance_paths = sorted(
        str(item.path)
        for item in _list_volume(
            volume,
            f"{run_root}/control/post-spawn-acceptances",
        )
    )
    volume_bookkeeping_present = bool(
        volume_claim is not None or ledger is not None or attempt_paths or acceptance_paths
    )
    if failures:
        evidence_status = "failed"
    elif registry_claim is not None and not volume_bookkeeping_present:
        evidence_status = "attempt_consumed_before_volume_bookkeeping"
    elif registry_claim is not None or volume_bookkeeping_present:
        evidence_status = "attempt_consumed_without_terminal_receipt"
    else:
        evidence_status = "not_terminal"
    attempt_consumed = bool(failures or registry_claim is not None or volume_bookkeeping_present)
    return {
        "evidence_status": evidence_status,
        "attempt_consumed": attempt_consumed,
        "failure_receipts": [failure["path"] for failure in failures],
        "failure_receipt_sha256": [failure["receipt_sha256"] for failure in failures],
        "failure_records": failures,
        "attempt_receipts": attempt_paths,
        "volume_attempt_claim_path": (volume_claim_path if volume_claim is not None else None),
        "post_spawn_acceptances": acceptance_paths,
        **registry_status,
    }


def inspect() -> None:
    config, reference, _export_config, control_plane, run_id = _load_checked_context()
    value: dict[str, Any] = {
        "run_id": run_id,
        "stage": SMOKE_STAGE,
        "smoke_config_hash": config.config_hash(),
        "verified_export_reference_sha256": reference.reference_sha256,
        "control_plane_sha256": control_plane.tree_sha256,
        "control_plane_file_count": control_plane.file_count,
        "app_name": _deployment_name(control_plane),
        "deployment_tag": _deployment_tag(control_plane),
        "subject_run_id": reference.subject_run_id,
        "subject_q3_shards": reference.q3_shard_count,
        "subject_q3_bytes": reference.q3_total_bytes,
        "subject_mtp": reference.mtp,
        "claims": config.claims.model_dump(mode="json"),
    }
    if _deployment_path(run_id).exists():
        value["local_deployment"] = _read_object(_deployment_path(run_id))
    value["local_calls"] = _validated_local_calls(
        config,
        reference,
        control_plane,
        run_id,
    )
    print(json.dumps(value, indent=2, sort_keys=True))


def _call_graph_records(nodes: Sequence[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for node in nodes:
        status = getattr(node, "status", None)
        status_name = getattr(status, "name", None)
        children = getattr(node, "children", None)
        record = {
            "input_id": getattr(node, "input_id", None),
            "function_call_id": getattr(node, "function_call_id", None),
            "task_id": getattr(node, "task_id", None),
            "status": status_name.casefold() if isinstance(status_name, str) else None,
            "function_name": getattr(node, "function_name", None),
            "module_name": getattr(node, "module_name", None),
        }
        stable_identity = (
            record["input_id"],
            record["function_call_id"],
            record["status"],
            record["function_name"],
            record["module_name"],
        )
        if not isinstance(record["task_id"], str) or any(
            not isinstance(value, str) or not value for value in stable_identity
        ):
            raise RuntimeError("Modal call graph contains an incomplete input identity")
        records.append(record)
        if not isinstance(children, list):
            raise RuntimeError("Modal call graph contains an invalid child collection")
        records.extend(_call_graph_records(children))
    return records


def status(call_id: str | None) -> None:
    config, reference, _export_config, control_plane, run_id = _load_checked_context()
    deployment = _deployment_identity(
        _validated_deployment(config, reference, control_plane, run_id)
    )
    local_calls = _validated_local_calls(config, reference, control_plane, run_id)
    if call_id is None and len(local_calls) == 1:
        local_call_id = local_calls[0].get("call_id")
        call_id = local_call_id if isinstance(local_call_id, str) else None
    remote = _remote_status(config, reference, control_plane, deployment, run_id)
    if remote["evidence_status"] in {"passed", "failed"}:
        print(
            json.dumps(
                {"run_id": run_id, "call_id": call_id, **remote},
                indent=2,
                sort_keys=True,
            )
        )
        return
    function_state = "no_call_id"
    function_result: object | None = None
    call_graph: list[dict[str, Any]] = []
    if call_id is not None:
        call = modal.FunctionCall.from_id(call_id)
        call_graph = _call_graph_records(call.get_call_graph())
        statuses = {record["status"] for record in call_graph}
        failed_statuses = {
            "failure",
            "init_failure",
            "terminated",
            "timeout",
        }
        if statuses & failed_statuses:
            function_state = "failed_before_terminal_evidence"
        elif call_graph and statuses == {"success"}:
            try:
                function_result = call.get(timeout=0)
                function_state = "returned_without_terminal_evidence"
            except (TimeoutError, modal.exception.TimeoutError) as error:
                if type(error) not in {TimeoutError, modal.exception.TimeoutError}:
                    raise
                function_state = "success_result_pending_without_terminal_evidence"
        else:
            function_state = "running_or_queued"
    print(
        json.dumps(
            {
                "run_id": run_id,
                "call_id": call_id,
                "function_state": function_state,
                "function_result": function_result,
                "call_graph": call_graph,
                **remote,
                "warning": "Function return is not evidence; only a validated terminal receipt is.",
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect")
    for command in ("deploy", "launch"):
        child = subparsers.add_parser(command)
        child.add_argument("--workspace-budget-usd", type=Decimal, required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--call-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "inspect":
        inspect()
    elif args.command == "deploy":
        deploy(args.workspace_budget_usd)
    elif args.command == "launch":
        launch(args.workspace_budget_usd)
    elif args.command == "status":
        status(args.call_id)
    else:  # pragma: no cover - argparse rejects this branch.
        raise RuntimeError(f"Unsupported command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
