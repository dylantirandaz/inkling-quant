"""Manage the isolated BF16 prompt-interface diagnostic on Modal.

Preparation is host-side only. Deployment and the single eight-B300 launch use
separate short-lived, content-addressed confirmations. A provider return is not
result evidence; only a validated immutable terminal receipt is.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import io
import json
import os
import re
import secrets
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, Final, Literal, cast

if __name__ == "__main__":
    sys.dont_write_bytecode = True

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
SRC_ROOT: Final = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inkling_quant_lab.gguf.inkling_bf16_interface_diagnostic import (  # noqa: E402
    DIAGNOSTIC_ATTEMPT_REGISTRY_NAME,
    DIAGNOSTIC_CONFIG_RELATIVE_PATH,
    DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
    DIAGNOSTIC_DEPLOY_CHALLENGE_MAX_AGE_SECONDS,
    DIAGNOSTIC_ENVIRONMENT_NAME,
    DIAGNOSTIC_EVIDENCE_VOLUME_NAME,
    DIAGNOSTIC_FUNCTION_NAME,
    DIAGNOSTIC_LAUNCH_CHALLENGE_MAX_AGE_SECONDS,
    DIAGNOSTIC_RAW_RECORD_MAX_BYTES,
    DIAGNOSTIC_STAGE,
    DiagnosticAttemptClaim,
    DiagnosticControlPlaneFile,
    DiagnosticControlPlaneProvenance,
    DiagnosticDeployConfirmationChallenge,
    DiagnosticDeploymentIdentity,
    DiagnosticLaunchConfirmationChallenge,
    DiagnosticLaunchIntent,
    DiagnosticPostSpawnAcceptance,
    DiagnosticPrivateRawEvidence,
    DiagnosticPrivateRawReference,
    DiagnosticReviewedInputs,
    DiagnosticSuccessTerminalReceipt,
    DiagnosticTerminalReceipt,
    DiagnosticTerminalReceiptReference,
    InklingBF16InterfaceDiagnosticBundle,
    build_diagnostic_control_plane_provenance,
    build_diagnostic_launch_intent,
    build_diagnostic_post_spawn_acceptance,
    build_diagnostic_rollup,
    build_diagnostic_server_command,
    build_diagnostic_terminal_receipt_reference,
    canonical_diagnostic_json_bytes,
    diagnostic_app_name,
    diagnostic_attempt_claim_path,
    diagnostic_attempt_registry_key,
    diagnostic_deployment_tag,
    diagnostic_launch_intent_path,
    diagnostic_post_spawn_acceptance_path,
    diagnostic_protocol_sha256,
    diagnostic_runtime_identity_sha256,
    diagnostic_workload_sha256,
    load_bf16_interface_diagnostic_bundle,
    parse_diagnostic_private_raw_evidence,
    parse_diagnostic_terminal_receipt,
    strict_diagnostic_json_object,
    validate_diagnostic_attempt_claim,
    validate_diagnostic_control_plane_provenance,
    validate_diagnostic_deploy_challenge_not_expired,
    validate_diagnostic_launch_intent,
    validate_diagnostic_post_spawn_acceptance,
    validate_diagnostic_private_raw_reference,
    validate_diagnostic_private_trials,
    validate_diagnostic_terminal_receipt_reference,
    validate_repository_relative_path,
)

EXPECTED_MODAL_VERSION: Final = "1.5.0"
RUNNER_RELATIVE_PATH: Final = "scripts/run_inkling_measurement_modal.py"
MANAGER_RELATIVE_PATH: Final = "scripts/manage_inkling_bf16_interface_diagnostic_modal.py"
ARTIFACT_ROOT: Final = PROJECT_ROOT / "artifacts" / "inkling-bf16-interface-diagnostic"
EVIDENCE_MOUNT_ROOT: Final = "/evidence"
CONTROL_SHA_ENV: Final = "IQL_BF16_DIAGNOSTIC_CONTROL_PLANE_SHA256"
PROVENANCE_PATH_ENV: Final = "IQL_BF16_DIAGNOSTIC_CONTROL_PLANE_PROVENANCE_PATH"
CALL_ID_PATTERN: Final = re.compile(r"^fc-[A-Za-z0-9]+$")
RUN_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
OBJECT_ID_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "function": re.compile(r"^fu-[A-Za-z0-9]+$"),
    "attempt registry": re.compile(r"^di-[A-Za-z0-9]+$"),
    "evidence volume": re.compile(r"^vo-[A-Za-z0-9]+$"),
}
MAX_CALL_GRAPH_NODES: Final = 256


@dataclass(frozen=True)
class _ReviewedContext:
    bundle: InklingBF16InterfaceDiagnosticBundle
    provenance: DiagnosticControlPlaneProvenance
    reviewed_inputs: DiagnosticReviewedInputs
    run_id: str
    provenance_path: Path


@dataclass(frozen=True)
class _LaunchBinding:
    deployment: DiagnosticDeploymentIdentity
    intent: DiagnosticLaunchIntent
    acceptance: DiagnosticPostSpawnAcceptance


@dataclass(frozen=True)
class _AttemptInspection:
    claim: DiagnosticAttemptClaim | None
    durable: bool


def _utc_microseconds(value: datetime | None = None) -> str:
    instant = datetime.now(UTC) if value is None else value.astimezone(UTC)
    return instant.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_utc_microseconds(value: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise RuntimeError("diagnostic evidence contains an invalid UTC time") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:
        raise RuntimeError("diagnostic evidence UTC time is not canonical")
    return parsed


def _timestamp_microseconds(timestamp: float) -> str:
    if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or timestamp <= 0:
        raise RuntimeError("Modal resource creation time is invalid")
    return _utc_microseconds(datetime.fromtimestamp(float(timestamp), UTC))


def _run_root(run_id: str) -> Path:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("diagnostic run ID is invalid")
    return ARTIFACT_ROOT / run_id


def _control_path(run_id: str, *parts: str) -> Path:
    return _run_root(run_id) / "control" / Path(*parts)


def _provenance_path(run_id: str, digest: str) -> Path:
    return _control_path(run_id, "provenance", f"{digest}.json")


def _deploy_challenge_path(run_id: str, digest: str) -> Path:
    return _control_path(run_id, "deploy-challenges", f"{digest}.json")


def _deployment_path(run_id: str) -> Path:
    return _control_path(run_id, "deployment.json")


def _deploy_consumption_path(run_id: str, digest: str) -> Path:
    return _control_path(run_id, "deploy-consumptions", f"{digest}.json")


def _launch_challenge_path(run_id: str, digest: str) -> Path:
    return _control_path(run_id, "launch-challenges", f"{digest}.json")


def _launch_intent_local_path(run_id: str, digest: str) -> Path:
    return _control_path(run_id, "launch-intents", f"{digest}.json")


def _launch_consumption_path(run_id: str) -> Path:
    return _control_path(run_id, "launch-consumption.json")


def _acceptance_local_path(run_id: str, digest: str) -> Path:
    return _control_path(run_id, "post-spawn-acceptances", f"{digest}.json")


def _call_receipt_path(run_id: str, digest: str) -> Path:
    return _run_root(run_id) / "calls" / f"{digest}.json"


def _assert_local_artifact_path(path: Path) -> Path:
    root = ARTIFACT_ROOT.resolve()
    candidate = Path(os.path.abspath(path))
    if candidate == root or root not in candidate.parents:
        raise RuntimeError("local record is outside the diagnostic artifact root")
    current = root
    for part in candidate.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("local diagnostic artifact path contains a symbolic link")
    if candidate.resolve(strict=False) != candidate:
        raise RuntimeError("local diagnostic artifact path is not canonical")
    return candidate


def _write_immutable(path: Path, payload: bytes) -> None:
    target = _assert_local_artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    _assert_local_artifact_path(target)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError:
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError(
                f"immutable record exists with different bytes: {target}"
            ) from None
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_immutable(path, canonical_diagnostic_json_bytes(value))


def _read_control_model(path: Path, model: type[Any]) -> Any:
    resolved = _assert_local_artifact_path(path)
    if resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError("local diagnostic control record must be one regular file")
    payload = resolved.read_bytes()
    strict_diagnostic_json_object(
        payload,
        maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
    )
    value = model.model_validate_json(payload, strict=True)
    if payload != value.canonical_bytes():
        raise RuntimeError("local diagnostic control record is not canonical")
    return value


@contextmanager
def _operation_lock(run_id: str, operation: Literal["deploy", "launch"]) -> Iterator[None]:
    path = _control_path(run_id, f".{operation}.lock")
    target = _assert_local_artifact_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _git_text(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    return result.stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        shell=False,
    )
    return result.stdout


def _require_reviewed_main() -> tuple[str, str]:
    if _git_text("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("diagnostic deployment requires a clean reviewed worktree")
    branch = _git_text("symbolic-ref", "--quiet", "HEAD")
    if branch != "refs/heads/main":
        raise RuntimeError("diagnostic deployment requires the checked-out main branch")
    _git_text("fetch", "--quiet", "--no-tags", "origin", "main")
    head = _git_text("rev-parse", "HEAD")
    local_main = _git_text("rev-parse", "refs/heads/main")
    origin_main = _git_text("rev-parse", "refs/remotes/origin/main")
    if head != local_main or head != origin_main:
        raise RuntimeError(
            "diagnostic deployment requires HEAD, local main, and fetched origin/main to match"
        )
    tree = _git_text("rev-parse", "HEAD^{tree}")
    if GIT_OBJECT_PATTERN.fullmatch(head) is None or GIT_OBJECT_PATTERN.fullmatch(tree) is None:
        raise RuntimeError("reviewed Git identity is invalid")
    return head, tree


def _closed_control_paths(
    bundle: InklingBF16InterfaceDiagnosticBundle,
) -> tuple[str, ...]:
    config = bundle.config
    fixed = {
        MANAGER_RELATIVE_PATH,
        RUNNER_RELATIVE_PATH,
        DIAGNOSTIC_CONFIG_RELATIVE_PATH,
        config.bf16_subject_reference.path,
        config.source_adoption_reference.path,
        config.diagnostic_dataset.path,
        config.runtime.instrumentation_patch_path,
        config.runtime_measurement_patch.path,
    }
    tracked_sources = {
        path
        for path in _git_text("ls-files", "src/inkling_quant_lab").splitlines()
        if path.endswith(".py")
    }
    if not tracked_sources:
        raise RuntimeError("diagnostic source closure is empty")
    paths = tuple(sorted(fixed | tracked_sources))
    for path in paths:
        validate_repository_relative_path(path)
    return paths


def _read_project_files(paths: Sequence[str]) -> dict[str, bytes]:
    root = PROJECT_ROOT.resolve()
    result: dict[str, bytes] = {}
    for relative in paths:
        path = PROJECT_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"diagnostic control file is not regular: {relative}")
        resolved = path.resolve()
        if resolved == root or root not in resolved.parents:
            raise RuntimeError(f"diagnostic control file escapes the project: {relative}")
        result[relative] = path.read_bytes()
    return result


def _manifest_file(
    provenance: DiagnosticControlPlaneProvenance,
    path: str,
) -> DiagnosticControlPlaneFile:
    matches = tuple(item for item in provenance.files if item.path == path)
    if len(matches) != 1:
        raise RuntimeError(f"reviewed file is absent from diagnostic closure: {path}")
    return matches[0]


def _diagnostic_run_id(
    bundle: InklingBF16InterfaceDiagnosticBundle,
    control_sha256: str,
) -> str:
    return f"inkling-bf16-diag-86b4d430-{bundle.config.config_hash()[:12]}-{control_sha256[:12]}"


def _build_reviewed_context() -> _ReviewedContext:
    commit, tree = _require_reviewed_main()
    bundle = load_bf16_interface_diagnostic_bundle(PROJECT_ROOT)
    paths = _closed_control_paths(bundle)
    provenance = build_diagnostic_control_plane_provenance(
        reviewed_commit_sha=commit,
        reviewed_tree_sha=tree,
        files=_read_project_files(paths),
        required_paths=paths,
    )
    config = bundle.config
    reviewed = DiagnosticReviewedInputs(
        control_plane=provenance,
        diagnostic_config=_manifest_file(provenance, DIAGNOSTIC_CONFIG_RELATIVE_PATH),
        resolved_config_sha256=config.config_hash(),
        protocol_sha256=diagnostic_protocol_sha256(config),
        workload_sha256=diagnostic_workload_sha256(config),
        bf16_subject_reference=_manifest_file(
            provenance,
            config.bf16_subject_reference.path,
        ),
        source_adoption_reference=_manifest_file(
            provenance,
            config.source_adoption_reference.path,
        ),
        diagnostic_dataset=_manifest_file(provenance, config.diagnostic_dataset.path),
        runtime_measurement_patch=_manifest_file(
            provenance,
            config.runtime_measurement_patch.path,
        ),
        resources=config.resources,
    )
    if (
        config.storage.evidence_volume != DIAGNOSTIC_EVIDENCE_VOLUME_NAME
        or config.storage.attempt_registry != DIAGNOSTIC_ATTEMPT_REGISTRY_NAME
    ):
        raise RuntimeError("diagnostic storage differs from its control constants")
    run_id = _diagnostic_run_id(bundle, provenance.control_plane_sha256)
    return _ReviewedContext(
        bundle=bundle,
        provenance=provenance,
        reviewed_inputs=reviewed,
        run_id=run_id,
        provenance_path=_provenance_path(run_id, provenance.control_plane_sha256),
    )


def _require_current_review(
    context: _ReviewedContext,
    reviewed: DiagnosticReviewedInputs,
) -> None:
    if reviewed != context.reviewed_inputs:
        raise RuntimeError("diagnostic control record differs from current reviewed main")


def inspect(*, as_json: bool) -> None:
    bundle = load_bf16_interface_diagnostic_bundle(PROJECT_ROOT)
    config = bundle.config
    config_bytes = (PROJECT_ROOT / DIAGNOSTIC_CONFIG_RELATIVE_PATH).read_bytes()
    payload = {
        "status": "planned_not_executed",
        "diagnostic_config_content_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "diagnostic_config_semantic_sha256": config.config_hash(),
        "model": f"{config.model_id}@{config.revision}",
        "subject": config.execution.subject,
        "cells": list(cell.name for cell in config.protocol.cells),
        "request_count": config.protocol.request_count,
        "resources": config.resources.model_dump(mode="json"),
        "remote_execution_default_enabled": False,
        "paid_compute_started": False,
        "next_action": "prepare-deploy",
    }
    if as_json:
        print(canonical_diagnostic_json_bytes(payload).decode(), end="")
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


def prepare_deploy() -> None:
    context = _build_reviewed_context()
    created = datetime.now(UTC)
    challenge = DiagnosticDeployConfirmationChallenge(
        created_at_utc=_utc_microseconds(created),
        expires_at_utc=_utc_microseconds(
            created + timedelta(seconds=DIAGNOSTIC_DEPLOY_CHALLENGE_MAX_AGE_SECONDS)
        ),
        confirmation_nonce=secrets.token_hex(32),
        reviewed_inputs=context.reviewed_inputs,
        app_name=diagnostic_app_name(context.provenance.control_plane_sha256),
    )
    _write_immutable(context.provenance_path, context.provenance.canonical_bytes())
    path = _deploy_challenge_path(context.run_id, challenge.challenge_sha256())
    _write_immutable(path, challenge.canonical_bytes())
    print(
        canonical_diagnostic_json_bytes(
            {
                "status": "prepared_before_deploy",
                "run_id": context.run_id,
                "challenge_path": str(path),
                "challenge_sha256": challenge.challenge_sha256(),
                "expires_at_utc": challenge.expires_at_utc,
                "confirmation": challenge.confirmation_text(),
                "warning": "No Modal request or GPU work was started.",
            }
        ).decode(),
        end="",
    )


def _load_modal() -> ModuleType:
    modal = importlib.import_module("modal")
    if getattr(modal, "__version__", None) != EXPECTED_MODAL_VERSION:
        raise RuntimeError(f"diagnostic manager requires Modal {EXPECTED_MODAL_VERSION}")
    return modal


def _modal_history(app_name: str, *, allow_missing: bool = False) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "app",
                "history",
                app_name,
                "-e",
                DIAGNOSTIC_ENVIRONMENT_NAME,
                "--json",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            shell=False,
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr if isinstance(error.stderr, str) else ""
        if allow_missing and ("not found" in message.lower() or "no app" in message.lower()):
            return []
        raise RuntimeError("Modal diagnostic App history lookup failed") from error
    value = json.loads(result.stdout)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise RuntimeError("Modal diagnostic App history has an unexpected shape")
    return value


def _deployment_version(history: Sequence[Mapping[str, Any]], tag: str) -> int:
    if len(history) != 1:
        raise RuntimeError("content-addressed diagnostic App must have one deployment")
    row = history[0]
    if row.get("tag") != tag or row.get("version") != "v1":
        raise RuntimeError("Modal diagnostic deployment tag or version is invalid")
    return 1


def _reviewed_deployment_version(
    context: _ReviewedContext,
    history: Sequence[Mapping[str, Any]],
    tag: str,
) -> int:
    version = _deployment_version(history, tag)
    row = history[0]
    if row.get("commit") != context.provenance.reviewed_commit_sha[:7]:
        raise RuntimeError("Modal diagnostic deployment commit differs from reviewed main")
    if row.get("client") != EXPECTED_MODAL_VERSION:
        raise RuntimeError("Modal diagnostic deployment client differs from the pin")
    return version


def _object_id(value: object, kind: str) -> str:
    pattern = OBJECT_ID_PATTERNS[kind]
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RuntimeError(f"Modal {kind} object ID is invalid")
    return value


def _resource_created_at(resource: Any) -> str:
    metadata = resource._get_metadata()
    created_at = getattr(getattr(metadata, "creation_info", None), "created_at", None)
    if not isinstance(created_at, (int, float)) or isinstance(created_at, bool):
        raise RuntimeError("Modal resource creation time is invalid")
    return _timestamp_microseconds(float(created_at))


def _function_binding(
    function: Any,
) -> tuple[str, Literal["run_bf16_interface_diagnostic"]]:
    function_id = _object_id(function.object_id, "function")
    name = getattr(function._get_metadata(), "function_name", None)
    if name != DIAGNOSTIC_FUNCTION_NAME:
        raise RuntimeError("Modal returned the wrong diagnostic Function")
    return function_id, cast(Literal["run_bf16_interface_diagnostic"], name)


def _deployment_resources(
    modal: ModuleType,
    *,
    create_if_missing: bool,
) -> tuple[Any, Any]:
    registry = modal.Dict.from_name(
        DIAGNOSTIC_ATTEMPT_REGISTRY_NAME,
        environment_name=DIAGNOSTIC_ENVIRONMENT_NAME,
        create_if_missing=create_if_missing,
    )
    registry.hydrate()
    evidence = modal.Volume.from_name(
        DIAGNOSTIC_EVIDENCE_VOLUME_NAME,
        environment_name=DIAGNOSTIC_ENVIRONMENT_NAME,
        create_if_missing=create_if_missing,
        version=1,
    )
    evidence.hydrate()
    return registry, evidence


def _deploy_remote(context: _ReviewedContext) -> DiagnosticDeploymentIdentity:
    modal = _load_modal()
    control_hash = context.provenance.control_plane_sha256
    app_name = diagnostic_app_name(control_hash)
    tag = diagnostic_deployment_tag(control_hash)
    history = _modal_history(app_name, allow_missing=True)
    recovering = bool(history)
    if recovering:
        version = _reviewed_deployment_version(context, history, tag)
    registry, evidence = _deployment_resources(modal, create_if_missing=not recovering)

    if not recovering:
        environment = os.environ.copy()
        environment.pop("IQL_MEASUREMENT_CONTROL_PLANE_SHA256", None)
        environment.pop("IQL_MEASUREMENT_CONTROL_PLANE_PROVENANCE_PATH", None)
        environment[CONTROL_SHA_ENV] = control_hash
        environment[PROVENANCE_PATH_ENV] = str(context.provenance_path)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "deploy",
                "-e",
                DIAGNOSTIC_ENVIRONMENT_NAME,
                "--name",
                app_name,
                "--tag",
                tag,
                str(PROJECT_ROOT / RUNNER_RELATIVE_PATH),
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            shell=False,
        )
        history = _modal_history(app_name)
        version = _reviewed_deployment_version(context, history, tag)

    function = modal.Function.from_name(
        app_name,
        DIAGNOSTIC_FUNCTION_NAME,
        environment_name=DIAGNOSTIC_ENVIRONMENT_NAME,
    )
    function.hydrate()
    function_id, function_name = _function_binding(function)
    registry_id = _object_id(registry.object_id, "attempt registry")
    registry_created_at = _resource_created_at(registry)
    evidence_id = _object_id(evidence.object_id, "evidence volume")
    stable_registry, stable_evidence = _deployment_resources(modal, create_if_missing=False)
    if (
        _object_id(stable_registry.object_id, "attempt registry") != registry_id
        or _resource_created_at(stable_registry) != registry_created_at
        or _object_id(stable_evidence.object_id, "evidence volume") != evidence_id
        or _modal_history(app_name) != history
    ):
        raise RuntimeError("Modal diagnostic deployment changed while being sealed")
    return DiagnosticDeploymentIdentity(
        deployed_at_utc=_utc_microseconds(),
        control_plane_sha256=control_hash,
        app_name=app_name,
        deployment_version=version,
        deployment_tag=tag,
        function_id=function_id,
        function_name=function_name,
        attempt_registry_id=registry_id,
        attempt_registry_created_at_utc=registry_created_at,
        evidence_volume_id=evidence_id,
    )


def deploy(*, challenge_path: Path, confirmation: str) -> None:
    challenge = cast(
        DiagnosticDeployConfirmationChallenge,
        _read_control_model(challenge_path, DiagnosticDeployConfirmationChallenge),
    )
    challenge.confirm(confirmation)
    context = _build_reviewed_context()
    _require_current_review(context, challenge.reviewed_inputs)
    expected = _deploy_challenge_path(context.run_id, challenge.challenge_sha256()).resolve()
    if _assert_local_artifact_path(challenge_path) != expected:
        raise RuntimeError("deploy challenge path is not content addressed")
    with _operation_lock(context.run_id, "deploy"):
        deployment_path = _deployment_path(context.run_id)
        if deployment_path.exists() or deployment_path.is_symlink():
            raise RuntimeError("diagnostic control plane is already deployed")
        consumed = _deploy_consumption_path(context.run_id, challenge.challenge_sha256())
        if consumed.exists() or consumed.is_symlink():
            raise RuntimeError("diagnostic deploy confirmation was already consumed")
        consumed_at = _utc_microseconds()
        validate_diagnostic_deploy_challenge_not_expired(
            challenge,
            observed_at_utc=consumed_at,
        )
        _write_immutable_json(
            consumed,
            {
                "schema_version": "inkling-bf16-interface-deploy-consumption-v1",
                "status": "authorized_before_deploy",
                "consumed_at_utc": consumed_at,
                "run_id": context.run_id,
                "challenge_sha256": challenge.challenge_sha256(),
                "control_plane_sha256": context.provenance.control_plane_sha256,
            },
        )
        deployment = _deploy_remote(context)
        _write_immutable(deployment_path, deployment.canonical_bytes())
    print(
        canonical_diagnostic_json_bytes(
            {
                "status": "deployed_without_launch",
                "run_id": context.run_id,
                "deployment": deployment.model_dump(mode="json"),
                "paid_gpu_compute_started": False,
                "next_action": "prepare-launch",
            }
        ).decode(),
        end="",
    )


def _read_deployment(run_id: str) -> DiagnosticDeploymentIdentity:
    return cast(
        DiagnosticDeploymentIdentity,
        _read_control_model(_deployment_path(run_id), DiagnosticDeploymentIdentity),
    )


def _launch_already_consumed(run_id: str) -> bool:
    consumption = _launch_consumption_path(run_id)
    intents = _control_path(run_id, "launch-intents")
    if intents.is_symlink():
        raise RuntimeError("diagnostic launch-intent directory is a symbolic link")
    return (
        consumption.exists()
        or consumption.is_symlink()
        or (intents.is_dir() and any(intents.iterdir()))
    )


def prepare_launch() -> None:
    context = _build_reviewed_context()
    if _launch_already_consumed(context.run_id):
        raise RuntimeError("the one diagnostic launch was already consumed")
    deployment = _read_deployment(context.run_id)
    deployment.validate_reviewed_inputs(context.reviewed_inputs)
    created = datetime.now(UTC)
    challenge = DiagnosticLaunchConfirmationChallenge(
        created_at_utc=_utc_microseconds(created),
        expires_at_utc=_utc_microseconds(
            created + timedelta(seconds=DIAGNOSTIC_LAUNCH_CHALLENGE_MAX_AGE_SECONDS)
        ),
        authorization_nonce=secrets.token_hex(32),
        run_id=context.run_id,
        reviewed_inputs=context.reviewed_inputs,
        deployment=deployment,
    )
    path = _launch_challenge_path(context.run_id, challenge.challenge_sha256())
    _write_immutable(path, challenge.canonical_bytes())
    print(
        canonical_diagnostic_json_bytes(
            {
                "status": "prepared_before_launch",
                "run_id": context.run_id,
                "challenge_path": str(path),
                "challenge_sha256": challenge.challenge_sha256(),
                "expires_at_utc": challenge.expires_at_utc,
                "confirmation": challenge.confirmation_text(),
                "warning": "No Modal request or GPU work was started.",
            }
        ).decode(),
        end="",
    )


def _validate_remote_deployment(
    deployment: DiagnosticDeploymentIdentity,
) -> tuple[Any, Any, Any]:
    modal = _load_modal()
    history = _modal_history(deployment.app_name)
    if _deployment_version(history, deployment.deployment_tag) != deployment.deployment_version:
        raise RuntimeError("Modal diagnostic deployment differs from the local seal")
    function = modal.Function.from_name(
        deployment.app_name,
        DIAGNOSTIC_FUNCTION_NAME,
        environment_name=DIAGNOSTIC_ENVIRONMENT_NAME,
    )
    function.hydrate()
    if _function_binding(function) != (deployment.function_id, deployment.function_name):
        raise RuntimeError("Modal diagnostic Function differs from the local seal")
    registry = modal.Dict.from_name(
        DIAGNOSTIC_ATTEMPT_REGISTRY_NAME,
        environment_name=DIAGNOSTIC_ENVIRONMENT_NAME,
        create_if_missing=False,
    )
    registry.hydrate()
    if (
        _object_id(registry.object_id, "attempt registry") != deployment.attempt_registry_id
        or _resource_created_at(registry) != deployment.attempt_registry_created_at_utc
    ):
        raise RuntimeError("Modal diagnostic attempt registry differs from the local seal")
    evidence = modal.Volume.from_name(
        DIAGNOSTIC_EVIDENCE_VOLUME_NAME,
        environment_name=DIAGNOSTIC_ENVIRONMENT_NAME,
        create_if_missing=False,
        version=1,
    )
    evidence.hydrate()
    if _object_id(evidence.object_id, "evidence volume") != deployment.evidence_volume_id:
        raise RuntimeError("Modal diagnostic evidence Volume differs from the local seal")
    if _modal_history(deployment.app_name) != history:
        raise RuntimeError("Modal diagnostic App changed during validation")
    return function, registry, evidence


def _remote_path(relative: str) -> str:
    canonical = validate_repository_relative_path(relative)
    return f"/{canonical}"


def _remote_read(
    volume: Any,
    relative: str,
    *,
    maximum_bytes: int = DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
    expected_size_bytes: int | None = None,
) -> bytes | None:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("remote diagnostic read limit must be positive")
    if expected_size_bytes is not None and (
        type(expected_size_bytes) is not int
        or expected_size_bytes <= 0
        or expected_size_bytes > maximum_bytes
    ):
        raise ValueError("remote diagnostic expected size exceeds its read limit")
    modal = _load_modal()
    payload = bytearray()
    read_limit = maximum_bytes if expected_size_bytes is None else expected_size_bytes
    try:
        for chunk in volume.read_file(_remote_path(relative)):
            if not isinstance(chunk, bytes):
                raise RuntimeError("Modal returned non-byte diagnostic evidence")
            if len(chunk) > read_limit - len(payload):
                raise RuntimeError("remote diagnostic record exceeds its size limit")
            payload.extend(chunk)
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
    if expected_size_bytes is not None and len(payload) != expected_size_bytes:
        raise RuntimeError("remote diagnostic record size differs from its reference")
    return bytes(payload)


def _remote_write_immutable(volume: Any, relative: str, payload: bytes) -> None:
    if _remote_read(volume, relative) is not None:
        raise RuntimeError("remote immutable diagnostic record already exists")
    with volume.batch_upload(force=False) as upload:
        upload.put_file(io.BytesIO(payload), _remote_path(relative), mode=0o400)
    if _remote_read(volume, relative) != payload:
        raise RuntimeError("remote diagnostic record readback differs from uploaded bytes")


def _list_remote_files(
    volume: Any,
    relative: str,
    *,
    maximum_file_bytes: int = DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
    maximum_entries: int = 1,
) -> tuple[tuple[str, int], ...]:
    if type(maximum_file_bytes) is not int or maximum_file_bytes <= 0:
        raise ValueError("remote diagnostic listing byte limit must be positive")
    if type(maximum_entries) is not int or maximum_entries <= 0:
        raise ValueError("remote diagnostic listing count must be positive")
    modal = _load_modal()
    try:
        entries = volume.listdir(_remote_path(relative))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return ()
    result: list[tuple[str, int]] = []
    for entry in entries:
        if len(result) >= maximum_entries:
            raise RuntimeError("remote diagnostic directory exceeds its entry limit")
        path = getattr(entry, "path", None)
        size = getattr(entry, "size", None)
        type_name = getattr(getattr(entry, "type", None), "name", None)
        if (
            not isinstance(path, str)
            or type(size) is not int
            or size <= 0
            or size > maximum_file_bytes
            or type_name != "FILE"
        ):
            raise RuntimeError("remote diagnostic directory contains an invalid entry")
        result.append((path.removeprefix("/"), size))
    return tuple(result)


def _read_only_remote_file(
    volume: Any,
    *,
    relative_path: str,
    expected_size_bytes: int,
    maximum_bytes: int,
    label: str,
) -> bytes:
    entries = _list_remote_files(
        volume,
        PurePosixPath(relative_path).parent.as_posix(),
        maximum_file_bytes=maximum_bytes,
    )
    if entries != ((relative_path, expected_size_bytes),):
        raise RuntimeError(f"remote {label} directory differs from its exact reference")
    payload = _remote_read(
        volume,
        relative_path,
        maximum_bytes=maximum_bytes,
        expected_size_bytes=expected_size_bytes,
    )
    if payload is None:
        raise RuntimeError(f"remote {label} disappeared during validation")
    return payload


def _cancel_call(call: Any, reason: str) -> RuntimeError:
    try:
        call.cancel(terminate_containers=True)
    except Exception as error:
        return RuntimeError(f"{reason}; call cancellation also failed: {error}")
    return RuntimeError(f"{reason}; call cancellation was requested")


def _validated_call_id(call: Any) -> str:
    call_id = getattr(call, "object_id", None)
    if not isinstance(call_id, str) or CALL_ID_PATTERN.fullmatch(call_id) is None:
        raise RuntimeError("Modal diagnostic call ID is invalid")
    return call_id


def _assert_remote_attempt_unconsumed(registry: Any, volume: Any, *, run_id: str) -> None:
    key = diagnostic_attempt_registry_key(run_id)
    present = registry.contains(key)
    if type(present) is not bool:
        raise RuntimeError("Modal attempt registry returned an invalid presence value")
    root = f"runs/{run_id}/{DIAGNOSTIC_STAGE}"
    evidence_roots = (
        f"{root}/control/launch-intents",
        f"{root}/control/post-spawn-acceptances",
        f"{root}/control/attempt-claims",
        f"{root}/private/raw",
        f"{root}/terminal/success",
        f"{root}/terminal/failure",
    )
    if present or any(_list_remote_files(volume, path) for path in evidence_roots):
        raise RuntimeError("remote evidence shows that the diagnostic attempt was consumed")


def _publish_acceptance(
    volume: Any,
    acceptance: DiagnosticPostSpawnAcceptance,
) -> None:
    path = diagnostic_post_spawn_acceptance_path(
        acceptance.run_id,
        acceptance.launch_intent_sha256,
    )
    payload = acceptance.canonical_bytes()
    _remote_write_immutable(volume, path, payload)
    validate_diagnostic_post_spawn_acceptance(
        _remote_read(volume, path) or b"",
        expected=acceptance,
        acceptance_sha256=acceptance.acceptance_sha256(),
        evidence_path=path,
    )


def launch(*, challenge_path: Path, confirmation: str) -> None:
    challenge = cast(
        DiagnosticLaunchConfirmationChallenge,
        _read_control_model(challenge_path, DiagnosticLaunchConfirmationChallenge),
    )
    challenge.confirm(confirmation)
    context = _build_reviewed_context()
    _require_current_review(context, challenge.reviewed_inputs)
    if challenge.run_id != context.run_id:
        raise RuntimeError("diagnostic launch challenge has the wrong run ID")
    deployment = _read_deployment(context.run_id)
    if challenge.deployment != deployment:
        raise RuntimeError("diagnostic launch challenge differs from sealed deployment")
    expected = _launch_challenge_path(context.run_id, challenge.challenge_sha256()).resolve()
    if _assert_local_artifact_path(challenge_path) != expected:
        raise RuntimeError("diagnostic launch challenge path is not content addressed")

    with _operation_lock(context.run_id, "launch"):
        if _launch_already_consumed(context.run_id):
            raise RuntimeError("the one diagnostic launch was already consumed")
        function, registry, evidence = _validate_remote_deployment(deployment)
        _assert_remote_attempt_unconsumed(registry, evidence, run_id=context.run_id)
        authorized_at = _utc_microseconds()
        intent = build_diagnostic_launch_intent(
            challenge,
            confirmation=confirmation,
            authorized_at_utc=authorized_at,
        )
        intent_sha256 = intent.intent_sha256()
        _write_immutable_json(
            _launch_consumption_path(context.run_id),
            {
                "schema_version": "inkling-bf16-interface-launch-consumption-v1",
                "status": "authorized_before_spawn",
                "consumed_at_utc": authorized_at,
                "run_id": context.run_id,
                "challenge_sha256": challenge.challenge_sha256(),
                "launch_intent_sha256": intent_sha256,
                "control_plane_sha256": context.provenance.control_plane_sha256,
            },
        )
        _write_immutable(
            _launch_intent_local_path(context.run_id, intent_sha256),
            intent.canonical_bytes(),
        )
        remote_intent_path = diagnostic_launch_intent_path(context.run_id, intent_sha256)
        _remote_write_immutable(evidence, remote_intent_path, intent.canonical_bytes())
        validate_diagnostic_launch_intent(
            _remote_read(evidence, remote_intent_path) or b"",
            expected=intent,
            intent_sha256=intent_sha256,
            evidence_path=remote_intent_path,
        )

        call = function.spawn(context.run_id, intent_sha256)
        try:
            call_id = _validated_call_id(call)
            acceptance = build_diagnostic_post_spawn_acceptance(
                intent,
                accepted_at_utc=_utc_microseconds(),
                call_id=call_id,
            )
            _publish_acceptance(evidence, acceptance)
            _write_immutable(
                _acceptance_local_path(context.run_id, intent_sha256),
                acceptance.canonical_bytes(),
            )
            receipt = {
                "schema_version": "inkling-bf16-interface-call-receipt-v1",
                "status": "accepted_after_spawn",
                "run_id": context.run_id,
                "call_id": call_id,
                "launch_intent_sha256": intent_sha256,
                "post_spawn_acceptance_sha256": acceptance.acceptance_sha256(),
                "deployment_sha256": hashlib.sha256(deployment.canonical_bytes()).hexdigest(),
                "function_return_is_success_evidence": False,
            }
            _write_immutable_json(
                _call_receipt_path(context.run_id, intent_sha256),
                receipt,
            )
        except Exception as error:
            raise _cancel_call(call, "post-spawn diagnostic acceptance failed") from error
    print(canonical_diagnostic_json_bytes(receipt).decode(), end="")


def _read_launch_binding(run_id: str, call_id: str) -> _LaunchBinding:
    deployment = _read_deployment(run_id)
    calls = _run_root(run_id) / "calls"
    if calls.is_symlink() or not calls.is_dir():
        raise RuntimeError("local diagnostic call receipt is missing")
    entries = tuple(calls.glob("*.json"))
    if len(entries) != 1 or entries[0].is_symlink():
        raise RuntimeError("expected exactly one local diagnostic call receipt")
    payload = entries[0].read_bytes()
    raw = strict_diagnostic_json_object(
        payload,
        maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
    )
    if payload != canonical_diagnostic_json_bytes(raw):
        raise RuntimeError("local diagnostic call receipt is not canonical")
    intent_sha256 = raw.get("launch_intent_sha256")
    if (
        raw.get("schema_version") != "inkling-bf16-interface-call-receipt-v1"
        or raw.get("status") != "accepted_after_spawn"
        or raw.get("run_id") != run_id
        or raw.get("call_id") != call_id
        or not isinstance(intent_sha256, str)
        or SHA256_PATTERN.fullmatch(intent_sha256) is None
        or entries[0].name != f"{intent_sha256}.json"
        or raw.get("deployment_sha256") != hashlib.sha256(deployment.canonical_bytes()).hexdigest()
        or raw.get("function_return_is_success_evidence") is not False
    ):
        raise RuntimeError("local call receipt differs from requested diagnostic call")
    intent = cast(
        DiagnosticLaunchIntent,
        _read_control_model(
            _launch_intent_local_path(run_id, intent_sha256),
            DiagnosticLaunchIntent,
        ),
    )
    if intent.intent_sha256() != intent_sha256 or intent.deployment != deployment:
        raise RuntimeError("local diagnostic intent differs from sealed deployment")
    acceptance = cast(
        DiagnosticPostSpawnAcceptance,
        _read_control_model(
            _acceptance_local_path(run_id, intent_sha256),
            DiagnosticPostSpawnAcceptance,
        ),
    )
    if (
        acceptance.call_id != call_id
        or acceptance.launch_intent_sha256 != intent_sha256
        or acceptance.deployment != deployment
        or raw.get("post_spawn_acceptance_sha256") != acceptance.acceptance_sha256()
    ):
        raise RuntimeError("local diagnostic acceptance differs from requested call")
    return _LaunchBinding(deployment=deployment, intent=intent, acceptance=acceptance)


def _reviewed_git_blobs(
    provenance: DiagnosticControlPlaneProvenance,
) -> dict[str, bytes]:
    commit = provenance.reviewed_commit_sha
    resolved_commit = _git_text("rev-parse", "--verify", f"{commit}^{{commit}}")
    if resolved_commit != commit:
        raise RuntimeError("accepted diagnostic Git commit is unavailable or ambiguous")
    resolved_tree = _git_text("rev-parse", "--verify", f"{commit}^{{tree}}")
    if resolved_tree != provenance.reviewed_tree_sha:
        raise RuntimeError("accepted diagnostic Git tree differs from its provenance")

    expected_paths = {item.path.encode("utf-8"): item.path for item in provenance.files}
    entries: dict[str, tuple[str, str, str]] = {}
    for record in _git_bytes("ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if not record:
            continue
        header, separator, path_bytes = record.partition(b"\t")
        path = expected_paths.get(path_bytes)
        if path is None:
            continue
        fields = header.split()
        if separator != b"\t" or len(fields) != 3:
            raise RuntimeError("accepted diagnostic Git tree has an invalid entry")
        try:
            mode, object_type, object_id = (field.decode("ascii") for field in fields)
        except UnicodeDecodeError as error:
            raise RuntimeError("accepted diagnostic Git tree has an invalid identity") from error
        if path in entries:
            raise RuntimeError("accepted diagnostic Git tree contains a duplicate path")
        entries[path] = (mode, object_type, object_id)

    files: dict[str, bytes] = {}
    for item in provenance.files:
        entry = entries.get(item.path)
        if entry is None:
            raise RuntimeError(f"accepted diagnostic Git path is missing: {item.path}")
        mode, object_type, object_id = entry
        if mode not in {"100644", "100755"} or object_type != "blob":
            raise RuntimeError(f"accepted diagnostic Git path is not a regular blob: {item.path}")
        if GIT_OBJECT_PATTERN.fullmatch(object_id) is None:
            raise RuntimeError(f"accepted diagnostic Git blob identity is invalid: {item.path}")
        payload = _git_bytes("cat-file", "blob", object_id)
        if len(payload) != item.size_bytes or hashlib.sha256(payload).hexdigest() != item.sha256:
            raise RuntimeError(
                f"accepted diagnostic Git blob differs from its provenance: {item.path}"
            )
        files[item.path] = payload

    validate_diagnostic_control_plane_provenance(
        provenance,
        reviewed_commit_sha=commit,
        reviewed_tree_sha=resolved_tree,
        files=files,
        required_paths=tuple(item.path for item in provenance.files),
    )
    return files


def _load_reviewed_bundle(
    binding: _LaunchBinding,
) -> InklingBF16InterfaceDiagnosticBundle:
    reviewed = binding.intent.reviewed_inputs
    files = _reviewed_git_blobs(reviewed.control_plane)
    with TemporaryDirectory(prefix="inkling-bf16-diagnostic-status-") as directory:
        root = Path(directory).resolve()
        for item in reviewed.control_plane.files:
            target = root.joinpath(*PurePosixPath(item.path).parts)
            if target.relative_to(root).as_posix() != item.path:
                raise RuntimeError(
                    f"reconstructed diagnostic path differs from its provenance: {item.path}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(files[item.path])
            payload = target.read_bytes()
            if (
                len(payload) != item.size_bytes
                or hashlib.sha256(payload).hexdigest() != item.sha256
            ):
                raise RuntimeError(
                    f"reconstructed diagnostic file differs from its provenance: {item.path}"
                )
        bundle = load_bf16_interface_diagnostic_bundle(root)
    if (
        bundle.config.config_hash() != reviewed.resolved_config_sha256
        or diagnostic_protocol_sha256(bundle.config) != reviewed.protocol_sha256
        or diagnostic_workload_sha256(bundle.config) != reviewed.workload_sha256
    ):
        raise RuntimeError("local diagnostic scope differs from accepted launch")
    return bundle


def _bound_attempt_claim(
    registry: Any,
    volume: Any,
    binding: _LaunchBinding,
) -> _AttemptInspection:
    run_id = binding.intent.run_id
    key = diagnostic_attempt_registry_key(run_id)
    present = registry.contains(key)
    if type(present) is not bool:
        raise RuntimeError("Modal attempt registry returned an invalid presence value")
    live_payload = registry.get(key) if present else None
    root = f"runs/{run_id}/{DIAGNOSTIC_STAGE}/control/attempt-claims"
    durable = _list_remote_files(volume, root)
    if len(durable) > 1:
        raise RuntimeError("diagnostic evidence has multiple attempt claims")
    if not present and not durable:
        return _AttemptInspection(claim=None, durable=False)
    if present and not isinstance(live_payload, bytes):
        raise RuntimeError("live diagnostic attempt claim is missing or invalid")
    durable_payload: bytes | None = None
    durable_path: str | None = None
    if durable:
        durable_path, durable_size = durable[0]
        durable_payload = _remote_read(
            volume,
            durable_path,
            expected_size_bytes=durable_size,
        )
        if durable_payload is None:
            raise RuntimeError("durable diagnostic attempt claim disappeared")
    claim_payload = live_payload if isinstance(live_payload, bytes) else durable_payload
    if claim_payload is None:
        raise RuntimeError("diagnostic attempt claim is unavailable")
    strict_diagnostic_json_object(
        claim_payload,
        maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
    )
    claim = DiagnosticAttemptClaim.model_validate_json(claim_payload, strict=True)
    digest = claim.claim_sha256()
    if claim_payload != claim.canonical_bytes():
        raise RuntimeError("diagnostic attempt claim is not canonical")
    expected_path = diagnostic_attempt_claim_path(run_id, digest)
    if durable_payload is not None:
        if durable_path != expected_path:
            raise RuntimeError("durable diagnostic attempt path is not content addressed")
        if isinstance(live_payload, bytes) and durable_payload != live_payload:
            raise RuntimeError("durable diagnostic attempt differs from atomic claim")
        validate_diagnostic_attempt_claim(
            durable_payload,
            expected=claim,
            claim_sha256=digest,
            evidence_path=expected_path,
        )
    acceptance = binding.acceptance
    if (
        claim.registry_name != binding.deployment.attempt_registry_name
        or claim.registry_id != binding.deployment.attempt_registry_id
        or claim.registry_created_at_utc != binding.deployment.attempt_registry_created_at_utc
        or claim.registry_key != key
        or claim.run_id != run_id
        or claim.call_id != acceptance.call_id
        or claim.launch_intent_sha256 != binding.intent.intent_sha256()
        or claim.post_spawn_acceptance_sha256 != acceptance.acceptance_sha256()
        or claim.reviewed_config_file_sha256
        != binding.intent.reviewed_inputs.diagnostic_config.sha256
        or claim.resolved_config_sha256 != binding.intent.reviewed_inputs.resolved_config_sha256
        or claim.control_plane_sha256 != binding.deployment.control_plane_sha256
        or _parse_utc_microseconds(claim.claimed_at_utc)
        < _parse_utc_microseconds(acceptance.accepted_at_utc)
    ):
        raise RuntimeError("diagnostic attempt differs from accepted launch")
    return _AttemptInspection(claim=claim, durable=bool(durable))


def _provider_state(call_id: str) -> tuple[str, int]:
    modal = _load_modal()
    roots = modal.FunctionCall.from_id(call_id).get_call_graph()
    if not isinstance(roots, list):
        raise RuntimeError("Modal diagnostic call graph has an unexpected shape")
    pending = list(roots)
    states: list[str] = []
    while pending:
        if len(states) >= MAX_CALL_GRAPH_NODES:
            raise RuntimeError("Modal diagnostic call graph exceeds its size limit")
        node = pending.pop()
        status = getattr(getattr(node, "status", None), "name", None)
        children = getattr(node, "children", None)
        if not isinstance(status, str) or not isinstance(children, list):
            raise RuntimeError("Modal diagnostic call graph contains invalid metadata")
        states.append(status.casefold())
        pending.extend(children)
    if set(states) & {"failure", "init_failure", "terminated", "timeout"}:
        return "failed_without_terminal_evidence", 1
    if states and set(states) == {"success"}:
        return "returned_without_terminal_evidence", 1
    return "running_or_queued", 0


def _read_private_raw(
    volume: Any,
    reference: DiagnosticPrivateRawReference,
) -> tuple[bytes, DiagnosticPrivateRawEvidence]:
    if reference.evidence_root != EVIDENCE_MOUNT_ROOT:
        raise RuntimeError("diagnostic private evidence has the wrong mounted root")
    payload = _read_only_remote_file(
        volume,
        relative_path=reference.relative_path,
        expected_size_bytes=reference.size_bytes,
        maximum_bytes=DIAGNOSTIC_RAW_RECORD_MAX_BYTES,
        label="private diagnostic evidence",
    )
    validate_diagnostic_private_raw_reference(payload, expected=reference)
    raw = parse_diagnostic_private_raw_evidence(payload, run_id=reference.run_id)
    return payload, raw


def _validate_private_scope(
    raw: DiagnosticPrivateRawEvidence,
    *,
    binding: _LaunchBinding,
    claim: DiagnosticAttemptClaim,
    bundle: InklingBF16InterfaceDiagnosticBundle,
) -> None:
    config = bundle.config
    bf16 = bundle.bf16
    validate_diagnostic_private_trials(raw, bundle=bundle)
    source_asset_manifest_sha256 = hashlib.sha256(
        canonical_diagnostic_json_bytes(
            {
                "schema_version": "inkling-bf16-interface-source-assets-v1",
                "assets": [
                    config.source_assets.config.model_dump(mode="json"),
                    config.source_assets.chat_template.model_dump(mode="json"),
                    config.source_assets.tokenizer_json.model_dump(mode="json"),
                    config.source_assets.tokenizer_config.model_dump(mode="json"),
                ],
            }
        )
    ).hexdigest()
    expected = {
        "run_id": binding.intent.run_id,
        "control_plane_sha256": binding.deployment.control_plane_sha256,
        "reviewed_config_file_sha256": binding.intent.reviewed_inputs.diagnostic_config.sha256,
        "resolved_config_sha256": binding.intent.reviewed_inputs.resolved_config_sha256,
        "launch_intent_sha256": binding.intent.intent_sha256(),
        "post_spawn_acceptance_sha256": binding.acceptance.acceptance_sha256(),
        "call_id": binding.acceptance.call_id,
        "attempt_claim_sha256": claim.claim_sha256(),
        "model_id": config.model_id,
        "model_revision": config.revision,
        "architecture": config.architecture,
        "protocol_sha256": diagnostic_protocol_sha256(config),
        "workload_sha256": diagnostic_workload_sha256(config),
        "bf16_inventory_sha256": bf16.bf16_inventory_sha256,
        "bf16_shard_count": bf16.bf16_shard_count,
        "bf16_total_bytes": bf16.bf16_total_bytes,
        "source_asset_manifest_sha256": source_asset_manifest_sha256,
    }
    if any(getattr(raw, field) != value for field, value in expected.items()):
        raise RuntimeError("private diagnostic evidence differs from the accepted scope")

    runtime = raw.runtime_identity
    base_runtime = config.runtime
    expected_runtime_fields = {
        "repository": base_runtime.repository,
        "repository_commit": base_runtime.commit,
        "cuda_image": base_runtime.cuda_image,
        "cuda_image_digest": base_runtime.cuda_image_digest,
        "platform": base_runtime.platform,
    }
    if any(getattr(runtime, field) != value for field, value in expected_runtime_fields.items()):
        raise RuntimeError("private diagnostic runtime differs from the reviewed runtime")
    expected_patches = (
        (
            base_runtime.instrumentation_patch_path,
            base_runtime.instrumentation_patch_sha256,
        ),
        (
            config.runtime_measurement_patch.path,
            config.runtime_measurement_patch.sha256,
        ),
    )
    observed_patches = tuple(
        (patch.path, patch.sha256) for patch in runtime.patches_applied_in_order
    )
    expected_base_binaries = tuple(
        (binary.name, binary.path, binary.sha256, binary.size_bytes)
        for binary in base_runtime.binaries
    )
    observed_base_binaries = tuple(
        (binary.name, binary.path, binary.sha256, binary.size_bytes)
        for binary in runtime.base_pre_measurement_patch_executables
    )
    expected_cmake_definitions = (
        "CMAKE_BUILD_TYPE=Release",
        "BUILD_SHARED_LIBS=ON",
        *base_runtime.cmake_definitions,
        "CMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE",
    )
    if (
        raw.runtime_identity_sha256 != diagnostic_runtime_identity_sha256(runtime)
        or raw.runtime_manifest_sha256 != runtime.manifest_sha256
        or observed_patches != expected_patches
        or runtime.patches_applied_in_order[1].size_bytes
        != config.runtime_measurement_patch.size_bytes
        or observed_base_binaries != expected_base_binaries
        or runtime.effective_cmake_definitions != expected_cmake_definitions
    ):
        raise RuntimeError("private diagnostic runtime identity is not the reviewed build")

    hardware = raw.hardware_identity
    expected_logical_devices = tuple(config.placement.logical_devices.split(","))
    if (
        raw.hardware_identity_sha256 != hardware.identity_sha256
        or hardware.backend != config.placement.backend
        or hardware.logical_devices != expected_logical_devices
        or len(hardware.gpus) != config.resources.gpu_count
        or any(
            gpu.name != "NVIDIA B300 SXM6 AC"
            or gpu.compute_capability != config.resources.compute_capability
            for gpu in hardware.gpus
        )
        or hardware.gpu_layers != config.placement.gpu_layers
        or hardware.cpu_moe_layers != config.placement.cpu_moe_layers
        or hardware.cpu_fallback != config.placement.cpu_fallback
    ):
        raise RuntimeError("private diagnostic hardware is not the reviewed eight-B300 cell")

    staged_model_path = PurePosixPath(
        config.execution.subject_staging_root,
        "bf16",
        bf16.bf16_shards[0].path,
    ).as_posix()
    if raw.command != build_diagnostic_server_command(staged_model_path):
        raise RuntimeError("private diagnostic command differs from the reviewed server command")
    if _parse_utc_microseconds(raw.started_at_utc) < _parse_utc_microseconds(claim.claimed_at_utc):
        raise RuntimeError("private diagnostic evidence predates its durable attempt")


def _validate_success_scope(
    receipt: DiagnosticSuccessTerminalReceipt,
    raw: DiagnosticPrivateRawEvidence,
    *,
    bundle: InklingBF16InterfaceDiagnosticBundle,
) -> None:
    config = bundle.config
    bf16 = bundle.bf16
    expected = {
        "model_id": config.model_id,
        "model_revision": config.revision,
        "architecture": config.architecture,
        "bf16_inventory_sha256": bf16.bf16_inventory_sha256,
        "bf16_shard_count": bf16.bf16_shard_count,
        "bf16_total_bytes": bf16.bf16_total_bytes,
        "protocol_sha256": diagnostic_protocol_sha256(config),
        "workload_sha256": diagnostic_workload_sha256(config),
    }
    if any(getattr(receipt, field) != value for field, value in expected.items()):
        raise RuntimeError("diagnostic success receipt differs from reviewed scope")
    linked = {
        "runtime_identity": raw.runtime_identity,
        "runtime_identity_sha256": raw.runtime_identity_sha256,
        "runtime_manifest_sha256": raw.runtime_manifest_sha256,
        "hardware_identity_sha256": raw.hardware_identity_sha256,
        "command_sha256": raw.command_sha256,
        "server_log_sha256": raw.server_log_sha256,
        "server_log_size_bytes": raw.server_log_size_bytes,
    }
    if any(getattr(receipt, field) != value for field, value in linked.items()):
        raise RuntimeError("diagnostic success receipt differs from private runtime evidence")
    rebuilt = build_diagnostic_rollup(
        raw,
        private_raw_content_sha256=receipt.private_raw_reference.content_sha256,
    )
    if receipt.rollup != rebuilt:
        raise RuntimeError("diagnostic compact rollup differs from private evidence")


def _terminal_evidence(
    volume: Any,
    binding: _LaunchBinding,
    attempt: _AttemptInspection,
    *,
    bundle: InklingBF16InterfaceDiagnosticBundle,
) -> tuple[DiagnosticTerminalReceiptReference, DiagnosticTerminalReceipt] | None:
    run_id = binding.intent.run_id
    candidates: list[tuple[Literal["success", "failure"], str, int]] = []
    for outcome in ("success", "failure"):
        root = f"runs/{run_id}/{DIAGNOSTIC_STAGE}/terminal/{outcome}"
        for path, size in _list_remote_files(volume, root):
            candidates.append((outcome, path, size))
    if not candidates:
        return None
    claim = attempt.claim
    if len(candidates) != 1 or claim is None or not attempt.durable:
        raise RuntimeError(
            "diagnostic terminal evidence conflicts or lacks a durable attempt claim"
        )
    outcome, path, size = candidates[0]
    payload = _remote_read(volume, path, expected_size_bytes=size)
    if payload is None:
        raise RuntimeError("diagnostic terminal receipt disappeared during validation")
    receipt = parse_diagnostic_terminal_receipt(
        payload,
        run_id=run_id,
        outcome=outcome,
    )
    reference = build_diagnostic_terminal_receipt_reference(
        payload,
        evidence_root=EVIDENCE_MOUNT_ROOT,
        run_id=run_id,
        outcome=outcome,
    )
    validate_diagnostic_terminal_receipt_reference(payload, expected=reference)
    if path != reference.relative_path or size != reference.size_bytes:
        raise RuntimeError("diagnostic terminal path or size is not content addressed")
    expected_bindings = {
        "stage": DIAGNOSTIC_STAGE,
        "run_id": run_id,
        "control_plane_sha256": binding.deployment.control_plane_sha256,
        "reviewed_config_file_sha256": binding.intent.reviewed_inputs.diagnostic_config.sha256,
        "resolved_config_sha256": binding.intent.reviewed_inputs.resolved_config_sha256,
        "launch_intent_sha256": binding.intent.intent_sha256(),
        "post_spawn_acceptance_sha256": binding.acceptance.acceptance_sha256(),
        "call_id": binding.acceptance.call_id,
        "attempt_claim_sha256": claim.claim_sha256(),
    }
    if any(getattr(receipt, field) != value for field, value in expected_bindings.items()):
        raise RuntimeError("diagnostic terminal receipt differs from accepted attempt")
    if _parse_utc_microseconds(receipt.completed_at_utc) < _parse_utc_microseconds(
        claim.claimed_at_utc
    ):
        raise RuntimeError("diagnostic terminal receipt predates its durable attempt")
    if receipt.private_raw_reference is not None:
        _, raw = _read_private_raw(volume, receipt.private_raw_reference)
        _validate_private_scope(raw, binding=binding, claim=claim, bundle=bundle)
        runtime_links = {
            "runtime_identity_sha256": raw.runtime_identity_sha256,
            "runtime_manifest_sha256": raw.runtime_manifest_sha256,
            "hardware_identity_sha256": raw.hardware_identity_sha256,
        }
        if any(getattr(receipt, field) != value for field, value in runtime_links.items()):
            raise RuntimeError("diagnostic terminal receipt differs from private runtime identity")
        if _parse_utc_microseconds(raw.completed_at_utc) > _parse_utc_microseconds(
            receipt.completed_at_utc
        ):
            raise RuntimeError("diagnostic terminal receipt predates private completion")
        if receipt.status == "completed":
            _validate_success_scope(
                receipt,
                raw,
                bundle=bundle,
            )
    return reference, receipt


def status(*, run_id: str, call_id: str) -> int:
    if RUN_ID_PATTERN.fullmatch(run_id) is None or CALL_ID_PATTERN.fullmatch(call_id) is None:
        raise ValueError("diagnostic run or Modal call ID is invalid")
    binding = _read_launch_binding(run_id, call_id)
    bundle = _load_reviewed_bundle(binding)
    _, registry, volume = _validate_remote_deployment(binding.deployment)
    intent_sha256 = binding.intent.intent_sha256()
    intent_path = diagnostic_launch_intent_path(run_id, intent_sha256)
    remote_intent = _read_only_remote_file(
        volume,
        relative_path=intent_path,
        expected_size_bytes=len(binding.intent.canonical_bytes()),
        maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
        label="diagnostic launch intent",
    )
    validate_diagnostic_launch_intent(
        remote_intent,
        expected=binding.intent,
        intent_sha256=intent_sha256,
        evidence_path=intent_path,
    )
    acceptance_path = diagnostic_post_spawn_acceptance_path(run_id, intent_sha256)
    remote_acceptance = _read_only_remote_file(
        volume,
        relative_path=acceptance_path,
        expected_size_bytes=len(binding.acceptance.canonical_bytes()),
        maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
        label="diagnostic post-spawn acceptance",
    )
    validate_diagnostic_post_spawn_acceptance(
        remote_acceptance,
        expected=binding.acceptance,
        acceptance_sha256=binding.acceptance.acceptance_sha256(),
        evidence_path=acceptance_path,
    )
    attempt = _bound_attempt_claim(registry, volume, binding)
    terminal = _terminal_evidence(volume, binding, attempt, bundle=bundle)
    claim = attempt.claim
    if terminal is not None:
        reference, receipt = terminal
        success = receipt.status == "completed"
        payload: dict[str, Any] = {
            "status": "completed" if success else "failed",
            "evidence_status": (
                "validated_terminal_success" if success else "validated_terminal_failure"
            ),
            "run_id": run_id,
            "call_id": call_id,
            "terminal_receipt": reference.model_dump(mode="json"),
            "attempt_claim_sha256": None if claim is None else claim.claim_sha256(),
            "diagnostic_completed": receipt.diagnostic_completed,
            "function_return_is_success_evidence": False,
        }
        if receipt.status == "completed":
            payload.update(
                {
                    "request_count": receipt.rollup.request_count,
                    "whole_output_pass_count": receipt.rollup.whole_output_pass_count,
                    "extracted_content_pass_count": (receipt.rollup.extracted_content_pass_count),
                    "gpu_placement_verified": receipt.gpu_placement_verified,
                    "cpu_fallback_observed": receipt.cpu_fallback_observed,
                    "diagnostic_only": True,
                    "quality_claim_allowed": False,
                    "performance_claim_allowed": False,
                }
            )
        print(canonical_diagnostic_json_bytes(payload).decode(), end="")
        return 0 if success else 1
    provider_status, exit_code = _provider_state(call_id)
    print(
        canonical_diagnostic_json_bytes(
            {
                "status": provider_status,
                "evidence_status": "no_terminal_receipt",
                "run_id": run_id,
                "call_id": call_id,
                "attempt_consumed": claim is not None,
                "attempt_claim_sha256": None if claim is None else claim.claim_sha256(),
                "retry_allowed": False,
                "function_return_is_success_evidence": False,
            }
        ).decode(),
        end="",
    )
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--json", action="store_true")
    commands.add_parser("prepare-deploy")
    deploy_parser = commands.add_parser("deploy")
    deploy_parser.add_argument("--challenge", type=Path, required=True)
    deploy_parser.add_argument("--confirm", required=True)
    commands.add_parser("prepare-launch")
    launch_parser = commands.add_parser("launch")
    launch_parser.add_argument("--challenge", type=Path, required=True)
    launch_parser.add_argument("--confirm", required=True)
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--run-id", required=True)
    status_parser.add_argument("--call-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        if arguments.command == "inspect":
            inspect(as_json=arguments.json)
        elif arguments.command == "prepare-deploy":
            prepare_deploy()
        elif arguments.command == "deploy":
            deploy(challenge_path=arguments.challenge, confirmation=arguments.confirm)
        elif arguments.command == "prepare-launch":
            prepare_launch()
        elif arguments.command == "launch":
            launch(challenge_path=arguments.challenge, confirmation=arguments.confirm)
        elif arguments.command == "status":
            return status(run_id=arguments.run_id, call_id=arguments.call_id)
        else:
            raise RuntimeError("unsupported diagnostic manager command")
    except (
        FileExistsError,
        ImportError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
