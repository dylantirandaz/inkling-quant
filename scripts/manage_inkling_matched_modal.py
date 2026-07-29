"""Prepare, deploy, launch, and inspect one matched Inkling Modal smoke run.

The prepare and inspect commands are local-only.  Deployment and launch require
separate, exact, content-addressed confirmations.  Deployment never launches
compute.  Launch uploads its immutable authorization before it starts one call.
"""

from __future__ import annotations

import argparse
import fcntl
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
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, Final

# Direct CLI use must not create package bytecode in the project tree.
if __name__ == "__main__":
    sys.dont_write_bytecode = True

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
SRC_ROOT: Final = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inkling_quant_lab.gguf.inkling import (  # noqa: E402
    INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
)
from inkling_quant_lab.gguf.inkling_matched import (  # noqa: E402
    BF16_SUBJECT_REFERENCE_RELATIVE_PATH,
    MATCHED_CELL_CONFIG_RELATIVE_PATH,
)
from inkling_quant_lab.gguf.inkling_matched_control import (  # noqa: E402
    MATCHED_ATTEMPT_REGISTRY_NAME,
    MATCHED_ENVIRONMENT_NAME,
    MATCHED_EVIDENCE_VOLUME_NAME,
    MATCHED_FUNCTION_NAME,
    MatchedAttemptClaim,
    MatchedControlPlaneProvenance,
    MatchedDeployConfirmationChallenge,
    MatchedDeploymentIdentity,
    MatchedExecutionResources,
    MatchedLaunchConfirmationChallenge,
    MatchedLaunchIntent,
    MatchedPostSpawnAcceptance,
    MatchedPublicationSnapshot,
    MatchedReviewedInputs,
    MatchedTerminalReceiptReference,
    build_matched_control_plane_provenance,
    build_matched_launch_intent,
    build_matched_terminal_receipt_reference,
    matched_app_name,
    matched_attempt_claim_path,
    matched_attempt_registry_key,
    matched_deployment_tag,
    matched_launch_intent_path,
    matched_post_spawn_acceptance_path,
    matched_publication_state_path,
    strict_matched_json_object,
    validate_matched_attempt_claim,
    validate_matched_post_spawn_acceptance,
    validate_matched_publication_state,
    validate_matched_publication_transition,
    validate_matched_terminal_receipt_reference,
)
from inkling_quant_lab.gguf.inkling_matched_execution import (  # noqa: E402
    MatchedFailureReceipt,
    MatchedRollupReceipt,
)
from inkling_quant_lab.gguf.inkling_matched_preflight import (  # noqa: E402
    InklingMatchedPreflightReport,
    build_matched_preflight_report,
)
from inkling_quant_lab.gguf.inkling_smoke import (  # noqa: E402
    VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH,
)

EXPECTED_MODAL_VERSION: Final = "1.5.0"
RUNNER_RELATIVE_PATH: Final = "scripts/run_inkling_matched_modal.py"
MANAGER_RELATIVE_PATH: Final = "scripts/manage_inkling_matched_modal.py"
OFFLINE_MANAGER_RELATIVE_PATH: Final = "scripts/manage_inkling_matched.py"
PATCH_RELATIVE_PATH: Final = "patches/inkling-smoke-a015409.patch"
ARTIFACT_ROOT: Final = PROJECT_ROOT / "artifacts" / "inkling-matched-modal"
CALL_ID_PATTERN: Final = re.compile(r"^fc-[A-Za-z0-9]+$")
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
MAX_REMOTE_TERMINAL_RECEIPT_BYTES: Final = 512 * 1024
MAX_PROVIDER_CALL_GRAPH_NODES: Final = 256
OBJECT_ID_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "function": re.compile(r"^fu-[A-Za-z0-9]+$"),
    "attempt registry": re.compile(r"^di-[A-Za-z0-9]+$"),
    "evidence volume": re.compile(r"^vo-[A-Za-z0-9]+$"),
}
FIXED_CONTROL_PATHS: Final = (
    MATCHED_CELL_CONFIG_RELATIVE_PATH,
    BF16_SUBJECT_REFERENCE_RELATIVE_PATH,
    VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH,
    INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
    PATCH_RELATIVE_PATH,
    OFFLINE_MANAGER_RELATIVE_PATH,
    MANAGER_RELATIVE_PATH,
    RUNNER_RELATIVE_PATH,
)


@dataclass(frozen=True)
class _ReviewedContext:
    """Exact local state bound into one reviewed deploy or launch."""

    report: InklingMatchedPreflightReport
    provenance: MatchedControlPlaneProvenance
    reviewed_inputs: MatchedReviewedInputs
    run_id: str
    provenance_path: Path


class _PostSpawnAcceptanceMismatchError(RuntimeError):
    """A fresh remote read proved exact acceptance absent or different."""


class _PostSpawnAcceptanceStateUnknownError(RuntimeError):
    """A fresh remote read could not determine acceptance state."""


@dataclass(frozen=True)
class _StatusLaunchBinding:
    """Exact local authorization and provider acceptance for one status query."""

    launch_intent: MatchedLaunchIntent
    acceptance: MatchedPostSpawnAcceptance


@dataclass(frozen=True)
class _ValidatedTerminalEvidence:
    """One terminal receipt proved by its schema, self-hash, and remote path."""

    reference: MatchedTerminalReceiptReference
    receipt: MatchedRollupReceipt | MatchedFailureReceipt


@dataclass(frozen=True)
class _PublicationInspection:
    """Validated publication history observed in one fresh Volume view."""

    snapshots: tuple[MatchedPublicationSnapshot, ...]

    @property
    def final(self) -> MatchedPublicationSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None


@dataclass(frozen=True)
class _AttemptClaimInspection:
    """Reconciled live and durable identities for one consumed attempt."""

    registry_key: str
    live_claim: MatchedAttemptClaim | None
    live_payload: bytes | None
    durable_claim: MatchedAttemptClaim | None
    durable_payload: bytes | None
    durable_path: str | None

    @property
    def claim_sha256(self) -> str | None:
        claim = self.durable_claim or self.live_claim
        return None if claim is None else claim.claim_sha256()

    @property
    def attempt_consumed_before_volume_bookkeeping(self) -> bool:
        return self.live_claim is not None and self.durable_claim is None


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _utc_microseconds() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _timestamp_microseconds(timestamp: float) -> str:
    if not isinstance(timestamp, (float, int)) or timestamp <= 0:
        raise RuntimeError("Modal resource creation time is invalid")
    return datetime.fromtimestamp(float(timestamp), UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _run_root(run_id: str) -> Path:
    if re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,95}", run_id) is None:
        raise ValueError("matched run ID is invalid")
    return ARTIFACT_ROOT / run_id


def _provenance_path(run_id: str, control_plane_sha256: str) -> Path:
    return _run_root(run_id) / "control" / "provenance" / f"{control_plane_sha256}.json"


def _deploy_challenge_path(run_id: str, challenge_sha256: str) -> Path:
    return _run_root(run_id) / "control" / "deploy-challenges" / f"{challenge_sha256}.json"


def _deployment_path(run_id: str) -> Path:
    return _run_root(run_id) / "control" / "deployment.json"


def _deploy_acceptance_path(run_id: str, challenge_sha256: str) -> Path:
    return _run_root(run_id) / "control" / "deploy-acceptances" / f"{challenge_sha256}.json"


def _deploy_consumption_path(run_id: str, challenge_sha256: str) -> Path:
    return _run_root(run_id) / "control" / "deploy-consumptions" / f"{challenge_sha256}.json"


def _launch_challenge_path(run_id: str, challenge_sha256: str) -> Path:
    return _run_root(run_id) / "control" / "launch-challenges" / f"{challenge_sha256}.json"


def _local_launch_intent_path(run_id: str, launch_intent_sha256: str) -> Path:
    return _run_root(run_id) / "control" / "launch-intents" / f"{launch_intent_sha256}.json"


def _launch_consumption_path(run_id: str) -> Path:
    return _run_root(run_id) / "control" / "launch-consumption.json"


def _local_post_spawn_acceptance_path(
    run_id: str,
    launch_intent_sha256: str,
) -> Path:
    return _run_root(run_id) / "control" / "post-spawn-acceptances" / f"{launch_intent_sha256}.json"


def _call_receipt_path(run_id: str, launch_intent_sha256: str) -> Path:
    return _run_root(run_id) / "calls" / f"{launch_intent_sha256}.json"


def _reject_symlinked_artifact_ancestors(path: Path) -> None:
    try:
        relative = path.relative_to(ARTIFACT_ROOT)
    except ValueError:
        return
    current = ARTIFACT_ROOT
    if current.is_symlink():
        raise RuntimeError("matched artifact root is a symbolic link")
    for part in relative.parts[:-1]:
        current /= part
        if current.is_symlink():
            raise RuntimeError("matched artifact path contains a symbolic link")


@contextmanager
def _exclusive_remote_operation(run_id: str, operation: str) -> Iterator[None]:
    if operation not in {"deploy", "launch"}:
        raise ValueError("matched manager operation is invalid")
    path = _run_root(run_id) / "control" / "manager.lock"
    _reject_symlinked_artifact_ancestors(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError("matched manager lock is a symbolic link")
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another matched manager {operation} operation is active"
            ) from error
        yield
    finally:
        os.close(descriptor)


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write canonical JSON once, allowing only an identical retry."""

    payload = _canonical_json_bytes(value)
    _reject_symlinked_artifact_ancestors(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError(f"immutable path is a symbolic link: {path}")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(f"immutable record exists with different bytes: {path}") from None
        return
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # The exclusive create ensures no previous record is replaced.  Retain a
        # partial file for explicit operator reconciliation instead of hiding it.
        raise


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


def _require_reviewed_main() -> tuple[str, str]:
    status = _git_text("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "matched deployment requires a clean reviewed worktree; commit or remove changes"
        )
    commit_sha = _git_text("rev-parse", "HEAD")
    tree_sha = _git_text("rev-parse", "HEAD^{tree}")
    origin_main = _git_text("rev-parse", "refs/remotes/origin/main")
    if commit_sha != origin_main:
        raise RuntimeError(
            "matched deployment requires HEAD to equal the locally fetched origin/main"
        )
    if (
        re.fullmatch(r"[0-9a-f]{40}", commit_sha) is None
        or re.fullmatch(r"[0-9a-f]{40}", tree_sha) is None
    ):
        raise RuntimeError("reviewed Git identity is invalid")
    return commit_sha, tree_sha


def _closed_control_paths() -> tuple[str, ...]:
    tracked = _git_text("ls-files", "src/inkling_quant_lab").splitlines()
    source_paths = tuple(sorted(path for path in tracked if path.endswith(".py") and path))
    paths = tuple(sorted((*FIXED_CONTROL_PATHS, *source_paths)))
    if len(paths) != len(set(paths)):
        raise RuntimeError("matched control-plane path set contains duplicates")
    if not source_paths:
        raise RuntimeError("matched control-plane source file set is empty")
    return paths


def _read_control_files(paths: Sequence[str]) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    root = PROJECT_ROOT.resolve()
    for relative in paths:
        path = PROJECT_ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"matched control file is not a regular file: {relative}")
        resolved = path.resolve()
        if resolved == root or root not in resolved.parents:
            raise RuntimeError(f"matched control file escapes the project: {relative}")
        files[relative] = path.read_bytes()
    return files


def _matched_run_id(reviewed_inputs: MatchedReviewedInputs) -> str:
    return (
        "inkling-matched-86b4d430-"
        f"{reviewed_inputs.matched_plan_sha256[:12]}-"
        f"{reviewed_inputs.control_plane_sha256[:12]}"
    )


def _build_reviewed_context() -> _ReviewedContext:
    commit_sha, tree_sha = _require_reviewed_main()
    report = build_matched_preflight_report(PROJECT_ROOT)
    paths = _closed_control_paths()
    provenance = build_matched_control_plane_provenance(
        reviewed_commit_sha=commit_sha,
        reviewed_tree_sha=tree_sha,
        files=_read_control_files(paths),
        required_paths=paths,
    )
    reviewed_inputs = MatchedReviewedInputs(
        reviewed_commit_sha=commit_sha,
        reviewed_tree_sha=tree_sha,
        matched_config_sha256=report.config_hash,
        matched_plan_sha256=report.plan_sha256,
        bf16_subject_reference_sha256=report.bf16_reference.reference_sha256,
        q3_verified_export_reference_sha256=report.q3_reference.reference_sha256,
        source_adoption_reference_sha256=report.source_reference.reference_sha256,
        control_plane_sha256=provenance.control_plane_sha256,
        subject_order=("bf16", "q3"),
        resources=MatchedExecutionResources(function_timeout_seconds=14_400),
    )
    reviewed_inputs.validate_control_plane(provenance)
    run_id = _matched_run_id(reviewed_inputs)
    return _ReviewedContext(
        report=report,
        provenance=provenance,
        reviewed_inputs=reviewed_inputs,
        run_id=run_id,
        provenance_path=_provenance_path(
            run_id,
            provenance.control_plane_sha256,
        ),
    )


def _resolve_local_control_path(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("local control record must be one regular file")
    resolved = path.resolve()
    allowed = ARTIFACT_ROOT.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise RuntimeError("local control record is outside the matched artifact root")
    return resolved


def _read_deploy_challenge(path: Path) -> MatchedDeployConfirmationChallenge:
    resolved = _resolve_local_control_path(path)
    payload = resolved.read_bytes()
    challenge = MatchedDeployConfirmationChallenge.model_validate(
        strict_matched_json_object(payload)
    )
    if payload != challenge.canonical_bytes():
        raise RuntimeError("deploy challenge bytes are not canonical")
    return challenge


def _read_launch_challenge(path: Path) -> MatchedLaunchConfirmationChallenge:
    resolved = _resolve_local_control_path(path)
    payload = resolved.read_bytes()
    challenge = MatchedLaunchConfirmationChallenge.model_validate(
        strict_matched_json_object(payload)
    )
    if payload != challenge.canonical_bytes():
        raise RuntimeError("launch challenge bytes are not canonical")
    return challenge


def _read_deployment(run_id: str) -> MatchedDeploymentIdentity:
    path = _deployment_path(run_id)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("matched deployment identity is missing")
    payload = path.read_bytes()
    deployment = MatchedDeploymentIdentity.model_validate(strict_matched_json_object(payload))
    if payload != deployment.canonical_bytes():
        raise RuntimeError("matched deployment identity bytes are not canonical")
    return deployment


def _require_current_review(
    context: _ReviewedContext,
    reviewed_inputs: MatchedReviewedInputs,
) -> None:
    if reviewed_inputs != context.reviewed_inputs:
        raise RuntimeError("challenge differs from the current reviewed origin/main")


def inspect(*, as_json: bool = False) -> None:
    """Inspect local matched controls without importing Modal or contacting a provider."""

    report = build_matched_preflight_report(PROJECT_ROOT)
    payload = {
        "status": report.status,
        "plan_sha256": report.plan_sha256,
        "matched_config_sha256": report.config_hash,
        "model": f"{report.model_id}@{report.revision}",
        "execution_record_status": report.execution.record_status,
        "remote_execution_default_enabled": False,
        "paid_compute_started": False,
        "next_action": "prepare-deploy",
    }
    if as_json:
        print(_canonical_json_bytes(payload).decode("utf-8"), end="")
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


def prepare_deploy() -> None:
    """Create one local content-addressed deployment challenge."""

    context = _build_reviewed_context()
    challenge = MatchedDeployConfirmationChallenge(
        created_at_utc=_utc_microseconds(),
        confirmation_nonce=secrets.token_hex(32),
        reviewed_inputs=context.reviewed_inputs,
        app_name=matched_app_name(context.provenance.control_plane_sha256),
    )
    _write_immutable_json(
        context.provenance_path,
        context.provenance.model_dump(mode="json"),
    )
    path = _deploy_challenge_path(context.run_id, challenge.challenge_sha256())
    _write_immutable_json(path, challenge.model_dump(mode="json"))
    print(
        _canonical_json_bytes(
            {
                "status": "prepared_before_deploy",
                "run_id": context.run_id,
                "challenge_path": str(path),
                "challenge_sha256": challenge.challenge_sha256(),
                "app_name": challenge.app_name,
                "confirmation": challenge.confirmation_text(),
                "warning": "This preparation did not contact Modal or start compute.",
            }
        ).decode("utf-8"),
        end="",
    )


def _load_modal() -> ModuleType:
    modal_module = importlib.import_module("modal")
    version = getattr(modal_module, "__version__", None)
    if version != EXPECTED_MODAL_VERSION:
        raise RuntimeError(
            f"matched manager requires Modal {EXPECTED_MODAL_VERSION}, got {version!r}"
        )
    return modal_module


def _modal_history(app_name: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "app",
            "history",
            app_name,
            "-e",
            MATCHED_ENVIRONMENT_NAME,
            "--json",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise RuntimeError("Modal App history returned an unexpected shape")
    return value


def _modal_history_or_empty(app_name: str) -> list[dict[str, Any]]:
    try:
        return _modal_history(app_name)
    except subprocess.CalledProcessError as error:
        message = error.stderr if isinstance(error.stderr, str) else ""
        if "not found" in message.lower() or "no app" in message.lower():
            return []
        raise RuntimeError("Modal App history lookup failed") from error


def _deployment_version(
    history: Sequence[Mapping[str, Any]],
    *,
    deployment_tag: str,
) -> int:
    matching = [row for row in history if row.get("tag") == deployment_tag]
    if len(matching) != 1:
        raise RuntimeError("expected exactly one deployment with the reviewed tag")
    version_text = matching[0].get("version")
    if not isinstance(version_text, str) or re.fullmatch(r"v[1-9][0-9]*", version_text) is None:
        raise RuntimeError("Modal deployment version is invalid")
    version = int(version_text[1:])
    if not history or history[0].get("version") != version_text:
        raise RuntimeError("reviewed deployment is not the newest App version")
    return version


def _validated_object_id(value: object, *, kind: str) -> str:
    pattern = OBJECT_ID_PATTERNS[kind]
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise RuntimeError(f"Modal {kind} object ID is invalid")
    return value


def _resource_creation_time(resource: Any) -> str:
    metadata = resource._get_metadata()
    creation_info = getattr(metadata, "creation_info", None)
    created_at = getattr(creation_info, "created_at", None)
    if not isinstance(created_at, (float, int)):
        raise RuntimeError("Modal resource creation time is unavailable")
    return _timestamp_microseconds(created_at)


def _function_binding(function: Any) -> tuple[str, str]:
    function_id = _validated_object_id(function.object_id, kind="function")
    metadata = function._get_metadata()
    function_name = getattr(metadata, "function_name", None)
    if function_name != MATCHED_FUNCTION_NAME:
        raise RuntimeError("Modal returned the wrong matched Function name")
    return function_id, function_name


def _deploy_control_plane(
    context: _ReviewedContext,
) -> MatchedDeploymentIdentity:
    modal_module = _load_modal()
    app_name = matched_app_name(context.provenance.control_plane_sha256)
    deployment_tag = matched_deployment_tag(context.provenance.control_plane_sha256)
    if _modal_history_or_empty(app_name):
        raise RuntimeError(
            "the implementation-addressed Modal App already exists; refusing redeployment"
        )

    registry = modal_module.Dict.from_name(
        MATCHED_ATTEMPT_REGISTRY_NAME,
        environment_name=MATCHED_ENVIRONMENT_NAME,
        create_if_missing=True,
    )
    registry.hydrate()
    evidence = modal_module.Volume.from_name(
        MATCHED_EVIDENCE_VOLUME_NAME,
        environment_name=MATCHED_ENVIRONMENT_NAME,
        create_if_missing=True,
        version=1,
    )
    evidence.hydrate()

    environment = os.environ.copy()
    environment["IQL_MATCHED_CONTROL_PLANE_SHA256"] = context.provenance.control_plane_sha256
    environment["IQL_MATCHED_CONTROL_PLANE_PROVENANCE_PATH"] = str(context.provenance_path)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "deploy",
            "-e",
            MATCHED_ENVIRONMENT_NAME,
            "--name",
            app_name,
            "--tag",
            deployment_tag,
            str(PROJECT_ROOT / RUNNER_RELATIVE_PATH),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        shell=False,
    )
    history = _modal_history(app_name)
    version = _deployment_version(history, deployment_tag=deployment_tag)
    function = modal_module.Function.from_name(
        app_name,
        MATCHED_FUNCTION_NAME,
        environment_name=MATCHED_ENVIRONMENT_NAME,
    )
    function.hydrate()
    function_id, function_name = _function_binding(function)
    history_after_hydration = _modal_history(app_name)
    if history_after_hydration != history:
        raise RuntimeError("Modal App changed while its deployment was being sealed")
    if (
        _deployment_version(
            history_after_hydration,
            deployment_tag=deployment_tag,
        )
        != version
    ):
        raise RuntimeError("Modal deployment version changed while sealing")

    return MatchedDeploymentIdentity(
        control_plane_sha256=context.provenance.control_plane_sha256,
        app_name=app_name,
        deployment_version=version,
        deployment_tag=deployment_tag,
        function_id=function_id,
        function_name=function_name,
        attempt_registry_id=_validated_object_id(
            registry.object_id,
            kind="attempt registry",
        ),
        attempt_registry_created_at_utc=_resource_creation_time(registry),
        evidence_volume_id=_validated_object_id(
            evidence.object_id,
            kind="evidence volume",
        ),
    )


def deploy(*, challenge_path: Path, confirmation: str) -> None:
    """Confirm and deploy the reviewed App without launching its Function."""

    challenge = _read_deploy_challenge(challenge_path)
    challenge.confirm(confirmation)
    context = _build_reviewed_context()
    _require_current_review(context, challenge.reviewed_inputs)
    _write_immutable_json(
        context.provenance_path,
        context.provenance.model_dump(mode="json"),
    )
    expected_path = _deploy_challenge_path(
        context.run_id,
        challenge.challenge_sha256(),
    ).resolve()
    if _resolve_local_control_path(challenge_path) != expected_path:
        raise RuntimeError("deploy challenge path is not content addressed")
    with _exclusive_remote_operation(context.run_id, "deploy"):
        acceptance = _deploy_confirmed(context, challenge)
    print(_canonical_json_bytes(acceptance).decode("utf-8"), end="")


def _deploy_confirmed(
    context: _ReviewedContext,
    challenge: MatchedDeployConfirmationChallenge,
) -> dict[str, Any]:
    if _deployment_path(context.run_id).exists():
        raise RuntimeError("matched control plane is already deployed")
    consumption_path = _deploy_consumption_path(
        context.run_id,
        challenge.challenge_sha256(),
    )
    if consumption_path.exists() or consumption_path.is_symlink():
        raise RuntimeError("deploy confirmation has already been consumed")
    _write_immutable_json(
        consumption_path,
        {
            "schema_version": "inkling-matched-deploy-consumption-v1",
            "status": "authorized_before_deploy",
            "consumed_at_utc": _utc_microseconds(),
            "run_id": context.run_id,
            "deploy_challenge_sha256": challenge.challenge_sha256(),
            "control_plane_sha256": context.provenance.control_plane_sha256,
        },
    )
    deployment = _deploy_control_plane(context)
    _write_immutable_json(
        _deployment_path(context.run_id),
        deployment.model_dump(mode="json"),
    )
    acceptance = {
        "schema_version": "inkling-matched-deploy-acceptance-v1",
        "status": "deployed_without_launch",
        "accepted_at_utc": _utc_microseconds(),
        "run_id": context.run_id,
        "deploy_challenge_sha256": challenge.challenge_sha256(),
        "deployment": deployment.model_dump(mode="json"),
        "paid_compute_started": False,
    }
    _write_immutable_json(
        _deploy_acceptance_path(context.run_id, challenge.challenge_sha256()),
        acceptance,
    )
    return acceptance


def prepare_launch(*, billing_cycle_end_utc: str) -> None:
    """Create one local launch challenge for an existing sealed deployment."""

    context = _build_reviewed_context()
    _require_launch_unused(context.run_id)
    deployment = _read_deployment(context.run_id)
    if deployment.control_plane_sha256 != context.provenance.control_plane_sha256:
        raise RuntimeError("local deployment differs from the current reviewed control")
    challenge = MatchedLaunchConfirmationChallenge(
        created_at_utc=_utc_microseconds(),
        authorization_nonce=secrets.token_hex(32),
        billing_cycle_end_utc=billing_cycle_end_utc,
        run_id=context.run_id,
        reviewed_inputs=context.reviewed_inputs,
        deployment=deployment,
    )
    path = _launch_challenge_path(
        context.run_id,
        challenge.challenge_sha256(),
    )
    _write_immutable_json(path, challenge.model_dump(mode="json"))
    print(
        _canonical_json_bytes(
            {
                "status": "prepared_before_launch",
                "run_id": context.run_id,
                "challenge_path": str(path),
                "challenge_sha256": challenge.challenge_sha256(),
                "confirmation": challenge.confirmation_text(),
                "warning": "This preparation did not contact Modal or start compute.",
            }
        ).decode("utf-8"),
        end="",
    )


def _validated_remote_deployment(
    deployment: MatchedDeploymentIdentity,
) -> tuple[Any, Any]:
    modal_module = _load_modal()
    history = _modal_history(deployment.app_name)
    if (
        _deployment_version(
            history,
            deployment_tag=deployment.deployment_tag,
        )
        != deployment.deployment_version
    ):
        raise RuntimeError("Modal deployment version differs from the local seal")
    function = modal_module.Function.from_name(
        deployment.app_name,
        MATCHED_FUNCTION_NAME,
        environment_name=MATCHED_ENVIRONMENT_NAME,
    )
    function.hydrate()
    function_id, function_name = _function_binding(function)
    if function_id != deployment.function_id or function_name != deployment.function_name:
        raise RuntimeError("Modal Function binding differs from the local seal")
    registry = modal_module.Dict.from_name(
        MATCHED_ATTEMPT_REGISTRY_NAME,
        environment_name=MATCHED_ENVIRONMENT_NAME,
        create_if_missing=False,
    )
    registry.hydrate()
    if (
        _validated_object_id(registry.object_id, kind="attempt registry")
        != deployment.attempt_registry_id
        or _resource_creation_time(registry) != deployment.attempt_registry_created_at_utc
    ):
        raise RuntimeError("Modal attempt registry differs from the local seal")
    evidence = modal_module.Volume.from_name(
        MATCHED_EVIDENCE_VOLUME_NAME,
        environment_name=MATCHED_ENVIRONMENT_NAME,
        create_if_missing=False,
        version=1,
    )
    evidence.hydrate()
    if (
        _validated_object_id(evidence.object_id, kind="evidence volume")
        != deployment.evidence_volume_id
    ):
        raise RuntimeError("Modal evidence Volume differs from the local seal")
    if _modal_history(deployment.app_name) != history:
        raise RuntimeError("Modal App changed during launch revalidation")
    return function, evidence


def _fresh_evidence_volume(deployment: MatchedDeploymentIdentity) -> Any:
    """Hydrate a new handle to the exact sealed evidence Volume."""

    modal_module = _load_modal()
    evidence = modal_module.Volume.from_name(
        MATCHED_EVIDENCE_VOLUME_NAME,
        environment_name=MATCHED_ENVIRONMENT_NAME,
        create_if_missing=False,
        version=1,
    )
    evidence.hydrate()
    if (
        _validated_object_id(evidence.object_id, kind="evidence volume")
        != deployment.evidence_volume_id
    ):
        raise RuntimeError("fresh Modal evidence Volume differs from the local seal")
    return evidence


def _fresh_attempt_registry(deployment: MatchedDeploymentIdentity) -> Any:
    """Hydrate a new handle to the exact sealed atomic attempt Dict."""

    modal_module = _load_modal()
    registry = modal_module.Dict.from_name(
        MATCHED_ATTEMPT_REGISTRY_NAME,
        environment_name=MATCHED_ENVIRONMENT_NAME,
        create_if_missing=False,
    )
    registry.hydrate()
    info = registry.info()
    if (
        _validated_object_id(registry.object_id, kind="attempt registry")
        != deployment.attempt_registry_id
        or getattr(info, "name", None) != deployment.attempt_registry_name
        or _resource_creation_time(registry) != deployment.attempt_registry_created_at_utc
    ):
        raise RuntimeError("fresh Modal attempt registry differs from the local seal")
    return registry


def _validate_attempt_claim_binding(
    claim: MatchedAttemptClaim,
    *,
    run_id: str,
    deployment: MatchedDeploymentIdentity,
    binding: _StatusLaunchBinding,
) -> None:
    """Require one attempt claim to describe the exact accepted launch."""

    acceptance = binding.acceptance
    expected_acceptance_path = matched_post_spawn_acceptance_path(
        run_id,
        acceptance.launch_intent_sha256,
    )
    if (
        claim.run_id != run_id
        or claim.registry_name != deployment.attempt_registry_name
        or claim.registry_id != deployment.attempt_registry_id
        or claim.registry_created_at_utc != deployment.attempt_registry_created_at_utc
        or claim.registry_key != matched_attempt_registry_key(run_id)
        or claim.call_id != acceptance.call_id
        or claim.launch_intent_sha256 != acceptance.launch_intent_sha256
        or claim.launch_intent_sha256 != binding.launch_intent.intent_sha256()
        or claim.post_spawn_acceptance_path != expected_acceptance_path
        or claim.post_spawn_acceptance_sha256 != acceptance.acceptance_sha256()
        or claim.matched_config_sha256 != acceptance.matched_config_sha256
        or claim.control_plane_sha256 != acceptance.control_plane_sha256
    ):
        raise ValueError("matched attempt claim differs from the exact accepted launch")


def _parse_bound_attempt_claim(
    payload: object,
    *,
    label: str,
    run_id: str,
    deployment: MatchedDeploymentIdentity,
    binding: _StatusLaunchBinding,
    evidence_path: str | None,
    expected_sha256: str | None,
) -> MatchedAttemptClaim:
    """Parse canonical claim bytes and validate their accepted-launch binding."""

    try:
        if not isinstance(payload, bytes):
            raise TypeError("matched attempt claim must be bytes")
        claim = MatchedAttemptClaim.model_validate(strict_matched_json_object(payload))
        claim_sha256 = claim.claim_sha256()
        if payload != claim.canonical_bytes():
            raise ValueError("matched attempt claim bytes are not canonical")
        if expected_sha256 is not None and claim_sha256 != expected_sha256:
            raise ValueError("matched attempt claim hash differs from its path")
        if evidence_path is not None:
            validate_matched_attempt_claim(
                payload,
                expected=claim,
                claim_sha256=claim_sha256,
                evidence_path=evidence_path,
            )
        _validate_attempt_claim_binding(
            claim,
            run_id=run_id,
            deployment=deployment,
            binding=binding,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} matched attempt claim is invalid") from error
    return claim


def _validated_attempt_claim_inspection(
    registry: Any,
    volume: Any,
    *,
    run_id: str,
    deployment: MatchedDeploymentIdentity,
    binding: _StatusLaunchBinding,
) -> _AttemptClaimInspection:
    """Reconcile the atomic Dict claim with its durable Volume copy."""

    registry_key = matched_attempt_registry_key(run_id)
    try:
        live_present = registry.contains(registry_key)
        if type(live_present) is not bool:
            raise TypeError("Modal Dict contains() did not return a bool")
        live_payload = registry.get(registry_key) if live_present else None
    except Exception as error:
        raise RuntimeError("live matched attempt registry is unreadable") from error

    if live_present:
        live_claim = _parse_bound_attempt_claim(
            live_payload,
            label="live",
            run_id=run_id,
            deployment=deployment,
            binding=binding,
            evidence_path=None,
            expected_sha256=None,
        )
    else:
        live_claim = None
        live_payload = None

    claim_root = f"runs/{run_id}/control/attempt-claims"
    entries = _remote_list_optional(volume, claim_root)
    if len(entries) > 1:
        raise RuntimeError("remote matched evidence has more than one durable attempt claim")

    durable_claim: MatchedAttemptClaim | None = None
    durable_payload: bytes | None = None
    durable_path: str | None = None
    if entries:
        entry = entries[0]
        path = getattr(entry, "path", None)
        type_name = getattr(getattr(entry, "type", None), "name", None)
        size = getattr(entry, "size", None)
        path_pattern = re.compile(rf"^{re.escape(claim_root)}/([0-9a-f]{{64}})\.json$")
        match = path_pattern.fullmatch(path) if isinstance(path, str) else None
        if (
            match is None
            or path != matched_attempt_claim_path(run_id, match.group(1))
            or type_name != "FILE"
            or type(size) is not int
            or not 0 < size <= MAX_REMOTE_TERMINAL_RECEIPT_BYTES
        ):
            raise RuntimeError("remote matched attempt-claim directory contains an unknown entry")
        durable_path = path
        durable_payload = _remote_read_bounded_optional(
            volume,
            durable_path,
            maximum_bytes=MAX_REMOTE_TERMINAL_RECEIPT_BYTES,
        )
        if durable_payload is None:
            raise RuntimeError("listed durable matched attempt claim disappeared before validation")
        durable_claim = _parse_bound_attempt_claim(
            durable_payload,
            label="durable",
            run_id=run_id,
            deployment=deployment,
            binding=binding,
            evidence_path=durable_path,
            expected_sha256=match.group(1),
        )

    if live_payload is not None and durable_payload is not None and live_payload != durable_payload:
        raise RuntimeError("live matched attempt claim differs from its durable Volume claim")
    return _AttemptClaimInspection(
        registry_key=registry_key,
        live_claim=live_claim,
        live_payload=live_payload,
        durable_claim=durable_claim,
        durable_payload=durable_payload,
        durable_path=durable_path,
    )


def _require_launch_unused(run_id: str) -> None:
    control_root = _run_root(run_id) / "control"
    consumption = _launch_consumption_path(run_id)
    if consumption.exists() or consumption.is_symlink():
        raise RuntimeError("the one configured matched launch has already been consumed")
    intents = control_root / "launch-intents"
    if intents.is_symlink():
        raise RuntimeError("matched launch control directory is a symbolic link")
    if intents.is_dir() and any(intents.iterdir()):
        raise RuntimeError("the one configured matched launch has already been consumed")


def _remote_path(relative: str) -> str:
    path = PurePosixPath(relative)
    if not relative or path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
        raise ValueError("remote matched evidence path is invalid")
    return f"/{relative}"


def _remote_read_optional(volume: Any, relative: str) -> bytes | None:
    try:
        return b"".join(volume.read_file(_remote_path(relative)))
    except FileNotFoundError:
        return None


def _remote_read_bounded_optional(
    volume: Any,
    relative: str,
    *,
    maximum_bytes: int,
) -> bytes | None:
    """Read one bounded remote record without accepting non-byte chunks."""

    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("remote matched evidence size limit is invalid")
    modal_module = _load_modal()
    payload = bytearray()
    try:
        for chunk in volume.read_file(_remote_path(relative)):
            if not isinstance(chunk, bytes):
                raise RuntimeError("Modal returned non-byte matched evidence")
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                raise RuntimeError("remote matched evidence exceeds its size limit")
    except (FileNotFoundError, modal_module.exception.NotFoundError):
        return None
    return bytes(payload)


def _remote_list_optional(volume: Any, relative: str) -> list[Any]:
    """List one remote directory, treating only provider not-found as empty."""

    modal_module = _load_modal()
    _remote_path(relative)
    try:
        return list(volume.listdir(relative, recursive=False))
    except (FileNotFoundError, modal_module.exception.NotFoundError):
        return []


def _remote_write_immutable(volume: Any, relative: str, payload: bytes) -> None:
    if _remote_read_optional(volume, relative) is not None:
        raise RuntimeError(f"immutable remote evidence already exists: {relative}")
    with volume.batch_upload(force=False) as upload:
        upload.put_file(io.BytesIO(payload), _remote_path(relative), mode=0o400)
    observed = _remote_read_optional(volume, relative)
    if observed != payload:
        raise RuntimeError(f"immutable remote evidence readback failed: {relative}")


def _publish_remote_launch_intent(
    evidence_volume: Any,
    launch_intent: MatchedLaunchIntent,
) -> None:
    relative = matched_launch_intent_path(
        launch_intent.run_id,
        launch_intent.intent_sha256(),
    )
    _remote_write_immutable(
        evidence_volume,
        relative,
        launch_intent.canonical_bytes(),
    )


def _publish_post_spawn_acceptance(
    evidence_volume: Any,
    acceptance: MatchedPostSpawnAcceptance,
) -> None:
    relative = matched_post_spawn_acceptance_path(
        acceptance.run_id,
        acceptance.launch_intent_sha256,
    )
    payload = acceptance.canonical_bytes()
    upload_error: Exception | None = None
    try:
        _remote_write_immutable(
            evidence_volume,
            relative,
            payload,
        )
    except Exception as error:
        upload_error = error

    try:
        fresh_evidence = _fresh_evidence_volume(acceptance.deployment)
        observed = _remote_read_optional(fresh_evidence, relative)
    except Exception as error:
        raise _PostSpawnAcceptanceStateUnknownError(
            "post-spawn acceptance state is unknown after fresh Volume hydration"
        ) from error

    if observed != payload:
        mismatch = _PostSpawnAcceptanceMismatchError(
            "fresh Volume read proved that exact post-spawn acceptance is absent or different"
        )
        if upload_error is not None:
            raise mismatch from upload_error
        raise mismatch


def _validated_call_id(call: Any) -> str:
    object_id = getattr(call, "object_id", None)
    if not isinstance(object_id, str) or CALL_ID_PATTERN.fullmatch(object_id) is None:
        raise RuntimeError("Modal call ID object is invalid")
    return object_id


def _launch_once(
    function: Any,
    evidence_volume: Any,
    launch_intent: MatchedLaunchIntent,
) -> tuple[str, MatchedPostSpawnAcceptance]:
    """Upload authorization, spawn once, then publish the accepted call ID."""

    _publish_remote_launch_intent(evidence_volume, launch_intent)
    call = function.spawn(
        launch_intent.run_id,
        launch_intent.intent_sha256(),
    )
    try:
        call_id = _validated_call_id(call)
        acceptance = MatchedPostSpawnAcceptance(
            accepted_at_utc=_utc_microseconds(),
            run_id=launch_intent.run_id,
            launch_intent_sha256=launch_intent.intent_sha256(),
            call_id=call_id,
            deployment=launch_intent.deployment,
            matched_config_sha256=launch_intent.reviewed_inputs.matched_config_sha256,
            control_plane_sha256=launch_intent.reviewed_inputs.control_plane_sha256,
        )
    except Exception as error:
        try:
            call.cancel(terminate_containers=True)
        except Exception as cancellation_error:
            raise RuntimeError(
                "post-spawn validation failed and call cancellation also failed"
            ) from cancellation_error
        raise RuntimeError(
            "post-spawn validation failed; call cancellation was requested"
        ) from error
    try:
        _publish_post_spawn_acceptance(evidence_volume, acceptance)
    except _PostSpawnAcceptanceMismatchError as error:
        try:
            call.cancel(terminate_containers=True)
        except Exception as cancellation_error:
            raise RuntimeError(
                "post-spawn acceptance mismatch and call cancellation also failed"
            ) from cancellation_error
        raise RuntimeError(
            "post-spawn acceptance mismatch; call cancellation was requested"
        ) from error
    except _PostSpawnAcceptanceStateUnknownError as error:
        raise RuntimeError(
            "post-spawn acceptance state is unknown; the call was not cancelled "
            "and this launch remains consumed"
        ) from error
    return call_id, acceptance


def launch(*, challenge_path: Path, confirmation: str) -> None:
    """Confirm one challenge and start one matched smoke Function call."""

    challenge = _read_launch_challenge(challenge_path)
    challenge.confirm(confirmation)
    context = _build_reviewed_context()
    _require_current_review(context, challenge.reviewed_inputs)
    if challenge.run_id != context.run_id:
        raise RuntimeError("launch challenge run ID differs from current reviewed control")
    deployment = _read_deployment(context.run_id)
    if challenge.deployment != deployment:
        raise RuntimeError("launch challenge deployment differs from the local seal")
    expected_path = _launch_challenge_path(
        context.run_id,
        challenge.challenge_sha256(),
    ).resolve()
    if _resolve_local_control_path(challenge_path) != expected_path:
        raise RuntimeError("launch challenge path is not content addressed")

    with _exclusive_remote_operation(context.run_id, "launch"):
        receipt = _launch_confirmed(
            context,
            challenge,
            confirmation=confirmation,
            deployment=deployment,
        )
    print(_canonical_json_bytes(receipt).decode("utf-8"), end="")


def _launch_confirmed(
    context: _ReviewedContext,
    challenge: MatchedLaunchConfirmationChallenge,
    *,
    confirmation: str,
    deployment: MatchedDeploymentIdentity,
) -> dict[str, Any]:
    _require_launch_unused(context.run_id)
    function, evidence = _validated_remote_deployment(deployment)
    authorized_at_utc = _utc_microseconds()
    launch_intent = build_matched_launch_intent(
        challenge,
        confirmation=confirmation,
        authorized_at_utc=authorized_at_utc,
    )
    launch_intent_sha256 = launch_intent.intent_sha256()
    local_intent_path = _local_launch_intent_path(
        context.run_id,
        launch_intent_sha256,
    )
    if local_intent_path.exists():
        raise RuntimeError("launch confirmation has already been consumed")
    _write_immutable_json(
        _launch_consumption_path(context.run_id),
        {
            "schema_version": "inkling-matched-launch-consumption-v1",
            "status": "authorized_before_spawn",
            "consumed_at_utc": authorized_at_utc,
            "run_id": context.run_id,
            "launch_challenge_sha256": challenge.challenge_sha256(),
            "launch_intent_sha256": launch_intent_sha256,
            "control_plane_sha256": context.provenance.control_plane_sha256,
        },
    )
    _write_immutable_json(
        local_intent_path,
        launch_intent.model_dump(mode="json"),
    )
    call_id, acceptance = _launch_once(function, evidence, launch_intent)
    _write_immutable_json(
        _local_post_spawn_acceptance_path(
            context.run_id,
            launch_intent_sha256,
        ),
        acceptance.model_dump(mode="json"),
    )
    receipt = {
        "schema_version": "inkling-matched-call-receipt-v1",
        "status": "accepted_after_spawn",
        "run_id": context.run_id,
        "call_id": call_id,
        "launch_intent_sha256": launch_intent_sha256,
        "post_spawn_acceptance_sha256": acceptance.acceptance_sha256(),
        "deployment": deployment.model_dump(mode="json"),
        "function_return_is_success_evidence": False,
    }
    _write_immutable_json(
        _call_receipt_path(context.run_id, launch_intent_sha256),
        receipt,
    )
    return receipt


def _read_status_launch_binding(
    context: _ReviewedContext,
    deployment: MatchedDeploymentIdentity,
    call_id: str,
) -> _StatusLaunchBinding:
    """Bind a status query to the one exact locally accepted launch."""

    calls_root = _run_root(context.run_id) / "calls"
    if calls_root.is_symlink() or not calls_root.is_dir():
        raise RuntimeError("matched call receipt directory is missing")
    entries = tuple(sorted(calls_root.iterdir()))
    if len(entries) != 1:
        raise RuntimeError("expected exactly one local matched call receipt")
    call_path = entries[0]
    if call_path.is_symlink() or not call_path.is_file():
        raise RuntimeError("matched call receipt is not one regular file")
    if re.fullmatch(r"[0-9a-f]{64}\.json", call_path.name) is None:
        raise RuntimeError("matched call receipt path is not launch addressed")

    call_payload = call_path.read_bytes()
    call_record = strict_matched_json_object(call_payload)
    if call_payload != _canonical_json_bytes(call_record):
        raise RuntimeError("matched call receipt bytes are not canonical")
    required_call_keys = {
        "schema_version",
        "status",
        "run_id",
        "call_id",
        "launch_intent_sha256",
        "post_spawn_acceptance_sha256",
        "deployment",
        "function_return_is_success_evidence",
    }
    if set(call_record) != required_call_keys:
        raise RuntimeError("matched call receipt has an unexpected schema")
    launch_intent_sha256 = call_record.get("launch_intent_sha256")
    acceptance_sha256 = call_record.get("post_spawn_acceptance_sha256")
    if (
        call_record.get("schema_version") != "inkling-matched-call-receipt-v1"
        or call_record.get("status") != "accepted_after_spawn"
        or call_record.get("run_id") != context.run_id
        or call_record.get("call_id") != call_id
        or not isinstance(launch_intent_sha256, str)
        or SHA256_PATTERN.fullmatch(launch_intent_sha256) is None
        or call_path.name != f"{launch_intent_sha256}.json"
        or not isinstance(acceptance_sha256, str)
        or SHA256_PATTERN.fullmatch(acceptance_sha256) is None
        or call_record.get("deployment") != deployment.model_dump(mode="json")
        or call_record.get("function_return_is_success_evidence") is not False
    ):
        raise RuntimeError("matched call receipt differs from the requested accepted call")

    intent_path = _local_launch_intent_path(context.run_id, launch_intent_sha256)
    if intent_path.is_symlink() or not intent_path.is_file():
        raise RuntimeError("local matched launch intent is missing")
    intent_payload = intent_path.read_bytes()
    launch_intent = MatchedLaunchIntent.model_validate(strict_matched_json_object(intent_payload))
    if (
        intent_payload != launch_intent.canonical_bytes()
        or launch_intent.intent_sha256() != launch_intent_sha256
        or launch_intent.run_id != context.run_id
        or launch_intent.reviewed_inputs != context.reviewed_inputs
        or launch_intent.deployment != deployment
    ):
        raise RuntimeError("local matched launch intent differs from current reviewed control")

    acceptance_path = _local_post_spawn_acceptance_path(
        context.run_id,
        launch_intent_sha256,
    )
    if acceptance_path.is_symlink() or not acceptance_path.is_file():
        raise RuntimeError("local matched post-spawn acceptance is missing")
    acceptance_payload = acceptance_path.read_bytes()
    acceptance = MatchedPostSpawnAcceptance.model_validate(
        strict_matched_json_object(acceptance_payload)
    )
    if (
        acceptance_payload != acceptance.canonical_bytes()
        or acceptance.acceptance_sha256() != acceptance_sha256
        or acceptance.run_id != context.run_id
        or acceptance.launch_intent_sha256 != launch_intent_sha256
        or acceptance.call_id != call_id
        or acceptance.deployment != deployment
        or acceptance.matched_config_sha256 != context.reviewed_inputs.matched_config_sha256
        or acceptance.control_plane_sha256 != context.reviewed_inputs.control_plane_sha256
    ):
        raise RuntimeError("local matched post-spawn acceptance differs from the requested call")
    return _StatusLaunchBinding(
        launch_intent=launch_intent,
        acceptance=acceptance,
    )


def _validate_remote_status_acceptance(
    volume: Any,
    binding: _StatusLaunchBinding,
) -> None:
    """Require the fresh Volume view to contain the exact accepted call."""

    acceptance = binding.acceptance
    relative = matched_post_spawn_acceptance_path(
        acceptance.run_id,
        acceptance.launch_intent_sha256,
    )
    payload = _remote_read_bounded_optional(
        volume,
        relative,
        maximum_bytes=MAX_REMOTE_TERMINAL_RECEIPT_BYTES,
    )
    if payload is None:
        raise RuntimeError("fresh Volume read found no matched post-spawn acceptance")
    validate_matched_post_spawn_acceptance(
        payload,
        expected=acceptance,
        acceptance_sha256=acceptance.acceptance_sha256(),
        evidence_path=relative,
    )


def _validated_publication_inspection(
    volume: Any,
    *,
    run_id: str,
) -> _PublicationInspection:
    """Validate the bounded monotonic publication history for one terminal."""

    publication_root = f"runs/{run_id}/control/publication-states"
    entries = _remote_list_optional(volume, publication_root)
    if len(entries) > 4:
        raise RuntimeError("remote matched publication history exceeds its bound")
    path_pattern = re.compile(rf"^{re.escape(publication_root)}/([0-9a-f]{{64}})\.json$")
    snapshots: list[MatchedPublicationSnapshot] = []
    seen_states: set[tuple[str, int]] = set()
    for entry in entries:
        path = getattr(entry, "path", None)
        type_name = getattr(getattr(entry, "type", None), "name", None)
        size = getattr(entry, "size", None)
        if not isinstance(path, str):
            raise RuntimeError("remote matched publication directory contains an unknown entry")
        match = path_pattern.fullmatch(path)
        if (
            match is None
            or type_name != "FILE"
            or type(size) is not int
            or not 0 < size <= MAX_REMOTE_TERMINAL_RECEIPT_BYTES
        ):
            raise RuntimeError("remote matched publication directory contains an unknown entry")
        payload = _remote_read_bounded_optional(
            volume,
            path,
            maximum_bytes=MAX_REMOTE_TERMINAL_RECEIPT_BYTES,
        )
        if payload is None:
            raise RuntimeError("listed matched publication snapshot disappeared before validation")
        try:
            snapshot = MatchedPublicationSnapshot.model_validate(
                strict_matched_json_object(payload)
            )
        except ValueError as error:
            raise RuntimeError("matched publication snapshot semantics are invalid") from error
        state_sha256 = match.group(1)
        if path != matched_publication_state_path(run_id, state_sha256):
            raise RuntimeError("matched publication snapshot path is not content addressed")
        validate_matched_publication_state(
            payload,
            expected=snapshot,
            state_sha256=state_sha256,
            evidence_path=path,
        )
        if snapshot.run_id != run_id:
            raise RuntimeError("matched publication snapshot has the wrong run ID")
        state_key = (snapshot.status, snapshot.cycle)
        if state_key in seen_states:
            raise RuntimeError("matched publication history contains a duplicate state")
        seen_states.add(state_key)
        snapshots.append(snapshot)

    status_rank = {
        "not_started": 0,
        "installing": 1,
        "confirmed": 2,
        "unknown": 2,
    }
    ordered = tuple(
        sorted(
            snapshots,
            key=lambda snapshot: (
                snapshot.cycle,
                status_rank[snapshot.status],
            ),
        )
    )
    if ordered:
        if ordered[0].status != "not_started" or ordered[0].cycle != 0:
            raise RuntimeError("matched publication history has no initial state")
        for previous, current in pairwise(ordered):
            try:
                validate_matched_publication_transition(previous, current)
            except ValueError as error:
                raise RuntimeError(
                    "matched publication history has an invalid transition"
                ) from error
    return _PublicationInspection(snapshots=ordered)


def _validated_terminal_evidence(
    volume: Any,
    *,
    run_id: str,
    durable_attempt_claim: MatchedAttemptClaim | None,
) -> _ValidatedTerminalEvidence | None:
    """Read at most one exact terminal receipt from a fresh Volume handle."""

    publication = _validated_publication_inspection(volume, run_id=run_id)
    candidates: list[tuple[str, str]] = []
    terminal_root = f"runs/{run_id}/terminal"
    for outcome in ("success", "failure"):
        outcome_root = f"{terminal_root}/{outcome}"
        entries = _remote_list_optional(volume, outcome_root)
        expected_path = re.compile(rf"^{re.escape(outcome_root)}/([0-9a-f]{{64}})\.json$")
        for entry in entries:
            path = getattr(entry, "path", None)
            type_name = getattr(getattr(entry, "type", None), "name", None)
            size = getattr(entry, "size", None)
            if (
                not isinstance(path, str)
                or expected_path.fullmatch(path) is None
                or type_name != "FILE"
                or type(size) is not int
                or not 0 < size <= MAX_REMOTE_TERMINAL_RECEIPT_BYTES
            ):
                raise RuntimeError("remote matched terminal directory contains an unknown entry")
            candidates.append((outcome, path))
    if not candidates and publication.final is not None:
        raise RuntimeError("matched publication state exists without its terminal receipt")
    if not candidates:
        return None
    if len(candidates) != 1:
        raise RuntimeError("remote matched evidence has conflicting terminal receipts")
    if publication.final is None:
        raise RuntimeError("matched terminal receipt has no publication history")
    if publication.final.status == "unknown":
        raise RuntimeError("matched terminal publication state is unknown")
    if publication.final.status != "confirmed":
        raise RuntimeError("matched terminal publication is not yet confirmed")
    if durable_attempt_claim is None:
        raise RuntimeError("matched terminal publication has no durable attempt claim")
    if publication.final.attempt_claim_sha256 != durable_attempt_claim.claim_sha256():
        raise RuntimeError("matched terminal publication differs from its durable attempt claim")
    expected_reference = publication.final.terminal_receipt
    if expected_reference is None:
        raise RuntimeError("confirmed matched publication has no terminal reference")

    outcome, path = candidates[0]
    if expected_reference.outcome != outcome or expected_reference.path != path:
        raise RuntimeError(
            "matched terminal receipt differs from its confirmed publication reference"
        )
    payload = _remote_read_bounded_optional(
        volume,
        path,
        maximum_bytes=MAX_REMOTE_TERMINAL_RECEIPT_BYTES,
    )
    if payload is None:
        raise RuntimeError("listed matched terminal receipt disappeared before validation")
    raw = strict_matched_json_object(payload)
    try:
        if outcome == "success":
            receipt: MatchedRollupReceipt | MatchedFailureReceipt = (
                MatchedRollupReceipt.model_validate(raw)
            )
        else:
            receipt = MatchedFailureReceipt.model_validate(raw)
    except ValueError as error:
        raise RuntimeError("matched terminal receipt semantics are invalid") from error
    if receipt.run_id != run_id:
        raise RuntimeError("matched terminal receipt has the wrong run ID")

    reference = build_matched_terminal_receipt_reference(
        payload,
        run_id=run_id,
        outcome=outcome,
    )
    if reference.path != path:
        raise RuntimeError("matched terminal receipt path differs from its exact content")
    if reference != expected_reference:
        raise RuntimeError("matched terminal receipt content differs from its confirmed reference")
    validate_matched_terminal_receipt_reference(
        payload,
        expected=reference,
    )
    return _ValidatedTerminalEvidence(
        reference=reference,
        receipt=receipt,
    )


def _provider_call_state(call_id: str) -> tuple[str, int]:
    """Inspect provider metadata without downloading the Function return value."""

    modal_module = _load_modal()
    call = modal_module.FunctionCall.from_id(call_id)
    roots = call.get_call_graph()
    if not isinstance(roots, list):
        raise RuntimeError("Modal call graph has an unexpected shape")
    pending = list(roots)
    states: list[str] = []
    while pending:
        if len(states) >= MAX_PROVIDER_CALL_GRAPH_NODES:
            raise RuntimeError("Modal call graph exceeds its size limit")
        node = pending.pop()
        status = getattr(node, "status", None)
        status_name = getattr(status, "name", None)
        children = getattr(node, "children", None)
        if (
            not isinstance(status_name, str)
            or re.fullmatch(r"[A-Z][A-Z_]*", status_name) is None
            or not isinstance(children, list)
        ):
            raise RuntimeError("Modal call graph contains invalid provider metadata")
        states.append(status_name.casefold())
        pending.extend(children)

    failed_states = {"failure", "init_failure", "terminated", "timeout"}
    if set(states) & failed_states:
        return "failed_without_terminal_evidence", 1
    if states and set(states) == {"success"}:
        return "returned_without_terminal_evidence", 1
    return "running_or_queued", 0


def _terminal_status_payload(
    call_id: str,
    terminal: _ValidatedTerminalEvidence,
    attempt: _AttemptClaimInspection,
) -> dict[str, Any]:
    """Expose only bounded fields from a fully validated terminal receipt."""

    receipt = terminal.receipt
    common: dict[str, Any] = {
        "call_id": call_id,
        "run_id": receipt.run_id,
        "terminal_receipt": terminal.reference.model_dump(mode="json"),
        "attempt_claim_sha256": attempt.claim_sha256,
        "attempt_consumed": True,
        "retry_allowed": False,
        "function_return_is_success_evidence": False,
        "quality_measured": False,
        "benchmark_measured": False,
    }
    if isinstance(receipt, MatchedRollupReceipt):
        return {
            "status": "passed",
            "evidence_status": "validated_terminal_success",
            "both_subjects_passed": receipt.both_subjects_passed,
            **common,
        }
    return {
        "status": "failed",
        "evidence_status": "validated_terminal_failure",
        "subject_at_failure": receipt.subject_at_failure.value,
        "failure_category": receipt.diagnostic.category,
        "failure_type": receipt.diagnostic.failure_type,
        "failure_message_sha256": receipt.diagnostic.message_sha256,
        **common,
    }


def status(*, call_id: str) -> int:
    """Validate terminal evidence, then inspect provider metadata if none exists."""

    if CALL_ID_PATTERN.fullmatch(call_id) is None:
        raise ValueError("Modal call ID is invalid")
    context = _build_reviewed_context()
    deployment = _read_deployment(context.run_id)
    if deployment.control_plane_sha256 != context.provenance.control_plane_sha256:
        raise RuntimeError("local deployment differs from current reviewed control")
    binding = _read_status_launch_binding(context, deployment, call_id)
    volume = _fresh_evidence_volume(deployment)
    registry = _fresh_attempt_registry(deployment)
    _validate_remote_status_acceptance(volume, binding)
    attempt = _validated_attempt_claim_inspection(
        registry,
        volume,
        run_id=context.run_id,
        deployment=deployment,
        binding=binding,
    )
    terminal = _validated_terminal_evidence(
        volume,
        run_id=context.run_id,
        durable_attempt_claim=attempt.durable_claim,
    )
    if terminal is not None:
        print(
            _canonical_json_bytes(_terminal_status_payload(call_id, terminal, attempt)).decode(
                "utf-8"
            ),
            end="",
        )
        return 0 if isinstance(terminal.receipt, MatchedRollupReceipt) else 1

    if attempt.attempt_consumed_before_volume_bookkeeping:
        print(
            _canonical_json_bytes(
                {
                    "status": "attempt_consumed",
                    "evidence_status": ("attempt_consumed_before_volume_bookkeeping"),
                    "call_id": call_id,
                    "run_id": context.run_id,
                    "attempt_registry_key": attempt.registry_key,
                    "attempt_registry_key_present": True,
                    "attempt_claim_sha256": attempt.claim_sha256,
                    "attempt_consumed": True,
                    "retry_allowed": False,
                    "volume_attempt_claim_path": None,
                    "function_return_is_success_evidence": False,
                }
            ).decode("utf-8"),
            end="",
        )
        return 1

    provider_state, exit_code = _provider_call_state(call_id)
    print(
        _canonical_json_bytes(
            {
                "status": provider_state,
                "call_id": call_id,
                "run_id": context.run_id,
                "evidence_status": "no_terminal_receipt",
                "attempt_claim_sha256": attempt.claim_sha256,
                "attempt_consumed": attempt.claim_sha256 is not None,
                "retry_allowed": False,
                "function_return_is_success_evidence": False,
                "warning": (
                    "Function return is not evidence; only a validated immutable "
                    "terminal receipt is."
                ),
            }
        ).decode("utf-8"),
        end="",
    )
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect_parser = commands.add_parser(
        "inspect",
        help="Inspect checked local controls without contacting Modal.",
    )
    inspect_parser.add_argument("--json", action="store_true")

    commands.add_parser(
        "prepare-deploy",
        help="Write the exact local deployment challenge.",
    )

    deploy_parser = commands.add_parser(
        "deploy",
        help="Deploy the reviewed App without launching compute.",
    )
    deploy_parser.add_argument("--challenge", type=Path, required=True)
    deploy_parser.add_argument("--confirm", required=True)

    prepare_launch_parser = commands.add_parser(
        "prepare-launch",
        help="Write the exact local launch challenge.",
    )
    prepare_launch_parser.add_argument(
        "--billing-cycle-end-utc",
        required=True,
        help="Exact current Modal cycle end in YYYY-MM-DDTHH:MM:SSZ form.",
    )

    launch_parser = commands.add_parser(
        "launch",
        help="Launch exactly one confirmed matched smoke call.",
    )
    launch_parser.add_argument("--challenge", type=Path, required=True)
    launch_parser.add_argument("--confirm", required=True)

    status_parser = commands.add_parser(
        "status",
        help="Poll one call without changing remote state.",
    )
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
            deploy(
                challenge_path=arguments.challenge,
                confirmation=arguments.confirm,
            )
        elif arguments.command == "prepare-launch":
            prepare_launch(
                billing_cycle_end_utc=arguments.billing_cycle_end_utc,
            )
        elif arguments.command == "launch":
            launch(
                challenge_path=arguments.challenge,
                confirmation=arguments.confirm,
            )
        elif arguments.command == "status":
            return status(call_id=arguments.call_id)
        else:
            raise RuntimeError("unsupported matched manager command")
    except (
        FileExistsError,
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
