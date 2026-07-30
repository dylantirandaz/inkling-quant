"""Manage one reviewed BF16-versus-Q3 Inkling measurement on Modal.

Local preparation never contacts Modal. Deployment and GPU launch use separate,
short-lived, content-addressed confirmations. Deployment does not call the
remote function. Only a validated immutable terminal receipt is a result.
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
from types import ModuleType
from typing import Any, Final, Literal, cast

if __name__ == "__main__":
    sys.dont_write_bytecode = True

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
SRC_ROOT: Final = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inkling_quant_lab.gguf.inkling_matched_execution import (  # noqa: E402
    ExactCudaPlacementPolicy,
    build_matched_cuda_placement_policy,
)
from inkling_quant_lab.gguf.inkling_measurement import (  # noqa: E402
    CORPUS_MATERIALIZER_RELATIVE_PATH,
    MEASUREMENT_CONFIG_RELATIVE_PATH,
    InklingMeasurementBundle,
    load_measurement_bundle,
    measurement_protocol_sha256,
    measurement_workload_sha256,
)
from inkling_quant_lab.gguf.inkling_measurement_control import (  # noqa: E402
    MEASUREMENT_ATTEMPT_REGISTRY_NAME,
    MEASUREMENT_CONTROL_RECORD_MAX_BYTES,
    MEASUREMENT_DEPLOY_CHALLENGE_MAX_AGE_SECONDS,
    MEASUREMENT_ENVIRONMENT_NAME,
    MEASUREMENT_EVIDENCE_VOLUME_NAME,
    MEASUREMENT_FUNCTION_NAME,
    MEASUREMENT_LAUNCH_CHALLENGE_MAX_AGE_SECONDS,
    MEASUREMENT_STAGE,
    MeasurementAttemptClaim,
    MeasurementControlPlaneFile,
    MeasurementControlPlaneProvenance,
    MeasurementDeployConfirmationChallenge,
    MeasurementDeploymentIdentity,
    MeasurementExecutionResources,
    MeasurementLaunchConfirmationChallenge,
    MeasurementLaunchIntent,
    MeasurementPostSpawnAcceptance,
    MeasurementReviewedInputs,
    MeasurementSuccessTerminalReceipt,
    MeasurementSupportingRecordReference,
    MeasurementTerminalReceipt,
    MeasurementTerminalReceiptReference,
    build_measurement_control_plane_provenance,
    build_measurement_launch_intent,
    build_measurement_post_spawn_acceptance,
    build_measurement_terminal_receipt_reference,
    canonical_measurement_json_bytes,
    measurement_app_name,
    measurement_attempt_claim_path,
    measurement_attempt_registry_key,
    measurement_deployment_tag,
    measurement_launch_intent_path,
    measurement_performance_rollup_sha256,
    measurement_post_spawn_acceptance_path,
    measurement_quality_rollup_sha256,
    parse_measurement_terminal_receipt,
    strict_measurement_json_object,
    validate_measurement_attempt_claim,
    validate_measurement_deploy_challenge_not_expired,
    validate_measurement_launch_intent,
    validate_measurement_post_spawn_acceptance,
    validate_measurement_supporting_record_reference,
    validate_repository_relative_path,
)
from inkling_quant_lab.gguf.inkling_measurement_evidence import (  # noqa: E402
    MEASUREMENT_RAW_BLOB_MAX_BYTES,
    MeasurementComparisonCompactRecord,
    MeasurementEvidenceSubject,
    MeasurementRawBlobReference,
    MeasurementSubjectCompactRecord,
    build_measurement_performance_rollup,
    build_measurement_placement_summaries,
    build_measurement_quality_rollup,
    measurement_subject_performance_projection_sha256,
    measurement_subject_quality_projection_sha256,
    parse_measurement_comparison_compact_record,
    parse_measurement_subject_compact_record,
    validate_measurement_comparison_links,
    validate_measurement_raw_blob_reference,
)
from inkling_quant_lab.gguf.inkling_measurement_raw_evidence import (  # noqa: E402
    MeasurementAttemptBindings,
    MeasurementBackendAuditEvidence,
    MeasurementPairingProjectionHashes,
    MeasurementRawEvidenceLinks,
    MeasurementRawTrialsEvidence,
    MeasurementResourceTelemetryEvidence,
    MeasurementSubjectPerformanceSummary,
    MeasurementSubjectQualitySummary,
    MeasurementTokenNllEvidence,
    parse_backend_audit_evidence,
    parse_raw_trials_evidence,
    parse_resource_telemetry_evidence,
    parse_token_nll_raw_evidence,
    recompute_pairing_projection_hashes,
    recompute_subject_performance_summary,
    recompute_subject_quality_summary,
    validate_measurement_diagnostic_evidence,
    validate_measurement_raw_evidence_links,
    validate_pairing_projection_hashes,
)

EXPECTED_MODAL_VERSION: Final = "1.5.0"
RUNNER_RELATIVE_PATH: Final = "scripts/run_inkling_measurement_modal.py"
MANAGER_RELATIVE_PATH: Final = "scripts/manage_inkling_measurement_modal.py"
ARTIFACT_ROOT: Final = PROJECT_ROOT / "artifacts" / "inkling-measurement-modal"
EVIDENCE_MOUNT_ROOT: Final = "/evidence"
CALL_ID_PATTERN: Final = re.compile(r"^fc-[A-Za-z0-9]+$")
RUN_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
OBJECT_ID_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "function": re.compile(r"^fu-[A-Za-z0-9]+$"),
    "attempt registry": re.compile(r"^di-[A-Za-z0-9]+$"),
    "evidence volume": re.compile(r"^vo-[A-Za-z0-9]+$"),
}
MAX_CALL_GRAPH_NODES: Final = 256


@dataclass(frozen=True)
class _ReviewedContext:
    bundle: InklingMeasurementBundle
    provenance: MeasurementControlPlaneProvenance
    reviewed_inputs: MeasurementReviewedInputs
    run_id: str
    provenance_path: Path


@dataclass(frozen=True)
class _LaunchBinding:
    deployment: MeasurementDeploymentIdentity
    intent: MeasurementLaunchIntent
    acceptance: MeasurementPostSpawnAcceptance


@dataclass(frozen=True)
class _AttemptInspection:
    claim: MeasurementAttemptClaim | None
    durable: bool


@dataclass(frozen=True)
class _ValidatedTerminalEvidence:
    reference: MeasurementTerminalReceiptReference
    receipt: MeasurementTerminalReceipt


@dataclass(frozen=True)
class _TerminalSupportingEvidence:
    subjects: tuple[MeasurementSubjectCompactRecord, ...]
    comparison: MeasurementComparisonCompactRecord | None
    raw_blobs: tuple[tuple[MeasurementRawBlobReference, bytes], ...]
    raw_subjects: tuple[_ValidatedSubjectRawEvidence, ...]


@dataclass(frozen=True)
class _ValidatedSubjectRawEvidence:
    subject: MeasurementEvidenceSubject
    token_nll: MeasurementTokenNllEvidence
    raw_trials: MeasurementRawTrialsEvidence
    telemetry: MeasurementResourceTelemetryEvidence
    backend_audit: MeasurementBackendAuditEvidence
    links: MeasurementRawEvidenceLinks
    quality: MeasurementSubjectQualitySummary
    performance: MeasurementSubjectPerformanceSummary
    pairing: MeasurementPairingProjectionHashes


def _utc_microseconds(value: datetime | None = None) -> str:
    instant = datetime.now(UTC) if value is None else value.astimezone(UTC)
    return instant.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _timestamp_microseconds(timestamp: float) -> str:
    if not isinstance(timestamp, (int, float)) or timestamp <= 0:
        raise RuntimeError("Modal resource creation time is invalid")
    return _utc_microseconds(datetime.fromtimestamp(float(timestamp), UTC))


def _parse_utc_microseconds(value: str) -> datetime:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise RuntimeError("measurement evidence contains an invalid UTC time") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:
        raise RuntimeError("measurement evidence UTC time is not canonical")
    return parsed


def _run_root(run_id: str) -> Path:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("measurement run ID is invalid")
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
        raise RuntimeError("local record is outside the measurement artifact root")
    current = root
    for part in candidate.relative_to(root).parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError("local measurement artifact path contains a symbolic link")
    resolved = candidate.resolve(strict=False)
    if resolved != candidate:
        raise RuntimeError("local measurement artifact path is not canonical")
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
    _write_immutable(path, canonical_measurement_json_bytes(value))


def _read_control_model(path: Path, model: type[Any]) -> Any:
    resolved = _assert_local_artifact_path(path)
    if resolved.is_symlink() or not resolved.is_file():
        raise RuntimeError("local control record must be one regular file")
    payload = resolved.read_bytes()
    value = model.model_validate(strict_measurement_json_object(payload))
    if payload != value.canonical_bytes():
        raise RuntimeError("local control record is not canonical")
    return value


@contextmanager
def _operation_lock(run_id: str, operation: str) -> Iterator[None]:
    if operation not in {"deploy", "launch"}:
        raise ValueError("measurement operation is invalid")
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


def _require_reviewed_main() -> tuple[str, str]:
    if _git_text("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("measurement deployment requires a clean reviewed worktree")
    _git_text("fetch", "--quiet", "--no-tags", "origin", "main")
    head = _git_text("rev-parse", "HEAD")
    local_main = _git_text("rev-parse", "refs/heads/main")
    origin_main = _git_text("rev-parse", "refs/remotes/origin/main")
    if head != local_main or head != origin_main:
        raise RuntimeError(
            "measurement deployment requires HEAD, local main, and fetched origin/main to match"
        )
    tree = _git_text("rev-parse", "HEAD^{tree}")
    if re.fullmatch(r"[0-9a-f]{40}", head) is None or re.fullmatch(r"[0-9a-f]{40}", tree) is None:
        raise RuntimeError("reviewed Git identity is invalid")
    return head, tree


def _closed_control_paths(bundle: InklingMeasurementBundle) -> tuple[str, ...]:
    config = bundle.config
    fixed = {
        MANAGER_RELATIVE_PATH,
        RUNNER_RELATIVE_PATH,
        MEASUREMENT_CONFIG_RELATIVE_PATH,
        config.matched_cell_config.path,
        config.bf16_subject_reference.path,
        config.q3_verified_export_reference.path,
        config.source_adoption_reference.path,
        config.base_runtime.instrumentation_patch_path,
        config.measurement_patch.path,
        config.quality.diagnostic.path,
        config.quality.corpus_reference.path,
        CORPUS_MATERIALIZER_RELATIVE_PATH,
    }
    tracked_sources = {
        path
        for path in _git_text("ls-files", "src/inkling_quant_lab").splitlines()
        if path.endswith(".py")
    }
    if not tracked_sources:
        raise RuntimeError("measurement source closure is empty")
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
            raise RuntimeError(f"measurement control file is not regular: {relative}")
        resolved = path.resolve()
        if resolved == root or root not in resolved.parents:
            raise RuntimeError(f"measurement control file escapes the project: {relative}")
        result[relative] = path.read_bytes()
    return result


def _manifest_file(
    provenance: MeasurementControlPlaneProvenance,
    path: str,
) -> MeasurementControlPlaneFile:
    matches = tuple(item for item in provenance.files if item.path == path)
    if len(matches) != 1:
        raise RuntimeError(f"reviewed file is absent from the control closure: {path}")
    return matches[0]


def _measurement_run_id(bundle: InklingMeasurementBundle, control_sha256: str) -> str:
    return f"inkling-measurement-86b4d430-{bundle.config.config_hash()[:12]}-{control_sha256[:12]}"


def _build_reviewed_context() -> _ReviewedContext:
    commit, tree = _require_reviewed_main()
    bundle = load_measurement_bundle(PROJECT_ROOT)
    paths = _closed_control_paths(bundle)
    provenance = build_measurement_control_plane_provenance(
        reviewed_commit_sha=commit,
        reviewed_tree_sha=tree,
        files=_read_project_files(paths),
        required_paths=paths,
    )
    config = bundle.config
    reviewed = MeasurementReviewedInputs(
        control_plane=provenance,
        measurement_config=_manifest_file(provenance, MEASUREMENT_CONFIG_RELATIVE_PATH),
        resolved_config_sha256=config.config_hash(),
        diagnostic_dataset=_manifest_file(provenance, config.quality.diagnostic.path),
        corpus_reference=_manifest_file(provenance, config.quality.corpus_reference.path),
        corpus_materializer=_manifest_file(
            provenance,
            CORPUS_MATERIALIZER_RELATIVE_PATH,
        ),
        bf16_subject_reference=_manifest_file(provenance, config.bf16_subject_reference.path),
        q3_verified_export_reference=_manifest_file(
            provenance, config.q3_verified_export_reference.path
        ),
        source_adoption_reference=_manifest_file(provenance, config.source_adoption_reference.path),
        resources=MeasurementExecutionResources(),
    )
    if (
        config.storage.evidence_volume != MEASUREMENT_EVIDENCE_VOLUME_NAME
        or config.storage.attempt_registry != MEASUREMENT_ATTEMPT_REGISTRY_NAME
    ):
        raise RuntimeError("measurement storage differs from its control-plane constants")
    run_id = _measurement_run_id(bundle, provenance.control_plane_sha256)
    return _ReviewedContext(
        bundle=bundle,
        provenance=provenance,
        reviewed_inputs=reviewed,
        run_id=run_id,
        provenance_path=_provenance_path(run_id, provenance.control_plane_sha256),
    )


def _require_current_review(
    context: _ReviewedContext,
    reviewed: MeasurementReviewedInputs,
) -> None:
    if reviewed != context.reviewed_inputs:
        raise RuntimeError("control record differs from the current reviewed origin/main")


def inspect(*, as_json: bool) -> None:
    bundle = load_measurement_bundle(PROJECT_ROOT)
    config = bundle.config
    config_bytes = (PROJECT_ROOT / MEASUREMENT_CONFIG_RELATIVE_PATH).read_bytes()
    payload = {
        "status": "planned_not_executed",
        "measurement_config_content_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "measurement_config_semantic_sha256": config.config_hash(),
        "model": f"{config.model_id}@{config.revision}",
        "subject_order": list(config.execution.subject_order),
        "resources": config.resources.model_dump(mode="json"),
        "remote_execution_default_enabled": False,
        "paid_compute_started": False,
        "next_action": "prepare-deploy",
    }
    if as_json:
        print(canonical_measurement_json_bytes(payload).decode(), end="")
    else:
        for key, value in payload.items():
            print(f"{key}: {value}")


def prepare_deploy() -> None:
    context = _build_reviewed_context()
    created = datetime.now(UTC)
    expires = created + timedelta(seconds=MEASUREMENT_DEPLOY_CHALLENGE_MAX_AGE_SECONDS)
    challenge = MeasurementDeployConfirmationChallenge(
        created_at_utc=_utc_microseconds(created),
        expires_at_utc=_utc_microseconds(expires),
        confirmation_nonce=secrets.token_hex(32),
        reviewed_inputs=context.reviewed_inputs,
        app_name=measurement_app_name(context.provenance.control_plane_sha256),
    )
    _write_immutable(context.provenance_path, context.provenance.canonical_bytes())
    path = _deploy_challenge_path(context.run_id, challenge.challenge_sha256())
    _write_immutable(path, challenge.canonical_bytes())
    print(
        canonical_measurement_json_bytes(
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
        raise RuntimeError(f"measurement manager requires Modal {EXPECTED_MODAL_VERSION}")
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
                MEASUREMENT_ENVIRONMENT_NAME,
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
        raise RuntimeError("Modal App history lookup failed") from error
    value = json.loads(result.stdout)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise RuntimeError("Modal App history has an unexpected shape")
    return value


def _deployment_version(history: Sequence[Mapping[str, Any]], tag: str) -> int:
    if len(history) != 1:
        raise RuntimeError("content-addressed Modal App must have exactly one deployment")
    row = history[0]
    if row.get("tag") != tag:
        raise RuntimeError("Modal deployment tag differs from the reviewed tag")
    version_text = row.get("version")
    if version_text != "v1":
        raise RuntimeError("content-addressed Modal App must be at deployment version v1")
    return 1


def _reviewed_deployment_version(
    context: _ReviewedContext,
    history: Sequence[Mapping[str, Any]],
    tag: str,
) -> int:
    version = _deployment_version(history, tag)
    row = history[0]
    expected_commit = context.provenance.reviewed_commit_sha[:7]
    if row.get("commit") != expected_commit:
        raise RuntimeError("Modal deployment commit differs from reviewed origin/main")
    if row.get("client") != EXPECTED_MODAL_VERSION:
        raise RuntimeError("Modal deployment client differs from the pinned Modal version")
    return version


def _object_id(value: object, kind: str) -> str:
    if not isinstance(value, str) or OBJECT_ID_PATTERNS[kind].fullmatch(value) is None:
        raise RuntimeError(f"Modal {kind} object ID is invalid")
    return value


def _resource_created_at(resource: Any) -> str:
    metadata = resource._get_metadata()
    creation = getattr(getattr(metadata, "creation_info", None), "created_at", None)
    if not isinstance(creation, (int, float)) or isinstance(creation, bool):
        raise RuntimeError("Modal resource creation time is invalid")
    return _timestamp_microseconds(float(creation))


def _function_binding(function: Any) -> tuple[str, Literal["run_measurement"]]:
    function_id = _object_id(function.object_id, "function")
    name = getattr(function._get_metadata(), "function_name", None)
    if name != MEASUREMENT_FUNCTION_NAME:
        raise RuntimeError("Modal returned the wrong measurement Function")
    return function_id, cast(Literal["run_measurement"], name)


def _deployment_resources(
    modal: ModuleType,
    *,
    create_if_missing: bool,
) -> tuple[Any, Any]:
    registry = modal.Dict.from_name(
        MEASUREMENT_ATTEMPT_REGISTRY_NAME,
        environment_name=MEASUREMENT_ENVIRONMENT_NAME,
        create_if_missing=create_if_missing,
    )
    registry.hydrate()
    evidence = modal.Volume.from_name(
        MEASUREMENT_EVIDENCE_VOLUME_NAME,
        environment_name=MEASUREMENT_ENVIRONMENT_NAME,
        create_if_missing=create_if_missing,
        version=1,
    )
    evidence.hydrate()
    return registry, evidence


def _deploy_remote(context: _ReviewedContext) -> MeasurementDeploymentIdentity:
    modal = _load_modal()
    control_hash = context.provenance.control_plane_sha256
    app_name = measurement_app_name(control_hash)
    tag = measurement_deployment_tag(control_hash)
    history = _modal_history(app_name, allow_missing=True)
    recovering = bool(history)
    if recovering:
        version = _reviewed_deployment_version(context, history, tag)
    registry, evidence = _deployment_resources(
        modal,
        create_if_missing=not recovering,
    )

    if not recovering:
        environment = os.environ.copy()
        environment["IQL_MEASUREMENT_CONTROL_PLANE_SHA256"] = control_hash
        environment["IQL_MEASUREMENT_CONTROL_PLANE_PROVENANCE_PATH"] = str(context.provenance_path)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "deploy",
                "-e",
                MEASUREMENT_ENVIRONMENT_NAME,
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
        MEASUREMENT_FUNCTION_NAME,
        environment_name=MEASUREMENT_ENVIRONMENT_NAME,
    )
    function.hydrate()
    function_id, function_name = _function_binding(function)
    registry_id = _object_id(registry.object_id, "attempt registry")
    registry_created_at = _resource_created_at(registry)
    evidence_id = _object_id(evidence.object_id, "evidence volume")
    stable_registry, stable_evidence = _deployment_resources(
        modal,
        create_if_missing=False,
    )
    if (
        _object_id(stable_registry.object_id, "attempt registry") != registry_id
        or _resource_created_at(stable_registry) != registry_created_at
        or _object_id(stable_evidence.object_id, "evidence volume") != evidence_id
    ):
        raise RuntimeError("Modal deployment resources changed while the deployment was sealed")
    if _modal_history(app_name) != history:
        raise RuntimeError("Modal App changed while the deployment was being sealed")
    return MeasurementDeploymentIdentity(
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
    challenge = _read_control_model(challenge_path, MeasurementDeployConfirmationChallenge)
    challenge.confirm(confirmation)
    context = _build_reviewed_context()
    _require_current_review(context, challenge.reviewed_inputs)
    expected = _deploy_challenge_path(context.run_id, challenge.challenge_sha256()).resolve()
    if _assert_local_artifact_path(challenge_path) != expected:
        raise RuntimeError("deploy challenge path is not content addressed")
    with _operation_lock(context.run_id, "deploy"):
        deployment_path = _deployment_path(context.run_id)
        if deployment_path.exists() or deployment_path.is_symlink():
            raise RuntimeError("measurement control plane is already deployed")
        consumed = _deploy_consumption_path(context.run_id, challenge.challenge_sha256())
        if consumed.exists() or consumed.is_symlink():
            raise RuntimeError("deploy confirmation was already consumed")
        consumed_at_utc = _utc_microseconds()
        validate_measurement_deploy_challenge_not_expired(
            challenge,
            observed_at_utc=consumed_at_utc,
        )
        _write_immutable_json(
            consumed,
            {
                "schema_version": "inkling-measurement-deploy-consumption-v1",
                "status": "authorized_before_deploy",
                "consumed_at_utc": consumed_at_utc,
                "run_id": context.run_id,
                "challenge_sha256": challenge.challenge_sha256(),
                "control_plane_sha256": context.provenance.control_plane_sha256,
            },
        )
        deployment = _deploy_remote(context)
        _write_immutable(_deployment_path(context.run_id), deployment.canonical_bytes())
    print(
        canonical_measurement_json_bytes(
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


def _read_deployment(run_id: str) -> MeasurementDeploymentIdentity:
    return cast(
        MeasurementDeploymentIdentity,
        _read_control_model(_deployment_path(run_id), MeasurementDeploymentIdentity),
    )


def _launch_already_consumed(run_id: str) -> bool:
    consumption = _launch_consumption_path(run_id)
    intent_root = _control_path(run_id, "launch-intents")
    if intent_root.is_symlink():
        raise RuntimeError("measurement launch-intent directory is a symbolic link")
    return (
        consumption.exists()
        or consumption.is_symlink()
        or (intent_root.is_dir() and any(intent_root.iterdir()))
    )


def prepare_launch() -> None:
    context = _build_reviewed_context()
    if _launch_already_consumed(context.run_id):
        raise RuntimeError("the one measurement launch was already consumed")
    deployment = _read_deployment(context.run_id)
    deployment.validate_reviewed_inputs(context.reviewed_inputs)
    created = datetime.now(UTC)
    expires = created + timedelta(seconds=MEASUREMENT_LAUNCH_CHALLENGE_MAX_AGE_SECONDS)
    challenge = MeasurementLaunchConfirmationChallenge(
        created_at_utc=_utc_microseconds(created),
        expires_at_utc=_utc_microseconds(expires),
        authorization_nonce=secrets.token_hex(32),
        run_id=context.run_id,
        reviewed_inputs=context.reviewed_inputs,
        deployment=deployment,
    )
    path = _launch_challenge_path(context.run_id, challenge.challenge_sha256())
    _write_immutable(path, challenge.canonical_bytes())
    print(
        canonical_measurement_json_bytes(
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
    deployment: MeasurementDeploymentIdentity,
) -> tuple[Any, Any, Any]:
    modal = _load_modal()
    history = _modal_history(deployment.app_name)
    if _deployment_version(history, deployment.deployment_tag) != deployment.deployment_version:
        raise RuntimeError("Modal deployment version differs from the local seal")
    function = modal.Function.from_name(
        deployment.app_name,
        MEASUREMENT_FUNCTION_NAME,
        environment_name=MEASUREMENT_ENVIRONMENT_NAME,
    )
    function.hydrate()
    if _function_binding(function) != (deployment.function_id, deployment.function_name):
        raise RuntimeError("Modal Function differs from the local deployment seal")
    registry = modal.Dict.from_name(
        MEASUREMENT_ATTEMPT_REGISTRY_NAME,
        environment_name=MEASUREMENT_ENVIRONMENT_NAME,
        create_if_missing=False,
    )
    registry.hydrate()
    if (
        _object_id(registry.object_id, "attempt registry") != deployment.attempt_registry_id
        or _resource_created_at(registry) != deployment.attempt_registry_created_at_utc
    ):
        raise RuntimeError("Modal attempt registry differs from the local deployment seal")
    evidence = modal.Volume.from_name(
        MEASUREMENT_EVIDENCE_VOLUME_NAME,
        environment_name=MEASUREMENT_ENVIRONMENT_NAME,
        create_if_missing=False,
        version=1,
    )
    evidence.hydrate()
    if _object_id(evidence.object_id, "evidence volume") != deployment.evidence_volume_id:
        raise RuntimeError("Modal evidence Volume differs from the local deployment seal")
    if _modal_history(deployment.app_name) != history:
        raise RuntimeError("Modal App changed during launch validation")
    return function, registry, evidence


def _remote_path(relative: str) -> str:
    canonical = validate_repository_relative_path(relative)
    return f"/{canonical}"


def _remote_read(
    volume: Any,
    relative: str,
    *,
    maximum_bytes: int = MEASUREMENT_CONTROL_RECORD_MAX_BYTES,
    expected_size_bytes: int | None = None,
) -> bytes | None:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("remote measurement read limit must be a positive integer")
    if expected_size_bytes is not None and (
        type(expected_size_bytes) is not int
        or expected_size_bytes <= 0
        or expected_size_bytes > maximum_bytes
    ):
        raise ValueError("remote measurement expected size exceeds its read limit")
    modal = _load_modal()
    payload = bytearray()
    read_limit = maximum_bytes if expected_size_bytes is None else expected_size_bytes
    try:
        for chunk in volume.read_file(_remote_path(relative)):
            if not isinstance(chunk, bytes):
                raise RuntimeError("Modal returned non-byte measurement evidence")
            if len(chunk) > read_limit - len(payload):
                raise RuntimeError("remote measurement record exceeds its size limit")
            payload.extend(chunk)
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
    if expected_size_bytes is not None and len(payload) != expected_size_bytes:
        raise RuntimeError("remote measurement record size differs from its reference")
    return bytes(payload)


def _remote_write_immutable(volume: Any, relative: str, payload: bytes) -> None:
    if _remote_read(volume, relative) is not None:
        raise RuntimeError("remote immutable measurement record already exists")
    with volume.batch_upload(force=False) as upload:
        upload.put_file(io.BytesIO(payload), _remote_path(relative), mode=0o400)
    volume.reload()
    if _remote_read(volume, relative) != payload:
        raise RuntimeError("remote measurement record readback differs from uploaded bytes")


def _cancel_call(call: Any, reason: str) -> RuntimeError:
    try:
        call.cancel(terminate_containers=True)
    except Exception as error:
        return RuntimeError(f"{reason}; call cancellation also failed: {error}")
    return RuntimeError(f"{reason}; call cancellation was requested")


def _validated_call_id(call: Any) -> str:
    call_id = getattr(call, "object_id", None)
    if not isinstance(call_id, str) or CALL_ID_PATTERN.fullmatch(call_id) is None:
        raise RuntimeError("Modal call ID is invalid")
    return call_id


def _publish_acceptance(
    volume: Any,
    acceptance: MeasurementPostSpawnAcceptance,
) -> None:
    path = measurement_post_spawn_acceptance_path(
        acceptance.run_id, acceptance.launch_intent_sha256
    )
    payload = acceptance.canonical_bytes()
    _remote_write_immutable(volume, path, payload)
    validate_measurement_post_spawn_acceptance(
        _remote_read(volume, path) or b"",
        expected=acceptance,
        acceptance_sha256=acceptance.acceptance_sha256(),
        evidence_path=path,
    )


def launch(*, challenge_path: Path, confirmation: str) -> None:
    challenge = _read_control_model(challenge_path, MeasurementLaunchConfirmationChallenge)
    challenge.confirm(confirmation)
    context = _build_reviewed_context()
    _require_current_review(context, challenge.reviewed_inputs)
    if challenge.run_id != context.run_id:
        raise RuntimeError("launch challenge has the wrong run ID")
    deployment = _read_deployment(context.run_id)
    if challenge.deployment != deployment:
        raise RuntimeError("launch challenge differs from the sealed deployment")
    expected = _launch_challenge_path(context.run_id, challenge.challenge_sha256()).resolve()
    if _assert_local_artifact_path(challenge_path) != expected:
        raise RuntimeError("launch challenge path is not content addressed")

    with _operation_lock(context.run_id, "launch"):
        if _launch_already_consumed(context.run_id):
            raise RuntimeError("the one measurement launch was already consumed")
        function, registry, evidence = _validate_remote_deployment(deployment)
        _assert_remote_attempt_unconsumed(
            registry,
            evidence,
            run_id=context.run_id,
        )
        authorized_at = _utc_microseconds()
        intent = build_measurement_launch_intent(
            challenge,
            confirmation=confirmation,
            authorized_at_utc=authorized_at,
        )
        intent_sha256 = intent.intent_sha256()
        _write_immutable_json(
            _launch_consumption_path(context.run_id),
            {
                "schema_version": "inkling-measurement-launch-consumption-v1",
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
        remote_intent_path = measurement_launch_intent_path(context.run_id, intent_sha256)
        _remote_write_immutable(evidence, remote_intent_path, intent.canonical_bytes())
        validate_measurement_launch_intent(
            _remote_read(evidence, remote_intent_path) or b"",
            expected=intent,
            intent_sha256=intent_sha256,
            evidence_path=remote_intent_path,
        )

        call = function.spawn(context.run_id, intent_sha256)
        try:
            call_id = _validated_call_id(call)
            acceptance = build_measurement_post_spawn_acceptance(
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
                "schema_version": "inkling-measurement-call-receipt-v1",
                "status": "accepted_after_spawn",
                "run_id": context.run_id,
                "call_id": call_id,
                "launch_intent_sha256": intent_sha256,
                "post_spawn_acceptance_sha256": acceptance.acceptance_sha256(),
                "deployment_sha256": hashlib.sha256(deployment.canonical_bytes()).hexdigest(),
                "function_return_is_success_evidence": False,
            }
            _write_immutable_json(_call_receipt_path(context.run_id, intent_sha256), receipt)
        except Exception as error:
            raise _cancel_call(call, "post-spawn acceptance failed") from error
    print(canonical_measurement_json_bytes(receipt).decode(), end="")


def _read_launch_binding(run_id: str, call_id: str) -> _LaunchBinding:
    deployment = _read_deployment(run_id)
    calls = _run_root(run_id) / "calls"
    if calls.is_symlink() or not calls.is_dir():
        raise RuntimeError("local measurement call receipt is missing")
    entries = tuple(calls.glob("*.json"))
    if len(entries) != 1 or entries[0].is_symlink():
        raise RuntimeError("expected exactly one local measurement call receipt")
    raw = strict_measurement_json_object(entries[0].read_bytes())
    if entries[0].read_bytes() != canonical_measurement_json_bytes(raw):
        raise RuntimeError("local measurement call receipt is not canonical")
    intent_sha256 = raw.get("launch_intent_sha256")
    if (
        raw.get("schema_version") != "inkling-measurement-call-receipt-v1"
        or raw.get("status") != "accepted_after_spawn"
        or raw.get("run_id") != run_id
        or raw.get("call_id") != call_id
        or not isinstance(intent_sha256, str)
        or SHA256_PATTERN.fullmatch(intent_sha256) is None
        or entries[0].name != f"{intent_sha256}.json"
        or raw.get("deployment_sha256") != hashlib.sha256(deployment.canonical_bytes()).hexdigest()
        or raw.get("function_return_is_success_evidence") is not False
    ):
        raise RuntimeError("local call receipt differs from the requested call")
    intent = _read_control_model(
        _launch_intent_local_path(run_id, intent_sha256), MeasurementLaunchIntent
    )
    if intent.intent_sha256() != intent_sha256 or intent.deployment != deployment:
        raise RuntimeError("local launch intent differs from the sealed deployment")
    acceptance = _read_control_model(
        _acceptance_local_path(run_id, intent_sha256), MeasurementPostSpawnAcceptance
    )
    if (
        acceptance.call_id != call_id
        or acceptance.launch_intent_sha256 != intent_sha256
        or acceptance.deployment != deployment
        or raw.get("post_spawn_acceptance_sha256") != acceptance.acceptance_sha256()
    ):
        raise RuntimeError("local post-spawn acceptance differs from the requested call")
    return _LaunchBinding(deployment=deployment, intent=intent, acceptance=acceptance)


def _fresh_remote_resources(
    deployment: MeasurementDeploymentIdentity,
) -> tuple[Any, Any]:
    _, registry, volume = _validate_remote_deployment(deployment)
    volume.reload()
    return registry, volume


def _list_remote_files(
    volume: Any,
    relative: str,
    *,
    maximum_file_bytes: int = MEASUREMENT_CONTROL_RECORD_MAX_BYTES,
    maximum_entries: int = 1,
) -> tuple[tuple[str, int], ...]:
    if type(maximum_file_bytes) is not int or maximum_file_bytes <= 0:
        raise ValueError("remote measurement listing limit must be a positive integer")
    if type(maximum_entries) is not int or maximum_entries <= 0:
        raise ValueError("remote measurement entry limit must be a positive integer")
    modal = _load_modal()
    try:
        entries = volume.listdir(_remote_path(relative))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return ()
    result: list[tuple[str, int]] = []
    for entry in entries:
        if len(result) >= maximum_entries:
            raise RuntimeError("remote measurement directory exceeds its entry limit")
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
            raise RuntimeError("remote measurement directory contains an invalid entry")
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
    parent = PurePosixPath(relative_path).parent.as_posix()
    entries = _list_remote_files(
        volume,
        parent,
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


def _assert_remote_attempt_unconsumed(
    registry: Any,
    volume: Any,
    *,
    run_id: str,
) -> None:
    key = measurement_attempt_registry_key(run_id)
    present = registry.contains(key)
    if type(present) is not bool:
        raise RuntimeError("Modal attempt registry returned an invalid presence value")
    volume.reload()
    evidence_roots = (
        f"runs/{run_id}/control/launch-intents",
        f"runs/{run_id}/control/post-spawn-acceptances",
        f"runs/{run_id}/control/attempt-claims",
        f"runs/{run_id}/terminal/success",
        f"runs/{run_id}/terminal/failure",
    )
    if present or any(_list_remote_files(volume, root) for root in evidence_roots):
        raise RuntimeError("remote measurement evidence shows that the attempt was consumed")


def _load_reviewed_bundle(binding: _LaunchBinding) -> InklingMeasurementBundle:
    reviewed = binding.intent.reviewed_inputs
    for item in reviewed.control_plane.files:
        payload = _read_project_files((item.path,))[item.path]
        if len(payload) != item.size_bytes or hashlib.sha256(payload).hexdigest() != item.sha256:
            raise RuntimeError(
                f"local reviewed measurement file differs from the accepted launch: {item.path}"
            )
    bundle = load_measurement_bundle(PROJECT_ROOT)
    if bundle.config.config_hash() != reviewed.resolved_config_sha256:
        raise RuntimeError("local resolved measurement config differs from the accepted launch")
    return bundle


def _expected_attempt_bindings(
    binding: _LaunchBinding,
    claim: MeasurementAttemptClaim,
) -> dict[str, str]:
    return {
        "control_plane_sha256": binding.deployment.control_plane_sha256,
        "reviewed_config_file_sha256": (binding.intent.reviewed_inputs.measurement_config.sha256),
        "resolved_config_sha256": binding.intent.reviewed_inputs.resolved_config_sha256,
        "launch_intent_sha256": binding.intent.intent_sha256(),
        "post_spawn_acceptance_sha256": binding.acceptance.acceptance_sha256(),
        "call_id": binding.acceptance.call_id,
        "attempt_claim_sha256": claim.claim_sha256(),
    }


def _validate_attempt_bound_record(
    record: MeasurementSubjectCompactRecord | MeasurementComparisonCompactRecord,
    *,
    binding: _LaunchBinding,
    claim: MeasurementAttemptClaim,
) -> None:
    if record.run_id != binding.intent.run_id:
        raise RuntimeError("measurement supporting record has the wrong run ID")
    expected = _expected_attempt_bindings(binding, claim)
    if any(getattr(record, field) != value for field, value in expected.items()):
        raise RuntimeError("measurement supporting record differs from the accepted attempt")


def _read_supporting_record(
    volume: Any,
    reference: MeasurementSupportingRecordReference,
) -> bytes:
    payload = _read_only_remote_file(
        volume,
        relative_path=reference.relative_path,
        expected_size_bytes=reference.size_bytes,
        maximum_bytes=MEASUREMENT_CONTROL_RECORD_MAX_BYTES,
        label=f"{reference.kind} supporting record",
    )
    validate_measurement_supporting_record_reference(payload, expected=reference)
    return payload


def _read_raw_blob(
    volume: Any,
    reference: MeasurementRawBlobReference,
) -> bytes:
    maximum_bytes = MEASUREMENT_RAW_BLOB_MAX_BYTES[reference.kind]
    payload = _read_only_remote_file(
        volume,
        relative_path=reference.relative_path,
        expected_size_bytes=reference.size_bytes,
        maximum_bytes=maximum_bytes,
        label=f"{reference.subject} {reference.kind} raw evidence",
    )
    validate_measurement_raw_blob_reference(payload, expected=reference)
    return payload


def _bound_attempt_claim(
    registry: Any,
    volume: Any,
    binding: _LaunchBinding,
) -> _AttemptInspection:
    run_id = binding.intent.run_id
    key = measurement_attempt_registry_key(run_id)
    present = registry.contains(key)
    if type(present) is not bool:
        raise RuntimeError("Modal attempt registry returned an invalid presence value")
    live_payload = registry.get(key) if present else None
    root = f"runs/{run_id}/control/attempt-claims"
    durable = _list_remote_files(volume, root)
    if len(durable) > 1:
        raise RuntimeError("measurement evidence has multiple attempt claims")
    if not present and not durable:
        return _AttemptInspection(claim=None, durable=False)
    if present and not isinstance(live_payload, bytes):
        raise RuntimeError("live measurement attempt claim is missing or invalid")
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
            raise RuntimeError("durable measurement attempt claim disappeared")
    claim_payload = live_payload if isinstance(live_payload, bytes) else durable_payload
    if claim_payload is None:
        raise RuntimeError("measurement attempt claim is unavailable")
    claim = MeasurementAttemptClaim.model_validate(strict_measurement_json_object(claim_payload))
    digest = claim.claim_sha256()
    if claim_payload != claim.canonical_bytes():
        raise RuntimeError("measurement attempt claim is not canonical")
    expected_path = measurement_attempt_claim_path(run_id, digest)
    if durable_payload is not None:
        if durable_path != expected_path:
            raise RuntimeError("durable measurement attempt claim path is not content addressed")
        if isinstance(live_payload, bytes) and durable_payload != live_payload:
            raise RuntimeError("durable measurement attempt claim differs from the atomic claim")
        validate_measurement_attempt_claim(
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
        != binding.intent.reviewed_inputs.measurement_config.sha256
        or claim.resolved_config_sha256 != binding.intent.reviewed_inputs.resolved_config_sha256
        or claim.control_plane_sha256 != binding.deployment.control_plane_sha256
        or _parse_utc_microseconds(claim.claimed_at_utc)
        < _parse_utc_microseconds(acceptance.accepted_at_utc)
    ):
        raise RuntimeError("measurement attempt claim differs from the accepted launch")
    return _AttemptInspection(claim=claim, durable=bool(durable))


def _validate_record_config_scope(
    record: MeasurementSubjectCompactRecord | MeasurementComparisonCompactRecord,
    *,
    bundle: InklingMeasurementBundle,
) -> None:
    config = bundle.config
    expected = {
        "model_id": config.model_id,
        "model_revision": config.revision,
        "protocol_sha256": measurement_protocol_sha256(config),
        "workload_sha256": measurement_workload_sha256(config),
    }
    if any(getattr(record, field) != value for field, value in expected.items()):
        raise RuntimeError(
            "measurement supporting record differs from the reviewed experiment scope"
        )


def _expected_subject_staging_projection(
    bundle: InklingMeasurementBundle,
    *,
    subject: MeasurementEvidenceSubject,
) -> tuple[tuple[str, str, str, int], ...]:
    """Build the reviewed full staging inventory in its required order."""

    paths = bundle.matched.paths
    tokenizer_artifacts = tuple(
        (path, artifact.sha256, artifact.size_bytes)
        for path, artifact in zip(
            paths.tokenizer_assets,
            bundle.matched.config.tokenizer_assets,
            strict=True,
        )
    )
    projector = bundle.matched.q3.projector
    shared_projector = (
        paths.shared_projector,
        projector.sha256,
        projector.size_bytes,
    )
    if subject == "bf16":
        artifacts = (
            *(
                (path, artifact.sha256, artifact.size_bytes)
                for path, artifact in zip(
                    paths.bf16_shards,
                    bundle.matched.bf16.bf16_shards,
                    strict=True,
                )
            ),
            (
                paths.bf16_conversion_receipt,
                bundle.matched.bf16.conversion_receipt.sha256,
                bundle.matched.bf16.conversion_receipt.size_bytes,
            ),
            shared_projector,
            *tokenizer_artifacts,
        )
    else:
        artifacts = (
            *(
                (path, artifact.sha256, artifact.size_bytes)
                for path, artifact in zip(
                    paths.q3_shards,
                    bundle.matched.q3.q3_shards,
                    strict=True,
                )
            ),
            shared_projector,
            (
                paths.q3_export_manifest,
                bundle.matched.q3.export_manifest.sha256,
                bundle.matched.q3.export_manifest.size_bytes,
            ),
            (
                paths.q3_verify_receipt,
                bundle.matched.q3.verify_receipt.sha256,
                bundle.matched.q3.verify_receipt.size_bytes,
            ),
            (
                paths.q3_quantize_receipt,
                bundle.matched.q3.quantize_receipt.sha256,
                bundle.matched.q3.quantize_receipt.size_bytes,
            ),
            (
                paths.projector_conversion_receipt,
                bundle.matched.q3.mmproj_receipt.sha256,
                bundle.matched.q3.mmproj_receipt.size_bytes,
            ),
            *tokenizer_artifacts,
        )
    staging_root = f"/cache/inkling-measurement-subject/{subject}"
    return tuple(
        (
            source_path,
            f"{staging_root}/{source_path.removeprefix('/')}",
            sha256,
            size_bytes,
        )
        for source_path, sha256, size_bytes in artifacts
    )


def _validate_subject_raw_evidence(
    record: MeasurementSubjectCompactRecord,
    payloads: Mapping[str, bytes],
    *,
    bundle: InklingMeasurementBundle,
    binding: _LaunchBinding,
    claim: MeasurementAttemptClaim,
    placement_policy: ExactCudaPlacementPolicy,
) -> _ValidatedSubjectRawEvidence:
    if tuple(payloads) != ("token_nll", "raw_trials", "resource_telemetry", "backend_audit"):
        raise RuntimeError("measurement raw evidence is incomplete or out of order")
    token_nll = parse_token_nll_raw_evidence(payloads["token_nll"])
    raw_trials = parse_raw_trials_evidence(payloads["raw_trials"])
    telemetry = parse_resource_telemetry_evidence(payloads["resource_telemetry"])
    backend_audit = parse_backend_audit_evidence(payloads["backend_audit"])
    expected_bindings = MeasurementAttemptBindings(
        run_id=record.run_id,
        subject=record.subject,
        reviewed_config_file_sha256=record.reviewed_config_file_sha256,
        resolved_config_sha256=record.resolved_config_sha256,
        protocol_sha256=record.protocol_sha256,
        workload_sha256=record.workload_sha256,
        launch_intent_sha256=binding.intent.intent_sha256(),
        post_spawn_acceptance_sha256=binding.acceptance.acceptance_sha256(),
        call_id=binding.acceptance.call_id,
        attempt_claim_sha256=claim.claim_sha256(),
    )
    if raw_trials.bindings != expected_bindings:
        raise RuntimeError("measurement raw evidence differs from the accepted subject attempt")
    links = validate_measurement_raw_evidence_links(
        token_nll,
        raw_trials,
        telemetry,
        backend_audit,
    )
    if (
        links.run_id != record.run_id
        or links.subject != record.subject
        or links.hardware_identity_sha256 != record.hardware_identity_sha256
    ):
        raise RuntimeError("measurement raw evidence differs from the compact subject identity")

    staging_artifacts = raw_trials.staging.artifacts
    observed_staging_projection = tuple(
        (
            item.source_path,
            item.staged_path,
            item.sha256,
            item.size_bytes,
        )
        for item in staging_artifacts
    )
    if observed_staging_projection != _expected_subject_staging_projection(
        bundle,
        subject=record.subject,
    ):
        raise RuntimeError("measurement staging inventory differs from the exact reviewed subject")
    if (
        raw_trials.server.load_pair_repetitions
        != bundle.config.performance.server.load_pair_repetitions
    ):
        raise RuntimeError("measurement load-pair count differs from the reviewed protocol")
    validate_measurement_diagnostic_evidence(
        bundle.diagnostic_items,
        prompt_template=bundle.config.quality.prompt_template,
        raw_trials=raw_trials,
    )
    projector = tuple(
        item
        for item in staging_artifacts
        if PurePosixPath(item.source_path).name == "mmproj-BF16.gguf"
    )
    if len(projector) != 1:
        raise RuntimeError("measurement staging evidence lacks one exact projector")
    executable_artifacts = (*staging_artifacts[:49], projector[0])
    compact_artifacts = tuple(
        (
            item.source_path,
            item.staged_path,
            item.sha256,
            item.size_bytes,
        )
        for item in record.artifact_inventory
    )
    raw_artifacts = tuple(
        (
            item.source_path,
            item.staged_path,
            item.sha256,
            item.size_bytes,
        )
        for item in executable_artifacts
    )
    if compact_artifacts != raw_artifacts:
        raise RuntimeError("measurement compact artifact inventory differs from staging evidence")

    quality = recompute_subject_quality_summary(token_nll, raw_trials)
    performance = recompute_subject_performance_summary(raw_trials)
    if record.quality_projection_sha256 != measurement_subject_quality_projection_sha256(
        quality
    ) or record.performance_projection_sha256 != measurement_subject_performance_projection_sha256(
        performance
    ):
        raise RuntimeError("measurement compact subject projections differ from raw evidence")
    placement_summaries = build_measurement_placement_summaries(
        backend_audit,
        backend_audit_content_sha256=record.raw_blobs[-1].content_sha256,
        policy=placement_policy,
    )
    if record.placement_summaries != placement_summaries:
        raise RuntimeError("measurement compact CUDA placement summaries differ from full logs")
    pairing = recompute_pairing_projection_hashes(token_nll, raw_trials)
    return _ValidatedSubjectRawEvidence(
        subject=record.subject,
        token_nll=token_nll,
        raw_trials=raw_trials,
        telemetry=telemetry,
        backend_audit=backend_audit,
        links=links,
        quality=quality,
        performance=performance,
        pairing=pairing,
    )


def _read_terminal_supporting_evidence(
    volume: Any,
    *,
    receipt: MeasurementTerminalReceipt,
    binding: _LaunchBinding,
    claim: MeasurementAttemptClaim,
    bundle: InklingMeasurementBundle,
) -> _TerminalSupportingEvidence:
    subjects: list[MeasurementSubjectCompactRecord] = []
    comparison: MeasurementComparisonCompactRecord | None = None
    raw_blobs: list[tuple[MeasurementRawBlobReference, bytes]] = []
    raw_subjects: list[_ValidatedSubjectRawEvidence] = []
    placement_policy = build_matched_cuda_placement_policy(bundle.matched.config)
    for reference in receipt.supporting_records:
        payload = _read_supporting_record(volume, reference)
        if reference.kind == "comparison":
            if comparison is not None:
                raise RuntimeError("measurement terminal has multiple comparison records")
            comparison = parse_measurement_comparison_compact_record(
                payload,
                run_id=receipt.run_id,
            )
            _validate_attempt_bound_record(comparison, binding=binding, claim=claim)
            _validate_record_config_scope(comparison, bundle=bundle)
            continue
        subject: MeasurementEvidenceSubject = "bf16" if reference.kind == "bf16_subject" else "q3"
        record = parse_measurement_subject_compact_record(
            payload,
            run_id=receipt.run_id,
            subject=subject,
        )
        _validate_attempt_bound_record(record, binding=binding, claim=claim)
        _validate_record_config_scope(record, bundle=bundle)
        subjects.append(record)
        subject_payloads: dict[str, bytes] = {}
        for raw_reference in record.raw_blobs:
            raw_payload = _read_raw_blob(volume, raw_reference)
            raw_blobs.append((raw_reference, raw_payload))
            subject_payloads[raw_reference.kind] = raw_payload
        raw_subjects.append(
            _validate_subject_raw_evidence(
                record,
                subject_payloads,
                bundle=bundle,
                binding=binding,
                claim=claim,
                placement_policy=placement_policy,
            )
        )

    subject_order = tuple(subject.subject for subject in subjects)
    if subject_order not in ((), ("bf16",), ("bf16", "q3")):
        raise RuntimeError("measurement subject records are incomplete or out of order")
    if len(subjects) == 2:
        shared_fields = (
            "run_id",
            "control_plane_sha256",
            "reviewed_config_file_sha256",
            "resolved_config_sha256",
            "launch_intent_sha256",
            "post_spawn_acceptance_sha256",
            "call_id",
            "attempt_claim_sha256",
            "runtime_manifest_sha256",
            "hardware_identity_sha256",
            "model_id",
            "model_revision",
            "protocol_sha256",
            "workload_sha256",
        )
        if any(
            getattr(subjects[0], field) != getattr(subjects[1], field) for field in shared_fields
        ):
            raise RuntimeError(
                "measurement subject records differ in their matched experiment scope"
            )
        validate_pairing_projection_hashes(
            raw_subjects[0].pairing,
            raw_subjects[1].pairing,
        )
    if comparison is not None:
        if len(subjects) != 2:
            raise RuntimeError("measurement comparison record lacks both exact subject records")
        validate_measurement_comparison_links(
            comparison,
            bf16=subjects[0],
            q3=subjects[1],
        )
        pairing = raw_subjects[0].pairing
        if (
            comparison.token_nll_pairing_sha256 != pairing.token_nll_pairing_sha256
            or comparison.diagnostic_pairing_sha256 != pairing.diagnostic_pairing_sha256
            or comparison.performance_pairing_sha256 != pairing.performance_pairing_sha256
        ):
            raise RuntimeError(
                "measurement comparison pairing differs from recomputed raw evidence"
            )
    return _TerminalSupportingEvidence(
        subjects=tuple(subjects),
        comparison=comparison,
        raw_blobs=tuple(raw_blobs),
        raw_subjects=tuple(raw_subjects),
    )


def _validate_success_scope(
    receipt: MeasurementSuccessTerminalReceipt,
    supporting: _TerminalSupportingEvidence,
    *,
    bundle: InklingMeasurementBundle,
) -> None:
    if len(supporting.subjects) != 2 or supporting.comparison is None:
        raise RuntimeError("completed measurement evidence is structurally incomplete")
    comparison = supporting.comparison
    bf16_raw, q3_raw = supporting.raw_subjects
    quality_rollup = build_measurement_quality_rollup(
        bf16_raw.quality,
        q3_raw.quality,
        paired_inputs_validated=True,
    )
    performance_rollup = build_measurement_performance_rollup(
        bf16_raw.performance,
        q3_raw.performance,
        llama_bench_workload_identity=(bundle.config.performance.llama_bench.workload_identity),
        server_workload_identity=bundle.config.performance.server.workload_identity,
        equivalent_trials_validated=True,
    )
    if receipt.quality_rollup != quality_rollup or receipt.performance_rollup != performance_rollup:
        raise RuntimeError("measurement terminal rollups differ from recomputed raw evidence")
    config = bundle.config
    quality_rollup_sha256 = measurement_quality_rollup_sha256(quality_rollup)
    performance_rollup_sha256 = measurement_performance_rollup_sha256(performance_rollup)
    expected = {
        "runtime_manifest_sha256": receipt.runtime_identity.manifest_sha256,
        "hardware_identity_sha256": receipt.hardware_identity_sha256,
        "model_id": config.model_id,
        "model_revision": config.revision,
        "protocol_sha256": measurement_protocol_sha256(config),
        "workload_sha256": measurement_workload_sha256(config),
        "quality_rollup_sha256": quality_rollup_sha256,
        "performance_rollup_sha256": performance_rollup_sha256,
    }
    if any(getattr(receipt, field) != value for field, value in expected.items()):
        raise RuntimeError("measurement terminal scope differs from its exact result")
    compared = {
        "runtime_manifest_sha256": receipt.runtime_manifest_sha256,
        "hardware_identity_sha256": receipt.hardware_identity_sha256,
        "model_id": receipt.model_id,
        "model_revision": receipt.model_revision,
        "protocol_sha256": receipt.protocol_sha256,
        "workload_sha256": receipt.workload_sha256,
        "quality_rollup_sha256": quality_rollup_sha256,
        "performance_rollup_sha256": performance_rollup_sha256,
    }
    if any(getattr(comparison, field) != value for field, value in compared.items()):
        raise RuntimeError("measurement comparison differs from the terminal scope")


def _terminal_evidence(
    volume: Any,
    binding: _LaunchBinding,
    attempt: _AttemptInspection,
    *,
    bundle: InklingMeasurementBundle,
) -> _ValidatedTerminalEvidence | None:
    run_id = binding.intent.run_id
    candidates: list[tuple[Literal["success", "failure"], str, int]] = []
    for outcome in ("success", "failure"):
        root = f"runs/{run_id}/terminal/{outcome}"
        for path, size in _list_remote_files(volume, root):
            candidates.append((outcome, path, size))
    if not candidates:
        return None
    claim = attempt.claim
    if len(candidates) != 1 or claim is None or not attempt.durable:
        raise RuntimeError(
            "measurement terminal evidence is conflicting or lacks a durable attempt claim"
        )
    outcome, path, size = candidates[0]
    payload = _remote_read(
        volume,
        path,
        expected_size_bytes=size,
    )
    if payload is None:
        raise RuntimeError("measurement terminal receipt disappeared during validation")
    receipt = parse_measurement_terminal_receipt(
        payload,
        run_id=run_id,
        outcome=outcome,
    )
    reference = build_measurement_terminal_receipt_reference(
        payload,
        evidence_root=EVIDENCE_MOUNT_ROOT,
        run_id=run_id,
        outcome=outcome,
    )
    if path != reference.relative_path:
        raise RuntimeError("measurement terminal receipt path is not content addressed")
    if reference.size_bytes != size:
        raise RuntimeError("measurement terminal receipt size differs from its listing")
    expected_bindings = _expected_attempt_bindings(binding, claim)
    if (
        receipt.stage != MEASUREMENT_STAGE
        or receipt.run_id != run_id
        or _parse_utc_microseconds(receipt.completed_at_utc)
        < _parse_utc_microseconds(claim.claimed_at_utc)
        or any(getattr(receipt, field) != value for field, value in expected_bindings.items())
    ):
        raise RuntimeError("measurement terminal receipt differs from the accepted attempt")
    supporting = _read_terminal_supporting_evidence(
        volume,
        receipt=receipt,
        binding=binding,
        claim=claim,
        bundle=bundle,
    )
    if isinstance(receipt, MeasurementSuccessTerminalReceipt):
        _validate_success_scope(receipt, supporting, bundle=bundle)
    elif supporting.subjects:
        runtime = receipt.runtime_identity
        if runtime is None or any(
            subject.runtime_manifest_sha256 != runtime.manifest_sha256
            for subject in supporting.subjects
        ):
            raise RuntimeError(
                "failed measurement supporting records differ from the runtime identity"
            )
    return _ValidatedTerminalEvidence(reference=reference, receipt=receipt)


def _provider_state(call_id: str) -> tuple[str, int]:
    modal = _load_modal()
    roots = modal.FunctionCall.from_id(call_id).get_call_graph()
    if not isinstance(roots, list):
        raise RuntimeError("Modal call graph has an unexpected shape")
    pending = list(roots)
    states: list[str] = []
    while pending:
        if len(states) >= MAX_CALL_GRAPH_NODES:
            raise RuntimeError("Modal call graph exceeds its size limit")
        node = pending.pop()
        status = getattr(getattr(node, "status", None), "name", None)
        children = getattr(node, "children", None)
        if not isinstance(status, str) or not isinstance(children, list):
            raise RuntimeError("Modal call graph contains invalid metadata")
        states.append(status.casefold())
        pending.extend(children)
    if set(states) & {"failure", "init_failure", "terminated", "timeout"}:
        return "failed_without_terminal_evidence", 1
    if states and set(states) == {"success"}:
        return "returned_without_terminal_evidence", 1
    return "running_or_queued", 0


def status(*, run_id: str, call_id: str) -> int:
    if RUN_ID_PATTERN.fullmatch(run_id) is None or CALL_ID_PATTERN.fullmatch(call_id) is None:
        raise ValueError("measurement run or Modal call ID is invalid")
    binding = _read_launch_binding(run_id, call_id)
    bundle = _load_reviewed_bundle(binding)
    registry, volume = _fresh_remote_resources(binding.deployment)
    intent_sha256 = binding.intent.intent_sha256()
    intent_path = measurement_launch_intent_path(run_id, intent_sha256)
    remote_intent = _read_only_remote_file(
        volume,
        relative_path=intent_path,
        expected_size_bytes=len(binding.intent.canonical_bytes()),
        maximum_bytes=MEASUREMENT_CONTROL_RECORD_MAX_BYTES,
        label="launch intent",
    )
    validate_measurement_launch_intent(
        remote_intent,
        expected=binding.intent,
        intent_sha256=intent_sha256,
        evidence_path=intent_path,
    )
    acceptance_path = measurement_post_spawn_acceptance_path(
        run_id,
        intent_sha256,
    )
    remote_acceptance = _read_only_remote_file(
        volume,
        relative_path=acceptance_path,
        expected_size_bytes=len(binding.acceptance.canonical_bytes()),
        maximum_bytes=MEASUREMENT_CONTROL_RECORD_MAX_BYTES,
        label="post-spawn acceptance",
    )
    validate_measurement_post_spawn_acceptance(
        remote_acceptance,
        expected=binding.acceptance,
        acceptance_sha256=binding.acceptance.acceptance_sha256(),
        evidence_path=acceptance_path,
    )
    attempt = _bound_attempt_claim(registry, volume, binding)
    claim = attempt.claim
    terminal = _terminal_evidence(
        volume,
        binding,
        attempt,
        bundle=bundle,
    )
    if terminal is not None:
        reference = terminal.reference
        receipt = terminal.receipt
        outcome = reference.outcome
        print(
            canonical_measurement_json_bytes(
                {
                    "status": "completed" if outcome == "success" else "failed",
                    "evidence_status": f"validated_terminal_{outcome}",
                    "run_id": run_id,
                    "call_id": call_id,
                    "terminal_receipt": reference.model_dump(mode="json"),
                    "attempt_claim_sha256": None if claim is None else claim.claim_sha256(),
                    "measurement_completed": receipt.measurement_completed,
                    "quality_retention_passed": receipt.quality_retention_passed,
                    "performance_comparison_complete": (receipt.performance_comparison_complete),
                    "speedup_claim_allowed": receipt.speedup_claim_allowed,
                    "function_return_is_success_evidence": False,
                }
            ).decode(),
            end="",
        )
        return 0 if outcome == "success" else 1
    provider_status, exit_code = _provider_state(call_id)
    print(
        canonical_measurement_json_bytes(
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
            raise RuntimeError("unsupported measurement manager command")
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
