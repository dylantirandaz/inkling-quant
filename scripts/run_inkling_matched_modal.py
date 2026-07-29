"""Run the exact BF16 then Q3 Inkling smoke cell on one Modal allocation.

This module is the paid, network-blocked data plane.  Do not invoke it
directly.  ``scripts/manage_inkling_matched_modal.py`` seals the reviewed
control plane, writes a one-use launch intent, and starts this function.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
import zlib
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, Protocol, cast

import modal
from pydantic import BaseModel

LOCAL_PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
LOCAL_SRC_ROOT: Final = LOCAL_PROJECT_ROOT / "src"
REMOTE_PROJECT_ROOT: Final = Path("/root/iql_project")
REMOTE_PROVENANCE_PATH: Final = Path("/root/iql-control-provenance.json")
REMOTE_PATCH_PATH: Final = REMOTE_PROJECT_ROOT / "patches/inkling-smoke-a015409.patch"
LLAMA_CPP_DIR: Final = Path("/opt/llama.cpp")
EVIDENCE_MOUNT: Final = Path("/evidence")
PINNED_LLAMA_CPP_COMMIT: Final = "a015409e6c27b84f60d688823d4c0126a11571fd"
PINNED_CUDA_IMAGE: Final = (
    "nvidia/cuda:13.1.2-devel-ubuntu24.04@"
    "sha256:952e42d23230610a2714c8484f38e9c934ed68e6f9c9c7fac62dcd5f98858a6e"
)
BUILD_TARGETS: Final = (
    "llama-cli",
    "llama-server",
    "llama-bench",
    "llama-perplexity",
)
SERVER_AUDIT_ENVIRONMENT: Final = {
    "IQL_SMOKE_BACKEND_AUDIT": "1",
    "IQL_SMOKE_RAW_LOGIT_AUDIT": "1",
    "LLAMA_MEDIA_MARKER": "<__media_iql_smoke_v1__>",
}
MAX_HTTP_RESPONSE_BYTES: Final = 16 * 1024 * 1024
MAX_SERVER_LOG_BYTES: Final = 64 * 1024 * 1024
ACCEPTANCE_TIMEOUT_SECONDS: Final = 120.0
MONITOR_INTERVAL_SECONDS: Final = 1.0
MONITOR_COMMAND_TIMEOUT_SECONDS: Final = 15.0
FUNCTION_TIMEOUT_SECONDS: Final = 14_400
TERMINAL_PUBLICATION_RESERVE_SECONDS: Final = 600.0
SERVER_READY_TIMEOUT_SECONDS: Final = 3_600.0
PROBE_HTTP_TIMEOUT_SECONDS: Final = 900.0
TIMING_RELATIVE_TOLERANCE: Final = 0.05
TIMING_ABSOLUTE_TOLERANCE: Final = 1.0e-3
RUN_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")

if str(LOCAL_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC_ROOT))

from inkling_quant_lab.gguf.inkling_matched import (  # noqa: E402
    InklingMatchedCellBundle,
    MatchedCapacityResult,
    load_matched_cell_bundle,
    screen_matched_capacity,
)
from inkling_quant_lab.gguf.inkling_matched_control import (  # noqa: E402
    MATCHED_ATTEMPT_REGISTRY_NAME,
    MatchedAttemptAcknowledgement,
    MatchedAttemptClaim,
    MatchedControlPlaneProvenance,
    MatchedLaunchIntent,
    MatchedPostSpawnAcceptance,
    MatchedPublicationSnapshot,
    MatchedTerminalReceiptReference,
    build_matched_terminal_receipt_reference,
    claim_matched_attempt,
    matched_app_name,
    matched_attempt_acknowledgement_path,
    matched_attempt_claim_path,
    matched_post_spawn_acceptance_path,
    matched_publication_state_path,
    strict_matched_json_object,
    validate_matched_attempt_acknowledgement,
    validate_matched_attempt_claim,
    validate_matched_control_plane_provenance,
    validate_matched_launch_intent,
    validate_matched_post_spawn_acceptance,
    validate_matched_publication_state,
    validate_matched_publication_transition,
)
from inkling_quant_lab.gguf.inkling_matched_execution import (  # noqa: E402
    MATCHED_SUBJECT_ORDER,
    MatchedArtifactHashObservation,
    MatchedCudaPeerTopologyEvidence,
    MatchedFailureCauseCode,
    MatchedFailureReceipt,
    MatchedGpuResourceEvidence,
    MatchedNvidiaSmiGpuEvidence,
    MatchedProbeEvidence,
    MatchedProbeTrialEvidence,
    MatchedPublicationState,
    MatchedSanitizedFailureDiagnostic,
    MatchedServerCleanupEvidence,
    MatchedServerCommandSpec,
    MatchedSubject,
    MatchedSubjectArtifactRehashEvidence,
    MatchedSubjectReceiptReference,
    MatchedSubjectSmokeReceipt,
    build_matched_capacity_inputs,
    build_matched_cuda_placement_policy,
    build_matched_rollup_receipt,
    build_matched_server_command,
    build_matched_server_environment,
    enumerate_matched_cuda_peer_topology,
    matched_failure_receipt_sha256,
    matched_shard_inventory_sha256,
    matched_subject_smoke_receipt_sha256,
    order_matched_nvidia_smi_identity_by_cuda_uuid,
    parse_exact_cuda_backend_audit,
    parse_matched_nvidia_smi_identity_csv,
    parse_matched_nvidia_smi_monitor_csv,
)
from inkling_quant_lab.gguf.inkling_smoke import (  # noqa: E402
    parse_artifact_load_evidence,
    parse_cuda_driver_linkage,
    parse_loader_offload_evidence,
    parse_raw_logit_audit_evidence,
    parse_server_completion,
)


class _CanonicalControlRecord(Protocol):
    def canonical_bytes(self) -> bytes: ...


class _PublicationCollisionError(RuntimeError):
    """An immutable path already contains different or unsafe bytes."""


class _PublicationUnknownError(RuntimeError):
    """Publication began, but its mounted durable result cannot be proved."""


class _EvidenceStateUnknownError(RuntimeError):
    """Installed immutable evidence cannot be reconciled with the mounted Volume."""


class _FileSizeMismatchError(RuntimeError):
    """A regular file differs from its reviewed byte count."""


_FailureCategory = Literal[
    "artifact_rehash",
    "hardware_identity",
    "hardware_capacity",
    "peer_topology",
    "server_start",
    "server_health",
    "probe",
    "backend_placement",
    "resource_monitor",
    "cleanup",
    "publication",
]


def _default_failure_cause(category: _FailureCategory) -> MatchedFailureCauseCode:
    return {
        "artifact_rehash": MatchedFailureCauseCode.ARTIFACT_CONTRACT_FAILED,
        "hardware_identity": MatchedFailureCauseCode.HARDWARE_IDENTITY_FAILED,
        "hardware_capacity": MatchedFailureCauseCode.HARDWARE_CAPACITY_FAILED,
        "peer_topology": MatchedFailureCauseCode.PEER_TOPOLOGY_FAILED,
        "server_start": MatchedFailureCauseCode.SERVER_START_FAILED,
        "server_health": MatchedFailureCauseCode.SERVER_HEALTH_FAILED,
        "probe": MatchedFailureCauseCode.COMPLETION_CONTRACT_FAILED,
        "backend_placement": MatchedFailureCauseCode.BACKEND_PLACEMENT_FAILED,
        "resource_monitor": MatchedFailureCauseCode.RESOURCE_MONITOR_FAILED,
        "cleanup": MatchedFailureCauseCode.CLEANUP_FAILED,
        "publication": MatchedFailureCauseCode.PUBLICATION_FAILED,
    }[category]


class _MatchedStageError(RuntimeError):
    """A bounded stage identity with no raw provider or artifact detail."""

    def __init__(
        self,
        category: _FailureCategory,
        *,
        cause_code: MatchedFailureCauseCode | None = None,
        artifact_path: str | None = None,
    ) -> None:
        super().__init__(f"matched smoke stage failed: {category}")
        self.category = category
        self.cause_code = _default_failure_cause(category) if cause_code is None else cause_code
        if artifact_path is not None and (
            artifact_path.startswith("/")
            or "\\" in artifact_path
            or "\x00" in artifact_path
            or any(part in {"", ".", ".."} for part in artifact_path.split("/"))
        ):
            raise ValueError("matched failure artifact path is not safe and relative")
        self.artifact_path = artifact_path


def _remaining_work_timeout(
    work_deadline_monotonic: float,
    maximum_seconds: float,
    category: _FailureCategory,
) -> float:
    """Clamp one blocking operation to the work budget before terminal reserve."""

    if (
        not math.isfinite(work_deadline_monotonic)
        or not math.isfinite(maximum_seconds)
        or maximum_seconds <= 0.0
    ):
        raise ValueError("matched work deadline inputs are invalid")
    remaining = work_deadline_monotonic - time.monotonic()
    if remaining <= 0.0:
        raise _MatchedStageError(
            category,
            cause_code=MatchedFailureCauseCode.DEADLINE_EXHAUSTED,
        )
    return min(maximum_seconds, remaining)


def _json_value(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _execution_json_bytes(value: object) -> bytes:
    """Serialize execution receipts without a trailing line feed."""

    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity_json_bytes(value: object) -> bytes:
    """Serialize supporting identity inputs with exactly one line feed."""

    return _execution_json_bytes(value) + b"\n"


def _identity_sha256(domain: str, value: object) -> str:
    return hashlib.sha256(domain.encode("ascii") + _identity_json_bytes(value)).hexdigest()


def _utc_microseconds() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("evidence path is not a contained canonical relative path")
    return path


def _mounted_path(relative: str) -> Path:
    return EVIDENCE_MOUNT / Path(_safe_relative_path(relative).as_posix())


def _resolved_evidence_mount() -> Path:
    """Resolve the platform-owned mount root without trusting child symlinks."""

    mount = EVIDENCE_MOUNT.absolute()
    try:
        mount_metadata = os.lstat(mount)
        resolved = mount.resolve(strict=True)
        resolved_metadata = os.lstat(resolved)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise RuntimeError("evidence mount is missing or unsafe") from error
    if not (
        stat.S_ISDIR(mount_metadata.st_mode) or stat.S_ISLNK(mount_metadata.st_mode)
    ) or not stat.S_ISDIR(resolved_metadata.st_mode):
        raise RuntimeError("evidence mount is missing or unsafe")
    return resolved


def _create_safe_evidence_parent(path: Path) -> None:
    """Create contained ancestors while rejecting symlinks and non-directories."""

    logical_mount = EVIDENCE_MOUNT.absolute()
    parent = path.parent.absolute()
    if not parent.is_relative_to(logical_mount):
        raise RuntimeError("evidence parent escapes its mounted root")
    current = _resolved_evidence_mount()
    for part in parent.relative_to(logical_mount).parts:
        current /= part
        with suppress(FileExistsError):
            current.mkdir()
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise RuntimeError("evidence path has an unreadable ancestor") from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("evidence path has a symlink or non-directory ancestor")


def _read_regular_bytes(path: Path, *, maximum_bytes: int = 16 * 1024 * 1024) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("mounted evidence is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(maximum_bytes + 1)
    finally:
        os.close(descriptor)
    if len(payload) > maximum_bytes:
        raise RuntimeError("mounted evidence exceeds its byte limit")
    return payload


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Use the reviewed Volume-v1 same-directory rename contract."""

    if source.parent != destination.parent:
        raise RuntimeError("immutable evidence rename must remain in one directory")
    try:
        os.lstat(destination)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(destination)
    os.rename(source, destination)


def _atomic_install_exact(path: Path, payload: bytes) -> None:
    """Install immutable bytes atomically, accepting only an identical retry."""

    _create_safe_evidence_parent(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(16)}.tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            _rename_noreplace(temporary, path)
        except FileExistsError:
            try:
                observed = _read_regular_bytes(path, maximum_bytes=len(payload))
            except BaseException as error:
                raise _EvidenceStateUnknownError(
                    "existing immutable evidence cannot be validated"
                ) from error
            if observed != payload:
                raise _PublicationCollisionError(
                    "immutable path exists with different bytes"
                ) from None
            return
        except BaseException as error:
            raise _EvidenceStateUnknownError(
                "immutable evidence rename has an unknown result"
            ) from error
        try:
            if _read_regular_bytes(path, maximum_bytes=len(payload)) != payload:
                raise RuntimeError("installed immutable evidence failed readback")
            _fsync_directory(path.parent)
        except BaseException as error:
            raise _EvidenceStateUnknownError(
                "installed immutable evidence has an unknown durable state"
            ) from error
    finally:
        with suppress(OSError):
            temporary.unlink()


def _write_once(path: Path, payload: bytes) -> None:
    """Write immutable evidence through the one atomic installation primitive."""

    _atomic_install_exact(path, payload)


def _commit_and_verify(files: Mapping[str, bytes]) -> None:
    """Commit and prove exact mounted bytes with one bounded missing-file retry."""

    if not files:
        raise ValueError("durable publication requires at least one file")
    ordered = tuple(
        sorted(
            (
                _safe_relative_path(relative).as_posix(),
                expected,
            )
            for relative, expected in files.items()
        )
    )
    missing: set[str] = set()
    observed_errors: list[BaseException] = []
    for _cycle in (1, 2):
        for relative, expected in ordered:
            if relative in missing:
                try:
                    _write_once(_mounted_path(relative), expected)
                except BaseException as error:
                    observed_errors.append(error)
                    unknown = _EvidenceStateUnknownError(
                        "missing evidence could not be reinstalled safely"
                    )
                    for observed_error in observed_errors:
                        unknown.add_note(
                            "Observed error type: "
                            f"{type(observed_error).__module__}."
                            f"{type(observed_error).__qualname__}"
                        )
                    raise unknown from error
        missing.clear()
        try:
            evidence_volume.commit()
        except BaseException as error:
            observed_errors.append(error)
        try:
            evidence_volume.reload()
        except BaseException as error:
            observed_errors.append(error)
            continue
        all_exact = True
        for relative, expected in ordered:
            try:
                observed = _read_regular_bytes(
                    _mounted_path(relative),
                    maximum_bytes=len(expected),
                )
            except FileNotFoundError:
                missing.add(relative)
                all_exact = False
                continue
            except BaseException as error:
                observed_errors.append(error)
                unknown = _EvidenceStateUnknownError("reloaded evidence cannot be validated safely")
                for observed_error in observed_errors:
                    unknown.add_note(
                        "Observed error type: "
                        f"{type(observed_error).__module__}."
                        f"{type(observed_error).__qualname__}"
                    )
                raise unknown from error
            if observed != expected:
                raise _PublicationCollisionError(
                    "reloaded evidence differs from installed immutable bytes"
                )
        if all_exact:
            return
    unknown = _EvidenceStateUnknownError("installed evidence has an unknown committed state")
    for observed_error in observed_errors:
        unknown.add_note(
            "Observed error type: "
            f"{type(observed_error).__module__}."
            f"{type(observed_error).__qualname__}"
        )
    raise unknown


def _publish_control_records(files: Mapping[str, _CanonicalControlRecord]) -> None:
    payloads = {relative: value.canonical_bytes() for relative, value in files.items()}
    for relative, payload in payloads.items():
        _write_once(_mounted_path(relative), payload)
    _commit_and_verify(payloads)


def _load_local_deployment() -> tuple[InklingMatchedCellBundle, str, Path]:
    bundle = load_matched_cell_bundle(LOCAL_PROJECT_ROOT)
    expected_control = os.environ.get("IQL_MATCHED_CONTROL_PLANE_SHA256")
    provenance_path_text = os.environ.get("IQL_MATCHED_CONTROL_PLANE_PROVENANCE_PATH")
    if (
        not isinstance(expected_control, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_control) is None
        or not provenance_path_text
    ):
        raise RuntimeError(
            "Deploy this paid runner only through scripts/manage_inkling_matched_modal.py"
        )
    provenance_path = Path(provenance_path_text)
    provenance = MatchedControlPlaneProvenance.model_validate(
        strict_matched_json_object(provenance_path.read_bytes())
    )
    if provenance.canonical_bytes() != provenance_path.read_bytes():
        raise RuntimeError("local control-plane provenance is not canonical")
    files = {item.path: (LOCAL_PROJECT_ROOT / item.path).read_bytes() for item in provenance.files}
    validate_matched_control_plane_provenance(
        provenance,
        reviewed_commit_sha=provenance.reviewed_commit_sha,
        reviewed_tree_sha=provenance.reviewed_tree_sha,
        files=files,
        required_paths=tuple(item.path for item in provenance.files),
    )
    if provenance.control_plane_sha256 != expected_control:
        raise RuntimeError("local deployment control-plane hash drifted")
    return bundle, expected_control, provenance_path


if modal.is_local():
    _LOCAL_BUNDLE, _CONTROL_SHA256, _LOCAL_PROVENANCE = _load_local_deployment()
else:
    _LOCAL_BUNDLE = load_matched_cell_bundle(REMOTE_PROJECT_ROOT)
    _CONTROL_SHA256 = os.environ["IQL_MATCHED_CONTROL_PLANE_SHA256"]
    _LOCAL_PROVENANCE = REMOTE_PROVENANCE_PATH

app = modal.App(matched_app_name(_CONTROL_SHA256))

baseline_volume = (
    modal.Volume.from_name(
        "inkling-work-v1",
        environment_name="inkling-quant",
        create_if_missing=False,
        version=1,
    )
    .with_mount_options(sub_path=_LOCAL_BUNDLE.config.storage.bf16_run_subpath)
    .read_only()
)
final_volume = (
    modal.Volume.from_name(
        "inkling-final-v1",
        environment_name="inkling-quant",
        create_if_missing=False,
        version=1,
    )
    .with_mount_options(sub_path=_LOCAL_BUNDLE.config.storage.final_run_subpath)
    .read_only()
)
source_volume = (
    modal.Volume.from_name(
        "inkling-source-v1",
        environment_name="inkling-quant",
        create_if_missing=False,
        version=1,
    )
    .with_mount_options(sub_path=_LOCAL_BUNDLE.config.storage.source_run_subpath)
    .read_only()
)
evidence_volume = modal.Volume.from_name(
    "inkling-matched-evidence-v1",
    environment_name="inkling-quant",
    create_if_missing=True,
    version=1,
)

matched_image = (
    modal.Image.from_registry(PINNED_CUDA_IMAGE, add_python="3.12")
    .apt_install(
        "build-essential",
        "ca-certificates",
        "cmake",
        "git",
        "ninja-build",
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "src/inkling_quant_lab"),
        str(REMOTE_PROJECT_ROOT / "src/inkling_quant_lab"),
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc", "**/*.pyo"],
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "configs/experiments"),
        str(REMOTE_PROJECT_ROOT / "configs/experiments"),
        copy=True,
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "scripts"),
        str(REMOTE_PROJECT_ROOT / "scripts"),
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc", "**/*.pyo"],
    )
    .add_local_file(
        str(LOCAL_PROJECT_ROOT / "patches/inkling-smoke-a015409.patch"),
        str(REMOTE_PATCH_PATH),
        copy=True,
    )
    .add_local_file(
        str(_LOCAL_PROVENANCE),
        str(REMOTE_PROVENANCE_PATH),
        copy=True,
    )
    .run_commands(
        f"git init {LLAMA_CPP_DIR}",
        f"git -C {LLAMA_CPP_DIR} remote add origin https://github.com/danielhanchen/llama.cpp.git",
        f"git -C {LLAMA_CPP_DIR} fetch --depth 1 origin {PINNED_LLAMA_CPP_COMMIT}",
        f"git -C {LLAMA_CPP_DIR} checkout --detach FETCH_HEAD",
        f"git -C {LLAMA_CPP_DIR} apply --check {REMOTE_PATCH_PATH}",
        f"git -C {LLAMA_CPP_DIR} apply {REMOTE_PATCH_PATH}",
        "python -m pip install --no-cache-dir pydantic==2.13.4 PyYAML==6.0.3",
        "mkdir -p /opt/iql-cuda-driver-link",
        "ln -s /usr/local/cuda/lib64/stubs/libcuda.so /opt/iql-cuda-driver-link/libcuda.so.1",
        (
            f"cmake -S {LLAMA_CPP_DIR} -B {LLAMA_CPP_DIR}/build -G Ninja "
            "-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF "
            "-DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF "
            "-DCMAKE_CUDA_ARCHITECTURES=103 "
            "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/iql-cuda-driver-link "
            "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE"
        ),
        f"cmake --build {LLAMA_CPP_DIR}/build --parallel 16 --target " + " ".join(BUILD_TARGETS),
        "unlink /opt/iql-cuda-driver-link/libcuda.so.1",
        "rmdir /opt/iql-cuda-driver-link",
        f'test "$(git -C {LLAMA_CPP_DIR} rev-parse HEAD)" = "{PINNED_LLAMA_CPP_COMMIT}"',
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_PROJECT_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "IQL_MATCHED_CONTROL_PLANE_SHA256": _CONTROL_SHA256,
        }
    )
)


def _validate_remote_control_plane(expected_sha256: str) -> MatchedControlPlaneProvenance:
    payload = _read_regular_bytes(REMOTE_PROVENANCE_PATH)
    provenance = MatchedControlPlaneProvenance.model_validate(strict_matched_json_object(payload))
    files = {
        item.path: _read_regular_bytes(
            REMOTE_PROJECT_ROOT / item.path,
            maximum_bytes=item.size_bytes,
        )
        for item in provenance.files
    }
    observed = validate_matched_control_plane_provenance(
        payload,
        reviewed_commit_sha=provenance.reviewed_commit_sha,
        reviewed_tree_sha=provenance.reviewed_tree_sha,
        files=files,
        required_paths=tuple(item.path for item in provenance.files),
    )
    if observed.control_plane_sha256 != expected_sha256:
        raise RuntimeError("deployed control plane differs from the authorized hash")
    return observed


def _current_invocation_ids() -> tuple[str, str, str]:
    call_id = modal.current_function_call_id()
    input_id = modal.current_input_id()
    task_id = os.environ.get("MODAL_TASK_ID")
    if (
        not isinstance(call_id, str)
        or re.fullmatch(r"fc-[A-Za-z0-9]+", call_id) is None
        or not isinstance(input_id, str)
        or re.fullmatch(r"in-[A-Za-z0-9]+(?::[0-9]+-[0-9]+)?", input_id) is None
        or not isinstance(task_id, str)
        or re.fullmatch(r"ta-[A-Za-z0-9]+", task_id) is None
    ):
        raise RuntimeError("Modal invocation identity is unavailable")
    return call_id, input_id, task_id


def _wait_for_acceptance(
    intent: MatchedLaunchIntent,
    *,
    call_id: str,
) -> tuple[MatchedPostSpawnAcceptance, str, str]:
    relative = matched_post_spawn_acceptance_path(intent.run_id, intent.intent_sha256())
    path = _mounted_path(relative)
    deadline = time.monotonic() + ACCEPTANCE_TIMEOUT_SECONDS
    while True:
        evidence_volume.reload()
        try:
            payload = _read_regular_bytes(path)
            break
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise RuntimeError("post-spawn acceptance was not published in time") from None
            time.sleep(0.25)
    raw = strict_matched_json_object(payload)
    observed = MatchedPostSpawnAcceptance.model_validate(raw)
    expected = MatchedPostSpawnAcceptance(
        accepted_at_utc=observed.accepted_at_utc,
        run_id=intent.run_id,
        launch_intent_sha256=intent.intent_sha256(),
        call_id=call_id,
        deployment=intent.deployment,
        matched_config_sha256=intent.reviewed_inputs.matched_config_sha256,
        control_plane_sha256=intent.reviewed_inputs.control_plane_sha256,
    )
    validate_matched_post_spawn_acceptance(
        payload,
        expected=expected,
        acceptance_sha256=observed.acceptance_sha256(),
        evidence_path=relative,
    )
    return observed, relative, observed.acceptance_sha256()


def _prepare_attempt(
    intent: MatchedLaunchIntent,
    acceptance_path: str,
    acceptance_sha256: str,
    invocation_ids: tuple[str, str, str],
) -> _PreparedAttempt:
    """Prepare exact claim records without consuming the atomic attempt."""

    call_id, input_id, task_id = invocation_ids
    registry = modal.Dict.from_id(intent.deployment.attempt_registry_id)
    registry.hydrate()
    info = registry.info()
    created_value = cast(object, info.created_at)
    if isinstance(created_value, datetime):
        if created_value.tzinfo is None:
            raise RuntimeError("sealed attempt registry creation time has no time zone")
        created_datetime = created_value.astimezone(UTC)
    elif isinstance(created_value, (int, float)) and not isinstance(created_value, bool):
        created_datetime = datetime.fromtimestamp(float(created_value), UTC)
    else:
        raise RuntimeError("sealed attempt registry creation time is unavailable")
    created_at = created_datetime.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if (
        registry.object_id != intent.deployment.attempt_registry_id
        or info.name != MATCHED_ATTEMPT_REGISTRY_NAME
        or created_at != intent.deployment.attempt_registry_created_at_utc
    ):
        raise RuntimeError("sealed attempt registry identity changed")
    claim = MatchedAttemptClaim(
        registry_id=registry.object_id,
        registry_created_at_utc=created_at,
        registry_key=f"{intent.run_id}:matched_smoke",
        run_id=intent.run_id,
        call_id=call_id,
        input_id=input_id,
        task_id=task_id,
        launch_intent_sha256=intent.intent_sha256(),
        post_spawn_acceptance_path=acceptance_path,
        post_spawn_acceptance_sha256=acceptance_sha256,
        matched_config_sha256=intent.reviewed_inputs.matched_config_sha256,
        control_plane_sha256=intent.reviewed_inputs.control_plane_sha256,
    )
    claim_path = matched_attempt_claim_path(intent.run_id, claim.claim_sha256())
    acknowledgement = MatchedAttemptAcknowledgement(
        acknowledged_at_utc=_utc_microseconds(),
        run_id=intent.run_id,
        registry_key=claim.registry_key,
        attempt_claim_path=claim_path,
        attempt_claim_sha256=claim.claim_sha256(),
        call_id=call_id,
        input_id=input_id,
        task_id=task_id,
        launch_intent_sha256=intent.intent_sha256(),
        matched_config_sha256=intent.reviewed_inputs.matched_config_sha256,
        control_plane_sha256=intent.reviewed_inputs.control_plane_sha256,
    )
    acknowledgement_path = matched_attempt_acknowledgement_path(
        intent.run_id,
        acknowledgement.acknowledgement_sha256(),
    )
    return _PreparedAttempt(
        registry=registry,
        claim=claim,
        claim_path=claim_path,
        acknowledgement=acknowledgement,
        acknowledgement_path=acknowledgement_path,
    )


def _claim_attempt(prepared: _PreparedAttempt) -> None:
    """Atomically consume the one attempt without performing later I/O."""

    if claim_matched_attempt(prepared.registry, prepared.claim) != prepared.claim.claim_sha256():
        raise RuntimeError("atomic attempt claim returned the wrong identity")


def _publish_attempt_records(prepared: _PreparedAttempt) -> None:
    """Persist and validate exact claim records after the atomic claim wins."""

    _publish_control_records(prepared.records())
    validate_matched_attempt_claim(
        _read_regular_bytes(_mounted_path(prepared.claim_path)),
        expected=prepared.claim,
        claim_sha256=prepared.claim.claim_sha256(),
        evidence_path=prepared.claim_path,
    )
    validate_matched_attempt_acknowledgement(
        _read_regular_bytes(_mounted_path(prepared.acknowledgement_path)),
        expected=prepared.acknowledgement,
        acknowledgement_sha256=prepared.acknowledgement.acknowledgement_sha256(),
        evidence_path=prepared.acknowledgement_path,
    )


def _sha256_file(
    path: Path,
    *,
    expected_size: int,
    failure_category: _FailureCategory,
    work_deadline_monotonic: float,
) -> tuple[str, int]:
    _remaining_work_timeout(
        work_deadline_monotonic,
        maximum_seconds=1.0,
        category=failure_category,
    )
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("subject artifact is not a regular file")
        if before.st_size != expected_size:
            raise _FileSizeMismatchError("subject artifact size drifted")
        while chunk := os.read(descriptor, 16 * 1024 * 1024):
            digest.update(chunk)
            _remaining_work_timeout(
                work_deadline_monotonic,
                maximum_seconds=1.0,
                category=failure_category,
            )
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    if stable_identity != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError("subject artifact changed while it was hashed")
    return digest.hexdigest(), before.st_size


@dataclass(frozen=True)
class _AllocationObservation:
    """One exact eight-GPU allocation observation."""

    gpus: tuple[MatchedNvidiaSmiGpuEvidence, ...]
    topology: MatchedCudaPeerTopologyEvidence
    capacity: MatchedCapacityResult
    cuda_driver_path: str
    identity_sha256: str


@dataclass
class _PublicationTracker:
    """Caller-visible terminal publication state."""

    snapshot: MatchedPublicationSnapshot
    mounted_reload_completed: bool = False


@dataclass(frozen=True)
class _PreparedAttempt:
    """Exact attempt records prepared before the atomic one-use claim."""

    registry: Any
    claim: MatchedAttemptClaim
    claim_path: str
    acknowledgement: MatchedAttemptAcknowledgement
    acknowledgement_path: str

    def records(self) -> dict[str, _CanonicalControlRecord]:
        return {
            self.claim_path: self.claim,
            self.acknowledgement_path: self.acknowledgement,
        }

    def payloads(self) -> dict[str, bytes]:
        return {relative: record.canonical_bytes() for relative, record in self.records().items()}


def _observe_allocation(
    bundle: InklingMatchedCellBundle,
    *,
    work_deadline_monotonic: float,
) -> _AllocationObservation:
    """Observe the exact CUDA identity, topology, and aggregate capacity."""

    try:
        identity_csv = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=_remaining_work_timeout(
                work_deadline_monotonic,
                maximum_seconds=30.0,
                category="hardware_identity",
            ),
            shell=False,
        ).stdout
        unordered_gpus = parse_matched_nvidia_smi_identity_csv(identity_csv)
        linkage = subprocess.run(
            ["ldd", str(LLAMA_CPP_DIR / "build/bin/libggml-cuda.so")],
            check=True,
            capture_output=True,
            text=True,
            timeout=_remaining_work_timeout(
                work_deadline_monotonic,
                maximum_seconds=30.0,
                category="hardware_identity",
            ),
            shell=False,
        ).stdout
        cuda_driver_path = parse_cuda_driver_linkage(linkage)
    except _MatchedStageError:
        raise
    except BaseException as error:
        raise _MatchedStageError("hardware_identity") from error
    try:
        _remaining_work_timeout(
            work_deadline_monotonic,
            maximum_seconds=1.0,
            category="peer_topology",
        )
        topology = enumerate_matched_cuda_peer_topology(
            cuda_driver_path,
            nvidia_smi_gpus=unordered_gpus,
        )
        gpus = order_matched_nvidia_smi_identity_by_cuda_uuid(
            unordered_gpus,
            cuda_gpu_uuids=topology.gpu_uuids,
        )
    except _MatchedStageError:
        raise
    except BaseException as error:
        raise _MatchedStageError("peer_topology") from error
    try:
        _remaining_work_timeout(
            work_deadline_monotonic,
            maximum_seconds=1.0,
            category="hardware_capacity",
        )
        capacity_inputs = build_matched_capacity_inputs(gpus)
        capacity = screen_matched_capacity(
            bundle.config,
            bundle.bf16,
            bundle.q3,
            observed_gpu_memory_bytes=capacity_inputs.observed_gpu_memory_bytes,
        )
    except _MatchedStageError:
        raise
    except BaseException as error:
        raise _MatchedStageError("hardware_capacity") from error
    identity_sha256 = _identity_sha256(
        "inkling-matched-allocation-identity-v1\n",
        {
            "gpus": [gpu.model_dump(mode="json") for gpu in gpus],
            "peer_topology": topology.model_dump(mode="json"),
            "capacity": capacity.model_dump(mode="json"),
            "cuda_driver_path": cuda_driver_path,
        },
    )
    return _AllocationObservation(
        gpus=gpus,
        topology=topology,
        capacity=capacity,
        cuda_driver_path=cuda_driver_path,
        identity_sha256=identity_sha256,
    )


def _observe_runtime_identity(
    bundle: InklingMatchedCellBundle,
    *,
    work_deadline_monotonic: float,
) -> str:
    """Rehash all reviewed runtime binaries and return their identity."""

    observed_binaries: list[dict[str, object]] = []
    try:
        for binary in bundle.config.runtime.binaries:
            digest, size = _sha256_file(
                Path(binary.path),
                expected_size=binary.size_bytes,
                failure_category="hardware_identity",
                work_deadline_monotonic=work_deadline_monotonic,
            )
            if size != binary.size_bytes or digest != binary.sha256:
                raise RuntimeError("runtime binary identity drifted")
            observed_binaries.append(
                {
                    "name": binary.name,
                    "sha256": digest,
                    "size_bytes": size,
                }
            )
    except _MatchedStageError:
        raise
    except BaseException as error:
        raise _MatchedStageError("hardware_identity") from error
    return _identity_sha256(
        "inkling-matched-runtime-identity-v1\n",
        {
            "runtime": bundle.config.runtime.model_dump(mode="json"),
            "binaries": observed_binaries,
        },
    )


def _probe_control_identity(bundle: InklingMatchedCellBundle) -> str:
    """Bind the checked prompts and output-vocabulary contract."""

    return _identity_sha256(
        "inkling-matched-probe-control-v1\n",
        {
            "probes": [probe.model_dump(mode="json") for probe in bundle.config.probes],
            "output_vocabulary": bundle.config.output_vocabulary.model_dump(mode="json"),
        },
    )


def _observe_artifact(
    *,
    subject: MatchedSubject,
    kind: str,
    artifact: Any,
    absolute_path: str,
    work_deadline_monotonic: float,
    shard_ordinal: int | None = None,
) -> MatchedArtifactHashObservation:
    try:
        digest, size = _sha256_file(
            Path(absolute_path),
            expected_size=artifact.size_bytes,
            failure_category="artifact_rehash",
            work_deadline_monotonic=work_deadline_monotonic,
        )
    except _FileSizeMismatchError as error:
        raise _MatchedStageError(
            "artifact_rehash",
            cause_code=MatchedFailureCauseCode.ARTIFACT_SIZE_MISMATCH,
            artifact_path=artifact.path,
        ) from error
    except _MatchedStageError as error:
        raise _MatchedStageError(
            "artifact_rehash",
            cause_code=error.cause_code,
            artifact_path=artifact.path,
        ) from error
    except BaseException as error:
        raise _MatchedStageError(
            "artifact_rehash",
            cause_code=MatchedFailureCauseCode.ARTIFACT_READ_FAILED,
            artifact_path=artifact.path,
        ) from error
    if size != artifact.size_bytes:
        raise _MatchedStageError(
            "artifact_rehash",
            cause_code=MatchedFailureCauseCode.ARTIFACT_SIZE_MISMATCH,
            artifact_path=artifact.path,
        )
    if digest != artifact.sha256:
        raise _MatchedStageError(
            "artifact_rehash",
            cause_code=MatchedFailureCauseCode.ARTIFACT_HASH_MISMATCH,
            artifact_path=artifact.path,
        )
    return MatchedArtifactHashObservation(
        subject=subject,
        kind=kind,
        relative_path=artifact.path,
        absolute_path=absolute_path,
        shard_ordinal=shard_ordinal,
        expected_sha256=artifact.sha256,
        observed_sha256=digest,
        expected_size_bytes=artifact.size_bytes,
        observed_size_bytes=size,
        hash_matches=True,
        size_matches=True,
    )


@dataclass(frozen=True)
class _ArtifactHashJob:
    subject: MatchedSubject
    kind: str
    artifact: Any
    absolute_path: str
    shard_ordinal: int | None = None
    work_deadline_monotonic: float = 0.0


def _run_artifact_hash_job(job: _ArtifactHashJob) -> MatchedArtifactHashObservation:
    return _observe_artifact(
        subject=job.subject,
        kind=job.kind,
        artifact=job.artifact,
        absolute_path=job.absolute_path,
        work_deadline_monotonic=job.work_deadline_monotonic,
        shard_ordinal=job.shard_ordinal,
    )


def _rehash_subject(
    bundle: InklingMatchedCellBundle,
    subject: MatchedSubject,
    *,
    work_deadline_monotonic: float,
) -> MatchedSubjectArtifactRehashEvidence:
    if subject is MatchedSubject.BF16:
        shard_pairs = zip(
            bundle.bf16.bf16_shards,
            bundle.paths.bf16_shards,
            strict=True,
        )
        jobs = [
            _ArtifactHashJob(
                subject,
                "text_shard",
                artifact,
                absolute,
                index,
                work_deadline_monotonic,
            )
            for index, (artifact, absolute) in enumerate(shard_pairs, start=1)
        ]
        jobs.append(
            _ArtifactHashJob(
                subject,
                "receipt",
                bundle.bf16.conversion_receipt,
                bundle.paths.bf16_conversion_receipt,
                work_deadline_monotonic=work_deadline_monotonic,
            )
        )
        reference_sha256 = bundle.bf16.reference_sha256
        expected_inventory: str = bundle.bf16.bf16_inventory_sha256
    else:
        shard_pairs = zip(
            bundle.q3.q3_shards,
            bundle.paths.q3_shards,
            strict=True,
        )
        jobs = [
            _ArtifactHashJob(
                subject,
                "text_shard",
                artifact,
                absolute,
                index,
                work_deadline_monotonic,
            )
            for index, (artifact, absolute) in enumerate(shard_pairs, start=1)
        ]
        jobs.extend(
            (
                _ArtifactHashJob(
                    subject,
                    "projector",
                    bundle.q3.projector,
                    bundle.paths.shared_projector,
                    work_deadline_monotonic=work_deadline_monotonic,
                ),
                _ArtifactHashJob(
                    subject,
                    "manifest",
                    bundle.q3.export_manifest,
                    bundle.paths.q3_export_manifest,
                    work_deadline_monotonic=work_deadline_monotonic,
                ),
                _ArtifactHashJob(
                    subject,
                    "receipt",
                    bundle.q3.verify_receipt,
                    bundle.paths.q3_verify_receipt,
                    work_deadline_monotonic=work_deadline_monotonic,
                ),
                _ArtifactHashJob(
                    subject,
                    "receipt",
                    bundle.q3.quantize_receipt,
                    bundle.paths.q3_quantize_receipt,
                    work_deadline_monotonic=work_deadline_monotonic,
                ),
                _ArtifactHashJob(
                    subject,
                    "receipt",
                    bundle.q3.mmproj_receipt,
                    bundle.paths.projector_conversion_receipt,
                    work_deadline_monotonic=work_deadline_monotonic,
                ),
            )
        )
        jobs.extend(
            _ArtifactHashJob(
                subject,
                "tokenizer",
                artifact,
                absolute,
                work_deadline_monotonic=work_deadline_monotonic,
            )
            for artifact, absolute in zip(
                bundle.config.tokenizer_assets,
                bundle.paths.tokenizer_assets,
                strict=True,
            )
        )
        reference_sha256 = bundle.q3.reference_sha256
        expected_inventory = bundle.q3.q3_inventory_sha256
    projector_job = _ArtifactHashJob(
        MatchedSubject.Q3,
        "projector",
        bundle.q3.projector,
        bundle.paths.shared_projector,
        work_deadline_monotonic=work_deadline_monotonic,
    )
    if subject is MatchedSubject.BF16:
        jobs.append(projector_job)

    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="iql-rehash") as executor:
        observations = tuple(executor.map(_run_artifact_hash_job, jobs))
    projector = next(item for item in observations if item.kind == "projector")
    assignments = tuple(
        item
        for item in observations
        if not (subject is MatchedSubject.BF16 and item.kind == "projector")
    )
    shards = tuple(item for item in assignments if item.kind == "text_shard")
    observed_inventory = matched_shard_inventory_sha256(subject, shards)
    return MatchedSubjectArtifactRehashEvidence(
        schema_version="inkling-matched-artifact-rehash-v1",
        subject=subject,
        subject_reference_sha256=reference_sha256,
        assignments=assignments,
        assignment_count=len(assignments),
        text_shard_count=49,
        text_shard_total_bytes=sum(item.observed_size_bytes for item in shards),
        expected_text_shard_inventory_sha256=expected_inventory,
        observed_text_shard_inventory_sha256=observed_inventory,
        first_shard_path=shards[0].absolute_path,
        metadata_only_first_shard=True,
        shared_projector=projector,
        rehash_completed=True,
        all_hashes_match=True,
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


def _checkerboard_png() -> bytes:
    rows = bytearray()
    for y in range(16):
        rows.append(0)
        for x in range(16):
            value = 255 if (x + y) % 2 == 0 else 0
            rows.extend((value, value, value))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(bytes(rows), level=9))
        + _png_chunk(b"IEND", b"")
    )


def _silence_wav() -> bytes:
    payload = io.BytesIO()
    with wave.open(payload, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(b"\x00\x00" * 4_000)
    return payload.getvalue()


def _fixture_bytes(fixture: str) -> bytes | None:
    if fixture == "none":
        return None
    if fixture == "synthetic_rgb8_png_16x16_checkerboard_v1":
        return _checkerboard_png()
    if fixture == "synthetic_pcm_s16le_wav_16000hz_mono_silence_250ms_v1":
        return _silence_wav()
    raise RuntimeError("unsupported checked matched probe fixture")


def _strict_http_json_object(raw: bytes) -> dict[str, Any]:
    if not raw or b"\x00" in raw:
        raise RuntimeError("llama-server returned empty or NUL-containing JSON")

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("llama-server returned invalid strict JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("llama-server response was not one JSON object")
    return value


def _http_json(
    port: int,
    method: str,
    path: str,
    *,
    body: object | None,
    timeout: float,
) -> tuple[dict[str, Any], str]:
    encoded = None if body is None else _execution_json_bytes(body)
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=encoded,
        headers={"Content-Type": "application/json"} if encoded is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"llama-server returned HTTP {error.code}") from None
    if len(raw) > MAX_HTTP_RESPONSE_BYTES:
        raise RuntimeError("llama-server response exceeded its evidence limit")
    value = _strict_http_json_object(raw)
    return value, hashlib.sha256(raw).hexdigest()


def _wait_ready(
    process: subprocess.Popen[bytes],
    port: int,
    *,
    work_deadline_monotonic: float,
) -> float:
    started = time.monotonic()
    deadline = min(
        started + SERVER_READY_TIMEOUT_SECONDS,
        work_deadline_monotonic,
    )
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited before becoming ready")
        try:
            payload, _ = _http_json(
                port,
                "GET",
                "/health",
                body=None,
                timeout=_remaining_work_timeout(
                    deadline,
                    maximum_seconds=5.0,
                    category="server_health",
                ),
            )
            if payload.get("status") == "ok":
                return time.monotonic() - started
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
            pass
        time.sleep(
            _remaining_work_timeout(
                deadline,
                maximum_seconds=2.0,
                category="server_health",
            )
        )
    raise RuntimeError("llama-server did not become ready before the load deadline")


def _model_properties(
    port: int,
    *,
    work_deadline_monotonic: float,
) -> tuple[int, str]:
    props, _ = _http_json(
        port,
        "GET",
        "/props",
        body=None,
        timeout=_remaining_work_timeout(
            work_deadline_monotonic,
            maximum_seconds=30.0,
            category="server_health",
        ),
    )
    modalities = props.get("modalities")
    marker = props.get("media_marker")
    if (
        not isinstance(modalities, Mapping)
        or modalities.get("vision") is not True
        or modalities.get("audio") is not True
        or not isinstance(marker, str)
        or not marker
    ):
        raise RuntimeError("llama-server did not expose the checked multimodal contract")
    build_info = props.get("build_info")
    if not isinstance(build_info, str) or PINNED_LLAMA_CPP_COMMIT[:7] not in build_info:
        raise RuntimeError("llama-server build info does not bind the pinned commit")
    models, _ = _http_json(
        port,
        "GET",
        "/v1/models",
        body=None,
        timeout=_remaining_work_timeout(
            work_deadline_monotonic,
            maximum_seconds=30.0,
            category="server_health",
        ),
    )
    data = models.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], Mapping):
        raise RuntimeError("llama-server model metadata has the wrong shape")
    meta = data[0].get("meta")
    if not isinstance(meta, Mapping) or type(meta.get("n_vocab")) is not int:
        raise RuntimeError("llama-server model metadata has no valid vocabulary")
    return int(meta["n_vocab"]), marker


def _completion_logprobs(payload: Mapping[str, Any]) -> tuple[float, ...]:
    probabilities = payload.get("completion_probabilities")
    if not isinstance(probabilities, list):
        raise RuntimeError("completion lacks retained probability evidence")
    values = tuple(float(item["logprob"]) for item in probabilities if isinstance(item, Mapping))
    if len(values) != len(probabilities) or any(
        not math.isfinite(value) or value > 0.0 for value in values
    ):
        raise RuntimeError("completion probability evidence is not finite")
    return values


def _positive_timing_number(timings: Mapping[str, Any], field: str) -> float:
    value = timings.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RuntimeError(f"llama-server timing {field} is not finite and positive")
    return float(value)


def _exact_nonnegative_timing_integer(
    timings: Mapping[str, Any],
    field: str,
) -> int:
    value = timings.get(field)
    if type(value) is not int or value < 0:
        raise RuntimeError(f"llama-server timing {field} is not a nonnegative integer")
    return value


def _require_timing_ratio(
    observed: float,
    expected: float,
    *,
    field: str,
) -> None:
    if not math.isclose(
        observed,
        expected,
        rel_tol=TIMING_RELATIVE_TOLERANCE,
        abs_tol=TIMING_ABSOLUTE_TOLERANCE,
    ):
        raise RuntimeError(f"llama-server timing {field} is internally inconsistent")


def _completion_timings(
    payload: Mapping[str, Any],
    *,
    tokens_predicted: int,
) -> tuple[float, float]:
    timings = payload.get("timings")
    if not isinstance(timings, Mapping):
        raise RuntimeError("completion lacks timing evidence")
    prompt_n = _exact_nonnegative_timing_integer(timings, "prompt_n")
    predicted_n = _exact_nonnegative_timing_integer(timings, "predicted_n")
    cache_n = _exact_nonnegative_timing_integer(timings, "cache_n")
    if prompt_n <= 0 or predicted_n != tokens_predicted or cache_n != 0:
        raise RuntimeError("completion timing token counts differ from the probe contract")
    prompt_ms = _positive_timing_number(timings, "prompt_ms")
    prompt_per_token_ms = _positive_timing_number(timings, "prompt_per_token_ms")
    prompt_per_second = _positive_timing_number(timings, "prompt_per_second")
    predicted_ms = _positive_timing_number(timings, "predicted_ms")
    predicted_per_token_ms = _positive_timing_number(
        timings,
        "predicted_per_token_ms",
    )
    predicted_per_second = _positive_timing_number(
        timings,
        "predicted_per_second",
    )
    _require_timing_ratio(
        prompt_per_token_ms,
        prompt_ms / prompt_n,
        field="prompt_per_token_ms",
    )
    _require_timing_ratio(
        prompt_per_second,
        1000.0 * prompt_n / prompt_ms,
        field="prompt_per_second",
    )
    _require_timing_ratio(
        predicted_per_token_ms,
        predicted_ms / predicted_n,
        field="predicted_per_token_ms",
    )
    _require_timing_ratio(
        predicted_per_second,
        1000.0 * predicted_n / predicted_ms,
        field="predicted_per_second",
    )
    return prompt_ms, predicted_ms


def _validate_completion_envelope(
    payload: Mapping[str, Any],
    *,
    tokens_predicted: int,
) -> None:
    if payload.get("error") is not None:
        raise RuntimeError("llama-server completion returned an error object")
    if (
        payload.get("stop") is not True
        or payload.get("truncated") is not False
        or type(payload.get("index")) is not int
        or payload.get("index") != 0
        or type(payload.get("id_slot")) is not int
        or cast(int, payload["id_slot"]) < 0
    ):
        raise RuntimeError("llama-server completion envelope is invalid")
    stop_type = payload.get("stop_type")
    if stop_type not in {"limit", "eos", "word"}:
        raise RuntimeError("llama-server completion stop type is invalid")
    tokens_evaluated = payload.get("tokens_evaluated")
    if type(tokens_evaluated) is not int or tokens_evaluated <= 0:
        raise RuntimeError("llama-server completion token evaluation count is invalid")
    if not 1 <= tokens_predicted <= 8:
        raise RuntimeError("llama-server completion token count exceeds the probe bound")


def _run_probe(
    port: int,
    probe: Any,
    marker: str,
    vocab_size: int,
    unpadded_vocab_size: int,
    *,
    work_deadline_monotonic: float,
) -> MatchedProbeEvidence:
    fixture = _fixture_bytes(probe.fixture)
    prompt: object = probe.prompt
    if fixture is not None:
        prompt = {
            "prompt_string": f"{marker}\n{probe.prompt}",
            "multimodal_data": [base64.b64encode(fixture).decode("ascii")],
        }
    request = {
        "prompt": prompt,
        "seed": probe.seed,
        "temperature": probe.temperature,
        "n_predict": probe.n_predict,
        "n_probs": probe.n_probs,
        "post_sampling_probs": probe.post_sampling_probs,
        "stream": False,
        "cache_prompt": False,
        "return_tokens": True,
        "timings_per_token": True,
    }
    trials: list[MatchedProbeTrialEvidence] = []
    for trial_index in (1, 2):
        payload, response_sha256 = _http_json(
            port,
            "POST",
            "/completion",
            body=request,
            timeout=_remaining_work_timeout(
                work_deadline_monotonic,
                maximum_seconds=PROBE_HTTP_TIMEOUT_SECONDS,
                category="probe",
            ),
        )
        completion = parse_server_completion(
            payload,
            vocab_size=vocab_size,
            unpadded_vocab_size=unpadded_vocab_size,
            expected_n_probs=probe.n_probs,
        )
        _validate_completion_envelope(
            payload,
            tokens_predicted=completion.tokens_predicted,
        )
        logprobs = _completion_logprobs(payload)
        prompt_ms, decode_ms = _completion_timings(
            payload,
            tokens_predicted=completion.tokens_predicted,
        )
        trials.append(
            MatchedProbeTrialEvidence(
                trial_index=trial_index,
                token_ids=completion.token_ids,
                generated_token_count=completion.tokens_predicted,
                minimum_logprob=min(logprobs),
                maximum_logprob=max(logprobs),
                mean_logprob=sum(logprobs) / len(logprobs),
                prompt_processing_ms=prompt_ms,
                decode_ms=decode_ms,
                response_sha256=response_sha256,
                finite_logits=True,
                valid_token_ids=True,
            )
        )
    if trials[0].token_ids != trials[1].token_ids:
        raise _MatchedStageError(
            "probe",
            cause_code=MatchedFailureCauseCode.GREEDY_REPEATABILITY_FAILED,
        )
    return MatchedProbeEvidence(
        probe_id=probe.probe_id,
        modality=probe.modality,
        prompt_sha256=probe.prompt_sha256,
        fixture_sha256=None if fixture is None else hashlib.sha256(fixture).hexdigest(),
        fixture_size_bytes=None if fixture is None else len(fixture),
        seed=probe.seed,
        temperature=probe.temperature,
        n_predict=probe.n_predict,
        n_probs=probe.n_probs,
        usable_vocab_size=unpadded_vocab_size,
        trials=(trials[0], trials[1]),
        repeatable_greedy_token_ids=True,
        prompt_text_recorded=False,
        output_text_recorded=False,
    )


class _RuntimeMonitor:
    def __init__(self, pid: int, gpu_uuids: Sequence[str]) -> None:
        self._pid = pid
        self._gpu_uuids = tuple(gpu_uuids)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._samples = 0
        self._peak_rss_mib = 0
        self._peak_memory = [0] * 8
        self._peak_utilization = [0] * 8
        self._error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self) -> None:
        self._thread.join(timeout=MONITOR_COMMAND_TIMEOUT_SECONDS + 5)
        if self._thread.is_alive():
            raise RuntimeError("resource monitor did not stop")
        if self._error is not None:
            raise RuntimeError("resource monitor failed") from self._error

    def evidence(self) -> MatchedGpuResourceEvidence:
        return MatchedGpuResourceEvidence(
            schema_version="inkling-matched-resource-evidence-v1",
            sampling_interval_seconds=MONITOR_INTERVAL_SECONDS,
            sample_count=self._samples,
            server_peak_host_rss_mib=self._peak_rss_mib,
            gpu_peak_memory_used_mib=tuple(self._peak_memory),
            gpu_peak_utilization_percent=tuple(self._peak_utilization),
        )

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._sample()
                self._stop_event.wait(MONITOR_INTERVAL_SECONDS)
        except BaseException as error:
            self._error = error

    def _sample(self) -> None:
        status_path = Path(f"/proc/{self._pid}/status")
        if status_path.is_file():
            match = re.search(
                r"^VmRSS:\s+([0-9]+)\s+kB$",
                status_path.read_text(encoding="utf-8"),
                re.MULTILINE,
            )
            if match is not None:
                self._peak_rss_mib = max(self._peak_rss_mib, int(match.group(1)) // 1024)
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=MONITOR_COMMAND_TIMEOUT_SECONDS,
            shell=False,
        ).stdout
        rows = parse_matched_nvidia_smi_monitor_csv(
            output,
            expected_uuids=self._gpu_uuids,
        )
        for index, row in enumerate(rows):
            self._peak_memory[index] = max(self._peak_memory[index], row.memory_used_mib)
            self._peak_utilization[index] = max(
                self._peak_utilization[index],
                row.utilization_percent,
            )
        self._samples += 1


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    termination_error: BaseException | None = None
    try:
        process.terminate()
    except BaseException as error:
        termination_error = error
    if termination_error is None:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        except BaseException as error:
            termination_error = error
        else:
            return
    try:
        process.kill()
        process.wait(timeout=30)
    except BaseException as kill_error:
        if termination_error is not None:
            termination_error.add_note(
                "Forced server kill also failed with "
                f"{type(kill_error).__module__}.{type(kill_error).__qualname__}"
            )
            raise termination_error from kill_error
        raise
    if isinstance(termination_error, (KeyboardInterrupt, SystemExit, GeneratorExit)):
        raise termination_error


def _stop_subject_runtime(
    monitor: _RuntimeMonitor,
    process: subprocess.Popen[bytes],
) -> None:
    try:
        monitor.stop()
    finally:
        try:
            monitor.join()
        finally:
            _terminate_process(process)


def _merge_cleanup_error(
    primary: BaseException | None,
    cleanup_error: BaseException,
) -> BaseException:
    """Retain the primary failure while recording only a safe cleanup type."""

    if primary is None:
        wrapped = _MatchedStageError("cleanup")
        wrapped.__cause__ = cleanup_error
        return wrapped
    primary.add_note(
        "Cleanup also failed with "
        f"{type(cleanup_error).__module__}.{type(cleanup_error).__qualname__}"
    )
    return primary


def _close_log_handle(
    log_handle: Any,
    primary: BaseException | None,
) -> BaseException | None:
    """Close one server log without replacing an existing stage failure."""

    try:
        log_handle.close()
    except BaseException as error:
        return _merge_cleanup_error(primary, error)
    return primary


def _read_server_log(path: Path) -> bytes:
    return _read_regular_bytes(path, maximum_bytes=MAX_SERVER_LOG_BYTES)


def _audit_log_text(log_payload: bytes) -> str:
    text = log_payload.decode("utf-8", errors="strict")
    retained_markers = (
        "ggml_cuda_init: found ",
        "load_tensors: offloading output layer to GPU",
        "load_tensors: offloaded ",
        "llama_model_loader: loaded meta data ",
        "llama_model_loader: additional ",
        "srv load_model: loaded multimodal model, ",
        "warning: no usable GPU found",
    )
    lines = [
        line
        for line in text.splitlines()
        if "IQL_SMOKE_" in line or any(marker in line for marker in retained_markers)
    ]
    retained = "\n".join(lines)
    if len(retained) > 16 * 1024 * 1024:
        raise RuntimeError("retained server audit evidence exceeds its parser bound")
    return retained


def _run_subject(
    *,
    subject: MatchedSubject,
    bundle: InklingMatchedCellBundle,
    rehash: MatchedSubjectArtifactRehashEvidence,
    gpu_uuids: Sequence[str],
    allocation_identity_sha256: str,
    runtime_identity_sha256: str,
    probe_control_sha256: str,
    run_id: str,
    work_deadline_monotonic: float,
) -> MatchedSubjectSmokeReceipt:
    port = 18_080 if subject is MatchedSubject.BF16 else 18_081
    log_path = Path(f"/tmp/inkling-matched-{subject.value}-llama-server.log")
    spec = MatchedServerCommandSpec(
        schema_version="inkling-matched-server-command-v1",
        subject=subject,
        server_binary="/opt/llama.cpp/build/bin/llama-server",
        first_shard_path=rehash.first_shard_path,
        projector_path=rehash.shared_projector.absolute_path,
        host="127.0.0.1",
        port=port,
        server_log_path=str(log_path),
        endpoint="/completion",
        log_verbosity=4,
        context_size=8192,
    )
    command = build_matched_server_command(spec)
    log_path.unlink(missing_ok=True)
    try:
        log_handle = log_path.open("xb")
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=build_matched_server_environment(
                    os.environ,
                    audit_environment=SERVER_AUDIT_ENVIRONMENT,
                ),
                shell=False,
            )
        except BaseException as spawn_error:
            try:
                log_handle.close()
            except BaseException as close_error:
                spawn_error.add_note(
                    "Server log close also failed with "
                    f"{type(close_error).__module__}."
                    f"{type(close_error).__qualname__}"
                )
            raise
    except _MatchedStageError:
        raise
    except BaseException as error:
        raise _MatchedStageError("server_start") from error
    monitor = _RuntimeMonitor(process.pid, gpu_uuids)
    try:
        monitor.start()
    except BaseException as error:
        primary = _MatchedStageError("resource_monitor")
        primary.__cause__ = error
        try:
            _terminate_process(process)
        except BaseException as cleanup_error:
            primary = cast(
                _MatchedStageError,
                _merge_cleanup_error(primary, cleanup_error),
            )
        primary = cast(
            _MatchedStageError,
            _close_log_handle(log_handle, primary),
        )
        raise primary from error
    load_time = 0.0
    probes: tuple[MatchedProbeEvidence, ...] = ()
    subject_error: BaseException | None = None
    try:
        try:
            try:
                load_time = _wait_ready(
                    process,
                    port,
                    work_deadline_monotonic=work_deadline_monotonic,
                )
                vocab_size, media_marker = _model_properties(
                    port,
                    work_deadline_monotonic=work_deadline_monotonic,
                )
            except _MatchedStageError:
                raise
            except BaseException as error:
                raise _MatchedStageError("server_health") from error
            output_vocabulary = bundle.config.output_vocabulary
            if vocab_size != output_vocabulary.vocab_size:
                raise _MatchedStageError("server_health")
            try:
                probes = tuple(
                    _run_probe(
                        port,
                        probe,
                        media_marker,
                        vocab_size,
                        output_vocabulary.unpadded_vocab_size,
                        work_deadline_monotonic=work_deadline_monotonic,
                    )
                    for probe in bundle.config.probes
                )
            except _MatchedStageError:
                raise
            except BaseException as error:
                raise _MatchedStageError("probe") from error
        except BaseException as error:
            subject_error = error
    finally:
        try:
            _stop_subject_runtime(monitor, process)
        except BaseException as error:
            subject_error = _merge_cleanup_error(subject_error, error)
        finally:
            subject_error = _close_log_handle(log_handle, subject_error)
    if subject_error is not None:
        raise subject_error
    try:
        resources = monitor.evidence()
    except _MatchedStageError:
        raise
    except BaseException as error:
        raise _MatchedStageError("resource_monitor") from error
    try:
        log_payload = _read_server_log(log_path)
        log_text = _audit_log_text(log_payload)
        loader = parse_loader_offload_evidence(log_text, expected_gpu_count=8)
        artifact_load = parse_artifact_load_evidence(
            log_text,
            expected_first_shard_path=rehash.first_shard_path,
            expected_projector_path=rehash.shared_projector.absolute_path,
        )
        expected_vectors = sum(
            trial.generated_token_count for probe in probes for trial in probe.trials
        )
        raw_logits = parse_raw_logit_audit_evidence(
            log_text,
            expected_generated_token_vectors=expected_vectors,
            vocab_size=output_vocabulary.vocab_size,
            unpadded_vocab_size=output_vocabulary.unpadded_vocab_size,
        )
        backend = parse_exact_cuda_backend_audit(
            log_text,
            policy=build_matched_cuda_placement_policy(bundle.config),
        )
    except _MatchedStageError:
        raise
    except BaseException as error:
        raise _MatchedStageError("backend_placement") from error
    payload: dict[str, object] = {
        "schema_version": "inkling-matched-subject-smoke-v1",
        "status": "passed",
        "stage": "matched_smoke",
        "run_id": run_id,
        "subject": subject,
        "subject_ordinal": MATCHED_SUBJECT_ORDER.index(subject),
        "allocation_identity_sha256": allocation_identity_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "probe_control_sha256": probe_control_sha256,
        "server_process_id": process.pid,
        "server_command": command,
        "server_log_sha256": hashlib.sha256(log_payload).hexdigest(),
        "load_time_seconds": load_time,
        "artifact_rehash": rehash,
        "loader_offload": loader,
        "artifact_load": artifact_load,
        "raw_logit_audit": raw_logits,
        "backend_audit": backend,
        "probes": probes,
        "resources": resources,
        "cleanup": MatchedServerCleanupEvidence(
            monitor_stopped_before_server=True,
            monitor_joined=True,
            server_stopped=True,
            process_exit_observed=True,
            log_scan_completed=True,
        ),
        "prompt_text_recorded": False,
        "output_text_recorded": False,
        "raw_server_log_recorded": False,
        "quality_measured": False,
        "benchmark_measured": False,
        "completed_at_utc": _utc_microseconds(),
    }
    payload["receipt_sha256"] = matched_subject_smoke_receipt_sha256(payload)
    return MatchedSubjectSmokeReceipt.model_validate(payload)


def _subject_reference(receipt: MatchedSubjectSmokeReceipt) -> MatchedSubjectReceiptReference:
    projector = receipt.artifact_rehash.shared_projector
    return MatchedSubjectReceiptReference(
        subject=receipt.subject,
        subject_ordinal=receipt.subject_ordinal,
        receipt_sha256=receipt.receipt_sha256,
        server_process_id=receipt.server_process_id,
        allocation_identity_sha256=receipt.allocation_identity_sha256,
        runtime_identity_sha256=receipt.runtime_identity_sha256,
        probe_control_sha256=receipt.probe_control_sha256,
        projector_path=projector.absolute_path,
        projector_sha256=projector.observed_sha256,
    )


def _subject_receipt_path(receipt: MatchedSubjectSmokeReceipt) -> str:
    return f"runs/{receipt.run_id}/subjects/{receipt.subject.value}/{receipt.receipt_sha256}.json"


def _validate_subject_receipt_bytes(
    payload: bytes,
    *,
    expected: MatchedSubjectSmokeReceipt,
) -> MatchedSubjectSmokeReceipt:
    observed = MatchedSubjectSmokeReceipt.model_validate(strict_matched_json_object(payload))
    if _execution_json_bytes(observed) != payload or observed != expected:
        raise RuntimeError("persisted subject receipt differs from its exact bytes")
    return observed


def _persist_subject_receipt(
    receipt: MatchedSubjectSmokeReceipt,
) -> MatchedSubjectReceiptReference:
    """Persist and prove one subject receipt before the next subject can start."""

    relative = _subject_receipt_path(receipt)
    payload = _execution_json_bytes(receipt)
    _write_once(_mounted_path(relative), payload)
    _commit_and_verify({relative: payload})
    observed = _read_regular_bytes(
        _mounted_path(relative),
        maximum_bytes=len(payload),
    )
    _validate_subject_receipt_bytes(observed, expected=receipt)
    return _subject_reference(receipt)


def _verify_persisted_subject_receipts(
    receipts: Sequence[MatchedSubjectSmokeReceipt],
) -> None:
    """Prove all subject receipts together in one freshly reloaded Volume view."""

    expected_files = {
        _subject_receipt_path(receipt): _execution_json_bytes(receipt) for receipt in receipts
    }
    _commit_and_verify(expected_files)
    for receipt in receipts:
        relative = _subject_receipt_path(receipt)
        payload = expected_files[relative]
        observed = _read_regular_bytes(
            _mounted_path(relative),
            maximum_bytes=len(payload),
        )
        _validate_subject_receipt_bytes(observed, expected=receipt)


def _publication_snapshot(
    *,
    publication_id: str,
    run_id: str,
    claim_sha256: str,
    status: Literal["not_started", "installing", "confirmed", "unknown"],
    cycle: int,
    terminal: MatchedTerminalReceiptReference | None,
    reloaded: bool,
) -> MatchedPublicationSnapshot:
    return MatchedPublicationSnapshot(
        publication_id=publication_id,
        run_id=run_id,
        attempt_claim_sha256=claim_sha256,
        status=status,
        cycle=cycle,
        terminal_receipt=terminal,
        mounted_reload_completed=reloaded,
        runner_network_blocked=True,
        runner_volume_read_method="mounted_after_reload",
        manager_cross_container_read_required=True,
    )


def _new_publication_tracker(claim: MatchedAttemptClaim) -> _PublicationTracker:
    publication_id = _identity_sha256(
        "inkling-matched-publication-v1\n",
        {
            "run_id": claim.run_id,
            "attempt_claim_sha256": claim.claim_sha256(),
        },
    )
    not_started = _publication_snapshot(
        publication_id=publication_id,
        run_id=claim.run_id,
        claim_sha256=claim.claim_sha256(),
        status="not_started",
        cycle=0,
        terminal=None,
        reloaded=False,
    )
    not_started.canonical_bytes()
    return _PublicationTracker(snapshot=not_started)


def _mark_publication_unknown(tracker: _PublicationTracker) -> None:
    """Suppress further terminal receipts and best-effort record the ambiguity."""

    previous = tracker.snapshot
    if previous.status != "installing":
        return
    unknown = _publication_snapshot(
        publication_id=previous.publication_id,
        run_id=previous.run_id,
        claim_sha256=previous.attempt_claim_sha256,
        status="unknown",
        cycle=previous.cycle,
        terminal=previous.terminal_receipt,
        reloaded=tracker.mounted_reload_completed,
    )
    unknown_payload = unknown.canonical_bytes()
    validate_matched_publication_transition(previous, unknown)
    tracker.snapshot = unknown
    relative = matched_publication_state_path(
        unknown.run_id,
        unknown.state_sha256(),
    )
    with suppress(BaseException):
        _write_once(_mounted_path(relative), unknown_payload)
        _commit_and_verify({relative: unknown_payload})


def _publication_unknown(
    tracker: _PublicationTracker,
    error: BaseException,
) -> _PublicationUnknownError:
    _mark_publication_unknown(tracker)
    unknown = _PublicationUnknownError(
        "terminal publication started but exact mounted durability was not proved"
    )
    unknown.add_note(f"Observed error type: {type(error).__module__}.{type(error).__qualname__}")
    return unknown


def _publish_terminal(
    *,
    claim: MatchedAttemptClaim,
    publication_tracker: _PublicationTracker,
    terminal_receipt: object,
    outcome: Literal["success", "failure"],
    required_files: Mapping[str, bytes],
) -> MatchedPublicationSnapshot:
    run_id = claim.run_id
    if not required_files:
        raise ValueError("terminal publication requires attempt-control files")
    required = {
        _safe_relative_path(relative).as_posix(): payload
        for relative, payload in required_files.items()
    }
    if any(not isinstance(payload, bytes) for payload in required.values()):
        raise TypeError("terminal publication files must contain bytes")
    terminal_payload = _execution_json_bytes(terminal_receipt)
    terminal_reference = build_matched_terminal_receipt_reference(
        terminal_payload,
        run_id=run_id,
        outcome=outcome,
    )
    if publication_tracker.snapshot.status != "not_started":
        raise RuntimeError("terminal publication was already started")
    not_started = publication_tracker.snapshot
    not_started_payload = not_started.canonical_bytes()
    state0 = matched_publication_state_path(
        run_id,
        not_started.state_sha256(),
    )
    try:
        _write_once(_mounted_path(state0), not_started_payload)
        expected_files: dict[str, bytes] = {
            **required,
            state0: not_started_payload,
            terminal_reference.path: terminal_payload,
        }
        observed_errors: list[BaseException] = []
        previous = not_started
        for cycle in (1, 2):
            installing = _publication_snapshot(
                publication_id=previous.publication_id,
                run_id=run_id,
                claim_sha256=claim.claim_sha256(),
                status="installing",
                cycle=cycle,
                terminal=terminal_reference,
                reloaded=False,
            )
            installing_payload = installing.canonical_bytes()
            validate_matched_publication_transition(previous, installing)
            publication_tracker.snapshot = installing
            installing_path = matched_publication_state_path(
                run_id,
                installing.state_sha256(),
            )
            expected_files[installing_path] = installing_payload
            for relative, payload in expected_files.items():
                _write_once(_mounted_path(relative), payload)
            try:
                evidence_volume.commit()
            except BaseException as error:
                observed_errors.append(error)
            try:
                evidence_volume.reload()
            except BaseException as error:
                observed_errors.append(error)
                previous = installing
                continue
            publication_tracker.mounted_reload_completed = True

            all_exact = True
            for relative, expected in expected_files.items():
                try:
                    observed = _read_regular_bytes(
                        _mounted_path(relative),
                        maximum_bytes=len(expected),
                    )
                except FileNotFoundError:
                    all_exact = False
                    continue
                if observed != expected:
                    raise _PublicationCollisionError(
                        "terminal evidence differs after mounted Volume reload"
                    )
            if not all_exact:
                previous = installing
                continue

            confirmed = _publication_snapshot(
                publication_id=installing.publication_id,
                run_id=run_id,
                claim_sha256=claim.claim_sha256(),
                status="confirmed",
                cycle=cycle,
                terminal=terminal_reference,
                reloaded=True,
            )
            confirmed_payload = confirmed.canonical_bytes()
            validate_matched_publication_transition(installing, confirmed)
            confirmed_path = matched_publication_state_path(
                run_id,
                confirmed.state_sha256(),
            )
            _write_once(_mounted_path(confirmed_path), confirmed_payload)
            _commit_and_verify({confirmed_path: confirmed_payload})
            validate_matched_publication_state(
                _read_regular_bytes(
                    _mounted_path(confirmed_path),
                    maximum_bytes=len(confirmed_payload),
                ),
                expected=confirmed,
                state_sha256=confirmed.state_sha256(),
                evidence_path=confirmed_path,
            )
            publication_tracker.snapshot = confirmed
            return confirmed
        exhausted = _EvidenceStateUnknownError(
            "terminal evidence remained missing after two mounted reload cycles"
        )
        for observed_error in observed_errors:
            exhausted.add_note(
                "Observed error type: "
                f"{type(observed_error).__module__}."
                f"{type(observed_error).__qualname__}"
            )
        raise exhausted
    except BaseException as error:
        if publication_tracker.snapshot.status == "not_started":
            raise
        if publication_tracker.snapshot.status == "confirmed":
            return publication_tracker.snapshot
        raise _publication_unknown(publication_tracker, error) from error


def _failure_publication_state() -> MatchedPublicationState:
    return MatchedPublicationState(
        state="not_started",
        success_files_installed=False,
        commit_requested=False,
        volume_reloaded=False,
        mounted_readback_verified=False,
        terminal_success_proven=False,
        failure_receipt_allowed=True,
    )


def _failure_category(error: BaseException) -> _FailureCategory:
    if isinstance(error, _MatchedStageError):
        return error.category
    return "publication"


def _failure_cause_code(error: BaseException) -> MatchedFailureCauseCode:
    if isinstance(error, _MatchedStageError):
        return error.cause_code
    return MatchedFailureCauseCode.PUBLICATION_FAILED


def _failure_artifact_path(error: BaseException) -> str | None:
    if isinstance(error, _MatchedStageError):
        return error.artifact_path
    return None


def _failure_detail(error: BaseException) -> BaseException:
    """Return the first in-memory non-stage cause without retaining it."""

    current = error
    for _ in range(8):
        cause = current.__cause__
        if not isinstance(current, _MatchedStageError) or cause is None or cause is current:
            break
        current = cause
    return current


def _failure_type(error: BaseException) -> str:
    name = type(_failure_detail(error)).__name__
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]{0,127}", name) is None:
        return "RuntimeError"
    return name


def _failure_message_sha256(error: BaseException) -> str:
    """Hash a bounded diagnostic even when an exception cannot be stringified."""

    detail = _failure_detail(error)
    try:
        message = str(detail)
    except BaseException:
        message = f"<unprintable:{type(detail).__module__}.{type(detail).__qualname__}>"
    return hashlib.sha256(message.encode("utf-8", errors="strict")).hexdigest()


def _build_failure_receipt(
    *,
    error: BaseException,
    run_id: str,
    subject: MatchedSubject,
    completed_subject_receipts: Sequence[MatchedSubjectReceiptReference],
    allocation_identity_sha256: str,
    runtime_identity_sha256: str,
    probe_control_sha256: str,
) -> MatchedFailureReceipt:
    diagnostic = MatchedSanitizedFailureDiagnostic(
        schema_version="inkling-matched-sanitized-failure-v1",
        category=_failure_category(error),
        failure_type=_failure_type(error),
        subject=subject,
        cause_code=_failure_cause_code(error),
        artifact_path=_failure_artifact_path(error),
        message_sha256=_failure_message_sha256(error),
        raw_message_recorded=False,
        traceback_recorded=False,
        raw_server_log_recorded=False,
    )
    completed = tuple(completed_subject_receipts)
    publication = _failure_publication_state()
    completed_at_utc = _utc_microseconds()
    payload: dict[str, object] = {
        "schema_version": "inkling-matched-failure-v1",
        "status": "failed",
        "stage": "matched_smoke",
        "run_id": run_id,
        "allocation_identity_sha256": allocation_identity_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "probe_control_sha256": probe_control_sha256,
        "subject_at_failure": subject,
        "completed_subject_receipts": completed,
        "diagnostic": diagnostic,
        "publication": publication,
        "prompt_text_recorded": False,
        "output_text_recorded": False,
        "completed_at_utc": completed_at_utc,
    }
    receipt_sha256 = matched_failure_receipt_sha256(payload)
    return MatchedFailureReceipt(
        schema_version="inkling-matched-failure-v1",
        status="failed",
        stage="matched_smoke",
        run_id=run_id,
        allocation_identity_sha256=allocation_identity_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        probe_control_sha256=probe_control_sha256,
        subject_at_failure=subject,
        completed_subject_receipts=completed,
        diagnostic=diagnostic,
        publication=publication,
        prompt_text_recorded=False,
        output_text_recorded=False,
        completed_at_utc=completed_at_utc,
        receipt_sha256=receipt_sha256,
    )


@app.function(
    image=matched_image,
    gpu="B300:8",
    cpu=16,
    memory=64 * 1024,
    ephemeral_disk=512 * 1024,
    retries=0,
    timeout=FUNCTION_TIMEOUT_SECONDS,
    startup_timeout=1_800,
    max_containers=1,
    single_use_containers=True,
    block_network=True,
    volumes={
        "/baseline": baseline_volume,
        "/final": final_volume,
        "/source": source_volume,
        "/evidence": evidence_volume,
    },
)
def matched_smoke_test(run_id: str, launch_intent_sha256: str) -> dict[str, Any]:
    """Consume one authorization and run both subjects on this allocation."""

    work_deadline_monotonic = (
        time.monotonic() + FUNCTION_TIMEOUT_SECONDS - TERMINAL_PUBLICATION_RESERVE_SECONDS
    )
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("matched run ID is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", launch_intent_sha256) is None:
        raise ValueError("matched launch-intent hash is invalid")
    bundle = load_matched_cell_bundle(REMOTE_PROJECT_ROOT)
    provenance = _validate_remote_control_plane(_CONTROL_SHA256)
    evidence_volume.reload()
    intent_relative = f"runs/{run_id}/control/launch-intents/{launch_intent_sha256}.json"
    intent_payload = _read_regular_bytes(_mounted_path(intent_relative))
    intent = MatchedLaunchIntent.model_validate(strict_matched_json_object(intent_payload))
    validate_matched_launch_intent(
        intent_payload,
        expected=intent,
        launch_intent_sha256=launch_intent_sha256,
        evidence_path=intent_relative,
    )
    if (
        intent.run_id != run_id
        or intent.reviewed_inputs.control_plane_sha256 != provenance.control_plane_sha256
        or intent.reviewed_inputs.matched_config_sha256 != bundle.config.config_hash()
        or intent.subject_order != MATCHED_SUBJECT_ORDER
    ):
        raise RuntimeError("remote matched authorization differs from deployed inputs")
    invocation_ids = _current_invocation_ids()
    acceptance, acceptance_path, acceptance_sha256 = _wait_for_acceptance(
        intent,
        call_id=invocation_ids[0],
    )
    if acceptance.call_id != invocation_ids[0]:
        raise RuntimeError("post-spawn acceptance belongs to a different call")
    prepared_attempt = _prepare_attempt(
        intent,
        acceptance_path,
        acceptance_sha256,
        invocation_ids,
    )
    # Observe the identities needed by a sanitized terminal failure before the
    # irreversible Dict claim. No model process starts before the claim wins.
    initial_allocation = _observe_allocation(
        bundle,
        work_deadline_monotonic=work_deadline_monotonic,
    )
    runtime_identity_sha256 = _observe_runtime_identity(
        bundle,
        work_deadline_monotonic=work_deadline_monotonic,
    )
    probe_control_sha256 = _probe_control_identity(bundle)
    claim = prepared_attempt.claim
    attempt_files = prepared_attempt.payloads()
    publication_tracker = _new_publication_tracker(claim)
    receipts: list[MatchedSubjectSmokeReceipt] = []
    completed_references: list[MatchedSubjectReceiptReference] = []
    current_subject = MatchedSubject.BF16
    claim_won = False
    try:
        _claim_attempt(prepared_attempt)
        claim_won = True
        _publish_attempt_records(prepared_attempt)
        for subject in MATCHED_SUBJECT_ORDER:
            current_subject = subject
            try:
                rehash = _rehash_subject(
                    bundle,
                    subject,
                    work_deadline_monotonic=work_deadline_monotonic,
                )
            except _MatchedStageError:
                raise
            except BaseException as error:
                raise _MatchedStageError("artifact_rehash") from error
            subject_allocation = _observe_allocation(
                bundle,
                work_deadline_monotonic=work_deadline_monotonic,
            )
            if subject_allocation.identity_sha256 != initial_allocation.identity_sha256:
                raise _MatchedStageError("peer_topology")
            subject_runtime_identity_sha256 = _observe_runtime_identity(
                bundle,
                work_deadline_monotonic=work_deadline_monotonic,
            )
            if subject_runtime_identity_sha256 != runtime_identity_sha256:
                raise _MatchedStageError("hardware_identity")
            receipt = _run_subject(
                subject=subject,
                bundle=bundle,
                rehash=rehash,
                gpu_uuids=subject_allocation.topology.gpu_uuids,
                allocation_identity_sha256=initial_allocation.identity_sha256,
                runtime_identity_sha256=runtime_identity_sha256,
                probe_control_sha256=probe_control_sha256,
                run_id=run_id,
                work_deadline_monotonic=work_deadline_monotonic,
            )
            try:
                reference = _persist_subject_receipt(receipt)
            except _MatchedStageError:
                raise
            except BaseException as error:
                raise _MatchedStageError("publication") from error
            receipts.append(receipt)
            completed_references.append(reference)
        try:
            _remaining_work_timeout(
                work_deadline_monotonic,
                maximum_seconds=1.0,
                category="publication",
            )
            _verify_persisted_subject_receipts(receipts)
            rollup = build_matched_rollup_receipt(
                run_id=run_id,
                subject_receipts=receipts,
                completed_at_utc=_utc_microseconds(),
            )
        except _MatchedStageError:
            raise
        except BaseException as error:
            raise _MatchedStageError("publication") from error
        publication = _publish_terminal(
            claim=claim,
            publication_tracker=publication_tracker,
            terminal_receipt=rollup,
            outcome="success",
            required_files=attempt_files,
        )
        if publication.terminal_receipt is None:
            raise RuntimeError("confirmed publication has no terminal receipt")
        return {
            "schema_version": "inkling-matched-runner-result-v1",
            "status": "passed",
            "run_id": run_id,
            "terminal_receipt": publication.terminal_receipt.model_dump(mode="json"),
            "quality_measured": False,
            "benchmark_measured": False,
        }
    except _PublicationUnknownError:
        raise
    except BaseException as error:
        if claim_won and publication_tracker.snapshot.status == "not_started":
            failure = _build_failure_receipt(
                error=error,
                run_id=run_id,
                subject=current_subject,
                completed_subject_receipts=completed_references,
                allocation_identity_sha256=initial_allocation.identity_sha256,
                runtime_identity_sha256=runtime_identity_sha256,
                probe_control_sha256=probe_control_sha256,
            )
            try:
                _publish_terminal(
                    claim=claim,
                    publication_tracker=publication_tracker,
                    terminal_receipt=failure,
                    outcome="failure",
                    required_files=attempt_files,
                )
            except _PublicationUnknownError:
                raise
            except BaseException as publication_error:
                error.add_note(
                    "Failure receipt publication also failed with "
                    f"{type(publication_error).__module__}."
                    f"{type(publication_error).__qualname__}"
                )
        raise
