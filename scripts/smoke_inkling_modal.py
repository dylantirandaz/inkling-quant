"""Run one sealed, read-only Inkling GGUF inference smoke test on Modal.

Deploy this file through ``manage_inkling_smoke_modal.py``.  It mounts the
completed export read-only, writes evidence to a different Volume, and never
downloads or uploads model data.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import secrets
import shlex
import stat
import struct
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.error
import urllib.request
import wave
import zlib
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

ENTRYPOINT_PATH = Path(__file__).resolve()
LOCAL_PROJECT_ROOT = ENTRYPOINT_PATH.parents[1]
LOCAL_SRC_ROOT = LOCAL_PROJECT_ROOT / "src"
if LOCAL_SRC_ROOT.is_dir() and str(LOCAL_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC_ROOT))

import modal  # noqa: E402

EXPECTED_MODAL_VERSION: Final = "1.5.0"
if modal.__version__ != EXPECTED_MODAL_VERSION:
    raise RuntimeError(
        f"This smoke boundary requires Modal {EXPECTED_MODAL_VERSION}, got {modal.__version__}"
    )

from inkling_quant_lab.gguf.inkling import (  # noqa: E402
    InklingGGUFConfig,
    load_inkling_gguf_config,
    require_stage_billing_window,
)
from inkling_quant_lab.gguf.inkling_smoke import (  # noqa: E402
    BACKEND_FAILURE_MARKER_TOKENS,
    CUDA_DRIVER_STUB_RPATH_LINK_DEFINITION,
    INSTRUMENTATION_PATCH_RELATIVE_PATH,
    PINNED_CUDA_IMAGE,
    PINNED_CUDA_IMAGE_DIGEST,
    PINNED_LLAMA_CPP_COMMIT,
    BackendFailureDiagnosticAccumulator,
    InklingSmokeConfig,
    InklingVerifiedExportReference,
    SmokeProbeConfig,
    combine_gpu_identity,
    enumerate_cuda_driver_gpus,
    enumerate_cuda_driver_peer_topology,
    load_inkling_smoke_config,
    load_verified_export_reference,
    parse_artifact_load_evidence,
    parse_backend_audit_evidence,
    parse_cuda_driver_linkage,
    parse_loader_offload_evidence,
    parse_nvidia_smi_csv,
    parse_nvidia_smi_monitor_csv,
    parse_raw_logit_audit_evidence,
    parse_server_completion,
    redacted_smoke_config_record,
)
from inkling_quant_lab.gguf.inkling_smoke_acceptance import (  # noqa: E402
    SMOKE_POST_SPAWN_ACCEPTANCE_MAX_BYTES,
    smoke_post_spawn_acceptance_path,
    validate_smoke_post_spawn_acceptance,
)
from inkling_quant_lab.gguf.inkling_smoke_attempt import (  # noqa: E402
    SmokeAttemptRegistryClaim,
    claim_smoke_attempt,
    smoke_attempt_registry_key,
)
from inkling_quant_lab.gguf.inkling_smoke_authorization import (  # noqa: E402
    smoke_launch_intent_remote_path,
    validate_smoke_launch_intent,
)
from inkling_quant_lab.gguf.inkling_smoke_execution import (  # noqa: E402
    SMOKE_STAGE,
    SmokeControlPlaneProvenance,
    SmokeGpuTopologyEvidence,
    SmokeHostEvidence,
    SmokeLaunchAcknowledgement,
    SmokeNvidiaSmiTopologyDiagnostic,
    SmokeServerLogFailureEvidence,
    SmokeSubprocessFailureEvidence,
    canonical_python_package_inventory,
    canonical_smoke_attempt_registry_created_at_utc,
    immutable_source_tree_identity,
    parse_cgroup_cpu_quota_millicores,
    parse_cgroup_memory_limit_bytes,
    parse_dpkg_inventory,
    parse_nvcc_version,
    parse_proc_cpu_model,
    parse_proc_mem_total_bytes,
    resolve_current_process_cgroup_hierarchy_paths,
    smoke_control_plane_provenance,
    smoke_hardware_topology_sha256,
    smoke_package_manifest_sha256,
    smoke_run_id,
    smoke_terminal_receipt_sha256,
    strict_json_object,
    validate_deployed_smoke_control_plane,
    validate_smoke_failure_receipt,
    validate_smoke_launch_acknowledgement,
    validate_smoke_terminal_receipt,
)

CONFIG_PATH = LOCAL_PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_smoke_modal.yaml"
REFERENCE_PATH = LOCAL_PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_verified_export.json"
PATCH_PATH = LOCAL_PROJECT_ROOT / INSTRUMENTATION_PATCH_RELATIVE_PATH
EXPORT_CONFIG_PATH = LOCAL_PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_modal.yaml"
REMOTE_PACKAGE = Path("/root/inkling_quant_lab")
REMOTE_CONFIG = Path("/root/inkling_q3_k_m_smoke_modal.yaml")
REMOTE_REFERENCE = Path("/root/inkling_q3_k_m_verified_export.json")
REMOTE_PATCH = Path("/root/inkling-smoke-a015409.patch")
LLAMA_CPP_DIR = Path("/opt/llama.cpp")
CUDA_DRIVER_STUB_DIR = Path("/usr/local/cuda/lib64/stubs")
CUDA_DRIVER_STUB = CUDA_DRIVER_STUB_DIR / "libcuda.so"
CUDA_DRIVER_LINK_DIR = Path("/opt/iql-cuda-driver-link")
CUDA_DRIVER_LINK_SONAME = CUDA_DRIVER_LINK_DIR / "libcuda.so.1"
CUDA_RUNTIME_LIBRARY_DIR = Path("/usr/local/cuda/lib64")
SUBJECT_MOUNT = Path("/subject")
EVIDENCE_MOUNT = Path("/evidence")
SERVER_LOG = Path("/tmp/inkling-llama-server.log")
SERVER_PORT: Final = 18080
MAX_HTTP_RESPONSE_BYTES: Final = 16 * 1024 * 1024
MAX_SERVER_LOG_BYTES: Final = 64 * 1024 * 1024
MAX_FAILURE_LOG_LINE_BYTES: Final = 4 * 1024
MAX_LAUNCH_INTENT_BYTES: Final = 64 * 1024
MAX_TERMINAL_RECEIPT_BYTES: Final = 16 * 1024 * 1024
POST_SPAWN_ACCEPTANCE_TIMEOUT_SECONDS: Final = 120.0
POST_SPAWN_ACCEPTANCE_POLL_SECONDS: Final = 0.25
HASH_WORKERS: Final = 8
GPU_SAMPLE_INTERVAL_SECONDS: Final = 1.0
GPU_MONITOR_COMMAND_TIMEOUT_SECONDS: Final = 15.0
GPU_MONITOR_STOP_TIMEOUT_SECONDS: Final = GPU_MONITOR_COMMAND_TIMEOUT_SECONDS + 5.0
STORAGE_DELETION_LAG_DAYS: Final = 4
BUILD_TARGETS: Final = (
    "llama-cli",
    "llama-server",
    "llama-bench",
    "llama-perplexity",
)
ELF_RPATH_AUDIT_PATHS: Final = (
    LLAMA_CPP_DIR / "build/bin/libggml-cuda.so",
    *(LLAMA_CPP_DIR / "build/bin" / target for target in BUILD_TARGETS),
)
SERVER_REQUIRED_FLAGS: Final = (
    "--model",
    "--mmproj",
    "--host",
    "--port",
    "--ctx-size",
    "--n-gpu-layers",
    "--n-cpu-moe",
    "--split-mode",
    "--tensor-split",
    "--flash-attn",
    "--mmap",
    "--mmproj-offload",
    "--parallel",
    "--threads",
    "--threads-batch",
    "--batch-size",
    "--ubatch-size",
    "--log-verbosity",
    "--no-webui",
)
SOURCE_BLOB_PINS: Final = (
    (
        "ggml/src/ggml-backend.cpp",
        "87615921c09be5ef8c4996faa70fb3f49c385031",
    ),
    (
        "src/llama-model-loader.cpp",
        "28f8bb7934bbc807a08dc13ad58724ec77281903",
    ),
    (
        "src/llama-model-loader.h",
        "c476026d3e510ad03d3e6f0d619ecea7fc95319c",
    ),
    ("tools/mtmd/clip.cpp", "dbd07081bf73f336a17bd3b8d8359830128c424b"),
    ("tools/mtmd/mtmd.cpp", "3e81e44143fa635e56e0a757ce1ba33d34d107e4"),
    (
        "tools/server/server-context.cpp",
        "7564ad4e9cfb8e77d610e90c7530121214a4c483",
    ),
    (
        "tools/server/server.cpp",
        "20effbb14851b201118843bf14fa5bc51de1e304",
    ),
)
SOURCE_CONTRACT_ASSERTIONS: Final = (
    ("CMakeLists.txt", "option(LLAMA_BUILD_UI"),
    ("CMakeLists.txt", "option(LLAMA_USE_PREBUILT_UI"),
    ("common/arg.cpp", '{"--mmproj-offload"},'),
    ("common/log.h", "#define LOG_LEVEL_TRACE  4"),
    ("common/log.cpp", "case GGML_LOG_LEVEL_INFO:  return LOG_LEVEL_TRACE;"),
    (
        "tools/server/server-common.cpp",
        'constexpr char JSON_STRING_PROMPT_KEY[] = "prompt_string";',
    ),
    (
        "tools/server/server-common.cpp",
        'constexpr char JSON_MTMD_DATA_KEY[] = "multimodal_data";',
    ),
    ("tools/server/server-context.cpp", '{ "media_marker",                get_media_marker() },'),
    ("tools/server/server-context.cpp", '{ "build_info",                  meta->build_info },'),
    ("tools/server/server-context.cpp", '{"n_vocab",     meta->model_vocab_n_tokens},'),
    ("tools/server/server-task.cpp", 'res["completion_probabilities"] ='),
)
PATCHED_SOURCE_BLOB_PINS: Final = (
    (
        "tools/server/server-context.cpp",
        "58b90ccbecd60cb0784810224d79e70e4152b521",
    ),
    (
        "tools/server/server.cpp",
        "9be8c02497080fa57ad9460084c2337a1997f89b",
    ),
)
SERVER_AUDIT_ENVIRONMENT: Final = {
    "IQL_SMOKE_BACKEND_AUDIT": "1",
    "IQL_SMOKE_RAW_LOGIT_AUDIT": "1",
    "LLAMA_MEDIA_MARKER": "<__media_iql_smoke_v1__>",
}

DEFAULT_CONFIG = load_inkling_smoke_config(CONFIG_PATH if modal.is_local() else REMOTE_CONFIG)
DEFAULT_REFERENCE = load_verified_export_reference(
    REFERENCE_PATH if modal.is_local() else REMOTE_REFERENCE
)
if DEFAULT_CONFIG.verified_export_reference_sha256 != DEFAULT_REFERENCE.reference_sha256:
    raise RuntimeError("Checked smoke config and verified-export reference disagree")
if modal.is_local():
    billing_cycle_end_utc = os.environ.get("IQL_MODAL_BILLING_CYCLE_END_CONFIRMED")
    control_plane_sha256 = os.environ.get("IQL_MODAL_SMOKE_CONTROL_PLANE_SHA256")
    local_control_plane = smoke_control_plane_provenance(LOCAL_PROJECT_ROOT)
    if (
        os.environ.get("IQL_MODAL_WORKSPACE_BUDGET_CONFIRMED") != "800"
        or not billing_cycle_end_utc
        or control_plane_sha256 != local_control_plane.tree_sha256
    ):
        raise RuntimeError(
            "Do not run or deploy this paid App directly. Use "
            "scripts/manage_inkling_smoke_modal.py after confirming the exact "
            "workspace hard budget and billing-cycle end."
        )
    if hashlib.sha256(PATCH_PATH.read_bytes()).hexdigest() != (
        DEFAULT_CONFIG.runtime.instrumentation_patch_sha256
    ):
        raise RuntimeError("Checked llama.cpp smoke instrumentation patch SHA-256 drifted")
    export_config = load_inkling_gguf_config(EXPORT_CONFIG_PATH)
    if export_config.config_hash() != InklingGGUFConfig().config_hash():
        raise RuntimeError("Checked export YAML differs from the frozen deployment schema")
    require_stage_billing_window(
        export_config,
        billing_cycle_end_utc,
        SMOKE_STAGE,
        include_startup=True,
        invocations=1,
    )

CUDA_IMAGE_REFERENCE = f"{PINNED_CUDA_IMAGE}@{PINNED_CUDA_IMAGE_DIGEST}"
app = modal.App(f"inkling-q3-smoke-{DEFAULT_CONFIG.config_hash()[:12]}")

subject_volume = modal.Volume.from_name(
    DEFAULT_CONFIG.storage.final_volume,
    environment_name="inkling-quant",
    create_if_missing=False,
    version=1,
).with_mount_options(
    read_only=True,
    sub_path=DEFAULT_CONFIG.storage.final_run_subpath,
)
evidence_volume = modal.Volume.from_name(
    DEFAULT_CONFIG.storage.evidence_volume,
    environment_name="inkling-quant",
    create_if_missing=True,
    version=1,
)


def _python_inventory_build_command(output_path: Path) -> str:
    return (
        "python -m pip freeze --all --path "
        '"$(python -c \'import sysconfig; print(sysconfig.get_path("purelib"))\')" '
        f"| LC_ALL=C sort > {output_path}"
    )


def _elf_link_audit_build_command(paths: Sequence[Path], cuda_library: Path) -> str:
    path_values = [str(path) for path in paths]
    return (
        'python -c "import subprocess; '
        f"paths={path_values!r}; cuda={str(cuda_library)!r}; "
        f"forbidden={(str(CUDA_DRIVER_STUB_DIR), str(CUDA_DRIVER_LINK_DIR))!r}; "
        "dynamic={path: subprocess.check_output(['readelf', '-d', path], text=True) "
        "for path in paths}; "
        "assert '[libcuda.so.1]' in dynamic[cuda], "
        "'libggml-cuda.so must require libcuda.so.1'; "
        "assert all(root not in value for value in dynamic.values() for root in forbidden), "
        "'built ELF objects must not retain the CUDA driver stub path'\""
    )


smoke_image = modal.Image.from_registry(CUDA_IMAGE_REFERENCE, add_python="3.12").apt_install(
    "build-essential",
    "ca-certificates",
    "cmake",
    "git",
    "ninja-build",
)
if modal.is_local():
    smoke_image = smoke_image.add_local_file(PATCH_PATH, str(REMOTE_PATCH), copy=True)
smoke_image = smoke_image.run_commands(
    f"git init {LLAMA_CPP_DIR}",
    f"git -C {LLAMA_CPP_DIR} remote add origin {DEFAULT_CONFIG.runtime.repository}",
    f"git -C {LLAMA_CPP_DIR} fetch --depth 1 origin {DEFAULT_CONFIG.runtime.commit}",
    f"git -C {LLAMA_CPP_DIR} checkout --detach FETCH_HEAD",
    *(
        f'test "$(git -C {LLAMA_CPP_DIR} hash-object {shlex.quote(path)})" = '
        f"{shlex.quote(expected)}"
        for path, expected in SOURCE_BLOB_PINS
    ),
    *(
        f"grep -F -- {shlex.quote(snippet)} {shlex.quote(str(LLAMA_CPP_DIR / path))} > /dev/null"
        for path, snippet in SOURCE_CONTRACT_ASSERTIONS
    ),
    (
        'python -c "import hashlib,pathlib; '
        f"assert hashlib.sha256(pathlib.Path('{REMOTE_PATCH}').read_bytes()).hexdigest() "
        f"== '{DEFAULT_CONFIG.runtime.instrumentation_patch_sha256}'\""
    ),
    f"git -C {LLAMA_CPP_DIR} apply --check {REMOTE_PATCH}",
    f"git -C {LLAMA_CPP_DIR} apply {REMOTE_PATCH}",
    *(
        f'test "$(git -C {LLAMA_CPP_DIR} hash-object {shlex.quote(path)})" = '
        f"{shlex.quote(expected)}"
        for path, expected in PATCHED_SOURCE_BLOB_PINS
    ),
    f"git -C {LLAMA_CPP_DIR} diff --check",
    "python -m pip install --no-cache-dir pydantic==2.13.4 PyYAML==6.0.3",
    f"test -f {CUDA_DRIVER_STUB}",
    f"test ! -e {CUDA_DRIVER_LINK_DIR}",
    f"mkdir {CUDA_DRIVER_LINK_DIR}",
    f"ln -s {CUDA_DRIVER_STUB} {CUDA_DRIVER_LINK_SONAME}",
    (f'test "$(readlink -f {CUDA_DRIVER_LINK_SONAME})" = "$(readlink -f {CUDA_DRIVER_STUB})"'),
    (
        f"cmake -S {LLAMA_CPP_DIR} -B {LLAMA_CPP_DIR}/build -G Ninja "
        "-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF "
        "-DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF "
        "-DCMAKE_CUDA_ARCHITECTURES=103 "
        f"-D{CUDA_DRIVER_STUB_RPATH_LINK_DEFINITION}"
    ),
    (f"grep -Fx -- 'LLAMA_BUILD_UI:BOOL=OFF' {LLAMA_CPP_DIR}/build/CMakeCache.txt > /dev/null"),
    (
        f"grep -Fx -- 'LLAMA_USE_PREBUILT_UI:BOOL=OFF' "
        f"{LLAMA_CPP_DIR}/build/CMakeCache.txt > /dev/null"
    ),
    (f"cmake --build {LLAMA_CPP_DIR}/build --parallel 16 --target " + " ".join(BUILD_TARGETS)),
    f"test ! -e {LLAMA_CPP_DIR}/build/tools/ui/dist/index.html",
    f"test ! -e {LLAMA_CPP_DIR}/build/tools/ui/dist.tar.gz",
    _elf_link_audit_build_command(
        ELF_RPATH_AUDIT_PATHS,
        LLAMA_CPP_DIR / "build/bin/libggml-cuda.so",
    ),
    (
        f"LD_LIBRARY_PATH={CUDA_DRIVER_LINK_DIR}:{CUDA_RUNTIME_LIBRARY_DIR} "
        f"{LLAMA_CPP_DIR}/build/bin/llama-server --help > "
        f"{LLAMA_CPP_DIR}/.iql-llama-server-help.txt 2>&1"
    ),
    f"unlink {CUDA_DRIVER_LINK_SONAME}",
    f"rmdir {CUDA_DRIVER_LINK_DIR}",
    f"test ! -e {CUDA_DRIVER_LINK_DIR}",
    *(
        f"grep -F -- '{flag}' {LLAMA_CPP_DIR}/.iql-llama-server-help.txt > /dev/null"
        for flag in SERVER_REQUIRED_FLAGS
    ),
    f"git -C {LLAMA_CPP_DIR} rev-parse HEAD > {LLAMA_CPP_DIR}/.iql-build-commit",
    f"sha256sum {REMOTE_PATCH} > {LLAMA_CPP_DIR}/.iql-smoke-patch.sha256",
    (
        f"git -C {LLAMA_CPP_DIR} diff --binary -- "
        + " ".join(shlex.quote(path) for path, _ in SOURCE_BLOB_PINS)
        + f" | sha256sum > {LLAMA_CPP_DIR}/.iql-patched-diff.sha256"
    ),
    (
        "python -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))' > "
        f"{LLAMA_CPP_DIR}/.iql-python-purelib.txt"
    ),
    _python_inventory_build_command(LLAMA_CPP_DIR / ".iql-python-freeze.txt"),
    (
        "dpkg-query -W -f='${binary:Package}=${Version}\\n' | LC_ALL=C sort > "
        f"{LLAMA_CPP_DIR}/.iql-dpkg-inventory.txt"
    ),
    *(
        f"sha256sum {LLAMA_CPP_DIR}/build/bin/{target} > {LLAMA_CPP_DIR}/.iql-{target}.sha256"
        for target in BUILD_TARGETS
    ),
).env(
    {
        "PYTHONPATH": "/root",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
)
if modal.is_local():
    smoke_image = smoke_image.add_local_dir(
        LOCAL_PROJECT_ROOT / "src/inkling_quant_lab",
        str(REMOTE_PACKAGE),
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc", "**/*.pyo", "**/.DS_Store"],
    )
    smoke_image = smoke_image.add_local_file(CONFIG_PATH, str(REMOTE_CONFIG), copy=True)
    smoke_image = smoke_image.add_local_file(
        REFERENCE_PATH,
        str(REMOTE_REFERENCE),
        copy=True,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_child(root: Path, *parts: str) -> Path:
    if any(
        not part
        or "\x00" in part
        or "\\" in part
        or part in {".", ".."}
        or Path(part).is_absolute()
        or ".." in Path(part).parts
        for part in parts
    ):
        raise RuntimeError("Unsafe evidence path")
    path = root.joinpath(*parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("Evidence path escapes its run root")
    return path


def _resolved_evidence_mount() -> Path:
    """Resolve the platform-owned mount root without trusting child symlinks."""

    mount = EVIDENCE_MOUNT.absolute()
    try:
        mount_metadata = os.lstat(mount)
        resolved = mount.resolve(strict=True)
        resolved_metadata = os.lstat(resolved)
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise RuntimeError("Evidence mount is missing or unsafe") from error
    if not (
        stat.S_ISDIR(mount_metadata.st_mode) or stat.S_ISLNK(mount_metadata.st_mode)
    ) or not stat.S_ISDIR(resolved_metadata.st_mode):
        raise RuntimeError("Evidence mount is missing or unsafe")
    return resolved


def _create_safe_evidence_parent(path: Path) -> None:
    logical_mount = EVIDENCE_MOUNT.absolute()
    parent = path.parent.absolute()
    if not parent.is_relative_to(logical_mount):
        raise RuntimeError("Evidence parent escapes its mounted root")
    current = _resolved_evidence_mount()
    for part in parent.relative_to(logical_mount).parts:
        current /= part
        with suppress(FileExistsError):
            current.mkdir()
        try:
            metadata = os.lstat(current)
        except OSError as error:
            raise RuntimeError("Evidence path has an unreadable ancestor") from error
        if not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError("Evidence path has a symlink or non-directory ancestor")


def _read_existing_regular_bytes(
    path: Path,
    *,
    max_bytes: int | None = None,
) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("Immutable evidence destination is not a regular file")
        if max_bytes is not None and metadata.st_size > max_bytes:
            raise RuntimeError("Immutable evidence exceeds its size limit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
        if max_bytes is not None and len(payload) > max_bytes:
            raise RuntimeError("Immutable evidence exceeds its size limit")
        return payload
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Publish one file with the Volume v1 same-directory rename contract."""

    if source.parent != destination.parent:
        raise RuntimeError("Immutable evidence rename must remain in one directory")
    try:
        os.lstat(destination)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(destination)
    os.rename(source, destination)


def _atomic_bytes(path: Path, payload: bytes, *, allow_identical: bool) -> None:
    _create_safe_evidence_parent(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(16)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            _rename_noreplace(temporary, path)
        except FileExistsError:
            try:
                existing = _read_existing_regular_bytes(path)
            except BaseException as error:
                raise _DurablePublicationStateUnknownError(
                    "Existing immutable evidence has an unknown state"
                ) from error
            if allow_identical and existing == payload:
                return
            raise RuntimeError(f"Refusing to replace immutable evidence at {path.name}") from None
        except BaseException as error:
            raise _DurablePublicationStateUnknownError(
                "Immutable evidence rename has an unknown result"
            ) from error
        try:
            if _read_existing_regular_bytes(path) != payload:
                raise RuntimeError(f"Published immutable evidence failed read-back at {path.name}")
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException as error:
            raise _DurablePublicationStateUnknownError(
                "Installed immutable evidence has an unknown durable state"
            ) from error
    finally:
        with suppress(OSError):
            temporary.unlink()


def _atomic_json(path: Path, value: Mapping[str, Any], *, allow_identical: bool) -> None:
    _atomic_bytes(
        path,
        (_canonical_json(value) + "\n").encode("utf-8"),
        allow_identical=allow_identical,
    )


class _DurablePublicationStateUnknownError(RuntimeError):
    """The runner cannot prove whether installed evidence is durable."""


def _read_reloaded_volume_bytes(remote_path: str) -> bytes | None:
    """Read one exact file after the mounted evidence Volume reloads."""

    if not remote_path or remote_path.startswith("/"):
        raise RuntimeError("Durable publication path must be relative")
    path = _safe_child(EVIDENCE_MOUNT, *remote_path.split("/"))
    try:
        return _read_existing_regular_bytes(
            path,
            max_bytes=MAX_TERMINAL_RECEIPT_BYTES,
        )
    except FileNotFoundError:
        return None


def _commit_and_reconcile_volume_files(
    expected_files: Mapping[str, bytes],
    *,
    event: str,
    run_id: str,
) -> None:
    """Commit installed files and verify them through the reloaded mount."""

    if not expected_files:
        raise ValueError("Durable publication requires at least one installed file")
    ordered_files = tuple(sorted(expected_files.items()))
    if any(not remote_path or remote_path.startswith("/") for remote_path, _ in ordered_files):
        raise ValueError("Durable publication paths must be relative")
    mounted_paths = {
        remote_path: _safe_child(EVIDENCE_MOUNT, *remote_path.split("/"))
        for remote_path, _ in ordered_files
    }
    observed_errors: list[BaseException] = []
    missing_after_reload: set[str] = set()
    for commit_sequence in (1, 2):
        if missing_after_reload:
            for remote_path, payload in ordered_files:
                if remote_path not in missing_after_reload:
                    continue
                try:
                    _atomic_bytes(
                        mounted_paths[remote_path],
                        payload,
                        allow_identical=True,
                    )
                except BaseException as error:
                    observed_errors.append(error)
                    unknown = _DurablePublicationStateUnknownError(
                        "Missing evidence could not be reinstalled safely"
                    )
                    for observed_error in observed_errors:
                        unknown.add_note(
                            "Observed during durable publication: "
                            f"{type(observed_error).__module__}."
                            f"{type(observed_error).__qualname__}"
                        )
                    raise unknown from error
            missing_after_reload.clear()
        try:
            evidence_volume.commit()
        except BaseException as error:
            observed_errors.append(error)
        try:
            evidence_volume.reload()
        except BaseException as error:
            observed_errors.append(error)
            continue
        all_persisted = True
        for remote_path, payload in ordered_files:
            try:
                persisted = _read_reloaded_volume_bytes(remote_path)
            except BaseException as error:
                observed_errors.append(error)
                all_persisted = False
                continue
            if persisted is None:
                all_persisted = False
                missing_after_reload.add(remote_path)
                continue
            if persisted != payload:
                unknown = _DurablePublicationStateUnknownError(
                    "Committed evidence differs from the installed bytes"
                )
                for observed_error in observed_errors:
                    unknown.add_note(
                        "Observed during durable publication: "
                        f"{type(observed_error).__module__}."
                        f"{type(observed_error).__qualname__}"
                    )
                raise unknown
        if all_persisted:
            if observed_errors:
                with suppress(BaseException):
                    print(
                        _canonical_json(
                            {
                                "event": event,
                                "run_id": run_id,
                                "stage": SMOKE_STAGE,
                                "commit_sequence": commit_sequence,
                                "readback": "mounted_volume_after_reload",
                                "observed_error_types": [
                                    f"{type(observed_error).__module__}."
                                    f"{type(observed_error).__qualname__}"
                                    for observed_error in observed_errors
                                ],
                            }
                        ),
                        file=sys.stderr,
                    )
            return
    unknown = _DurablePublicationStateUnknownError(
        "Installed evidence has an unknown committed state"
    )
    for observed_error in observed_errors:
        unknown.add_note(
            "Observed during durable publication: "
            f"{type(observed_error).__module__}."
            f"{type(observed_error).__qualname__}"
        )
    raise unknown


def _publish_success_receipt(
    success_path: Path,
    receipt: Mapping[str, Any],
    *,
    run_id: str,
) -> None:
    """Install one success receipt and reconcile ambiguous Volume commits."""

    payload = (_canonical_json(receipt) + "\n").encode("utf-8")
    if len(payload) > MAX_TERMINAL_RECEIPT_BYTES:
        raise RuntimeError("Terminal success receipt exceeds its size limit")
    try:
        _atomic_bytes(success_path, payload, allow_identical=False)
        remote_path = success_path.relative_to(EVIDENCE_MOUNT).as_posix()
        _commit_and_reconcile_volume_files(
            {remote_path: payload},
            event="success_commit_reconciled",
            run_id=run_id,
        )
    except _DurablePublicationStateUnknownError:
        raise
    except BaseException as error:
        raise _DurablePublicationStateUnknownError(
            "Terminal success publication has an unknown committed state"
        ) from error


def _publish_failure_receipt(
    failure_path: Path,
    receipt: Mapping[str, Any],
    *,
    run_id: str,
) -> None:
    """Install one failure receipt and reconcile ambiguous Volume commits."""

    payload = (_canonical_json(receipt) + "\n").encode("utf-8")
    if len(payload) > MAX_TERMINAL_RECEIPT_BYTES:
        raise RuntimeError("Terminal failure receipt exceeds its size limit")
    _atomic_bytes(failure_path, payload, allow_identical=True)
    remote_path = failure_path.relative_to(EVIDENCE_MOUNT).as_posix()
    _commit_and_reconcile_volume_files(
        {remote_path: payload},
        event="failure_commit_reconciled",
        run_id=run_id,
    )


def _remote_config(config_json: str) -> InklingSmokeConfig:
    config = InklingSmokeConfig.model_validate_json(config_json)
    embedded = load_inkling_smoke_config(REMOTE_CONFIG)
    if config.config_hash() != DEFAULT_CONFIG.config_hash() or config != embedded:
        raise RuntimeError("Remote smoke config differs from the checked deployment config")
    return config


def _remote_reference(reference_json: str) -> InklingVerifiedExportReference:
    reference = InklingVerifiedExportReference.model_validate_json(reference_json)
    embedded = load_verified_export_reference(REMOTE_REFERENCE)
    if reference != embedded or reference.reference_sha256 != DEFAULT_REFERENCE.reference_sha256:
        raise RuntimeError("Remote verified-export reference differs from checked bytes")
    return reference


def _require_stage_window(config: InklingSmokeConfig, cycle_end_text: str) -> None:
    cycle_end = datetime.strptime(cycle_end_text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    required = timedelta(
        days=STORAGE_DELETION_LAG_DAYS,
        hours=config.resources.max_hours,
    )
    if datetime.now(UTC) + required > cycle_end:
        raise RuntimeError("The smoke body and storage-deletion window cross the cycle end")


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
        raise RuntimeError("Modal invocation identity is unavailable at the evidence boundary")
    return call_id, input_id, task_id


def _sealed_attempt_registry(
    config: InklingSmokeConfig,
    acknowledgement: SmokeLaunchAcknowledgement,
) -> Any:
    """Resolve and verify the exact modern Modal Dict sealed by the manager."""

    deployment = acknowledgement.deployment
    if (
        config.storage.attempt_registry_append_only is not True
        or deployment.attempt_registry_name != config.storage.attempt_registry
    ):
        raise RuntimeError("The smoke attempt registry configuration is invalid")
    registry = modal.Dict.from_id(deployment.attempt_registry_id)
    registry.hydrate()
    info = registry.info()
    try:
        created_at_utc = canonical_smoke_attempt_registry_created_at_utc(info.created_at)
    except (TypeError, ValueError) as error:
        raise RuntimeError("The sealed smoke attempt Dict is unsupported") from error
    observed = (
        registry.object_id,
        info.name,
        created_at_utc,
    )
    expected = (
        deployment.attempt_registry_id,
        deployment.attempt_registry_name,
        deployment.attempt_registry_created_at_utc,
    )
    if observed != expected:
        raise RuntimeError("The sealed smoke attempt Dict identity changed")
    return registry


def _wait_for_post_spawn_acceptance(
    *,
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    acknowledgement: SmokeLaunchAcknowledgement,
    run_id: str,
    launch_intent_sha256: str,
) -> tuple[tuple[str, str, str], str, str]:
    """Wait for the manager to bind the launch to this exact Modal call."""

    invocation_ids = _current_invocation_ids()
    call_id, _input_id, _task_id = invocation_ids
    remote_path = smoke_post_spawn_acceptance_path(run_id, launch_intent_sha256)
    acceptance_path = _safe_child(EVIDENCE_MOUNT, *remote_path.split("/"))
    deadline = time.monotonic() + POST_SPAWN_ACCEPTANCE_TIMEOUT_SECONDS
    while True:
        evidence_volume.reload()
        try:
            payload = _read_existing_regular_bytes(acceptance_path)
        except FileNotFoundError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "The exact Modal call did not receive post-spawn acceptance"
                ) from None
            time.sleep(min(POST_SPAWN_ACCEPTANCE_POLL_SECONDS, remaining))
            continue
        except OSError as error:
            raise RuntimeError("Post-spawn acceptance evidence is unsafe") from error
        break

    if len(payload) > SMOKE_POST_SPAWN_ACCEPTANCE_MAX_BYTES:
        raise RuntimeError("Post-spawn acceptance evidence exceeds its size limit")
    try:
        raw = strict_json_object(payload)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Post-spawn acceptance evidence is invalid JSON") from error
    accepted_at = raw.get("accepted_at")
    if not isinstance(accepted_at, str):
        raise RuntimeError("Post-spawn acceptance evidence has no canonical time")
    acceptance_sha256 = hashlib.sha256(payload).hexdigest()
    deployment = acknowledgement.deployment
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
        raise RuntimeError("Post-spawn acceptance does not bind this exact Modal call") from error
    run_prefix = f"runs/{run_id}/"
    if not remote_path.startswith(run_prefix):
        raise RuntimeError("Post-spawn acceptance path differs from its run")
    return invocation_ids, remote_path.removeprefix(run_prefix), acceptance_sha256


def _bind_run_inputs(
    run_root: Path,
    *,
    config: InklingSmokeConfig,
    reference_json: str,
    control_plane_json: str,
) -> None:
    _atomic_bytes(
        _safe_child(run_root, "resolved_config.json"),
        (_canonical_json(redacted_smoke_config_record(config)) + "\n").encode(),
        allow_identical=True,
    )
    _atomic_bytes(
        _safe_child(run_root, "verified_export_reference.json"),
        (reference_json + "\n").encode(),
        allow_identical=True,
    )
    _atomic_bytes(
        _safe_child(run_root, "control_plane.json"),
        (control_plane_json + "\n").encode(),
        allow_identical=True,
    )


def _begin_only_attempt(
    run_root: Path,
    *,
    config: InklingSmokeConfig,
    control_plane: SmokeControlPlaneProvenance,
    acknowledgement: SmokeLaunchAcknowledgement,
    attempt_registry: Any,
    run_id: str,
    launch_intent_sha256: str,
    invocation_ids: tuple[str, str, str],
    post_spawn_acceptance_path: str,
    post_spawn_acceptance_sha256: str,
    invocation_state: dict[str, Any],
) -> dict[str, Any]:
    success = _safe_child(run_root, "smoke_test.success.json")
    if success.exists() or success.is_symlink():
        raise RuntimeError("The smoke stage already has a terminal success marker")
    call_id, input_id, task_id = invocation_ids
    deployment = acknowledgement.deployment
    full_acceptance_path = f"runs/{run_id}/{post_spawn_acceptance_path}"
    registry_claim = SmokeAttemptRegistryClaim(
        registry_name=deployment.attempt_registry_name,
        registry_id=deployment.attempt_registry_id,
        registry_created_at_utc=deployment.attempt_registry_created_at_utc,
        registry_key=smoke_attempt_registry_key(run_id),
        run_id=run_id,
        call_id=call_id,
        input_id=input_id,
        task_id=task_id,
        launch_intent_sha256=launch_intent_sha256,
        post_spawn_acceptance_path=full_acceptance_path,
        post_spawn_acceptance_sha256=post_spawn_acceptance_sha256,
        smoke_config_hash=config.config_hash(),
        control_plane_sha256=control_plane.tree_sha256,
    )
    registry_claim_sha256 = registry_claim.claim_sha256()
    event = {
        "schema_version": "inkling-smoke-invocation-v3",
        "run_id": run_id,
        "stage": SMOKE_STAGE,
        "sequence": 1,
        "limit": config.resources.max_attempts,
        "call_id": call_id,
        "input_id": input_id,
        "task_id": task_id,
        "launch_intent_sha256": launch_intent_sha256,
        "smoke_config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "post_spawn_acceptance_path": post_spawn_acceptance_path,
        "post_spawn_acceptance_sha256": post_spawn_acceptance_sha256,
        "attempt_registry_name": deployment.attempt_registry_name,
        "attempt_registry_id": deployment.attempt_registry_id,
        "attempt_registry_created_at_utc": (deployment.attempt_registry_created_at_utc),
        "attempt_registry_key": registry_claim.registry_key,
        "attempt_registry_claim_sha256": registry_claim_sha256,
    }
    claim_path = _safe_child(run_root, "control", "smoke_test.attempt.claim.json")
    event_payload = (_canonical_json(event) + "\n").encode("utf-8")
    event_id = hashlib.sha256(_canonical_json(event).encode()).hexdigest()
    history_path = _safe_child(
        run_root,
        "control",
        "history",
        f"smoke_test.attempt.1.{event_id}.json",
    )
    event_sha256 = hashlib.sha256(event_payload).hexdigest()
    complete_invocation = {
        **event,
        "attempt_claim_path": claim_path.relative_to(run_root).as_posix(),
        "attempt_claim_sha256": registry_claim_sha256,
        "invocation_history_path": history_path.relative_to(run_root).as_posix(),
        "invocation_history_sha256": event_sha256,
    }
    if invocation_state:
        raise RuntimeError("Smoke invocation state must be empty before the claim")
    observed_registry_claim_sha256 = claim_smoke_attempt(
        attempt_registry,
        registry_claim,
    )
    if observed_registry_claim_sha256 != registry_claim_sha256:
        raise RuntimeError("The atomic smoke attempt claim returned the wrong hash")
    invocation_state.update(complete_invocation)
    _complete_owned_attempt_records(
        run_root,
        config=config,
        invocation=invocation_state,
    )
    return dict(invocation_state)


def _complete_owned_attempt_records(
    run_root: Path,
    *,
    config: InklingSmokeConfig,
    invocation: Mapping[str, Any],
) -> None:
    """Complete or verify bookkeeping for an invocation that owns the claim."""

    persisted_event = {
        name: value
        for name, value in invocation.items()
        if name
        not in {
            "attempt_claim_path",
            "attempt_claim_sha256",
            "invocation_history_path",
            "invocation_history_sha256",
        }
    }
    event_payload = (_canonical_json(persisted_event) + "\n").encode("utf-8")
    run_id = str(invocation["run_id"])
    registry_claim = SmokeAttemptRegistryClaim(
        registry_name=invocation["attempt_registry_name"],
        registry_id=invocation["attempt_registry_id"],
        registry_created_at_utc=invocation["attempt_registry_created_at_utc"],
        registry_key=invocation["attempt_registry_key"],
        run_id=run_id,
        call_id=invocation["call_id"],
        input_id=invocation["input_id"],
        task_id=invocation["task_id"],
        launch_intent_sha256=invocation["launch_intent_sha256"],
        post_spawn_acceptance_path=(f"runs/{run_id}/{invocation['post_spawn_acceptance_path']}"),
        post_spawn_acceptance_sha256=invocation["post_spawn_acceptance_sha256"],
        smoke_config_hash=invocation["smoke_config_hash"],
        control_plane_sha256=invocation["control_plane_sha256"],
    )
    registry_claim_payload = registry_claim.canonical_bytes()
    registry_claim_sha256 = registry_claim.claim_sha256()
    if (
        registry_claim_sha256 != invocation["attempt_registry_claim_sha256"]
        or registry_claim_sha256 != invocation["attempt_claim_sha256"]
    ):
        raise RuntimeError("Owned smoke attempt registry claim differs from its invocation")
    claim_path = _safe_child(run_root, *str(invocation["attempt_claim_path"]).split("/"))
    _atomic_bytes(claim_path, registry_claim_payload, allow_identical=True)
    if _read_existing_regular_bytes(claim_path) != registry_claim_payload:
        raise RuntimeError("Owned smoke attempt claim differs from its registry claim")
    _commit_and_reconcile_volume_files(
        {claim_path.relative_to(EVIDENCE_MOUNT).as_posix(): (registry_claim_payload)},
        event="attempt_claim_commit_reconciled",
        run_id=run_id,
    )
    history_path = _safe_child(
        run_root,
        *str(invocation["invocation_history_path"]).split("/"),
    )
    if hashlib.sha256(event_payload).hexdigest() != invocation["invocation_history_sha256"]:
        raise RuntimeError("Owned smoke invocation history differs from its invocation")
    _atomic_bytes(history_path, event_payload, allow_identical=True)
    if _sha256(history_path) != invocation["invocation_history_sha256"]:
        raise RuntimeError("Owned smoke invocation history differs after publication")
    ledger = {
        "schema_version": "inkling-smoke-attempt-ledger-v1",
        "stage": SMOKE_STAGE,
        "attempts": 1,
        "limit": config.resources.max_attempts,
        "last_history_path": history_path.relative_to(run_root).as_posix(),
        "last_history_sha256": invocation["invocation_history_sha256"],
        "last_call_id": invocation["call_id"],
        "last_input_id": invocation["input_id"],
        "last_task_id": invocation["task_id"],
        "launch_intent_sha256": invocation["launch_intent_sha256"],
        "smoke_config_hash": invocation["smoke_config_hash"],
        "control_plane_sha256": invocation["control_plane_sha256"],
    }
    ledger_path = _safe_child(run_root, "control", "smoke_test.attempts.json")
    ledger_payload = (_canonical_json(ledger) + "\n").encode("utf-8")
    _atomic_bytes(
        ledger_path,
        ledger_payload,
        allow_identical=True,
    )
    _commit_and_reconcile_volume_files(
        {
            history_path.relative_to(EVIDENCE_MOUNT).as_posix(): event_payload,
            ledger_path.relative_to(EVIDENCE_MOUNT).as_posix(): ledger_payload,
        },
        event="attempt_bookkeeping_commit_reconciled",
        run_id=run_id,
    )


def _regular_file_identity(path: Path, *, root: Path, expected_size: int) -> dict[str, Any]:
    try:
        before = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError("A required verified-export artifact is missing") from error
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or not path.resolve().is_relative_to(root.resolve())
        or before.st_size != expected_size
    ):
        raise RuntimeError("A required verified-export artifact is unsafe or size-drifted")
    digest = _sha256(path)
    after = path.lstat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("A verified-export artifact changed while it was hashed")
    return {"sha256": digest, "size_bytes": before.st_size}


def _verify_export_artifact(
    artifact: Any,
    *,
    subject_root: Path,
) -> dict[str, Any]:
    path = _safe_child(subject_root, *str(artifact.path).split("/"))
    identity = _regular_file_identity(
        path,
        root=subject_root,
        expected_size=int(artifact.size_bytes),
    )
    if identity["sha256"] != artifact.sha256:
        raise RuntimeError("A verified-export artifact SHA-256 differs from its reference")
    return {"path": artifact.path, **identity}


def _verify_complete_subject(
    reference: InklingVerifiedExportReference,
) -> tuple[dict[str, Any], ...]:
    artifacts = (
        *reference.q3_shards,
        reference.projector,
        reference.export_manifest,
        reference.verify_receipt,
        reference.quantize_receipt,
        reference.mmproj_receipt,
    )
    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as executor:
        records = tuple(
            executor.map(
                lambda artifact: _verify_export_artifact(
                    artifact,
                    subject_root=SUBJECT_MOUNT,
                ),
                artifacts,
            )
        )
    if len(records) != 54:
        raise RuntimeError("Verified-export artifact count changed")
    return records


def _runtime_python_inventory() -> tuple[str, bytes]:
    purelib = sysconfig.get_path("purelib")
    if not purelib:
        raise RuntimeError("Python purelib path is unavailable")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all", "--path", purelib],
        check=True,
        capture_output=True,
        timeout=120,
        shell=False,
    )
    lines = sorted(line for line in result.stdout.splitlines(keepends=True) if line.strip())
    return purelib, b"".join(lines)


def _runtime_python_packages(purelib: str) -> dict[str, str]:
    pairs: list[tuple[str, str]] = []
    for distribution in importlib.metadata.distributions(path=[purelib]):
        try:
            name = distribution.metadata["Name"]
        except KeyError as error:
            raise RuntimeError(
                "Python distribution metadata lacks an exact package name"
            ) from error
        version = distribution.version
        if not isinstance(name, str) or not isinstance(version, str):
            raise RuntimeError("Python distribution metadata lacks an exact name or version")
        pairs.append((name, version))
    try:
        return canonical_python_package_inventory(pairs)
    except ValueError as error:
        raise RuntimeError("Python package metadata inventory is invalid") from error


def _runtime_toolchain_evidence() -> dict[str, Any]:
    commit = (LLAMA_CPP_DIR / ".iql-build-commit").read_text().strip()
    if commit != PINNED_LLAMA_CPP_COMMIT:
        raise RuntimeError("llama.cpp checkout differs from the pinned smoke runtime")
    patch_sha256 = _sha256(REMOTE_PATCH)
    if patch_sha256 != DEFAULT_CONFIG.runtime.instrumentation_patch_sha256:
        raise RuntimeError("Runtime smoke instrumentation patch differs from its config")
    build_patch_sha256 = (
        (LLAMA_CPP_DIR / ".iql-smoke-patch.sha256").read_text(encoding="utf-8").split(maxsplit=1)[0]
    )
    if build_patch_sha256 != patch_sha256:
        raise RuntimeError("Build-time smoke instrumentation patch identity drifted")
    patched_paths = tuple(
        sorted(
            line
            for line in subprocess.run(
                ["git", "-C", str(LLAMA_CPP_DIR), "diff", "--name-only"],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
                shell=False,
            ).stdout.splitlines()
            if line
        )
    )
    expected_patched_paths = tuple(path for path, _ in SOURCE_BLOB_PINS)
    if set(patched_paths) != set(expected_patched_paths):
        raise RuntimeError("Patched llama.cpp source inventory differs from its contract")
    patched_diff = subprocess.run(
        [
            "git",
            "-C",
            str(LLAMA_CPP_DIR),
            "diff",
            "--binary",
            "--",
            *expected_patched_paths,
        ],
        check=True,
        capture_output=True,
        timeout=60,
        shell=False,
    ).stdout
    patched_diff_sha256 = hashlib.sha256(patched_diff).hexdigest()
    build_diff_sha256 = (
        (LLAMA_CPP_DIR / ".iql-patched-diff.sha256")
        .read_text(encoding="utf-8")
        .split(maxsplit=1)[0]
    )
    if build_diff_sha256 != patched_diff_sha256:
        raise RuntimeError("Patched llama.cpp diff identity changed after build")
    build_purelib = (LLAMA_CPP_DIR / ".iql-python-purelib.txt").read_text().strip()
    runtime_purelib, runtime_freeze = _runtime_python_inventory()
    build_freeze = (LLAMA_CPP_DIR / ".iql-python-freeze.txt").read_bytes()
    if runtime_purelib != build_purelib or runtime_freeze != build_freeze:
        raise RuntimeError("Image-owned Python inventory changed after build")
    python_packages = _runtime_python_packages(runtime_purelib)
    runtime_dpkg = subprocess.run(
        ["dpkg-query", "-W", "-f=${binary:Package}=${Version}\\n"],
        check=True,
        capture_output=True,
        timeout=120,
        shell=False,
    ).stdout
    runtime_dpkg = b"".join(
        sorted(line for line in runtime_dpkg.splitlines(keepends=True) if line.strip())
    )
    build_dpkg = (LLAMA_CPP_DIR / ".iql-dpkg-inventory.txt").read_bytes()
    if runtime_dpkg != build_dpkg:
        raise RuntimeError("Image-owned operating-system package inventory changed")
    try:
        dpkg_packages = parse_dpkg_inventory(runtime_dpkg)
    except ValueError as error:
        raise RuntimeError("Image-owned operating-system package inventory is invalid") from error
    cuda_backend = LLAMA_CPP_DIR / "build/bin/libggml-cuda.so"
    cuda_linkage = subprocess.run(
        ["ldd", str(cuda_backend)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        shell=False,
    ).stdout
    cuda_driver_path = Path(parse_cuda_driver_linkage(cuda_linkage)).resolve(strict=True)
    cuda_stub_root = CUDA_DRIVER_STUB_DIR.resolve(strict=True)
    if (
        "stubs" in cuda_driver_path.parts
        or cuda_driver_path in (cuda_stub_root, CUDA_DRIVER_LINK_DIR)
        or cuda_stub_root in cuda_driver_path.parents
        or CUDA_DRIVER_LINK_DIR in cuda_driver_path.parents
    ):
        raise RuntimeError("Runtime CUDA driver resolved to the build stub")
    if not cuda_driver_path.is_file():
        raise RuntimeError("Runtime CUDA driver is not a regular file")
    binaries: list[dict[str, Any]] = []
    for target in BUILD_TARGETS:
        path = LLAMA_CPP_DIR / "build" / "bin" / target
        expected_line = (LLAMA_CPP_DIR / f".iql-{target}.sha256").read_text().strip()
        expected_sha256 = expected_line.split(maxsplit=1)[0]
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"Built {target} binary drifted")
        binaries.append(
            {
                "name": target,
                "path": str(path),
                "sha256": actual_sha256,
                "size_bytes": path.stat().st_size,
            }
        )
    nvcc = subprocess.run(
        ["nvcc", "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        shell=False,
    ).stdout
    try:
        nvcc_version = parse_nvcc_version(nvcc)
    except ValueError as error:
        raise RuntimeError("CUDA compiler version inventory is invalid") from error
    python_inventory_sha256 = hashlib.sha256(runtime_freeze).hexdigest()
    dpkg_inventory_sha256 = hashlib.sha256(runtime_dpkg).hexdigest()
    nvcc_version_sha256 = hashlib.sha256(nvcc.encode()).hexdigest()
    python_implementation = platform.python_implementation()
    python_version = platform.python_version()
    python_executable_path = Path(sys.executable).resolve(strict=True)
    if not python_executable_path.is_file():
        raise RuntimeError("Python executable is not a regular file")
    python_executable_sha256 = _sha256(python_executable_path)
    modal_module_path_text = getattr(modal, "__file__", None)
    if not isinstance(modal_module_path_text, str):
        raise RuntimeError("Modal runtime module path is unavailable")
    modal_package_root = Path(modal_module_path_text).resolve(strict=True).parent
    try:
        modal_package_file_count, modal_package_tree_sha256 = immutable_source_tree_identity(
            modal_package_root
        )
    except ValueError as error:
        raise RuntimeError("Modal runtime source-tree identity is invalid") from error
    package_manifest_sha256 = smoke_package_manifest_sha256(
        python_implementation=python_implementation,
        python_version=python_version,
        python_executable_path=str(python_executable_path),
        python_executable_sha256=python_executable_sha256,
        python_inventory_scope="image_sysconfig_purelib_v1",
        python_purelib=runtime_purelib,
        python_inventory_sha256=python_inventory_sha256,
        python_packages=python_packages,
        modal_runtime_version=modal.__version__,
        modal_package_root=str(modal_package_root),
        modal_package_tree_schema_version="inkling-smoke-source-tree-v1",
        modal_package_file_count=modal_package_file_count,
        modal_package_tree_sha256=modal_package_tree_sha256,
        dpkg_inventory_sha256=dpkg_inventory_sha256,
        dpkg_packages=dpkg_packages,
        nvcc_version=nvcc_version,
        nvcc_version_sha256=nvcc_version_sha256,
    )
    return {
        "llama_cpp_repository": DEFAULT_CONFIG.runtime.repository,
        "llama_cpp_commit": commit,
        "instrumentation_schema_version": (DEFAULT_CONFIG.runtime.instrumentation_schema_version),
        "instrumentation_patch_path": str(REMOTE_PATCH),
        "instrumentation_patch_sha256": patch_sha256,
        "patched_source_paths": list(patched_paths),
        "patched_diff_sha256": patched_diff_sha256,
        "base_source_blob_ids": [
            {"path": path, "git_blob_id": blob_id} for path, blob_id in SOURCE_BLOB_PINS
        ],
        "cuda_image": CUDA_IMAGE_REFERENCE,
        "cuda_driver_library_path": str(cuda_driver_path),
        "cuda_driver_library_sha256": _sha256(cuda_driver_path),
        "cmake_definitions": list(DEFAULT_CONFIG.runtime.cmake_definitions),
        "binaries": binaries,
        "python_implementation": python_implementation,
        "python_version": python_version,
        "python_executable_path": str(python_executable_path),
        "python_executable_sha256": python_executable_sha256,
        "python_inventory_scope": "image_sysconfig_purelib_v1",
        "python_purelib": runtime_purelib,
        "python_inventory_sha256": python_inventory_sha256,
        "python_packages": python_packages,
        "modal_runtime_version": modal.__version__,
        "modal_package_root": str(modal_package_root),
        "modal_package_tree_schema_version": "inkling-smoke-source-tree-v1",
        "modal_package_file_count": modal_package_file_count,
        "modal_package_tree_sha256": modal_package_tree_sha256,
        "dpkg_inventory_sha256": dpkg_inventory_sha256,
        "dpkg_packages": dpkg_packages,
        "package_manifest_schema_version": "inkling-smoke-package-manifest-v2",
        "package_manifest_sha256": package_manifest_sha256,
        "nvcc_version": nvcc_version,
        "nvcc_version_sha256": nvcc_version_sha256,
    }


def _nvidia_identity(cuda_driver_library_path: str) -> tuple[dict[str, Any], ...]:
    driver_api_version, cuda_gpus = enumerate_cuda_driver_gpus(cuda_driver_library_path)
    fields = "uuid,name,memory.total,driver_version,compute_cap"
    payload = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        shell=False,
    ).stdout
    joined = combine_gpu_identity(
        driver_api_version,
        cuda_gpus,
        parse_nvidia_smi_csv(payload),
    )
    return tuple(item.model_dump(mode="json") for item in joined)


def _require_gpu_inventory_unchanged(
    expected: Sequence[Mapping[str, Any]],
    observed: Sequence[Mapping[str, Any]],
) -> None:
    stable_fields = (
        "identity_protocol",
        "cuda_driver_api_version",
        "cuda_ordinal",
        "uuid",
        "cuda_driver_name",
        "nvidia_smi_name",
        "cuda_compute_capability",
        "nvidia_smi_compute_capability",
        "cuda_total_memory_bytes",
        "nvidia_smi_memory_total_mib",
        "driver_version",
    )
    expected_identity = tuple(tuple(gpu.get(field) for field in stable_fields) for gpu in expected)
    observed_identity = tuple(tuple(gpu.get(field) for field in stable_fields) for gpu in observed)
    if observed_identity != expected_identity:
        raise RuntimeError("GPU UUID inventory changed before server start")


def _cuda_peer_topology(cuda_driver_library_path: str) -> dict[str, Any]:
    topology = enumerate_cuda_driver_peer_topology(cuda_driver_library_path)
    return topology.model_dump(mode="json")


def _require_peer_topology_matches_hardware(
    hardware: Sequence[Mapping[str, Any]],
    peer_topology: Mapping[str, Any],
) -> None:
    links = peer_topology.get("links")
    if not isinstance(links, Sequence) or isinstance(links, str | bytes) or len(links) != 2:
        raise RuntimeError("CUDA peer topology has the wrong link cardinality")
    expected = (
        (
            hardware[0].get("cuda_ordinal"),
            hardware[0].get("uuid"),
            hardware[1].get("cuda_ordinal"),
            hardware[1].get("uuid"),
        ),
        (
            hardware[1].get("cuda_ordinal"),
            hardware[1].get("uuid"),
            hardware[0].get("cuda_ordinal"),
            hardware[0].get("uuid"),
        ),
    )
    observed: list[tuple[object, object, object, object]] = []
    for link in links:
        if not isinstance(link, Mapping):
            raise RuntimeError("CUDA peer topology contains an invalid link")
        observed.append(
            (
                link.get("source_cuda_ordinal"),
                link.get("source_uuid"),
                link.get("destination_cuda_ordinal"),
                link.get("destination_uuid"),
            )
        )
    if tuple(observed) != expected:
        raise RuntimeError("CUDA peer topology differs from the joined GPU identity")
    driver_versions = {gpu.get("cuda_driver_api_version") for gpu in hardware}
    if driver_versions != {peer_topology.get("cuda_driver_api_version")}:
        raise RuntimeError("CUDA peer topology uses a different driver API version")


def _require_peer_topology_unchanged(
    expected: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> None:
    if dict(observed) != dict(expected):
        raise RuntimeError("CUDA peer topology changed before server start")


def _nvidia_topology_diagnostic() -> dict[str, Any]:
    argv = ["nvidia-smi", "topo", "-m"]
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            timeout=60,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        try:
            stdout = _captured_subprocess_bytes(error.stdout, label="stdout")
            stderr = _captured_subprocess_bytes(error.stderr, label="stderr")
        except RuntimeError:
            stdout = b""
            stderr = b""
            status = "invalid_result"
        else:
            status = "timed_out"
        return_code: int | None = None
    except OSError:
        stdout = b""
        stderr = b""
        status = "unavailable"
        return_code = None
    else:
        stdout = result.stdout
        stderr = result.stderr
        return_code = result.returncode
        if (
            not isinstance(stdout, bytes)
            or not isinstance(stderr, bytes)
            or type(return_code) is not int
        ):
            stdout = b""
            stderr = b""
            return_code = None
            status = "invalid_result"
        elif return_code != 0:
            status = "command_failed"
        elif not stdout.strip():
            status = "empty_output"
        else:
            status = "available"
    return SmokeNvidiaSmiTopologyDiagnostic.model_validate(
        {
            "schema_version": "inkling-smoke-nvidia-smi-topology-diagnostic-v1",
            "argv": argv,
            "status": status,
            "return_code": return_code,
            "stdout_size_bytes": len(stdout),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_size_bytes": len(stderr),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "stdout_recorded": False,
            "stderr_recorded": False,
        }
    ).model_dump(mode="json")


def _gpu_topology_identity(
    peer_topology: Mapping[str, Any],
    nvidia_smi_topology: Mapping[str, Any],
) -> dict[str, Any]:
    links = peer_topology.get("links")
    if not isinstance(links, Sequence) or isinstance(links, str | bytes):
        raise RuntimeError("CUDA peer topology links are invalid")
    return SmokeGpuTopologyEvidence.model_validate(
        {
            "schema_version": "inkling-smoke-gpu-topology-v1",
            "protocol": "cuda-driver-p2p-v1+nvidia-smi-topo-diagnostic-v1",
            "cuda_driver_api_version": peer_topology.get("cuda_driver_api_version"),
            "edges": list(links),
            "nvidia_smi_topology": dict(nvidia_smi_topology),
        }
    ).model_dump(mode="json")


def _cgroup_control_file_exists(path: Path) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise RuntimeError("cgroup control-file inventory is unavailable") from error
    if stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode):
        raise RuntimeError("cgroup control-file inventory contains a symlink or special file")
    return True


def _read_cgroup_control(path: Path, *, label: str) -> str:
    if not _cgroup_control_file_exists(path):
        raise RuntimeError(f"{label} is unavailable")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise RuntimeError(f"{label} is unreadable") from error


def _cgroup_leaf_pid_identity(leaf: Path, *, pid: int) -> str:
    path = leaf / "cgroup.procs"
    payload = _read_cgroup_control(path, label="process cgroup membership file")
    identifiers = payload.splitlines()
    if not identifiers or any(re.fullmatch(r"[1-9][0-9]*", item) is None for item in identifiers):
        raise RuntimeError("process cgroup membership file is invalid")
    if str(pid) not in identifiers:
        raise RuntimeError("resolved cgroup leaf does not contain the current process")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cgroup_path_sha256(path: Path) -> str:
    payload = b"inkling-smoke-cgroup-path-v1\0" + str(path).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cgroup_cpu_quota(
    leaf: Path,
    *,
    hierarchy_root: Path,
) -> tuple[int, str, tuple[str, ...], tuple[int | None, ...]]:
    try:
        leaf.relative_to(hierarchy_root)
    except ValueError as error:
        raise RuntimeError("process CPU cgroup leaf is outside its mount boundary") from error
    v2_path = leaf / "cpu.max"
    quota_path = leaf / "cpu.cfs_quota_us"
    period_path = leaf / "cpu.cfs_period_us"
    v2_exists = _cgroup_control_file_exists(v2_path)
    quota_exists = _cgroup_control_file_exists(quota_path)
    period_exists = _cgroup_control_file_exists(period_path)
    if v2_exists and (quota_exists or period_exists):
        raise RuntimeError("process CPU cgroup exposes ambiguous v1 and v2 limits")
    paths: list[str] = []
    values: list[int | None] = []
    try:
        if v2_exists:
            current = leaf
            while True:
                current_path = current / "cpu.max"
                if not _cgroup_control_file_exists(current_path):
                    if current == hierarchy_root:
                        break
                    raise RuntimeError("visible cgroup v2 CPU hierarchy is incomplete")
                values.append(
                    parse_cgroup_cpu_quota_millicores(
                        _read_cgroup_control(current_path, label="cgroup v2 CPU quota")
                    )
                )
                paths.append(str(current_path))
                if current == hierarchy_root:
                    break
                current = current.parent
            source = "cgroup_v2_visible_hierarchy_cpu.max"
        else:
            if not quota_exists or not period_exists:
                raise RuntimeError("process cgroup v1 CPU quota inventory is incomplete")
            current = leaf
            while True:
                current_quota = current / "cpu.cfs_quota_us"
                current_period = current / "cpu.cfs_period_us"
                current_quota_exists = _cgroup_control_file_exists(current_quota)
                current_period_exists = _cgroup_control_file_exists(current_period)
                if current_quota_exists != current_period_exists:
                    raise RuntimeError("visible cgroup v1 CPU quota inventory is incomplete")
                if not current_quota_exists:
                    raise RuntimeError("visible cgroup v1 CPU hierarchy is incomplete")
                normalized = (
                    f"{_read_cgroup_control(current_quota, label='cgroup v1 CPU quota').strip()} "
                    f"{_read_cgroup_control(current_period, label='cgroup v1 CPU period').strip()}"
                )
                values.append(parse_cgroup_cpu_quota_millicores(normalized))
                paths.extend((str(current_quota), str(current_period)))
                if current == hierarchy_root:
                    break
                current = current.parent
            source = "cgroup_v1_visible_hierarchy_cpu.cfs_quota_us"
    except ValueError as error:
        raise RuntimeError("process cgroup CPU quota inventory is invalid") from error
    finite = [value for value in values if value is not None]
    if not finite:
        raise RuntimeError("effective process cgroup CPU quota must be finite")
    return min(finite), source, tuple(paths), tuple(values)


def _cgroup_memory_limit(
    leaf: Path,
    *,
    hierarchy_root: Path,
) -> tuple[int, str, tuple[str, ...], tuple[int | None, ...]]:
    try:
        leaf.relative_to(hierarchy_root)
    except ValueError as error:
        raise RuntimeError("process memory cgroup leaf is outside its mount boundary") from error
    v2_path = leaf / "memory.max"
    v1_path = leaf / "memory.limit_in_bytes"
    v2_exists = _cgroup_control_file_exists(v2_path)
    v1_exists = _cgroup_control_file_exists(v1_path)
    if v2_exists == v1_exists:
        raise RuntimeError("process memory cgroup limit is missing or ambiguous")
    filename = "memory.max" if v2_exists else "memory.limit_in_bytes"
    paths: list[str] = []
    values: list[int | None] = []
    try:
        current = leaf
        while True:
            selected = current / filename
            if not _cgroup_control_file_exists(selected):
                if current == hierarchy_root and v2_exists:
                    break
                raise RuntimeError("visible cgroup memory hierarchy is incomplete")
            values.append(
                parse_cgroup_memory_limit_bytes(
                    _read_cgroup_control(selected, label="cgroup memory limit")
                )
            )
            paths.append(str(selected))
            if current == hierarchy_root:
                break
            current = current.parent
    except ValueError as error:
        raise RuntimeError("process cgroup memory-limit inventory is invalid") from error
    finite = [value for value in values if value is not None]
    if not finite:
        raise RuntimeError("effective process cgroup memory limit must be finite")
    source = (
        "cgroup_v2_visible_hierarchy_memory.max"
        if v2_exists
        else "cgroup_v1_visible_hierarchy_memory.limit_in_bytes"
    )
    return min(finite), source, tuple(paths), tuple(values)


def _cgroup_limit_inventory() -> dict[str, Any]:
    try:
        membership_payload = Path("/proc/self/cgroup").read_bytes()
        mountinfo_payload = Path("/proc/self/mountinfo").read_bytes()
        membership_text = membership_payload.decode("utf-8", errors="strict")
        mountinfo_text = mountinfo_payload.decode("utf-8", errors="strict")
        hierarchies = resolve_current_process_cgroup_hierarchy_paths(
            proc_self_cgroup=membership_text,
            proc_self_mountinfo=mountinfo_text,
        )
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError("current-process cgroup membership inventory is invalid") from error
    leaves = {controller: paths[0] for controller, paths in hierarchies.items()}
    hierarchy_roots = {controller: paths[1] for controller, paths in hierarchies.items()}
    pid = os.getpid()
    _cgroup_leaf_pid_identity(leaves["cpu"], pid=pid)
    _cgroup_leaf_pid_identity(leaves["memory"], pid=pid)
    cpu_value, cpu_source, cpu_paths, cpu_values = _cgroup_cpu_quota(
        leaves["cpu"], hierarchy_root=hierarchy_roots["cpu"]
    )
    memory_value, memory_source, memory_paths, memory_values = _cgroup_memory_limit(
        leaves["memory"], hierarchy_root=hierarchy_roots["memory"]
    )
    try:
        if (
            Path("/proc/self/cgroup").read_bytes() != membership_payload
            or Path("/proc/self/mountinfo").read_bytes() != mountinfo_payload
        ):
            raise RuntimeError("current-process cgroup membership changed during preflight")
    except OSError as error:
        raise RuntimeError("current-process cgroup inventory changed or disappeared") from error
    final_cpu_membership_sha256 = _cgroup_leaf_pid_identity(leaves["cpu"], pid=pid)
    final_memory_membership_sha256 = _cgroup_leaf_pid_identity(leaves["memory"], pid=pid)
    final_cpu = _cgroup_cpu_quota(leaves["cpu"], hierarchy_root=hierarchy_roots["cpu"])
    final_memory = _cgroup_memory_limit(leaves["memory"], hierarchy_root=hierarchy_roots["memory"])
    if final_cpu != (cpu_value, cpu_source, cpu_paths, cpu_values):
        raise RuntimeError("process cgroup CPU quota changed during preflight")
    if final_memory != (memory_value, memory_source, memory_paths, memory_values):
        raise RuntimeError("process cgroup memory limit changed during preflight")
    return {
        "cgroup_membership_sha256": hashlib.sha256(membership_payload).hexdigest(),
        "cgroup_mountinfo_sha256": hashlib.sha256(mountinfo_payload).hexdigest(),
        "cgroup_visibility_scope": "process_mount_namespace_visible_hierarchy",
        "cgroup_process_pid": pid,
        "cgroup_cpu_leaf_path_sha256": _cgroup_path_sha256(leaves["cpu"]),
        "cgroup_cpu_leaf_pid_verified": True,
        "cgroup_cpu_leaf_cgroup_procs_sha256": final_cpu_membership_sha256,
        "cgroup_cpu_quota_millicores": cpu_value,
        "cgroup_cpu_quota_source": cpu_source,
        "cgroup_cpu_limit_path_sha256s": [_cgroup_path_sha256(Path(path)) for path in cpu_paths],
        "cgroup_cpu_limit_values_millicores": list(cpu_values),
        "cgroup_memory_leaf_path_sha256": _cgroup_path_sha256(leaves["memory"]),
        "cgroup_memory_leaf_pid_verified": True,
        "cgroup_memory_leaf_cgroup_procs_sha256": final_memory_membership_sha256,
        "cgroup_memory_limit_bytes": memory_value,
        "cgroup_memory_limit_source": memory_source,
        "cgroup_memory_limit_path_sha256s": [
            _cgroup_path_sha256(Path(path)) for path in memory_paths
        ],
        "cgroup_memory_limit_values_bytes": list(memory_values),
    }


def _require_cgroup_inventory_unchanged(host: Mapping[str, Any]) -> None:
    observed = _cgroup_limit_inventory()
    if any(key not in host or host[key] != value for key, value in observed.items()):
        raise RuntimeError("process cgroup inventory changed before server start")


def _host_identity(
    gpus: Sequence[Mapping[str, Any]],
    *,
    gpu_topology: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        cpu_model = parse_proc_cpu_model(Path("/proc/cpuinfo").read_text(encoding="utf-8"))
        host_ram_bytes = parse_proc_mem_total_bytes(
            Path("/proc/meminfo").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise RuntimeError("Linux host hardware inventory is unavailable or invalid") from error
    host_logical_cpu_count = os.cpu_count()
    affinity_function = getattr(os, "sched_getaffinity", None)
    if (
        not isinstance(host_logical_cpu_count, int)
        or host_logical_cpu_count <= 0
        or not callable(affinity_function)
    ):
        raise RuntimeError("Linux logical CPU or affinity inventory is unavailable")
    affinity_ids = tuple(sorted(affinity_function(0)))
    if not affinity_ids:
        raise RuntimeError("Linux CPU affinity inventory is empty")
    cgroup = _cgroup_limit_inventory()
    cgroup_cpu_quota_millicores = int(cgroup["cgroup_cpu_quota_millicores"])
    logical_cpu_count = len(affinity_ids)
    logical_cpu_count_scope = "container_effective_sched_getaffinity"
    if cgroup_cpu_quota_millicores // 1000 <= logical_cpu_count:
        logical_cpu_count = cgroup_cpu_quota_millicores // 1000
        logical_cpu_count_scope = "container_cgroup_cpu_quota"
    cgroup_memory_limit_bytes = int(cgroup["cgroup_memory_limit_bytes"])
    ram_bytes = host_ram_bytes
    ram_scope = "host_physical_no_lower_cgroup_limit"
    if cgroup_memory_limit_bytes < host_ram_bytes:
        ram_bytes = cgroup_memory_limit_bytes
        ram_scope = "container_cgroup_memory_limit"
    host: dict[str, Any] = {
        "provider": "Modal",
        "cpu_model": cpu_model,
        "host_logical_cpu_count": host_logical_cpu_count,
        "host_logical_cpu_count_scope": "host_online_os_cpu_count",
        "logical_cpu_count": logical_cpu_count,
        "logical_cpu_count_scope": logical_cpu_count_scope,
        "requested_cpu_cores": DEFAULT_CONFIG.resources.cpu_cores,
        "requested_cpu_scope": "modal_physical_cores_hard_request_and_limit",
        "cgroup_membership_sha256": cgroup["cgroup_membership_sha256"],
        "cgroup_mountinfo_sha256": cgroup["cgroup_mountinfo_sha256"],
        "cgroup_visibility_scope": cgroup["cgroup_visibility_scope"],
        "cgroup_process_pid": cgroup["cgroup_process_pid"],
        "cgroup_cpu_leaf_path_sha256": cgroup["cgroup_cpu_leaf_path_sha256"],
        "cgroup_cpu_leaf_pid_verified": cgroup["cgroup_cpu_leaf_pid_verified"],
        "cgroup_cpu_leaf_cgroup_procs_sha256": cgroup["cgroup_cpu_leaf_cgroup_procs_sha256"],
        "cgroup_cpu_quota_millicores": cgroup_cpu_quota_millicores,
        "cgroup_cpu_quota_source": cgroup["cgroup_cpu_quota_source"],
        "cgroup_cpu_limit_path_sha256s": cgroup["cgroup_cpu_limit_path_sha256s"],
        "cgroup_cpu_limit_values_millicores": cgroup["cgroup_cpu_limit_values_millicores"],
        "cpu_affinity_ids": list(affinity_ids),
        "cpu_affinity_scope": "container_effective_sched_getaffinity",
        "host_ram_bytes": host_ram_bytes,
        "host_ram_scope": "host_physical_proc_meminfo_memtotal",
        "ram_bytes": ram_bytes,
        "ram_scope": ram_scope,
        "requested_ram_bytes": DEFAULT_CONFIG.resources.memory_gib * 1024**3,
        "requested_ram_scope": "modal_bytes_hard_request_and_limit",
        "cgroup_memory_leaf_path_sha256": cgroup["cgroup_memory_leaf_path_sha256"],
        "cgroup_memory_leaf_pid_verified": cgroup["cgroup_memory_leaf_pid_verified"],
        "cgroup_memory_leaf_cgroup_procs_sha256": cgroup["cgroup_memory_leaf_cgroup_procs_sha256"],
        "cgroup_memory_limit_bytes": cgroup_memory_limit_bytes,
        "cgroup_memory_limit_source": cgroup["cgroup_memory_limit_source"],
        "cgroup_memory_limit_path_sha256s": cgroup["cgroup_memory_limit_path_sha256s"],
        "cgroup_memory_limit_values_bytes": cgroup["cgroup_memory_limit_values_bytes"],
        "topology_schema_version": "inkling-smoke-hardware-topology-v4",
    }
    host["topology_sha256"] = smoke_hardware_topology_sha256(
        host,
        gpus,
        gpu_topology,
    )
    return SmokeHostEvidence.model_validate(host).model_dump(mode="json", exclude_none=True)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = struct.pack(">I", zlib.crc32(kind + payload))
    return struct.pack(">I", len(payload)) + kind + payload + checksum


def _checkerboard_png() -> bytes:
    rows = bytearray()
    for y in range(16):
        rows.append(0)
        for x in range(16):
            value = 255 if (x + y) % 2 == 0 else 0
            rows.extend((value, value, value))
    header = struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
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


def _fixture_bytes(probe: SmokeProbeConfig) -> bytes | None:
    if probe.fixture == "none":
        return None
    if probe.fixture == "synthetic_rgb8_png_16x16_checkerboard_v1":
        return _checkerboard_png()
    if probe.fixture == "synthetic_pcm_s16le_wav_16000hz_mono_silence_250ms_v1":
        return _silence_wav()
    raise RuntimeError("Unsupported checked smoke fixture")


def _http_json(
    method: str,
    path: str,
    *,
    body: Mapping[str, Any] | None = None,
    timeout: float,
) -> tuple[dict[str, Any], str]:
    payload = None if body is None else _canonical_json(body).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{SERVER_PORT}{path}",
        data=payload,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"llama-server returned HTTP status {error.code}") from None
    if len(raw) > MAX_HTTP_RESPONSE_BYTES:
        raise RuntimeError("llama-server response exceeded the smoke evidence limit")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("llama-server returned invalid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError("llama-server response must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _wait_until_ready(process: subprocess.Popen[bytes], *, timeout: float) -> float:
    started = time.monotonic()
    deadline = started + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited before its health endpoint became ready")
        try:
            health, _ = _http_json("GET", "/health", timeout=5)
        except (OSError, RuntimeError):
            time.sleep(2)
            continue
        if health.get("status") == "ok":
            return time.monotonic() - started
        time.sleep(2)
    raise RuntimeError("llama-server did not become ready before the load deadline")


def _read_server_log() -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(SERVER_LOG, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("llama-server log is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(MAX_SERVER_LOG_BYTES + 1)
    finally:
        os.close(descriptor)
    if len(payload) > MAX_SERVER_LOG_BYTES:
        raise RuntimeError("llama-server log exceeded the bounded smoke limit")
    return payload


class _RuntimeMonitor:
    def __init__(self, pid: int, gpu_uuids: Sequence[str]) -> None:
        self._pid = pid
        self._gpu_uuids = tuple(gpu_uuids)
        if len(self._gpu_uuids) != 2 or len(set(self._gpu_uuids)) != 2:
            raise ValueError("Runtime monitor requires two unique GPU UUIDs")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._samples = 0
        self._peak_rss_mib = 0
        self._gpu_peak_memory_mib = [0, 0]
        self._gpu_peak_utilization_percent = [0, 0]
        self._error: BaseException | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=GPU_MONITOR_STOP_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError("Runtime monitor did not stop")
        if self._error is not None:
            raise RuntimeError("Runtime monitor failed") from self._error
        if self._samples < 1:
            raise RuntimeError("Runtime monitor captured no resource samples")
        return {
            "sampling_interval_seconds": GPU_SAMPLE_INTERVAL_SECONDS,
            "sample_count": self._samples,
            "server_peak_host_rss_mib": self._peak_rss_mib,
            "gpu_peak_memory_used_mib": self._gpu_peak_memory_mib,
            "gpu_peak_utilization_percent": self._gpu_peak_utilization_percent,
        }

    def _sample_loop(self) -> None:
        try:
            while not self._stop.is_set():
                self._sample_once()
                self._stop.wait(GPU_SAMPLE_INTERVAL_SECONDS)
        except BaseException as error:
            self._error = error

    def _sample_once(self) -> None:
        status = Path(f"/proc/{self._pid}/status")
        if status.is_file():
            match = re.search(r"^VmRSS:\s+([0-9]+)\s+kB$", status.read_text(), re.MULTILINE)
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
            timeout=GPU_MONITOR_COMMAND_TIMEOUT_SECONDS,
            shell=False,
        ).stdout
        samples = parse_nvidia_smi_monitor_csv(output, expected_uuids=self._gpu_uuids)
        for ordinal, sample in enumerate(samples):
            self._gpu_peak_memory_mib[ordinal] = max(
                self._gpu_peak_memory_mib[ordinal], sample.memory_used_mib
            )
            self._gpu_peak_utilization_percent[ordinal] = max(
                self._gpu_peak_utilization_percent[ordinal], sample.utilization_percent
            )
        self._samples += 1


def _model_properties() -> tuple[dict[str, Any], int, str]:
    props, props_sha256 = _http_json("GET", "/props", timeout=30)
    modalities = props.get("modalities")
    if (
        not isinstance(modalities, Mapping)
        or modalities.get("vision") is not True
        or modalities.get("audio") is not True
    ):
        raise RuntimeError("llama-server did not expose both checked multimodal capabilities")
    marker = props.get("media_marker")
    if not isinstance(marker, str) or not marker or marker in {"<image>", "<audio>"}:
        raise RuntimeError("llama-server returned an invalid dynamic media marker")
    build_info = props.get("build_info")
    if not isinstance(build_info, str) or PINNED_LLAMA_CPP_COMMIT[:7] not in build_info:
        raise RuntimeError("llama-server build info does not bind the pinned commit")
    models, models_sha256 = _http_json("GET", "/v1/models", timeout=30)
    data = models.get("data")
    if not isinstance(data, Sequence) or isinstance(data, str | bytes) or len(data) != 1:
        raise RuntimeError("llama-server model metadata has the wrong cardinality")
    model = data[0]
    if not isinstance(model, Mapping) or not isinstance(model.get("meta"), Mapping):
        raise RuntimeError("llama-server model metadata is incomplete")
    n_vocab = model["meta"].get("n_vocab")
    if type(n_vocab) is not int or n_vocab <= 0:
        raise RuntimeError("llama-server model metadata lacks a valid vocabulary size")
    return (
        {
            "props_sha256": props_sha256,
            "models_sha256": models_sha256,
            "build_info": build_info,
            "modalities": {"text": True, "vision": True, "audio": True},
            "vocab_size": n_vocab,
            "media_marker_sha256": hashlib.sha256(marker.encode()).hexdigest(),
        },
        n_vocab,
        marker,
    )


def _probe_request(
    probe: SmokeProbeConfig,
    *,
    media_marker: str,
    n_predict: int,
    n_probs: int,
) -> dict[str, Any]:
    fixture = _fixture_bytes(probe)
    prompt: str | dict[str, Any]
    if fixture is None:
        prompt = probe.prompt
    else:
        prompt = {
            "prompt_string": f"{media_marker}\n{probe.prompt}",
            "multimodal_data": [base64.b64encode(fixture).decode("ascii")],
        }
    return {
        "prompt": prompt,
        "seed": probe.seed,
        "temperature": probe.temperature,
        "n_predict": n_predict,
        "n_probs": n_probs,
        "post_sampling_probs": probe.post_sampling_probs,
        "stream": probe.stream,
        "cache_prompt": probe.cache_prompt,
        "return_tokens": probe.return_tokens,
        "timings_per_token": probe.timings_per_token,
    }


def _timing_evidence(payload: Mapping[str, Any]) -> dict[str, int | float]:
    timings = payload.get("timings")
    if not isinstance(timings, Mapping):
        raise RuntimeError("llama-server completion lacks timing evidence")
    integer_fields = ("prompt_n", "predicted_n", "cache_n")
    float_fields = (
        "prompt_ms",
        "prompt_per_token_ms",
        "prompt_per_second",
        "predicted_ms",
        "predicted_per_token_ms",
        "predicted_per_second",
    )
    result: dict[str, int | float] = {}
    for field in integer_fields:
        value = timings.get(field)
        if type(value) is not int or value < 0:
            raise RuntimeError(f"llama-server timing {field} is invalid")
        result[field] = value
    for field in float_fields:
        value = timings.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
            or value < 0
        ):
            raise RuntimeError(f"llama-server timing {field} is invalid")
        result[field] = float(value)
    if result["predicted_n"] < 1 or result["predicted_ms"] <= 0:
        raise RuntimeError("llama-server timing does not contain measured generation")
    return result


def _run_probe(
    probe: SmokeProbeConfig,
    *,
    vocab_size: int,
    unpadded_vocab_size: int,
    media_marker: str,
) -> dict[str, Any]:
    fixture = _fixture_bytes(probe)
    trials: list[dict[str, Any]] = []
    for trial in range(1, probe.trials + 1):
        payload, response_sha256 = _http_json(
            "POST",
            DEFAULT_CONFIG.runtime.server_endpoint,
            body=_probe_request(
                probe,
                media_marker=media_marker,
                n_predict=probe.n_predict,
                n_probs=probe.n_probs,
            ),
            timeout=900,
        )
        parsed = parse_server_completion(
            payload,
            vocab_size=vocab_size,
            expected_n_probs=probe.n_probs,
            unpadded_vocab_size=unpadded_vocab_size,
        )
        trials.append(
            {
                "trial": trial,
                "token_ids": list(parsed.token_ids),
                "tokens_predicted": parsed.tokens_predicted,
                "minimum_sampled_token_logprob": parsed.minimum_logprob,
                "maximum_sampled_token_logprob": parsed.maximum_logprob,
                "all_returned_logprobs_finite": parsed.all_returned_logprobs_finite,
                "response_sha256": response_sha256,
                "timings": _timing_evidence(payload),
            }
        )
    token_sequences = {tuple(trial["token_ids"]) for trial in trials}
    if len(token_sequences) != 1:
        raise RuntimeError("Repeated greedy output is not token-identical")
    return {
        "probe_id": probe.probe_id,
        "modality": probe.modality,
        "prompt_sha256": probe.prompt_sha256,
        "prompt_recorded": False,
        "output_text_recorded": False,
        "fixture": probe.fixture,
        "fixture_sha256": None if fixture is None else hashlib.sha256(fixture).hexdigest(),
        "fixture_size_bytes": None if fixture is None else len(fixture),
        "seed": probe.seed,
        "temperature": probe.temperature,
        "repeatable_greedy_token_ids": True,
        "trials": trials,
    }


def _server_command(reference: InklingVerifiedExportReference) -> list[str]:
    first_shard = SUBJECT_MOUNT / reference.q3_shards[0].path
    projector = SUBJECT_MOUNT / reference.projector.path
    return [
        str(LLAMA_CPP_DIR / "build/bin/llama-server"),
        "--log-verbosity",
        str(DEFAULT_CONFIG.runtime.log_verbosity),
        "--model",
        str(first_shard),
        "--mmproj",
        str(projector),
        "--host",
        "127.0.0.1",
        "--port",
        str(SERVER_PORT),
        "--ctx-size",
        str(DEFAULT_CONFIG.runtime.context_size),
        "--n-gpu-layers",
        "all",
        "--n-cpu-moe",
        "0",
        "--split-mode",
        DEFAULT_CONFIG.runtime.split_mode,
        "--tensor-split",
        "1,1",
        "--flash-attn",
        "on",
        "--mmap",
        "--mmproj-offload",
        "--parallel",
        "1",
        "--threads",
        "16",
        "--threads-batch",
        "16",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--no-webui",
    ]


def _server_environment() -> dict[str, str]:
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("LLAMA_ARG_")
    }
    environment.update(SERVER_AUDIT_ENVIRONMENT)
    return environment


def _terminate_process(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
            return {"method": "SIGTERM", "return_code": process.returncode, "clean": True}
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
            return {"method": "SIGKILL", "return_code": process.returncode, "clean": False}
    return {"method": "already_exited", "return_code": process.returncode, "clean": False}


def _failure_server_log_evidence() -> dict[str, Any]:
    """Hash the full log and retain only bounded structural failure evidence."""

    patterns = {
        "out_of_memory_observed": b"out of memory",
        "no_usable_gpu_observed": b"no usable gpu found",
        "model_load_failure_observed": b"failed to load model",
        "projector_load_failure_observed": b"failed to load multimodal projector",
        "unsupported_architecture_observed": b"unsupported model architecture",
    }
    observed = dict.fromkeys(patterns, False)
    digest = hashlib.sha256()
    size_bytes = 0
    backend = BackendFailureDiagnosticAccumulator()

    def result(*, present: bool, scan_integrity: str) -> dict[str, Any]:
        backend_diagnostic, backend_malformed = backend.finish()
        if backend_malformed and scan_integrity != "missing":
            scan_integrity = "malformed"
        return SmokeServerLogFailureEvidence.model_validate(
            {
                "schema_version": "inkling-smoke-server-log-failure-v1",
                "present": present,
                "size_bytes": size_bytes,
                "sha256": digest.hexdigest(),
                "raw_log_recorded": False,
                "scan_integrity": scan_integrity,
                "safe_failure_signals": observed,
                "backend_diagnostic": backend_diagnostic.model_dump(mode="json"),
            }
        ).model_dump(mode="json")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(SERVER_LOG, flags)
    except FileNotFoundError:
        return result(present=False, scan_integrity="missing")
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError("llama-server log is not a regular file")
        maximum_pattern_bytes = max(len(pattern) for pattern in patterns.values())
        signal_tail = b""
        marker_tails = {marker: b"" for marker in BACKEND_FAILURE_MARKER_TOKENS}
        line_buffer = bytearray()
        line_is_overlong = False

        def observe_overlong_marker_bytes(value: bytes) -> None:
            counts: dict[bytes, int] = {}
            for marker in BACKEND_FAILURE_MARKER_TOKENS:
                searchable = marker_tails[marker] + value
                counts[marker] = searchable.count(marker)
                marker_tails[marker] = searchable[-(len(marker) - 1) :]
            graph_count = counts[BACKEND_FAILURE_MARKER_TOKENS[0]]
            cpu_count = counts[BACKEND_FAILURE_MARKER_TOKENS[1]]
            if graph_count + cpu_count:
                backend.observe_unparsed_marker_counts(
                    graph_marker_count=graph_count,
                    cpu_node_marker_count=cpu_count,
                )

        def reset_marker_tails() -> None:
            for marker in BACKEND_FAILURE_MARKER_TOKENS:
                marker_tails[marker] = b""

        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while segment := handle.readline(MAX_FAILURE_LOG_LINE_BYTES + 1):
                digest.update(segment)
                size_bytes += len(segment)
                searchable = (signal_tail + segment).lower()
                for name, pattern in patterns.items():
                    if not observed[name] and pattern in searchable:
                        observed[name] = True
                signal_tail = searchable[-(maximum_pattern_bytes - 1) :]

                if line_is_overlong:
                    observe_overlong_marker_bytes(segment)
                    if segment.endswith(b"\n"):
                        line_is_overlong = False
                        reset_marker_tails()
                    continue

                line_buffer.extend(segment)
                if len(line_buffer) > MAX_FAILURE_LOG_LINE_BYTES:
                    line_is_overlong = True
                    observe_overlong_marker_bytes(bytes(line_buffer))
                    line_buffer.clear()
                    if segment.endswith(b"\n"):
                        line_is_overlong = False
                        reset_marker_tails()
                    continue
                if segment.endswith(b"\n"):
                    backend.observe_line(bytes(line_buffer))
                    line_buffer.clear()
            if line_buffer:
                backend.observe_line(bytes(line_buffer))
    finally:
        os.close(descriptor)
    return result(present=True, scan_integrity="complete")


def _failed_subprocess_command_id(command: object) -> str:
    if (
        not isinstance(command, Sequence)
        or isinstance(command, str | bytes)
        or any(not isinstance(item, str) for item in command)
    ):
        raise RuntimeError("failed subprocess command is not a string argument sequence")
    argv = tuple(command)
    purelib = sysconfig.get_path("purelib")
    expected_patched_paths = tuple(path for path, _git_blob_id in SOURCE_BLOB_PINS)
    allowlist: dict[tuple[str, ...], str] = {
        (
            sys.executable,
            "-m",
            "pip",
            "freeze",
            "--all",
            "--path",
            purelib or "",
        ): "python_package_inventory_v1",
        (
            "git",
            "-C",
            str(LLAMA_CPP_DIR),
            "diff",
            "--name-only",
        ): "llama_cpp_git_changed_paths_v1",
        (
            "git",
            "-C",
            str(LLAMA_CPP_DIR),
            "diff",
            "--binary",
            "--",
            *expected_patched_paths,
        ): "llama_cpp_git_patched_diff_v1",
        (
            "dpkg-query",
            "-W",
            "-f=${binary:Package}=${Version}\\n",
        ): "dpkg_inventory_v1",
        (
            "ldd",
            str(LLAMA_CPP_DIR / "build/bin/libggml-cuda.so"),
        ): "cuda_driver_linkage_v1",
        ("nvcc", "--version"): "cuda_compiler_version_v1",
        (
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ): "nvidia_smi_identity_v1",
    }
    try:
        return allowlist[argv]
    except KeyError as error:
        raise RuntimeError("failed subprocess command is not allowlisted") from error


def _captured_subprocess_bytes(value: object, *, label: str) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    raise RuntimeError(f"failed subprocess {label} has an unsupported type")


def _safe_subprocess_failure(
    error: BaseException,
) -> dict[str, Any] | None:
    if not isinstance(error, subprocess.CalledProcessError):
        return None
    if type(error.returncode) is not int or error.returncode == 0:
        raise RuntimeError("failed subprocess has an invalid return code")
    stdout = _captured_subprocess_bytes(getattr(error, "stdout", None), label="stdout")
    stderr = _captured_subprocess_bytes(getattr(error, "stderr", None), label="stderr")
    return SmokeSubprocessFailureEvidence.model_validate(
        {
            "schema_version": "inkling-smoke-subprocess-failure-v1",
            "command_id": _failed_subprocess_command_id(error.cmd),
            "return_code": error.returncode,
            "stdout_size_bytes": len(stdout),
            "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
            "stderr_size_bytes": len(stderr),
            "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
            "stdout_recorded": False,
            "stderr_recorded": False,
        }
    ).model_dump(mode="json")


def _record_failure(
    run_root: Path,
    *,
    config: InklingSmokeConfig,
    control_plane: SmokeControlPlaneProvenance,
    invocation: Mapping[str, Any],
    launch_intent_sha256: str,
    phase: str,
    error: BaseException,
) -> None:
    _complete_owned_attempt_records(
        run_root,
        config=config,
        invocation=invocation,
    )
    server_log_evidence = _failure_server_log_evidence()
    receipt: dict[str, Any] = {
        "schema_version": "inkling-smoke-terminal-v6",
        "status": "failed",
        "stage": SMOKE_STAGE,
        "run_id": run_root.name,
        "subject_run_id": DEFAULT_REFERENCE.subject_run_id,
        "smoke_config_hash": config.config_hash(),
        "verified_export_reference_sha256": DEFAULT_REFERENCE.reference_sha256,
        "control_plane_sha256": control_plane.tree_sha256,
        "launch_intent_sha256": launch_intent_sha256,
        "call_id": invocation["call_id"],
        "input_id": invocation["input_id"],
        "task_id": invocation["task_id"],
        "invocation": dict(invocation),
        "failure_phase": phase,
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
        "safe_subprocess_failure": _safe_subprocess_failure(error),
        "server_log_sha256": server_log_evidence["sha256"],
        "safe_failure_signals": server_log_evidence["safe_failure_signals"],
        "server_log_evidence": server_log_evidence,
        "prompt_text_recorded": False,
        "output_text_recorded": False,
    }
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)
    path = _safe_child(
        run_root,
        "control",
        "outcomes",
        f"smoke_test.failed.{receipt['receipt_sha256']}.json",
    )
    validate_smoke_failure_receipt(
        receipt,
        config=config,
        reference=DEFAULT_REFERENCE,
        control_plane=control_plane,
        run_id=run_root.name,
        launch_intent_sha256=launch_intent_sha256,
        outcome_path=path.relative_to(EVIDENCE_MOUNT).as_posix(),
    )
    _publish_failure_receipt(path, receipt, run_id=run_root.name)


@app.function(
    image=smoke_image,
    gpu="B300:2",
    cpu=(16, 16),
    memory=(65_536, 65_536),
    ephemeral_disk=524_288,
    retries=0,
    timeout=7_200,
    startup_timeout=900,
    max_containers=1,
    single_use_containers=True,
    block_network=True,
    volumes={str(SUBJECT_MOUNT): subject_volume, str(EVIDENCE_MOUNT): evidence_volume},
)
def smoke_test(
    config_json: str,
    reference_json: str,
    run_id: str,
    launch_intent_sha256: str,
    acknowledgement_json: str,
    control_plane_json: str,
) -> dict[str, Any]:
    """Rehash, load, and probe the exact verified Q3 export once."""

    phase = "validate_remote_inputs"
    invocation: Mapping[str, Any] | None = None
    process: subprocess.Popen[bytes] | None = None
    monitor: _RuntimeMonitor | None = None
    log_handle: Any | None = None
    run_root: Path | None = None
    success_path: Path | None = None
    success_publication_started = False
    success_publication_confirmed = False
    try:
        config = _remote_config(config_json)
        reference = _remote_reference(reference_json)
        control_plane = validate_deployed_smoke_control_plane(
            control_plane_json,
            deployment_script=ENTRYPOINT_PATH,
            deployed_package_root=REMOTE_PACKAGE,
            deployed_config=REMOTE_CONFIG,
            deployed_reference=REMOTE_REFERENCE,
            deployed_patch=REMOTE_PATCH,
        )
        expected_run_id = smoke_run_id(config, control_plane.tree_sha256)
        if run_id != expected_run_id:
            raise RuntimeError("Remote smoke run ID differs from its deterministic identity")
        acknowledgement = validate_smoke_launch_acknowledgement(
            acknowledgement_json,
            config=config,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )
        attempt_registry = _sealed_attempt_registry(config, acknowledgement)
        _require_stage_window(config, acknowledgement.billing_cycle_end_utc)
        candidate_run_root = _safe_child(EVIDENCE_MOUNT, "runs", run_id)
        phase = "validate_remote_launch_authorization"
        remote_launch_intent = smoke_launch_intent_remote_path(
            run_id,
            launch_intent_sha256,
        )
        authorization_path = _safe_child(
            EVIDENCE_MOUNT,
            *remote_launch_intent.split("/"),
        )
        authorization_bytes = _read_existing_regular_bytes(authorization_path)
        if len(authorization_bytes) > MAX_LAUNCH_INTENT_BYTES:
            raise RuntimeError("Remote smoke launch authorization exceeds its size limit")
        authorization = validate_smoke_launch_intent(
            authorization_bytes,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
            deployment=acknowledgement.deployment,
        )
        if authorization.billing_cycle_end_utc != acknowledgement.billing_cycle_end_utc:
            raise RuntimeError(
                "Remote smoke launch authorization and acknowledgement use different cycles"
            )
        phase = "wait_for_post_spawn_acceptance"
        (
            invocation_ids,
            post_spawn_acceptance_path,
            post_spawn_acceptance_sha256,
        ) = _wait_for_post_spawn_acceptance(
            config=config,
            reference=reference,
            control_plane=control_plane,
            acknowledgement=acknowledgement,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )
        run_root = candidate_run_root
        phase = "commit_attempt"
        claimed_invocation: dict[str, Any] = {}
        invocation = claimed_invocation
        invocation = _begin_only_attempt(
            run_root,
            config=config,
            control_plane=control_plane,
            acknowledgement=acknowledgement,
            attempt_registry=attempt_registry,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
            invocation_ids=invocation_ids,
            post_spawn_acceptance_path=post_spawn_acceptance_path,
            post_spawn_acceptance_sha256=post_spawn_acceptance_sha256,
            invocation_state=claimed_invocation,
        )
        _bind_run_inputs(
            run_root,
            config=config,
            reference_json=reference_json,
            control_plane_json=control_plane_json,
        )
        phase = "verify_runtime_and_allocation"
        toolchain = _runtime_toolchain_evidence()
        cuda_driver_library_path = str(toolchain["cuda_driver_library_path"])
        hardware = _nvidia_identity(cuda_driver_library_path)
        peer_topology = _cuda_peer_topology(cuda_driver_library_path)
        _require_peer_topology_matches_hardware(hardware, peer_topology)
        gpu_topology = _gpu_topology_identity(
            peer_topology,
            _nvidia_topology_diagnostic(),
        )
        host = _host_identity(
            hardware,
            gpu_topology=gpu_topology,
        )
        phase = "verify_subject_hashes"
        artifact_started = time.monotonic()
        artifacts = _verify_complete_subject(reference)
        artifact_seconds = time.monotonic() - artifact_started
        phase = "revalidate_allocation_before_server_start"
        _require_cgroup_inventory_unchanged(host)
        _require_gpu_inventory_unchanged(
            hardware,
            _nvidia_identity(cuda_driver_library_path),
        )
        observed_peer_topology = _cuda_peer_topology(cuda_driver_library_path)
        _require_peer_topology_matches_hardware(hardware, observed_peer_topology)
        _require_peer_topology_unchanged(peer_topology, observed_peer_topology)
        phase = "start_server"
        command = _server_command(reference)
        SERVER_LOG.unlink(missing_ok=True)
        log_handle = SERVER_LOG.open("xb")
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=_server_environment(),
            shell=False,
        )
        monitor = _RuntimeMonitor(process.pid, [str(gpu["uuid"]) for gpu in hardware])
        monitor.start()
        load_seconds = _wait_until_ready(process, timeout=3_600)
        phase = "validate_server_properties"
        properties, vocab_size, media_marker = _model_properties()
        output_vocabulary = config.output_vocabulary
        if output_vocabulary is None:
            raise RuntimeError("Smoke configuration lacks the output vocabulary contract")
        if vocab_size != output_vocabulary.vocab_size:
            raise RuntimeError(
                "llama-server vocabulary size differs from the output vocabulary contract"
            )
        phase = "run_deterministic_probes"
        probes = [
            _run_probe(
                probe,
                vocab_size=vocab_size,
                unpadded_vocab_size=output_vocabulary.unpadded_vocab_size,
                media_marker=media_marker,
            )
            for probe in config.probes
        ]
        phase = "stop_server"
        resources = monitor.stop()
        monitor = None
        cleanup = _terminate_process(process)
        process = None
        log_handle.close()
        log_handle = None
        log_bytes = _read_server_log()
        if cleanup["clean"] is not True:
            raise RuntimeError("llama-server did not exit cleanly after its smoke probes")
        log_text = log_bytes.decode("utf-8", errors="strict")
        loader = parse_loader_offload_evidence(log_text)
        expected_generated_token_vectors = sum(
            trial["tokens_predicted"] for probe in probes for trial in probe["trials"]
        )
        raw_logit_audit = parse_raw_logit_audit_evidence(
            log_text,
            expected_generated_token_vectors=expected_generated_token_vectors,
            vocab_size=vocab_size,
            unpadded_vocab_size=output_vocabulary.unpadded_vocab_size,
        )
        backend_audit = parse_backend_audit_evidence(log_text)
        artifact_load = parse_artifact_load_evidence(log_text)
        phase = "publish_success"
        receipt: dict[str, Any] = {
            "schema_version": "inkling-smoke-terminal-v5",
            "status": "passed",
            "stage": SMOKE_STAGE,
            "run_id": run_id,
            "subject": {
                "run_id": reference.subject_run_id,
                "model_id": reference.model_id,
                "revision": reference.revision,
                "architecture": reference.architecture,
                "quant_type": reference.quant_type,
                "mtp": reference.mtp,
                "verified_export_reference_sha256": reference.reference_sha256,
                "q3_shard_count": reference.q3_shard_count,
                "q3_total_bytes": reference.q3_total_bytes,
                "projector_sha256": reference.projector.sha256,
            },
            "smoke_config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "control_plane_file_count": control_plane.file_count,
            "launch_intent_sha256": launch_intent_sha256,
            "invocation": dict(invocation),
            "artifact_rehash": {
                "algorithm": "sha256",
                "worker_count": HASH_WORKERS,
                "elapsed_seconds": artifact_seconds,
                "artifact_count": len(artifacts),
                "artifacts": list(artifacts),
            },
            "runtime": toolchain,
            "host": host,
            "hardware": list(hardware),
            "gpu_topology": gpu_topology,
            "server": {
                "command": command,
                "audit_environment": dict(SERVER_AUDIT_ENVIRONMENT),
                "network_scope": "loopback_only_with_modal_external_network_blocked",
                "post_rehash_load_to_health_seconds": load_seconds,
                "loader_offload": loader.model_dump(mode="json"),
                "artifact_load": artifact_load.model_dump(mode="json"),
                "raw_logit_audit": raw_logit_audit.model_dump(mode="json"),
                "backend_audit": backend_audit.model_dump(mode="json"),
                "properties": properties,
                "server_log_sha256": hashlib.sha256(log_bytes).hexdigest(),
                "cleanup": cleanup,
            },
            "probes": probes,
            "resources": resources,
            "claims": config.claims.model_dump(mode="json"),
            "evidence_policy": config.evidence.model_dump(mode="json"),
            "prompt_text_recorded": False,
            "output_text_recorded": False,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )
        success_path = _safe_child(run_root, "smoke_test.success.json")
        success_publication_started = True
        _publish_success_receipt(
            success_path,
            receipt,
            run_id=run_id,
        )
        success_publication_confirmed = True
        return {
            "status": "passed",
            "stage": SMOKE_STAGE,
            "run_id": run_id,
            "receipt_path": success_path.relative_to(EVIDENCE_MOUNT).as_posix(),
            "receipt_sha256": receipt["receipt_sha256"],
            "warning": "Function return is not evidence; validate the committed success receipt.",
        }
    except BaseException as error:
        if process is not None:
            with suppress(BaseException):
                _terminate_process(process)
        if monitor is not None:
            with suppress(BaseException):
                monitor.stop()
        if log_handle is not None:
            with suppress(BaseException):
                log_handle.close()
        if (
            run_root is not None
            and invocation
            and (
                success_publication_started
                or isinstance(error, _DurablePublicationStateUnknownError)
            )
        ):
            if success_publication_confirmed:
                error.add_note(
                    "Terminal success publication was confirmed. No failure receipt "
                    "was written because a conflicting terminal result is unsafe."
                )
            elif success_publication_started:
                error.add_note(
                    "Terminal success publication started. No failure receipt was "
                    "written because a conflicting terminal result is unsafe."
                )
            else:
                error.add_note(
                    "Durable publication has an unknown committed state. No failure "
                    "receipt was written because a conflicting result is unsafe."
                )
        elif run_root is not None and invocation:
            try:
                _record_failure(
                    run_root,
                    config=config,
                    control_plane=control_plane,
                    invocation=invocation,
                    launch_intent_sha256=launch_intent_sha256,
                    phase=phase,
                    error=error,
                )
            except BaseException as recording_error:
                error.add_note(
                    "Could not commit the immutable smoke failure receipt: "
                    f"{type(recording_error).__module__}.{type(recording_error).__qualname__}"
                )
        raise
