"""Content-addressed execution contracts for the Inkling Modal smoke stage.

The completed export has its own sealed five-stage control plane.  This module
creates a new, one-function control plane for read-only inference.  It keeps
deployment identity, operator acknowledgement, and terminal evidence separate
from the immutable export run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    AfterValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.gguf.inkling_smoke import (
    EXPECTED_PROJECTOR_BYTES,
    EXPECTED_Q3_SHARD_COUNT,
    EXPECTED_Q3_TOTAL_BYTES,
    HISTORICAL_INSTRUMENTATION_PATCH_SHA256,
    INSTRUMENTATION_PATCH_SHA256,
    INSTRUMENTATION_SCHEMA_VERSION,
    LEGACY_CURRENT_INSTRUMENTATION_PATCH_SHA256,
    PINNED_LLAMA_CPP_COMMIT,
    PINNED_MODEL_REVISION,
    InklingSmokeConfig,
    InklingVerifiedExportReference,
)

SMOKE_CONTROL_PLANE_HASH_DOMAIN = b"inkling-smoke-control-plane-v1\0"
_SMOKE_RECEIPT_HASH_DOMAIN_V2 = b"inkling-smoke-terminal-receipt-v2\0"
_SMOKE_RECEIPT_HASH_DOMAIN_V3 = b"inkling-smoke-terminal-receipt-v3\0"
_SMOKE_RECEIPT_HASH_DOMAIN_V4 = b"inkling-smoke-terminal-receipt-v4\0"
SMOKE_RECEIPT_HASH_DOMAIN = b"inkling-smoke-terminal-receipt-v5\0"
SMOKE_PACKAGE_MANIFEST_HASH_DOMAIN = b"inkling-smoke-package-manifest-v2\0"
_SMOKE_HARDWARE_TOPOLOGY_HASH_DOMAIN_V2 = b"inkling-smoke-hardware-topology-v2\0"
_SMOKE_HARDWARE_TOPOLOGY_HASH_DOMAIN_V3 = b"inkling-smoke-hardware-topology-v3\0"
SMOKE_HARDWARE_TOPOLOGY_HASH_DOMAIN = b"inkling-smoke-hardware-topology-v4\0"
SMOKE_SOURCE_TREE_HASH_DOMAIN = b"inkling-smoke-source-tree-v1\0"
SMOKE_STAGE = "smoke_test"
SMOKE_ENVIRONMENT_NAME = "inkling-quant"
SMOKE_WORKSPACE_BUDGET_USD = Decimal("800")
SMOKE_DEPLOYMENT_TAG_HASH_PREFIX_LENGTH = 40
SMOKE_ATTEMPT_REGISTRY_MIN_CREATED_AT_UTC = datetime(2025, 5, 20, tzinfo=UTC)
SMOKE_CONTROL_PLANE_REQUIRED_FILES = (
    "pyproject.toml",
    "uv.lock",
    "configs/experiments/inkling_q3_k_m_modal.yaml",
    "configs/experiments/inkling_q3_k_m_smoke_modal.yaml",
    "configs/experiments/inkling_q3_k_m_verified_export.json",
    "patches/inkling-smoke-a015409.patch",
    "scripts/manage_inkling_smoke_modal.py",
    "scripts/smoke_inkling_modal.py",
)


def smoke_deployment_tag(control_plane_sha256: str) -> str:
    """Return the Modal-safe tag for one full control-plane SHA-256."""

    if (
        not isinstance(control_plane_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", control_plane_sha256) is None
    ):
        raise ValueError("smoke control-plane SHA-256 is invalid")
    return f"iql-smoke-{control_plane_sha256[:SMOKE_DEPLOYMENT_TAG_HASH_PREFIX_LENGTH]}"


def canonical_smoke_attempt_registry_created_at_utc(value: datetime) -> str:
    """Normalize a Modal Dict creation time to canonical UTC microsecond text."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("smoke attempt Dict creation time must be timezone-aware")
    normalized = value.astimezone(UTC)
    if normalized < SMOKE_ATTEMPT_REGISTRY_MIN_CREATED_AT_UTC:
        raise ValueError("smoke attempt Dict uses the unsupported legacy implementation")
    return normalized.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def validate_smoke_attempt_registry_created_at_utc(value: str) -> str:
    """Validate a sealed Modal Dict creation time and its implementation epoch."""

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z", value) is None:
        raise ValueError("smoke attempt Dict creation time must use canonical UTC text")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError("smoke attempt Dict creation time is not a real UTC time") from error
    if canonical_smoke_attempt_registry_created_at_utc(parsed) != value:
        raise ValueError("smoke attempt Dict creation time must use canonical UTC text")
    return value


_SMOKE_HISTORICAL_PATCHED_SOURCE_BLOB_IDS = (
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
    (
        "tools/mtmd/clip.cpp",
        "dbd07081bf73f336a17bd3b8d8359830128c424b",
    ),
    (
        "tools/mtmd/mtmd.cpp",
        "3e81e44143fa635e56e0a757ce1ba33d34d107e4",
    ),
    (
        "tools/server/server-context.cpp",
        "7564ad4e9cfb8e77d610e90c7530121214a4c483",
    ),
)
_SMOKE_PATCHED_SOURCE_BLOB_IDS = (
    *_SMOKE_HISTORICAL_PATCHED_SOURCE_BLOB_IDS,
    (
        "tools/server/server.cpp",
        "20effbb14851b201118843bf14fa5bc51de1e304",
    ),
)
_SMOKE_HISTORICAL_PATCHED_SOURCE_PATHS = tuple(
    sorted(path for path, _git_blob_id in _SMOKE_HISTORICAL_PATCHED_SOURCE_BLOB_IDS)
)
_SMOKE_PATCHED_SOURCE_PATHS = tuple(
    sorted(path for path, _git_blob_id in _SMOKE_PATCHED_SOURCE_BLOB_IDS)
)
_SMOKE_SERVER_AUDIT_ENVIRONMENT = {
    "IQL_SMOKE_BACKEND_AUDIT": "1",
    "IQL_SMOKE_RAW_LOGIT_AUDIT": "1",
    "LLAMA_MEDIA_MARKER": "<__media_iql_smoke_v1__>",
}


def _require_true(value: bool) -> bool:
    if value is not True:
        raise ValueError("evidence flag must be true")
    return value


def _require_false(value: bool) -> bool:
    if value is not False:
        raise ValueError("evidence flag must be false")
    return value


RequiredTrue: TypeAlias = Annotated[StrictBool, AfterValidator(_require_true)]
RequiredFalse: TypeAlias = Annotated[StrictBool, AfterValidator(_require_false)]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _require_inventory_text(value: str, *, label: str) -> str:
    if (
        not value
        or value.strip() != value
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        raise ValueError(f"{label} must be non-empty, trimmed single-line text")
    return value


def canonical_python_package_inventory(
    packages: Iterable[tuple[str, str]],
) -> dict[str, str]:
    """Return one canonical, duplicate-free Python distribution map."""

    inventory: dict[str, str] = {}
    for raw_name, raw_version in packages:
        name = re.sub(
            r"[-_.]+", "-", _require_inventory_text(raw_name, label="package name").casefold()
        )
        version = _require_inventory_text(raw_version, label="package version")
        if re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", name) is None:
            raise ValueError("Python package name is not canonicalizable")
        if name in inventory:
            raise ValueError(f"duplicate Python package distribution: {name}")
        inventory[name] = version
    if not inventory:
        raise ValueError("Python package inventory must not be empty")
    return dict(sorted(inventory.items()))


def parse_dpkg_inventory(payload: bytes) -> dict[str, str]:
    """Parse the exact sorted ``dpkg-query`` inventory used by the image audit."""

    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Debian package inventory is not UTF-8") from error
    packages: dict[str, str] = {}
    for line in text.splitlines():
        if not line:
            continue
        name, separator, version = line.partition("=")
        if not separator:
            raise ValueError("Debian package inventory line lacks a version separator")
        name = _require_inventory_text(name, label="Debian package name")
        version = _require_inventory_text(version, label="Debian package version")
        if re.fullmatch(r"[a-z0-9][a-z0-9+.-]*(?::[a-z0-9][a-z0-9-]*)?", name) is None:
            raise ValueError("Debian package name is invalid")
        if name in packages:
            raise ValueError(f"duplicate Debian package: {name}")
        packages[name] = version
    if not packages:
        raise ValueError("Debian package inventory must not be empty")
    return dict(sorted(packages.items()))


def parse_nvcc_version(payload: str) -> str:
    """Extract the one exact CUDA compiler build version from ``nvcc`` output."""

    versions = re.findall(r"(?<![A-Za-z0-9])V([0-9]+\.[0-9]+\.[0-9]+)(?![A-Za-z0-9])", payload)
    if len(versions) != 1:
        raise ValueError("nvcc output must contain exactly one compiler build version")
    return f"V{versions[0]}"


def parse_proc_cpu_model(payload: str) -> str:
    """Return a deterministic CPU model identity from Linux ``/proc/cpuinfo``."""

    fields: dict[str, list[str]] = {}
    for line in payload.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        normalized_key = key.strip().casefold()
        normalized_value = value.strip()
        if normalized_value:
            fields.setdefault(normalized_key, []).append(normalized_value)
    for key in ("model name", "hardware", "processor"):
        candidates = fields.get(key, [])
        if key == "processor":
            candidates = [value for value in candidates if not value.isdecimal()]
        if candidates:
            unique = sorted(set(candidates))
            return " | ".join(unique)
    raise ValueError("Linux CPU inventory contains no model identity")


def parse_proc_mem_total_bytes(payload: str) -> int:
    """Return physical RAM capacity from the one Linux ``MemTotal`` field."""

    matches = re.findall(r"^MemTotal:\s+([1-9][0-9]*)\s+kB$", payload, re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("Linux memory inventory must contain one MemTotal field")
    return int(matches[0]) * 1024


def parse_cgroup_cpu_quota_millicores(payload: str) -> int | None:
    """Parse cgroup v2 ``cpu.max`` or normalized v1 quota/period text."""

    fields = payload.split()
    if len(fields) != 2:
        raise ValueError("cgroup CPU quota inventory must contain quota and period")
    quota_text, period_text = fields
    if re.fullmatch(r"[1-9][0-9]*", period_text) is None:
        raise ValueError("cgroup CPU period is invalid")
    if quota_text in {"max", "-1"}:
        return None
    if re.fullmatch(r"[1-9][0-9]*", quota_text) is None:
        raise ValueError("cgroup CPU quota is invalid")
    numerator = int(quota_text) * 1000
    period = int(period_text)
    if numerator % period:
        raise ValueError("cgroup CPU quota is not exactly representable in millicores")
    return numerator // period


def parse_cgroup_memory_limit_bytes(payload: str) -> int | None:
    """Parse cgroup v2 or v1 memory-limit text without inventing a limit."""

    value = payload.strip()
    if value in {"max", "-1"}:
        return None
    if re.fullmatch(r"[1-9][0-9]*", value) is None:
        raise ValueError("cgroup memory limit is invalid")
    parsed = int(value)
    # Linux cgroup v1 reports an effectively unlimited 64-bit memory limit as
    # LONG_MAX rounded down to a page boundary instead of as ``-1``.
    if parsed >= (1 << 63) - 4096:
        return None
    return parsed


def _canonical_proc_path(value: str, *, label: str) -> PurePosixPath:
    if not value or "\x00" in value or not value.startswith("/"):
        raise ValueError(f"{label} must be an absolute POSIX path")
    path = PurePosixPath(value)
    if path.as_posix() != value or any(part in {".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be canonical")
    return path


def _decode_mountinfo_path(value: str, *, label: str) -> PurePosixPath:
    escapes = {
        r"\040": " ",
        r"\011": "\t",
        r"\012": "\n",
        r"\134": "\\",
    }
    decoded = value
    for encoded, plain in escapes.items():
        decoded = decoded.replace(encoded, plain)
    if re.search(r"\\[0-7]{3}", decoded):
        raise ValueError(f"{label} contains an unsupported mountinfo escape")
    return _canonical_proc_path(decoded, label=label)


def resolve_process_cgroup_file(
    cgroup_payload: str,
    mountinfo_payload: str,
    *,
    controller: Literal["cpu", "memory"],
    filename: str,
) -> tuple[Path, Literal[1, 2], Path]:
    """Resolve one cgroup control file for the current process namespace."""

    if not filename or PurePosixPath(filename).name != filename or "/" in filename:
        raise ValueError("cgroup control filename must be one path component")

    unified_path: PurePosixPath | None = None
    legacy_paths: list[PurePosixPath] = []
    hierarchy_ids: set[str] = set()
    lines = [line for line in cgroup_payload.splitlines() if line]
    if not lines:
        raise ValueError("process cgroup membership inventory is empty")
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            raise ValueError("process cgroup membership line is malformed")
        hierarchy_id, controller_text, path_text = fields
        if re.fullmatch(r"0|[1-9][0-9]*", hierarchy_id) is None:
            raise ValueError("process cgroup hierarchy ID is invalid")
        if hierarchy_id in hierarchy_ids:
            raise ValueError("process cgroup hierarchy ID is duplicated")
        hierarchy_ids.add(hierarchy_id)
        membership = _canonical_proc_path(path_text, label="process cgroup membership")
        controllers = controller_text.split(",") if controller_text else []
        if any(
            re.fullmatch(r"(?:[A-Za-z0-9_.-]+|name=[A-Za-z0-9_.-]+)", item) is None
            for item in controllers
        ) or len(controllers) != len(set(controllers)):
            raise ValueError("process cgroup controller inventory is invalid")
        if hierarchy_id == "0":
            if controllers or unified_path is not None:
                raise ValueError("unified cgroup membership is invalid or duplicated")
            unified_path = membership
        elif controller in controllers:
            legacy_paths.append(membership)

    if len(legacy_paths) > 1:
        raise ValueError(f"process cgroup has multiple {controller} memberships")
    if legacy_paths:
        membership = legacy_paths[0]
        version: Literal[1, 2] = 1
    elif unified_path is not None:
        membership = unified_path
        version = 2
    else:
        raise ValueError(f"process cgroup has no {controller} membership")

    candidates: list[tuple[Path, Path, Path]] = []
    for line in mountinfo_payload.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator < 6 or len(fields) < separator + 4:
            raise ValueError("cgroup mountinfo line is malformed")
        filesystem_type = fields[separator + 1]
        super_options = set(fields[separator + 3].split(","))
        if version == 2:
            if filesystem_type != "cgroup2":
                continue
        elif filesystem_type != "cgroup" or controller not in super_options:
            continue
        root = _decode_mountinfo_path(fields[3], label="cgroup mount root")
        mount_point = _decode_mountinfo_path(fields[4], label="cgroup mount point")
        try:
            relative = membership.relative_to(root)
        except ValueError:
            continue
        resolved = mount_point.joinpath(relative, filename)
        candidates.append(
            (
                Path(root.as_posix()),
                Path(resolved.as_posix()),
                Path(mount_point.as_posix()),
            )
        )

    if not candidates:
        raise ValueError(f"process cgroup {controller} membership has no matching mount")
    if len(candidates) != 1:
        raise ValueError(f"process cgroup {controller} membership has ambiguous mounts")
    _mount_root, path, selected_mount_point = candidates[0]
    return path, version, selected_mount_point


def resolve_current_process_cgroup_hierarchy_paths(
    *,
    proc_self_cgroup: str,
    proc_self_mountinfo: str,
) -> dict[str, tuple[Path, Path]]:
    """Return each process cgroup leaf and its inclusive mount boundary."""

    cpu_file, _cpu_version, cpu_root = resolve_process_cgroup_file(
        proc_self_cgroup,
        proc_self_mountinfo,
        controller="cpu",
        filename="cgroup.procs",
    )
    memory_file, _memory_version, memory_root = resolve_process_cgroup_file(
        proc_self_cgroup,
        proc_self_mountinfo,
        controller="memory",
        filename="cgroup.procs",
    )
    return {
        "cpu": (cpu_file.parent, cpu_root),
        "memory": (memory_file.parent, memory_root),
    }


def resolve_current_process_cgroup_leaf_paths(
    *,
    proc_self_cgroup: str,
    proc_self_mountinfo: str,
) -> dict[str, Path]:
    """Resolve the current process CPU and memory cgroup leaf directories."""

    hierarchies = resolve_current_process_cgroup_hierarchy_paths(
        proc_self_cgroup=proc_self_cgroup,
        proc_self_mountinfo=proc_self_mountinfo,
    )
    return {controller: leaf for controller, (leaf, _hierarchy_root) in hierarchies.items()}


def immutable_source_tree_identity(root: Path) -> tuple[int, str]:
    """Hash every immutable regular file in one runtime source tree."""

    try:
        root_status = root.lstat()
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("runtime source-tree root is unavailable") from error
    if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(root_status.st_mode):
        raise ValueError("runtime source-tree root must be a real directory")
    if resolved_root != root:
        raise ValueError("runtime source-tree root must be resolved and canonical")

    files: list[dict[str, object]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        try:
            status = path.lstat()
        except OSError as error:
            raise ValueError("runtime source-tree entry is unavailable") from error
        if stat.S_ISDIR(status.st_mode):
            continue
        if not stat.S_ISREG(status.st_mode):
            raise ValueError("runtime source tree contains a symlink or special file")
        if path.suffix in {".pyc", ".pyo"}:
            continue
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError("runtime source-tree file is unreadable") from error
        if len(payload) != status.st_size:
            raise ValueError("runtime source-tree file changed while it was hashed")
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": status.st_size,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not files:
        raise ValueError("runtime source tree contains no immutable regular files")
    manifest = {
        "schema_version": "inkling-smoke-source-tree-v1",
        "files": files,
    }
    digest = hashlib.sha256(
        SMOKE_SOURCE_TREE_HASH_DOMAIN + _canonical_json(manifest).encode("utf-8")
    ).hexdigest()
    return len(files), digest


class SmokeControlPlaneFile(StrictFrozenModel):
    """One regular file bound into the smoke implementation identity."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class SmokeControlPlaneProvenance(StrictFrozenModel):
    """Canonical manifest of all code and local launch inputs for one smoke run."""

    schema_version: Literal["inkling-smoke-control-plane-v1"] = "inkling-smoke-control-plane-v1"
    tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_count: int = Field(gt=0)
    files: tuple[SmokeControlPlaneFile, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> SmokeControlPlaneProvenance:
        if self.file_count != len(self.files):
            raise ValueError("smoke control-plane file count does not match its manifest")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("smoke control-plane paths must be sorted and unique")
        if self.tree_sha256 != smoke_control_plane_tree_sha256(self.files):
            raise ValueError("smoke control-plane tree hash does not match its manifest")
        return self

    def canonical_json(self) -> str:
        """Return canonical JSON without a trailing newline for Modal arguments."""

        return _canonical_json(self.model_dump(mode="json"))


class SmokeLaunchDeploymentIdentity(StrictFrozenModel):
    """Exact sealed Modal deployment authorized for one smoke call."""

    app_name: StrictStr = Field(pattern=r"^inkling-q3-smoke-[0-9a-f]{12}$")
    environment_name: Literal["inkling-quant"] = "inkling-quant"
    deployment_version: StrictInt = Field(gt=0)
    deployment_tag: StrictStr = Field(
        pattern=(
            rf"^iql-smoke-[0-9a-f]"
            rf"{{{SMOKE_DEPLOYMENT_TAG_HASH_PREFIX_LENGTH}}}$"
        )
    )
    function_id: StrictStr = Field(pattern=r"^fu-[A-Za-z0-9]+$")
    function_name: Literal["smoke_test"] = "smoke_test"
    attempt_registry_name: Literal["inkling-smoke-attempt-registry-v1"]
    attempt_registry_id: StrictStr = Field(pattern=r"^di-[A-Za-z0-9]+$")
    attempt_registry_created_at_utc: StrictStr

    @field_validator("attempt_registry_created_at_utc")
    @classmethod
    def registry_creation_is_supported(cls, value: str) -> str:
        return validate_smoke_attempt_registry_created_at_utc(value)


class SmokeLaunchAcknowledgement(StrictFrozenModel):
    """Operator acknowledgement bound to exactly one accepted smoke call."""

    schema_version: Literal["inkling-smoke-launch-ack-v4"] = "inkling-smoke-launch-ack-v4"
    smoke_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified_export_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control_plane_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    launch_intent_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    deployment: SmokeLaunchDeploymentIdentity
    environment_name: Literal["inkling-quant"] = "inkling-quant"
    workspace_budget_usd: Decimal
    billing_cycle_end_utc: str
    confirmation: Literal["dashboard-workspace-hard-budget-confirmed"] = (
        "dashboard-workspace-hard-budget-confirmed"
    )

    @field_validator("billing_cycle_end_utc")
    @classmethod
    def cycle_end_is_canonical_utc(cls, value: str) -> str:
        try:
            parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
        except ValueError as error:
            raise ValueError(
                "billing cycle end must use a real YYYY-MM-DDTHH:MM:SSZ time"
            ) from error
        if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
            raise ValueError("billing cycle end must use canonical UTC text")
        return value

    @model_validator(mode="after")
    def exact_external_gate(self) -> SmokeLaunchAcknowledgement:
        if self.workspace_budget_usd != SMOKE_WORKSPACE_BUDGET_USD:
            raise ValueError("smoke launch acknowledgement has the wrong workspace cap")
        if self.deployment.environment_name != self.environment_name:
            raise ValueError("smoke launch acknowledgement has different environments")
        if self.deployment.app_name != (f"inkling-q3-smoke-{self.control_plane_sha256[:12]}"):
            raise ValueError("smoke launch acknowledgement has the wrong app identity")
        if self.deployment.deployment_tag != smoke_deployment_tag(self.control_plane_sha256):
            raise ValueError("smoke launch acknowledgement has the wrong deployment tag")
        return self

    def canonical_json(self) -> str:
        """Return canonical JSON without a trailing newline for Modal arguments."""

        return _canonical_json(self.model_dump(mode="json"))


class _SmokeReceiptModel(StrictFrozenModel):
    """Base for fail-closed terminal smoke evidence records."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class SmokeSafeFailureSignals(_SmokeReceiptModel):
    """Non-sensitive failure categories derived from the server log."""

    out_of_memory_observed: StrictBool
    no_usable_gpu_observed: StrictBool
    model_load_failure_observed: StrictBool
    projector_load_failure_observed: StrictBool
    unsupported_architecture_observed: StrictBool


class SmokeSubprocessFailureEvidence(_SmokeReceiptModel):
    """Safe identity of one failed allowlisted preflight subprocess."""

    schema_version: Literal["inkling-smoke-subprocess-failure-v1"]
    command_id: Literal[
        "python_package_inventory_v1",
        "llama_cpp_git_changed_paths_v1",
        "llama_cpp_git_patched_diff_v1",
        "dpkg_inventory_v1",
        "cuda_driver_linkage_v1",
        "cuda_compiler_version_v1",
        "nvidia_smi_identity_v1",
    ]
    return_code: StrictInt
    stdout_size_bytes: StrictInt = Field(ge=0)
    stdout_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    stderr_size_bytes: StrictInt = Field(ge=0)
    stderr_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    stdout_recorded: RequiredFalse
    stderr_recorded: RequiredFalse

    @model_validator(mode="after")
    def return_code_is_failure(self) -> SmokeSubprocessFailureEvidence:
        if self.return_code == 0:
            raise ValueError("subprocess failure return code must be nonzero")
        return self


class SmokeInvocationEvidence(_SmokeReceiptModel):
    """The single claimed Modal invocation that produced the receipt."""

    schema_version: Literal["inkling-smoke-invocation-v3"]
    run_id: StrictStr = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    stage: Literal["smoke_test"]
    sequence: StrictInt = Field(ge=1, le=1)
    limit: StrictInt = Field(ge=1, le=1)
    call_id: StrictStr = Field(pattern=r"^fc-[A-Za-z0-9]+$")
    input_id: StrictStr = Field(pattern=r"^in-[A-Za-z0-9]+(?::[0-9]+-[0-9]+)?$")
    task_id: StrictStr = Field(pattern=r"^ta-[A-Za-z0-9]+$")
    launch_intent_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    smoke_config_hash: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    control_plane_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    post_spawn_acceptance_path: StrictStr = Field(
        pattern=r"^control/post-spawn-acceptances/[0-9a-f]{64}\.json$"
    )
    post_spawn_acceptance_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_registry_name: Literal["inkling-smoke-attempt-registry-v1"]
    attempt_registry_id: StrictStr = Field(pattern=r"^di-[A-Za-z0-9]+$")
    attempt_registry_created_at_utc: StrictStr
    attempt_registry_key: StrictStr
    attempt_registry_claim_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    attempt_claim_path: Literal["control/smoke_test.attempt.claim.json"]
    attempt_claim_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_history_path: StrictStr = Field(
        pattern=r"^control/history/smoke_test\.attempt\.1\.[0-9a-f]{64}\.json$"
    )
    invocation_history_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def acceptance_path_matches_launch(self) -> SmokeInvocationEvidence:
        expected = f"control/post-spawn-acceptances/{self.launch_intent_sha256}.json"
        if self.post_spawn_acceptance_path != expected:
            raise ValueError("invocation acceptance path differs from its launch intent")
        validate_smoke_attempt_registry_created_at_utc(self.attempt_registry_created_at_utc)
        if self.attempt_registry_key != f"{self.run_id}:{self.stage}":
            raise ValueError("invocation attempt registry key differs from its run and stage")
        if self.attempt_claim_sha256 != self.attempt_registry_claim_sha256:
            raise ValueError("invocation attempt claim hash differs from its registry claim")
        return self


class _SmokeFailureReceipt(_SmokeReceiptModel):
    """Fields shared by terminal failure receipts."""

    status: Literal["failed"]
    stage: Literal["smoke_test"]
    run_id: StrictStr = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    subject_run_id: StrictStr = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    smoke_config_hash: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    verified_export_reference_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    control_plane_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    launch_intent_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    call_id: StrictStr = Field(pattern=r"^fc-[A-Za-z0-9]+$")
    input_id: StrictStr = Field(pattern=r"^in-[A-Za-z0-9]+(?::[0-9]+-[0-9]+)?$")
    task_id: StrictStr = Field(pattern=r"^ta-[A-Za-z0-9]+$")
    failure_phase: Literal[
        "commit_attempt",
        "verify_runtime_and_allocation",
        "verify_subject_hashes",
        "revalidate_allocation_before_server_start",
        "start_server",
        "validate_server_properties",
        "run_deterministic_probes",
        "stop_server",
        "publish_success",
    ]
    exception_type: StrictStr = Field(
        min_length=3,
        max_length=256,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$",
    )
    server_log_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    safe_failure_signals: SmokeSafeFailureSignals
    prompt_text_recorded: RequiredFalse
    output_text_recorded: RequiredFalse
    receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class SmokeFailureReceiptV2(_SmokeFailureReceipt):
    """Historical immutable failure evidence using the v2 hash domain."""

    schema_version: Literal["inkling-smoke-terminal-v2"]


class SmokeFailureReceiptV3(_SmokeFailureReceipt):
    """Historical immutable failure evidence using the v3 hash domain."""

    schema_version: Literal["inkling-smoke-terminal-v3"]
    invocation: SmokeInvocationEvidence

    @model_validator(mode="after")
    def invocation_matches_top_level_identity(self) -> SmokeFailureReceiptV3:
        if (
            self.invocation.run_id,
            self.invocation.call_id,
            self.invocation.input_id,
            self.invocation.task_id,
            self.invocation.launch_intent_sha256,
            self.invocation.smoke_config_hash,
            self.invocation.control_plane_sha256,
        ) != (
            self.run_id,
            self.call_id,
            self.input_id,
            self.task_id,
            self.launch_intent_sha256,
            self.smoke_config_hash,
            self.control_plane_sha256,
        ):
            raise ValueError("failure invocation differs from the top-level receipt identity")
        return self


class SmokeFailureReceiptV4(_SmokeFailureReceipt):
    """Historical version 4 failure evidence with safe subprocess diagnostics."""

    schema_version: Literal["inkling-smoke-terminal-v4"]
    invocation: SmokeInvocationEvidence
    safe_subprocess_failure: SmokeSubprocessFailureEvidence | None

    @model_validator(mode="after")
    def invocation_and_failure_are_consistent(self) -> SmokeFailureReceiptV4:
        if (
            self.invocation.run_id,
            self.invocation.call_id,
            self.invocation.input_id,
            self.invocation.task_id,
            self.invocation.launch_intent_sha256,
            self.invocation.smoke_config_hash,
            self.invocation.control_plane_sha256,
        ) != (
            self.run_id,
            self.call_id,
            self.input_id,
            self.task_id,
            self.launch_intent_sha256,
            self.smoke_config_hash,
            self.control_plane_sha256,
        ):
            raise ValueError("failure invocation differs from the top-level receipt identity")
        is_called_process_error = self.exception_type == "subprocess.CalledProcessError"
        if is_called_process_error != (self.safe_subprocess_failure is not None):
            raise ValueError("subprocess failure evidence differs from the exception type")
        return self


class SmokeFailureReceiptV5(_SmokeFailureReceipt):
    """Version 5 failure evidence with corrected logit instrumentation."""

    schema_version: Literal["inkling-smoke-terminal-v5"]
    invocation: SmokeInvocationEvidence
    safe_subprocess_failure: SmokeSubprocessFailureEvidence | None

    @model_validator(mode="after")
    def invocation_and_failure_are_consistent(self) -> SmokeFailureReceiptV5:
        if (
            self.invocation.run_id,
            self.invocation.call_id,
            self.invocation.input_id,
            self.invocation.task_id,
            self.invocation.launch_intent_sha256,
            self.invocation.smoke_config_hash,
            self.invocation.control_plane_sha256,
        ) != (
            self.run_id,
            self.call_id,
            self.input_id,
            self.task_id,
            self.launch_intent_sha256,
            self.smoke_config_hash,
            self.control_plane_sha256,
        ):
            raise ValueError("failure invocation differs from the top-level receipt identity")
        is_called_process_error = self.exception_type == "subprocess.CalledProcessError"
        if is_called_process_error != (self.safe_subprocess_failure is not None):
            raise ValueError("subprocess failure evidence differs from the exception type")
        return self


SmokeFailureReceipt: TypeAlias = (
    SmokeFailureReceiptV2 | SmokeFailureReceiptV3 | SmokeFailureReceiptV4 | SmokeFailureReceiptV5
)


class SmokeReceiptSubject(_SmokeReceiptModel):
    """Identity of the exact immutable export exercised by the smoke run."""

    run_id: StrictStr = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,127}$")
    model_id: Literal["thinkingmachines/Inkling"]
    revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"]
    architecture: Literal["InklingForConditionalGeneration"]
    quant_type: Literal["Q3_K_M"]
    mtp: Literal["omitted_unsupported"]
    verified_export_reference_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    q3_shard_count: StrictInt = Field(ge=49, le=49)
    q3_total_bytes: StrictInt = Field(ge=451_035_400_288, le=451_035_400_288)
    projector_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class SmokeArtifactIdentity(_SmokeReceiptModel):
    """One artifact whose bytes were rehashed before model load."""

    path: StrictStr = Field(min_length=1)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: StrictInt = Field(gt=0)

    @field_validator("path")
    @classmethod
    def path_is_canonical_relative(cls, value: str) -> str:
        parsed = Path(value)
        if (
            parsed.is_absolute()
            or "\\" in value
            or "//" in value
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or parsed.as_posix() != value
        ):
            raise ValueError("smoke artifact path must be canonical relative POSIX text")
        return value


class SmokeArtifactRehashEvidence(_SmokeReceiptModel):
    """Complete byte identities observed before server startup."""

    algorithm: Literal["sha256"]
    worker_count: StrictInt = Field(ge=8, le=8)
    elapsed_seconds: float = Field(gt=0)
    artifact_count: StrictInt = Field(ge=54, le=54)
    artifacts: tuple[SmokeArtifactIdentity, ...]

    @field_validator("elapsed_seconds", mode="before")
    @classmethod
    def elapsed_is_finite_number(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("artifact rehash duration must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError("artifact rehash duration must be finite")
        return value

    @model_validator(mode="after")
    def count_matches_inventory(self) -> SmokeArtifactRehashEvidence:
        if len(self.artifacts) != self.artifact_count:
            raise ValueError("artifact rehash count differs from its inventory")
        paths = tuple(artifact.path for artifact in self.artifacts)
        if len(paths) != len(set(paths)):
            raise ValueError("artifact rehash inventory contains duplicate paths")
        return self


class SmokeRuntimeBinaryEvidence(_SmokeReceiptModel):
    """Identity of one executable in the pinned llama.cpp build."""

    name: Literal["llama-cli", "llama-server", "llama-bench", "llama-perplexity"]
    path: StrictStr = Field(pattern=r"^/opt/llama\.cpp/build/bin/llama-[a-z]+$")
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: StrictInt = Field(gt=0)


class SmokeBaseSourceBlobIdentity(_SmokeReceiptModel):
    """Pinned Git blob identity for one instrumented llama.cpp source file."""

    path: StrictStr = Field(min_length=1)
    git_blob_id: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")


def smoke_package_manifest_sha256(
    *,
    python_implementation: str,
    python_version: str,
    python_executable_path: str,
    python_executable_sha256: str,
    python_inventory_scope: str,
    python_purelib: str,
    python_inventory_sha256: str,
    python_packages: Mapping[str, str],
    modal_runtime_version: str,
    modal_package_root: str,
    modal_package_tree_schema_version: str,
    modal_package_file_count: int,
    modal_package_tree_sha256: str,
    dpkg_inventory_sha256: str,
    dpkg_packages: Mapping[str, str],
    nvcc_version: str,
    nvcc_version_sha256: str,
) -> str:
    """Hash the exact package maps and their audited raw inventories."""

    payload = {
        "schema_version": "inkling-smoke-package-manifest-v2",
        "python_implementation": python_implementation,
        "python_version": python_version,
        "python_executable_path": python_executable_path,
        "python_executable_sha256": python_executable_sha256,
        "python_inventory_scope": python_inventory_scope,
        "python_purelib": python_purelib,
        "python_inventory_sha256": python_inventory_sha256,
        "python_packages": dict(python_packages),
        "modal_runtime_version": modal_runtime_version,
        "modal_package_root": modal_package_root,
        "modal_package_tree_schema_version": modal_package_tree_schema_version,
        "modal_package_file_count": modal_package_file_count,
        "modal_package_tree_sha256": modal_package_tree_sha256,
        "dpkg_inventory_sha256": dpkg_inventory_sha256,
        "dpkg_packages": dict(dpkg_packages),
        "nvcc_version": nvcc_version,
        "nvcc_version_sha256": nvcc_version_sha256,
    }
    return hashlib.sha256(
        SMOKE_PACKAGE_MANIFEST_HASH_DOMAIN + _canonical_json(payload).encode("utf-8")
    ).hexdigest()


class SmokeRuntimeEvidence(_SmokeReceiptModel):
    """Pinned runtime and instrumentation provenance."""

    llama_cpp_repository: Literal["https://github.com/danielhanchen/llama.cpp.git"]
    llama_cpp_commit: Literal["a015409e6c27b84f60d688823d4c0126a11571fd"]
    cuda_image: Literal[
        "nvidia/cuda:13.1.2-devel-ubuntu24.04@sha256:952e42d23230610a2714c8484f38e9c934ed68e6f9c9c7fac62dcd5f98858a6e"
    ]
    cuda_driver_library_path: StrictStr = Field(min_length=1)
    cuda_driver_library_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    cmake_definitions: tuple[StrictStr, ...]
    binaries: tuple[SmokeRuntimeBinaryEvidence, ...]
    python_implementation: Literal["CPython"]
    python_version: StrictStr = Field(pattern=r"^3\.12\.[0-9]+$")
    python_executable_path: StrictStr = Field(min_length=1)
    python_executable_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    python_inventory_scope: Literal["image_sysconfig_purelib_v1"]
    python_purelib: StrictStr = Field(min_length=1)
    python_inventory_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    python_packages: dict[StrictStr, StrictStr]
    modal_runtime_version: Literal["1.5.0"]
    modal_package_root: Literal["/pkg/modal"]
    modal_package_tree_schema_version: Literal["inkling-smoke-source-tree-v1"]
    modal_package_file_count: StrictInt = Field(gt=0)
    modal_package_tree_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    dpkg_inventory_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    dpkg_packages: dict[StrictStr, StrictStr]
    package_manifest_schema_version: Literal["inkling-smoke-package-manifest-v2"]
    package_manifest_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    nvcc_version: StrictStr = Field(pattern=r"^V[0-9]+\.[0-9]+\.[0-9]+$")
    nvcc_version_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    instrumentation_schema_version: Literal[
        "inkling-llama-smoke-instrumentation-v1",
        "inkling-llama-smoke-instrumentation-v2",
    ]
    instrumentation_patch_path: Literal["/root/inkling-smoke-a015409.patch"]
    instrumentation_patch_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    patched_source_paths: tuple[StrictStr, ...]
    patched_diff_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    base_source_blob_ids: tuple[SmokeBaseSourceBlobIdentity, ...]

    @field_validator("cuda_driver_library_path", "python_executable_path")
    @classmethod
    def canonical_runtime_file_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        stub_root = PurePosixPath("/usr/local/cuda/lib64/stubs")
        link_root = PurePosixPath("/opt/iql-cuda-driver-link")
        if (
            "\x00" in value
            or "\\" in value
            or "//" in value
            or not path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != value
        ):
            raise ValueError("runtime file path must be canonical and absolute")
        if value == "/pkg/modal":
            raise ValueError("runtime file path must identify a file")
        if (
            "stubs" in path.parts
            or path in (stub_root, link_root)
            or stub_root in path.parents
            or link_root in path.parents
        ):
            raise ValueError("runtime file path must not identify the build stub")
        return value

    @field_validator("python_packages")
    @classmethod
    def python_package_map_is_canonical(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        if not value or tuple(value) != tuple(sorted(value)):
            raise ValueError("Python package map must be non-empty and sorted")
        expected = canonical_python_package_inventory(value.items())
        if expected != value:
            raise ValueError("Python package map is not canonical")
        return value

    @field_validator("dpkg_packages")
    @classmethod
    def debian_package_map_is_canonical(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        if not value or tuple(value) != tuple(sorted(value)):
            raise ValueError("Debian package map must be non-empty and sorted")
        encoded = "".join(f"{name}={version}\n" for name, version in value.items())
        if parse_dpkg_inventory(encoded.encode("utf-8")) != value:
            raise ValueError("Debian package map is not canonical")
        return value

    @model_validator(mode="after")
    def exact_build_inventory(self) -> SmokeRuntimeEvidence:
        expected_names = (
            "llama-cli",
            "llama-server",
            "llama-bench",
            "llama-perplexity",
        )
        names = tuple(binary.name for binary in self.binaries)
        if names != expected_names:
            raise ValueError("runtime binaries must be the exact ordered checked set")
        expected_paths = tuple(f"/opt/llama.cpp/build/bin/{name}" for name in expected_names)
        if tuple(binary.path for binary in self.binaries) != expected_paths:
            raise ValueError("runtime binary paths differ from the pinned build")
        observed_blobs = tuple(
            (identity.path, identity.git_blob_id) for identity in self.base_source_blob_ids
        )
        expected_blobs: tuple[tuple[str, str], ...]
        if self.instrumentation_patch_sha256 == HISTORICAL_INSTRUMENTATION_PATCH_SHA256:
            expected_blobs = _SMOKE_HISTORICAL_PATCHED_SOURCE_BLOB_IDS
            expected_paths = _SMOKE_HISTORICAL_PATCHED_SOURCE_PATHS
            expected_instrumentation_schema = "inkling-llama-smoke-instrumentation-v1"
        elif self.instrumentation_patch_sha256 == LEGACY_CURRENT_INSTRUMENTATION_PATCH_SHA256:
            expected_blobs = _SMOKE_PATCHED_SOURCE_BLOB_IDS
            expected_paths = _SMOKE_PATCHED_SOURCE_PATHS
            expected_instrumentation_schema = "inkling-llama-smoke-instrumentation-v1"
        elif self.instrumentation_patch_sha256 == INSTRUMENTATION_PATCH_SHA256:
            expected_blobs = _SMOKE_PATCHED_SOURCE_BLOB_IDS
            expected_paths = _SMOKE_PATCHED_SOURCE_PATHS
            expected_instrumentation_schema = INSTRUMENTATION_SCHEMA_VERSION
        else:
            raise ValueError("runtime instrumentation patch SHA-256 is unsupported")
        if self.instrumentation_schema_version != expected_instrumentation_schema:
            raise ValueError("runtime instrumentation schema differs from its source patch")
        if observed_blobs != expected_blobs:
            raise ValueError("runtime base-source blob identities differ from their patch")
        if self.patched_source_paths != expected_paths:
            raise ValueError("runtime patched-source inventory differs from its patch")
        expected_package_manifest = smoke_package_manifest_sha256(
            python_implementation=self.python_implementation,
            python_version=self.python_version,
            python_executable_path=self.python_executable_path,
            python_executable_sha256=self.python_executable_sha256,
            python_inventory_scope=self.python_inventory_scope,
            python_purelib=self.python_purelib,
            python_inventory_sha256=self.python_inventory_sha256,
            python_packages=self.python_packages,
            modal_runtime_version=self.modal_runtime_version,
            modal_package_root=self.modal_package_root,
            modal_package_tree_schema_version=self.modal_package_tree_schema_version,
            modal_package_file_count=self.modal_package_file_count,
            modal_package_tree_sha256=self.modal_package_tree_sha256,
            dpkg_inventory_sha256=self.dpkg_inventory_sha256,
            dpkg_packages=self.dpkg_packages,
            nvcc_version=self.nvcc_version,
            nvcc_version_sha256=self.nvcc_version_sha256,
        )
        if self.package_manifest_sha256 != expected_package_manifest:
            raise ValueError("runtime package manifest hash differs from its package maps")
        return self


class SmokeCudaPeerEdgeEvidence(_SmokeReceiptModel):
    """One directed CUDA Driver peer-access capability observation."""

    source_cuda_ordinal: StrictInt = Field(ge=0)
    source_uuid: StrictStr = Field(
        pattern=r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$"
    )
    destination_cuda_ordinal: StrictInt = Field(ge=0)
    destination_uuid: StrictStr = Field(
        pattern=r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$"
    )
    can_access_peer: StrictBool
    performance_rank: StrictInt = Field(ge=0)
    access_supported: StrictBool
    native_atomic_supported: StrictBool
    cuda_array_access_supported: StrictBool
    only_partial_native_atomic_supported: StrictBool

    @model_validator(mode="after")
    def directed_edge_is_consistent(self) -> SmokeCudaPeerEdgeEvidence:
        if (
            self.source_cuda_ordinal == self.destination_cuda_ordinal
            or self.source_uuid == self.destination_uuid
        ):
            raise ValueError("CUDA peer evidence must describe two different devices")
        if self.can_access_peer != self.access_supported:
            raise ValueError("CUDA peer access APIs returned different support values")
        if self.native_atomic_supported and self.only_partial_native_atomic_supported:
            raise ValueError("CUDA peer atomic support cannot be both full and partial")
        return self


class SmokeNvidiaSmiTopologyDiagnostic(_SmokeReceiptModel):
    """Hashed, non-authoritative result from ``nvidia-smi topo -m``."""

    schema_version: Literal["inkling-smoke-nvidia-smi-topology-diagnostic-v1"]
    argv: tuple[Literal["nvidia-smi"], Literal["topo"], Literal["-m"]]
    status: Literal[
        "available",
        "command_failed",
        "empty_output",
        "timed_out",
        "unavailable",
        "invalid_result",
    ]
    return_code: StrictInt | None
    stdout_size_bytes: StrictInt = Field(ge=0)
    stdout_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    stderr_size_bytes: StrictInt = Field(ge=0)
    stderr_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    stdout_recorded: RequiredFalse
    stderr_recorded: RequiredFalse

    @model_validator(mode="after")
    def outcome_matches_return_code(self) -> SmokeNvidiaSmiTopologyDiagnostic:
        if self.status == "available":
            if self.return_code != 0 or self.stdout_size_bytes == 0:
                raise ValueError("available nvidia-smi topology evidence is incomplete")
        elif self.status == "command_failed":
            if self.return_code is None or self.return_code == 0:
                raise ValueError("failed nvidia-smi topology evidence needs a nonzero return code")
        elif self.status == "empty_output":
            if self.return_code != 0:
                raise ValueError("empty nvidia-smi topology evidence needs a zero return code")
        elif self.return_code is not None:
            raise ValueError("unavailable nvidia-smi topology evidence cannot have a return code")
        if self.status in {"unavailable", "invalid_result"} and (
            self.stdout_size_bytes != 0 or self.stderr_size_bytes != 0
        ):
            raise ValueError("unavailable nvidia-smi topology evidence cannot contain output")
        return self


class SmokeGpuTopologyEvidence(_SmokeReceiptModel):
    """Process-visible CUDA peer capability topology for the two tested GPUs."""

    schema_version: Literal["inkling-smoke-gpu-topology-v1"]
    protocol: Literal["cuda-driver-p2p-v1+nvidia-smi-topo-diagnostic-v1"]
    cuda_driver_api_version: StrictInt = Field(gt=0)
    edges: tuple[SmokeCudaPeerEdgeEvidence, SmokeCudaPeerEdgeEvidence]
    nvidia_smi_topology: SmokeNvidiaSmiTopologyDiagnostic

    @model_validator(mode="after")
    def exact_two_gpu_topology(self) -> SmokeGpuTopologyEvidence:
        ordinal_pairs = tuple(
            (edge.source_cuda_ordinal, edge.destination_cuda_ordinal) for edge in self.edges
        )
        if ordinal_pairs != ((0, 1), (1, 0)):
            raise ValueError("CUDA peer topology must contain exact ordered zero-to-one edges")
        first, second = self.edges
        if (
            first.source_uuid,
            first.destination_uuid,
        ) != (
            second.destination_uuid,
            second.source_uuid,
        ):
            raise ValueError("CUDA peer topology reverse edges use different GPU UUIDs")
        return self


class SmokeHostEvidence(_SmokeReceiptModel):
    """Exact host identity for the tested Modal hardware allocation."""

    provider: Literal["Modal"]
    cpu_model: StrictStr = Field(min_length=1)
    host_logical_cpu_count: StrictInt = Field(gt=0)
    host_logical_cpu_count_scope: Literal["host_online_os_cpu_count"]
    logical_cpu_count: StrictInt = Field(gt=0)
    logical_cpu_count_scope: Literal[
        "container_cgroup_cpu_quota",
        "container_effective_sched_getaffinity",
    ]
    requested_cpu_cores: Literal[16]
    requested_cpu_scope: Literal["modal_physical_cores_hard_request_and_limit"]
    cgroup_membership_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    cgroup_mountinfo_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    cgroup_visibility_scope: Literal["process_mount_namespace_visible_hierarchy"]
    cgroup_process_pid: StrictInt = Field(gt=0)
    cgroup_cpu_leaf_path_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    cgroup_cpu_leaf_pid_verified: Literal[True]
    cgroup_cpu_leaf_cgroup_procs_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    cgroup_cpu_quota_millicores: StrictInt = Field(gt=0)
    cgroup_cpu_quota_source: Literal[
        "cgroup_v2_visible_hierarchy_cpu.max",
        "cgroup_v1_visible_hierarchy_cpu.cfs_quota_us",
    ]
    cgroup_cpu_limit_path_sha256s: tuple[StrictStr, ...]
    cgroup_cpu_limit_values_millicores: tuple[StrictInt | None, ...]
    cpu_affinity_ids: tuple[StrictInt, ...]
    cpu_affinity_scope: Literal["container_effective_sched_getaffinity"]
    host_ram_bytes: StrictInt = Field(gt=0)
    host_ram_scope: Literal["host_physical_proc_meminfo_memtotal"]
    ram_bytes: StrictInt = Field(gt=0)
    ram_scope: Literal[
        "container_cgroup_memory_limit",
        "host_physical_no_lower_cgroup_limit",
    ]
    requested_ram_bytes: Literal[68719476736]
    requested_ram_scope: Literal["modal_bytes_hard_request_and_limit"]
    cgroup_memory_leaf_path_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    cgroup_memory_leaf_pid_verified: Literal[True]
    cgroup_memory_leaf_cgroup_procs_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    cgroup_memory_limit_bytes: StrictInt = Field(gt=0)
    cgroup_memory_limit_source: Literal[
        "cgroup_v2_visible_hierarchy_memory.max",
        "cgroup_v1_visible_hierarchy_memory.limit_in_bytes",
    ]
    cgroup_memory_limit_path_sha256s: tuple[StrictStr, ...]
    cgroup_memory_limit_values_bytes: tuple[StrictInt | None, ...]
    nvidia_smi_topo_m_sha256: StrictStr | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    topology_schema_version: Literal[
        "inkling-smoke-hardware-topology-v2",
        "inkling-smoke-hardware-topology-v3",
        "inkling-smoke-hardware-topology-v4",
    ]
    topology_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("cpu_model")
    @classmethod
    def cpu_model_is_exact(cls, value: str) -> str:
        return _require_inventory_text(value, label="CPU model")

    @field_validator(
        "cgroup_cpu_limit_path_sha256s",
        "cgroup_memory_limit_path_sha256s",
    )
    @classmethod
    def cgroup_path_hashes_are_sha256(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in values):
            raise ValueError("cgroup path hashes must be non-empty SHA-256 inventories")
        if len(values) != len(set(values)):
            raise ValueError("cgroup path hash inventories must not contain duplicates")
        return values

    @model_validator(mode="after")
    def host_capacity_is_consistent(self) -> SmokeHostEvidence:
        if self.topology_schema_version in {
            "inkling-smoke-hardware-topology-v2",
            "inkling-smoke-hardware-topology-v3",
        }:
            if self.nvidia_smi_topo_m_sha256 is None:
                raise ValueError("historical host topology lacks its nvidia-smi digest")
        elif (
            self.nvidia_smi_topo_m_sha256 is not None
            or "nvidia_smi_topo_m_sha256" in self.model_fields_set
        ):
            raise ValueError("current host topology must not contain legacy nvidia-smi evidence")
        if (
            not self.cpu_affinity_ids
            or self.cpu_affinity_ids != tuple(sorted(self.cpu_affinity_ids))
            or len(self.cpu_affinity_ids) != len(set(self.cpu_affinity_ids))
            or any(identifier < 0 for identifier in self.cpu_affinity_ids)
        ):
            raise ValueError("CPU affinity IDs must be non-empty, sorted, and unique")
        if len(self.cpu_affinity_ids) > self.host_logical_cpu_count:
            raise ValueError("CPU affinity count exceeds the host logical CPU count")
        affinity_count = len(self.cpu_affinity_ids)
        expected_logical_cpu_count = affinity_count
        expected_logical_cpu_scope = "container_effective_sched_getaffinity"
        quota_whole_cores = self.cgroup_cpu_quota_millicores // 1000
        if quota_whole_cores <= affinity_count:
            expected_logical_cpu_count = self.cgroup_cpu_quota_millicores // 1000
            expected_logical_cpu_scope = "container_cgroup_cpu_quota"
        if (
            self.logical_cpu_count != expected_logical_cpu_count
            or self.logical_cpu_count_scope != expected_logical_cpu_scope
        ):
            raise ValueError("effective logical CPU count differs from its measured scope")
        if not self.cgroup_cpu_limit_values_millicores:
            raise ValueError("cgroup CPU hierarchy inventory is empty")
        expected_cpu_path_count = len(self.cgroup_cpu_limit_values_millicores)
        if self.cgroup_cpu_quota_source == "cgroup_v1_visible_hierarchy_cpu.cfs_quota_us":
            expected_cpu_path_count *= 2
        if len(self.cgroup_cpu_limit_path_sha256s) != expected_cpu_path_count:
            raise ValueError("cgroup CPU limit path hashes differ from their source")
        finite_cpu_limits = [
            value for value in self.cgroup_cpu_limit_values_millicores if value is not None
        ]
        if not finite_cpu_limits or self.cgroup_cpu_quota_millicores != min(finite_cpu_limits):
            raise ValueError("effective cgroup CPU quota differs from its visible hierarchy")
        if self.cgroup_cpu_quota_millicores != self.requested_cpu_cores * 1000:
            raise ValueError("cgroup CPU hard quota differs from the requested limit")
        if self.logical_cpu_count < self.requested_cpu_cores:
            raise ValueError("effective CPU capacity is below the requested allocation")
        if self.host_ram_bytes % 1024 != 0:
            raise ValueError("host RAM bytes must come from the Linux KiB inventory")
        expected_ram_bytes = self.host_ram_bytes
        expected_ram_scope = "host_physical_no_lower_cgroup_limit"
        if self.cgroup_memory_limit_bytes < self.host_ram_bytes:
            expected_ram_bytes = self.cgroup_memory_limit_bytes
            expected_ram_scope = "container_cgroup_memory_limit"
        if self.ram_bytes != expected_ram_bytes or self.ram_scope != expected_ram_scope:
            raise ValueError("effective RAM differs from its measured scope")
        if not self.cgroup_memory_limit_values_bytes:
            raise ValueError("cgroup memory hierarchy inventory is empty")
        if len(self.cgroup_memory_limit_path_sha256s) != len(self.cgroup_memory_limit_values_bytes):
            raise ValueError("cgroup memory limit path hashes differ from their source")
        finite_memory_limits = [
            value for value in self.cgroup_memory_limit_values_bytes if value is not None
        ]
        if not finite_memory_limits or self.cgroup_memory_limit_bytes != min(finite_memory_limits):
            raise ValueError("effective cgroup memory limit differs from its visible hierarchy")
        if self.cgroup_memory_limit_bytes != self.requested_ram_bytes:
            raise ValueError("cgroup memory hard limit differs from the requested limit")
        if self.ram_bytes < self.requested_ram_bytes:
            raise ValueError("effective RAM is below the requested allocation")
        return self


class SmokeGpuEvidence(_SmokeReceiptModel):
    """One exact GPU identity in the tested hardware matrix cell."""

    identity_protocol: Literal["cuda-driver-uuid+nvidia-smi-uuid-v1"]
    cuda_driver_api_version: StrictInt = Field(gt=0)
    cuda_ordinal: StrictInt = Field(ge=0)
    uuid: StrictStr = Field(pattern=r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$")
    cuda_driver_name: StrictStr = Field(min_length=1)
    nvidia_smi_name: StrictStr = Field(min_length=1)
    cuda_compute_capability: Literal["10.3"]
    nvidia_smi_compute_capability: Literal["10.3"]
    cuda_total_memory_bytes: StrictInt = Field(gt=0)
    nvidia_smi_memory_total_mib: StrictInt = Field(gt=0)
    driver_version: StrictStr = Field(pattern=r"^[0-9]+(?:\.[0-9]+)+$")
    pci_bus_id: StrictStr | None = Field(
        default=None,
        pattern=r"^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$",
    )
    pci_bus_id_status: Literal["available", "unavailable"]
    pci_bus_id_source: Literal["cuda_driver_api"]
    pci_bus_id_error_code: StrictInt | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def gpu_sources_are_consistent(self) -> SmokeGpuEvidence:
        if "B300" not in self.cuda_driver_name.upper():
            raise ValueError("CUDA driver smoke GPU name is not B300")
        if "B300" not in self.nvidia_smi_name.upper():
            raise ValueError("nvidia-smi smoke GPU name is not B300")
        if self.pci_bus_id_status == "available":
            if self.pci_bus_id is None or self.pci_bus_id_error_code is not None:
                raise ValueError("available CUDA PCI evidence is incomplete")
        elif self.pci_bus_id is not None or self.pci_bus_id_error_code is None:
            raise ValueError("unavailable CUDA PCI evidence lacks its error code")
        return self


def smoke_hardware_topology_sha256(
    host: SmokeHostEvidence | Mapping[str, object],
    gpus: Sequence[SmokeGpuEvidence | Mapping[str, object]],
    gpu_topology: SmokeGpuTopologyEvidence | Mapping[str, object] | None = None,
) -> str:
    """Hash the complete host and ordered accelerator identity for one cell."""

    host_value = (
        host.model_dump(mode="json", exclude_none=True)
        if isinstance(host, SmokeHostEvidence)
        else dict(host)
    )
    gpu_values = [
        gpu.model_dump(mode="json") if isinstance(gpu, SmokeGpuEvidence) else dict(gpu)
        for gpu in gpus
    ]
    payload = {
        "schema_version": host_value.get("topology_schema_version"),
        "provider": host_value.get("provider"),
        "cpu_model": host_value.get("cpu_model"),
        "host_logical_cpu_count": host_value.get("host_logical_cpu_count"),
        "host_logical_cpu_count_scope": host_value.get("host_logical_cpu_count_scope"),
        "logical_cpu_count": host_value.get("logical_cpu_count"),
        "logical_cpu_count_scope": host_value.get("logical_cpu_count_scope"),
        "requested_cpu_cores": host_value.get("requested_cpu_cores"),
        "requested_cpu_scope": host_value.get("requested_cpu_scope"),
        "cgroup_membership_sha256": host_value.get("cgroup_membership_sha256"),
        "cgroup_mountinfo_sha256": host_value.get("cgroup_mountinfo_sha256"),
        "cgroup_visibility_scope": host_value.get("cgroup_visibility_scope"),
        "cgroup_process_pid": host_value.get("cgroup_process_pid"),
        "cgroup_cpu_leaf_path_sha256": host_value.get("cgroup_cpu_leaf_path_sha256"),
        "cgroup_cpu_leaf_pid_verified": host_value.get("cgroup_cpu_leaf_pid_verified"),
        "cgroup_cpu_leaf_cgroup_procs_sha256": host_value.get(
            "cgroup_cpu_leaf_cgroup_procs_sha256"
        ),
        "cgroup_cpu_quota_millicores": host_value.get("cgroup_cpu_quota_millicores"),
        "cgroup_cpu_quota_source": host_value.get("cgroup_cpu_quota_source"),
        "cgroup_cpu_limit_path_sha256s": host_value.get("cgroup_cpu_limit_path_sha256s"),
        "cgroup_cpu_limit_values_millicores": host_value.get("cgroup_cpu_limit_values_millicores"),
        "cpu_affinity_ids": host_value.get("cpu_affinity_ids"),
        "cpu_affinity_scope": host_value.get("cpu_affinity_scope"),
        "host_ram_bytes": host_value.get("host_ram_bytes"),
        "host_ram_scope": host_value.get("host_ram_scope"),
        "ram_bytes": host_value.get("ram_bytes"),
        "ram_scope": host_value.get("ram_scope"),
        "requested_ram_bytes": host_value.get("requested_ram_bytes"),
        "requested_ram_scope": host_value.get("requested_ram_scope"),
        "cgroup_memory_leaf_path_sha256": host_value.get("cgroup_memory_leaf_path_sha256"),
        "cgroup_memory_leaf_pid_verified": host_value.get("cgroup_memory_leaf_pid_verified"),
        "cgroup_memory_leaf_cgroup_procs_sha256": host_value.get(
            "cgroup_memory_leaf_cgroup_procs_sha256"
        ),
        "cgroup_memory_limit_bytes": host_value.get("cgroup_memory_limit_bytes"),
        "cgroup_memory_limit_source": host_value.get("cgroup_memory_limit_source"),
        "cgroup_memory_limit_path_sha256s": host_value.get("cgroup_memory_limit_path_sha256s"),
        "cgroup_memory_limit_values_bytes": host_value.get("cgroup_memory_limit_values_bytes"),
        "nvidia_smi_topo_m_sha256": host_value.get("nvidia_smi_topo_m_sha256"),
        "gpus": sorted(gpu_values, key=lambda value: str(value.get("uuid"))),
    }
    schema_version = host_value.get("topology_schema_version")
    if schema_version == "inkling-smoke-hardware-topology-v2":
        if gpu_topology is not None:
            raise ValueError("historical smoke topology must not contain CUDA peer evidence")
        domain = _SMOKE_HARDWARE_TOPOLOGY_HASH_DOMAIN_V2
    elif schema_version == "inkling-smoke-hardware-topology-v3":
        if gpu_topology is not None:
            raise ValueError("historical smoke topology must not contain CUDA peer evidence")
        domain = _SMOKE_HARDWARE_TOPOLOGY_HASH_DOMAIN_V3
    elif schema_version == "inkling-smoke-hardware-topology-v4":
        if "nvidia_smi_topo_m_sha256" in host_value:
            raise ValueError("current smoke topology contains legacy nvidia-smi evidence")
        if gpu_topology is None:
            raise ValueError("current smoke topology lacks CUDA peer evidence")
        topology_value = (
            gpu_topology.model_dump(mode="json")
            if isinstance(gpu_topology, SmokeGpuTopologyEvidence)
            else SmokeGpuTopologyEvidence.model_validate(dict(gpu_topology)).model_dump(mode="json")
        )
        payload.pop("nvidia_smi_topo_m_sha256")
        payload["gpu_topology"] = topology_value
        domain = SMOKE_HARDWARE_TOPOLOGY_HASH_DOMAIN
    else:
        raise ValueError("unsupported smoke hardware topology schema version")
    return hashlib.sha256(domain + _canonical_json(payload).encode("utf-8")).hexdigest()


class SmokeServerProperties(_SmokeReceiptModel):
    """Redacted model metadata returned by the pinned server."""

    props_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    models_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    build_info: StrictStr = Field(min_length=1)
    modalities: dict[Literal["text", "vision", "audio"], RequiredTrue]
    vocab_size: StrictInt = Field(gt=0)
    media_marker_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_properties(self) -> SmokeServerProperties:
        if self.modalities != {"text": True, "vision": True, "audio": True}:
            raise ValueError("server properties must report text, vision, and audio")
        if PINNED_LLAMA_CPP_COMMIT[:7] not in self.build_info:
            raise ValueError("server build information lacks the pinned commit")
        return self


class SmokeTextShardEvidence(_SmokeReceiptModel):
    """Metadata-open proof for the complete split text model."""

    expected: StrictInt = Field(ge=49, le=49)
    opened: StrictInt = Field(ge=49, le=49)
    contexts: StrictInt = Field(ge=49, le=49)
    tensors: StrictInt = Field(gt=0)


class SmokeTextLoadEvidence(_SmokeReceiptModel):
    """Tensor-data accounting for the complete split text model."""

    opened: StrictInt = Field(ge=49, le=49)
    accounted: StrictInt = Field(ge=49, le=49)
    tensors: StrictInt = Field(gt=0)
    bytes: StrictInt = Field(gt=0)
    size_done: StrictInt = Field(gt=0)
    size_data: StrictInt = Field(gt=0)
    mmap: RequiredTrue

    @model_validator(mode="after")
    def byte_accounting_is_complete(self) -> SmokeTextLoadEvidence:
        if not self.bytes == self.size_done == self.size_data:
            raise ValueError("text tensor byte accounting is incomplete")
        return self


class SmokeProjectorTensorEvidence(_SmokeReceiptModel):
    """Loaded tensor count and bytes for one Inkling projector modality."""

    modality: Literal["vision", "audio"]
    projector: Literal["inkling"]
    tensors: StrictInt = Field(gt=0)
    bytes: StrictInt = Field(gt=0)


class SmokeProjectorReadyEvidence(_SmokeReceiptModel):
    """Proof that both Inkling projector contexts reached ready state."""

    opened: RequiredTrue
    vision: RequiredTrue
    audio: RequiredTrue
    vision_type: Literal["inkling"]
    audio_type: Literal["inkling"]
    n_embd: StrictInt = Field(gt=0)


class SmokeArtifactLoadEvidence(_SmokeReceiptModel):
    """Pinned loader proof for all shards and the multimodal projector."""

    schema_version: Literal["inkling-artifact-load-v1"]
    first_shard_path: Literal["/subject/q3_k_m/inkling-Q3_K_M-00001-of-00049.gguf"]
    additional_shards_loaded: StrictInt = Field(ge=48, le=48)
    total_shards_loaded: StrictInt = Field(ge=49, le=49)
    projector_path: Literal["/subject/mmproj/mmproj-BF16.gguf"]
    text_shards: SmokeTextShardEvidence
    text_load: SmokeTextLoadEvidence
    projector_tensors: tuple[
        SmokeProjectorTensorEvidence,
        SmokeProjectorTensorEvidence,
    ]
    projector_ready: SmokeProjectorReadyEvidence
    all_expected_artifacts_loaded: RequiredTrue

    @model_validator(mode="after")
    def all_tensor_sets_are_loaded(self) -> SmokeArtifactLoadEvidence:
        if self.text_shards.tensors != self.text_load.tensors:
            raise ValueError("text metadata and loaded tensor counts differ")
        if tuple(row.modality for row in self.projector_tensors) != (
            "vision",
            "audio",
        ):
            raise ValueError("projector tensor evidence must be ordered vision then audio")
        return self


class SmokeRawLogitRowEvidence(_SmokeReceiptModel):
    """One pre-softmax vocabulary vector scanned by pinned instrumentation."""

    task_id: StrictInt = Field(ge=0)
    slot_id: StrictInt = Field(ge=0)
    completion_index: StrictInt = Field(gt=0)
    batch_index: StrictInt = Field(ge=0)
    count: StrictInt = Field(gt=0)
    finite: StrictInt = Field(gt=0)
    nan: StrictInt = Field(ge=0, le=0)
    pos_inf: StrictInt = Field(ge=0, le=0)
    neg_inf: StrictInt = Field(ge=0, le=0)


class SmokeRawLogitAudit(_SmokeReceiptModel):
    """Historical full-vocabulary finiteness audit for generated vectors."""

    schema_version: Literal["inkling-raw-logit-audit-v1"]
    expected_generated_token_vectors: StrictInt = Field(gt=0)
    observed_generated_token_vectors: StrictInt = Field(gt=0)
    vocab_size: StrictInt = Field(gt=0)
    rows: tuple[SmokeRawLogitRowEvidence, ...]
    all_rows_complete: RequiredTrue
    all_values_finite: RequiredTrue

    @model_validator(mode="after")
    def audit_is_complete(self) -> SmokeRawLogitAudit:
        if not (
            self.expected_generated_token_vectors
            == self.observed_generated_token_vectors
            == len(self.rows)
        ):
            raise ValueError("raw-logit vector cardinality is incomplete")
        for row in self.rows:
            if row.count != self.vocab_size or row.finite != self.vocab_size:
                raise ValueError("raw-logit row does not cover the full vocabulary")
        identities = tuple(
            (row.task_id, row.slot_id, row.completion_index, row.batch_index) for row in self.rows
        )
        if len(identities) != len(set(identities)):
            raise ValueError("raw-logit audit contains duplicate vector identities")
        return self


class SmokeRawLogitRowEvidenceV2(_SmokeReceiptModel):
    """One logit vector split at the pinned unpadded vocabulary boundary."""

    task_id: StrictInt = Field(ge=0)
    slot_id: StrictInt = Field(ge=0)
    completion_index: StrictInt = Field(gt=0)
    batch_index: StrictInt = Field(ge=0)
    count: StrictInt = Field(gt=0)
    unpadded_count: StrictInt = Field(gt=0)
    padded_count: StrictInt = Field(gt=0)
    unpadded_finite: StrictInt = Field(gt=0)
    unpadded_nan: StrictInt = Field(ge=0, le=0)
    unpadded_pos_inf: StrictInt = Field(ge=0, le=0)
    unpadded_neg_inf: StrictInt = Field(ge=0, le=0)
    padded_finite: StrictInt = Field(ge=0, le=0)
    padded_nan: StrictInt = Field(ge=0, le=0)
    padded_pos_inf: StrictInt = Field(ge=0, le=0)
    padded_neg_inf: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def regions_have_exact_value_classes(self) -> SmokeRawLogitRowEvidenceV2:
        if self.unpadded_count + self.padded_count != self.count:
            raise ValueError("raw-logit row regions do not cover the vocabulary")
        if self.unpadded_finite != self.unpadded_count:
            raise ValueError("raw-logit unpadded region is not exactly finite")
        if self.padded_neg_inf != self.padded_count:
            raise ValueError("raw-logit padded suffix is not exactly negative infinity")
        return self


class SmokeRawLogitAuditV2(_SmokeReceiptModel):
    """Complete audit of finite active logits and the masked padded suffix."""

    schema_version: Literal["inkling-raw-logit-audit-v2"]
    expected_generated_token_vectors: StrictInt = Field(gt=0)
    observed_generated_token_vectors: StrictInt = Field(gt=0)
    vocab_size: StrictInt = Field(gt=0)
    unpadded_vocab_size: StrictInt = Field(gt=0)
    padded_vocab_size: StrictInt = Field(gt=0)
    rows: tuple[SmokeRawLogitRowEvidenceV2, ...]
    all_rows_complete: RequiredTrue
    all_unpadded_values_finite: RequiredTrue
    all_padded_values_negative_infinity: RequiredTrue

    @model_validator(mode="after")
    def audit_is_complete(self) -> SmokeRawLogitAuditV2:
        if self.unpadded_vocab_size >= self.vocab_size:
            raise ValueError("raw-logit audit has no padded vocabulary suffix")
        if self.padded_vocab_size != self.vocab_size - self.unpadded_vocab_size:
            raise ValueError("raw-logit padded vocabulary cardinality is inconsistent")
        if not (
            self.expected_generated_token_vectors
            == self.observed_generated_token_vectors
            == len(self.rows)
        ):
            raise ValueError("raw-logit vector cardinality is incomplete")
        for row in self.rows:
            if (
                row.count != self.vocab_size
                or row.unpadded_count != self.unpadded_vocab_size
                or row.padded_count != self.padded_vocab_size
            ):
                raise ValueError("raw-logit row vocabulary boundary differs from its audit")
        identities = tuple(
            (row.task_id, row.slot_id, row.completion_index, row.batch_index) for row in self.rows
        )
        if len(identities) != len(set(identities)):
            raise ValueError("raw-logit audit contains duplicate vector identities")
        return self


SmokeRawLogitAuditEvidence: TypeAlias = SmokeRawLogitAudit | SmokeRawLogitAuditV2


class SmokeBackendGraphEvidence(_SmokeReceiptModel):
    """One graph's post-assignment non-view compute placement."""

    graph_uid: StrictInt = Field(gt=0)
    phase: Literal["post_assignment_pre_split"]
    scope: Literal["non_view_compute"]
    compute: StrictInt = Field(gt=0)
    gpu: StrictInt = Field(gt=0)
    cpu: StrictInt = Field(ge=0, le=0)
    accel: StrictInt = Field(ge=0)
    other: StrictInt = Field(ge=0, le=0)
    unassigned: StrictInt = Field(ge=0, le=0)

    @model_validator(mode="after")
    def categories_cover_compute(self) -> SmokeBackendGraphEvidence:
        if self.compute != (self.gpu + self.cpu + self.accel + self.other + self.unassigned):
            raise ValueError("backend graph categories do not cover its compute nodes")
        return self


class SmokeBackendIdentityEvidence(_SmokeReceiptModel):
    """One backend device's compute-node count within one graph."""

    graph_uid: StrictInt = Field(gt=0)
    backend_index: StrictInt = Field(ge=0)
    backend_name: StrictStr = Field(pattern=r"^\S+$")
    device_name: StrictStr = Field(pattern=r"^\S+$")
    device_type: Literal["cpu", "gpu", "igpu", "accel", "meta", "unassigned"]
    compute: StrictInt = Field(gt=0)


class SmokeBackendAudit(_SmokeReceiptModel):
    """Graph-operation placement proof with no CPU or unassigned fallback."""

    schema_version: Literal["inkling-backend-audit-v1"]
    graphs: tuple[SmokeBackendGraphEvidence, ...]
    identities: tuple[SmokeBackendIdentityEvidence, ...]
    observed_graphs: StrictInt = Field(gt=0)
    compute_operations: StrictInt = Field(gt=0)
    gpu_operations: StrictInt = Field(gt=0)
    accelerator_operations: StrictInt = Field(ge=0)
    cpu_operations: StrictInt = Field(ge=0, le=0)
    other_operations: StrictInt = Field(ge=0, le=0)
    unassigned_operations: StrictInt = Field(ge=0, le=0)
    all_compute_operations_accelerated: RequiredTrue
    no_cpu_model_graph_fallback: RequiredTrue

    @model_validator(mode="after")
    def all_operations_are_accelerated(self) -> SmokeBackendAudit:
        graph_uids = tuple(graph.graph_uid for graph in self.graphs)
        if len(graph_uids) != self.observed_graphs or len(graph_uids) != len(set(graph_uids)):
            raise ValueError("backend graph identities are missing or duplicated")
        identity_keys = tuple(
            (identity.graph_uid, identity.backend_index) for identity in self.identities
        )
        if len(identity_keys) != len(set(identity_keys)):
            raise ValueError("backend device identities are duplicated")
        if {identity.graph_uid for identity in self.identities} != set(graph_uids):
            raise ValueError("backend identities do not cover the exact graph set")

        graph_by_uid = {graph.graph_uid: graph for graph in self.graphs}
        for graph_uid, graph in graph_by_uid.items():
            rows = tuple(
                identity for identity in self.identities if identity.graph_uid == graph_uid
            )
            category_counts = {
                "gpu": sum(row.compute for row in rows if row.device_type in {"gpu", "igpu"}),
                "cpu": sum(row.compute for row in rows if row.device_type == "cpu"),
                "accel": sum(row.compute for row in rows if row.device_type == "accel"),
                "other": sum(row.compute for row in rows if row.device_type == "meta"),
                "unassigned": sum(row.compute for row in rows if row.device_type == "unassigned"),
            }
            if sum(row.compute for row in rows) != graph.compute:
                raise ValueError("backend identities do not cover a graph's compute nodes")
            if any(category_counts[name] != getattr(graph, name) for name in category_counts):
                raise ValueError("backend identity categories differ from graph evidence")

        aggregates = {
            "compute_operations": sum(graph.compute for graph in self.graphs),
            "gpu_operations": sum(graph.gpu for graph in self.graphs),
            "accelerator_operations": sum(graph.accel for graph in self.graphs),
            "cpu_operations": sum(graph.cpu for graph in self.graphs),
            "other_operations": sum(graph.other for graph in self.graphs),
            "unassigned_operations": sum(graph.unassigned for graph in self.graphs),
        }
        if any(getattr(self, name) != value for name, value in aggregates.items()):
            raise ValueError("backend aggregates differ from per-graph evidence")
        if self.gpu_operations + self.accelerator_operations != self.compute_operations:
            raise ValueError("backend audit does not assign every operation to an accelerator")
        return self


class SmokeServerCleanup(_SmokeReceiptModel):
    """Bounded server shutdown result."""

    method: Literal["SIGTERM"]
    return_code: StrictInt
    clean: RequiredTrue


class SmokeLoaderOffloadEvidence(_SmokeReceiptModel):
    """Strict proof that every offloadable model layer reached CUDA."""

    cuda_device_count: StrictInt = Field(gt=0)
    offloaded_layers: StrictInt = Field(gt=0)
    offloadable_layers: StrictInt = Field(gt=0)
    output_layer_offloaded: RequiredTrue
    all_offloadable_layers_on_gpu: RequiredTrue
    no_gpu_warning_observed: RequiredTrue

    @model_validator(mode="after")
    def exact_layer_offload(self) -> SmokeLoaderOffloadEvidence:
        if self.offloaded_layers != self.offloadable_layers:
            raise ValueError("loader did not offload every offloadable layer")
        return self


class SmokeServerEvidence(_SmokeReceiptModel):
    """Server invocation, load, graph, logit, and cleanup evidence."""

    command: tuple[StrictStr, ...]
    audit_environment: dict[StrictStr, StrictStr]
    network_scope: Literal["loopback_only_with_modal_external_network_blocked"]
    post_rehash_load_to_health_seconds: float = Field(gt=0, le=3_600)
    loader_offload: SmokeLoaderOffloadEvidence
    artifact_load: SmokeArtifactLoadEvidence
    raw_logit_audit: SmokeRawLogitAuditEvidence
    backend_audit: SmokeBackendAudit
    properties: SmokeServerProperties
    server_log_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    cleanup: SmokeServerCleanup

    @field_validator("audit_environment")
    @classmethod
    def exact_audit_environment(cls, value: dict[str, str]) -> dict[str, str]:
        if value != _SMOKE_SERVER_AUDIT_ENVIRONMENT:
            raise ValueError("server instrumentation environment differs from its contract")
        return value

    @field_validator("post_rehash_load_to_health_seconds", mode="before")
    @classmethod
    def load_time_is_finite(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("server load time must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError("server load time must be finite")
        return value


class SmokeTimingEvidence(_SmokeReceiptModel):
    """One measured llama-server completion timing record."""

    prompt_n: StrictInt = Field(gt=0)
    predicted_n: StrictInt = Field(gt=0)
    cache_n: StrictInt = Field(ge=0)
    prompt_ms: float = Field(gt=0)
    prompt_per_token_ms: float = Field(gt=0)
    prompt_per_second: float = Field(gt=0)
    predicted_ms: float = Field(gt=0)
    predicted_per_token_ms: float = Field(gt=0)
    predicted_per_second: float = Field(gt=0)

    @field_validator(
        "prompt_ms",
        "prompt_per_token_ms",
        "prompt_per_second",
        "predicted_ms",
        "predicted_per_token_ms",
        "predicted_per_second",
        mode="before",
    )
    @classmethod
    def timing_is_finite(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("completion timing must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError("completion timing must be finite")
        return value


class SmokeProbeTrialEvidence(_SmokeReceiptModel):
    """One redacted deterministic completion trial."""

    trial: StrictInt = Field(ge=1, le=2)
    token_ids: tuple[StrictInt, ...]
    tokens_predicted: StrictInt = Field(gt=0, le=8)
    minimum_sampled_token_logprob: float
    maximum_sampled_token_logprob: float
    all_returned_logprobs_finite: RequiredTrue
    response_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    timings: SmokeTimingEvidence

    @field_validator(
        "minimum_sampled_token_logprob",
        "maximum_sampled_token_logprob",
        mode="before",
    )
    @classmethod
    def logprob_is_finite(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("sampled-token log probability must be numeric")
        parsed = float(value)
        if not math.isfinite(parsed) or parsed > 0:
            raise ValueError("sampled-token log probability must be finite and nonpositive")
        return value

    @model_validator(mode="after")
    def tokens_and_timings_match(self) -> SmokeProbeTrialEvidence:
        if len(self.token_ids) != self.tokens_predicted:
            raise ValueError("trial token count differs from returned token IDs")
        if self.timings.predicted_n != self.tokens_predicted:
            raise ValueError("trial timing token count differs from returned token IDs")
        if self.minimum_sampled_token_logprob > self.maximum_sampled_token_logprob:
            raise ValueError("sampled-token log-probability range is inverted")
        return self


class SmokeProbeEvidence(_SmokeReceiptModel):
    """Two deterministic trials for one exact modality probe."""

    probe_id: StrictStr = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    modality: Literal["text", "image", "audio"]
    prompt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_recorded: RequiredFalse
    output_text_recorded: RequiredFalse
    fixture: Literal[
        "none",
        "synthetic_rgb8_png_16x16_checkerboard_v1",
        "synthetic_pcm_s16le_wav_16000hz_mono_silence_250ms_v1",
    ]
    fixture_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    fixture_size_bytes: StrictInt | None = Field(default=None, gt=0)
    seed: StrictInt = Field(ge=42, le=42)
    temperature: float
    repeatable_greedy_token_ids: RequiredTrue
    trials: tuple[SmokeProbeTrialEvidence, ...]

    @field_validator("temperature", mode="before")
    @classmethod
    def exact_greedy_temperature(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("probe temperature must be numeric")
        if float(value) != 0.0:
            raise ValueError("probe temperature must use llama.cpp greedy mode")
        return value

    @model_validator(mode="after")
    def exact_trials_and_repeatability(self) -> SmokeProbeEvidence:
        if tuple(trial.trial for trial in self.trials) != (1, 2):
            raise ValueError("probe must contain exactly trials one and two")
        if self.trials[0].token_ids != self.trials[1].token_ids:
            raise ValueError("probe trials are not token-identical")
        expected_fixture_identity = {
            "none": (None, None),
            "synthetic_rgb8_png_16x16_checkerboard_v1": (
                "95b4e645a67edfb972c4ca1f2a0b8ed97e60988adfaa020d015d6a334576c2d7",
                86,
            ),
            "synthetic_pcm_s16le_wav_16000hz_mono_silence_250ms_v1": (
                "59460d5690616336b990fc7b1629428e3bd825e422da84469d2c8c8ecfaff43b",
                8_044,
            ),
        }[self.fixture]
        if (self.fixture_sha256, self.fixture_size_bytes) != expected_fixture_identity:
            raise ValueError("probe fixture identity differs from its checked constructor")
        return self


class SmokeResourceEvidence(_SmokeReceiptModel):
    """Peak host and per-GPU resource observations."""

    sampling_interval_seconds: float
    sample_count: StrictInt = Field(gt=0)
    server_peak_host_rss_mib: StrictInt = Field(gt=0)
    gpu_peak_memory_used_mib: tuple[StrictInt, StrictInt]
    gpu_peak_utilization_percent: tuple[StrictInt, StrictInt]

    @field_validator("sampling_interval_seconds", mode="before")
    @classmethod
    def exact_sampling_interval(cls, value: object) -> object:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("resource sampling interval must be numeric")
        if float(value) != 1.0:
            raise ValueError("resource sampling interval must be exactly one second")
        return value

    @model_validator(mode="after")
    def valid_gpu_peaks(self) -> SmokeResourceEvidence:
        if any(value <= 0 for value in self.gpu_peak_memory_used_mib):
            raise ValueError("each GPU must report positive peak memory use")
        if any(not 0 <= value <= 100 for value in self.gpu_peak_utilization_percent):
            raise ValueError("GPU utilization is outside zero to one hundred percent")
        return self


class SmokeReceiptClaimLimits(_SmokeReceiptModel):
    """Exact claim boundary for a load-and-inference smoke result."""

    purpose: Literal["load_and_inference_smoke_only"]
    mtp_included: RequiredFalse
    mtp_supported: RequiredFalse
    quality_measured: RequiredFalse
    benchmark_measured: RequiredFalse
    performance_claim_allowed: RequiredFalse
    quality_retention_claim_allowed: RequiredFalse
    compatibility_scope: Literal["single_exact_matrix_cell"]


class SmokeReceiptEvidencePolicy(_SmokeReceiptModel):
    """Exact evidence and privacy policy required for a passing receipt."""

    record_prompt_text: RequiredFalse
    record_output_text: RequiredFalse
    record_token_ids: RequiredTrue
    record_logprob_summary: RequiredTrue
    record_command_arguments: RequiredTrue
    record_artifact_hashes: RequiredTrue
    record_runtime_commit: RequiredTrue
    record_hardware: RequiredTrue
    record_timings: RequiredTrue
    record_peak_memory: RequiredTrue


class SmokeTerminalReceipt(_SmokeReceiptModel):
    """Strict success evidence for the exact Inkling Q3 inference smoke stage."""

    schema_version: Literal[
        "inkling-smoke-terminal-v3",
        "inkling-smoke-terminal-v4",
        "inkling-smoke-terminal-v5",
    ]
    status: Literal["passed"]
    stage: Literal["smoke_test"]
    run_id: StrictStr = Field(pattern=r"^[a-z0-9][a-z0-9._-]{0,95}$")
    subject: SmokeReceiptSubject
    smoke_config_hash: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    control_plane_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    control_plane_file_count: StrictInt = Field(gt=0)
    launch_intent_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    invocation: SmokeInvocationEvidence
    artifact_rehash: SmokeArtifactRehashEvidence
    runtime: SmokeRuntimeEvidence
    host: SmokeHostEvidence
    hardware: tuple[SmokeGpuEvidence, SmokeGpuEvidence]
    gpu_topology: SmokeGpuTopologyEvidence | None = None
    server: SmokeServerEvidence
    probes: tuple[SmokeProbeEvidence, SmokeProbeEvidence, SmokeProbeEvidence]
    resources: SmokeResourceEvidence
    claims: SmokeReceiptClaimLimits
    evidence_policy: SmokeReceiptEvidencePolicy
    prompt_text_recorded: RequiredFalse
    output_text_recorded: RequiredFalse
    completed_at_utc: StrictStr
    receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("completed_at_utc")
    @classmethod
    def completion_time_is_utc(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise ValueError("completion time must be ISO 8601") from error
        if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
            raise ValueError("completion time must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def evidence_is_internally_complete(self) -> SmokeTerminalReceipt:
        identities = tuple((probe.probe_id, probe.modality) for probe in self.probes)
        if identities != (
            ("text_greedy_v1", "text"),
            ("image_greedy_v1", "image"),
            ("audio_greedy_v1", "audio"),
        ):
            raise ValueError("receipt must contain exact ordered text, image, and audio probes")
        vocabulary = self.server.properties.vocab_size
        if self.server.raw_logit_audit.vocab_size != vocabulary:
            raise ValueError("raw-logit and server vocabulary sizes differ")
        predicted = sum(trial.tokens_predicted for probe in self.probes for trial in probe.trials)
        if self.server.raw_logit_audit.expected_generated_token_vectors != predicted:
            raise ValueError("raw-logit audit does not cover every generated token")
        if self.schema_version == "inkling-smoke-terminal-v5":
            if not isinstance(self.server.raw_logit_audit, SmokeRawLogitAuditV2):
                raise ValueError("terminal receipt v5 requires raw-logit audit v2")
            token_id_limit = self.server.raw_logit_audit.unpadded_vocab_size
        else:
            if not isinstance(self.server.raw_logit_audit, SmokeRawLogitAudit):
                raise ValueError("historical terminal receipt requires raw-logit audit v1")
            token_id_limit = vocabulary
        for probe in self.probes:
            for trial in probe.trials:
                if any(not 0 <= token_id < token_id_limit for token_id in trial.token_ids):
                    raise ValueError("probe returned a token outside the usable vocabulary")
        indices = tuple(gpu.cuda_ordinal for gpu in self.hardware)
        if indices != (0, 1):
            raise ValueError("receipt hardware must contain CUDA ordinals zero and one")
        if len({gpu.uuid for gpu in self.hardware}) != 2:
            raise ValueError("receipt GPU UUIDs must be unique")
        available_pci_ids = tuple(
            gpu.pci_bus_id for gpu in self.hardware if gpu.pci_bus_id_status == "available"
        )
        if len(available_pci_ids) != len(set(available_pci_ids)):
            raise ValueError("available receipt GPU PCI bus IDs must be unique")
        if len({gpu.driver_version for gpu in self.hardware}) != 1:
            raise ValueError("receipt GPUs must use one driver version")
        if len({gpu.cuda_driver_api_version for gpu in self.hardware}) != 1:
            raise ValueError("receipt GPUs must use one CUDA driver API version")
        if self.schema_version == "inkling-smoke-terminal-v3":
            historical_source_blobs = tuple(
                (identity.path, identity.git_blob_id)
                for identity in self.runtime.base_source_blob_ids
            )
            if (
                self.runtime.instrumentation_patch_sha256 != HISTORICAL_INSTRUMENTATION_PATCH_SHA256
                or self.runtime.patched_source_paths != _SMOKE_HISTORICAL_PATCHED_SOURCE_PATHS
                or historical_source_blobs != _SMOKE_HISTORICAL_PATCHED_SOURCE_BLOB_IDS
            ):
                raise ValueError("historical success receipt uses the current source patch")
            if self.gpu_topology is not None or "gpu_topology" in self.model_fields_set:
                raise ValueError("historical success receipt contains current topology evidence")
            if self.host.topology_schema_version != "inkling-smoke-hardware-topology-v3":
                raise ValueError("historical success receipt uses the wrong topology schema")
        elif self.schema_version == "inkling-smoke-terminal-v4":
            _require_current_cuda_backend_identities(self.server.backend_audit.identities)
            current_source_blobs = tuple(
                (identity.path, identity.git_blob_id)
                for identity in self.runtime.base_source_blob_ids
            )
            if (
                self.runtime.instrumentation_patch_sha256
                != LEGACY_CURRENT_INSTRUMENTATION_PATCH_SHA256
                or self.runtime.patched_source_paths != _SMOKE_PATCHED_SOURCE_PATHS
                or current_source_blobs != _SMOKE_PATCHED_SOURCE_BLOB_IDS
            ):
                raise ValueError("version 4 success receipt uses the wrong source patch")
        else:
            _require_current_cuda_backend_identities(self.server.backend_audit.identities)
            current_source_blobs = tuple(
                (identity.path, identity.git_blob_id)
                for identity in self.runtime.base_source_blob_ids
            )
            if (
                self.runtime.instrumentation_patch_sha256 != INSTRUMENTATION_PATCH_SHA256
                or self.runtime.patched_source_paths != _SMOKE_PATCHED_SOURCE_PATHS
                or current_source_blobs != _SMOKE_PATCHED_SOURCE_BLOB_IDS
            ):
                raise ValueError("version 5 success receipt uses the wrong source patch")
        if self.schema_version in {
            "inkling-smoke-terminal-v4",
            "inkling-smoke-terminal-v5",
        }:
            if self.gpu_topology is None:
                raise ValueError("current success receipt lacks CUDA peer topology evidence")
            if self.host.topology_schema_version != "inkling-smoke-hardware-topology-v4":
                raise ValueError("current success receipt uses the wrong topology schema")
            if self.gpu_topology.cuda_driver_api_version != (
                self.hardware[0].cuda_driver_api_version
            ):
                raise ValueError("CUDA peer topology uses a different driver API version")
            expected_edges = (
                (
                    self.hardware[0].cuda_ordinal,
                    self.hardware[0].uuid,
                    self.hardware[1].cuda_ordinal,
                    self.hardware[1].uuid,
                ),
                (
                    self.hardware[1].cuda_ordinal,
                    self.hardware[1].uuid,
                    self.hardware[0].cuda_ordinal,
                    self.hardware[0].uuid,
                ),
            )
            observed_edges = tuple(
                (
                    edge.source_cuda_ordinal,
                    edge.source_uuid,
                    edge.destination_cuda_ordinal,
                    edge.destination_uuid,
                )
                for edge in self.gpu_topology.edges
            )
            if observed_edges != expected_edges:
                raise ValueError("CUDA peer topology differs from the receipt GPU inventory")
        if self.host.topology_sha256 != smoke_hardware_topology_sha256(
            self.host,
            self.hardware,
            self.gpu_topology,
        ):
            raise ValueError("receipt hardware topology hash differs from its inventory")
        for gpu, peak_mib in zip(
            self.hardware,
            self.resources.gpu_peak_memory_used_mib,
            strict=True,
        ):
            if peak_mib > gpu.nvidia_smi_memory_total_mib:
                raise ValueError("receipt GPU peak memory exceeds installed memory")
        return self


def smoke_control_plane_tree_sha256(
    files: tuple[SmokeControlPlaneFile, ...] | list[SmokeControlPlaneFile],
) -> str:
    """Hash a canonical smoke control-plane file manifest."""

    payload = _canonical_json([item.model_dump(mode="json") for item in files]).encode()
    return hashlib.sha256(SMOKE_CONTROL_PLANE_HASH_DOMAIN + payload).hexdigest()


def smoke_control_plane_provenance(project_root: Path) -> SmokeControlPlaneProvenance:
    """Hash the runnable package plus exact smoke scripts, configs, and lock files."""

    root = project_root.resolve()
    package_root = root / "src" / "inkling_quant_lab"
    required = tuple(root / relative for relative in SMOKE_CONTROL_PLANE_REQUIRED_FILES)
    if not package_root.is_dir() or any(not path.is_file() for path in required):
        raise ConfigurationError(
            "Inkling smoke control-plane source tree is incomplete",
            component="inkling_smoke_control_plane",
        )
    candidates = (*required, *sorted(package_root.rglob("*"), key=lambda path: path.as_posix()))
    if package_root.is_symlink() or any(path.is_symlink() for path in candidates):
        raise ConfigurationError(
            "Inkling smoke control plane rejects symlinks",
            component="inkling_smoke_control_plane",
        )
    files: list[SmokeControlPlaneFile] = []
    for path in candidates:
        relative = path.relative_to(root)
        if not path.is_file() or "__pycache__" in relative.parts or path.name == ".DS_Store":
            continue
        if path.suffix in {".pyc", ".pyo"}:
            raise ConfigurationError(
                "Inkling smoke control plane rejects bytecode",
                component="inkling_smoke_control_plane",
            )
        payload = path.read_bytes()
        files.append(
            SmokeControlPlaneFile(
                path=relative.as_posix(),
                sha256=hashlib.sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
        )
    files.sort(key=lambda item: item.path)
    return SmokeControlPlaneProvenance(
        tree_sha256=smoke_control_plane_tree_sha256(files),
        file_count=len(files),
        files=tuple(files),
    )


def _file_identity(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def validate_deployed_smoke_control_plane(
    provenance_json: str,
    *,
    deployment_script: Path,
    deployed_package_root: Path,
    deployed_config: Path,
    deployed_reference: Path,
    deployed_patch: Path,
) -> SmokeControlPlaneProvenance:
    """Verify all remotely executable bytes against the caller's manifest."""

    try:
        provenance = SmokeControlPlaneProvenance.model_validate_json(provenance_json)
    except ValidationError as error:
        raise ConfigurationError(
            "Deployed smoke control-plane manifest is invalid",
            component="inkling_smoke_control_plane",
        ) from error
    expected = {item.path: item for item in provenance.files}
    if not set(SMOKE_CONTROL_PLANE_REQUIRED_FILES).issubset(expected):
        raise ConfigurationError(
            "Smoke control-plane manifest lacks a required file",
            component="inkling_smoke_control_plane",
        )
    runtime_paths = {
        "scripts/smoke_inkling_modal.py": Path(deployment_script),
        "configs/experiments/inkling_q3_k_m_smoke_modal.yaml": Path(deployed_config),
        "configs/experiments/inkling_q3_k_m_verified_export.json": Path(deployed_reference),
        "patches/inkling-smoke-a015409.patch": Path(deployed_patch),
    }
    package_root = Path(deployed_package_root)
    if package_root.is_symlink() or not package_root.is_dir():
        raise ConfigurationError(
            "Deployed Inkling Quant Lab package is missing or unsafe",
            component="inkling_smoke_control_plane",
        )
    observed: dict[str, tuple[str, int]] = {}
    for canonical, path in runtime_paths.items():
        if path.is_symlink() or not path.is_file():
            raise ConfigurationError(
                f"Deployed smoke runtime file is missing or unsafe: {canonical}",
                component="inkling_smoke_control_plane",
            )
        observed[canonical] = _file_identity(path)
    for path in sorted(package_root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(package_root)
        if (
            "__pycache__" in relative.parts
            or path.name == ".DS_Store"
            or path.suffix in {".pyc", ".pyo"}
        ):
            raise ConfigurationError(
                f"Deployed package contains unmanifested generated code: {relative.as_posix()}",
                component="inkling_smoke_control_plane",
            )
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise ConfigurationError(
                f"Deployed package has an unsafe entry: {relative.as_posix()}",
                component="inkling_smoke_control_plane",
            )
        if path.is_file():
            canonical = (Path("src/inkling_quant_lab") / relative).as_posix()
            observed[canonical] = _file_identity(path)
    expected_runtime = {
        path: (item.sha256, item.size_bytes)
        for path, item in expected.items()
        if path.startswith("src/inkling_quant_lab/") or path in runtime_paths
    }
    if observed != expected_runtime:
        raise ConfigurationError(
            "Deployed smoke source inventory differs from its control plane",
            component="inkling_smoke_control_plane",
        )
    return provenance


def smoke_run_id(config: InklingSmokeConfig, control_plane_sha256: str) -> str:
    """Derive the evidence namespace from subject, runtime, config, and code."""

    if re.fullmatch(r"[0-9a-f]{64}", control_plane_sha256) is None:
        raise ConfigurationError("smoke control-plane SHA-256 is invalid")
    return (
        f"inkling-smoke-{PINNED_MODEL_REVISION[:8]}-{PINNED_LLAMA_CPP_COMMIT[:8]}-"
        f"{config.config_hash()[:10]}-{control_plane_sha256[:10]}"
    )


def validate_smoke_launch_acknowledgement(
    acknowledgement_json: str,
    *,
    config: InklingSmokeConfig,
    control_plane: SmokeControlPlaneProvenance,
    run_id: str,
    launch_intent_sha256: str,
) -> SmokeLaunchAcknowledgement:
    """Validate the exact operator acknowledgement at the remote boundary."""

    try:
        acknowledgement = SmokeLaunchAcknowledgement.model_validate_json(acknowledgement_json)
    except ValidationError as error:
        raise ConfigurationError(
            "Smoke launch acknowledgement is invalid",
            component="inkling_smoke_paid_gate",
        ) from error
    expected = (
        config.config_hash(),
        config.verified_export_reference_sha256,
        control_plane.tree_sha256,
        launch_intent_sha256,
        run_id,
    )
    observed = (
        acknowledgement.smoke_config_hash,
        acknowledgement.verified_export_reference_sha256,
        acknowledgement.control_plane_sha256,
        acknowledgement.launch_intent_sha256,
        acknowledgement.run_id,
    )
    if observed != expected:
        raise ConfigurationError(
            "Smoke launch acknowledgement does not bind this exact run",
            component="inkling_smoke_paid_gate",
        )
    if acknowledgement.deployment.attempt_registry_name != (config.storage.attempt_registry):
        raise ConfigurationError(
            "Smoke launch acknowledgement uses the wrong attempt registry",
            component="inkling_smoke_paid_gate",
        )
    return acknowledgement


def strict_json_object(payload: str | bytes) -> dict[str, Any]:
    """Decode one JSON object while rejecting duplicate keys and non-finite constants."""

    if isinstance(payload, bytes):
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("JSON evidence must use UTF-8") from error
    elif isinstance(payload, str):
        text = payload
    else:
        raise TypeError("JSON evidence must be text or bytes")

    def object_without_duplicates(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON evidence contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite_constant(value: str) -> None:
        raise ValueError(f"JSON evidence contains non-finite constant {value}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError("JSON evidence contains a non-finite number")
        return parsed

    try:
        value = json.loads(
            text,
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_nonfinite_constant,
            parse_float=finite_float,
        )
    except json.JSONDecodeError as error:
        raise ValueError("JSON evidence is syntactically invalid") from error
    if not isinstance(value, dict):
        raise ValueError("JSON evidence root must be an object")
    return value


def _require_current_cuda_backend_identities(
    identities: tuple[SmokeBackendIdentityEvidence, ...],
) -> None:
    if any(identity.device_type != "gpu" for identity in identities):
        raise ValueError("current backend audit used a non-CUDA accelerator")
    cuda0_identity = (0, "CUDA0", "CUDA0")
    expected = {cuda0_identity, (1, "CUDA1", "CUDA1")}
    observed_dual_cuda_graph = False
    for graph_uid in {row.graph_uid for row in identities}:
        graph_rows = tuple(row for row in identities if row.graph_uid == graph_uid)
        observed = {(row.backend_index, row.backend_name, row.device_name) for row in graph_rows}
        if (
            len(graph_rows) != len(observed)
            or not observed.issubset(expected)
            or cuda0_identity not in observed
        ):
            raise ValueError(
                "current backend graph lacks the exact CUDA index and device identities"
            )
        observed_dual_cuda_graph |= observed == expected
    if not observed_dual_cuda_graph:
        raise ValueError("current backend audit lacks one exact dual-CUDA graph")


def _expected_server_command(
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    *,
    schema_version: Literal[
        "inkling-smoke-terminal-v3",
        "inkling-smoke-terminal-v4",
        "inkling-smoke-terminal-v5",
    ],
) -> tuple[str, ...]:
    first_shard = f"/subject/{reference.q3_shards[0].path}"
    projector = f"/subject/{reference.projector.path}"
    common_arguments = (
        "--model",
        first_shard,
        "--mmproj",
        projector,
        "--host",
        "127.0.0.1",
        "--port",
        "18080",
        "--ctx-size",
        str(config.runtime.context_size),
        "--n-gpu-layers",
        config.runtime.gpu_layers,
        "--n-cpu-moe",
        "0",
        "--split-mode",
        config.runtime.split_mode,
        "--tensor-split",
        ",".join(str(value) for value in config.runtime.tensor_split),
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
    )
    verbosity_arguments = ("--log-verbosity", str(config.runtime.log_verbosity))
    return (
        "/opt/llama.cpp/build/bin/llama-server",
        *(
            (*verbosity_arguments, *common_arguments)
            if schema_version
            in {
                "inkling-smoke-terminal-v4",
                "inkling-smoke-terminal-v5",
            }
            else (*common_arguments, *verbosity_arguments)
        ),
        "--no-webui",
    )


def _validate_smoke_failure_outcome_path(
    outcome_path: str | PurePosixPath,
    *,
    run_id: str,
    receipt_sha256: str,
) -> None:
    raw_path = str(outcome_path)
    parsed = PurePosixPath(raw_path)
    if (
        not raw_path
        or parsed.is_absolute()
        or "\\" in raw_path
        or "//" in raw_path
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or parsed.as_posix() != raw_path
    ):
        raise ValueError("smoke failure outcome path must be canonical relative POSIX text")
    filename = f"smoke_test.failed.{receipt_sha256}.json"
    if parsed.name != filename:
        raise ValueError("smoke failure outcome filename does not bind the receipt digest")
    if len(parsed.parts) > 1 and parsed.parts != (
        "runs",
        run_id,
        "control",
        "outcomes",
        filename,
    ):
        raise ValueError("smoke failure outcome path differs from the exact run location")


def _validate_smoke_failure_launch_schema(
    *,
    schema_version: str,
    config: InklingSmokeConfig,
    control_plane: SmokeControlPlaneProvenance,
    run_id: str,
) -> None:
    expected_run_id = smoke_run_id(config, control_plane.tree_sha256)
    if run_id != expected_run_id:
        raise ValueError("terminal smoke failure run ID is not derived from its launch inputs")

    patch_entries = tuple(
        item
        for item in control_plane.files
        if item.path == config.runtime.instrumentation_patch_path
    )
    if len(patch_entries) != 1:
        raise ValueError("smoke failure control plane lacks one instrumentation patch")
    patch_sha256 = patch_entries[0].sha256
    if patch_sha256 != config.runtime.instrumentation_patch_sha256:
        raise ValueError("smoke failure control-plane patch differs from the smoke config")

    expected_patch_by_schema = {
        "inkling-smoke-terminal-v2": HISTORICAL_INSTRUMENTATION_PATCH_SHA256,
        "inkling-smoke-terminal-v3": HISTORICAL_INSTRUMENTATION_PATCH_SHA256,
        "inkling-smoke-terminal-v4": LEGACY_CURRENT_INSTRUMENTATION_PATCH_SHA256,
        "inkling-smoke-terminal-v5": INSTRUMENTATION_PATCH_SHA256,
    }
    expected_patch = expected_patch_by_schema.get(schema_version)
    if expected_patch is None:
        raise ValueError("terminal smoke failure schema version is unsupported")
    if patch_sha256 != expected_patch:
        raise ValueError("terminal smoke failure schema differs from its instrumentation patch")


def validate_smoke_failure_receipt(
    value: Mapping[str, Any] | str | bytes,
    *,
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    run_id: str,
    launch_intent_sha256: str,
    outcome_path: str | PurePosixPath | None = None,
) -> SmokeFailureReceipt:
    """Validate immutable failure evidence against its exact launch inputs.

    Text and byte inputs use the duplicate-key-rejecting JSON parser. A mapping
    is accepted for callers that have already parsed trusted local bytes.
    """

    if isinstance(value, str | bytes):
        raw = strict_json_object(value)
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise TypeError("terminal smoke failure receipt must be JSON text, bytes, or a mapping")

    receipt_sha256 = raw.get("receipt_sha256")
    if not isinstance(receipt_sha256, str):
        raise ValueError("terminal smoke failure receipt lacks a valid self-hash")
    try:
        computed_sha256 = smoke_terminal_receipt_sha256(raw)
    except ValueError as error:
        raise ValueError("terminal smoke failure receipt hash domain is invalid") from error
    if receipt_sha256 != computed_sha256:
        raise ValueError("terminal smoke failure receipt self-hash does not match its payload")

    schema_version = raw.get("schema_version")
    try:
        if schema_version == "inkling-smoke-terminal-v2":
            receipt: SmokeFailureReceipt = SmokeFailureReceiptV2.model_validate(raw)
        elif schema_version == "inkling-smoke-terminal-v3":
            receipt = SmokeFailureReceiptV3.model_validate(raw)
        elif schema_version == "inkling-smoke-terminal-v4":
            receipt = SmokeFailureReceiptV4.model_validate(raw)
        elif schema_version == "inkling-smoke-terminal-v5":
            receipt = SmokeFailureReceiptV5.model_validate(raw)
        else:
            raise ValueError("unsupported terminal smoke failure receipt schema version")
    except ValidationError as error:
        raise ValueError("terminal smoke failure receipt schema is invalid") from error

    if config.verified_export_reference_sha256 != reference.reference_sha256:
        raise ValueError("smoke failure validation inputs use different export references")
    _validate_smoke_failure_launch_schema(
        schema_version=receipt.schema_version,
        config=config,
        control_plane=control_plane,
        run_id=run_id,
    )
    observed_bindings = (
        receipt.stage,
        receipt.run_id,
        receipt.subject_run_id,
        receipt.smoke_config_hash,
        receipt.verified_export_reference_sha256,
        receipt.control_plane_sha256,
        receipt.launch_intent_sha256,
    )
    expected_bindings = (
        SMOKE_STAGE,
        run_id,
        reference.subject_run_id,
        config.config_hash(),
        reference.reference_sha256,
        control_plane.tree_sha256,
        launch_intent_sha256,
    )
    if observed_bindings != expected_bindings:
        raise ValueError("terminal smoke failure receipt does not bind the exact launch")

    if outcome_path is not None:
        _validate_smoke_failure_outcome_path(
            outcome_path,
            run_id=run_id,
            receipt_sha256=receipt.receipt_sha256,
        )
    return receipt


def validate_smoke_terminal_receipt(
    value: Mapping[str, Any],
    *,
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
    run_id: str,
) -> SmokeTerminalReceipt:
    """Validate complete success evidence against the exact run and export inputs."""

    raw = dict(value)
    receipt_sha256 = raw.get("receipt_sha256")
    if not isinstance(receipt_sha256, str) or receipt_sha256 != smoke_terminal_receipt_sha256(raw):
        raise ValueError("terminal smoke receipt self-hash does not match its payload")
    try:
        receipt = SmokeTerminalReceipt.model_validate(raw)
    except ValidationError as error:
        raise ValueError("terminal smoke receipt schema is invalid") from error

    expected_subject = {
        "run_id": reference.subject_run_id,
        "model_id": reference.model_id,
        "revision": reference.revision,
        "architecture": reference.architecture,
        "quant_type": reference.quant_type,
        "mtp": reference.mtp,
        "verified_export_reference_sha256": reference.reference_sha256,
        "q3_shard_count": EXPECTED_Q3_SHARD_COUNT,
        "q3_total_bytes": EXPECTED_Q3_TOTAL_BYTES,
        "projector_sha256": reference.projector.sha256,
    }
    if receipt.subject.model_dump(mode="json") != expected_subject:
        raise ValueError("terminal smoke receipt subject differs from the exact export")

    exact_identity = (
        receipt.run_id,
        receipt.smoke_config_hash,
        receipt.control_plane_sha256,
        receipt.control_plane_file_count,
    )
    expected_identity = (
        run_id,
        config.config_hash(),
        control_plane.tree_sha256,
        control_plane.file_count,
    )
    if exact_identity != expected_identity:
        raise ValueError("terminal smoke receipt differs from the exact run identity")
    if receipt.schema_version == "inkling-smoke-terminal-v5":
        output_vocabulary = config.output_vocabulary
        raw_logit_audit = receipt.server.raw_logit_audit
        if output_vocabulary is None or not isinstance(
            raw_logit_audit,
            SmokeRawLogitAuditV2,
        ):
            raise ValueError("terminal smoke v5 lacks its configured output vocabulary")
        if (
            raw_logit_audit.vocab_size,
            raw_logit_audit.unpadded_vocab_size,
            raw_logit_audit.padded_vocab_size,
        ) != (
            output_vocabulary.vocab_size,
            output_vocabulary.unpadded_vocab_size,
            output_vocabulary.padded_vocab_size,
        ):
            raise ValueError("terminal smoke raw-logit vocabulary differs from the exact config")
    invocation_identity = (
        receipt.invocation.run_id,
        receipt.invocation.launch_intent_sha256,
        receipt.invocation.smoke_config_hash,
        receipt.invocation.control_plane_sha256,
        receipt.invocation.attempt_registry_name,
    )
    if invocation_identity != (
        run_id,
        receipt.launch_intent_sha256,
        config.config_hash(),
        control_plane.tree_sha256,
        config.storage.attempt_registry,
    ):
        raise ValueError("terminal smoke invocation does not bind the exact launch")

    expected_artifacts = tuple(
        SmokeArtifactIdentity.model_validate(artifact.model_dump(mode="json"))
        for artifact in (
            *reference.q3_shards,
            reference.projector,
            reference.export_manifest,
            reference.verify_receipt,
            reference.quantize_receipt,
            reference.mmproj_receipt,
        )
    )
    if receipt.artifact_rehash.artifacts != expected_artifacts:
        raise ValueError("terminal smoke artifact rehash differs from the exact export")
    if (
        sum(
            artifact.size_bytes
            for artifact in receipt.artifact_rehash.artifacts[:EXPECTED_Q3_SHARD_COUNT]
        )
        != EXPECTED_Q3_TOTAL_BYTES
    ):
        raise ValueError("terminal smoke Q3 rehash bytes differ from the exact export")
    if receipt.artifact_rehash.artifacts[EXPECTED_Q3_SHARD_COUNT].size_bytes != (
        EXPECTED_PROJECTOR_BYTES
    ):
        raise ValueError("terminal smoke projector bytes differ from the exact export")

    if receipt.runtime.cmake_definitions != config.runtime.cmake_definitions:
        raise ValueError("terminal smoke CMake definitions differ from the exact config")
    runtime_instrumentation = (
        receipt.runtime.instrumentation_schema_version,
        receipt.runtime.instrumentation_patch_sha256,
    )
    configured_instrumentation = (
        config.runtime.instrumentation_schema_version,
        config.runtime.instrumentation_patch_sha256,
    )
    if runtime_instrumentation != configured_instrumentation:
        raise ValueError("terminal smoke instrumentation differs from the exact config")
    patch_entries = tuple(
        item for item in control_plane.files if item.path == "patches/inkling-smoke-a015409.patch"
    )
    if len(patch_entries) != 1:
        raise ValueError("smoke control plane lacks the pinned instrumentation patch")
    if receipt.runtime.instrumentation_patch_sha256 != patch_entries[0].sha256:
        raise ValueError("terminal smoke instrumentation patch differs from its control plane")

    if receipt.server.command != _expected_server_command(
        config,
        reference,
        schema_version=receipt.schema_version,
    ):
        raise ValueError("terminal smoke server command differs from the exact config")
    if receipt.server.loader_offload.cuda_device_count != config.resources.gpu_count:
        raise ValueError("terminal smoke loader observed the wrong CUDA device count")
    if len(receipt.hardware) != config.resources.gpu_count:
        raise ValueError("terminal smoke hardware cardinality differs from the exact config")
    requested_host_capacity = (
        receipt.host.requested_cpu_cores,
        receipt.host.requested_ram_bytes,
    )
    configured_host_capacity = (
        config.resources.cpu_cores,
        config.resources.memory_gib * 1024**3,
    )
    if requested_host_capacity != configured_host_capacity:
        raise ValueError("terminal smoke host request differs from the exact config")

    if receipt.claims.model_dump(mode="json") != config.claims.model_dump(mode="json"):
        raise ValueError("terminal smoke claim limits differ from the exact config")
    if receipt.evidence_policy.model_dump(mode="json") != config.evidence.model_dump(mode="json"):
        raise ValueError("terminal smoke evidence policy differs from the exact config")

    for observed, expected in zip(receipt.probes, config.probes, strict=True):
        identity = (
            observed.probe_id,
            observed.modality,
            observed.prompt_sha256,
            observed.fixture,
            observed.seed,
            observed.temperature,
        )
        expected_probe = (
            expected.probe_id,
            expected.modality,
            expected.prompt_sha256,
            expected.fixture,
            expected.seed,
            expected.temperature,
        )
        if identity != expected_probe:
            raise ValueError("terminal smoke probe differs from the exact config")
        if len(observed.trials) != expected.trials:
            raise ValueError("terminal smoke probe trial count differs from the exact config")

    return receipt


def smoke_terminal_receipt_sha256(value: Mapping[str, Any]) -> str:
    """Return the domain-separated self-hash for one terminal smoke receipt."""

    payload = dict(value)
    payload.pop("receipt_sha256", None)
    schema_version = payload.get("schema_version")
    if schema_version == "inkling-smoke-terminal-v2":
        if payload.get("status") != "failed":
            raise ValueError("terminal smoke receipt v2 is valid only for historical failures")
        domain = _SMOKE_RECEIPT_HASH_DOMAIN_V2
    elif schema_version == "inkling-smoke-terminal-v3":
        domain = _SMOKE_RECEIPT_HASH_DOMAIN_V3
    elif schema_version == "inkling-smoke-terminal-v4":
        domain = _SMOKE_RECEIPT_HASH_DOMAIN_V4
    elif schema_version == "inkling-smoke-terminal-v5":
        domain = SMOKE_RECEIPT_HASH_DOMAIN
    else:
        raise ValueError("unsupported terminal smoke receipt schema version")
    return hashlib.sha256(domain + _canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = [
    "SMOKE_ATTEMPT_REGISTRY_MIN_CREATED_AT_UTC",
    "SMOKE_CONTROL_PLANE_REQUIRED_FILES",
    "SMOKE_DEPLOYMENT_TAG_HASH_PREFIX_LENGTH",
    "SMOKE_ENVIRONMENT_NAME",
    "SMOKE_STAGE",
    "SMOKE_WORKSPACE_BUDGET_USD",
    "SmokeControlPlaneFile",
    "SmokeControlPlaneProvenance",
    "SmokeCudaPeerEdgeEvidence",
    "SmokeFailureReceipt",
    "SmokeFailureReceiptV2",
    "SmokeFailureReceiptV3",
    "SmokeFailureReceiptV4",
    "SmokeFailureReceiptV5",
    "SmokeGpuTopologyEvidence",
    "SmokeHostEvidence",
    "SmokeInvocationEvidence",
    "SmokeLaunchAcknowledgement",
    "SmokeLaunchDeploymentIdentity",
    "SmokeNvidiaSmiTopologyDiagnostic",
    "SmokeRawLogitAudit",
    "SmokeRawLogitAuditV2",
    "SmokeSafeFailureSignals",
    "SmokeSubprocessFailureEvidence",
    "SmokeTerminalReceipt",
    "canonical_python_package_inventory",
    "canonical_smoke_attempt_registry_created_at_utc",
    "immutable_source_tree_identity",
    "parse_cgroup_cpu_quota_millicores",
    "parse_cgroup_memory_limit_bytes",
    "parse_dpkg_inventory",
    "parse_nvcc_version",
    "parse_proc_cpu_model",
    "parse_proc_mem_total_bytes",
    "resolve_current_process_cgroup_hierarchy_paths",
    "resolve_current_process_cgroup_leaf_paths",
    "resolve_process_cgroup_file",
    "smoke_control_plane_provenance",
    "smoke_control_plane_tree_sha256",
    "smoke_deployment_tag",
    "smoke_hardware_topology_sha256",
    "smoke_package_manifest_sha256",
    "smoke_run_id",
    "smoke_terminal_receipt_sha256",
    "strict_json_object",
    "validate_deployed_smoke_control_plane",
    "validate_smoke_attempt_registry_created_at_utc",
    "validate_smoke_failure_receipt",
    "validate_smoke_launch_acknowledgement",
    "validate_smoke_terminal_receipt",
]
