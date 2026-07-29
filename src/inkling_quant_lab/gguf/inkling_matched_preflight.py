"""Compile the checked Inkling matched plan without starting remote work."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, Self, TypeVar

from pydantic import Field, ValidationError, model_validator

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.gguf.inkling import (
    INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
    InklingSourceAdoptionReference,
)
from inkling_quant_lab.gguf.inkling_matched import (
    BF16_SUBJECT_REFERENCE_RELATIVE_PATH,
    CAPACITY_SCREEN_LIMITATION,
    MATCHED_CELL_CONFIG_RELATIVE_PATH,
    InklingBF16SubjectReference,
    InklingMatchedCellBundle,
    InklingMatchedCellConfig,
    MatchedClaimLimits,
    MatchedExecutionConfig,
    MatchedRuntimeConfig,
    build_matched_cell_bundle,
    parse_matched_cell_config_bytes,
)
from inkling_quant_lab.gguf.inkling_smoke import (
    VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH,
    InklingVerifiedExportReference,
)

MATCHED_PREFLIGHT_PLAN_HASH_DOMAIN: Final = b"inkling-matched-preflight-plan-v1\0"
MATCHED_PREFLIGHT_INVENTORY_HASH_DOMAIN: Final = b"inkling-matched-preflight-inventory-v1\0"
MATCHED_SCOPE_WARNING: Final = (
    "Read each machine-readable experiment record before using its result. "
    "Do not apply a result to a different model, dataset, runtime, software, "
    "hardware, or protocol."
)

_CONTROL_FILE_ROLES: Final = (
    ("matched_config", MATCHED_CELL_CONFIG_RELATIVE_PATH),
    ("bf16_reference", BF16_SUBJECT_REFERENCE_RELATIVE_PATH),
    ("q3_reference", VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH),
    ("source_reference", INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH),
    ("instrumentation_patch", "patches/inkling-smoke-a015409.patch"),
)
_NOT_EXECUTED_STAGES: Final = (
    "rehash_bf16_subject",
    "rehash_q3_subject",
    "screen_aggregate_capacity",
    "smoke_bf16_subject",
    "smoke_q3_subject",
    "verify_matched_smoke_evidence",
)
# Update this complete raw identity set in one reviewed change when a control changes.
_MATCHED_CONFIG_CONTROL_SHA256: Final = (
    "d3cbf6ae5abc9b1c798820ffc2d3b8e4181b318078b538312534ce6641ad21f5"
)
_MATCHED_CONFIG_CONTROL_SIZE_BYTES: Final = 7_655
_BF16_REFERENCE_CONTROL_SHA256: Final = (
    "1aa0fc8be7c1a5f6eb4ea480684a660ce443e0830c05668b141d5b88e5a762ff"
)
_BF16_REFERENCE_CONTROL_SIZE_BYTES: Final = 10_234
_Q3_REFERENCE_CONTROL_SHA256: Final = (
    "1086fec05c9b6b4400caf9be21cbbbb8e6e5c4138164c5e02af010359f84ad96"
)
_Q3_REFERENCE_CONTROL_SIZE_BYTES: Final = 9_518
_SOURCE_REFERENCE_CONTROL_SHA256: Final = (
    "c156c4d772a8b0a039af97c83971d4042975ea4bec8b2ae57e52cef267e21187"
)
_SOURCE_REFERENCE_CONTROL_SIZE_BYTES: Final = 4_959
_INSTRUMENTATION_PATCH_CONTROL_SHA256: Final = (
    "005f1f342511fc3fc843bdcc7be814ed8a60e67033b733eb7e7e4af53925be04"
)
_INSTRUMENTATION_PATCH_SIZE_BYTES: Final = 48_409
_EXPECTED_CONTROL_IDENTITIES: Final[tuple[tuple[str, str, str, int], ...]] = (
    (
        "matched_config",
        MATCHED_CELL_CONFIG_RELATIVE_PATH,
        _MATCHED_CONFIG_CONTROL_SHA256,
        _MATCHED_CONFIG_CONTROL_SIZE_BYTES,
    ),
    (
        "bf16_reference",
        BF16_SUBJECT_REFERENCE_RELATIVE_PATH,
        _BF16_REFERENCE_CONTROL_SHA256,
        _BF16_REFERENCE_CONTROL_SIZE_BYTES,
    ),
    (
        "q3_reference",
        VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH,
        _Q3_REFERENCE_CONTROL_SHA256,
        _Q3_REFERENCE_CONTROL_SIZE_BYTES,
    ),
    (
        "source_reference",
        INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
        _SOURCE_REFERENCE_CONTROL_SHA256,
        _SOURCE_REFERENCE_CONTROL_SIZE_BYTES,
    ),
    (
        "instrumentation_patch",
        "patches/inkling-smoke-a015409.patch",
        _INSTRUMENTATION_PATCH_CONTROL_SHA256,
        _INSTRUMENTATION_PATCH_SIZE_BYTES,
    ),
)
_READ_CHUNK_SIZE: Final = 1024 * 1024
_MAX_CONTROL_FILE_SIZE_BYTES: Final = 16 * 1024 * 1024
_NATIVE_OPEN_SUPPORTS_DIR_FD: Final = os.open in os.supports_dir_fd
_NATIVE_STAT_SUPPORTS_FOLLOW_SYMLINKS: Final = os.stat in os.supports_follow_symlinks

_ReferenceT = TypeVar("_ReferenceT")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + _canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class _CapturedControlFile:
    """One descriptor-bound local control and the exact bytes read from it."""

    report: MatchedLocalControlFile
    content: bytes


class MatchedLocalControlFile(StrictFrozenModel):
    """One local control input read during the offline preflight."""

    role: Literal[
        "matched_config",
        "bf16_reference",
        "q3_reference",
        "source_reference",
        "instrumentation_patch",
    ]
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    regular_file: Literal[True]
    symlink: Literal[False]
    contained_in_project: Literal[True]


class MatchedMountPlan(StrictFrozenModel):
    """One exact future mount declaration."""

    name: Literal["bf16", "q3", "source", "evidence"]
    volume: str
    volume_version: Literal[1]
    sub_path: str | None
    mount_path: str
    read_only: bool
    create_if_missing: bool

    @model_validator(mode="after")
    def exact_mount_policy(self) -> Self:
        subject = self.name != "evidence"
        if subject and (self.sub_path is None or not self.read_only or self.create_if_missing):
            raise ValueError("subject mounts must be existing read-only subpath mounts")
        if not subject and (
            self.sub_path is not None or self.read_only or not self.create_if_missing
        ):
            raise ValueError("evidence must use one separate writable volume")
        return self


class MatchedPreflightStage(StrictFrozenModel):
    """Truth state for one checked stage."""

    ordinal: int = Field(ge=0)
    name: str
    status: Literal["passed", "not_executed"]
    execution_performed: Literal[False]
    measurement_observed: Literal[False]
    external_artifact_bytes_rehashed: Literal[False]
    paid_compute_started: Literal[False]


class MatchedInventoryAssignment(StrictFrozenModel):
    """One declared remote file owned by one future rehash stage."""

    stage: Literal["rehash_bf16_subject", "rehash_q3_subject"]
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    declared_only: Literal[True]
    rehash_performed: Literal[False]


class MatchedSubjectDeclaration(StrictFrozenModel):
    """One exact subject identity from a checked reference."""

    role: Literal["bf16", "q3"]
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: Literal["BF16", "Q3_K_M"]
    shard_count: Literal[49]
    shard_total_bytes: int = Field(gt=0)
    shard_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_artifact_bytes_rehashed: Literal[False]


class MatchedSharedProjectorDeclaration(StrictFrozenModel):
    """The shared projector declared by both subject references."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)
    reference_identity_matches: Literal[True]
    external_artifact_bytes_rehashed: Literal[False]


class MatchedTokenizerDeclaration(StrictFrozenModel):
    """The exact tokenizer subset declared for both subjects."""

    artifact_count: Literal[6]
    declared_total_bytes: int = Field(gt=0)
    declared_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    external_artifact_bytes_rehashed: Literal[False]


class MatchedDeclaredCapacity(StrictFrozenModel):
    """Planning arithmetic that is not observed hardware evidence."""

    gpu_count: Literal[8]
    configured_minimum_gpu_memory_bytes: int = Field(gt=0)
    configured_minimum_total_gpu_memory_bytes: int = Field(gt=0)
    capacity_reserve_basis_points: Literal[1000]
    declared_headroom_bytes: int = Field(gt=0)
    declared_usable_bytes: int = Field(gt=0)
    bf16_subject_bytes: int = Field(gt=0)
    q3_subject_bytes: int = Field(gt=0)
    sequential_peak_subject_bytes: int = Field(gt=0)
    declared_remaining_bytes: int
    hardware_probed: Literal[False]
    allocation_observed: Literal[False]
    capacity_screen_executed: Literal[False]
    runtime_fit_proven: Literal[False]
    limitation: Literal[
        "The aggregate capacity screen does not prove per-device tensor placement, "
        "runtime workspace fit, or successful inference."
    ]


class MatchedDeclaredResourceCell(StrictFrozenModel):
    """The exact requested cell, recorded as a declaration and not an observation."""

    provider: Literal["modal"]
    gpu_type: Literal["B300"]
    gpu_count: Literal[8]
    compute_capability: Literal["10.3"]
    minimum_gpu_memory_bytes: Literal[287000000000]
    capacity_reserve_basis_points: Literal[1000]
    capacity_strategy: Literal["sequential_peak_plus_reserve"]
    cpu_cores: Literal[16]
    memory_gib: Literal[64]
    ephemeral_disk_mib: Literal[524288]
    startup_timeout_seconds: Literal[1800]
    function_timeout_seconds: Literal[14400]
    max_attempts: Literal[1]
    max_recovery_attempts: Literal[0]
    declared_only: Literal[True]


class MatchedPreflightFacts(StrictFrozenModel):
    """Facts that keep the offline and paid boundaries explicit."""

    local_control_files_verified: Literal[True]
    provider_contacted: Literal[False]
    remote_volume_inspected: Literal[False]
    network_access_performed: Literal[False]
    subprocess_execution_performed: Literal[False]
    local_write_performed: Literal[False]
    external_artifact_bytes_rehashed: Literal[False]
    artifact_rehash_performed: Literal[False]
    remote_runtime_binaries_verified: Literal[False]
    remote_execution_performed: Literal[False]
    hardware_probed: Literal[False]
    allocation_observed: Literal[False]
    capacity_screen_executed: Literal[False]
    bf16_smoke_executed: Literal[False]
    q3_smoke_executed: Literal[False]
    matched_smoke_verified: Literal[False]
    smoke_passed: Literal[False]
    measurement_execution_performed: Literal[False]
    measurement_ready: Literal[False]
    launch_authorized: Literal[False]
    paid_compute_started: Literal[False]


class InklingMatchedPreflightReport(StrictFrozenModel):
    """Deterministic read-only result for the checked planning record."""

    schema_version: Literal["inkling-matched-preflight-v1"]
    status: Literal["ready_for_operator_review"]
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: Literal["thinkingmachines/Inkling"]
    revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"]
    architecture: Literal["InklingForConditionalGeneration"]
    control_files: tuple[MatchedLocalControlFile, ...]
    config: InklingMatchedCellConfig
    bf16_reference: InklingBF16SubjectReference
    q3_reference: InklingVerifiedExportReference
    source_reference: InklingSourceAdoptionReference
    subjects: tuple[MatchedSubjectDeclaration, MatchedSubjectDeclaration]
    shared_projector: MatchedSharedProjectorDeclaration
    tokenizer: MatchedTokenizerDeclaration
    runtime: MatchedRuntimeConfig
    mounts: tuple[MatchedMountPlan, MatchedMountPlan, MatchedMountPlan, MatchedMountPlan]
    inventory_assignments: tuple[MatchedInventoryAssignment, ...]
    stages: tuple[
        MatchedPreflightStage,
        MatchedPreflightStage,
        MatchedPreflightStage,
        MatchedPreflightStage,
        MatchedPreflightStage,
        MatchedPreflightStage,
        MatchedPreflightStage,
    ]
    declared_capacity: MatchedDeclaredCapacity
    declared_resource_cell: MatchedDeclaredResourceCell
    execution: MatchedExecutionConfig
    claims: MatchedClaimLimits
    facts: MatchedPreflightFacts
    warning: Literal[
        "Read each machine-readable experiment record before using its result. "
        "Do not apply a result to a different model, dataset, runtime, software, "
        "hardware, or protocol."
    ]

    @model_validator(mode="after")
    def fail_closed_truth(self) -> Self:
        if _control_identities(self.control_files) != _EXPECTED_CONTROL_IDENTITIES:
            raise ValueError("matched preflight control files differ from the checked inputs")
        try:
            _validate_control_config_binding(self.control_files, self.config)
        except ConfigurationError as error:
            raise ValueError(
                "matched preflight control files differ from the embedded config"
            ) from error

        try:
            bundle = build_matched_cell_bundle(
                self.config,
                self.bf16_reference,
                self.q3_reference,
                self.source_reference,
            )
        except ConfigurationError as error:
            raise ValueError("matched preflight embeds inconsistent reference records") from error

        if self.config_hash != self.config.config_hash():
            raise ValueError("matched preflight config hash differs from its embedded config")
        if (
            self.model_id != self.config.model_id
            or self.revision != self.config.revision
            or self.architecture != self.config.architecture
        ):
            raise ValueError("matched preflight model identity differs from its embedded config")

        expected_fields: tuple[tuple[str, object, object], ...] = (
            ("subjects", self.subjects, _subject_declarations(bundle)),
            (
                "shared projector",
                self.shared_projector,
                _shared_projector_declaration(bundle),
            ),
            ("tokenizer", self.tokenizer, _tokenizer_declaration(bundle)),
            ("runtime", self.runtime, bundle.config.runtime),
            ("mounts", self.mounts, _mounts(bundle)),
            (
                "inventory assignments",
                self.inventory_assignments,
                _inventory_assignments(bundle),
            ),
            ("stages", self.stages, _stages(bundle)),
            ("declared capacity", self.declared_capacity, _declared_capacity(bundle)),
            (
                "declared resource cell",
                self.declared_resource_cell,
                _declared_resource_cell(bundle),
            ),
            ("execution", self.execution, bundle.config.execution),
            ("claims", self.claims, bundle.config.claims),
            ("facts", self.facts, _offline_facts()),
        )
        for label, observed, expected in expected_fields:
            if observed != expected:
                raise ValueError(f"matched preflight {label} differs from the checked plan")
        if self.plan_sha256 != self.computed_plan_sha256():
            raise ValueError("matched preflight plan SHA-256 differs from its payload")
        return self

    def canonical_dict(self) -> dict[str, Any]:
        """Return the complete deterministic report."""

        return self.model_dump(mode="json")

    def canonical_json(self) -> str:
        """Serialize the complete report without a trailing line feed."""

        return _canonical_json(self.canonical_dict())

    def computed_plan_sha256(self) -> str:
        """Hash all report fields except the stored plan hash."""

        payload = self.model_dump(mode="json", exclude={"plan_sha256"})
        return _domain_hash(MATCHED_PREFLIGHT_PLAN_HASH_DOMAIN, payload)


def _canonical_control_relative_path(value: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or "\\" in value
        or "\x00" in value
        or value != relative.as_posix()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ConfigurationError(
            "Matched preflight control path must be canonical and project-relative",
            component="inkling_matched_preflight",
            details={"path": value},
        )
    return relative


def _filesystem_error(action: str, path: object, error: OSError) -> ConfigurationError:
    return ConfigurationError(
        f"Unable to {action} matched preflight input {path}: {error}",
        component="inkling_matched_preflight",
    )


def _close_descriptors(
    descriptors: tuple[tuple[int, object], ...],
    *,
    suppress_errors: bool = False,
) -> None:
    """Close every owned descriptor once and preserve the first close failure."""

    first_failure: tuple[object, OSError] | None = None
    for descriptor, label in descriptors:
        try:
            os.close(descriptor)
        except OSError as caught_error:
            if first_failure is None:
                first_failure = (label, caught_error)
    if first_failure is not None and not suppress_errors:
        label, close_error = first_failure
        raise _filesystem_error("close", label, close_error) from close_error


def _descriptor_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_secure_descriptor_support() -> None:
    required_flags = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    missing = tuple(name for name in required_flags if not getattr(os, name, 0))
    if missing or not _NATIVE_OPEN_SUPPORTS_DIR_FD or not _NATIVE_STAT_SUPPORTS_FOLLOW_SYMLINKS:
        raise ConfigurationError(
            "Matched preflight requires POSIX no-follow descriptor traversal",
            component="inkling_matched_preflight",
            details={"missing_os_flags": list(missing)},
        )


def _open_project_root(project_root: str | Path) -> tuple[Path, int]:
    _require_secure_descriptor_support()
    try:
        root = Path(project_root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ConfigurationError(
            f"Unable to resolve matched preflight project root {project_root}: {error}",
            component="inkling_matched_preflight",
        ) from error

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise _filesystem_error("open project root", root, error) from error
    try:
        try:
            opened = os.fstat(descriptor)
            named = os.stat(root, follow_symlinks=False)
        except OSError as error:
            raise _filesystem_error("inspect project root", root, error) from error
        if not stat.S_ISDIR(opened.st_mode) or _descriptor_identity(opened) != (
            _descriptor_identity(named)
        ):
            raise ConfigurationError(
                f"Matched preflight project root changed while it was opened: {root}",
                component="inkling_matched_preflight",
            )
        return root, descriptor
    except BaseException:
        _close_descriptors(((descriptor, root),), suppress_errors=True)
        raise


def _open_control_parent(
    root_descriptor: int,
    *,
    root: Path,
    relative: PurePosixPath,
) -> int:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        current = os.dup(root_descriptor)
    except OSError as error:
        raise _filesystem_error("duplicate project root descriptor for", root, error) from error
    owned: list[tuple[int, object]] = [(current, root)]
    try:
        for part in relative.parts[:-1]:
            try:
                child = os.open(part, flags, dir_fd=current)
            except OSError as error:
                raise _filesystem_error(
                    "open directory component of",
                    root / Path(relative.as_posix()),
                    error,
                ) from error
            owned.append((child, part))
            try:
                metadata = os.fstat(child)
            except OSError as error:
                raise _filesystem_error(
                    "inspect directory component of",
                    root / Path(relative.as_posix()),
                    error,
                ) from error
            if not stat.S_ISDIR(metadata.st_mode):
                raise ConfigurationError(
                    "Matched preflight control path contains a non-directory component",
                    component="inkling_matched_preflight",
                    details={"path": relative.as_posix(), "component": part},
                )
            current = child
    except BaseException:
        _close_descriptors(tuple(reversed(owned)), suppress_errors=True)
        raise

    parent = owned.pop()
    try:
        _close_descriptors(tuple(reversed(owned)))
    except BaseException:
        _close_descriptors((parent,), suppress_errors=True)
        raise
    return parent[0]


def _capture_control_file(
    root: Path,
    root_descriptor: int,
    *,
    role: str,
    relative_path: str,
) -> _CapturedControlFile:
    relative = _canonical_control_relative_path(relative_path)
    candidate = root / Path(relative.as_posix())
    parent_descriptor = _open_control_parent(
        root_descriptor,
        root=root,
        relative=relative,
    )
    descriptor: int | None = None
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        try:
            descriptor = os.open(relative.name, flags, dir_fd=parent_descriptor)
        except OSError as error:
            raise _filesystem_error("open", candidate, error) from error
        try:
            before = os.fstat(descriptor)
        except OSError as error:
            raise _filesystem_error("inspect", candidate, error) from error
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > _MAX_CONTROL_FILE_SIZE_BYTES
        ):
            raise ConfigurationError(
                f"Matched preflight input is not an allowed non-empty regular file: {candidate}",
                component="inkling_matched_preflight",
            )

        chunks: list[bytes] = []
        total_bytes = 0
        while True:
            try:
                chunk = os.read(descriptor, _READ_CHUNK_SIZE)
            except OSError as error:
                raise _filesystem_error("read", candidate, error) from error
            if not chunk:
                break
            chunks.append(chunk)
            total_bytes += len(chunk)
            if total_bytes > _MAX_CONTROL_FILE_SIZE_BYTES:
                raise ConfigurationError(
                    f"Matched preflight input grew beyond the size limit: {candidate}",
                    component="inkling_matched_preflight",
                )
        content = b"".join(chunks)
        try:
            after = os.fstat(descriptor)
        except OSError as error:
            raise _filesystem_error("reinspect", candidate, error) from error
        if _descriptor_identity(before) != _descriptor_identity(after):
            raise ConfigurationError(
                f"Matched preflight input changed while it was read: {candidate}",
                component="inkling_matched_preflight",
            )
        if len(content) != before.st_size:
            raise ConfigurationError(
                f"Matched preflight input ended before its declared size: {candidate}",
                component="inkling_matched_preflight",
                details={
                    "declared_size_bytes": before.st_size,
                    "read_size_bytes": len(content),
                },
            )
        report = MatchedLocalControlFile.model_validate(
            {
                "role": role,
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
                "regular_file": True,
                "symlink": False,
                "contained_in_project": True,
            }
        )
        return _CapturedControlFile(report=report, content=content)
    finally:
        suppress_close_errors = sys.exc_info()[0] is not None
        descriptors = (((descriptor, candidate),) if descriptor is not None else ()) + (
            (parent_descriptor, candidate.parent),
        )
        _close_descriptors(
            descriptors,
            suppress_errors=suppress_close_errors,
        )


def _capture_controls(
    project_root: str | Path,
) -> tuple[Path, tuple[_CapturedControlFile, ...]]:
    root, root_descriptor = _open_project_root(project_root)
    try:
        captures = tuple(
            _capture_control_file(
                root,
                root_descriptor,
                role=role,
                relative_path=relative_path,
            )
            for role, relative_path in _CONTROL_FILE_ROLES
        )
        try:
            opened = os.fstat(root_descriptor)
            named = os.stat(root, follow_symlinks=False)
        except OSError as error:
            raise _filesystem_error("reinspect project root", root, error) from error
        if _descriptor_identity(opened) != _descriptor_identity(named):
            raise ConfigurationError(
                f"Matched preflight project root changed during validation: {root}",
                component="inkling_matched_preflight",
            )
        return root, captures
    finally:
        _close_descriptors(
            ((root_descriptor, root),),
            suppress_errors=sys.exc_info()[0] is not None,
        )


def _parse_canonical_json_reference(
    raw_bytes: bytes,
    *,
    source: str,
    validator: Callable[[object], _ReferenceT],
    canonical_json: Callable[[_ReferenceT], str],
) -> _ReferenceT:
    try:
        raw = json.loads(raw_bytes)
        if not isinstance(raw, Mapping):
            raise ValueError("reference root must be a JSON object")
        reference = validator(raw)
    except (RecursionError, UnicodeError, ValueError, ValidationError) as error:
        raise ConfigurationError(
            f"Unable to parse matched preflight reference {source}: {error}",
            component="inkling_matched_preflight",
        ) from error
    expected_bytes = (canonical_json(reference) + "\n").encode("utf-8")
    if raw_bytes != expected_bytes:
        raise ConfigurationError(
            f"Matched preflight reference is not canonical JSON plus one newline: {source}",
            component="inkling_matched_preflight",
        )
    return reference


def _control_identities(
    controls: tuple[MatchedLocalControlFile, ...],
) -> tuple[tuple[str, str, str, int], ...]:
    return tuple((item.role, item.path, item.sha256, item.size_bytes) for item in controls)


def _validate_control_config_binding(
    controls: tuple[MatchedLocalControlFile, ...],
    config: InklingMatchedCellConfig,
) -> None:
    """Bind each checked control path and patch hash to the embedded config."""

    expected_paths = (
        ("matched_config", MATCHED_CELL_CONFIG_RELATIVE_PATH),
        ("bf16_reference", config.bf16_subject_reference_path),
        ("q3_reference", config.q3_verified_export_reference_path),
        ("source_reference", config.source_adoption_reference_path),
        ("instrumentation_patch", config.runtime.instrumentation_patch_path),
    )
    observed_paths = tuple((item.role, item.path) for item in controls)
    if observed_paths != expected_paths:
        raise ConfigurationError(
            "Matched preflight control paths differ from the embedded configuration",
            component="inkling_matched_preflight",
        )

    instrumentation = controls[-1]
    if (
        instrumentation.role != "instrumentation_patch"
        or instrumentation.sha256 != config.runtime.instrumentation_patch_sha256
    ):
        raise ConfigurationError(
            "Matched instrumentation patch control SHA-256 differs from the embedded runtime "
            "configuration",
            component="inkling_matched_preflight",
        )


def _mounts(bundle: InklingMatchedCellBundle) -> tuple[MatchedMountPlan, ...]:
    storage = bundle.config.storage
    mounts = (
        MatchedMountPlan(
            name="bf16",
            volume=storage.bf16_volume,
            volume_version=storage.bf16_volume_version,
            sub_path=storage.bf16_run_subpath,
            mount_path=storage.bf16_mount_path,
            read_only=storage.bf16_read_only,
            create_if_missing=storage.bf16_create_if_missing,
        ),
        MatchedMountPlan(
            name="q3",
            volume=storage.final_volume,
            volume_version=storage.final_volume_version,
            sub_path=storage.final_run_subpath,
            mount_path=storage.final_mount_path,
            read_only=storage.final_read_only,
            create_if_missing=storage.final_create_if_missing,
        ),
        MatchedMountPlan(
            name="source",
            volume=storage.source_volume,
            volume_version=storage.source_volume_version,
            sub_path=storage.source_run_subpath,
            mount_path=storage.source_mount_path,
            read_only=storage.source_read_only,
            create_if_missing=storage.source_create_if_missing,
        ),
        MatchedMountPlan(
            name="evidence",
            volume=storage.evidence_volume,
            volume_version=storage.evidence_volume_version,
            sub_path=None,
            mount_path=storage.evidence_mount_path,
            read_only=storage.evidence_read_only,
            create_if_missing=storage.evidence_create_if_missing,
        ),
    )
    mount_paths = tuple(PurePosixPath(item.mount_path) for item in mounts)
    if any(not path.is_absolute() for path in mount_paths):
        raise ConfigurationError(
            "Matched preflight mount paths must be absolute",
            component="inkling_matched_preflight",
        )
    for index, left in enumerate(mount_paths):
        for right in mount_paths[index + 1 :]:
            if left == right or left.is_relative_to(right) or right.is_relative_to(left):
                raise ConfigurationError(
                    "Matched preflight mount paths overlap",
                    component="inkling_matched_preflight",
                )
    return mounts


def _artifact_record(
    *,
    path: str,
    sha256: str,
    size_bytes: int,
) -> dict[str, str | int]:
    return {"path": path, "sha256": sha256, "size_bytes": size_bytes}


def _inventory_assignments(
    bundle: InklingMatchedCellBundle,
) -> tuple[MatchedInventoryAssignment, ...]:
    paths = bundle.paths
    bf16_records = (
        *(
            _artifact_record(path=path, sha256=item.sha256, size_bytes=item.size_bytes)
            for path, item in zip(
                paths.bf16_shards,
                bundle.bf16.bf16_shards,
                strict=True,
            )
        ),
        _artifact_record(
            path=paths.bf16_conversion_receipt,
            sha256=bundle.bf16.conversion_receipt.sha256,
            size_bytes=bundle.bf16.conversion_receipt.size_bytes,
        ),
    )
    q3_pairs = (
        (paths.q3_export_manifest, bundle.q3.export_manifest),
        (paths.q3_verify_receipt, bundle.q3.verify_receipt),
        (paths.q3_quantize_receipt, bundle.q3.quantize_receipt),
        (paths.projector_conversion_receipt, bundle.q3.mmproj_receipt),
    )
    q3_records = (
        *(
            _artifact_record(path=path, sha256=item.sha256, size_bytes=item.size_bytes)
            for path, item in zip(
                paths.q3_shards,
                bundle.q3.q3_shards,
                strict=True,
            )
        ),
        _artifact_record(
            path=paths.shared_projector,
            sha256=bundle.q3.projector.sha256,
            size_bytes=bundle.q3.projector.size_bytes,
        ),
        *(
            _artifact_record(path=path, sha256=item.sha256, size_bytes=item.size_bytes)
            for path, item in q3_pairs
        ),
        *(
            _artifact_record(path=path, sha256=item.sha256, size_bytes=item.size_bytes)
            for path, item in zip(
                paths.tokenizer_assets,
                bundle.config.tokenizer_assets,
                strict=True,
            )
        ),
    )
    all_paths = tuple(str(item["path"]) for item in (*bf16_records, *q3_records))
    if len(all_paths) != 110 or len(set(all_paths)) != len(all_paths):
        raise ConfigurationError(
            "Matched preflight remote artifact assignment is incomplete or duplicated",
            component="inkling_matched_preflight",
            details={
                "expected_artifact_count": 110,
                "observed_artifact_count": len(all_paths),
                "unique_artifact_count": len(set(all_paths)),
            },
        )
    storage = bundle.config.storage
    forbidden_prefixes = (
        f"{storage.bf16_mount_path}/{storage.bf16_run_subpath}/",
        f"{storage.final_mount_path}/{storage.final_run_subpath}/",
        f"{storage.source_mount_path}/{storage.source_run_subpath}/",
    )
    if any(path.startswith(forbidden_prefixes) for path in all_paths):
        raise ConfigurationError(
            "Matched preflight path repeats a mounted volume subpath",
            component="inkling_matched_preflight",
        )
    return tuple(
        MatchedInventoryAssignment(
            stage=stage,
            path=str(item["path"]),
            sha256=str(item["sha256"]),
            size_bytes=int(item["size_bytes"]),
            declared_only=True,
            rehash_performed=False,
        )
        for stage, records in (
            ("rehash_bf16_subject", bf16_records),
            ("rehash_q3_subject", q3_records),
        )
        for item in records
    )


def _subject_declarations(
    bundle: InklingMatchedCellBundle,
) -> tuple[MatchedSubjectDeclaration, MatchedSubjectDeclaration]:
    return (
        MatchedSubjectDeclaration(
            role="bf16",
            reference_sha256=bundle.bf16.reference_sha256,
            format="BF16",
            shard_count=bundle.bf16.bf16_shard_count,
            shard_total_bytes=bundle.bf16.bf16_total_bytes,
            shard_inventory_sha256=bundle.bf16.bf16_inventory_sha256,
            external_artifact_bytes_rehashed=False,
        ),
        MatchedSubjectDeclaration(
            role="q3",
            reference_sha256=bundle.q3.reference_sha256,
            format="Q3_K_M",
            shard_count=bundle.q3.q3_shard_count,
            shard_total_bytes=bundle.q3.q3_total_bytes,
            shard_inventory_sha256=bundle.q3.q3_inventory_sha256,
            external_artifact_bytes_rehashed=False,
        ),
    )


def _shared_projector_declaration(
    bundle: InklingMatchedCellBundle,
) -> MatchedSharedProjectorDeclaration:
    return MatchedSharedProjectorDeclaration(
        path=bundle.paths.shared_projector,
        sha256=bundle.q3.projector.sha256,
        size_bytes=bundle.q3.projector.size_bytes,
        reference_identity_matches=True,
        external_artifact_bytes_rehashed=False,
    )


def _declared_capacity(bundle: InklingMatchedCellBundle) -> MatchedDeclaredCapacity:
    resources = bundle.config.resources
    minimum_total = resources.gpu_count * resources.minimum_gpu_memory_bytes
    reserve_numerator = minimum_total * resources.capacity_reserve_basis_points
    headroom = (reserve_numerator + 9_999) // 10_000
    usable = minimum_total - headroom
    bf16_bytes = bundle.bf16.bf16_total_bytes + bundle.bf16.projector.size_bytes
    q3_bytes = bundle.q3.q3_total_bytes + bundle.q3.projector.size_bytes
    sequential_peak = max(bf16_bytes, q3_bytes)
    return MatchedDeclaredCapacity(
        gpu_count=resources.gpu_count,
        configured_minimum_gpu_memory_bytes=resources.minimum_gpu_memory_bytes,
        configured_minimum_total_gpu_memory_bytes=minimum_total,
        capacity_reserve_basis_points=resources.capacity_reserve_basis_points,
        declared_headroom_bytes=headroom,
        declared_usable_bytes=usable,
        bf16_subject_bytes=bf16_bytes,
        q3_subject_bytes=q3_bytes,
        sequential_peak_subject_bytes=sequential_peak,
        declared_remaining_bytes=usable - sequential_peak,
        hardware_probed=False,
        allocation_observed=False,
        capacity_screen_executed=False,
        runtime_fit_proven=False,
        limitation=CAPACITY_SCREEN_LIMITATION,
    )


def _declared_resource_cell(
    bundle: InklingMatchedCellBundle,
) -> MatchedDeclaredResourceCell:
    resources = bundle.config.resources
    return MatchedDeclaredResourceCell(
        provider=resources.provider,
        gpu_type=resources.gpu_type,
        gpu_count=resources.gpu_count,
        compute_capability=resources.compute_capability,
        minimum_gpu_memory_bytes=resources.minimum_gpu_memory_bytes,
        capacity_reserve_basis_points=resources.capacity_reserve_basis_points,
        capacity_strategy=resources.capacity_strategy,
        cpu_cores=resources.cpu_cores,
        memory_gib=resources.memory_gib,
        ephemeral_disk_mib=resources.ephemeral_disk_mib,
        startup_timeout_seconds=resources.startup_timeout_seconds,
        function_timeout_seconds=resources.function_timeout_seconds,
        max_attempts=resources.max_attempts,
        max_recovery_attempts=resources.max_recovery_attempts,
        declared_only=True,
    )


def _stages(bundle: InklingMatchedCellBundle) -> tuple[MatchedPreflightStage, ...]:
    return tuple(
        MatchedPreflightStage(
            ordinal=ordinal,
            name=name,
            status="passed" if ordinal == 0 else "not_executed",
            execution_performed=False,
            measurement_observed=False,
            external_artifact_bytes_rehashed=False,
            paid_compute_started=False,
        )
        for ordinal, name in enumerate(bundle.config.execution.planned_stages)
    )


def _tokenizer_declaration(bundle: InklingMatchedCellBundle) -> MatchedTokenizerDeclaration:
    artifacts = tuple(
        _artifact_record(
            path=path,
            sha256=item.sha256,
            size_bytes=item.size_bytes,
        )
        for path, item in zip(
            bundle.paths.tokenizer_assets,
            bundle.config.tokenizer_assets,
            strict=True,
        )
    )
    return MatchedTokenizerDeclaration(
        artifact_count=6,
        declared_total_bytes=sum(item.size_bytes for item in bundle.config.tokenizer_assets),
        declared_inventory_sha256=_domain_hash(
            MATCHED_PREFLIGHT_INVENTORY_HASH_DOMAIN,
            artifacts,
        ),
        external_artifact_bytes_rehashed=False,
    )


def _offline_facts() -> MatchedPreflightFacts:
    return MatchedPreflightFacts(
        local_control_files_verified=True,
        provider_contacted=False,
        remote_volume_inspected=False,
        network_access_performed=False,
        subprocess_execution_performed=False,
        local_write_performed=False,
        external_artifact_bytes_rehashed=False,
        artifact_rehash_performed=False,
        remote_runtime_binaries_verified=False,
        remote_execution_performed=False,
        hardware_probed=False,
        allocation_observed=False,
        capacity_screen_executed=False,
        bf16_smoke_executed=False,
        q3_smoke_executed=False,
        matched_smoke_verified=False,
        smoke_passed=False,
        measurement_execution_performed=False,
        measurement_ready=False,
        launch_authorized=False,
        paid_compute_started=False,
    )


def build_matched_preflight_report(
    project_root: str | Path,
    *,
    config_relative_path: str = MATCHED_CELL_CONFIG_RELATIVE_PATH,
) -> InklingMatchedPreflightReport:
    """Validate local controls and compile a launch-disabled matched plan."""

    if config_relative_path != MATCHED_CELL_CONFIG_RELATIVE_PATH:
        raise ConfigurationError(
            "Matched preflight accepts only the checked configuration path",
            component="inkling_matched_preflight",
        )
    _, captures = _capture_controls(project_root)
    controls = tuple(item.report for item in captures)
    observed_controls = _control_identities(controls)
    if observed_controls != _EXPECTED_CONTROL_IDENTITIES:
        mismatch_index = next(
            index
            for index, (observed, expected) in enumerate(
                zip(observed_controls, _EXPECTED_CONTROL_IDENTITIES, strict=True)
            )
            if observed != expected
        )
        role = _EXPECTED_CONTROL_IDENTITIES[mismatch_index][0].replace("_", " ")
        raise ConfigurationError(
            f"Matched {role} control file SHA-256 or size differs from the checked input",
            component="inkling_matched_preflight",
        )

    captured_by_role = {item.report.role: item for item in captures}
    config_capture = captured_by_role["matched_config"]
    bf16_capture = captured_by_role["bf16_reference"]
    q3_capture = captured_by_role["q3_reference"]
    source_capture = captured_by_role["source_reference"]

    config = parse_matched_cell_config_bytes(
        config_capture.content,
        source=config_capture.report.path,
    )
    _validate_control_config_binding(controls, config)
    bf16 = _parse_canonical_json_reference(
        bf16_capture.content,
        source=bf16_capture.report.path,
        validator=InklingBF16SubjectReference.model_validate,
        canonical_json=lambda value: value.canonical_json(),
    )
    q3 = _parse_canonical_json_reference(
        q3_capture.content,
        source=q3_capture.report.path,
        validator=InklingVerifiedExportReference.model_validate,
        canonical_json=lambda value: value.canonical_json(),
    )
    source = _parse_canonical_json_reference(
        source_capture.content,
        source=source_capture.report.path,
        validator=InklingSourceAdoptionReference.model_validate,
        canonical_json=lambda value: value.canonical_json(),
    )
    bundle = build_matched_cell_bundle(config, bf16, q3, source)

    payload: dict[str, Any] = {
        "schema_version": "inkling-matched-preflight-v1",
        "status": "ready_for_operator_review",
        "config_hash": bundle.config.config_hash(),
        "model_id": bundle.config.model_id,
        "revision": bundle.config.revision,
        "architecture": bundle.config.architecture,
        "control_files": tuple(item.model_dump(mode="json") for item in controls),
        "config": bundle.config.model_dump(mode="json"),
        "bf16_reference": bundle.bf16.model_dump(mode="json"),
        "q3_reference": bundle.q3.model_dump(mode="json"),
        "source_reference": bundle.source.model_dump(mode="json"),
        "subjects": tuple(item.model_dump(mode="json") for item in _subject_declarations(bundle)),
        "shared_projector": _shared_projector_declaration(bundle).model_dump(mode="json"),
        "tokenizer": _tokenizer_declaration(bundle).model_dump(mode="json"),
        "runtime": bundle.config.runtime.model_dump(mode="json"),
        "mounts": tuple(item.model_dump(mode="json") for item in _mounts(bundle)),
        "inventory_assignments": tuple(
            item.model_dump(mode="json") for item in _inventory_assignments(bundle)
        ),
        "stages": tuple(item.model_dump(mode="json") for item in _stages(bundle)),
        "declared_capacity": _declared_capacity(bundle).model_dump(mode="json"),
        "declared_resource_cell": _declared_resource_cell(bundle).model_dump(mode="json"),
        "execution": bundle.config.execution.model_dump(mode="json"),
        "claims": bundle.config.claims.model_dump(mode="json"),
        "facts": _offline_facts().model_dump(mode="json"),
        "warning": MATCHED_SCOPE_WARNING,
    }
    payload["plan_sha256"] = _domain_hash(MATCHED_PREFLIGHT_PLAN_HASH_DOMAIN, payload)
    try:
        return InklingMatchedPreflightReport.model_validate(payload)
    except ValidationError as error:
        raise ConfigurationError(
            f"Unable to validate the matched preflight report: {error}",
            component="inkling_matched_preflight",
        ) from error


__all__ = [
    "MATCHED_PREFLIGHT_INVENTORY_HASH_DOMAIN",
    "MATCHED_PREFLIGHT_PLAN_HASH_DOMAIN",
    "MATCHED_SCOPE_WARNING",
    "InklingMatchedPreflightReport",
    "MatchedDeclaredCapacity",
    "MatchedDeclaredResourceCell",
    "MatchedInventoryAssignment",
    "MatchedLocalControlFile",
    "MatchedMountPlan",
    "MatchedPreflightFacts",
    "MatchedPreflightStage",
    "MatchedSharedProjectorDeclaration",
    "MatchedSubjectDeclaration",
    "MatchedTokenizerDeclaration",
    "build_matched_preflight_report",
]
