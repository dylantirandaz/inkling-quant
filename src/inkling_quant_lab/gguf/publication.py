"""Crash-safe publication of immutable GGUF stage directories.

This module deliberately has no Modal dependency.  A worker first records an
immutable intent while its outputs still live in an attempt directory.  A later
worker can then verify and atomically rename that directory, or recover when the
rename happened before the success receipt was durable.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from inkling_quant_lab.artifacts import sha256_file
from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.manifests import ArtifactChecksum

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_STAGE_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
_COMPONENT = "gguf_publication"
ModelT = TypeVar("ModelT", bound=BaseModel)


class _ImmutableRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PublicationIntent(_ImmutableRecord):
    """Immutable, self-checking commitment to one attempt directory."""

    schema_version: Literal["inkling-gguf-publication-intent-v1"] = (
        "inkling-gguf-publication-intent-v1"
    )
    config_hash: str = Field(pattern=_SHA256_PATTERN)
    stage: str = Field(pattern=_STAGE_PATTERN)
    partial_directory: str = Field(min_length=1)
    canonical_directory: str = Field(min_length=1)
    outputs: tuple[ArtifactChecksum, ...] = Field(min_length=1)
    binding_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_binding(self) -> PublicationIntent:
        """Reject reordered inventories, duplicates, and intent field drift."""

        paths = tuple(output.path for output in self.outputs)
        if paths != tuple(sorted(paths)):
            raise ValueError("publication outputs must be ordered by path")
        if len(paths) != len(set(paths)):
            raise ValueError("publication output paths must be unique")
        if self.binding_sha256 != _intent_binding_sha256(
            config_hash=self.config_hash,
            stage=self.stage,
            partial_directory=self.partial_directory,
            canonical_directory=self.canonical_directory,
            outputs=self.outputs,
        ):
            raise ValueError("publication intent binding hash drifted")
        return self


class PublicationReceipt(_ImmutableRecord):
    """Durable proof that all recorded outputs live below the canonical directory."""

    schema_version: Literal["inkling-gguf-publication-receipt-v1"] = (
        "inkling-gguf-publication-receipt-v1"
    )
    status: Literal["success"] = "success"
    config_hash: str = Field(pattern=_SHA256_PATTERN)
    stage: str = Field(pattern=_STAGE_PATTERN)
    canonical_directory: str = Field(min_length=1)
    intent_sha256: str = Field(pattern=_SHA256_PATTERN)
    outputs: tuple[ArtifactChecksum, ...] = Field(min_length=1)


def publication_intent_path(run_root: Path, stage: str) -> Path:
    """Return the deterministic intent path after validating its stage binding."""

    root = _validated_root(run_root)
    _validate_stage(stage)
    return _control_directory(root, create=False) / f"{stage}.publication.intent.json"


def publication_receipt_path(run_root: Path, stage: str) -> Path:
    """Return the deterministic success-receipt path for a stage."""

    root = _validated_root(run_root)
    _validate_stage(stage)
    return _control_directory(root, create=False) / f"{stage}.publication.success.json"


def prepare_publication_intent(
    run_root: Path,
    *,
    config_hash: str,
    stage: str,
    partial_directory: Path,
    canonical_directory: Path,
    outputs: Sequence[ArtifactChecksum | Mapping[str, Any]],
) -> PublicationIntent:
    """Verify an attempt completely, then atomically create its immutable intent.

    Output paths are relative to ``partial_directory``.  The records must cover
    every regular file in that directory; unrecorded files and all symlinks or
    special files are rejected.
    """

    root = _validated_root(run_root)
    _validate_hash(config_hash, "config_hash")
    _validate_stage(stage)
    partial, partial_relative = _bound_directory(root, partial_directory)
    canonical, canonical_relative = _bound_directory(root, canonical_directory)
    _validate_directory_pair(
        partial,
        partial_relative,
        canonical,
        canonical_relative,
    )
    _require_directory(partial, "partial directory")
    _require_absent(canonical, "canonical directory")

    records = tuple(
        sorted((_coerce_output(value, stage) for value in outputs), key=lambda item: item.path)
    )
    if not records:
        raise _integrity_error("A publication intent requires at least one output", stage)
    _verify_complete_inventory(partial, records, stage)

    binding_sha256 = _intent_binding_sha256(
        config_hash=config_hash,
        stage=stage,
        partial_directory=partial_relative.as_posix(),
        canonical_directory=canonical_relative.as_posix(),
        outputs=records,
    )
    intent = PublicationIntent(
        config_hash=config_hash,
        stage=stage,
        partial_directory=partial_relative.as_posix(),
        canonical_directory=canonical_relative.as_posix(),
        outputs=records,
        binding_sha256=binding_sha256,
    )
    control = _control_directory(root, create=True)
    _write_immutable_json(control / f"{stage}.publication.intent.json", intent)
    return intent


def finalize_publication(
    run_root: Path,
    *,
    config_hash: str,
    stage: str,
    partial_directory: Path,
    canonical_directory: Path,
) -> PublicationReceipt:
    """Finish or recover one publication and return canonical output records.

    Exactly one bound directory must exist.  A partial directory is verified and
    atomically renamed; an already-canonical directory represents a crash after
    that rename and is verified in place.  The same deterministic receipt is
    returned on every successful invocation.
    """

    root = _validated_root(run_root)
    _validate_hash(config_hash, "config_hash")
    _validate_stage(stage)
    partial, partial_relative = _bound_directory(root, partial_directory)
    canonical, canonical_relative = _bound_directory(root, canonical_directory)
    _validate_directory_pair(
        partial,
        partial_relative,
        canonical,
        canonical_relative,
    )

    intent_path = publication_intent_path(root, stage)
    intent = _read_model(intent_path, PublicationIntent, stage)
    _require_canonical_encoding(intent_path, intent, stage)
    if intent.config_hash != config_hash:
        raise _integrity_error("Publication intent config hash drifted", stage)
    if intent.stage != stage:
        raise _integrity_error("Publication intent stage binding drifted", stage)
    if intent.partial_directory != partial_relative.as_posix():
        raise _integrity_error("Publication intent partial path binding drifted", stage)
    if intent.canonical_directory != canonical_relative.as_posix():
        raise _integrity_error("Publication intent canonical path binding drifted", stage)

    partial_exists = _directory_exists_or_raise(partial, "partial directory", stage)
    canonical_exists = _directory_exists_or_raise(canonical, "canonical directory", stage)
    if partial_exists == canonical_exists:
        state = "both exist" if partial_exists else "neither exists"
        raise _integrity_error(
            f"Invalid publication state: {state}; exactly one directory is required",
            stage,
        )

    active = partial if partial_exists else canonical
    _verify_complete_inventory(active, intent.outputs, stage)
    if partial_exists:
        # Refuse any observed canonical object.  os.rename provides the atomic
        # directory publication after this no-overwrite precondition.
        _require_absent(canonical, "canonical directory")
        os.rename(partial, canonical)
        if partial.exists() or partial.is_symlink():
            raise _integrity_error("Partial directory remained after atomic rename", stage)
        _require_directory(canonical, "canonical directory")
        _verify_complete_inventory(canonical, intent.outputs, stage)

    canonical_outputs = tuple(
        ArtifactChecksum(
            path=(canonical_relative / output.path).as_posix(),
            sha256=output.sha256,
            size_bytes=output.size_bytes,
        )
        for output in intent.outputs
    )
    receipt = PublicationReceipt(
        config_hash=config_hash,
        stage=stage,
        canonical_directory=canonical_relative.as_posix(),
        intent_sha256=sha256_file(intent_path),
        outputs=canonical_outputs,
    )
    receipt_path = publication_receipt_path(root, stage)
    _write_immutable_json(receipt_path, receipt)
    return receipt


def _intent_binding_sha256(
    *,
    config_hash: str,
    stage: str,
    partial_directory: str,
    canonical_directory: str,
    outputs: Sequence[ArtifactChecksum],
) -> str:
    payload = {
        "schema": "inkling-gguf-publication-binding-v1",
        "config_hash": config_hash,
        "stage": stage,
        "partial_directory": partial_directory,
        "canonical_directory": canonical_directory,
        "outputs": [output.model_dump(mode="json") for output in outputs],
    }
    return hashlib.sha256(_canonical_json_bytes(payload, trailing_newline=False)).hexdigest()


def _coerce_output(value: ArtifactChecksum | Mapping[str, Any], stage: str) -> ArtifactChecksum:
    try:
        if isinstance(value, ArtifactChecksum):
            return value
        return ArtifactChecksum.model_validate(value)
    except ValidationError as error:
        raise _integrity_error("Publication output record is malformed", stage) from error


def _validated_root(run_root: Path) -> Path:
    requested = Path(run_root).expanduser()
    if requested.is_symlink() or not requested.is_dir():
        raise _integrity_error(f"Run root is not a regular directory: {run_root}")
    return requested.resolve(strict=True)


def _validate_hash(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise _integrity_error(f"{label} must be a lowercase SHA-256 digest")


def _validate_stage(stage: str) -> None:
    if re.fullmatch(_STAGE_PATTERN, stage) is None:
        raise _integrity_error(f"Unsafe publication stage name: {stage!r}")


def _bound_directory(root: Path, requested: Path) -> tuple[Path, Path]:
    raw = Path(requested).expanduser()
    if ".." in raw.parts:
        raise _integrity_error(f"Publication path contains parent traversal: {requested}")
    candidate = raw if raw.is_absolute() else root / raw
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise _integrity_error(f"Publication path escapes run root: {requested}") from error
    if not relative.parts or relative == Path("."):
        raise _integrity_error("Publication directory cannot be the run root")
    _reject_symlink_components(root, relative)
    resolved = candidate.resolve(strict=False)
    if resolved == root or not resolved.is_relative_to(root):
        raise _integrity_error(f"Publication path escapes run root: {requested}")
    return resolved, resolved.relative_to(root)


def _validate_directory_pair(
    partial: Path,
    partial_relative: Path,
    canonical: Path,
    canonical_relative: Path,
) -> None:
    if partial == canonical:
        raise _integrity_error("Partial and canonical publication directories must differ")
    if partial.parent != canonical.parent:
        raise _integrity_error("Partial and canonical directories must be atomic-rename siblings")
    if partial_relative.parts[0] == "control" or canonical_relative.parts[0] == "control":
        raise _integrity_error("Published stage directories cannot reside under control")


def _reject_symlink_components(root: Path, relative: Path) -> None:
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise _integrity_error(f"Publication path contains a symlink: {current}")
        if not current.exists():
            return


def _require_directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise _integrity_error(f"Missing {label}: {path}") from error
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise _integrity_error(f"Unsafe {label}: {path}")


def _require_absent(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise _integrity_error(f"Refusing to replace existing {label}: {path}")


def _directory_exists_or_raise(path: Path, label: str, stage: str) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise _integrity_error(f"Unsafe {label}: {path}", stage)
    return True


def _validate_output_relative_path(relative_text: str, stage: str) -> Path:
    relative = Path(relative_text)
    if (
        relative.is_absolute()
        or not relative.parts
        or relative == Path(".")
        or ".." in relative.parts
    ):
        raise _integrity_error(f"Unsafe publication output path: {relative_text}", stage)
    return relative


def _verify_complete_inventory(
    directory: Path, records: Sequence[ArtifactChecksum], stage: str
) -> None:
    _require_directory(directory, "publication directory")
    expected: dict[str, ArtifactChecksum] = {}
    for record in records:
        record_relative = _validate_output_relative_path(record.path, stage)
        normalized = record_relative.as_posix()
        if normalized != record.path:
            raise _integrity_error(f"Output path is not normalized: {record.path}", stage)
        if normalized in expected:
            raise _integrity_error(f"Duplicate publication output: {normalized}", stage)
        expected[normalized] = record

    observed: set[str] = set()
    for candidate in sorted(directory.rglob("*")):
        observed_relative = candidate.relative_to(directory).as_posix()
        mode = candidate.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise _integrity_error(
                f"Publication tree contains a symlink: {observed_relative}", stage
            )
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise _integrity_error(
                f"Publication tree contains a special file: {observed_relative}", stage
            )
        observed.add(observed_relative)

    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        unrecorded = sorted(observed - set(expected))
        raise _integrity_error(
            f"Publication inventory differs (missing={missing}, unrecorded={unrecorded})",
            stage,
        )
    for expected_relative, record in expected.items():
        _verify_regular_file(directory / expected_relative, record, stage)


def _verify_regular_file(path: Path, record: ArtifactChecksum, stage: str) -> None:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise _integrity_error(f"Recorded output is not a regular file: {record.path}", stage)
    if before.st_size != record.size_bytes:
        raise _integrity_error(f"Recorded output size drifted: {record.path}", stage)
    observed_hash = sha256_file(path)
    after = path.lstat()
    stable_fields = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    if stable_fields != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise _integrity_error(f"Recorded output changed while hashing: {record.path}", stage)
    if observed_hash != record.sha256:
        raise _integrity_error(f"Recorded output checksum drifted: {record.path}", stage)


def _control_directory(root: Path, *, create: bool) -> Path:
    control = root / "control"
    if create:
        with suppress(FileExistsError):
            control.mkdir()
    if control.exists() or control.is_symlink():
        _require_directory(control, "publication control directory")
    return control


def _canonical_json_bytes(value: Any, *, trailing_newline: bool = True) -> bytes:
    suffix = "\n" if trailing_newline else ""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + suffix
    ).encode("utf-8")


def _model_bytes(model: BaseModel) -> bytes:
    return _canonical_json_bytes(model.model_dump(mode="json"))


def _write_immutable_json(path: Path, model: BaseModel) -> None:
    payload = _model_bytes(model)
    if path.exists() or path.is_symlink():
        _require_same_immutable_file(path, payload)
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
        # Modal Volume v1 documents atomic rename but not hard links.  The
        # deployed workflow serializes each stage (one container, one sealed
        # Function ID), so an absent-path check followed by same-directory
        # rename is the portable crash-safe publication primitive here.
        if path.exists() or path.is_symlink():
            _require_same_immutable_file(path, payload)
        else:
            os.rename(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_same_immutable_file(path: Path, expected: bytes) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise _integrity_error(f"Immutable publication file disappeared: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise _integrity_error(f"Immutable publication path is unsafe: {path}")
    if path.read_bytes() != expected:
        raise _integrity_error(f"Refusing to overwrite immutable publication file: {path}")


def _read_model(path: Path, model_type: type[ModelT], stage: str) -> ModelT:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise _integrity_error(f"Missing publication intent: {path}", stage) from error
    if path.is_symlink() or not stat.S_ISREG(mode):
        raise _integrity_error(f"Publication metadata is not a regular file: {path}", stage)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return model_type.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise _integrity_error(f"Publication metadata is malformed: {path}", stage) from error


def _require_canonical_encoding(path: Path, model: BaseModel, stage: str) -> None:
    if path.read_bytes() != _model_bytes(model):
        raise _integrity_error(f"Publication metadata encoding drifted: {path}", stage)


def _integrity_error(message: str, stage: str | None = None) -> ArtifactIntegrityError:
    return ArtifactIntegrityError(message, stage=stage, component=_COMPONENT)


__all__ = [
    "PublicationIntent",
    "PublicationReceipt",
    "finalize_publication",
    "prepare_publication_intent",
    "publication_intent_path",
    "publication_receipt_path",
]
