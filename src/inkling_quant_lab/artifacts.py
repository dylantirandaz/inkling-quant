"""Local append-only artifact store with atomic writes and checksums."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.security import safe_path


class ArtifactStore(Protocol):
    """Minimal store boundary used by the local pipeline."""

    @property
    def root(self) -> Path: ...

    def path(self, relative: str | Path) -> Path: ...

    def write_bytes(self, relative: str | Path, data: bytes, *, replace: bool = False) -> Path: ...

    def read_bytes(self, relative: str | Path) -> bytes: ...

    def checksum(self, relative: str | Path) -> str: ...


def sha256_bytes(data: bytes) -> str:
    """Hash bytes with SHA-256."""

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file into a SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class LocalArtifactStore:
    """Filesystem artifact store that rejects traversal and partial commits."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        """Return the normalized artifact root."""

        return self._root

    def path(self, relative: str | Path) -> Path:
        """Resolve a safe artifact path without creating it."""

        return safe_path(self._root, relative)

    def ensure_directory(self, relative: str | Path) -> Path:
        """Create a directory inside the store and return it."""

        directory = safe_path(self._root, relative)
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.resolve().is_relative_to(self._root):
            raise ArtifactIntegrityError(f"Directory escaped artifact root: {relative}")
        return directory

    def write_bytes(self, relative: str | Path, data: bytes, *, replace: bool = False) -> Path:
        """Atomically write a file, refusing replacement by default."""

        destination = safe_path(self._root, relative, create_parent=True)
        if destination.exists() and not replace:
            raise ArtifactIntegrityError(
                f"Refusing to overwrite immutable artifact: {relative}",
                component="artifact_store",
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if destination.exists() and not replace:
                raise ArtifactIntegrityError(
                    f"Refusing to overwrite immutable artifact: {relative}",
                    component="artifact_store",
                )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def write_text(
        self, relative: str | Path, text: str, *, replace: bool = False, append: bool = False
    ) -> Path:
        """Write UTF-8 text atomically, or append to an append-only event stream."""

        if append:
            destination = safe_path(self._root, relative, create_parent=True)
            with destination.open("a", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            return destination
        return self.write_bytes(relative, text.encode("utf-8"), replace=replace)

    def write_json(
        self, relative: str | Path, value: Any, *, replace: bool = False, indent: int = 2
    ) -> Path:
        """Write stable JSON with a trailing newline."""

        payload = json.dumps(value, sort_keys=True, indent=indent, ensure_ascii=False) + "\n"
        return self.write_text(relative, payload, replace=replace)

    def read_bytes(self, relative: str | Path) -> bytes:
        """Read bytes from a safe artifact path."""

        return self.path(relative).read_bytes()

    def read_json(self, relative: str | Path) -> Any:
        """Read a JSON artifact."""

        return json.loads(self.path(relative).read_text(encoding="utf-8"))

    def checksum(self, relative: str | Path) -> str:
        """Return the SHA-256 checksum of an existing artifact."""

        path = self.path(relative)
        if not path.is_file():
            raise ArtifactIntegrityError(f"Artifact is missing or not a file: {relative}")
        return sha256_file(path)

    @contextmanager
    def staged_directory(
        self,
        relative: str | Path,
        *,
        replace_empty_placeholder: bool = False,
    ) -> Iterator[Path]:
        """Yield a temporary stage directory and atomically publish it on success.

        A run may pre-create an empty category directory so failed runs still
        expose the documented topology. A stage that owns that entire category
        can explicitly replace only that empty placeholder; non-empty evidence
        is never overwritten.
        """

        destination = safe_path(self._root, relative)
        removed_placeholder = False
        if destination.exists():
            if (
                replace_empty_placeholder
                and destination.is_dir()
                and not any(destination.iterdir())
            ):
                destination.rmdir()
                removed_placeholder = True
            else:
                raise ArtifactIntegrityError(f"Stage output already exists: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        )
        committed = False
        try:
            yield temporary
            if destination.exists():
                raise ArtifactIntegrityError(f"Stage output appeared concurrently: {relative}")
            os.replace(temporary, destination)
            committed = True
        finally:
            if not committed and temporary.exists():
                for child in sorted(temporary.rglob("*"), reverse=True):
                    if child.is_file() or child.is_symlink():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        child.rmdir()
                temporary.rmdir()
            if removed_placeholder and not committed and not destination.exists():
                destination.mkdir(parents=True)
