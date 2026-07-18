"""Security primitives used by config, logging, loaders, and artifact storage."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from inkling_quant_lab.exceptions import ArtifactIntegrityError, ModelLoadError

_TOKEN_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"),
    re.compile(r"\bhf_[A-Za-z0-9]{8,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)
_SENSITIVE_CONFIG_KEYS = {
    "access_token",
    "api_key",
    "authorization",
    "auth_token",
    "bearer_token",
    "client_secret",
    "password",
    "secret",
    "token",
}
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(?:access[_-]?token|api[_-]?key|authorization|password|secret|token)"
    r"\s*[=:]\s*[^\s,;&]+"
)


def sensitive_literal_path(value: Any, path: tuple[str, ...] = ()) -> tuple[str, ...] | None:
    """Return the first path containing literal credential material, if any."""

    if isinstance(value, str):
        if _SENSITIVE_ASSIGNMENT.search(value) or any(
            pattern.search(value) for pattern in _TOKEN_PATTERNS
        ):
            return path
        return None
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            item_path = (*path, key_text)
            if key_text.lower() in _SENSITIVE_CONFIG_KEYS and isinstance(item, str) and item:
                return item_path
            found = sensitive_literal_path(item, item_path)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = sensitive_literal_path(item, (*path, str(index)))
            if found is not None:
                return found
    return None


class Redactor:
    """Redact known secrets and common authorization-token forms."""

    def __init__(self, secret_values: Sequence[str] = ()) -> None:
        self._secrets = tuple(
            sorted(
                {value for value in secret_values if value},
                key=lambda value: (-len(value), value),
            )
        )

    def text(self, value: str) -> str:
        """Return text with secret material replaced by ``[REDACTED]``."""

        redacted = value
        for secret in self._secrets:
            redacted = redacted.replace(secret, "[REDACTED]")
        for pattern in _TOKEN_PATTERNS:
            redacted = pattern.sub(
                lambda match: f"{match.group(1)}[REDACTED]" if match.lastindex else "[REDACTED]",
                redacted,
            )
        return redacted

    def value(self, value: Any) -> Any:
        """Recursively redact strings while preserving JSON-compatible shape."""

        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, Mapping):
            return {str(key): self.value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.value(item) for item in value)
        return value


def safe_path(root: Path, relative: str | Path, *, create_parent: bool = False) -> Path:
    """Resolve a relative artifact path and prove it remains below ``root``.

    Absolute paths, parent traversal, prefix-collision paths, and existing symlink
    escapes are rejected before any output file is opened.
    """

    root_resolved = root.expanduser().resolve()
    requested = Path(relative)
    if requested.is_absolute() or any(part == ".." for part in requested.parts):
        raise ArtifactIntegrityError(
            f"Artifact path must be relative and cannot contain '..': {relative}",
            component="artifact_store",
        )
    candidate = (root_resolved / requested).resolve(strict=False)
    if candidate == root_resolved or not candidate.is_relative_to(root_resolved):
        raise ArtifactIntegrityError(
            f"Artifact path escapes configured root: {relative}",
            component="artifact_store",
        )
    if create_parent:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        parent = candidate.parent.resolve()
        if not parent.is_relative_to(root_resolved):
            raise ArtifactIntegrityError(
                f"Artifact parent resolves outside configured root: {relative}",
                component="artifact_store",
            )
    return candidate


def validate_checkpoint_path(path: Path, *, allow_pickle: bool) -> None:
    """Reject known pickle-based model formats unless explicitly authorized."""

    unsafe_suffixes = {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
    if path.suffix.lower() in unsafe_suffixes and not allow_pickle:
        raise ModelLoadError(
            f"Pickle-based checkpoint is disabled: {path.name}",
            component="model_loader",
            remediation="Use safetensors or set security.allow_pickle_weights=true explicitly.",
        )


def run_command(
    args: Sequence[str], *, cwd: Path | None = None, timeout_seconds: float = 30.0
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess without shell interpolation."""

    if not args:
        raise ValueError("args must not be empty")
    return subprocess.run(
        list(args),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        shell=False,
    )
