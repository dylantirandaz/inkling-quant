"""Versioned run ledger and validated lifecycle transitions."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from inkling_quant_lab.artifacts import LocalArtifactStore
from inkling_quant_lab.exceptions import ArtifactIntegrityError


def utc_now() -> datetime:
    """Return an aware UTC timestamp."""

    return datetime.now(UTC)


class RunStatus(StrEnum):
    """Legal top-level run states."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class StageStatus(StrEnum):
    """Legal stage states, including optional and resume states."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED_NOT_REQUIRED = "skipped_not_required"
    UNSUPPORTED = "unsupported"
    INVALIDATED = "invalidated"


class ImmutableRecord(BaseModel):
    """Strict immutable record base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactChecksum(ImmutableRecord):
    """One immutable stage output."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class StageError(ImmutableRecord):
    """Actionable, redacted stage failure context."""

    code: str
    message: str
    component: str | None = None
    remediation: str | None = None
    details_path: str | None = None


class StageRecord(ImmutableRecord):
    """Lifecycle, fingerprint, and outputs for one pipeline stage."""

    name: str
    required: bool = True
    status: StageStatus = StageStatus.PENDING
    fingerprint: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    outputs: tuple[ArtifactChecksum, ...] = ()
    error: StageError | None = None
    reason: str | None = None
    attempt: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_state_fields(self) -> StageRecord:
        """Require timing and fingerprint evidence for completed work."""

        if self.status is StageStatus.SUCCESS and (
            self.started_at is None or self.completed_at is None or self.fingerprint is None
        ):
            raise ValueError("successful stage requires timestamps and fingerprint")
        if self.status is StageStatus.FAILED and self.error is None:
            raise ValueError("failed stage requires error context")
        if self.status is StageStatus.UNSUPPORTED and not self.reason:
            raise ValueError("unsupported stage requires a reason")
        return self


class GitProvenance(ImmutableRecord):
    """Git state where a repository is available."""

    commit: str | None = None
    dirty: bool | None = None


class ModelProvenance(ImmutableRecord):
    """Exact configured and resolved model identity."""

    id: str
    revision: str | None
    resolved_class: str | None = None
    architecture: str | None = None
    checksum: str | None = None


class RunManifest(ImmutableRecord):
    """Authoritative append-only run ledger."""

    schema_version: str = "1.0"
    run_id: str
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    git: GitProvenance = Field(default_factory=GitProvenance)
    model: ModelProvenance
    environment: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    status: RunStatus = RunStatus.PENDING
    warnings: tuple[str, ...] = ()

    def start(self, *, at: datetime | None = None) -> RunManifest:
        """Transition ``pending`` to ``running``."""

        if self.status is not RunStatus.PENDING:
            raise ArtifactIntegrityError(
                f"Invalid run transition {self.status} -> running", component="manifest"
            )
        return self.model_copy(update={"status": RunStatus.RUNNING, "started_at": at or utc_now()})

    def start_stage(
        self, name: str, fingerprint: str, *, at: datetime | None = None
    ) -> RunManifest:
        """Start a pending, failed, or invalidated stage while the run is active."""

        if self.status is not RunStatus.RUNNING:
            raise ArtifactIntegrityError("Stages can start only while a run is running")
        current = self.stages[name]
        if current.status not in {
            StageStatus.PENDING,
            StageStatus.FAILED,
            StageStatus.INVALIDATED,
        }:
            raise ArtifactIntegrityError(
                f"Invalid stage transition {current.status} -> running for {name}"
            )
        updated = current.model_copy(
            update={
                "status": StageStatus.RUNNING,
                "fingerprint": fingerprint,
                "started_at": at or utc_now(),
                "completed_at": None,
                "outputs": (),
                "error": None,
                "reason": None,
                "attempt": current.attempt + 1,
            }
        )
        return self._replace_stage(updated)

    def finish_stage(
        self,
        name: str,
        outputs: tuple[ArtifactChecksum, ...],
        *,
        at: datetime | None = None,
    ) -> RunManifest:
        """Transition a running stage to success."""

        current = self.stages[name]
        if current.status is not StageStatus.RUNNING:
            raise ArtifactIntegrityError(
                f"Invalid stage transition {current.status} -> success for {name}"
            )
        return self._replace_stage(
            current.model_copy(
                update={
                    "status": StageStatus.SUCCESS,
                    "completed_at": at or utc_now(),
                    "outputs": outputs,
                }
            )
        )

    def fail_stage(
        self, name: str, error: StageError, *, at: datetime | None = None
    ) -> RunManifest:
        """Transition a running stage to failed."""

        current = self.stages[name]
        if current.status is not StageStatus.RUNNING:
            raise ArtifactIntegrityError(
                f"Invalid stage transition {current.status} -> failed for {name}"
            )
        return self._replace_stage(
            current.model_copy(
                update={
                    "status": StageStatus.FAILED,
                    "completed_at": at or utc_now(),
                    "error": error,
                }
            )
        )

    def mark_stage(
        self, name: str, status: StageStatus, reason: str, *, at: datetime | None = None
    ) -> RunManifest:
        """Mark a pending optional stage skipped or unsupported."""

        if status not in {StageStatus.SKIPPED_NOT_REQUIRED, StageStatus.UNSUPPORTED}:
            raise ArtifactIntegrityError(f"mark_stage does not support status {status}")
        current = self.stages[name]
        if current.status not in {StageStatus.PENDING, StageStatus.RUNNING}:
            raise ArtifactIntegrityError(f"Cannot mark stage {name} from {current.status}")
        return self._replace_stage(
            current.model_copy(
                update={
                    "status": status,
                    "reason": reason,
                    "completed_at": at or utc_now(),
                }
            )
        )

    def invalidate(self, names: set[str]) -> RunManifest:
        """Invalidate selected successful or failed stages for forced execution."""

        stages = dict(self.stages)
        for name in names:
            current = stages[name]
            if current.status in {StageStatus.RUNNING, StageStatus.PENDING}:
                continue
            stages[name] = current.model_copy(
                update={
                    "status": StageStatus.INVALIDATED,
                    "completed_at": None,
                    "error": None,
                    "reason": "invalidated by forced stage execution",
                }
            )
        return self.model_copy(update={"stages": stages})

    def succeed(self, *, at: datetime | None = None) -> RunManifest:
        """Complete a run only when every required stage succeeded."""

        if self.status is not RunStatus.RUNNING:
            raise ArtifactIntegrityError(f"Invalid run transition {self.status} -> success")
        incomplete = [
            name
            for name, stage in self.stages.items()
            if stage.required and stage.status is not StageStatus.SUCCESS
        ]
        if incomplete:
            raise ArtifactIntegrityError(
                "Cannot complete run; required stages are not successful: " + ", ".join(incomplete),
                component="manifest",
            )
        return self.model_copy(
            update={"status": RunStatus.SUCCESS, "completed_at": at or utc_now()}
        )

    def fail(self, *, at: datetime | None = None) -> RunManifest:
        """Transition a running run to failed."""

        if self.status is not RunStatus.RUNNING:
            raise ArtifactIntegrityError(f"Invalid run transition {self.status} -> failed")
        return self.model_copy(update={"status": RunStatus.FAILED, "completed_at": at or utc_now()})

    def _replace_stage(self, stage: StageRecord) -> RunManifest:
        stages = dict(self.stages)
        stages[stage.name] = stage
        return self.model_copy(update={"stages": stages})


def persist_manifest(store: LocalArtifactStore, manifest: RunManifest) -> Path:
    """Atomically create or update the authoritative manifest ledger."""

    return store.write_json(
        "manifest.json",
        manifest.model_dump(mode="json"),
        replace=store.path("manifest.json").exists(),
    )


def load_manifest(run_directory: str | Path) -> RunManifest:
    """Load and validate a manifest from a run directory."""

    path = Path(run_directory) / "manifest.json"
    try:
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ArtifactIntegrityError(f"Unable to read manifest {path}: {error}") from error


def verify_outputs(run_directory: str | Path, stage: StageRecord) -> None:
    """Verify every recorded stage output checksum before resume."""

    from inkling_quant_lab.artifacts import sha256_file

    root = Path(run_directory).resolve()
    for output in stage.outputs:
        relative = Path(output.path)
        unresolved = root / relative
        has_symlink_component = False
        if not relative.is_absolute() and ".." not in relative.parts:
            current = root
            for part in relative.parts:
                current /= part
                if current.is_symlink():
                    has_symlink_component = True
                    break
        path = unresolved.resolve(strict=False)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or has_symlink_component
            or not path.is_relative_to(root)
            or not path.is_file()
        ):
            raise ArtifactIntegrityError(
                f"Missing or unsafe output for completed stage {stage.name}: {output.path}"
            )
        actual = sha256_file(path)
        if actual != output.sha256 or path.stat().st_size != output.size_bytes:
            raise ArtifactIntegrityError(
                f"Checksum mismatch for completed stage {stage.name}: {output.path}",
                component="resume",
                details={"expected": output.sha256, "actual": actual},
            )
