"""Integrity-checked resume planning and append-only forced-stage archival."""

from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError

from inkling_quant_lab.artifacts import LocalArtifactStore, sha256_file
from inkling_quant_lab.config import ExperimentConfig, load_config
from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.manifests import (
    ArtifactChecksum,
    GitProvenance,
    ModelProvenance,
    RunManifest,
    RunStatus,
    StageRecord,
    StageStatus,
    utc_now,
    verify_outputs,
)
from inkling_quant_lab.pipeline.stages import (
    STAGE_ORDER,
    STAGE_OUTPUT_LOCATIONS,
    StageName,
    record_stage_result,
    stage_descendants,
    stage_fingerprint_from_manifest,
)


class _ImmutableResumeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArchivedStage(_ImmutableResumeRecord):
    """Durable mapping from prior outputs to their forced-run archive paths."""

    name: StageName
    previous_status: StageStatus
    previous_fingerprint: str | None
    original_outputs: tuple[ArtifactChecksum, ...]
    archived_outputs: tuple[ArtifactChecksum, ...]


class ForcedArchiveLedger(_ImmutableResumeRecord):
    """Durable two-phase evidence for one forced-stage archive transaction."""

    schema_version: str = "1.0"
    archive_id: str
    forced_stage: StageName
    state: Literal[
        "preparing",
        "outputs_archived",
        "manifest_committed",
        "recovered",
        "rolled_back",
    ]
    created_at: datetime
    stages: tuple[ArchivedStage, ...]


class ResumePreparation(_ImmutableResumeRecord):
    """Copy-on-write manifest plus deterministic skip/restart decisions."""

    manifest: RunManifest
    skip_stages: tuple[StageName, ...]
    restart_stages: tuple[StageName, ...]
    invalidated_stages: tuple[StageName, ...] = ()
    archived_stages: tuple[ArchivedStage, ...] = ()
    forced_archive_ledger: str | None = None


@dataclass(frozen=True)
class _ForcedArchiveRecoveryPlan:
    """A fully validated, still read-only forced-archive recovery decision."""

    ledger_path: Path
    ledger: ForcedArchiveLedger
    target_state: Literal["manifest_committed", "recovered"]
    restorations: tuple[tuple[ArtifactChecksum, ArtifactChecksum], ...]
    archived_fallbacks: tuple[tuple[ArtifactChecksum, ArtifactChecksum], ...]


def preflight_resume_integrity(
    run_directory: str | Path,
    manifest: RunManifest,
    config: ExperimentConfig,
) -> None:
    """Validate every resume input before archive recovery or ledger mutation.

    An interrupted forced-stage transaction can legitimately leave canonical
    outputs absent while the old manifest still records them as successful.
    In that narrow case, the pending archive ledger is read-only recovery
    evidence. Every archive transaction is validated first, then successful
    outputs and fingerprints are checked against canonical files or that
    verified fallback.
    """

    if manifest.config_hash != config.config_hash():
        raise ArtifactIntegrityError(
            "resolved configuration hash does not match the resume target",
            component="resume",
            details={"expected": manifest.config_hash, "actual": config.config_hash()},
        )
    _verify_manifest_stage_set(manifest)
    plans = _plan_forced_archive_recovery(run_directory, manifest)
    archived_fallbacks = {
        original.path: archived for plan in plans for original, archived in plan.archived_fallbacks
    }
    _verify_successful_stage_outputs_with_fallback(
        run_directory,
        manifest,
        archived_fallbacks=archived_fallbacks,
    )
    verify_successful_stage_fingerprints(config, manifest)


def recover_forced_archive_transactions(
    run_directory: str | Path,
    manifest: RunManifest,
) -> tuple[str, ...]:
    """Restore outputs moved before their invalidated manifest was committed.

    Forced execution is a two-file-system-object transaction: prior canonical
    outputs move first, then the run manifest records invalidation. A durable
    archive ledger lets a later resume restore byte-identical outputs when the
    old manifest is still authoritative after an abrupt termination.
    """

    store = LocalArtifactStore(run_directory)
    plans = _plan_forced_archive_recovery(store.root, manifest)
    recovered: list[str] = []
    for plan in plans:
        for original, archived in plan.restorations:
            store.write_bytes(original.path, store.read_bytes(archived.path))
            recovered.append(original.path)

    # Do not advance any transaction ledger until every pending transaction has
    # been validated and every required restoration has completed.
    for plan in plans:
        store.write_json(
            plan.ledger_path.relative_to(store.root),
            plan.ledger.model_copy(update={"state": plan.target_state}).model_dump(mode="json"),
            replace=True,
        )
    return tuple(recovered)


def _plan_forced_archive_recovery(
    run_directory: str | Path,
    manifest: RunManifest,
) -> tuple[_ForcedArchiveRecoveryPlan, ...]:
    """Validate all pending archive transactions without changing the store."""

    store = LocalArtifactStore(run_directory)
    archive_root = store.root / "archive" / "forced"
    if not archive_root.is_dir():
        return ()

    plans: list[_ForcedArchiveRecoveryPlan] = []
    claimed_original_paths: set[str] = set()
    for ledger_path in sorted(archive_root.glob("*/archive_manifest.json")):
        try:
            ledger = ForcedArchiveLedger.model_validate_json(
                ledger_path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise ArtifactIntegrityError(
                f"Unable to validate forced-stage archive ledger {ledger_path}: {error}",
                component="resume",
            ) from error
        if ledger_path.parent.name != ledger.archive_id:
            raise ArtifactIntegrityError(
                f"Forced archive ledger path does not match archive_id {ledger.archive_id!r}",
                component="resume",
            )
        if ledger.state in {"manifest_committed", "recovered", "rolled_back"}:
            continue
        if not ledger.stages:
            raise ArtifactIntegrityError(
                f"Forced archive ledger {ledger_path} records no stages",
                component="resume",
            )

        stage_names = tuple(stage.name for stage in ledger.stages)
        if len(set(stage_names)) != len(stage_names):
            raise ArtifactIntegrityError(
                f"Forced archive ledger {ledger_path} repeats a stage",
                component="resume",
            )
        claims = tuple(_manifest_still_claims_stage(manifest, stage) for stage in ledger.stages)
        if any(claims) and not all(claims):
            raise ArtifactIntegrityError(
                f"Forced archive ledger {ledger_path} only partially matches the manifest",
                component="resume",
            )
        manifest_still_claims_archive = all(claims)

        restorations: list[tuple[ArtifactChecksum, ArtifactChecksum]] = []
        fallbacks: list[tuple[ArtifactChecksum, ArtifactChecksum]] = []
        ledger_original_paths: set[str] = set()
        for archived_stage in ledger.stages:
            if len(archived_stage.original_outputs) != len(archived_stage.archived_outputs):
                raise ArtifactIntegrityError(
                    f"Forced archive ledger has mismatched output lists for {archived_stage.name}",
                    component="resume",
                )
            for original, archived in zip(
                archived_stage.original_outputs,
                archived_stage.archived_outputs,
                strict=True,
            ):
                if original.path in ledger_original_paths:
                    raise ArtifactIntegrityError(
                        f"Forced archive ledger repeats original output {original.path}",
                        component="resume",
                    )
                ledger_original_paths.add(original.path)
                if original.sha256 != archived.sha256 or original.size_bytes != archived.size_bytes:
                    raise ArtifactIntegrityError(
                        f"Forced archive ledger checksum metadata differs for {original.path}",
                        component="resume",
                    )
                expected_archived = (
                    Path("archive")
                    / "forced"
                    / ledger.archive_id
                    / archived_stage.name
                    / original.path
                ).as_posix()
                if archived.path != expected_archived:
                    raise ArtifactIntegrityError(
                        f"Forced archive ledger has an unexpected destination for {original.path}",
                        component="resume",
                    )

                canonical_path = store.path(original.path)
                archived_path = store.path(archived.path)
                canonical_exists = canonical_path.exists()
                archived_exists = archived_path.exists()
                if canonical_exists:
                    _verify_recovery_file(
                        canonical_path,
                        original,
                        message=(
                            f"Cannot recover forced output {original.path}; canonical copy is "
                            "not a regular file or is corrupt"
                        ),
                    )
                if archived_exists:
                    _verify_recovery_file(
                        archived_path,
                        archived,
                        message=(
                            f"Cannot recover forced output {original.path}; archive copy is "
                            "missing or corrupt"
                        ),
                    )

                archive_required = (
                    ledger.state == "outputs_archived" or not manifest_still_claims_archive
                )
                if archive_required and not archived_exists:
                    raise ArtifactIntegrityError(
                        f"Cannot recover forced output {original.path}; archive copy is missing "
                        "or corrupt",
                        component="resume",
                    )
                if manifest_still_claims_archive and not canonical_exists and not archived_exists:
                    raise ArtifactIntegrityError(
                        f"Cannot recover forced output {original.path}; archive copy is missing "
                        "or corrupt",
                        component="resume",
                    )
                if manifest_still_claims_archive and not canonical_exists:
                    restorations.append((original, archived))
                    fallbacks.append((original, archived))

        if manifest_still_claims_archive:
            overlap = claimed_original_paths.intersection(ledger_original_paths)
            if overlap:
                raise ArtifactIntegrityError(
                    "Multiple pending forced archives claim the same outputs: "
                    + ", ".join(sorted(overlap)),
                    component="resume",
                )
            claimed_original_paths.update(ledger_original_paths)
        plans.append(
            _ForcedArchiveRecoveryPlan(
                ledger_path=ledger_path,
                ledger=ledger,
                target_state=(
                    "recovered" if manifest_still_claims_archive else "manifest_committed"
                ),
                restorations=tuple(restorations),
                archived_fallbacks=tuple(fallbacks),
            )
        )
    return tuple(plans)


def _manifest_still_claims_stage(manifest: RunManifest, archived_stage: ArchivedStage) -> bool:
    current = manifest.stages.get(archived_stage.name)
    return current is not None and (
        current.status is archived_stage.previous_status
        and current.fingerprint == archived_stage.previous_fingerprint
        and current.outputs == archived_stage.original_outputs
    )


def _verify_recovery_file(
    path: Path,
    expected: ArtifactChecksum,
    *,
    message: str,
) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != expected.size_bytes
        or sha256_file(path) != expected.sha256
    ):
        raise ArtifactIntegrityError(message, component="resume")


def _verify_successful_stage_outputs_with_fallback(
    run_directory: str | Path,
    manifest: RunManifest,
    *,
    archived_fallbacks: dict[str, ArtifactChecksum],
) -> None:
    store = LocalArtifactStore(run_directory)
    for name in sorted(manifest.stages):
        stage = manifest.stages[name]
        if stage.status is not StageStatus.SUCCESS:
            continue
        for output in stage.outputs:
            canonical = store.path(output.path)
            if canonical.is_file():
                actual = sha256_file(canonical)
                if actual != output.sha256 or canonical.stat().st_size != output.size_bytes:
                    raise ArtifactIntegrityError(
                        f"Checksum mismatch for completed stage {stage.name}: {output.path}",
                        component="resume",
                        details={"expected": output.sha256, "actual": actual},
                    )
                continue
            archived = archived_fallbacks.get(output.path)
            if archived is None:
                raise ArtifactIntegrityError(
                    f"Missing or unsafe output for completed stage {stage.name}: {output.path}"
                )
            archived_path = store.path(archived.path)
            _verify_recovery_file(
                archived_path,
                output,
                message=(
                    f"Cannot recover forced output {output.path}; archive copy is missing "
                    "or corrupt"
                ),
            )


def load_resume_config(run_directory: str | Path, manifest: RunManifest) -> ExperimentConfig:
    """Load the canonical resolved config or its most recent forced-archive copy."""

    store = LocalArtifactStore(run_directory)
    canonical = store.root / "resolved_config.yaml"
    candidates = [canonical]
    archive_root = store.root / "archive" / "forced"
    if archive_root.is_dir():
        for ledger_path in sorted(archive_root.glob("*/archive_manifest.json"), reverse=True):
            try:
                ledger = ForcedArchiveLedger.model_validate_json(
                    ledger_path.read_text(encoding="utf-8")
                )
            except (OSError, ValidationError) as error:
                raise ArtifactIntegrityError(
                    f"Unable to validate forced-stage archive ledger {ledger_path}: {error}",
                    component="resume",
                ) from error
            for stage in ledger.stages:
                if len(stage.original_outputs) != len(stage.archived_outputs):
                    raise ArtifactIntegrityError(
                        f"Forced archive ledger has mismatched output lists for {stage.name}",
                        component="resume",
                    )
                for original, archived in zip(
                    stage.original_outputs,
                    stage.archived_outputs,
                    strict=True,
                ):
                    if original.path == "resolved_config.yaml":
                        candidates.append(store.path(archived.path))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        config = load_config(candidate)
        if config.config_hash() != manifest.config_hash:
            raise ArtifactIntegrityError(
                f"Resolved configuration hash does not match manifest for {candidate}",
                component="resume",
            )
        return config
    raise ArtifactIntegrityError(
        f"Unable to locate resolved_config.yaml for run {store.root}",
        component="resume",
    )


def mark_forced_archive_committed(run_directory: str | Path, relative_path: str) -> None:
    """Finish a forced-archive transaction after invalidation is durable."""

    store = LocalArtifactStore(run_directory)
    path = store.path(relative_path)
    try:
        ledger = ForcedArchiveLedger.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ArtifactIntegrityError(
            f"Unable to validate forced-stage archive ledger {path}: {error}",
            component="resume",
        ) from error
    store.write_json(
        relative_path,
        ledger.model_copy(update={"state": "manifest_committed"}).model_dump(mode="json"),
        replace=True,
    )


def recover_published_stage_outputs(
    run_directory: str | Path,
    manifest: RunManifest,
    config: ExperimentConfig,
) -> tuple[RunManifest, tuple[StageName, ...]]:
    """Adopt fully published outputs left behind by a pre-ledger process exit."""

    if manifest.status is not RunStatus.RUNNING:
        return manifest, ()
    store = LocalArtifactStore(run_directory)
    working = manifest
    recovered: list[StageName] = []
    for name in STAGE_ORDER:
        stage = working.stages.get(name)
        if stage is None or stage.status is not StageStatus.RUNNING:
            continue
        locations = tuple(store.path(value) for value in STAGE_OUTPUT_LOCATIONS[name])
        if not all(location.exists() for location in locations):
            continue
        if name == "probe_runtime":
            environment = cast(dict[str, Any], store.read_json("environment.json"))
            if not isinstance(environment.get("capabilities"), dict):
                continue
        output_paths: list[Path] = []
        for location in locations:
            if location.is_file() or location.is_symlink():
                output_paths.append(location)
            elif location.is_dir():
                output_paths.extend(
                    path
                    for path in sorted(location.rglob("*"))
                    if path.is_file() or path.is_symlink()
                )
        if not output_paths:
            continue
        expected_fingerprint = stage_fingerprint_from_manifest(name, config, working)
        if stage.fingerprint != expected_fingerprint:
            raise ArtifactIntegrityError(
                f"fingerprint mismatch for interrupted published stage {name}: "
                f"expected {expected_fingerprint}, recorded {stage.fingerprint}",
                component="resume",
            )
        result = record_stage_result(store.root, name, output_paths)
        working = working.finish_stage(name, result.outputs)
        if name == "probe_runtime":
            environment = cast(dict[str, Any], store.read_json("environment.json"))
            git_data = cast(dict[str, Any], environment.get("git", {}))
            working = working.model_copy(
                update={
                    "environment": environment,
                    "git": GitProvenance(
                        commit=cast(str | None, git_data.get("commit")),
                        dirty=cast(bool | None, git_data.get("dirty")),
                    ),
                }
            )
        elif name == "load_baseline":
            descriptor = cast(
                dict[str, Any], store.read_json("checkpoints/baseline/descriptor.json")
            )
            working = working.model_copy(
                update={
                    "model": ModelProvenance(
                        id=cast(str, descriptor["model_id"]),
                        revision=cast(str | None, descriptor["revision"]),
                        resolved_class=cast(str | None, descriptor.get("resolved_class")),
                        architecture=cast(str | None, descriptor.get("architecture")),
                        checksum=cast(str | None, descriptor.get("checksum")),
                    )
                }
            )
        recovered.append(name)
    return working, tuple(recovered)


def verify_successful_stage_outputs(run_directory: str | Path, manifest: RunManifest) -> None:
    """Verify every file of every successful stage before any resume mutation."""

    for name in sorted(manifest.stages):
        stage = manifest.stages[name]
        if stage.status is StageStatus.SUCCESS:
            verify_outputs(run_directory, stage)


def verify_successful_stage_fingerprints(config: ExperimentConfig, manifest: RunManifest) -> None:
    """Prove that successful stages still match config and direct input checksums."""

    for name in STAGE_ORDER:
        stage = _stage_record(manifest, name)
        if stage.status is not StageStatus.SUCCESS:
            continue
        expected = stage_fingerprint_from_manifest(name, config, manifest)
        if stage.fingerprint != expected:
            raise ArtifactIntegrityError(
                f"fingerprint mismatch for completed stage {name}: "
                f"expected {expected}, recorded {stage.fingerprint}",
                component="resume",
                details={
                    "stage": name,
                    "expected": expected,
                    "recorded": stage.fingerprint,
                },
            )


def reopen_manifest_for_resume(manifest: RunManifest, *, at: datetime | None = None) -> RunManifest:
    """Reopen failed/successful runs and restart interrupted running stages.

    ``RunManifest.start`` intentionally permits only a fresh pending run. Resume
    needs this separate copy-on-write transition so completed ledger evidence is
    retained while the top-level run returns to ``running``.
    """

    resumed_at = at or utc_now()
    stages = dict(manifest.stages)
    for name, stage in sorted(stages.items()):
        if stage.status is not StageStatus.RUNNING:
            continue
        stages[name] = stage.model_copy(
            update={
                "status": StageStatus.INVALIDATED,
                "fingerprint": None,
                "started_at": None,
                "completed_at": None,
                "outputs": (),
                "error": None,
                "reason": "interrupted before atomic completion; restart on resume",
            }
        )

    if manifest.status is RunStatus.PENDING:
        return manifest.model_copy(update={"stages": stages}).start(at=resumed_at)
    return manifest.model_copy(
        update={
            "status": RunStatus.RUNNING,
            "started_at": manifest.started_at or resumed_at,
            "completed_at": None,
            "stages": stages,
        }
    )


def prepare_resume(
    run_directory: str | Path,
    manifest: RunManifest,
    config: ExperimentConfig,
    *,
    force_stage: StageName | None = None,
    archive_id: str | None = None,
    at: datetime | None = None,
) -> ResumePreparation:
    """Verify a run, optionally archive a forced subtree, and build a resume plan."""

    if manifest.config_hash != config.config_hash():
        raise ArtifactIntegrityError(
            "resolved configuration hash does not match the resume target",
            component="resume",
            details={"expected": manifest.config_hash, "actual": config.config_hash()},
        )
    _verify_manifest_stage_set(manifest)
    verify_successful_stage_outputs(run_directory, manifest)
    verify_successful_stage_fingerprints(config, manifest)

    working = manifest
    invalidated: tuple[StageName, ...] = ()
    archived: tuple[ArchivedStage, ...] = ()
    forced_archive_ledger: str | None = None
    if force_stage is not None:
        invalidated = stage_descendants(force_stage, include_self=True)
        working, archived, forced_archive_ledger = archive_and_invalidate(
            run_directory,
            working,
            invalidated,
            forced_stage=force_stage,
            archive_id=archive_id or _default_archive_id(at),
        )

    working = reopen_manifest_for_resume(working, at=at)
    skip: list[StageName] = []
    restart: list[StageName] = []
    for name in STAGE_ORDER:
        stage = _stage_record(working, name)
        if stage.status is StageStatus.SUCCESS or (
            stage.status
            in {
                StageStatus.SKIPPED_NOT_REQUIRED,
                StageStatus.UNSUPPORTED,
            }
            and not stage.required
        ):
            skip.append(name)
        else:
            restart.append(name)

    return ResumePreparation(
        manifest=working,
        skip_stages=tuple(skip),
        restart_stages=tuple(restart),
        invalidated_stages=invalidated,
        archived_stages=archived,
        forced_archive_ledger=forced_archive_ledger,
    )


def archive_and_invalidate(
    run_directory: str | Path,
    manifest: RunManifest,
    names: tuple[StageName, ...],
    *,
    forced_stage: StageName,
    archive_id: str,
) -> tuple[RunManifest, tuple[ArchivedStage, ...], str]:
    """Move prior outputs to a unique archive, then invalidate a forced subtree."""

    store = LocalArtifactStore(run_directory)
    archive_root = store.path(Path("archive") / "forced" / archive_id)
    if archive_root.exists():
        raise ArtifactIntegrityError(
            f"forced-stage archive already exists: {archive_root.relative_to(store.root)}",
            component="resume",
        )

    planned: list[tuple[StageName, ArtifactChecksum, Path, Path]] = []
    seen_sources: set[Path] = set()
    for name in names:
        stage = _stage_record(manifest, name)
        if stage.outputs:
            verify_outputs(store.root, stage)
        for output in stage.outputs:
            source = store.path(output.path)
            if source in seen_sources:
                raise ArtifactIntegrityError(
                    f"multiple stages claim the same forced output: {output.path}",
                    component="resume",
                )
            seen_sources.add(source)
            destination = store.path(Path("archive") / "forced" / archive_id / name / output.path)
            if destination.exists():
                raise ArtifactIntegrityError(
                    f"forced-stage archive destination already exists: {destination}",
                    component="resume",
                )
            planned.append((name, output, source, destination))

    archived_by_stage: dict[StageName, list[ArtifactChecksum]] = {name: [] for name in names}
    for name, output, _, destination in planned:
        archived_by_stage[name].append(
            ArtifactChecksum(
                path=destination.relative_to(store.root).as_posix(),
                sha256=output.sha256,
                size_bytes=output.size_bytes,
            )
        )
    archived_records = tuple(
        ArchivedStage(
            name=name,
            previous_status=_stage_record(manifest, name).status,
            previous_fingerprint=_stage_record(manifest, name).fingerprint,
            original_outputs=_stage_record(manifest, name).outputs,
            archived_outputs=tuple(sorted(archived_by_stage[name], key=lambda output: output.path)),
        )
        for name in names
    )
    ledger_relative = (Path("archive") / "forced" / archive_id / "archive_manifest.json").as_posix()
    ledger = ForcedArchiveLedger(
        archive_id=archive_id,
        forced_stage=forced_stage,
        state="preparing",
        created_at=utc_now(),
        stages=archived_records,
    )
    store.write_json(ledger_relative, ledger.model_dump(mode="json"))

    moved: list[tuple[Path, Path]] = []
    try:
        for _, _, source, destination in planned:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, destination)
            moved.append((source, destination))
    except OSError as error:
        for source, destination in reversed(moved):
            source.parent.mkdir(parents=True, exist_ok=True)
            os.replace(destination, source)
        store.write_json(
            ledger_relative,
            ledger.model_copy(update={"state": "rolled_back"}).model_dump(mode="json"),
            replace=True,
        )
        raise ArtifactIntegrityError(
            f"unable to archive forced-stage outputs: {error}",
            component="resume",
        ) from error
    store.write_json(
        ledger_relative,
        ledger.model_copy(update={"state": "outputs_archived"}).model_dump(mode="json"),
        replace=True,
    )

    # Canonical stage outputs can be top-level directories such as ``reports``
    # or ``checkpoints/candidate``. Remove parents left empty by the moves so a
    # forced rerun can atomically commit a replacement at the same path.
    for source, _ in moved:
        _prune_empty_parents(source.parent, store.root)

    stages = dict(manifest.stages)
    for name in names:
        previous = _stage_record(manifest, name)
        stages[name] = previous.model_copy(
            update={
                "status": StageStatus.INVALIDATED,
                "fingerprint": None,
                "started_at": None,
                "completed_at": None,
                "outputs": (),
                "error": None,
                "reason": (
                    f"invalidated by --force-stage {forced_stage}; "
                    f"prior outputs archived under archive/forced/{archive_id}"
                ),
            }
        )
        _prune_standard_stage_directory(store.root, name)

    return manifest.model_copy(update={"stages": stages}), archived_records, ledger_relative


def _verify_manifest_stage_set(manifest: RunManifest) -> None:
    expected = set(STAGE_ORDER)
    actual = set(manifest.stages)
    if actual != expected:
        raise ArtifactIntegrityError(
            f"manifest stage set mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}",
            component="resume",
        )


def _stage_record(manifest: RunManifest, name: StageName) -> StageRecord:
    try:
        return manifest.stages[name]
    except KeyError as error:
        raise ArtifactIntegrityError(
            f"manifest is missing governed stage {name!r}",
            component="resume",
        ) from error


def _default_archive_id(at: datetime | None) -> str:
    value = at or utc_now()
    return value.strftime("%Y%m%dT%H%M%S.%fZ")


def _prune_standard_stage_directory(root: Path, name: StageName) -> None:
    directory = root / "stages" / name
    if not directory.exists() or not directory.is_dir():
        return
    for candidate in sorted(directory.rglob("*"), reverse=True):
        if candidate.is_dir():
            with suppress(OSError):
                candidate.rmdir()
    with suppress(OSError):
        directory.rmdir()


def _prune_empty_parents(directory: Path, root: Path) -> None:
    candidate = directory
    while candidate != root and candidate.is_relative_to(root):
        try:
            candidate.rmdir()
        except OSError:
            break
        candidate = candidate.parent


__all__ = [
    "ArchivedStage",
    "ForcedArchiveLedger",
    "ResumePreparation",
    "archive_and_invalidate",
    "load_resume_config",
    "mark_forced_archive_committed",
    "preflight_resume_integrity",
    "prepare_resume",
    "recover_forced_archive_transactions",
    "recover_published_stage_outputs",
    "reopen_manifest_for_resume",
    "verify_successful_stage_fingerprints",
    "verify_successful_stage_outputs",
]
