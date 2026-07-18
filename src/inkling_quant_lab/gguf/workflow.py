"""Small immutable stage ledger for the external Inkling GGUF workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, TypeAlias

from inkling_quant_lab.artifacts import sha256_file
from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.gguf.inkling import InklingGGUFConfig, InklingSourceAudit
from inkling_quant_lab.manifests import (
    ArtifactChecksum,
    ModelProvenance,
    RunManifest,
    StageRecord,
    StageStatus,
    verify_outputs,
)

GGUFStageName: TypeAlias = Literal[
    "preflight",
    "materialize_source",
    "convert_text_bf16",
    "convert_multimodal_projector",
    "quantize_text",
    "verify_export",
    "smoke_test",
    "finalize",
]

STAGE_ORDER: tuple[GGUFStageName, ...] = (
    "preflight",
    "materialize_source",
    "convert_text_bf16",
    "convert_multimodal_projector",
    "quantize_text",
    "verify_export",
    "smoke_test",
    "finalize",
)


def initial_manifest(
    config: InklingGGUFConfig,
    audit: InklingSourceAudit,
    *,
    run_id: str,
) -> RunManifest:
    """Create the isolated ledger without pretending it is the 14-stage model pipeline."""

    if (
        not audit.verified
        or audit.model_id != config.source.model_id
        or audit.revision != config.source.revision
        or audit.architecture != config.source.architecture
        or audit.model_type != config.source.model_type
    ):
        raise ArtifactIntegrityError("Source audit does not match the configured pinned Inkling")

    return RunManifest(
        run_id=run_id,
        config_hash=config.config_hash(),
        model=ModelProvenance(
            id=audit.model_id,
            revision=audit.revision,
            resolved_class=audit.architecture,
            architecture=audit.model_type,
            checksum=audit.weight_index_sha256,
        ),
        environment={
            "workflow": "manual-large-inkling-gguf-v1",
            "llama_cpp_repository": config.toolchain.repository,
            "llama_cpp_commit": config.toolchain.commit,
            "quant_type": config.quantization.quant_type,
            "license": config.source.license,
            "coverage": config.coverage.model_dump(mode="json"),
        },
        stages={name: StageRecord(name=name) for name in STAGE_ORDER},
        warnings=audit.warnings,
    )


def stage_fingerprint(
    name: GGUFStageName,
    config: InklingGGUFConfig,
    dependency_checksums: tuple[str, ...],
) -> str:
    """Hash exact config, ordered stage identity, and immutable dependency outputs."""

    if any(len(checksum) != 64 for checksum in dependency_checksums):
        raise ArtifactIntegrityError("stage dependency checksum is not SHA-256")
    payload = {
        "schema": "inkling-gguf-stage-v1",
        "name": name,
        "config_hash": config.config_hash(),
        "dependency_checksums": sorted(dependency_checksums),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def output_record(root: Path, path: Path) -> ArtifactChecksum:
    """Hash a regular non-symlink output proven to reside below the run root."""

    root_resolved = root.resolve()
    if path.is_symlink() or not path.is_file():
        raise ArtifactIntegrityError(f"Stage output is not a regular file: {path}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ArtifactIntegrityError(f"Stage output escapes run root: {path}")
    return ArtifactChecksum(
        path=str(resolved.relative_to(root_resolved)),
        sha256=sha256_file(resolved),
        size_bytes=resolved.stat().st_size,
    )


def verify_completed_stage(
    run_directory: Path,
    stage: StageRecord,
    expected_fingerprint: str,
) -> None:
    """Allow resume skipping only for a successful, current, checksum-valid stage."""

    if stage.status is not StageStatus.SUCCESS:
        raise ArtifactIntegrityError(f"Stage {stage.name} is not successful")
    if not stage.outputs:
        raise ArtifactIntegrityError(f"Completed stage {stage.name} has no recorded outputs")
    if stage.fingerprint != expected_fingerprint:
        raise ArtifactIntegrityError(
            f"Completed stage fingerprint mismatch for {stage.name}", component="gguf_resume"
        )
    verify_outputs(run_directory, stage)


def next_pending_stage(
    manifest: RunManifest,
    *,
    run_directory: Path | None = None,
    config: InklingGGUFConfig | None = None,
) -> GGUFStageName | None:
    """Return the first non-success stage after verifying every completed predecessor."""

    dependency_checksums: tuple[str, ...] = ()
    for name in STAGE_ORDER:
        stage = manifest.stages[name]
        if stage.status is not StageStatus.SUCCESS:
            return name
        if run_directory is None or config is None:
            raise ArtifactIntegrityError(
                "run_directory and config are required to skip completed GGUF stages"
            )
        expected = stage_fingerprint(name, config, dependency_checksums)
        verify_completed_stage(run_directory, stage, expected)
        dependency_checksums = tuple(output.sha256 for output in stage.outputs)
    return None


__all__ = [
    "STAGE_ORDER",
    "GGUFStageName",
    "initial_manifest",
    "next_pending_stage",
    "output_record",
    "stage_fingerprint",
    "verify_completed_stage",
]
