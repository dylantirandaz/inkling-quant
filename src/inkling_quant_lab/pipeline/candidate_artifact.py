"""Integrity-bound loading for a governed exported candidate artifact."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.manifests import RunManifest, StageRecord, StageStatus, load_manifest
from inkling_quant_lab.models.base import (
    ModelCapabilities,
    ModelDescriptor,
    RuntimeForModel,
    SourceWeightFreeModelAdapter,
    SourceWeightFreeReloadProvenance,
)
from inkling_quant_lab.quantization.base import QuantizationManifest, QuantizedModel
from inkling_quant_lab.quantization.reference import (
    HF_CAUSAL_LM_SOURCE_WEIGHT_FREE_RELOAD_ADAPTER,
    reload_exported_model_source_weight_free,
)
from inkling_quant_lab.security import safe_path

_SOURCE_WEIGHT_FREE_MODE = "isolated_subject_artifact_peak_rss"
_BASELINE_DESCRIPTOR_PATH = "checkpoints/baseline/descriptor.json"
_CANDIDATE_METADATA_PATH = "checkpoints/candidate/candidate/metadata.json"
_CANDIDATE_TENSOR_PATH = "checkpoints/candidate/candidate/model.safetensors"
_CANDIDATE_MANIFEST_PATH = "checkpoints/candidate/quantization_manifest.json"
_CANDIDATE_BUNDLE_PATH = "checkpoints/candidate/candidate"
_MAX_RETAINED_JSON_BYTES = 16 * 1024 * 1024


class _ImmutablePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _CapabilitiesPayload(_ImmutablePayload):
    supports_text: bool
    supports_images: bool
    supports_audio: bool
    is_moe: bool
    supports_router_logits: bool
    supports_token_level_routes: bool
    supported_dtypes: tuple[str, ...]
    supported_device_maps: tuple[str, ...]
    max_context_length: int = Field(gt=0)
    requires_remote_code: bool


class _BaselineDescriptorPayload(_ImmutablePayload):
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    resolved_class: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    capabilities: _CapabilitiesPayload
    serialized_size_bytes: int = Field(gt=0)


class _ExportModelPayload(_ImmutablePayload):
    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    resolved_class: str = Field(min_length=1)
    architecture: str = Field(min_length=1)
    source_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")


class _ExportReloadPayload(_ImmutablePayload):
    adapter: Literal["hf_causal_lm_source_weight_free_v1"]
    backend: str = Field(min_length=1)
    format: Literal["safetensors"]
    metadata_file: Literal["metadata.json"]
    tensor_file: Literal["model.safetensors"]
    tensor_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _ExportMetadataPayload(_ImmutablePayload):
    schema_version: Literal["2.0"]
    model: _ExportModelPayload
    quantization: QuantizationManifest
    reload: _ExportReloadPayload


@dataclass(frozen=True, slots=True)
class _VerifiedArtifact:
    path: Path
    sha256: str
    size_bytes: int
    payload: bytes | None


@dataclass(frozen=True, slots=True)
class VerifiedBaselineDescriptor:
    """A baseline descriptor whose exact stage output bytes passed verification."""

    descriptor: ModelDescriptor
    serialized_size_bytes: int
    artifact_sha256: str
    artifact_size_bytes: int


def _resolve_run_directory(run_directory: str | Path) -> Path:
    requested = Path(run_directory).expanduser()
    if requested.is_symlink():
        raise ArtifactIntegrityError(
            f"governed run directory must not be a symlink: {requested}",
            component="candidate_artifact",
        )
    try:
        root = requested.resolve(strict=True)
    except OSError as error:
        raise ArtifactIntegrityError(
            f"governed run directory is unavailable: {requested}: {error}",
            component="candidate_artifact",
        ) from error
    if not root.is_dir():
        raise ArtifactIntegrityError(
            f"governed run directory is not a directory: {root}",
            component="candidate_artifact",
        )
    return root


def _reject_symlink_components(root: Path, relative: str) -> None:
    current = root
    for part in PurePosixPath(relative).parts:
        current /= part
        if current.is_symlink():
            raise ArtifactIntegrityError(
                f"unsafe output path contains a symlink: {relative}",
                component="candidate_artifact",
            )


def _safe_output_path(root: Path, stage_name: str, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        not relative
        or pure.is_absolute()
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != relative
    ):
        raise ArtifactIntegrityError(
            f"stage {stage_name} declares an unsafe output path: {relative!r}",
            component="candidate_artifact",
        )
    _reject_symlink_components(root, relative)
    try:
        path = safe_path(root, relative)
    except ArtifactIntegrityError as error:
        raise ArtifactIntegrityError(
            f"stage {stage_name} declares an unsafe output path: {relative!r}",
            component="candidate_artifact",
        ) from error
    if not path.is_file():
        raise ArtifactIntegrityError(
            f"stage {stage_name} output is missing or not a regular file: {relative}",
            component="candidate_artifact",
        )
    return path


def _stream_verify(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    stage_name: str,
    relative: str,
    retain_payload: bool,
) -> _VerifiedArtifact:
    if retain_payload and expected_size_bytes > _MAX_RETAINED_JSON_BYTES:
        raise ArtifactIntegrityError(
            f"stage {stage_name} JSON output is unexpectedly large: {relative}",
            component="candidate_artifact",
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ArtifactIntegrityError(
            f"unable to open stage {stage_name} output safely: {relative}: {error}",
            component="candidate_artifact",
        ) from error
    digest = hashlib.sha256()
    retained = bytearray() if retain_payload else None
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ArtifactIntegrityError(
                    f"stage {stage_name} output is not a regular file: {relative}",
                    component="candidate_artifact",
                )
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                if retained is not None:
                    retained.extend(chunk)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise ArtifactIntegrityError(
            f"unable to stream stage {stage_name} output: {relative}: {error}",
            component="candidate_artifact",
        ) from error
    stable_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) == (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    actual_sha256 = digest.hexdigest()
    if not stable_identity:
        raise ArtifactIntegrityError(
            f"stage {stage_name} output changed while it was verified: {relative}",
            component="candidate_artifact",
        )
    if before.st_size != expected_size_bytes or actual_sha256 != expected_sha256:
        raise ArtifactIntegrityError(
            f"Checksum mismatch for completed stage {stage_name}: {relative}",
            component="candidate_artifact",
            details={
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
                "expected_size_bytes": expected_size_bytes,
                "actual_size_bytes": before.st_size,
            },
        )
    return _VerifiedArtifact(
        path=path,
        sha256=actual_sha256,
        size_bytes=before.st_size,
        payload=None if retained is None else bytes(retained),
    )


def _require_successful_stage(manifest: RunManifest, name: str) -> StageRecord:
    try:
        stage = manifest.stages[name]
    except KeyError as error:
        raise ArtifactIntegrityError(
            f"governed run manifest is missing required stage {name!r}",
            component="candidate_artifact",
        ) from error
    if stage.name != name:
        raise ArtifactIntegrityError(
            f"governed stage key/name mismatch for {name!r}: {stage.name!r}",
            component="candidate_artifact",
        )
    if stage.status is not StageStatus.SUCCESS:
        raise ArtifactIntegrityError(
            f"governed stage {name!r} is not successful: {stage.status}",
            component="candidate_artifact",
        )
    return stage


def _verify_exact_stage_outputs(
    root: Path,
    stage: StageRecord,
    expected_paths: tuple[str, ...],
    *,
    retained_paths: frozenset[str],
) -> dict[str, _VerifiedArtifact]:
    declared: dict[str, tuple[str, int, Path]] = {}
    for output in stage.outputs:
        if output.path in declared:
            raise ArtifactIntegrityError(
                f"stage {stage.name} declares duplicate output {output.path!r}",
                component="candidate_artifact",
            )
        path = _safe_output_path(root, stage.name, output.path)
        declared[output.path] = (output.sha256, output.size_bytes, path)
    if set(declared) != set(expected_paths) or len(declared) != len(expected_paths):
        raise ArtifactIntegrityError(
            f"stage {stage.name} output path set does not match the governed contract",
            component="candidate_artifact",
            details={
                "expected": list(expected_paths),
                "declared": sorted(declared),
            },
        )
    return {
        relative: _stream_verify(
            declared[relative][2],
            expected_sha256=declared[relative][0],
            expected_size_bytes=declared[relative][1],
            stage_name=stage.name,
            relative=relative,
            retain_payload=relative in retained_paths,
        )
        for relative in expected_paths
    }


def _load_governance(
    run_directory: str | Path, config: ExperimentConfig
) -> tuple[Path, RunManifest]:
    if config.benchmark.host_memory_mode != _SOURCE_WEIGHT_FREE_MODE:
        raise ArtifactIntegrityError(
            "governed source-weight-free artifact loading requires "
            f"benchmark.host_memory_mode={_SOURCE_WEIGHT_FREE_MODE!r}",
            component="candidate_artifact",
        )
    root = _resolve_run_directory(run_directory)
    _safe_output_path(root, "manifest", "manifest.json")
    manifest = load_manifest(root)
    active_hash = config.config_hash()
    if manifest.config_hash != active_hash:
        raise ArtifactIntegrityError(
            "governed run config hash does not match the active resolved configuration",
            component="candidate_artifact",
            details={"manifest": manifest.config_hash, "active": active_hash},
        )
    if (
        manifest.model.id != config.model.model_id
        or manifest.model.revision != config.model.revision
    ):
        raise ArtifactIntegrityError(
            "governed run model identity does not match the active resolved configuration",
            component="candidate_artifact",
        )
    return root, manifest


def _parse_baseline_descriptor(
    artifact: _VerifiedArtifact,
    *,
    config: ExperimentConfig,
    manifest: RunManifest,
) -> VerifiedBaselineDescriptor:
    assert artifact.payload is not None
    try:
        payload = _BaselineDescriptorPayload.model_validate_json(artifact.payload)
    except ValidationError as error:
        raise ArtifactIntegrityError(
            f"verified baseline descriptor is invalid: {error}",
            component="candidate_artifact",
        ) from error
    expected = (
        config.model.model_id,
        config.model.revision,
        manifest.model.resolved_class,
        manifest.model.architecture,
        manifest.model.checksum,
    )
    observed = (
        payload.model_id,
        payload.revision,
        payload.resolved_class,
        payload.architecture,
        payload.checksum,
    )
    if observed != expected:
        raise ArtifactIntegrityError(
            "verified baseline descriptor does not match config and manifest model identity",
            component="candidate_artifact",
            details={"expected": list(expected), "observed": list(observed)},
        )
    capabilities = ModelCapabilities(**payload.capabilities.model_dump(mode="python"))
    return VerifiedBaselineDescriptor(
        descriptor=ModelDescriptor(
            model_id=payload.model_id,
            revision=payload.revision,
            resolved_class=payload.resolved_class,
            architecture=payload.architecture,
            checksum=payload.checksum,
            capabilities=capabilities,
        ),
        serialized_size_bytes=payload.serialized_size_bytes,
        artifact_sha256=artifact.sha256,
        artifact_size_bytes=artifact.size_bytes,
    )


def _verified_baseline_descriptor(
    root: Path, manifest: RunManifest, config: ExperimentConfig
) -> VerifiedBaselineDescriptor:
    stage = _require_successful_stage(manifest, "load_baseline")
    artifacts = _verify_exact_stage_outputs(
        root,
        stage,
        (_BASELINE_DESCRIPTOR_PATH,),
        retained_paths=frozenset({_BASELINE_DESCRIPTOR_PATH}),
    )
    return _parse_baseline_descriptor(
        artifacts[_BASELINE_DESCRIPTOR_PATH], config=config, manifest=manifest
    )


def load_verified_baseline_descriptor(
    run_directory: str | Path, config: ExperimentConfig
) -> VerifiedBaselineDescriptor:
    """Verify and parse the persisted baseline descriptor without opening model weights."""

    root, manifest = _load_governance(run_directory, config)
    return _verified_baseline_descriptor(root, manifest, config)


def _parse_quantization_manifest(artifact: _VerifiedArtifact) -> QuantizationManifest:
    assert artifact.payload is not None
    try:
        return QuantizationManifest.model_validate_json(artifact.payload)
    except ValidationError as error:
        raise ArtifactIntegrityError(
            f"verified canonical quantization manifest is invalid: {error}",
            component="candidate_artifact",
        ) from error


def _parse_export_metadata(artifact: _VerifiedArtifact) -> _ExportMetadataPayload:
    assert artifact.payload is not None
    try:
        return _ExportMetadataPayload.model_validate_json(artifact.payload)
    except ValidationError as error:
        raise ArtifactIntegrityError(
            f"verified candidate export metadata is invalid: {error}",
            component="candidate_artifact",
        ) from error


def _bundle_sha256(metadata_sha256: str, tensor_sha256: str) -> str:
    digest = hashlib.sha256()
    for name, checksum in (
        ("metadata.json", metadata_sha256),
        ("model.safetensors", tensor_sha256),
    ):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(checksum))
    return digest.hexdigest()


def _validate_export_contract(
    *,
    config: ExperimentConfig,
    baseline: VerifiedBaselineDescriptor,
    artifacts: dict[str, _VerifiedArtifact],
) -> tuple[QuantizationManifest, _ExportMetadataPayload, str]:
    canonical = _parse_quantization_manifest(artifacts[_CANDIDATE_MANIFEST_PATH])
    metadata = _parse_export_metadata(artifacts[_CANDIDATE_METADATA_PATH])
    if metadata.quantization != canonical:
        raise ArtifactIntegrityError(
            "candidate embedded quantization manifest differs from the canonical persisted "
            "quantization manifest",
            component="candidate_artifact",
        )
    if (
        canonical.backend != config.quantization.backend
        or canonical.method != config.quantization.method
    ):
        raise ArtifactIntegrityError(
            "candidate quantization identity does not match the active resolved configuration",
            component="candidate_artifact",
        )
    if canonical.source_model_checksum != baseline.descriptor.checksum:
        raise ArtifactIntegrityError(
            "candidate source checksum does not match the verified baseline descriptor",
            component="candidate_artifact",
        )
    expected_model = (
        baseline.descriptor.model_id,
        baseline.descriptor.revision,
        baseline.descriptor.resolved_class,
        baseline.descriptor.architecture,
        baseline.descriptor.checksum,
    )
    observed_model = (
        metadata.model.model_id,
        metadata.model.revision,
        metadata.model.resolved_class,
        metadata.model.architecture,
        metadata.model.source_checksum,
    )
    if observed_model != expected_model:
        raise ArtifactIntegrityError(
            "candidate export model identity does not match the verified baseline descriptor",
            component="candidate_artifact",
        )
    tensor = artifacts[_CANDIDATE_TENSOR_PATH]
    metadata_artifact = artifacts[_CANDIDATE_METADATA_PATH]
    if (
        metadata.reload.adapter != HF_CAUSAL_LM_SOURCE_WEIGHT_FREE_RELOAD_ADAPTER
        or metadata.reload.backend != canonical.backend
        or metadata.reload.tensor_sha256 != tensor.sha256
    ):
        raise ArtifactIntegrityError(
            "candidate export reload metadata does not match the externally verified bundle",
            component="candidate_artifact",
        )
    actual_bundle_size = metadata_artifact.size_bytes + tensor.size_bytes
    if canonical.serialized_size_bytes != actual_bundle_size:
        raise ArtifactIntegrityError(
            "candidate serialized size does not match the externally verified bundle",
            component="candidate_artifact",
            details={
                "manifest": canonical.serialized_size_bytes,
                "actual": actual_bundle_size,
            },
        )
    return (
        canonical,
        metadata,
        _bundle_sha256(metadata_artifact.sha256, tensor.sha256),
    )


def _validate_reload_result(
    *,
    candidate: QuantizedModel,
    provenance: SourceWeightFreeReloadProvenance,
    canonical: QuantizationManifest,
    metadata: _ExportMetadataPayload,
    artifacts: dict[str, _VerifiedArtifact],
    expected_bundle_sha256: str,
) -> None:
    if candidate.manifest != canonical:
        raise ArtifactIntegrityError(
            "reloaded candidate manifest differs from the canonical persisted manifest",
            component="candidate_artifact",
        )
    identity_matches = (
        provenance.reload_adapter == metadata.reload.adapter
        and provenance.backend == canonical.backend
        and provenance.model_id == metadata.model.model_id
        and provenance.revision == metadata.model.revision
        and provenance.resolved_class == metadata.model.resolved_class
        and provenance.architecture == metadata.model.architecture
        and provenance.source_model_checksum == metadata.model.source_checksum
    )
    external_hashes_match = (
        provenance.metadata_file == metadata.reload.metadata_file
        and provenance.metadata_sha256 == artifacts[_CANDIDATE_METADATA_PATH].sha256
        and provenance.tensor_file == metadata.reload.tensor_file
        and provenance.tensor_sha256 == artifacts[_CANDIDATE_TENSOR_PATH].sha256
        and provenance.bundle_sha256 == expected_bundle_sha256
    )
    strict_source_weight_free = (
        provenance.strict_load
        and provenance.assign
        and not provenance.missing_keys
        and not provenance.unexpected_keys
        and not provenance.meta_tensor_names
        and not provenance.source_weights_loaded
    )
    if not identity_matches or not external_hashes_match or not strict_source_weight_free:
        raise ArtifactIntegrityError(
            "source-weight-free reload provenance does not match the governed artifact contract",
            component="candidate_artifact",
        )


def load_governed_candidate_artifact(
    run_directory: str | Path,
    config: ExperimentConfig,
    adapter: SourceWeightFreeModelAdapter,
    runtime: RuntimeForModel,
) -> tuple[QuantizedModel, SourceWeightFreeReloadProvenance]:
    """Verify and load the exact exported candidate without loading source model weights."""

    root, manifest = _load_governance(run_directory, config)
    baseline = _verified_baseline_descriptor(root, manifest, config)
    quantize = _require_successful_stage(manifest, "quantize")
    artifacts = _verify_exact_stage_outputs(
        root,
        quantize,
        (
            _CANDIDATE_METADATA_PATH,
            _CANDIDATE_TENSOR_PATH,
            _CANDIDATE_MANIFEST_PATH,
        ),
        retained_paths=frozenset({_CANDIDATE_METADATA_PATH, _CANDIDATE_MANIFEST_PATH}),
    )
    canonical, metadata, bundle_sha256 = _validate_export_contract(
        config=config,
        baseline=baseline,
        artifacts=artifacts,
    )
    try:
        candidate, provenance = reload_exported_model_source_weight_free(
            root / _CANDIDATE_BUNDLE_PATH,
            adapter,
            config,
            runtime,
        )
    except (OSError, ValueError) as error:
        raise ArtifactIntegrityError(
            f"source-weight-free candidate export reload failed: {error}",
            component="candidate_artifact",
        ) from error
    _validate_reload_result(
        candidate=candidate,
        provenance=provenance,
        canonical=canonical,
        metadata=metadata,
        artifacts=artifacts,
        expected_bundle_sha256=bundle_sha256,
    )
    return candidate, provenance


__all__ = [
    "VerifiedBaselineDescriptor",
    "load_governed_candidate_artifact",
    "load_verified_baseline_descriptor",
]
