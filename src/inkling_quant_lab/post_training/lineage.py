"""Fail-closed lineage for a six-router replacement overlay.

The records deliberately retain hashes and provenance rather than training data
or tensor payloads. Tensor application and audit use ordinary CPU-testable
PyTorch state dictionaries and never import an accelerator-specific runtime.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Final, Literal, Self, TypeAlias

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor

from inkling_quant_lab.security import sensitive_literal_path

ROUTER_TENSOR_COUNT: Final = 6
ROUTER_LAYER_INDICES: Final = tuple(range(ROUTER_TENSOR_COUNT))
TENSOR_HASH_SCHEME: Final = (
    "sha256(torch dtype ASCII + NUL + compact JSON shape + NUL + contiguous CPU tensor bytes)"
)
STATE_HASH_SCHEME: Final = (
    "sha256(sorted UTF-8 tensor name length/name + canonical tensor SHA-256 bytes)"
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_REVISION_PATTERN = r"^[0-9a-f]{40}$"

TensorState: TypeAlias = Mapping[str, Tensor]


class RouterOverlayError(ValueError):
    """A router overlay or its lineage failed a deterministic integrity check."""


class ImmutableLineageRecord(BaseModel):
    """Strict immutable and finite record base with canonical serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    def canonical_json(self) -> str:
        """Serialize the record deterministically for hashing and persistence."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def sha256(self) -> str:
        """Hash the canonical record bytes."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class ParentModelIdentity(ImmutableLineageRecord):
    """Exact immutable identity of the model receiving the router overlay."""

    model_id: str = Field(min_length=1)
    revision: str = Field(pattern=_REVISION_PATTERN)
    architecture: str = Field(min_length=1)
    resolved_class: str = Field(min_length=1)
    checkpoint_format: Literal["safetensors"] = "safetensors"
    trust_remote_code: Literal[False] = False
    config_sha256: str = Field(pattern=_SHA256_PATTERN)
    tokenizer_sha256: str = Field(pattern=_SHA256_PATTERN)
    weights_sha256: str = Field(pattern=_SHA256_PATTERN)
    state_dict_sha256: str = Field(pattern=_SHA256_PATTERN)
    tensor_count: int = Field(ge=ROUTER_TENSOR_COUNT)
    router_tensor_names: tuple[str, ...]
    router_shape: tuple[int, int]
    router_dtype: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")

    @model_validator(mode="after")
    def exact_router_inventory(self) -> Self:
        if len(self.router_tensor_names) != ROUTER_TENSOR_COUNT:
            raise ValueError("parent identity requires exactly six router tensor names")
        if len(set(self.router_tensor_names)) != ROUTER_TENSOR_COUNT:
            raise ValueError("parent router tensor names must be unique")
        if any(not name or "\0" in name for name in self.router_tensor_names):
            raise ValueError("parent router tensor names must be non-empty and contain no NUL")
        if any(dimension <= 0 for dimension in self.router_shape):
            raise ValueError("parent router shape dimensions must be positive")
        return self


class CorpusProvenance(ImmutableLineageRecord):
    """Exact selected corpus partition without retaining document contents."""

    corpus_id: str = Field(min_length=1)
    role: Literal["train", "validation", "test"]
    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    declared_license: str = Field(min_length=1)
    source_size_bytes: int = Field(gt=0)
    source_sha256: str = Field(pattern=_SHA256_PATTERN)
    encoding: Literal["utf-8"] = "utf-8"
    parser_version: str = Field(min_length=1)
    split: str = Field(min_length=1)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    ordered_sample_ids_sha256: str = Field(pattern=_SHA256_PATTERN)
    sample_count: int = Field(ge=1)
    token_count: int = Field(ge=1)


class TrainingRunProvenance(ImmutableLineageRecord):
    """Resolved hyperparameters and objective identity for one training run."""

    run_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    steps: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    sequence_length: int = Field(ge=1)
    optimizer: str = Field(min_length=1)
    learning_rate: float = Field(gt=0.0)
    optimizer_betas: tuple[float, float]
    optimizer_epsilon: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    bias_correction: bool
    objective_name: Literal["domain_pair_soft_target_cross_entropy"] = (
        "domain_pair_soft_target_cross_entropy"
    )
    objective_version: Literal["top2-domain-pair-ce-v1"] = "top2-domain-pair-ce-v1"
    training_config_sha256: str = Field(pattern=_SHA256_PATTERN)
    domain_pair_config_sha256: str = Field(pattern=_SHA256_PATTERN)

    @field_validator("optimizer_betas")
    @classmethod
    def optimizer_betas_are_probabilities(cls, value: tuple[float, float]) -> tuple[float, float]:
        if any(beta < 0.0 or beta >= 1.0 for beta in value):
            raise ValueError("optimizer betas must be in the half-open interval [0, 1)")
        return value


class TrainingSourceProvenance(ImmutableLineageRecord):
    """Exact source, environment, and dependency bytes used for training."""

    repository: str = Field(min_length=1)
    repository_revision: str | None = None
    repository_dirty: bool | None = None
    entrypoint: str = Field(min_length=1)
    entrypoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    environment_sha256: str = Field(pattern=_SHA256_PATTERN)
    dependency_lock_sha256: str = Field(pattern=_SHA256_PATTERN)
    framework: str = Field(min_length=1)
    framework_version: str = Field(min_length=1)


class RouterTensorOverlay(ImmutableLineageRecord):
    """One exact replacement tensor and the parent tensor it supersedes."""

    layer_index: int = Field(ge=0, le=ROUTER_TENSOR_COUNT - 1)
    name: str = Field(min_length=1)
    payload_kind: Literal["replacement_tensor"] = "replacement_tensor"
    shape: tuple[int, ...]
    dtype: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    parent_tensor_sha256: str = Field(pattern=_SHA256_PATTERN)
    overlay_tensor_sha256: str = Field(pattern=_SHA256_PATTERN)
    hash_scheme: Literal[
        "sha256(torch dtype ASCII + NUL + compact JSON shape + NUL + contiguous CPU tensor bytes)"
    ] = TENSOR_HASH_SCHEME

    @model_validator(mode="after")
    def replacement_is_well_formed(self) -> Self:
        if len(self.shape) != 2 or any(dimension <= 0 for dimension in self.shape):
            raise ValueError("router replacement tensors must have a positive rank-two shape")
        if "\0" in self.name:
            raise ValueError("router tensor name must not contain NUL")
        if self.parent_tensor_sha256 == self.overlay_tensor_sha256:
            raise ValueError("every learned router replacement must differ from its parent")
        return self


class UnchangedNonRouterProof(ImmutableLineageRecord):
    """Canonical equality proof for every tensor outside the six routers."""

    router_tensor_names: tuple[str, ...]
    parent_nonrouter_tensor_count: int = Field(ge=0)
    candidate_nonrouter_tensor_count: int = Field(ge=0)
    parent_nonrouter_numel: int = Field(ge=0)
    candidate_nonrouter_numel: int = Field(ge=0)
    parent_nonrouter_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_nonrouter_sha256: str = Field(pattern=_SHA256_PATTERN)
    hash_scheme: Literal[
        "sha256(sorted UTF-8 tensor name length/name + canonical tensor SHA-256 bytes)"
    ] = STATE_HASH_SCHEME

    @model_validator(mode="after")
    def nonrouters_are_exactly_unchanged(self) -> Self:
        if (
            len(self.router_tensor_names) != ROUTER_TENSOR_COUNT
            or len(set(self.router_tensor_names)) != ROUTER_TENSOR_COUNT
        ):
            raise ValueError("non-router proof requires six unique router exclusions")
        if self.parent_nonrouter_tensor_count != self.candidate_nonrouter_tensor_count:
            raise ValueError("non-router tensor counts differ")
        if self.parent_nonrouter_numel != self.candidate_nonrouter_numel:
            raise ValueError("non-router element counts differ")
        if self.parent_nonrouter_sha256 != self.candidate_nonrouter_sha256:
            raise ValueError("non-router aggregate hashes differ")
        return self


class ReloadProof(ImmutableLineageRecord):
    """Exact safe-reload equality for the reconstructed candidate state."""

    serialization_format: Literal["safetensors"] = "safetensors"
    pickle_used: Literal[False] = False
    trust_remote_code: Literal[False] = False
    overlay_bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    loader_name: str = Field(min_length=1)
    loader_version: str = Field(min_length=1)
    candidate_state_sha256: str = Field(pattern=_SHA256_PATTERN)
    reloaded_state_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_router_sha256: str = Field(pattern=_SHA256_PATTERN)
    reloaded_router_sha256: str = Field(pattern=_SHA256_PATTERN)
    hash_scheme: Literal[
        "sha256(sorted UTF-8 tensor name length/name + canonical tensor SHA-256 bytes)"
    ] = STATE_HASH_SCHEME

    @model_validator(mode="after")
    def reload_is_exact(self) -> Self:
        if self.candidate_state_sha256 != self.reloaded_state_sha256:
            raise ValueError("reloaded candidate state differs from the pre-save candidate")
        if self.candidate_router_sha256 != self.reloaded_router_sha256:
            raise ValueError("reloaded router tensors differ from the pre-save candidate")
        return self


class RouterOverlayLineage(ImmutableLineageRecord):
    """Complete immutable lineage for an exact six-router learned overlay."""

    schema_version: Literal["router-overlay-lineage-v1"] = "router-overlay-lineage-v1"
    parent: ParentModelIdentity
    corpus_contract_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus: tuple[CorpusProvenance, ...]
    training: TrainingRunProvenance
    source: TrainingSourceProvenance
    router_tensors: tuple[RouterTensorOverlay, ...]
    unchanged_nonrouters: UnchangedNonRouterProof
    reload: ReloadProof

    @model_validator(mode="after")
    def lineage_is_complete_and_narrow(self) -> Self:
        if not self.corpus or not any(item.role == "train" for item in self.corpus):
            raise ValueError("router training lineage requires at least one training corpus")
        corpus_ids = tuple(item.corpus_id for item in self.corpus)
        if len(set(corpus_ids)) != len(corpus_ids):
            raise ValueError("corpus provenance IDs must be unique")
        if len(self.router_tensors) != ROUTER_TENSOR_COUNT:
            raise ValueError("router overlay lineage requires exactly six tensors")
        if tuple(item.layer_index for item in self.router_tensors) != ROUTER_LAYER_INDICES:
            raise ValueError("router overlays must retain layers zero through five in order")
        overlay_names = tuple(item.name for item in self.router_tensors)
        if overlay_names != self.parent.router_tensor_names:
            raise ValueError("overlay names must exactly match the ordered parent router inventory")
        for item in self.router_tensors:
            if item.shape != self.parent.router_shape or item.dtype != self.parent.router_dtype:
                raise ValueError("router overlay shape/dtype differs from the parent contract")
        if self.unchanged_nonrouters.router_tensor_names != overlay_names:
            raise ValueError("non-router proof excludes a different router inventory")
        expected_nonrouter_count = self.parent.tensor_count - ROUTER_TENSOR_COUNT
        if self.unchanged_nonrouters.parent_nonrouter_tensor_count != expected_nonrouter_count:
            raise ValueError("non-router proof count differs from the exact parent inventory")
        expected_router_sha256 = _aggregate_prehashed(
            {item.name: item.overlay_tensor_sha256 for item in self.router_tensors}
        )
        if self.reload.candidate_router_sha256 != expected_router_sha256:
            raise ValueError("reload proof router digest differs from the overlay manifest")
        if self.parent.state_dict_sha256 == self.reload.candidate_state_sha256:
            raise ValueError("learned overlay must change the full candidate state digest")
        secret_path = sensitive_literal_path(self.model_dump(mode="json"))
        if secret_path is not None:
            raise ValueError(
                "router overlay lineage contains credential-like material at "
                + ".".join(secret_path)
            )
        return self


class RouterOverlayAudit(ImmutableLineageRecord):
    """Successful recomputation of all overlay and reload invariants."""

    lineage_sha256: str = Field(pattern=_SHA256_PATTERN)
    parent_state_sha256: str = Field(pattern=_SHA256_PATTERN)
    candidate_state_sha256: str = Field(pattern=_SHA256_PATTERN)
    reloaded_state_sha256: str = Field(pattern=_SHA256_PATTERN)
    router_tensor_count: Literal[6] = 6
    all_router_replacements_exact: Literal[True] = True
    all_nonrouter_tensors_unchanged: Literal[True] = True
    reload_exact: Literal[True] = True


def _checked_tensor(name: str, tensor: Tensor) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise RouterOverlayError(f"state entry {name!r} is not a torch.Tensor")
    if tensor.device.type == "meta":
        raise RouterOverlayError(f"state entry {name!r} has no materialized bytes")
    if tensor.layout is not torch.strided or tensor.is_quantized:
        raise RouterOverlayError(f"state entry {name!r} must be a dense non-quantized tensor")
    return tensor.detach().cpu().contiguous()


def _validate_names(state: TensorState) -> None:
    for name in state:
        if not isinstance(name, str) or not name or "\0" in name:
            raise RouterOverlayError("state tensor names must be non-empty strings without NUL")


def canonical_tensor_sha256(tensor: Tensor) -> str:
    """Hash dtype, shape, and exact contiguous CPU bytes for one tensor."""

    normalized = _checked_tensor("<tensor>", tensor)
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(normalized.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(normalized.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _aggregate_prehashed(tensor_hashes: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensor_hashes):
        if not name or "\0" in name:
            raise RouterOverlayError("aggregate tensor names must be non-empty and contain no NUL")
        tensor_sha256 = tensor_hashes[name]
        if len(tensor_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in tensor_sha256
        ):
            raise RouterOverlayError(f"invalid canonical tensor hash for {name!r}")
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(tensor_sha256))
    return digest.hexdigest()


def _subset_sha256(state: TensorState, names: Sequence[str]) -> str:
    _validate_names(state)
    missing = sorted(set(names) - set(state))
    if missing:
        raise RouterOverlayError("state is missing tensors: " + ", ".join(missing))
    return _aggregate_prehashed({name: canonical_tensor_sha256(state[name]) for name in names})


def state_dict_sha256(state: TensorState) -> str:
    """Hash a complete named tensor state independently of mapping iteration order."""

    _validate_names(state)
    return _subset_sha256(state, tuple(state))


def _require_router_names(router_names: Sequence[str]) -> tuple[str, ...]:
    resolved = tuple(router_names)
    if len(resolved) != ROUTER_TENSOR_COUNT or len(set(resolved)) != ROUTER_TENSOR_COUNT:
        raise RouterOverlayError("exactly six unique router tensor names are required")
    if any(not name or "\0" in name for name in resolved):
        raise RouterOverlayError("router tensor names must be non-empty and contain no NUL")
    return resolved


def _require_same_state_inventory(left: TensorState, right: TensorState) -> tuple[str, ...]:
    _validate_names(left)
    _validate_names(right)
    left_names = set(left)
    right_names = set(right)
    if left_names != right_names:
        missing = sorted(left_names - right_names)
        extra = sorted(right_names - left_names)
        raise RouterOverlayError(f"state inventories differ; missing={missing!r}, extra={extra!r}")
    return tuple(sorted(left_names))


def _build_router_records(
    parent_state: TensorState,
    overlay_state: TensorState,
    router_names: Sequence[str],
) -> tuple[RouterTensorOverlay, ...]:
    names = _require_router_names(router_names)
    if set(overlay_state) != set(names):
        missing = sorted(set(names) - set(overlay_state))
        extra = sorted(set(overlay_state) - set(names))
        raise RouterOverlayError(
            f"overlay must contain only the six routers; missing={missing!r}, extra={extra!r}"
        )
    records: list[RouterTensorOverlay] = []
    for layer_index, name in enumerate(names):
        if name not in parent_state:
            raise RouterOverlayError(f"parent state is missing router tensor {name!r}")
        parent = _checked_tensor(name, parent_state[name])
        overlay = _checked_tensor(name, overlay_state[name])
        if parent.shape != overlay.shape or parent.dtype != overlay.dtype:
            raise RouterOverlayError(f"router overlay shape/dtype mismatch for {name!r}")
        try:
            records.append(
                RouterTensorOverlay(
                    layer_index=layer_index,
                    name=name,
                    shape=tuple(parent.shape),
                    dtype=str(parent.dtype),
                    parent_tensor_sha256=canonical_tensor_sha256(parent),
                    overlay_tensor_sha256=canonical_tensor_sha256(overlay),
                )
            )
        except ValueError as error:
            raise RouterOverlayError(f"invalid router replacement {name!r}: {error}") from error
    return tuple(records)


def _apply_records(
    parent_state: TensorState,
    overlay_state: TensorState,
    records: Sequence[RouterTensorOverlay],
) -> dict[str, Tensor]:
    record_names = tuple(item.name for item in records)
    if set(overlay_state) != set(record_names):
        raise RouterOverlayError("overlay payload inventory differs from the lineage records")
    candidate = {name: tensor.detach().clone() for name, tensor in parent_state.items()}
    for record in records:
        if record.name not in parent_state:
            raise RouterOverlayError(f"parent state is missing router tensor {record.name!r}")
        parent = _checked_tensor(record.name, parent_state[record.name])
        overlay = _checked_tensor(record.name, overlay_state[record.name])
        if tuple(parent.shape) != record.shape or str(parent.dtype) != record.dtype:
            raise RouterOverlayError(f"parent router spec differs for {record.name!r}")
        if canonical_tensor_sha256(parent) != record.parent_tensor_sha256:
            raise RouterOverlayError(f"parent router hash differs for {record.name!r}")
        if tuple(overlay.shape) != record.shape or str(overlay.dtype) != record.dtype:
            raise RouterOverlayError(f"overlay router spec differs for {record.name!r}")
        if canonical_tensor_sha256(overlay) != record.overlay_tensor_sha256:
            raise RouterOverlayError(f"overlay router hash differs for {record.name!r}")
        candidate[record.name] = overlay_state[record.name].detach().clone()
    return candidate


def _build_unchanged_nonrouter_proof(
    parent_state: TensorState,
    candidate_state: TensorState,
    router_names: Sequence[str],
) -> UnchangedNonRouterProof:
    names = _require_same_state_inventory(parent_state, candidate_state)
    routers = _require_router_names(router_names)
    nonrouters = tuple(name for name in names if name not in set(routers))
    changed = tuple(
        name
        for name in nonrouters
        if canonical_tensor_sha256(parent_state[name])
        != canonical_tensor_sha256(candidate_state[name])
    )
    if changed:
        raise RouterOverlayError("non-router tensors changed: " + ", ".join(changed))
    parent_numel = sum(int(parent_state[name].numel()) for name in nonrouters)
    candidate_numel = sum(int(candidate_state[name].numel()) for name in nonrouters)
    return UnchangedNonRouterProof(
        router_tensor_names=routers,
        parent_nonrouter_tensor_count=len(nonrouters),
        candidate_nonrouter_tensor_count=len(nonrouters),
        parent_nonrouter_numel=parent_numel,
        candidate_nonrouter_numel=candidate_numel,
        parent_nonrouter_sha256=_subset_sha256(parent_state, nonrouters),
        candidate_nonrouter_sha256=_subset_sha256(candidate_state, nonrouters),
    )


def _build_reload_proof(
    candidate_state: TensorState,
    reloaded_state: TensorState,
    router_names: Sequence[str],
    *,
    overlay_bundle_sha256: str,
    loader_name: str,
    loader_version: str,
) -> ReloadProof:
    _require_same_state_inventory(candidate_state, reloaded_state)
    routers = _require_router_names(router_names)
    candidate_sha256 = state_dict_sha256(candidate_state)
    reloaded_sha256 = state_dict_sha256(reloaded_state)
    candidate_router_sha256 = _subset_sha256(candidate_state, routers)
    reloaded_router_sha256 = _subset_sha256(reloaded_state, routers)
    if candidate_sha256 != reloaded_sha256 or candidate_router_sha256 != reloaded_router_sha256:
        raise RouterOverlayError("reloaded state differs from the pre-save candidate")
    return ReloadProof(
        overlay_bundle_sha256=overlay_bundle_sha256,
        loader_name=loader_name,
        loader_version=loader_version,
        candidate_state_sha256=candidate_sha256,
        reloaded_state_sha256=reloaded_sha256,
        candidate_router_sha256=candidate_router_sha256,
        reloaded_router_sha256=reloaded_router_sha256,
    )


def _validate_parent_state(parent_state: TensorState, parent: ParentModelIdentity) -> None:
    if len(parent_state) != parent.tensor_count:
        raise RouterOverlayError("parent tensor count differs from its exact identity")
    if state_dict_sha256(parent_state) != parent.state_dict_sha256:
        raise RouterOverlayError("parent state digest differs from its exact identity")
    for name in parent.router_tensor_names:
        if name not in parent_state:
            raise RouterOverlayError(f"parent state is missing router tensor {name!r}")
        tensor = _checked_tensor(name, parent_state[name])
        if tuple(tensor.shape) != parent.router_shape or str(tensor.dtype) != parent.router_dtype:
            raise RouterOverlayError(f"parent router shape/dtype differs for {name!r}")


def build_router_overlay_lineage(
    *,
    parent: ParentModelIdentity,
    corpus_contract_sha256: str,
    corpus: Sequence[CorpusProvenance],
    training: TrainingRunProvenance,
    source: TrainingSourceProvenance,
    parent_state: TensorState,
    overlay_state: TensorState,
    reloaded_state: TensorState,
    overlay_bundle_sha256: str,
    loader_name: str,
    loader_version: str,
) -> RouterOverlayLineage:
    """Build complete lineage only after replacement, non-router, and reload proofs pass."""

    _validate_parent_state(parent_state, parent)
    records = _build_router_records(parent_state, overlay_state, parent.router_tensor_names)
    candidate = _apply_records(parent_state, overlay_state, records)
    unchanged = _build_unchanged_nonrouter_proof(
        parent_state, candidate, parent.router_tensor_names
    )
    reload_proof = _build_reload_proof(
        candidate,
        reloaded_state,
        parent.router_tensor_names,
        overlay_bundle_sha256=overlay_bundle_sha256,
        loader_name=loader_name,
        loader_version=loader_version,
    )
    return RouterOverlayLineage(
        parent=parent,
        corpus_contract_sha256=corpus_contract_sha256,
        corpus=tuple(corpus),
        training=training,
        source=source,
        router_tensors=records,
        unchanged_nonrouters=unchanged,
        reload=reload_proof,
    )


def apply_router_overlay(
    parent_state: TensorState,
    overlay_state: TensorState,
    lineage: RouterOverlayLineage,
) -> dict[str, Tensor]:
    """Apply a recorded six-router overlay after recomputing all input facts."""

    _validate_parent_state(parent_state, lineage.parent)
    candidate = _apply_records(parent_state, overlay_state, lineage.router_tensors)
    proof = _build_unchanged_nonrouter_proof(
        parent_state, candidate, lineage.parent.router_tensor_names
    )
    if proof != lineage.unchanged_nonrouters:
        raise RouterOverlayError("recomputed non-router proof differs from the lineage")
    candidate_sha256 = state_dict_sha256(candidate)
    if candidate_sha256 != lineage.reload.candidate_state_sha256:
        raise RouterOverlayError("applied candidate digest differs from the lineage reload proof")
    return candidate


def audit_router_overlay(
    parent_state: TensorState,
    overlay_state: TensorState,
    reloaded_state: TensorState,
    lineage: RouterOverlayLineage,
) -> RouterOverlayAudit:
    """Recompute parent, overlay, unchanged-nonrouter, and reload evidence."""

    candidate = apply_router_overlay(parent_state, overlay_state, lineage)
    reload_proof = _build_reload_proof(
        candidate,
        reloaded_state,
        lineage.parent.router_tensor_names,
        overlay_bundle_sha256=lineage.reload.overlay_bundle_sha256,
        loader_name=lineage.reload.loader_name,
        loader_version=lineage.reload.loader_version,
    )
    if reload_proof != lineage.reload:
        raise RouterOverlayError("recomputed reload proof differs from the lineage")
    return RouterOverlayAudit(
        lineage_sha256=lineage.sha256(),
        parent_state_sha256=state_dict_sha256(parent_state),
        candidate_state_sha256=state_dict_sha256(candidate),
        reloaded_state_sha256=state_dict_sha256(reloaded_state),
    )


__all__ = [
    "CorpusProvenance",
    "ParentModelIdentity",
    "ReloadProof",
    "RouterOverlayAudit",
    "RouterOverlayError",
    "RouterOverlayLineage",
    "RouterTensorOverlay",
    "TrainingRunProvenance",
    "TrainingSourceProvenance",
    "UnchangedNonRouterProof",
    "apply_router_overlay",
    "audit_router_overlay",
    "build_router_overlay_lineage",
    "canonical_tensor_sha256",
    "state_dict_sha256",
]
