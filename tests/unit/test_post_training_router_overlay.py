"""CPU contracts for exact learned-router overlay lineage and reload audit."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor

from inkling_quant_lab.post_training.lineage import (
    CorpusProvenance,
    ParentModelIdentity,
    ReloadProof,
    RouterOverlayError,
    RouterOverlayLineage,
    TrainingRunProvenance,
    TrainingSourceProvenance,
    UnchangedNonRouterProof,
    apply_router_overlay,
    audit_router_overlay,
    build_router_overlay_lineage,
    canonical_tensor_sha256,
    state_dict_sha256,
)

pytestmark = pytest.mark.unit

ROUTER_NAMES = tuple(f"model.layers.{layer}.block_sparse_moe.gate.weight" for layer in range(6))


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _parent_state() -> dict[str, Tensor]:
    state = {
        name: torch.arange(12, dtype=torch.float32).reshape(3, 4) + float(layer * 12)
        for layer, name in enumerate(ROUTER_NAMES)
    }
    state["model.embed_tokens.weight"] = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    state["lm_head.weight"] = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    return state


def _overlay_state(parent: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {
        name: parent[name] + float(layer + 1) / 100.0 for layer, name in enumerate(ROUTER_NAMES)
    }


def _candidate_state(
    parent: Mapping[str, Tensor], overlay: Mapping[str, Tensor]
) -> dict[str, Tensor]:
    candidate = {name: tensor.detach().clone() for name, tensor in parent.items()}
    candidate.update({name: tensor.detach().clone() for name, tensor in overlay.items()})
    return candidate


def _parent_identity(state: Mapping[str, Tensor]) -> ParentModelIdentity:
    return ParentModelIdentity(
        model_id="ggml-org/stories15M_MOE",
        revision="a" * 40,
        architecture="MixtralForCausalLM",
        resolved_class="transformers.MixtralForCausalLM",
        config_sha256=_digest("config"),
        tokenizer_sha256=_digest("tokenizer"),
        weights_sha256=_digest("weights"),
        state_dict_sha256=state_dict_sha256(state),
        tensor_count=len(state),
        router_tensor_names=ROUTER_NAMES,
        router_shape=(3, 4),
        router_dtype="torch.float32",
    )


def _corpus() -> tuple[CorpusProvenance, ...]:
    return (
        CorpusProvenance(
            corpus_id="public-domain-train",
            role="train",
            dataset_id="local://public-domain-corpus",
            revision="corpus-v1",
            declared_license="public-domain-fixture",
            source_size_bytes=100,
            source_sha256=_digest("corpus-source"),
            parser_version="fixture-parser-v1",
            split="train",
            content_sha256=_digest("train-content"),
            ordered_sample_ids_sha256=_digest("train-sample-ids"),
            sample_count=80,
            token_count=8_000,
        ),
        CorpusProvenance(
            corpus_id="public-domain-validation",
            role="validation",
            dataset_id="local://public-domain-corpus",
            revision="corpus-v1",
            declared_license="public-domain-fixture",
            source_size_bytes=100,
            source_sha256=_digest("corpus-source"),
            parser_version="fixture-parser-v1",
            split="validation",
            content_sha256=_digest("validation-content"),
            ordered_sample_ids_sha256=_digest("validation-sample-ids"),
            sample_count=10,
            token_count=1_000,
        ),
    )


def _training() -> TrainingRunProvenance:
    return TrainingRunProvenance(
        run_id="router-overlay-seed-17",
        seed=17,
        steps=500,
        batch_size=3,
        sequence_length=128,
        optimizer="adamw",
        learning_rate=1e-4,
        optimizer_betas=(0.9, 0.999),
        optimizer_epsilon=1e-8,
        weight_decay=0.0,
        bias_correction=False,
        training_config_sha256=_digest("resolved-training-config"),
        domain_pair_config_sha256=_digest("domain-pair-config"),
    )


def _source() -> TrainingSourceProvenance:
    return TrainingSourceProvenance(
        repository="local://inkling-quant-lab",
        repository_revision="b" * 40,
        repository_dirty=False,
        entrypoint="scripts/train_router_overlay.py",
        entrypoint_sha256=_digest("entrypoint"),
        source_bundle_sha256=_digest("source-bundle"),
        environment_sha256=_digest("environment"),
        dependency_lock_sha256=_digest("uv-lock"),
        framework="reference-cpu",
        framework_version="1.0",
    )


def _lineage_fixture() -> tuple[
    dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], RouterOverlayLineage
]:
    parent = _parent_state()
    overlay = _overlay_state(parent)
    candidate = _candidate_state(parent, overlay)
    lineage = build_router_overlay_lineage(
        parent=_parent_identity(parent),
        corpus_contract_sha256=_digest("corpus-contract"),
        corpus=_corpus(),
        training=_training(),
        source=_source(),
        parent_state=parent,
        overlay_state=overlay,
        reloaded_state=candidate,
        overlay_bundle_sha256=_digest("overlay.safetensors"),
        loader_name="safetensors.torch.load_file",
        loader_version="0.8.0",
    )
    return parent, overlay, candidate, lineage


def test_lineage_records_exact_six_router_and_provenance_contracts() -> None:
    parent, overlay, candidate, lineage = _lineage_fixture()

    assert lineage.schema_version == "router-overlay-lineage-v1"
    assert lineage.corpus_contract_sha256 == _digest("corpus-contract")
    assert all(item.source_sha256 == _digest("corpus-source") for item in lineage.corpus)
    assert all(item.declared_license == "public-domain-fixture" for item in lineage.corpus)
    assert lineage.training.optimizer_betas == (0.9, 0.999)
    assert lineage.training.optimizer_epsilon == 1e-8
    assert lineage.training.weight_decay == 0.0
    assert lineage.training.bias_correction is False
    assert tuple(item.layer_index for item in lineage.router_tensors) == tuple(range(6))
    assert tuple(item.name for item in lineage.router_tensors) == ROUTER_NAMES
    assert all(item.shape == (3, 4) for item in lineage.router_tensors)
    assert all(item.dtype == "torch.float32" for item in lineage.router_tensors)
    assert tuple(item.overlay_tensor_sha256 for item in lineage.router_tensors) == tuple(
        canonical_tensor_sha256(overlay[name]) for name in ROUTER_NAMES
    )
    assert tuple(item.parent_tensor_sha256 for item in lineage.router_tensors) == tuple(
        canonical_tensor_sha256(parent[name]) for name in ROUTER_NAMES
    )
    assert lineage.unchanged_nonrouters.parent_nonrouter_tensor_count == 2
    assert lineage.unchanged_nonrouters.parent_nonrouter_numel == 44
    assert lineage.reload.serialization_format == "safetensors"
    assert lineage.reload.pickle_used is False
    assert lineage.reload.trust_remote_code is False
    assert lineage.reload.candidate_state_sha256 == state_dict_sha256(candidate)
    assert {item.role for item in lineage.corpus} == {"train", "validation"}


def test_lineage_is_frozen_strict_canonical_and_round_trips() -> None:
    _, _, _, lineage = _lineage_fixture()

    assert lineage.sha256() == hashlib.sha256(lineage.canonical_json().encode()).hexdigest()
    assert RouterOverlayLineage.model_validate_json(lineage.canonical_json()) == lineage
    with pytest.raises(ValidationError):
        RouterOverlayLineage.model_validate({**lineage.model_dump(), "unexpected": True})
    with pytest.raises(ValidationError):
        lineage.training.steps = 1  # type: ignore[misc]


@pytest.mark.parametrize("betas", ((-0.1, 0.999), (0.9, 1.0)))
def test_training_lineage_rejects_invalid_optimizer_betas(
    betas: tuple[float, float],
) -> None:
    payload = _training().model_dump()
    payload["optimizer_betas"] = betas

    with pytest.raises(ValidationError, match="optimizer betas"):
        TrainingRunProvenance.model_validate(payload)


def test_apply_clones_state_and_audit_recomputes_all_proofs() -> None:
    parent, overlay, candidate, lineage = _lineage_fixture()

    applied = apply_router_overlay(parent, overlay, lineage)
    assert state_dict_sha256(applied) == state_dict_sha256(candidate)
    assert state_dict_sha256(parent) == lineage.parent.state_dict_sha256
    for name in ROUTER_NAMES:
        assert torch.equal(applied[name], overlay[name])
        assert applied[name].data_ptr() != overlay[name].data_ptr()
    audit = audit_router_overlay(parent, overlay, candidate, lineage)
    assert audit.lineage_sha256 == lineage.sha256()
    assert audit.parent_state_sha256 == lineage.parent.state_dict_sha256
    assert audit.candidate_state_sha256 == lineage.reload.candidate_state_sha256
    assert audit.reloaded_state_sha256 == lineage.reload.reloaded_state_sha256
    assert audit.router_tensor_count == 6
    assert audit.all_router_replacements_exact is True
    assert audit.all_nonrouter_tensors_unchanged is True
    assert audit.reload_exact is True


def test_tensor_and_state_hashes_bind_dtype_shape_bytes_names_and_not_mapping_order() -> None:
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    assert canonical_tensor_sha256(tensor) == canonical_tensor_sha256(tensor.clone())
    assert canonical_tensor_sha256(tensor) != canonical_tensor_sha256(tensor.reshape(2, 1))
    assert canonical_tensor_sha256(tensor) != canonical_tensor_sha256(tensor.to(torch.float64))

    state = {"b": tensor + 1, "a": tensor}
    reversed_state = {"a": tensor.clone(), "b": tensor + 1}
    renamed_state = {"a": tensor.clone(), "c": tensor + 1}
    assert state_dict_sha256(state) == state_dict_sha256(reversed_state)
    assert state_dict_sha256(state) != state_dict_sha256(renamed_state)


@pytest.mark.parametrize("kind", ("sparse", "meta", "quantized"))
def test_tensor_hash_rejects_non_materialized_or_non_dense_payloads(kind: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "sparse":
            bad_tensor = torch.sparse_coo_tensor(
                indices=[[0]], values=[1.0], size=(2,), check_invariants=False
            )
        elif kind == "meta":
            bad_tensor = torch.empty(2, device="meta")
        else:
            bad_tensor = torch.quantize_per_tensor(
                torch.tensor([1.0]), scale=0.1, zero_point=0, dtype=torch.qint8
            )
    with pytest.raises(RouterOverlayError):
        canonical_tensor_sha256(bad_tensor)


def test_state_hash_rejects_non_tensor_and_invalid_names() -> None:
    with pytest.raises(RouterOverlayError, match=r"not a torch\.Tensor"):
        state_dict_sha256({"value": "not-a-tensor"})  # type: ignore[dict-item]
    with pytest.raises(RouterOverlayError, match="names"):
        state_dict_sha256({"bad\0name": torch.ones(1)})


@pytest.mark.parametrize("mode", ("missing", "extra"))
def test_build_rejects_overlay_inventory_other_than_exact_six(mode: str) -> None:
    parent = _parent_state()
    overlay = _overlay_state(parent)
    if mode == "missing":
        overlay.pop(ROUTER_NAMES[-1])
    else:
        overlay["lm_head.weight"] = parent["lm_head.weight"].clone()

    with pytest.raises(RouterOverlayError, match="only the six routers"):
        build_router_overlay_lineage(
            parent=_parent_identity(parent),
            corpus_contract_sha256=_digest("corpus-contract"),
            corpus=_corpus(),
            training=_training(),
            source=_source(),
            parent_state=parent,
            overlay_state=overlay,
            reloaded_state=_candidate_state(parent, overlay),
            overlay_bundle_sha256=_digest("bundle"),
            loader_name="loader",
            loader_version="1",
        )


@pytest.mark.parametrize("mode", ("shape", "dtype", "unchanged"))
def test_build_rejects_invalid_router_replacement(mode: str) -> None:
    parent = _parent_state()
    overlay = _overlay_state(parent)
    if mode == "shape":
        overlay[ROUTER_NAMES[0]] = torch.ones(4, 3)
    elif mode == "dtype":
        overlay[ROUTER_NAMES[0]] = overlay[ROUTER_NAMES[0]].to(torch.float64)
    else:
        overlay[ROUTER_NAMES[0]] = parent[ROUTER_NAMES[0]].clone()

    with pytest.raises(RouterOverlayError, match=r"replacement|shape/dtype"):
        build_router_overlay_lineage(
            parent=_parent_identity(parent),
            corpus_contract_sha256=_digest("corpus-contract"),
            corpus=_corpus(),
            training=_training(),
            source=_source(),
            parent_state=parent,
            overlay_state=overlay,
            reloaded_state=_candidate_state(parent, overlay),
            overlay_bundle_sha256=_digest("bundle"),
            loader_name="loader",
            loader_version="1",
        )


@pytest.mark.parametrize("changed_name", (ROUTER_NAMES[2], "lm_head.weight"))
def test_build_rejects_reload_that_differs_from_candidate(changed_name: str) -> None:
    parent = _parent_state()
    overlay = _overlay_state(parent)
    reloaded = _candidate_state(parent, overlay)
    reloaded[changed_name] = reloaded[changed_name] + 1.0

    with pytest.raises(RouterOverlayError, match="reloaded state differs"):
        build_router_overlay_lineage(
            parent=_parent_identity(parent),
            corpus_contract_sha256=_digest("corpus-contract"),
            corpus=_corpus(),
            training=_training(),
            source=_source(),
            parent_state=parent,
            overlay_state=overlay,
            reloaded_state=reloaded,
            overlay_bundle_sha256=_digest("bundle"),
            loader_name="loader",
            loader_version="1",
        )


def test_apply_rejects_wrong_parent_or_overlay_bytes() -> None:
    parent, overlay, _, lineage = _lineage_fixture()
    wrong_parent = {name: tensor.clone() for name, tensor in parent.items()}
    wrong_parent[ROUTER_NAMES[0]] += 1.0
    with pytest.raises(RouterOverlayError, match="parent state digest"):
        apply_router_overlay(wrong_parent, overlay, lineage)

    wrong_overlay = {name: tensor.clone() for name, tensor in overlay.items()}
    wrong_overlay[ROUTER_NAMES[0]] += 1.0
    with pytest.raises(RouterOverlayError, match="overlay router hash"):
        apply_router_overlay(parent, wrong_overlay, lineage)


def test_audit_rejects_post_reload_mutation() -> None:
    parent, overlay, candidate, lineage = _lineage_fixture()
    candidate["lm_head.weight"] += 1.0

    with pytest.raises(RouterOverlayError, match="reloaded state differs"):
        audit_router_overlay(parent, overlay, candidate, lineage)


def test_parent_identity_rejects_nonexact_revision_and_router_inventory() -> None:
    state = _parent_state()
    payload = _parent_identity(state).model_dump()
    payload["revision"] = "main"
    with pytest.raises(ValidationError):
        ParentModelIdentity.model_validate(payload)
    payload = _parent_identity(state).model_dump()
    payload["router_tensor_names"] = ROUTER_NAMES[:-1]
    with pytest.raises(ValidationError, match="exactly six"):
        ParentModelIdentity.model_validate(payload)


def test_lineage_rejects_missing_training_corpus_duplicate_corpus_and_secret() -> None:
    _, _, _, lineage = _lineage_fixture()
    payload = lineage.model_dump(mode="json")
    payload["corpus"] = [
        {**payload["corpus"][1], "role": "validation"},
    ]
    with pytest.raises(ValidationError, match="training corpus"):
        RouterOverlayLineage.model_validate(payload)

    payload = lineage.model_dump(mode="json")
    payload["corpus"][1]["corpus_id"] = payload["corpus"][0]["corpus_id"]
    with pytest.raises(ValidationError, match="unique"):
        RouterOverlayLineage.model_validate(payload)

    payload = lineage.model_dump(mode="json")
    payload["parent"]["model_id"] = "https://example.invalid/model?token=super-secret"
    with pytest.raises(ValidationError, match="credential-like"):
        RouterOverlayLineage.model_validate(payload)


def test_lineage_rejects_reordered_or_tampered_router_contract() -> None:
    _, _, _, lineage = _lineage_fixture()
    payload = lineage.model_dump(mode="json")
    payload["router_tensors"] = list(reversed(payload["router_tensors"]))
    with pytest.raises(ValidationError, match="layers zero through five"):
        RouterOverlayLineage.model_validate(payload)

    payload = lineage.model_dump(mode="json")
    payload["router_tensors"][0]["overlay_tensor_sha256"] = _digest("tampered")
    with pytest.raises(ValidationError, match="reload proof router digest"):
        RouterOverlayLineage.model_validate(payload)


def test_unchanged_and_reload_proofs_reject_false_equality_claims() -> None:
    _, _, _, lineage = _lineage_fixture()
    unchanged: dict[str, Any] = lineage.unchanged_nonrouters.model_dump()
    unchanged["candidate_nonrouter_sha256"] = _digest("changed")
    with pytest.raises(ValidationError, match="aggregate hashes differ"):
        UnchangedNonRouterProof.model_validate(unchanged)

    reload: dict[str, Any] = lineage.reload.model_dump()
    reload["reloaded_state_sha256"] = _digest("changed")
    with pytest.raises(ValidationError, match="reloaded candidate state differs"):
        ReloadProof.model_validate(reload)


@pytest.mark.parametrize(
    ("field", "value"),
    (("pickle_used", True), ("trust_remote_code", True), ("serialization_format", "pickle")),
)
def test_reload_proof_cannot_claim_unsafe_reload(field: str, value: object) -> None:
    _, _, _, lineage = _lineage_fixture()
    payload = lineage.reload.model_dump()
    payload[field] = value
    with pytest.raises(ValidationError):
        ReloadProof.model_validate(payload)
