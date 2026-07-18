"""CPU contracts for the packed six-gate learned-router overlay."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError
from torch import Tensor

from inkling_quant_lab.post_training.lineage import (
    CorpusProvenance,
    ParentModelIdentity,
    RouterOverlayLineage,
    TrainingRunProvenance,
    TrainingSourceProvenance,
    build_router_overlay_lineage,
    state_dict_sha256,
)
from inkling_quant_lab.post_training.quantized_router import (
    ROUTER_SHAPE,
    ROUTER_WEIGHT_NAMES,
    QuantizedRouterOverlayError,
    QuantizedRouterOverlayManifest,
    affine_dequantize_router,
    affine_quantize_router,
    affine_rounding_error_bound,
    audit_quantized_router_overlay,
    build_quantized_router_overlay,
    expected_quantized_router_bytes,
    tensor_payload_sha256,
)

pytestmark = pytest.mark.unit


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _learned_fixture() -> tuple[dict[str, Tensor], RouterOverlayLineage]:
    generator = torch.Generator().manual_seed(2718)
    parent = {
        name: torch.randn(ROUTER_SHAPE, generator=generator, dtype=torch.float32)
        for name in ROUTER_WEIGHT_NAMES
    }
    parent["model.embed_tokens.weight"] = torch.randn(
        (8, ROUTER_SHAPE[1]), generator=generator, dtype=torch.float32
    )
    learned = {
        name: parent[name] + (layer + 1) * 0.003 for layer, name in enumerate(ROUTER_WEIGHT_NAMES)
    }
    candidate = {name: tensor.clone() for name, tensor in parent.items()}
    candidate.update({name: tensor.clone() for name, tensor in learned.items()})
    identity = ParentModelIdentity(
        model_id="local/inkling-compatible-moe",
        revision="a" * 40,
        architecture="InklingMoEForCausalLM",
        resolved_class="inkling.InklingMoEForCausalLM",
        config_sha256=_digest("config"),
        tokenizer_sha256=_digest("tokenizer"),
        weights_sha256=_digest("weights"),
        state_dict_sha256=state_dict_sha256(parent),
        tensor_count=len(parent),
        router_tensor_names=ROUTER_WEIGHT_NAMES,
        router_shape=ROUTER_SHAPE,
        router_dtype="torch.float32",
    )
    corpus = (
        CorpusProvenance(
            corpus_id="router-train",
            role="train",
            dataset_id="local://router-corpus",
            revision="v1",
            declared_license="fixture-license",
            source_size_bytes=100,
            source_sha256=_digest("corpus-source"),
            parser_version="fixture-parser-v1",
            split="train",
            content_sha256=_digest("content"),
            ordered_sample_ids_sha256=_digest("samples"),
            sample_count=12,
            token_count=1_200,
        ),
    )
    training = TrainingRunProvenance(
        run_id="learned-router-2718",
        seed=2718,
        steps=10,
        batch_size=2,
        sequence_length=64,
        optimizer="adamw",
        learning_rate=1e-4,
        optimizer_betas=(0.9, 0.999),
        optimizer_epsilon=1e-8,
        weight_decay=0.0,
        bias_correction=False,
        training_config_sha256=_digest("training-config"),
        domain_pair_config_sha256=_digest("domain-config"),
    )
    source = TrainingSourceProvenance(
        repository="local://inkling-quant-lab",
        repository_revision="b" * 40,
        repository_dirty=False,
        entrypoint="scripts/train_router_overlay.py",
        entrypoint_sha256=_digest("entrypoint"),
        source_bundle_sha256=_digest("source"),
        environment_sha256=_digest("environment"),
        dependency_lock_sha256=_digest("lock"),
        framework="torch-cpu",
        framework_version=torch.__version__,
    )
    lineage = build_router_overlay_lineage(
        parent=identity,
        corpus_contract_sha256=_digest("corpus-contract"),
        corpus=corpus,
        training=training,
        source=source,
        parent_state=parent,
        overlay_state=learned,
        reloaded_state=candidate,
        overlay_bundle_sha256=_digest("learned-router.safetensors"),
        loader_name="safetensors.torch.load_file",
        loader_version="0.8.0",
    )
    return learned, lineage


@pytest.mark.parametrize(
    ("bits", "packed_shape", "raw_bytes"),
    ((4, (4, 36), 5_184), (8, (4, 72), 8_640)),
)
def test_reference_affine_pack_shapes_dtypes_bytes_and_error_bound(
    bits: int, packed_shape: tuple[int, int], raw_bytes: int
) -> None:
    generator = torch.Generator().manual_seed(bits)
    weight = torch.randn(ROUTER_SHAPE, generator=generator, dtype=torch.float32)

    quantized = affine_quantize_router(weight, bits=bits)
    restored = affine_dequantize_router(
        quantized.weight, quantized.scales, quantized.biases, bits=bits
    )

    assert quantized.weight.shape == packed_shape
    assert quantized.weight.dtype is torch.uint32
    assert quantized.scales.shape == (4, 9)
    assert quantized.scales.dtype is torch.float32
    assert quantized.biases.shape == (4, 9)
    assert quantized.biases.dtype is torch.float32
    assert restored.shape == ROUTER_SHAPE
    assert restored.dtype is torch.float32
    assert float((weight - restored).abs().amax()) <= affine_rounding_error_bound(
        quantized.scales, weight
    )
    assert expected_quantized_router_bytes(bits) == raw_bytes


def test_packing_is_low_bit_first_and_constant_groups_are_finite() -> None:
    weight = torch.zeros(ROUTER_SHAPE, dtype=torch.float32)
    weight[:, :32] = torch.arange(32, dtype=torch.float32)

    quantized = affine_quantize_router(weight, bits=4)
    restored = affine_dequantize_router(
        quantized.weight, quantized.scales, quantized.biases, bits=4
    )

    first_codes = (
        torch.round((weight[0, :8] - quantized.biases[0, 0]) / quantized.scales[0, 0])
        .clamp(0, 15)
        .to(torch.int64)
    )
    expected_word = sum(int(code) << (position * 4) for position, code in enumerate(first_codes))
    assert int(quantized.weight[0, 0]) == expected_word
    assert torch.isfinite(quantized.scales).all()
    assert torch.isfinite(quantized.biases).all()
    assert torch.equal(restored[:, 32:], weight[:, 32:])


@pytest.mark.parametrize(("bits", "expected_bytes"), ((4, 5_184), (8, 8_640)))
def test_overlay_manifest_has_exact_eighteen_tensor_inventory_and_safe_recipe(
    bits: int, expected_bytes: int
) -> None:
    learned, lineage = _learned_fixture()

    tensors, manifest = build_quantized_router_overlay(learned, lineage, bits=bits)
    audit = audit_quantized_router_overlay(
        tensors,
        manifest,
        learned_router_state=learned,
        learned_float_lineage=lineage,
    )

    assert len(tensors) == 18
    assert len(manifest.tensor_inventory) == 18
    assert len(manifest.routers) == 6
    assert manifest.raw_tensor_bytes == expected_bytes
    assert sum(tensor.numel() * tensor.element_size() for tensor in tensors.values()) == (
        expected_bytes
    )
    assert manifest.parent_learned_float_lineage_sha256 == lineage.sha256()
    assert manifest.learned_float_overlay_bundle_sha256 == lineage.reload.overlay_bundle_sha256
    assert tuple(item.learned_float_weight_sha256 for item in manifest.routers) == tuple(
        item.overlay_tensor_sha256 for item in lineage.router_tensors
    )
    assert {item.role for item in manifest.tensor_inventory} == {
        "packed_weight",
        "scales",
        "affine_biases",
    }
    assert manifest.reconstruction.module_type == "mlx.nn.QuantizedLinear"
    assert manifest.reconstruction.input_dims == 288
    assert manifest.reconstruction.output_dims == 4
    assert manifest.reconstruction.linear_bias is False
    assert manifest.reconstruction.trust_remote_code is False
    assert manifest.reconstruction.pickle_used is False
    assert manifest.reconstruction.python_source_in_bundle is False
    assert audit.tensor_inventory_exact is True
    assert audit.raw_tensor_bytes == expected_bytes


def test_manifest_serialization_and_payload_hash_are_canonical_and_frozen() -> None:
    learned, lineage = _learned_fixture()
    tensors, manifest = build_quantized_router_overlay(learned, lineage, bits=4)
    reversed_hashes = {
        item.name: item.tensor_sha256 for item in reversed(manifest.tensor_inventory)
    }

    assert manifest.sha256() == hashlib.sha256(manifest.canonical_json().encode()).hexdigest()
    assert QuantizedRouterOverlayManifest.model_validate_json(manifest.canonical_json()) == manifest
    assert tensor_payload_sha256(reversed_hashes) == manifest.tensor_payload_sha256
    assert (
        tensor_payload_sha256(
            {
                name: next(
                    item.tensor_sha256 for item in manifest.tensor_inventory if item.name == name
                )
                for name in tensors
            }
        )
        == manifest.tensor_payload_sha256
    )
    with pytest.raises(ValidationError):
        manifest.bits = 8  # type: ignore[misc]
    with pytest.raises(ValidationError):
        QuantizedRouterOverlayManifest.model_validate(
            {**manifest.model_dump(), "raw_tensor_bytes": 5_183}
        )


@pytest.mark.parametrize("mode", ("tamper", "missing", "extra"))
def test_audit_rejects_tampered_missing_and_extra_quantized_tensors(mode: str) -> None:
    learned, lineage = _learned_fixture()
    tensors, manifest = build_quantized_router_overlay(learned, lineage, bits=4)
    invalid = {name: tensor.clone() for name, tensor in tensors.items()}
    if mode == "tamper":
        name = ROUTER_WEIGHT_NAMES[0]
        value = int(invalid[name][0, 0].item()) ^ 1
        invalid[name][0, 0] = value
        expected = "spec, bytes, or hash"
    elif mode == "missing":
        invalid.pop(next(iter(invalid)))
        expected = "inventory differs"
    else:
        invalid["model.layers.6.block_sparse_moe.gate.weight"] = torch.zeros(
            (4, 36), dtype=torch.uint32
        )
        expected = "inventory differs"

    with pytest.raises(QuantizedRouterOverlayError, match=expected):
        audit_quantized_router_overlay(
            invalid,
            manifest,
            learned_router_state=learned,
            learned_float_lineage=lineage,
        )


def test_build_and_audit_reject_wrong_learned_payload_or_lineage_hash() -> None:
    learned, lineage = _learned_fixture()
    tensors, manifest = build_quantized_router_overlay(learned, lineage, bits=8)
    tampered_learned = {name: tensor.clone() for name, tensor in learned.items()}
    tampered_learned[ROUTER_WEIGHT_NAMES[2]][0, 0] += 1.0

    with pytest.raises(QuantizedRouterOverlayError, match="learned router hash"):
        audit_quantized_router_overlay(
            tensors,
            manifest,
            learned_router_state=tampered_learned,
            learned_float_lineage=lineage,
        )
    bad_manifest = manifest.model_copy(
        update={"parent_learned_float_lineage_sha256": _digest("wrong-lineage")}
    )
    with pytest.raises(QuantizedRouterOverlayError, match="lineage hash"):
        audit_quantized_router_overlay(
            tensors,
            bad_manifest,
            learned_router_state=learned,
            learned_float_lineage=lineage,
        )


@pytest.mark.parametrize("bits", (0, 3, 6, 16, True))
def test_reference_quantizer_rejects_unsupported_bits(bits: int) -> None:
    with pytest.raises(QuantizedRouterOverlayError, match="exactly 4 or 8"):
        affine_quantize_router(torch.zeros(ROUTER_SHAPE), bits=bits)


def test_build_requires_exact_learned_inventory_shape_and_dtype() -> None:
    learned, lineage = _learned_fixture()
    cases: list[Mapping[str, Tensor]] = []
    missing = dict(learned)
    missing.pop(ROUTER_WEIGHT_NAMES[-1])
    cases.append(missing)
    extra = dict(learned)
    extra["extra"] = torch.zeros(ROUTER_SHAPE)
    cases.append(extra)
    wrong_shape = dict(learned)
    wrong_shape[ROUTER_WEIGHT_NAMES[0]] = torch.zeros((288, 4))
    cases.append(wrong_shape)
    wrong_dtype = dict(learned)
    wrong_dtype[ROUTER_WEIGHT_NAMES[0]] = learned[ROUTER_WEIGHT_NAMES[0]].double()
    cases.append(wrong_dtype)

    for case in cases:
        with pytest.raises(QuantizedRouterOverlayError):
            build_quantized_router_overlay(case, lineage, bits=4)


def test_quantized_router_module_imports_without_mlx_or_mlx_lm() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    command = (
        "import sys; "
        "import inkling_quant_lab.post_training.quantized_router; "
        "assert 'mlx' not in sys.modules; "
        "assert 'mlx_lm' not in sys.modules"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=source_root.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
