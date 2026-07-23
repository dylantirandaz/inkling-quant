from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest
from pydantic import ValidationError

from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.gguf.mtp import (
    MTP_ATTENTION_PATTERN,
    MTP_HEAD_COUNT,
    MTP_LOCAL_LAYER_IDS,
    MTP_SHARED_SOURCE_DEPENDENCIES,
    MTP_SOURCE_TENSOR_COUNT,
    MTP_SOURCE_TENSORS_PER_HEAD,
    MTP_TRUNK_LAYER_COUNT,
    PINNED_MTP_HEADER_BYTES,
    PINNED_MTP_HEADER_INVENTORY_SHA256,
    PINNED_MTP_HEADER_JSON_SHA256,
    PINNED_MTP_HEADER_PREFIX_BYTES,
    PINNED_MTP_HEADER_PREFIX_SHA256,
    PINNED_MTP_MODEL_ID,
    PINNED_MTP_REVISION,
    PINNED_MTP_SHARD,
    PINNED_MTP_SHARD_BYTES,
    PINNED_MTP_SHARD_SHA256,
    MTPSourceAudit,
    MTPVariantContract,
    audit_mtp_source_inventory,
    build_mtp_format_receipt,
    build_mtp_mapping_contract,
    deinterleave_dense_w13,
    expected_mtp_safetensors_header,
    expected_mtp_safetensors_header_prefix,
    expected_mtp_source_specs,
    parse_pinned_mtp_safetensors_header,
)

# Pinned llama.cpp a01540948fbb3b361bdd230fc165f8a72f846858 uses
# `blk.{bid}.ffn_gscale` in MODEL_TENSOR.FFN_GSCALE and applies the default
# `.weight` suffix in conversion/base.py::format_tensor_name.
_PINNED_LLAMA_CPP_FFN_GSCALE_PATTERN = "blk.{bid}.ffn_gscale"
_PINNED_LLAMA_CPP_DEFAULT_TENSOR_SUFFIX = ".weight"


def _weight_map() -> dict[str, object]:
    result: dict[str, object] = {
        spec.name: PINNED_MTP_SHARD for spec in expected_mtp_source_specs()
    }
    result.update(
        {
            "model.llm.embed.weight": "model-00001-of-00108.safetensors",
            "model.llm.embed_norm.weight": "model-00001-of-00108.safetensors",
            "model.llm.unembed.weight": "model-00108-of-00108.safetensors",
        }
    )
    return result


def _audit(
    *,
    header: Mapping[str, object] | None = None,
    weight_map: Mapping[str, object] | None = None,
    **identity: object,
) -> MTPSourceAudit:
    arguments: dict[str, object] = {
        "model_id": PINNED_MTP_MODEL_ID,
        "revision": PINNED_MTP_REVISION,
        "shard_name": PINNED_MTP_SHARD,
        "shard_size_bytes": PINNED_MTP_SHARD_BYTES,
        "shard_sha256": PINNED_MTP_SHARD_SHA256,
        "header_size_bytes": PINNED_MTP_HEADER_BYTES,
        "header_prefix_sha256": PINNED_MTP_HEADER_PREFIX_SHA256,
        "weight_map": _weight_map() if weight_map is None else weight_map,
        "safetensors_header": expected_mtp_safetensors_header() if header is None else header,
    }
    arguments.update(identity)
    return audit_mtp_source_inventory(**arguments)  # type: ignore[arg-type]


def test_tc_inkling_mtp_001_exact_source_inventory_is_8_by_20_bf16() -> None:
    specs = expected_mtp_source_specs()
    audit = _audit()

    assert len(specs) == MTP_SOURCE_TENSOR_COUNT == 160
    assert {spec.dtype for spec in specs} == {"BF16"}
    assert [sum(spec.head_index == head for spec in specs) for head in range(MTP_HEAD_COUNT)] == [
        MTP_SOURCE_TENSORS_PER_HEAD
    ] * MTP_HEAD_COUNT
    assert sum(spec.nbytes for spec in specs) == audit.payload_size_bytes
    assert audit.payload_size_bytes + 8 + PINNED_MTP_HEADER_BYTES == PINNED_MTP_SHARD_BYTES


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("model_id", "other/model"),
        ("revision", "0" * 40),
        ("shard_name", "other.safetensors"),
        ("shard_size_bytes", PINNED_MTP_SHARD_BYTES - 1),
        ("shard_sha256", "0" * 64),
        ("header_size_bytes", PINNED_MTP_HEADER_BYTES - 1),
        ("header_prefix_sha256", "0" * 64),
    ],
)
def test_tc_inkling_mtp_002_shard_identity_is_exact(field: str, bad_value: object) -> None:
    with pytest.raises(ConfigurationError, match="identity mismatch"):
        _audit(**{field: bad_value})  # type: ignore[arg-type]


def test_tc_inkling_mtp_003_local_global_pattern_controls_exact_shapes() -> None:
    specs = {spec.name: spec for spec in expected_mtp_source_specs()}

    assert MTP_ATTENTION_PATTERN == (
        "local",
        "global",
        "local",
        "global",
        "local",
        "local",
        "local",
        "local",
    )
    assert MTP_LOCAL_LAYER_IDS == (0, 2, 4, 5, 6, 7)
    assert specs["model.mtp.layers.0.transformer_block.attn.k_sconv.weight"].shape == (
        2_048,
        1,
        4,
    )
    assert specs["model.mtp.layers.1.transformer_block.attn.k_sconv.weight"].shape == (
        1_024,
        1,
        4,
    )
    assert specs["model.mtp.layers.0.transformer_block.attn.rel_logits_proj.proj"].shape == (
        16,
        512,
    )
    assert specs["model.mtp.layers.1.transformer_block.attn.rel_logits_proj.proj"].shape == (
        16,
        1_024,
    )
    assert specs["model.mtp.layers.3.transformer_block.attn.k_sconv.weight"].shape == (
        1_024,
        1,
        4,
    )
    assert specs["model.mtp.layers.7.transformer_block.attn.k_sconv.weight"].shape == (
        2_048,
        1,
        4,
    )


def test_tc_inkling_mtp_003b_reconstruction_matches_pinned_real_header() -> None:
    prefix = expected_mtp_safetensors_header_prefix()
    header = parse_pinned_mtp_safetensors_header(prefix)

    assert len(prefix) == PINNED_MTP_HEADER_PREFIX_BYTES
    assert __import__("hashlib").sha256(prefix).hexdigest() == PINNED_MTP_HEADER_PREFIX_SHA256
    assert __import__("hashlib").sha256(prefix[8:]).hexdigest() == PINNED_MTP_HEADER_JSON_SHA256
    assert header == expected_mtp_safetensors_header()
    assert _audit(header=header).header_inventory_sha256 == (PINNED_MTP_HEADER_INVENTORY_SHA256)

    damaged = bytearray(prefix)
    damaged[-1] ^= 1
    with pytest.raises(ConfigurationError, match="prefix identity mismatch"):
        parse_pinned_mtp_safetensors_header(bytes(damaged))


@pytest.mark.parametrize("mutation", ["missing", "extra", "dtype", "shape", "offset"])
def test_tc_inkling_mtp_004_header_mismatch_fails_closed(mutation: str) -> None:
    header = expected_mtp_safetensors_header()
    name = next(iter(header))
    if mutation == "missing":
        header.pop(name)
    elif mutation == "extra":
        header["model.mtp.layers.8.unexpected.weight"] = {
            "dtype": "BF16",
            "shape": [1],
            "data_offsets": [0, 2],
        }
    else:
        entry = dict(header[name])
        if mutation == "dtype":
            entry["dtype"] = "F16"
        elif mutation == "shape":
            entry["shape"] = [1]
        else:
            raw_offsets = entry["data_offsets"]
            assert isinstance(raw_offsets, list)
            offsets = list(raw_offsets)
            offsets[1] -= 2
            entry["data_offsets"] = offsets
        header[name] = entry

    with pytest.raises(ConfigurationError):
        _audit(header=header)


def test_tc_inkling_mtp_005_mapping_is_complete_unique_and_dense() -> None:
    mapping = build_mtp_mapping_contract()
    outputs = [output for entry in mapping.entries for output in entry.outputs]
    w13_entries = [
        entry for entry in mapping.entries if entry.source_name.endswith("mlp.w13_dn.weight")
    ]

    assert len(mapping.entries) == 160
    assert len(outputs) == mapping.mapped_tensor_count == 168
    assert len({output.gguf_name for output in outputs}) == 168
    assert len(w13_entries) == 8
    assert all(len(entry.outputs) == 2 for entry in w13_entries)
    assert mapping.metadata.appended_block_indices == tuple(range(66, 74))
    assert mapping.metadata.base_block_count == MTP_TRUNK_LAYER_COUNT
    assert mapping.metadata.total_block_count == 74
    assert mapping.metadata.gguf_block_count_key == "inkling.block_count"
    assert mapping.metadata.gguf_nextn_predict_layers_key == "inkling.nextn_predict_layers"
    assert mapping.metadata.attention_pattern == MTP_ATTENTION_PATTERN
    assert not any("exp" in output.gguf_name for output in outputs)


def test_tc_inkling_mtp_005b_ffn_gscale_matches_pinned_llama_cpp_name() -> None:
    mapping = build_mtp_mapping_contract()
    global_scale_entries = [
        entry for entry in mapping.entries if entry.source_name.endswith("mlp.global_scale")
    ]

    assert [entry.outputs[0].gguf_name for entry in global_scale_entries] == [
        _PINNED_LLAMA_CPP_FFN_GSCALE_PATTERN.format(bid=MTP_TRUNK_LAYER_COUNT + head_index)
        + _PINNED_LLAMA_CPP_DEFAULT_TENSOR_SUFFIX
        for head_index in range(MTP_HEAD_COUNT)
    ]


def test_tc_inkling_mtp_006_dense_w13_deinterleave_is_even_odd_and_contiguous() -> None:
    interleaved = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)

    gate, up = deinterleave_dense_w13(interleaved)

    np.testing.assert_array_equal(gate, interleaved[0::2])
    np.testing.assert_array_equal(up, interleaved[1::2])
    assert gate.flags.c_contiguous
    assert up.flags.c_contiguous
    with pytest.raises(ValueError, match="rank 2"):
        deinterleave_dense_w13(np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="positive and even"):
        deinterleave_dense_w13(np.zeros((3, 2), dtype=np.float32))


def test_tc_inkling_mtp_007_shared_dependencies_and_variants_fail_closed() -> None:
    mapping = build_mtp_mapping_contract()
    weight_map = _weight_map()
    weight_map.pop(MTP_SHARED_SOURCE_DEPENDENCIES[0])

    assert tuple(value.source_name for value in mapping.shared_dependencies) == (
        MTP_SHARED_SOURCE_DEPENDENCIES
    )
    assert mapping.artifact_layout == "unresolved"
    assert mapping.shared_tensor_strategy == "unresolved"
    assert mapping.export_ready is False
    assert all(value.resolution == "unresolved" for value in mapping.shared_dependencies)
    assert all(value.reusable_only_when_bundled for value in mapping.shared_dependencies)
    assert all(
        value.copy_or_runtime_alias_required_for_standalone for value in mapping.shared_dependencies
    )
    with pytest.raises(ConfigurationError, match="dependencies are missing"):
        _audit(weight_map=weight_map)
    with pytest.raises(ConfigurationError, match="Unsupported Inkling MTP variant"):
        build_mtp_mapping_contract(MTPVariantContract(hidden_size=4_096))
    with pytest.raises(ConfigurationError, match="Unsupported Inkling MTP variant"):
        build_mtp_mapping_contract(MTPVariantContract(chain_hidden_post_norm=True))


def test_tc_inkling_mtp_008_digest_is_deterministic_and_receipt_denies_runtime_claims() -> None:
    header = expected_mtp_safetensors_header()
    reverse_header = dict(reversed(tuple(header.items())))
    first = _audit(header=header)
    second = _audit(header=reverse_header)
    first_mapping = build_mtp_mapping_contract()
    second_mapping = build_mtp_mapping_contract()
    receipt = build_mtp_format_receipt(first, first_mapping)

    assert first.header_inventory_sha256 == second.header_inventory_sha256
    assert first.source_contract_sha256 == second.source_contract_sha256
    assert first_mapping.canonical_sha256 == second_mapping.canonical_sha256
    assert receipt.support_state == "format_verified_not_runtime_verified"
    assert receipt.format_verified is True
    assert receipt.runtime_graph_implemented is False
    assert receipt.cache_rollback_implemented is False
    assert receipt.speculative_decoding_verified is False
    assert receipt.runtime_verified is False
    with pytest.raises(ValidationError):
        receipt.runtime_verified = True  # type: ignore[assignment,misc]
