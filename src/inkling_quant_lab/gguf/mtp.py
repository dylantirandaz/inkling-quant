"""Pinned, CPU-safe source and mapping contracts for Inkling MTP tensors.

This module verifies metadata and array transformations only.  It does not add an
MTP runtime graph, cache rollback, or speculative decoding to llama.cpp.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from pydantic import Field, field_validator

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.exceptions import ConfigurationError

PINNED_MTP_MODEL_ID = "thinkingmachines/Inkling"
PINNED_MTP_REVISION = "86b4d430ab871652a707666b89203a866888c5e5"
PINNED_MTP_ARCHITECTURE = "InklingForConditionalGeneration"
PINNED_MTP_SHARD = "mtp.safetensors"
PINNED_MTP_SHARD_BYTES = 10_520_911_944
PINNED_MTP_SHARD_SHA256 = "a0a01bf72ac48c3d7bfd56d5c1efbc955f00f51db30eb0955996680d8e595cc0"
PINNED_MTP_HEADER_BYTES = 20_528
PINNED_MTP_HEADER_PREFIX_BYTES = 20_536
PINNED_MTP_HEADER_PREFIX_SHA256 = "6a8d221af93f9a48559bc5ffa9fa23af50c32cb326a776ce7ef9e96605aaf081"
PINNED_MTP_HEADER_JSON_SHA256 = "9a9cf85fd997a46a13f8906733f19f3963bed02cc352ab07c06177e6ff92a381"
PINNED_MTP_HEADER_INVENTORY_SHA256 = (
    "0eaa84a961319662cb6010a2abb34d84d7e70c63cd2180ec9a84833cdf9dea23"
)

MTP_HEAD_COUNT = 8
MTP_SOURCE_TENSORS_PER_HEAD = 20
MTP_SOURCE_TENSOR_COUNT = MTP_HEAD_COUNT * MTP_SOURCE_TENSORS_PER_HEAD
MTP_TRUNK_LAYER_COUNT = 66
MTP_TOTAL_LAYER_COUNT = MTP_TRUNK_LAYER_COUNT + MTP_HEAD_COUNT
MTP_MAPPED_TENSOR_COUNT = MTP_SOURCE_TENSOR_COUNT + MTP_HEAD_COUNT
MTP_LOCAL_LAYER_IDS = (0, 2, 4, 5, 6, 7)
MTP_ATTENTION_PATTERN: tuple[Literal["local", "global"], ...] = (
    "local",
    "global",
    "local",
    "global",
    "local",
    "local",
    "local",
    "local",
)

MTP_HIDDEN_SIZE = 6_144
MTP_DENSE_INTERMEDIATE_SIZE = 24_576
MTP_QUERY_HEAD_COUNT = 64
MTP_GLOBAL_KV_HEAD_COUNT = 8
MTP_LOCAL_KV_HEAD_COUNT = 16
MTP_HEAD_DIM = 128
MTP_D_REL = 16
MTP_GLOBAL_REL_EXTENT = 1_024
MTP_LOCAL_REL_EXTENT = 512
MTP_SHORTCONV_KERNEL = 4
MTP_LOGIT_MUP_DENOMINATOR = 24.0

MTP_SHARED_SOURCE_DEPENDENCIES = (
    "model.llm.embed.weight",
    "model.llm.embed_norm.weight",
    "model.llm.unembed.weight",
)

MTPSupportState: TypeAlias = Literal["format_verified_not_runtime_verified"]
AttentionKind: TypeAlias = Literal["local", "global"]
TensorDType: TypeAlias = Literal["BF16", "F32"]
MappingOperation: TypeAlias = Literal[
    "copy",
    "rename",
    "squeeze_axis_1",
    "deinterleave_even_odd_rows",
    "cast_f32",
]


def _configuration_error(message: str, **details: object) -> ConfigurationError:
    return ConfigurationError(
        message,
        component="gguf.mtp",
        remediation="Use the exact pinned Inkling revision and MTP metadata contract.",
        details=dict(details),
    )


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class MTPVariantContract(StrictFrozenModel):
    """Model dimensions required by the one supported MTP variant."""

    model_id: str = PINNED_MTP_MODEL_ID
    revision: str = PINNED_MTP_REVISION
    architecture: str = PINNED_MTP_ARCHITECTURE
    trunk_layer_count: int = Field(default=MTP_TRUNK_LAYER_COUNT, ge=1)
    mtp_layer_count: int = Field(default=MTP_HEAD_COUNT, ge=1)
    hidden_size: int = Field(default=MTP_HIDDEN_SIZE, ge=1)
    dense_intermediate_size: int = Field(default=MTP_DENSE_INTERMEDIATE_SIZE, ge=1)
    query_head_count: int = Field(default=MTP_QUERY_HEAD_COUNT, ge=1)
    global_kv_head_count: int = Field(default=MTP_GLOBAL_KV_HEAD_COUNT, ge=1)
    local_kv_head_count: int = Field(default=MTP_LOCAL_KV_HEAD_COUNT, ge=1)
    head_dim: int = Field(default=MTP_HEAD_DIM, ge=1)
    d_rel: int = Field(default=MTP_D_REL, ge=1)
    global_rel_extent: int = Field(default=MTP_GLOBAL_REL_EXTENT, ge=1)
    local_rel_extent: int = Field(default=MTP_LOCAL_REL_EXTENT, ge=1)
    shortconv_kernel: int = Field(default=MTP_SHORTCONV_KERNEL, ge=1)
    logit_mup_denominator: float = Field(default=MTP_LOGIT_MUP_DENOMINATOR, gt=0.0)
    local_layer_ids: tuple[int, ...] = MTP_LOCAL_LAYER_IDS
    chain_hidden_post_norm: bool = False
    use_embed_norm: bool = True
    use_shortconv: bool = True

    @field_validator("local_layer_ids")
    @classmethod
    def local_ids_are_unique_and_sorted(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if value != tuple(sorted(set(value))):
            raise ValueError("local_layer_ids must be sorted and unique")
        return value


class MTPSourceTensorSpec(StrictFrozenModel):
    """Expected metadata for one tensor in the pinned MTP safetensors shard."""

    name: str = Field(min_length=1)
    head_index: int = Field(ge=0, lt=MTP_HEAD_COUNT)
    source_suffix: str = Field(min_length=1)
    dtype: Literal["BF16"] = "BF16"
    shape: tuple[int, ...]
    nbytes: int = Field(ge=1)
    shard: Literal["mtp.safetensors"] = "mtp.safetensors"

    @field_validator("shape")
    @classmethod
    def shape_is_nonempty_and_positive(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(dimension <= 0 for dimension in value):
            raise ValueError("shape dimensions must be positive")
        return value


class MTPHeaderTensor(StrictFrozenModel):
    """Verified safetensors header entry without loading its tensor payload."""

    name: str = Field(min_length=1)
    dtype: Literal["BF16"] = "BF16"
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]


class MTPSharedDependency(StrictFrozenModel):
    """A trunk tensor whose standalone-versus-bundled resolution is pending."""

    source_name: str = Field(min_length=1)
    gguf_name: str = Field(min_length=1)
    role: Literal["token_embedding", "embedding_norm", "output_projection"]
    resolution: Literal["unresolved"] = "unresolved"
    reusable_only_when_bundled: Literal[True] = True
    copy_or_runtime_alias_required_for_standalone: Literal[True] = True


class MTPMappedTensor(StrictFrozenModel):
    """One converter-array output produced by an MTP source tensor."""

    gguf_name: str = Field(min_length=1)
    shape: tuple[int, ...]
    converter_dtype: TensorDType


class MTPMappingEntry(StrictFrozenModel):
    """Deterministic source-to-GGUF mapping and required array operations."""

    source_name: str = Field(min_length=1)
    head_index: int = Field(ge=0, lt=MTP_HEAD_COUNT)
    gguf_block_index: int = Field(ge=MTP_TRUNK_LAYER_COUNT)
    source_shape: tuple[int, ...]
    source_dtype: Literal["BF16"] = "BF16"
    operations: tuple[MappingOperation, ...]
    outputs: tuple[MTPMappedTensor, ...]


class MTPMetadataContract(StrictFrozenModel):
    """Metadata needed by a future Inkling-specific MTP runtime implementation."""

    gguf_nextn_predict_layers_key: Literal["inkling.nextn_predict_layers"] = (
        "inkling.nextn_predict_layers"
    )
    gguf_block_count_key: Literal["inkling.block_count"] = "inkling.block_count"
    mtp_layer_count: Literal[8] = 8
    base_block_count: Literal[66] = 66
    total_block_count: Literal[74] = 74
    appended_block_indices: tuple[int, ...] = tuple(
        range(MTP_TRUNK_LAYER_COUNT, MTP_TRUNK_LAYER_COUNT + MTP_HEAD_COUNT)
    )
    attention_pattern: tuple[AttentionKind, ...] = MTP_ATTENTION_PATTERN
    local_layer_ids: tuple[int, ...] = MTP_LOCAL_LAYER_IDS
    chain_hidden_post_norm: Literal[False] = False
    hidden_size: Literal[6144] = 6_144
    dense_intermediate_size: Literal[24576] = 24_576
    query_head_count: Literal[64] = 64
    global_kv_head_count: Literal[8] = 8
    local_kv_head_count: Literal[16] = 16
    head_dim: Literal[128] = 128
    d_rel: Literal[16] = 16
    global_rel_extent: Literal[1024] = 1_024
    local_rel_extent: Literal[512] = 512
    shortconv_kernel: Literal[4] = 4
    logit_mup_denominator: float = MTP_LOGIT_MUP_DENOMINATOR
    use_embed_norm: Literal[True] = True
    use_shortconv: Literal[True] = True


class MTPSourceAudit(StrictFrozenModel):
    """Result of verifying the pinned index and safetensors header."""

    schema_version: Literal["iql.inkling_mtp_source_audit.v1"] = "iql.inkling_mtp_source_audit.v1"
    model_id: Literal["thinkingmachines/Inkling"] = "thinkingmachines/Inkling"
    revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"] = (
        "86b4d430ab871652a707666b89203a866888c5e5"
    )
    shard: Literal["mtp.safetensors"] = "mtp.safetensors"
    shard_size_bytes: Literal[10520911944] = 10_520_911_944
    shard_sha256: Literal["a0a01bf72ac48c3d7bfd56d5c1efbc955f00f51db30eb0955996680d8e595cc0"] = (
        "a0a01bf72ac48c3d7bfd56d5c1efbc955f00f51db30eb0955996680d8e595cc0"
    )
    header_size_bytes: Literal[20528] = 20_528
    header_prefix_sha256: Literal[
        "6a8d221af93f9a48559bc5ffa9fa23af50c32cb326a776ce7ef9e96605aaf081"
    ] = "6a8d221af93f9a48559bc5ffa9fa23af50c32cb326a776ce7ef9e96605aaf081"
    payload_size_bytes: int = Field(ge=1)
    head_count: Literal[8] = 8
    tensor_count: Literal[160] = 160
    tensors_per_head: Literal[20] = 20
    header_tensors: tuple[MTPHeaderTensor, ...]
    shared_dependencies: tuple[MTPSharedDependency, ...]
    source_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    header_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_state: MTPSupportState = "format_verified_not_runtime_verified"
    verified: Literal[True] = True
    runtime_verified: Literal[False] = False


class MTPMappingContract(StrictFrozenModel):
    """Complete deterministic converter contract for all pinned MTP sources."""

    schema_version: Literal["iql.inkling_mtp_mapping.v1"] = "iql.inkling_mtp_mapping.v1"
    model_id: Literal["thinkingmachines/Inkling"] = "thinkingmachines/Inkling"
    revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"] = (
        "86b4d430ab871652a707666b89203a866888c5e5"
    )
    source_tensor_count: Literal[160] = 160
    mapped_tensor_count: Literal[168] = 168
    metadata: MTPMetadataContract
    entries: tuple[MTPMappingEntry, ...]
    shared_dependencies: tuple[MTPSharedDependency, ...]
    canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_layout: Literal["unresolved"] = "unresolved"
    shared_tensor_strategy: Literal["unresolved"] = "unresolved"
    export_ready: Literal[False] = False
    support_state: MTPSupportState = "format_verified_not_runtime_verified"
    runtime_graph_implemented: Literal[False] = False
    cache_rollback_implemented: Literal[False] = False
    speculative_decoding_verified: Literal[False] = False
    runtime_verified: Literal[False] = False


class MTPFormatReceipt(StrictFrozenModel):
    """Immutable proof that source metadata and mapping contracts agree."""

    schema_version: Literal["iql.inkling_mtp_format_receipt.v1"] = (
        "iql.inkling_mtp_format_receipt.v1"
    )
    model_id: Literal["thinkingmachines/Inkling"] = "thinkingmachines/Inkling"
    revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"] = (
        "86b4d430ab871652a707666b89203a866888c5e5"
    )
    shard_sha256: Literal["a0a01bf72ac48c3d7bfd56d5c1efbc955f00f51db30eb0955996680d8e595cc0"] = (
        "a0a01bf72ac48c3d7bfd56d5c1efbc955f00f51db30eb0955996680d8e595cc0"
    )
    source_inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mapping_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    canonical_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_tensor_count: Literal[160] = 160
    mapped_tensor_count: Literal[168] = 168
    shared_dependency_count: Literal[3] = 3
    support_state: MTPSupportState = "format_verified_not_runtime_verified"
    format_verified: Literal[True] = True
    runtime_graph_implemented: Literal[False] = False
    cache_rollback_implemented: Literal[False] = False
    speculative_decoding_verified: Literal[False] = False
    runtime_verified: Literal[False] = False


def pinned_mtp_variant() -> MTPVariantContract:
    """Return the sole supported Inkling MTP variant."""

    return MTPVariantContract()


def _require_pinned_variant(variant: MTPVariantContract) -> None:
    expected = pinned_mtp_variant().model_dump(mode="json")
    actual = variant.model_dump(mode="json")
    mismatches = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatches:
        raise _configuration_error("Unsupported Inkling MTP variant", mismatches=mismatches)


def _tensor_templates(
    attention_kind: AttentionKind,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    kv_width = (
        MTP_LOCAL_KV_HEAD_COUNT if attention_kind == "local" else MTP_GLOBAL_KV_HEAD_COUNT
    ) * MTP_HEAD_DIM
    rel_extent = MTP_LOCAL_REL_EXTENT if attention_kind == "local" else MTP_GLOBAL_REL_EXTENT
    return (
        ("embed_norm.weight", (MTP_HIDDEN_SIZE,)),
        ("hidden_norm.weight", (MTP_HIDDEN_SIZE,)),
        ("input_proj.weight", (MTP_HIDDEN_SIZE, 2 * MTP_HIDDEN_SIZE)),
        ("transformer_block.attn.k_norm.weight", (MTP_HEAD_DIM,)),
        (
            "transformer_block.attn.k_sconv.weight",
            (kv_width, 1, MTP_SHORTCONV_KERNEL),
        ),
        ("transformer_block.attn.q_norm.weight", (MTP_HEAD_DIM,)),
        ("transformer_block.attn.rel_logits_proj.proj", (MTP_D_REL, rel_extent)),
        (
            "transformer_block.attn.v_sconv.weight",
            (kv_width, 1, MTP_SHORTCONV_KERNEL),
        ),
        ("transformer_block.attn.wk_dv.weight", (kv_width, MTP_HIDDEN_SIZE)),
        (
            "transformer_block.attn.wo_ud.weight",
            (MTP_HIDDEN_SIZE, MTP_QUERY_HEAD_COUNT * MTP_HEAD_DIM),
        ),
        (
            "transformer_block.attn.wq_du.weight",
            (MTP_QUERY_HEAD_COUNT * MTP_HEAD_DIM, MTP_HIDDEN_SIZE),
        ),
        ("transformer_block.attn.wr_du.weight", (MTP_D_REL * 64, MTP_HIDDEN_SIZE)),
        ("transformer_block.attn.wv_dv.weight", (kv_width, MTP_HIDDEN_SIZE)),
        ("transformer_block.attn_norm.weight", (MTP_HIDDEN_SIZE,)),
        (
            "transformer_block.attn_sconv.weight",
            (MTP_HIDDEN_SIZE, 1, MTP_SHORTCONV_KERNEL),
        ),
        ("transformer_block.mlp.global_scale", (1,)),
        (
            "transformer_block.mlp.w13_dn.weight",
            (2 * MTP_DENSE_INTERMEDIATE_SIZE, MTP_HIDDEN_SIZE),
        ),
        (
            "transformer_block.mlp.w2_md.weight",
            (MTP_HIDDEN_SIZE, MTP_DENSE_INTERMEDIATE_SIZE),
        ),
        ("transformer_block.mlp_norm.weight", (MTP_HIDDEN_SIZE,)),
        (
            "transformer_block.mlp_sconv.weight",
            (MTP_HIDDEN_SIZE, 1, MTP_SHORTCONV_KERNEL),
        ),
    )


def expected_mtp_source_specs(
    variant: MTPVariantContract | None = None,
) -> tuple[MTPSourceTensorSpec, ...]:
    """Build the exact 8-by-20 pinned source inventory."""

    selected = pinned_mtp_variant() if variant is None else variant
    _require_pinned_variant(selected)
    specs: list[MTPSourceTensorSpec] = []
    for head_index, attention_kind in enumerate(MTP_ATTENTION_PATTERN):
        for suffix, shape in _tensor_templates(attention_kind):
            element_count = int(np.prod(shape, dtype=np.int64))
            specs.append(
                MTPSourceTensorSpec(
                    name=f"model.mtp.layers.{head_index}.{suffix}",
                    head_index=head_index,
                    source_suffix=suffix,
                    shape=shape,
                    nbytes=element_count * 2,
                )
            )
    if len(specs) != MTP_SOURCE_TENSOR_COUNT:
        raise RuntimeError("internal MTP source contract has the wrong tensor count")
    return tuple(specs)


def expected_mtp_safetensors_header() -> dict[str, dict[str, object]]:
    """Reconstruct the pinned header mapping with exact shapes and byte spans."""

    offset = 0
    header: dict[str, dict[str, object]] = {}
    for spec in expected_mtp_source_specs():
        header[spec.name] = {
            "dtype": spec.dtype,
            "shape": list(spec.shape),
            "data_offsets": [offset, offset + spec.nbytes],
        }
        offset += spec.nbytes
    return header


def expected_mtp_safetensors_header_prefix() -> bytes:
    """Reconstruct and verify the exact 20,536-byte pinned shard prefix."""

    payload = json.dumps(
        expected_mtp_safetensors_header(),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if len(payload) > PINNED_MTP_HEADER_BYTES:
        raise RuntimeError("internal MTP header exceeds its pinned byte length")
    padded = payload + b" " * (PINNED_MTP_HEADER_BYTES - len(payload))
    prefix = struct.pack("<Q", PINNED_MTP_HEADER_BYTES) + padded
    if (
        len(prefix) != PINNED_MTP_HEADER_PREFIX_BYTES
        or hashlib.sha256(prefix).hexdigest() != PINNED_MTP_HEADER_PREFIX_SHA256
        or hashlib.sha256(padded).hexdigest() != PINNED_MTP_HEADER_JSON_SHA256
    ):
        raise RuntimeError("internal MTP header reconstruction differs from the pinned shard")
    return prefix


def parse_pinned_mtp_safetensors_header(
    prefix: bytes,
) -> dict[str, dict[str, object]]:
    """Parse a range-read prefix only after its complete identity is verified."""

    if (
        len(prefix) != PINNED_MTP_HEADER_PREFIX_BYTES
        or hashlib.sha256(prefix).hexdigest() != PINNED_MTP_HEADER_PREFIX_SHA256
    ):
        raise _configuration_error(
            "Inkling MTP safetensors header prefix identity mismatch",
            expected_size=PINNED_MTP_HEADER_PREFIX_BYTES,
            actual_size=len(prefix),
            expected_sha256=PINNED_MTP_HEADER_PREFIX_SHA256,
            actual_sha256=hashlib.sha256(prefix).hexdigest(),
        )
    header_size = struct.unpack("<Q", prefix[:8])[0]
    if header_size != PINNED_MTP_HEADER_BYTES:
        raise _configuration_error(
            "Inkling MTP safetensors header length mismatch",
            expected=PINNED_MTP_HEADER_BYTES,
            actual=header_size,
        )
    encoded = prefix[8:]
    if hashlib.sha256(encoded).hexdigest() != PINNED_MTP_HEADER_JSON_SHA256:
        raise _configuration_error("Inkling MTP safetensors header JSON identity mismatch")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _configuration_error("Inkling MTP safetensors header is invalid JSON") from error
    if not isinstance(value, dict) or any(
        not isinstance(name, str) or not isinstance(entry, dict) for name, entry in value.items()
    ):
        raise _configuration_error("Inkling MTP safetensors header root is invalid")
    return value


def mtp_shared_dependencies() -> tuple[MTPSharedDependency, ...]:
    """Return shared tensors that require an explicit artifact-layout decision."""

    return (
        MTPSharedDependency(
            source_name="model.llm.embed.weight",
            gguf_name="token_embd.weight",
            role="token_embedding",
        ),
        MTPSharedDependency(
            source_name="model.llm.embed_norm.weight",
            gguf_name="token_embd_norm.weight",
            role="embedding_norm",
        ),
        MTPSharedDependency(
            source_name="model.llm.unembed.weight",
            gguf_name="output.weight",
            role="output_projection",
        ),
    )


def _parse_header_entry(name: str, raw: object) -> MTPHeaderTensor:
    if not isinstance(raw, Mapping):
        raise _configuration_error("MTP safetensors header entry must be an object", tensor=name)
    dtype = raw.get("dtype")
    shape = raw.get("shape")
    offsets = raw.get("data_offsets")
    if dtype != "BF16":
        raise _configuration_error(
            "MTP tensor dtype does not match the pinned BF16 contract",
            tensor=name,
            expected="BF16",
            actual=dtype,
        )
    if not isinstance(shape, (list, tuple)) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in shape
    ):
        raise _configuration_error("MTP tensor shape is invalid", tensor=name, actual=shape)
    if (
        not isinstance(offsets, (list, tuple))
        or len(offsets) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in offsets)
    ):
        raise _configuration_error("MTP tensor data_offsets are invalid", tensor=name)
    shape_tuple = tuple(shape)
    offset_tuple = (offsets[0], offsets[1])
    if not shape_tuple or any(value <= 0 for value in shape_tuple):
        raise _configuration_error("MTP tensor shape must be positive", tensor=name)
    if offset_tuple[0] < 0 or offset_tuple[1] <= offset_tuple[0]:
        raise _configuration_error("MTP tensor data_offsets must be increasing", tensor=name)
    return MTPHeaderTensor(
        name=name,
        dtype="BF16",
        shape=shape_tuple,
        data_offsets=offset_tuple,
    )


def _validate_identity(
    *,
    model_id: str,
    revision: str,
    shard_name: str,
    shard_size_bytes: int,
    shard_sha256: str,
    header_size_bytes: int,
    header_prefix_sha256: str,
) -> None:
    fields = {
        "model_id": (PINNED_MTP_MODEL_ID, model_id),
        "revision": (PINNED_MTP_REVISION, revision),
        "shard_name": (PINNED_MTP_SHARD, shard_name),
        "shard_size_bytes": (PINNED_MTP_SHARD_BYTES, shard_size_bytes),
        "shard_sha256": (PINNED_MTP_SHARD_SHA256, shard_sha256.lower()),
        "header_size_bytes": (PINNED_MTP_HEADER_BYTES, header_size_bytes),
        "header_prefix_sha256": (
            PINNED_MTP_HEADER_PREFIX_SHA256,
            header_prefix_sha256.lower(),
        ),
    }
    mismatches = {
        field: {"expected": expected, "actual": actual}
        for field, (expected, actual) in fields.items()
        if actual != expected
    }
    if mismatches:
        raise _configuration_error(
            "Pinned Inkling MTP shard identity mismatch",
            mismatches=mismatches,
        )


def audit_mtp_source_inventory(
    *,
    model_id: str,
    revision: str,
    shard_name: str,
    shard_size_bytes: int,
    shard_sha256: str,
    header_size_bytes: int,
    header_prefix_sha256: str,
    weight_map: Mapping[str, object],
    safetensors_header: Mapping[str, object],
    variant: MTPVariantContract | None = None,
) -> MTPSourceAudit:
    """Verify the exact pinned MTP index and header without reading tensor payloads."""

    selected = pinned_mtp_variant() if variant is None else variant
    _require_pinned_variant(selected)
    _validate_identity(
        model_id=model_id,
        revision=revision,
        shard_name=shard_name,
        shard_size_bytes=shard_size_bytes,
        shard_sha256=shard_sha256,
        header_size_bytes=header_size_bytes,
        header_prefix_sha256=header_prefix_sha256,
    )

    expected_specs = expected_mtp_source_specs(selected)
    expected_by_name = {spec.name: spec for spec in expected_specs}
    header_names = {name for name in safetensors_header if name != "__metadata__"}
    expected_names = set(expected_by_name)
    if header_names != expected_names:
        raise _configuration_error(
            "MTP safetensors header does not contain the exact 160-tensor inventory",
            missing=sorted(expected_names - header_names),
            unexpected=sorted(header_names - expected_names),
        )

    parsed = tuple(
        _parse_header_entry(name, safetensors_header[name]) for name in sorted(header_names)
    )
    for tensor in parsed:
        expected = expected_by_name[tensor.name]
        if tensor.shape != expected.shape:
            raise _configuration_error(
                "MTP tensor shape does not match the pinned contract",
                tensor=tensor.name,
                expected=list(expected.shape),
                actual=list(tensor.shape),
            )
        start, end = tensor.data_offsets
        if end - start != expected.nbytes:
            raise _configuration_error(
                "MTP tensor byte span does not match its BF16 shape",
                tensor=tensor.name,
                expected=expected.nbytes,
                actual=end - start,
            )

    intervals = sorted(
        (tensor.data_offsets[0], tensor.data_offsets[1], tensor.name) for tensor in parsed
    )
    cursor = 0
    for start, end, name in intervals:
        if start != cursor:
            raise _configuration_error(
                "MTP safetensors payload has a gap or overlap",
                tensor=name,
                expected_start=cursor,
                actual_start=start,
            )
        cursor = end
    expected_payload_bytes = shard_size_bytes - 8 - header_size_bytes
    if cursor != expected_payload_bytes:
        raise _configuration_error(
            "MTP safetensors payload size does not match the pinned shard",
            expected=expected_payload_bytes,
            actual=cursor,
        )

    indexed_mtp_names = {name for name in weight_map if name.startswith("model.mtp.")}
    if indexed_mtp_names != expected_names:
        raise _configuration_error(
            "MTP weight index does not contain the exact 160-tensor inventory",
            missing=sorted(expected_names - indexed_mtp_names),
            unexpected=sorted(indexed_mtp_names - expected_names),
        )
    wrong_shards = {
        name: weight_map[name]
        for name in sorted(expected_names)
        if weight_map[name] != PINNED_MTP_SHARD
    }
    if wrong_shards:
        raise _configuration_error(
            "MTP tensors are assigned to unexpected shards",
            tensors=wrong_shards,
        )

    dependencies = mtp_shared_dependencies()
    missing_dependencies = [
        dependency.source_name
        for dependency in dependencies
        if dependency.source_name not in weight_map
    ]
    if missing_dependencies:
        raise _configuration_error(
            "MTP shared trunk dependencies are missing from the weight index",
            missing=missing_dependencies,
        )
    invalid_dependencies = {
        dependency.source_name: weight_map[dependency.source_name]
        for dependency in dependencies
        if not isinstance(weight_map[dependency.source_name], str)
        or not str(weight_map[dependency.source_name]).endswith(".safetensors")
        or weight_map[dependency.source_name] == PINNED_MTP_SHARD
    }
    if invalid_dependencies:
        raise _configuration_error(
            "MTP shared dependencies must reference trunk safetensors shards",
            dependencies=invalid_dependencies,
        )

    source_contract_payload = {
        "variant": selected.model_dump(mode="json"),
        "shard": {
            "name": PINNED_MTP_SHARD,
            "size_bytes": PINNED_MTP_SHARD_BYTES,
            "sha256": PINNED_MTP_SHARD_SHA256,
            "header_size_bytes": PINNED_MTP_HEADER_BYTES,
        },
        "tensors": [spec.model_dump(mode="json") for spec in expected_specs],
        "shared_dependencies": [value.model_dump(mode="json") for value in dependencies],
    }
    header_payload = [tensor.model_dump(mode="json") for tensor in parsed]
    header_inventory_sha256 = _canonical_sha256(header_payload)
    if header_inventory_sha256 != PINNED_MTP_HEADER_INVENTORY_SHA256:
        raise _configuration_error(
            "MTP header inventory digest differs from the pinned shard",
            expected=PINNED_MTP_HEADER_INVENTORY_SHA256,
            actual=header_inventory_sha256,
        )
    return MTPSourceAudit(
        payload_size_bytes=cursor,
        header_tensors=parsed,
        shared_dependencies=dependencies,
        source_contract_sha256=_canonical_sha256(source_contract_payload),
        header_inventory_sha256=header_inventory_sha256,
    )


def _mapping_for_spec(spec: MTPSourceTensorSpec) -> MTPMappingEntry:
    block = MTP_TRUNK_LAYER_COUNT + spec.head_index
    suffix = spec.source_suffix
    base = f"blk.{block}"
    name_map: dict[str, str] = {
        "embed_norm.weight": f"{base}.nextn.enorm.weight",
        "hidden_norm.weight": f"{base}.nextn.hnorm.weight",
        "input_proj.weight": f"{base}.nextn.eh_proj.weight",
        "transformer_block.attn.k_norm.weight": f"{base}.attn_k_norm.weight",
        "transformer_block.attn.k_sconv.weight": f"{base}.shortconv_k.weight",
        "transformer_block.attn.q_norm.weight": f"{base}.attn_q_norm.weight",
        "transformer_block.attn.rel_logits_proj.proj": f"{base}.attn_rel_proj.weight",
        "transformer_block.attn.v_sconv.weight": f"{base}.shortconv_v.weight",
        "transformer_block.attn.wk_dv.weight": f"{base}.attn_k.weight",
        "transformer_block.attn.wo_ud.weight": f"{base}.attn_output.weight",
        "transformer_block.attn.wq_du.weight": f"{base}.attn_q.weight",
        "transformer_block.attn.wr_du.weight": f"{base}.attn_r.weight",
        "transformer_block.attn.wv_dv.weight": f"{base}.attn_v.weight",
        "transformer_block.attn_norm.weight": f"{base}.attn_norm.weight",
        "transformer_block.attn_sconv.weight": f"{base}.shortconv_attn.weight",
        "transformer_block.mlp.global_scale": f"{base}.ffn_gscale.weight",
        "transformer_block.mlp.w2_md.weight": f"{base}.ffn_down.weight",
        "transformer_block.mlp_norm.weight": f"{base}.ffn_norm.weight",
        "transformer_block.mlp_sconv.weight": f"{base}.shortconv_mlp.weight",
    }
    outputs: tuple[MTPMappedTensor, ...]
    output_shape: tuple[int, ...]
    if suffix == "transformer_block.mlp.w13_dn.weight":
        output_shape = (MTP_DENSE_INTERMEDIATE_SIZE, MTP_HIDDEN_SIZE)
        outputs = (
            MTPMappedTensor(
                gguf_name=f"{base}.ffn_gate.weight",
                shape=output_shape,
                converter_dtype="BF16",
            ),
            MTPMappedTensor(
                gguf_name=f"{base}.ffn_up.weight",
                shape=output_shape,
                converter_dtype="BF16",
            ),
        )
        operations: tuple[MappingOperation, ...] = ("deinterleave_even_odd_rows",)
    else:
        try:
            output_name = name_map[suffix]
        except KeyError as error:
            raise RuntimeError(f"internal MTP mapping is missing {suffix}") from error
        output_shape = spec.shape
        operations = ("copy",)
        dtype: TensorDType = "BF16"
        if suffix.endswith("_sconv.weight"):
            output_shape = (spec.shape[0], spec.shape[2])
            operations = ("squeeze_axis_1", "cast_f32")
            dtype = "F32"
        elif suffix == "transformer_block.attn.rel_logits_proj.proj":
            operations = ("rename", "cast_f32")
            dtype = "F32"
        elif suffix == "transformer_block.mlp.global_scale":
            operations = ("cast_f32",)
            dtype = "F32"
        elif output_name != spec.name:
            operations = ("rename",)
        outputs = (
            MTPMappedTensor(
                gguf_name=output_name,
                shape=output_shape,
                converter_dtype=dtype,
            ),
        )
    return MTPMappingEntry(
        source_name=spec.name,
        head_index=spec.head_index,
        gguf_block_index=block,
        source_shape=spec.shape,
        operations=operations,
        outputs=outputs,
    )


def build_mtp_mapping_contract(
    variant: MTPVariantContract | None = None,
) -> MTPMappingContract:
    """Build the complete mapping contract, failing closed on other variants."""

    selected = pinned_mtp_variant() if variant is None else variant
    _require_pinned_variant(selected)
    entries = tuple(_mapping_for_spec(spec) for spec in expected_mtp_source_specs(selected))
    output_names = [output.gguf_name for entry in entries for output in entry.outputs]
    if len(entries) != MTP_SOURCE_TENSOR_COUNT or len(output_names) != MTP_MAPPED_TENSOR_COUNT:
        raise RuntimeError("internal MTP mapping count is inconsistent")
    if len(set(output_names)) != len(output_names):
        raise RuntimeError("internal MTP mapping contains duplicate GGUF tensor names")
    dependencies = mtp_shared_dependencies()
    metadata = MTPMetadataContract()
    payload = {
        "schema_version": "iql.inkling_mtp_mapping.v1",
        "model_id": PINNED_MTP_MODEL_ID,
        "revision": PINNED_MTP_REVISION,
        "source_tensor_count": MTP_SOURCE_TENSOR_COUNT,
        "mapped_tensor_count": len(output_names),
        "metadata": metadata.model_dump(mode="json"),
        "entries": [entry.model_dump(mode="json") for entry in entries],
        "shared_dependencies": [value.model_dump(mode="json") for value in dependencies],
        "artifact_layout": "unresolved",
        "shared_tensor_strategy": "unresolved",
        "export_ready": False,
        "support_state": "format_verified_not_runtime_verified",
        "runtime_graph_implemented": False,
        "cache_rollback_implemented": False,
        "speculative_decoding_verified": False,
        "runtime_verified": False,
    }
    return MTPMappingContract(
        metadata=metadata,
        entries=entries,
        shared_dependencies=dependencies,
        canonical_sha256=_canonical_sha256(payload),
    )


def deinterleave_dense_w13(
    tensor: npt.NDArray[np.generic],
) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]:
    """Split interleaved SwiGLU rows into contiguous gate and up arrays."""

    if tensor.ndim != 2:
        raise ValueError(f"w13 tensor must be rank 2, got rank {tensor.ndim}")
    if tensor.shape[0] == 0 or tensor.shape[0] % 2 != 0:
        raise ValueError("w13 output-row dimension must be positive and even")
    if tensor.shape[1] == 0:
        raise ValueError("w13 input dimension must be positive")
    gate = np.ascontiguousarray(tensor[0::2, :])
    up = np.ascontiguousarray(tensor[1::2, :])
    return gate, up


def build_mtp_format_receipt(
    audit: MTPSourceAudit,
    mapping: MTPMappingContract,
) -> MTPFormatReceipt:
    """Bind an exact source audit to its mapping without claiming runtime support."""

    if audit.model_id != mapping.model_id or audit.revision != mapping.revision:
        raise _configuration_error("MTP source audit and mapping identity do not match")
    if audit.tensor_count != mapping.source_tensor_count:
        raise _configuration_error("MTP source audit and mapping tensor counts do not match")
    payload = {
        "schema_version": "iql.inkling_mtp_format_receipt.v1",
        "model_id": audit.model_id,
        "revision": audit.revision,
        "shard_sha256": audit.shard_sha256,
        "source_inventory_sha256": audit.header_inventory_sha256,
        "mapping_contract_sha256": mapping.canonical_sha256,
        "source_tensor_count": audit.tensor_count,
        "mapped_tensor_count": mapping.mapped_tensor_count,
        "shared_dependency_count": len(mapping.shared_dependencies),
        "support_state": "format_verified_not_runtime_verified",
        "format_verified": True,
        "runtime_graph_implemented": False,
        "cache_rollback_implemented": False,
        "speculative_decoding_verified": False,
        "runtime_verified": False,
    }
    return MTPFormatReceipt(
        source_inventory_sha256=audit.header_inventory_sha256,
        mapping_contract_sha256=mapping.canonical_sha256,
        canonical_sha256=_canonical_sha256(payload),
    )
