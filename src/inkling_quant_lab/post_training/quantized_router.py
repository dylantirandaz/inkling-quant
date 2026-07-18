"""CPU-only contract for an MLX-compatible quantized learned-router overlay.

The payload is deliberately only eighteen ordinary tensors: packed ``weight``,
``scales``, and affine ``biases`` for each of six gates.  This module mirrors
the pinned MLX 0.32.0 affine quantizer without importing MLX, and records enough
lineage and inventory evidence to reconstruct the six ``QuantizedLinear``
modules without serializing Python objects or pickle payloads.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal, Self, TypeAlias, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from inkling_quant_lab.post_training.lineage import (
    ROUTER_LAYER_INDICES,
    ROUTER_TENSOR_COUNT,
    TENSOR_HASH_SCHEME,
    RouterOverlayLineage,
    canonical_tensor_sha256,
)
from inkling_quant_lab.security import sensitive_literal_path

ROUTER_INPUT_DIMS: Final = 288
ROUTER_OUTPUT_DIMS: Final = 4
ROUTER_SHAPE: Final = (ROUTER_OUTPUT_DIMS, ROUTER_INPUT_DIMS)
AFFINE_GROUP_SIZE: Final = 32
AFFINE_GROUP_COUNT: Final = ROUTER_INPUT_DIMS // AFFINE_GROUP_SIZE
QUANTIZED_ROUTER_TENSOR_COUNT: Final = ROUTER_TENSOR_COUNT * 3
SUPPORTED_BITS: Final = (4, 8)
ROUTER_WEIGHT_NAMES: Final = tuple(
    f"model.layers.{layer}.block_sparse_moe.gate.weight" for layer in ROUTER_LAYER_INDICES
)
TENSOR_PAYLOAD_HASH_SCHEME: Final = (
    "sha256(sorted UTF-8 tensor name length/name + canonical tensor SHA-256 bytes)"
)

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_REVISION_PATTERN = r"^[0-9a-f]{40}$"

QuantizedTensorState: TypeAlias = Mapping[str, Tensor]
QuantizationBits: TypeAlias = Literal[4, 8]


class QuantizedRouterOverlayError(ValueError):
    """A learned or quantized router payload failed a fail-closed audit."""


class ImmutableQuantizedRouterRecord(BaseModel):
    """Strict immutable record with deterministic JSON and digest helpers."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, validate_default=True
    )

    def canonical_json(self) -> str:
        """Serialize this record deterministically."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def sha256(self) -> str:
        """Hash the canonical record bytes."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class QuantizedTensorFact(ImmutableQuantizedRouterRecord):
    """Exact name, representation, byte count, and digest for one tensor."""

    name: str = Field(min_length=1)
    layer_index: int = Field(ge=0, le=ROUTER_TENSOR_COUNT - 1)
    role: Literal["packed_weight", "scales", "affine_biases"]
    dtype: Literal["uint32", "float32"]
    shape: tuple[int, int]
    nbytes: int = Field(gt=0)
    tensor_sha256: str = Field(pattern=_SHA256_PATTERN)
    hash_scheme: Literal[
        "sha256(torch dtype ASCII + NUL + compact JSON shape + NUL + contiguous CPU tensor bytes)"
    ] = TENSOR_HASH_SCHEME

    @model_validator(mode="after")
    def valid_name(self) -> Self:
        if "\0" in self.name:
            raise ValueError("quantized tensor names must not contain NUL")
        return self


class QuantizedRouterFact(ImmutableQuantizedRouterRecord):
    """Learned-float parent and measured quantization error for one gate."""

    layer_index: int = Field(ge=0, le=ROUTER_TENSOR_COUNT - 1)
    gate_module_name: str = Field(min_length=1)
    learned_float_weight_name: str = Field(min_length=1)
    learned_float_weight_sha256: str = Field(pattern=_SHA256_PATTERN)
    packed_weight_name: str = Field(min_length=1)
    scales_name: str = Field(min_length=1)
    biases_name: str = Field(min_length=1)
    max_abs_error: float = Field(ge=0.0)
    rmse: float = Field(ge=0.0)
    max_rounding_error_bound: float = Field(ge=0.0)

    @model_validator(mode="after")
    def exact_gate_names(self) -> Self:
        weight_name = ROUTER_WEIGHT_NAMES[self.layer_index]
        module_name = weight_name.removesuffix(".weight")
        expected = (
            module_name,
            weight_name,
            weight_name,
            f"{module_name}.scales",
            f"{module_name}.biases",
        )
        actual = (
            self.gate_module_name,
            self.learned_float_weight_name,
            self.packed_weight_name,
            self.scales_name,
            self.biases_name,
        )
        if actual != expected:
            raise ValueError("router fact names differ from the exact six-gate contract")
        if self.max_abs_error > self.max_rounding_error_bound:
            raise ValueError("measured affine error exceeds its recorded rounding bound")
        return self


class QuantizedRouterReconstructionRecipe(ImmutableQuantizedRouterRecord):
    """Non-executable recipe for reconstructing six MLX quantized gates."""

    mlx_version: Literal["0.32.0"] = "0.32.0"
    mlx_lm_version: Literal["0.31.3"] = "0.31.3"
    module_type: Literal["mlx.nn.QuantizedLinear"] = "mlx.nn.QuantizedLinear"
    input_dims: Literal[288] = ROUTER_INPUT_DIMS
    output_dims: Literal[4] = ROUTER_OUTPUT_DIMS
    linear_bias: Literal[False] = False
    group_size: Literal[32] = AFFINE_GROUP_SIZE
    bits: QuantizationBits
    mode: Literal["affine"] = "affine"
    parameters_per_gate: tuple[Literal["weight", "scales", "biases"], ...] = (
        "weight",
        "scales",
        "biases",
    )
    replacement_order: tuple[int, ...] = ROUTER_LAYER_INDICES
    parent_checkpoint_format: Literal["safetensors"] = "safetensors"
    overlay_format: Literal["safetensors"] = "safetensors"
    load_parent_before_overlay: Literal[True] = True
    verify_lineage_and_inventory_before_mutation: Literal[True] = True
    strict_tensor_loading: Literal[True] = True
    trust_remote_code: Literal[False] = False
    pickle_used: Literal[False] = False
    python_source_in_bundle: Literal[False] = False

    @model_validator(mode="after")
    def exact_replacement_recipe(self) -> Self:
        if self.parameters_per_gate != ("weight", "scales", "biases"):
            raise ValueError("each quantized gate requires weight, scales, and biases in order")
        if self.replacement_order != ROUTER_LAYER_INDICES:
            raise ValueError("quantized gates must replace layers zero through five in order")
        return self


class QuantizedRouterOverlayManifest(ImmutableQuantizedRouterRecord):
    """Complete immutable contract for the eighteen-tensor overlay."""

    schema_version: Literal["quantized-router-overlay-v1"] = "quantized-router-overlay-v1"
    quantizer_reference: Literal["mlx-0.32.0-affine-cpu-reference"] = (
        "mlx-0.32.0-affine-cpu-reference"
    )
    bits: QuantizationBits
    group_size: Literal[32] = AFFINE_GROUP_SIZE
    mode: Literal["affine"] = "affine"
    router_shape: tuple[Literal[4], Literal[288]] = ROUTER_SHAPE
    router_count: Literal[6] = ROUTER_TENSOR_COUNT
    tensor_count: Literal[18] = 18
    raw_tensor_bytes: int = Field(gt=0)
    serialization_format: Literal["safetensors"] = "safetensors"
    pickle_used: Literal[False] = False
    python_source_in_bundle: Literal[False] = False
    parent_model_id: str = Field(min_length=1)
    parent_model_revision: str = Field(pattern=_REVISION_PATTERN)
    parent_weights_sha256: str = Field(pattern=_SHA256_PATTERN)
    parent_state_dict_sha256: str = Field(pattern=_SHA256_PATTERN)
    learned_float_candidate_state_sha256: str = Field(pattern=_SHA256_PATTERN)
    learned_float_overlay_bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    parent_learned_float_lineage_sha256: str = Field(pattern=_SHA256_PATTERN)
    tensor_payload_sha256: str = Field(pattern=_SHA256_PATTERN)
    tensor_payload_hash_scheme: Literal[
        "sha256(sorted UTF-8 tensor name length/name + canonical tensor SHA-256 bytes)"
    ] = TENSOR_PAYLOAD_HASH_SCHEME
    routers: tuple[QuantizedRouterFact, ...]
    tensor_inventory: tuple[QuantizedTensorFact, ...]
    reconstruction: QuantizedRouterReconstructionRecipe

    @model_validator(mode="after")
    def exact_overlay_contract(self) -> Self:
        expected_bytes = expected_quantized_router_bytes(self.bits)
        if self.raw_tensor_bytes != expected_bytes:
            raise ValueError(
                f"{self.bits}-bit overlay must contain exactly {expected_bytes} raw tensor bytes"
            )
        if self.reconstruction.bits != self.bits:
            raise ValueError("reconstruction bits differ from the overlay manifest")
        if len(self.routers) != ROUTER_TENSOR_COUNT:
            raise ValueError("quantized overlay requires exactly six router facts")
        if tuple(item.layer_index for item in self.routers) != ROUTER_LAYER_INDICES:
            raise ValueError("quantized router facts must retain layers zero through five in order")

        expected_specs = _expected_tensor_specs(self.bits)
        if len(self.tensor_inventory) != QUANTIZED_ROUTER_TENSOR_COUNT:
            raise ValueError("quantized overlay requires exactly eighteen tensor facts")
        if tuple(item.name for item in self.tensor_inventory) != tuple(expected_specs):
            raise ValueError("tensor inventory is not in canonical six-gate order")
        for item, (name, spec) in zip(self.tensor_inventory, expected_specs.items(), strict=True):
            role, dtype, shape, nbytes, layer_index = spec
            if (
                item.name,
                item.role,
                item.dtype,
                item.shape,
                item.nbytes,
                item.layer_index,
            ) != (name, role, dtype, shape, nbytes, layer_index):
                raise ValueError(f"tensor fact differs from the exact contract for {name!r}")
        inventory_sha256 = tensor_payload_sha256(
            {item.name: item.tensor_sha256 for item in self.tensor_inventory}
        )
        if inventory_sha256 != self.tensor_payload_sha256:
            raise ValueError("tensor payload digest differs from the inventory hashes")

        for router in self.routers:
            names = {
                router.packed_weight_name,
                router.scales_name,
                router.biases_name,
            }
            inventory_names = {
                item.name
                for item in self.tensor_inventory
                if item.layer_index == router.layer_index
            }
            if names != inventory_names:
                raise ValueError("router fact references a different tensor inventory")
        secret_path = sensitive_literal_path(self.model_dump(mode="json"))
        if secret_path is not None:
            raise ValueError(
                "quantized router manifest contains credential-like material at "
                + ".".join(secret_path)
            )
        return self


class QuantizedRouterOverlayAudit(ImmutableQuantizedRouterRecord):
    """Successful recomputation of lineage, inventory, hash, and error facts."""

    manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    parent_learned_float_lineage_sha256: str = Field(pattern=_SHA256_PATTERN)
    tensor_payload_sha256: str = Field(pattern=_SHA256_PATTERN)
    bits: QuantizationBits
    router_count: Literal[6] = ROUTER_TENSOR_COUNT
    tensor_count: Literal[18] = 18
    raw_tensor_bytes: int = Field(gt=0)
    learned_float_parents_exact: Literal[True] = True
    tensor_inventory_exact: Literal[True] = True
    quantization_errors_exact: Literal[True] = True
    safe_reconstruction_only: Literal[True] = True


@dataclass(frozen=True)
class AffineQuantizedRouter:
    """CPU tensor result of the pinned affine quantization operation."""

    weight: Tensor
    scales: Tensor
    biases: Tensor


def _require_bits(bits: int) -> QuantizationBits:
    if type(bits) is not int or bits not in SUPPORTED_BITS:
        raise QuantizedRouterOverlayError("router affine bits must be exactly 4 or 8")
    return cast(QuantizationBits, bits)


def _checked_dense_cpu_tensor(name: str, tensor: Tensor) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise QuantizedRouterOverlayError(f"tensor {name!r} is not a torch.Tensor")
    if tensor.device.type == "meta":
        raise QuantizedRouterOverlayError(f"tensor {name!r} has no materialized bytes")
    if tensor.layout is not torch.strided or tensor.is_quantized:
        raise QuantizedRouterOverlayError(f"tensor {name!r} must be a dense non-quantized tensor")
    return tensor.detach().cpu().contiguous()


def _require_learned_router(name: str, tensor: Tensor) -> Tensor:
    normalized = _checked_dense_cpu_tensor(name, tensor)
    if tuple(normalized.shape) != ROUTER_SHAPE or normalized.dtype is not torch.float32:
        raise QuantizedRouterOverlayError(
            f"learned router {name!r} must have shape {ROUTER_SHAPE} and dtype torch.float32"
        )
    if not bool(torch.isfinite(normalized).all().item()):
        raise QuantizedRouterOverlayError(f"learned router {name!r} contains non-finite values")
    return normalized


def affine_quantize_router(weight: Tensor, *, bits: int) -> AffineQuantizedRouter:
    """Pack one ``(4, 288)`` float32 gate like MLX 0.32.0 affine quantization.

    Values are packed low-bit first into uint32 words.  Group scales and biases
    intentionally follow MLX's signed, endpoint-anchored affine algorithm.
    """

    resolved_bits = _require_bits(bits)
    normalized = _require_learned_router("<router>", weight)
    groups = normalized.reshape(ROUTER_OUTPUT_DIMS, AFFINE_GROUP_COUNT, AFFINE_GROUP_SIZE)
    group_min = groups.amin(dim=-1)
    group_max = groups.amax(dim=-1)
    n_bins = float((1 << resolved_bits) - 1)
    scale = torch.maximum(
        (group_max - group_min) / n_bins,
        torch.full_like(group_min, 1e-7),
    )
    min_is_larger = group_min.abs() > group_max.abs()
    scale = torch.where(min_is_larger, scale, -scale)
    edge = torch.where(min_is_larger, group_min, group_max)
    q0 = torch.round(edge / scale)
    anchored = q0 != 0
    scale = torch.where(anchored, edge / q0, scale)
    biases = torch.where(anchored, edge, torch.zeros_like(edge))

    quantized = torch.round((groups - biases.unsqueeze(-1)) / scale.unsqueeze(-1))
    quantized = quantized.clamp_(0, int(n_bins)).to(torch.int64)
    values_per_word = 32 // resolved_bits
    words_per_group = AFFINE_GROUP_SIZE // values_per_word
    unpacked_words = quantized.reshape(
        ROUTER_OUTPUT_DIMS, AFFINE_GROUP_COUNT, words_per_group, values_per_word
    )
    shifts = torch.arange(values_per_word, dtype=torch.int64) * resolved_bits
    packed = torch.sum(unpacked_words << shifts, dim=-1).to(torch.uint32)
    packed = packed.reshape(ROUTER_OUTPUT_DIMS, AFFINE_GROUP_COUNT * words_per_group)
    return AffineQuantizedRouter(
        weight=packed.contiguous(),
        scales=scale.to(torch.float32).contiguous(),
        biases=biases.to(torch.float32).contiguous(),
    )


def affine_dequantize_router(
    weight: Tensor,
    scales: Tensor,
    biases: Tensor,
    *,
    bits: int,
) -> Tensor:
    """Dequantize one packed gate using the pinned MLX affine equation."""

    resolved_bits = _require_bits(bits)
    normalized_weight = _checked_dense_cpu_tensor("weight", weight)
    normalized_scales = _checked_dense_cpu_tensor("scales", scales)
    normalized_biases = _checked_dense_cpu_tensor("biases", biases)
    values_per_word = 32 // resolved_bits
    words_per_group = AFFINE_GROUP_SIZE // values_per_word
    expected_weight_shape = (
        ROUTER_OUTPUT_DIMS,
        AFFINE_GROUP_COUNT * words_per_group,
    )
    if (
        tuple(normalized_weight.shape) != expected_weight_shape
        or normalized_weight.dtype is not torch.uint32
    ):
        raise QuantizedRouterOverlayError(
            f"packed weight must have shape {expected_weight_shape} and dtype torch.uint32"
        )
    for name, tensor in (("scales", normalized_scales), ("biases", normalized_biases)):
        if tuple(tensor.shape) != (ROUTER_OUTPUT_DIMS, AFFINE_GROUP_COUNT):
            raise QuantizedRouterOverlayError(f"{name} must have shape (4, 9)")
        if tensor.dtype is not torch.float32:
            raise QuantizedRouterOverlayError(f"{name} must have dtype torch.float32")
        if not bool(torch.isfinite(tensor).all().item()):
            raise QuantizedRouterOverlayError(f"{name} contains non-finite values")

    words = normalized_weight.to(torch.int64).reshape(
        ROUTER_OUTPUT_DIMS, AFFINE_GROUP_COUNT, words_per_group
    )
    shifts = torch.arange(values_per_word, dtype=torch.int64) * resolved_bits
    values = ((words.unsqueeze(-1) >> shifts) & ((1 << resolved_bits) - 1)).reshape(
        ROUTER_OUTPUT_DIMS, AFFINE_GROUP_COUNT, AFFINE_GROUP_SIZE
    )
    output = values.to(torch.float32) * normalized_scales.unsqueeze(
        -1
    ) + normalized_biases.unsqueeze(-1)
    return output.reshape(ROUTER_SHAPE).contiguous()


def affine_rounding_error_bound(scales: Tensor, reference_weight: Tensor) -> float:
    """Return a float32-safe maximum error bound for the affine reconstruction."""

    normalized_scales = _checked_dense_cpu_tensor("scales", scales)
    normalized_reference = _require_learned_router("<reference-router>", reference_weight)
    if (
        tuple(normalized_scales.shape) != (ROUTER_OUTPUT_DIMS, AFFINE_GROUP_COUNT)
        or normalized_scales.dtype is not torch.float32
    ):
        raise QuantizedRouterOverlayError("scales must have shape (4, 9) and dtype torch.float32")
    # MLX re-anchors the scale to one endpoint after rounding its zero code.
    # That can clip the opposite endpoint by less than one full quantization
    # step, so a half-step textbook bound would be too narrow.
    quantization_step = float(normalized_scales.abs().amax().item())
    magnitude = max(1.0, float(normalized_reference.abs().amax().item()))
    return quantization_step + 8.0 * torch.finfo(torch.float32).eps * magnitude


def expected_quantized_router_bytes(bits: int) -> int:
    """Return the exact raw tensor bytes for all six gates."""

    resolved_bits = _require_bits(bits)
    packed_bytes = ROUTER_OUTPUT_DIMS * (ROUTER_INPUT_DIMS * resolved_bits // 32) * 4
    affine_bytes = ROUTER_OUTPUT_DIMS * AFFINE_GROUP_COUNT * 4
    return ROUTER_TENSOR_COUNT * (packed_bytes + affine_bytes + affine_bytes)


def tensor_payload_sha256(tensor_hashes: Mapping[str, str]) -> str:
    """Hash a named tensor inventory independently of mapping iteration order."""

    digest = hashlib.sha256()
    for name in sorted(tensor_hashes):
        if not isinstance(name, str) or not name or "\0" in name:
            raise QuantizedRouterOverlayError(
                "tensor payload names must be non-empty strings without NUL"
            )
        tensor_sha256 = tensor_hashes[name]
        if len(tensor_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in tensor_sha256
        ):
            raise QuantizedRouterOverlayError(f"invalid canonical tensor hash for {name!r}")
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded_name)
        digest.update(bytes.fromhex(tensor_sha256))
    return digest.hexdigest()


def _expected_tensor_specs(
    bits: QuantizationBits,
) -> dict[
    str,
    tuple[
        Literal["packed_weight", "scales", "affine_biases"],
        Literal["uint32", "float32"],
        tuple[int, int],
        int,
        int,
    ],
]:
    packed_shape = (ROUTER_OUTPUT_DIMS, ROUTER_INPUT_DIMS * bits // 32)
    affine_shape = (ROUTER_OUTPUT_DIMS, AFFINE_GROUP_COUNT)
    specs: dict[
        str,
        tuple[
            Literal["packed_weight", "scales", "affine_biases"],
            Literal["uint32", "float32"],
            tuple[int, int],
            int,
            int,
        ],
    ] = {}
    for layer_index, weight_name in enumerate(ROUTER_WEIGHT_NAMES):
        module_name = weight_name.removesuffix(".weight")
        specs[weight_name] = (
            "packed_weight",
            "uint32",
            packed_shape,
            math.prod(packed_shape) * 4,
            layer_index,
        )
        specs[f"{module_name}.scales"] = (
            "scales",
            "float32",
            affine_shape,
            math.prod(affine_shape) * 4,
            layer_index,
        )
        specs[f"{module_name}.biases"] = (
            "affine_biases",
            "float32",
            affine_shape,
            math.prod(affine_shape) * 4,
            layer_index,
        )
    return specs


def _validate_learned_lineage(
    learned_router_state: QuantizedTensorState,
    learned_float_lineage: RouterOverlayLineage,
) -> dict[str, Tensor]:
    if learned_float_lineage.parent.router_tensor_names != ROUTER_WEIGHT_NAMES:
        raise QuantizedRouterOverlayError("learned lineage differs from the exact six gate names")
    if (
        learned_float_lineage.parent.router_shape != ROUTER_SHAPE
        or learned_float_lineage.parent.router_dtype != "torch.float32"
    ):
        raise QuantizedRouterOverlayError(
            "learned lineage routers must have shape (4, 288) and dtype torch.float32"
        )
    if set(learned_router_state) != set(ROUTER_WEIGHT_NAMES):
        missing = sorted(set(ROUTER_WEIGHT_NAMES) - set(learned_router_state))
        extra = sorted(set(learned_router_state) - set(ROUTER_WEIGHT_NAMES))
        raise QuantizedRouterOverlayError(
            f"learned payload must contain exactly six gates; missing={missing!r}, extra={extra!r}"
        )
    normalized: dict[str, Tensor] = {}
    for record, expected_name in zip(
        learned_float_lineage.router_tensors, ROUTER_WEIGHT_NAMES, strict=True
    ):
        if (
            record.name != expected_name
            or record.shape != ROUTER_SHAPE
            or record.dtype != "torch.float32"
        ):
            raise QuantizedRouterOverlayError(
                f"learned lineage tensor contract differs for {expected_name!r}"
            )
        tensor = _require_learned_router(expected_name, learned_router_state[expected_name])
        if canonical_tensor_sha256(tensor) != record.overlay_tensor_sha256:
            raise QuantizedRouterOverlayError(
                f"learned router hash differs from its lineage for {expected_name!r}"
            )
        normalized[expected_name] = tensor
    return normalized


def _actual_tensor_fact(
    name: str,
    tensor: Tensor,
    *,
    bits: QuantizationBits,
) -> QuantizedTensorFact:
    specs = _expected_tensor_specs(bits)
    if name not in specs:
        raise QuantizedRouterOverlayError(f"unexpected quantized tensor {name!r}")
    role, dtype, shape, nbytes, layer_index = specs[name]
    normalized = _checked_dense_cpu_tensor(name, tensor)
    actual_dtype = "uint32" if normalized.dtype is torch.uint32 else "float32"
    if (
        actual_dtype != dtype
        or normalized.dtype not in (torch.uint32, torch.float32)
        or tuple(normalized.shape) != shape
        or normalized.numel() * normalized.element_size() != nbytes
    ):
        raise QuantizedRouterOverlayError(f"quantized tensor spec differs for {name!r}")
    return QuantizedTensorFact(
        name=name,
        layer_index=layer_index,
        role=role,
        dtype=dtype,
        shape=shape,
        nbytes=nbytes,
        tensor_sha256=canonical_tensor_sha256(normalized),
    )


def build_quantized_router_overlay(
    learned_router_state: QuantizedTensorState,
    learned_float_lineage: RouterOverlayLineage,
    *,
    bits: int,
) -> tuple[dict[str, Tensor], QuantizedRouterOverlayManifest]:
    """Build an eighteen-tensor overlay and its exact learned-float lineage contract."""

    resolved_bits = _require_bits(bits)
    learned = _validate_learned_lineage(learned_router_state, learned_float_lineage)
    tensors: dict[str, Tensor] = {}
    routers: list[QuantizedRouterFact] = []
    for record in learned_float_lineage.router_tensors:
        module_name = record.name.removesuffix(".weight")
        quantized = affine_quantize_router(learned[record.name], bits=resolved_bits)
        tensors[record.name] = quantized.weight
        tensors[f"{module_name}.scales"] = quantized.scales
        tensors[f"{module_name}.biases"] = quantized.biases
        dequantized = affine_dequantize_router(
            quantized.weight, quantized.scales, quantized.biases, bits=resolved_bits
        )
        difference = learned[record.name] - dequantized
        bound = affine_rounding_error_bound(quantized.scales, learned[record.name])
        max_abs_error = float(difference.abs().amax().item())
        if max_abs_error > bound:
            raise QuantizedRouterOverlayError(
                f"reference quantization error exceeds its bound for {record.name!r}"
            )
        routers.append(
            QuantizedRouterFact(
                layer_index=record.layer_index,
                gate_module_name=module_name,
                learned_float_weight_name=record.name,
                learned_float_weight_sha256=record.overlay_tensor_sha256,
                packed_weight_name=record.name,
                scales_name=f"{module_name}.scales",
                biases_name=f"{module_name}.biases",
                max_abs_error=max_abs_error,
                rmse=float(torch.sqrt(torch.mean(difference.square())).item()),
                max_rounding_error_bound=bound,
            )
        )

    inventory = tuple(
        _actual_tensor_fact(name, tensors[name], bits=resolved_bits)
        for name in _expected_tensor_specs(resolved_bits)
    )
    payload_sha256 = tensor_payload_sha256({item.name: item.tensor_sha256 for item in inventory})
    manifest = QuantizedRouterOverlayManifest(
        bits=resolved_bits,
        raw_tensor_bytes=sum(item.nbytes for item in inventory),
        parent_model_id=learned_float_lineage.parent.model_id,
        parent_model_revision=learned_float_lineage.parent.revision,
        parent_weights_sha256=learned_float_lineage.parent.weights_sha256,
        parent_state_dict_sha256=learned_float_lineage.parent.state_dict_sha256,
        learned_float_candidate_state_sha256=(learned_float_lineage.reload.candidate_state_sha256),
        learned_float_overlay_bundle_sha256=(learned_float_lineage.reload.overlay_bundle_sha256),
        parent_learned_float_lineage_sha256=learned_float_lineage.sha256(),
        tensor_payload_sha256=payload_sha256,
        routers=tuple(routers),
        tensor_inventory=inventory,
        reconstruction=QuantizedRouterReconstructionRecipe(bits=resolved_bits),
    )
    return tensors, manifest


def audit_quantized_router_overlay(
    quantized_router_state: QuantizedTensorState,
    manifest: QuantizedRouterOverlayManifest,
    *,
    learned_router_state: QuantizedTensorState,
    learned_float_lineage: RouterOverlayLineage,
) -> QuantizedRouterOverlayAudit:
    """Recompute every external lineage, tensor, and quantization-error fact."""

    learned = _validate_learned_lineage(learned_router_state, learned_float_lineage)
    if manifest.parent_learned_float_lineage_sha256 != learned_float_lineage.sha256():
        raise QuantizedRouterOverlayError("learned-float lineage hash differs from the manifest")
    lineage_facts = (
        manifest.parent_model_id,
        manifest.parent_model_revision,
        manifest.parent_weights_sha256,
        manifest.parent_state_dict_sha256,
        manifest.learned_float_candidate_state_sha256,
        manifest.learned_float_overlay_bundle_sha256,
    )
    actual_lineage_facts = (
        learned_float_lineage.parent.model_id,
        learned_float_lineage.parent.revision,
        learned_float_lineage.parent.weights_sha256,
        learned_float_lineage.parent.state_dict_sha256,
        learned_float_lineage.reload.candidate_state_sha256,
        learned_float_lineage.reload.overlay_bundle_sha256,
    )
    if lineage_facts != actual_lineage_facts:
        raise QuantizedRouterOverlayError("learned-float parent facts differ from the manifest")

    expected_names = tuple(_expected_tensor_specs(manifest.bits))
    if set(quantized_router_state) != set(expected_names):
        missing = sorted(set(expected_names) - set(quantized_router_state))
        extra = sorted(set(quantized_router_state) - set(expected_names))
        raise QuantizedRouterOverlayError(
            f"quantized payload inventory differs; missing={missing!r}, extra={extra!r}"
        )
    facts = tuple(
        _actual_tensor_fact(name, quantized_router_state[name], bits=manifest.bits)
        for name in expected_names
    )
    if facts != manifest.tensor_inventory:
        raise QuantizedRouterOverlayError("quantized tensor spec, bytes, or hash differs")
    payload_sha256 = tensor_payload_sha256({item.name: item.tensor_sha256 for item in facts})
    if payload_sha256 != manifest.tensor_payload_sha256:
        raise QuantizedRouterOverlayError("quantized tensor payload digest differs")

    recomputed_routers: list[QuantizedRouterFact] = []
    for record in learned_float_lineage.router_tensors:
        module_name = record.name.removesuffix(".weight")
        dequantized = affine_dequantize_router(
            quantized_router_state[record.name],
            quantized_router_state[f"{module_name}.scales"],
            quantized_router_state[f"{module_name}.biases"],
            bits=manifest.bits,
        )
        difference = learned[record.name] - dequantized
        bound = affine_rounding_error_bound(
            quantized_router_state[f"{module_name}.scales"], learned[record.name]
        )
        recomputed_routers.append(
            QuantizedRouterFact(
                layer_index=record.layer_index,
                gate_module_name=module_name,
                learned_float_weight_name=record.name,
                learned_float_weight_sha256=record.overlay_tensor_sha256,
                packed_weight_name=record.name,
                scales_name=f"{module_name}.scales",
                biases_name=f"{module_name}.biases",
                max_abs_error=float(difference.abs().amax().item()),
                rmse=float(torch.sqrt(torch.mean(difference.square())).item()),
                max_rounding_error_bound=bound,
            )
        )
    if tuple(recomputed_routers) != manifest.routers:
        raise QuantizedRouterOverlayError("quantization error or learned parent facts differ")
    return QuantizedRouterOverlayAudit(
        manifest_sha256=manifest.sha256(),
        parent_learned_float_lineage_sha256=learned_float_lineage.sha256(),
        tensor_payload_sha256=payload_sha256,
        bits=manifest.bits,
        raw_tensor_bytes=sum(item.nbytes for item in facts),
    )


__all__ = [
    "AFFINE_GROUP_SIZE",
    "ROUTER_SHAPE",
    "ROUTER_WEIGHT_NAMES",
    "AffineQuantizedRouter",
    "QuantizedRouterFact",
    "QuantizedRouterOverlayAudit",
    "QuantizedRouterOverlayError",
    "QuantizedRouterOverlayManifest",
    "QuantizedRouterReconstructionRecipe",
    "QuantizedTensorFact",
    "affine_dequantize_router",
    "affine_quantize_router",
    "affine_rounding_error_bound",
    "audit_quantized_router_overlay",
    "build_quantized_router_overlay",
    "expected_quantized_router_bytes",
    "tensor_payload_sha256",
]
