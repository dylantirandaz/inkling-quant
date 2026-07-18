#!/usr/bin/env python3
"""Evaluate fully quantized MLX Mixtral checkpoints without network or model code.

The parent process performs a fail-closed static audit before any MLX import.  It
then starts one fresh worker process per checkpoint so host-RSS and MLX allocator
measurements do not inherit another model's residency. Prompts and generated
tokens stay in memory. Raw route IDs cross the subprocess boundary only through
captured standard output and are omitted from the immutable final record.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from ctypes.util import find_library
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_ID = "ggml-org/stories15M_MOE"
MODEL_REVISION = "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
MODEL_LICENSE = "MIT"
MODEL_CONFIG_SHA256 = "e901e012953d1df93574b2cc3d7db5ed4758d52f8bd4a7dd4b647936e32261be"
SOURCE_TOKENIZER_CONFIG_SHA256 = "33d29c87e41f7dd1efb0434d852730320c82970f292be452d820539bce417052"
SOURCE_TOKENIZER_SHA256 = "8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1"
CONVERTED_TOKENIZER_CONFIG_SHA256 = (
    "95b5a0061c68c562ba0459f3409f3ff459db78fe201b4a3534d9f38d0c2c6238"
)
CONVERTED_TOKENIZER_SHA256 = "ed7c4e86a6d2b2a24cf7fb6e6d96c445a97bf9a22d682f598869afc2625bfd1f"
SOURCE_WEIGHT_SHA256 = "dbfa0289f68a8dd721d10eb12d8bd82e098455682027f6f9986ba548913f9082"
SOURCE_WEIGHT_SIZE_BYTES = 72_744_704
DATASET_ID = "hf://datasets/roneneldan/TinyStories/TinyStories-valid.txt"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
DATASET_SPLIT = "validation"
DATASET_LICENSE = "cdla-sharing-1.0"
DATASET_SHA256 = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
STORY_SEPARATOR = "<|endoftext|>"
SEED = 20_260_715
MAX_CONTEXT_TOKENS = 256
MAX_NEW_TOKENS = 8
PERPLEXITY_IDS = tuple(f"story-{index:06d}" for index in range(1, 33))
GENERATION_IDS = ("story-000001", "story-000008", "story-000016", "story-000024")
BENCHMARK_SAMPLE_ID = "story-000008"
PROMPT_TEMPLATE = "{text}"
PROMPT_TEMPLATE_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()

EXPECTED_CONVERSIONS: dict[str, dict[str, Any]] = {
    "float32": {
        "bits": None,
        "config_sha256": "8297307683b0a6b694ea74d5472b2fd6571ef880bec3bca4d45666e6cecf8495",
        "weight_sha256": "880068eacd8de271e4fd28e54e23fd2b4a1e9b3f9795c84581b534c56a562e3b",
        "weight_size_bytes": 145_441_603,
        "tensor_bytes": 145_434_240,
        "index_sha256": "907f7ce42994a8bef30158af095024674a6d4c46147cbc7936ee73a1be9c3bfc",
    },
    "mlx_affine_q4_g32": {
        "bits": 4,
        "config_sha256": "1cb73e9e5c63bb78a15bb07b4189fc1e528c180ad1dd8bfb6d89cbcd8afb283f",
        "weight_sha256": "610c0f80ed9e59648c652aa4e8c9227b59be1286c55cccad0d5356c064b20552",
        "weight_size_bytes": 27_299_860,
        "tensor_bytes": 27_281_088,
        "index_sha256": "a3c277e527d0ed45ba4a690ea3a102fd75f6ab181bd87db9a2fe652936cbb6c9",
    },
    "mlx_affine_q8_g32": {
        "bits": 8,
        "config_sha256": "97b00d3dcda1aef2d5c98b002d3c229fc38b29983c1354f9279d23a15b02221d",
        "weight_sha256": "a2a5a57afbad4aa8378fea33a6b51b0848f7cd665d3ed191bad80704d3756dec",
        "weight_size_bytes": 45_477_324,
        "tensor_bytes": 45_458_496,
        "index_sha256": "8bbbb87885bc997cc47c5d7888e64e86174275af8069a72208abbc2b6a4aa998",
    },
}

EXPECTED_MLX_LM_SOURCE_HASHES = {
    "convert.py": "dc60df164c2d51ee2f05f5f9f3324bc3a44a59dd2ccddb75dde680e854ce5e9a",
    "utils.py": "ba0371e9c88d52b34d71271945c2394005fbcb2bfb2ee9f6f82d627a33b72422",
    "models/mixtral.py": "a7d15990aa42b81b659c8679089b6f1571225466825c71d9350eace1964c3b5c",
    "models/switch_layers.py": ("073a6a808d5c90bb699a2ecca0e559b06727ae96dbc1f0253e4c7e77e4ee1ef2"),
    "generate.py": "270778ad53eaca55a8533d82e6752660fe5d2605c4aa0879b48a50a91f69345f",
    "tokenizer_utils.py": "25784bb03c922d0d7832ce6c66a6cd4eb3a4820b6c5a8e583dedb63a018fb56a",
}
EXPECTED_MLX_QUANTIZED_LAYER_SHA256 = (
    "e6c34d65bfb9f1c6f35ade3b3c7c021693c60afa31eea922a78ddceca96b3f37"
)
EXPECTED_RUNTIME_VERSIONS = {
    "mlx": "0.32.0",
    "mlx-lm": "0.31.3",
    "mlx-metal": "0.32.0",
    "safetensors": "0.8.0",
    "transformers": "5.12.1",
}

_SOURCE_SUFFIXES = frozenset({".json", ".safetensors", ".txt"})
_SOURCE_FILES = frozenset(
    {
        "config.json",
        "data.txt",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "moe_shakespeare15M/checkpoint-400/adapter_config.json",
        "moe_shakespeare15M/checkpoint-400/trainer_state.json",
        "moe_shakespeare15M/checkpoint-500/adapter_config.json",
        "moe_shakespeare15M/checkpoint-500/trainer_state.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
_CONVERSION_FILES = frozenset(
    {
        "README.md",
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
_EXECUTABLE_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".dylib", ".exe", ".joblib", ".pkl", ".pickle", ".pt", ".pth", ".py", ".so"}
)
_SENSITIVE_LABEL = re.compile(
    r"(?i)(authorization|bearer\s|password|secret|token\s*[=:]|hf_[a-z0-9]{8,}|sk-[a-z0-9])"
)


class _MachTimeValue(ctypes.Structure):
    _fields_ = (("seconds", ctypes.c_int32), ("microseconds", ctypes.c_int32))


class _MachTaskBasicInfo(ctypes.Structure):
    _fields_ = (
        ("virtual_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_max", ctypes.c_uint64),
        ("user_time", _MachTimeValue),
        ("system_time", _MachTimeValue),
        ("policy", ctypes.c_int32),
        ("suspend_count", ctypes.c_int32),
    )


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 for one regular file or safe symlink target."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _token_ids_sha256(tokens: Sequence[int]) -> str:
    payload = json.dumps(
        list(tokens),
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_model_config_security(config: Mapping[str, Any], *, source: str) -> None:
    """Reject every MLX-LM snapshot hook that can select repository Python."""

    if config.get("model_file") is not None:
        raise ValueError(f"{source} config.model_file must be absent or null")
    if config.get("auto_map") is not None:
        raise ValueError(f"{source} config.auto_map must be absent or null")
    text_config = config.get("text_config")
    if isinstance(text_config, Mapping) and (
        text_config.get("model_file") is not None or text_config.get("auto_map") is not None
    ):
        raise ValueError(f"{source} nested text config selects remote/custom code")
    if config.get("model_type") != "mixtral":
        raise ValueError(f"{source} must select MLX-LM's built-in mixtral implementation")
    if config.get("architectures") != ["MixtralForCausalLM"]:
        raise ValueError(f"{source} architecture must be exactly MixtralForCausalLM")


def validate_tokenizer_config_security(
    config: Mapping[str, Any],
    *,
    source: str,
    expected_class: str,
) -> None:
    """Require one built-in tokenizer path and reject dynamic helper selectors."""

    if config.get("tokenizer_class") != expected_class:
        raise ValueError(f"{source} tokenizer_class must be exactly {expected_class}")
    for field in (
        "auto_map",
        "chat_template",
        "chat_template_type",
        "tool_parser_type",
    ):
        if config.get(field) is not None:
            raise ValueError(f"{source} tokenizer config field {field} must be absent or null")
    if config.get("trust_remote_code") is True:
        raise ValueError(f"{source} tokenizer config must not enable remote code")


def _audit_tree(
    root: Path,
    *,
    allowed_suffixes: frozenset[str],
    allow_hf_cache_symlinks: bool,
) -> list[dict[str, Any]]:
    root = root.expanduser().resolve(strict=True)
    allowed_resolution_root = root.parents[1] if allow_hf_cache_symlinks else root
    inventory: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root).as_posix()
        is_symlink = candidate.is_symlink()
        if is_symlink and candidate.is_dir():
            raise ValueError(f"model tree must not contain directory symlinks: {relative}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"unsupported non-file snapshot entry: {candidate}")
        suffix = candidate.suffix.lower()
        if suffix in _EXECUTABLE_SUFFIXES or suffix not in allowed_suffixes:
            raise ValueError(f"unsafe or unsupported model file: {relative}")
        if is_symlink:
            if not allow_hf_cache_symlinks:
                raise ValueError(f"converted checkpoint must not contain symlinks: {relative}")
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(allowed_resolution_root):
                raise ValueError(f"snapshot symlink escapes its repository cache: {relative}")
        mode = candidate.stat().st_mode & 0o777
        if mode & 0o111:
            raise ValueError(f"model file must not be executable: {relative}")
        inventory.append(
            {
                "path": relative,
                "size_bytes": candidate.stat().st_size,
                "sha256": file_sha256(candidate),
                "symlink": is_symlink,
            }
        )
    if not inventory:
        raise ValueError(f"model directory is empty: {root}")
    return inventory


def audit_source_snapshot(
    snapshot: Path,
    *,
    expected_revision: str = MODEL_REVISION,
    expected_config_sha256: str = MODEL_CONFIG_SHA256,
    expected_tokenizer_config_sha256: str = SOURCE_TOKENIZER_CONFIG_SHA256,
    expected_tokenizer_sha256: str = SOURCE_TOKENIZER_SHA256,
    expected_weight_sha256: str = SOURCE_WEIGHT_SHA256,
    expected_weight_size_bytes: int = SOURCE_WEIGHT_SIZE_BYTES,
    expected_file_set: frozenset[str] | None = _SOURCE_FILES,
    expected_safetensors_tensor_count: int | None = 117,
) -> dict[str, Any]:
    """Prove this local source cannot activate MLX-LM's custom-model path."""

    snapshot = snapshot.expanduser().resolve(strict=True)
    if snapshot.name != expected_revision:
        raise ValueError(
            f"source snapshot directory must be the immutable revision {expected_revision}"
        )
    inventory = _audit_tree(
        snapshot,
        allowed_suffixes=_SOURCE_SUFFIXES,
        allow_hf_cache_symlinks=True,
    )
    names = {item["path"] for item in inventory}
    if expected_file_set is not None and names != expected_file_set:
        raise ValueError(
            "source snapshot file set mismatch; "
            f"missing={sorted(expected_file_set - names)}, "
            f"extra={sorted(names - expected_file_set)}"
        )
    config_path = snapshot / "config.json"
    tokenizer_config_path = snapshot / "tokenizer_config.json"
    tokenizer_path = snapshot / "tokenizer.json"
    weight_path = snapshot / "model.safetensors"
    if file_sha256(config_path) != expected_config_sha256:
        raise ValueError("source config.json checksum mismatch")
    if file_sha256(tokenizer_config_path) != expected_tokenizer_config_sha256:
        raise ValueError("source tokenizer_config.json checksum mismatch")
    if file_sha256(tokenizer_path) != expected_tokenizer_sha256:
        raise ValueError("source tokenizer.json checksum mismatch")
    if weight_path.stat().st_size != expected_weight_size_bytes:
        raise ValueError("source model.safetensors byte-size mismatch")
    if file_sha256(weight_path) != expected_weight_sha256:
        raise ValueError("source model.safetensors checksum mismatch")
    if expected_safetensors_tensor_count is not None:
        from safetensors import safe_open

        with safe_open(str(weight_path), framework="np") as handle:
            if (
                handle.metadata() != {"format": "pt"}
                or len(handle.keys()) != expected_safetensors_tensor_count
            ):
                raise ValueError("source model.safetensors structure/metadata mismatch")
            expert_identity_groups: list[dict[str, Any]] = []
            for layer in range(6):
                for projection in ("w1", "w2", "w3"):
                    hashes: list[str] = []
                    dtype: str | None = None
                    shape: list[int] | None = None
                    for expert in range(4):
                        key = (
                            f"model.layers.{layer}.block_sparse_moe.experts."
                            f"{expert}.{projection}.weight"
                        )
                        tensor = handle.get_tensor(key)
                        hashes.append(hashlib.sha256(tensor.tobytes()).hexdigest())
                        dtype = str(tensor.dtype)
                        shape = [int(value) for value in tensor.shape]
                    if len(set(hashes)) != 1:
                        raise ValueError(
                            f"source expert tensors differ in layer {layer} {projection}"
                        )
                    expert_identity_groups.append(
                        {
                            "layer_id": layer,
                            "projection": projection,
                            "expert_count": 4,
                            "all_four_byte_identical": True,
                            "representative_tensor_sha256": hashes[0],
                            "dtype": dtype,
                            "shape": shape,
                        }
                    )
    else:
        expert_identity_groups = []
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_model_config_security(config, source="source snapshot")
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    validate_tokenizer_config_security(
        tokenizer_config,
        source="source snapshot",
        expected_class="LlamaTokenizer",
    )
    if any(item["path"].endswith(".py") for item in inventory):
        raise ValueError("source snapshot contains Python despite the allowlist")
    return {
        "location": "local_huggingface_cache_snapshot",
        "revision": expected_revision,
        "config_sha256": expected_config_sha256,
        "tokenizer_config_sha256": expected_tokenizer_config_sha256,
        "tokenizer_sha256": expected_tokenizer_sha256,
        "weight_file": "model.safetensors",
        "weight_sha256": expected_weight_sha256,
        "weight_size_bytes": expected_weight_size_bytes,
        "safetensors_tensor_count": expected_safetensors_tensor_count,
        "safetensors_metadata": (
            {"format": "pt"} if expected_safetensors_tensor_count is not None else None
        ),
        "expert_tensor_identity": {
            "measurement": "direct SHA-256 of every source expert tensor byte sequence",
            "group_count": len(expert_identity_groups),
            "all_groups_repeat_four_identical_experts": bool(expert_identity_groups),
            "groups": expert_identity_groups,
        },
        "file_inventory": inventory,
        "model_file": None,
        "auto_map": None,
        "tokenizer_class": "LlamaTokenizer",
        "python_file_count": 0,
        "code_execution_guard": (
            "audited before importing MLX; local path; built-in mlx_lm.models.mixtral; "
            "config.model_file and auto_map null; no Python/native/pickle files"
        ),
    }


def _expected_quantized_scale_keys() -> set[str]:
    keys = {"model.embed_tokens.scales", "lm_head.scales"}
    for layer in range(6):
        prefix = f"model.layers.{layer}"
        keys.add(f"{prefix}.block_sparse_moe.gate.scales")
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            keys.add(f"{prefix}.self_attn.{projection}.scales")
        for projection in ("gate_proj", "up_proj", "down_proj"):
            keys.add(f"{prefix}.block_sparse_moe.switch_mlp.{projection}.scales")
    return keys


def _expected_runtime_quantized_linear_names() -> set[str]:
    names = {"lm_head"}
    for layer in range(6):
        prefix = f"model.layers.{layer}"
        names.add(f"{prefix}.block_sparse_moe.gate")
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.add(f"{prefix}.self_attn.{projection}")
    return names


def _expected_runtime_quantized_expert_names() -> set[str]:
    return {
        f"model.layers.{layer}.block_sparse_moe.switch_mlp.{projection}"
        for layer in range(6)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }


def validate_checkpoint_tensor_facts(
    facts: Mapping[str, tuple[str, tuple[int, ...], int]],
    *,
    bits: int,
) -> dict[str, Any]:
    """Validate all eligible leaves and the fused expert tensors are truly packed."""

    if bits not in (4, 8):
        raise ValueError("only the evidenced affine 4-bit and 8-bit formats are supported")
    scale_keys = {key for key in facts if key.endswith(".scales")}
    expected_scales = _expected_quantized_scale_keys()
    if scale_keys != expected_scales:
        missing = sorted(expected_scales - scale_keys)
        extra = sorted(scale_keys - expected_scales)
        raise ValueError(f"quantized scale-key mismatch; missing={missing}, extra={extra}")
    bias_keys = {key for key in facts if key.endswith(".biases")}
    expected_biases = {key.removesuffix(".scales") + ".biases" for key in expected_scales}
    if bias_keys != expected_biases:
        raise ValueError("affine zero-point bias tensors do not match the quantized leaves")
    expert_prefixes = []
    for layer in range(6):
        for projection, output_dims, input_dims in (
            ("gate_proj", 768, 288),
            ("up_proj", 768, 288),
            ("down_proj", 288, 768),
        ):
            prefix = f"model.layers.{layer}.block_sparse_moe.switch_mlp.{projection}"
            expert_prefixes.append(prefix)
            weight = facts.get(f"{prefix}.weight")
            scales = facts.get(f"{prefix}.scales")
            biases = facts.get(f"{prefix}.biases")
            packed_width = input_dims * bits // 32
            groups = input_dims // 32
            if weight is None or weight[:2] != ("uint32", (4, output_dims, packed_width)):
                raise ValueError(f"fused expert weight is not {bits}-bit uint32 packed: {prefix}")
            expected_aux = ("float32", (4, output_dims, groups))
            if scales is None or scales[:2] != expected_aux:
                raise ValueError(f"fused expert scales have the wrong shape/dtype: {prefix}")
            if biases is None or biases[:2] != expected_aux:
                raise ValueError(f"fused expert affine biases have the wrong shape/dtype: {prefix}")
    for scale_key in expected_scales:
        prefix = scale_key.removesuffix(".scales")
        weight = facts.get(f"{prefix}.weight")
        scales = facts[scale_key]
        biases = facts[f"{prefix}.biases"]
        if weight is None or weight[0] != "uint32":
            raise ValueError(f"quantized leaf is missing a uint32 packed weight: {prefix}")
        if scales[0] != "float32" or biases[0] != "float32":
            raise ValueError(f"quantized leaf has non-float32 affine metadata: {prefix}")
    return {
        "quantized_leaf_count": len(expected_scales),
        "quantized_fused_expert_projection_count": len(expert_prefixes),
        "fused_expert_layers": 6,
        "experts_per_layer": 4,
        "expert_projection_names": expert_prefixes,
        "packed_weight_dtype": "uint32",
        "scale_dtype": "float32",
        "affine_bias_dtype": "float32",
        "group_size": 32,
        "bits": bits,
    }


def audit_converted_checkpoint(path: Path, *, label: str) -> dict[str, Any]:
    """Validate a deterministic FP32 control or full eligible-leaf quantization."""

    expected = EXPECTED_CONVERSIONS[label]
    path = path.expanduser().resolve(strict=True)
    inventory = _audit_tree(
        path,
        allowed_suffixes=frozenset({".json", ".md", ".safetensors"}),
        allow_hf_cache_symlinks=False,
    )
    names = {item["path"] for item in inventory}
    if names != _CONVERSION_FILES:
        raise ValueError(
            f"converted checkpoint file set mismatch; missing={sorted(_CONVERSION_FILES - names)}, "
            f"extra={sorted(names - _CONVERSION_FILES)}"
        )
    config_path = path / "config.json"
    tokenizer_config_path = path / "tokenizer_config.json"
    tokenizer_path = path / "tokenizer.json"
    weight_path = path / "model.safetensors"
    index_path = path / "model.safetensors.index.json"
    actual_hashes = {
        "config_sha256": file_sha256(config_path),
        "weight_sha256": file_sha256(weight_path),
        "index_sha256": file_sha256(index_path),
    }
    for name, actual in actual_hashes.items():
        if actual != expected[name]:
            raise ValueError(
                f"{label} {name} mismatch: expected {expected[name]}, received {actual}"
            )
    if weight_path.stat().st_size != expected["weight_size_bytes"]:
        raise ValueError(f"{label} serialized weight byte-size mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_model_config_security(config, source=label)
    if file_sha256(tokenizer_config_path) != CONVERTED_TOKENIZER_CONFIG_SHA256:
        raise ValueError(f"{label} tokenizer_config.json checksum mismatch")
    if file_sha256(tokenizer_path) != CONVERTED_TOKENIZER_SHA256:
        raise ValueError(f"{label} tokenizer.json checksum mismatch")
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    validate_tokenizer_config_security(
        tokenizer_config,
        source=label,
        expected_class="TokenizersBackend",
    )
    quantization = config.get("quantization")
    bits = expected["bits"]
    if bits is None:
        if quantization is not None or config.get("quantization_config") is not None:
            raise ValueError("float32 control must not declare quantization")
    elif quantization != {"bits": bits, "group_size": 32, "mode": "affine"}:
        raise ValueError(f"{label} quantization config is not the evidenced affine contract")

    from safetensors import safe_open

    facts: dict[str, tuple[str, tuple[int, ...], int]] = {}
    with safe_open(str(weight_path), framework="np") as handle:
        if handle.metadata() != {"format": "mlx"}:
            raise ValueError(f"{label} safetensors metadata is not MLX format")
        for key in handle.keys():  # noqa: SIM118 - safetensors.safe_open is not iterable
            tensor = handle.get_tensor(key)
            facts[key] = (str(tensor.dtype), tuple(int(v) for v in tensor.shape), tensor.nbytes)
    tensor_bytes = sum(fact[2] for fact in facts.values())
    if tensor_bytes != expected["tensor_bytes"]:
        raise ValueError(f"{label} tensor-byte total mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("metadata", {}).get("total_size") != tensor_bytes:
        raise ValueError(f"{label} index total_size does not match tensor bytes")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or set(weight_map) != set(facts):
        raise ValueError(f"{label} index weight map does not exactly cover tensor keys")
    if set(weight_map.values()) != {"model.safetensors"}:
        raise ValueError(f"{label} index points outside its one safetensors shard")

    if bits is None:
        if len(facts) != 63 or {fact[0] for fact in facts.values()} != {"float32"}:
            raise ValueError("float32 control must contain exactly 63 float32 tensors")
        quantized_facts: dict[str, Any] | None = None
    else:
        quantized_facts = validate_checkpoint_tensor_facts(facts, bits=bits)
        if len(facts) != 163:
            raise ValueError(f"{label} must contain 163 weight/scale/affine-bias tensors")
    return {
        "label": label,
        "location": f"temporary_local_checkpoint/{label}",
        "config_sha256": actual_hashes["config_sha256"],
        "tokenizer_config_sha256": CONVERTED_TOKENIZER_CONFIG_SHA256,
        "tokenizer_sha256": CONVERTED_TOKENIZER_SHA256,
        "weight_sha256": actual_hashes["weight_sha256"],
        "weight_size_bytes": weight_path.stat().st_size,
        "index_sha256": actual_hashes["index_sha256"],
        "tensor_bytes": tensor_bytes,
        "stored_tensor_count": len(facts),
        "bundle_size_bytes": sum(item["size_bytes"] for item in inventory),
        "file_inventory": inventory,
        "quantization": quantization,
        "quantized_tensor_proof": quantized_facts,
        "model_file": None,
        "auto_map": None,
        "tokenizer_class": "TokenizersBackend",
        "python_file_count": 0,
        "symlink_count": 0,
    }


def split_stories(text: str) -> tuple[str, ...]:
    """Split the official flat TinyStories file into stable nonempty stories."""

    return tuple(story.strip() for story in text.split(STORY_SEPARATOR) if story.strip())


def story_index(sample_id: str) -> int:
    """Map the established story-NNNNNN identity to the source segment index."""

    if not sample_id.startswith("story-") or len(sample_id) != 12:
        raise ValueError(f"unsupported TinyStories sample ID: {sample_id}")
    suffix = sample_id.removeprefix("story-")
    if not suffix.isdigit():
        raise ValueError(f"unsupported TinyStories sample ID: {sample_id}")
    return int(suffix)


def select_stories(stories: Sequence[str], sample_ids: Sequence[str]) -> tuple[str, ...]:
    """Select unique stable IDs in declared order."""

    indexes = tuple(story_index(sample_id) for sample_id in sample_ids)
    if len(set(indexes)) != len(indexes):
        raise ValueError("TinyStories sample IDs must be unique")
    if not indexes or min(indexes) < 0 or max(indexes) >= len(stories):
        raise ValueError("TinyStories sample selection is empty or outside the dataset")
    return tuple(stories[index] for index in indexes)


def selection_sha256(sample_ids: Sequence[str], stories: Sequence[str]) -> str:
    records = [
        {
            "sample_id": sample_id,
            "content_sha256": hashlib.sha256(story.encode("utf-8")).hexdigest(),
        }
        for sample_id, story in zip(sample_ids, stories, strict=True)
    ]
    return _canonical_sha256(records)


def current_process_rss_bytes() -> int:
    """Read current macOS process RSS using Mach task_info, never ru_maxrss."""

    if platform.system() != "Darwin":
        raise OSError("this MLX/Metal evidence collector supports macOS only")
    library = ctypes.CDLL(find_library("System") or "/usr/lib/libSystem.B.dylib")
    library.mach_task_self.restype = ctypes.c_uint32
    library.task_info.argtypes = (
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    )
    library.task_info.restype = ctypes.c_int32
    info = _MachTaskBasicInfo()
    count = ctypes.c_uint32(ctypes.sizeof(info) // ctypes.sizeof(ctypes.c_uint32))
    status = int(
        library.task_info(
            library.mach_task_self(),
            20,
            ctypes.byref(info),
            ctypes.byref(count),
        )
    )
    if status != 0 or info.resident_size <= 0:
        raise OSError(f"macOS current-RSS collector failed with status {status}")
    return int(info.resident_size)


def percentile(values: Sequence[float], probability: float) -> float:
    """Return the linear-interpolated quantile used in benchmark summaries."""

    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("percentile values must be finite")
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def summarize_samples(values: Sequence[float]) -> dict[str, float]:
    """Summarize retained measured trials; warm-ups are never passed here."""

    if not values:
        raise ValueError("benchmark summary requires measured trials")
    normalized = [float(value) for value in values]
    if any(not math.isfinite(value) or value <= 0.0 for value in normalized):
        raise ValueError("benchmark samples must be finite and greater than zero")
    return {
        "median": statistics.median(normalized),
        "p10": percentile(normalized, 0.1),
        "p90": percentile(normalized, 0.9),
        "mean": statistics.fmean(normalized),
        "standard_deviation": statistics.pstdev(normalized),
        "minimum": min(normalized),
        "maximum": max(normalized),
    }


def require_finite(
    value: float,
    *,
    name: str,
    minimum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    """Validate one scientific or timing value before it reaches evidence JSON."""

    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and normalized <= 0.0:
        raise ValueError(f"{name} must be greater than zero")
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def validate_hardware_label(value: str) -> str:
    """Require useful operator provenance without accepting credential-like text."""

    if not value or value != value.strip():
        raise ValueError("--hardware-label must be exact, nonempty, and whitespace-trimmed")
    if _SENSITIVE_LABEL.search(value):
        raise ValueError("--hardware-label appears to contain a credential or secret")
    return value


def validate_worker_timeout(value: float) -> float:
    """Bound each isolated Metal worker so a deadlock cannot stall the parent."""

    if not math.isfinite(value) or not 30.0 <= value <= 3600.0:
        raise ValueError("--worker-timeout-seconds must be finite and between 30 and 3600")
    return value


def _js_divergence(left: Sequence[int], right: Sequence[int]) -> float:
    left_total = sum(left)
    right_total = sum(right)
    if left_total <= 0 or right_total <= 0 or len(left) != len(right):
        raise ValueError("routing distributions must be aligned and nonempty")
    p = [value / left_total for value in left]
    q = [value / right_total for value in right]
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q, strict=True)]

    def kl(values: Sequence[float], middle: Sequence[float]) -> float:
        return sum(
            value * math.log(value / center)
            for value, center in zip(values, middle, strict=True)
            if value > 0.0 and center > 0.0
        )

    return require_finite(
        0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint),
        name="routing Jensen-Shannon divergence",
        minimum=0.0,
    )


def compare_routes(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare exact captured top-2 routes at sample/layer/token alignment keys."""

    def flatten(
        records: Sequence[Mapping[str, Any]],
    ) -> dict[tuple[str, int, int], tuple[int, ...]]:
        flattened: dict[tuple[str, int, int], tuple[int, ...]] = {}
        for record in records:
            sample_id = str(record["sample_id"])
            layer_id = int(record["layer_id"])
            for token_position, experts in enumerate(record["routes"]):
                key = (sample_id, layer_id, token_position)
                if key in flattened:
                    raise ValueError(f"duplicate routing alignment key: {key}")
                route = tuple(int(expert) for expert in experts)
                if len(route) != 2 or len(set(route)) != 2 or any(not 0 <= e < 4 for e in route):
                    raise ValueError(f"invalid top-2 route at {key}: {route}")
                flattened[key] = route
        return flattened

    base = flatten(baseline)
    cand = flatten(candidate)
    if base.keys() != cand.keys() or not base:
        raise ValueError("baseline and candidate route traces are not exactly aligned")
    per_layer: list[dict[str, Any]] = []
    total_tokens = 0
    total_ordered = 0
    total_unordered = 0
    total_overlap = 0.0
    for layer_id in range(6):
        keys = sorted(key for key in base if key[1] == layer_id)
        baseline_counts = [0, 0, 0, 0]
        candidate_counts = [0, 0, 0, 0]
        ordered = 0
        unordered = 0
        overlap = 0.0
        for key in keys:
            left = base[key]
            right = cand[key]
            ordered += int(left == right)
            unordered += int(set(left) == set(right))
            overlap += len(set(left).intersection(right)) / 2.0
            for expert in left:
                baseline_counts[expert] += 1
            for expert in right:
                candidate_counts[expert] += 1
        count = len(keys)
        if count <= 0:
            raise ValueError(f"routing trace has no events for layer {layer_id}")
        per_layer.append(
            {
                "layer_id": layer_id,
                "token_count": count,
                "ordered_top2_agreement": ordered / count,
                "unordered_top2_agreement": unordered / count,
                "mean_top2_overlap": overlap / count,
                "selection_js_divergence": _js_divergence(baseline_counts, candidate_counts),
                "baseline_expert_selection_counts": baseline_counts,
                "candidate_expert_selection_counts": candidate_counts,
            }
        )
        total_tokens += count
        total_ordered += ordered
        total_unordered += unordered
        total_overlap += overlap
    return {
        "alignment_key_count": len(base),
        "alignment_sha256": _canonical_sha256([list(key) for key in sorted(base)]),
        "ordered_top2_agreement": total_ordered / total_tokens,
        "unordered_top2_agreement": total_unordered / total_tokens,
        "mean_top2_overlap": total_overlap / total_tokens,
        "token_weighted_selection_js_divergence": sum(
            float(layer["selection_js_divergence"]) * int(layer["token_count"])
            for layer in per_layer
        )
        / total_tokens,
        "per_layer": per_layer,
        "capture_semantics": (
            "wrapper records each exact MixtralSparseMoeBlock gate(x) result; selected expert IDs "
            "are recomputed with the block's identical argpartition(-gates, kth=1)[..., :2]"
        ),
    }


def resolve_output_path(project_root: Path, requested: Path) -> Path:
    """Keep generated raw evidence below gitignored artifacts and refuse replacement."""

    project_root = project_root.expanduser().resolve(strict=True)
    if requested.is_absolute() or any(part == ".." for part in requested.parts):
        raise ValueError("output path must be project-relative and cannot contain '..'")
    artifact_root = (project_root / "artifacts").resolve(strict=False)
    output = (project_root / requested).resolve(strict=False)
    if output == artifact_root or not output.is_relative_to(artifact_root):
        raise ValueError("raw MLX evidence output must be below artifacts/")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"immutable output already exists: {output}")
    resolved_parent = parent.resolve(strict=True)
    if resolved_parent == artifact_root or resolved_parent.is_relative_to(artifact_root):
        return output
    raise ValueError("output parent resolves outside artifacts/")


def _installed_environment_projection() -> tuple[dict[str, str], str]:
    packages = {
        str(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    packages = dict(sorted(packages.items(), key=lambda item: item[0].lower()))
    return packages, _canonical_sha256(packages)


def validate_runtime_versions() -> dict[str, str]:
    """Fail before MLX import unless the complete evidenced runtime matrix matches."""

    actual = {package: importlib.metadata.version(package) for package in EXPECTED_RUNTIME_VERSIONS}
    if actual != EXPECTED_RUNTIME_VERSIONS:
        raise ValueError(
            f"MLX runtime version mismatch: expected {EXPECTED_RUNTIME_VERSIONS}, got {actual}"
        )
    return actual


def _git_provenance(project_root: Path) -> dict[str, Any]:
    """Return commit/dirty state, or the exact reason this workspace cannot."""

    if not (project_root / ".git").exists():
        return {
            "commit": None,
            "dirty": None,
            "unavailable_reason": "workspace has no .git metadata",
        }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status), "unavailable_reason": None}


def _mlx_lm_source_provenance() -> dict[str, Any]:
    distribution = importlib.metadata.distribution("mlx-lm")
    package_root = Path(str(distribution.locate_file("mlx_lm"))).resolve(strict=True)
    hashes = {
        relative: file_sha256(package_root / relative) for relative in EXPECTED_MLX_LM_SOURCE_HASHES
    }
    if hashes != EXPECTED_MLX_LM_SOURCE_HASHES:
        raise ValueError("installed MLX-LM source does not match the audited 0.31.3 files")
    mlx_distribution = importlib.metadata.distribution("mlx")
    quantized_layer = Path(str(mlx_distribution.locate_file("mlx/nn/layers/quantized.py")))
    quantized_layer_hash = file_sha256(quantized_layer)
    if quantized_layer_hash != EXPECTED_MLX_QUANTIZED_LAYER_SHA256:
        raise ValueError("installed MLX quantized-layer source checksum mismatch")
    return {
        "mlx_lm_package_location": "installed_distribution/mlx_lm",
        "mlx_lm_source_sha256": hashes,
        "mlx_quantized_layer_sha256": quantized_layer_hash,
        "audited_loader_behavior": (
            "local paths bypass snapshot_download; load_model reads config.json and "
            "model*.safetensors; model_file would execute Python and is therefore rejected "
            "before import"
        ),
        "audited_quantizer_behavior": (
            "quantize_model calls nn.quantize only on modules exposing to_quantized with an input "
            "width divisible by group_size; Mixtral fused SwitchLinear.to_quantized returns "
            "QuantizedSwitchLinear backed by mx.gather_qmm"
        ),
    }


def _route_trace_summary(routes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = [[0, 0, 0, 0] for _ in range(6)]
    event_count = 0
    projection: list[Any] = []
    for record in routes:
        sample_id = str(record["sample_id"])
        layer_id = int(record["layer_id"])
        for position, route in enumerate(record["routes"]):
            experts = [int(expert) for expert in route]
            for expert in experts:
                counts[layer_id][expert] += 1
            projection.append([sample_id, layer_id, position, experts])
            event_count += 1
    return {
        "token_layer_event_count": event_count,
        "selected_expert_count": event_count * 2,
        "trace_sha256": _canonical_sha256(projection),
        "expert_selection_counts_by_layer": counts,
        "router_logits_persisted": False,
        "raw_routes_persisted": False,
    }


def _benchmark_loaded_model(
    *,
    mx: Any,
    model: Any,
    tokenizer: Any,
    generate_step: Any,
    story: str,
    warmup_count: int,
    repetition_count: int,
) -> dict[str, Any]:
    """Benchmark immediately after load, before quality or routing allocations."""

    content_sha256 = hashlib.sha256(story.encode("utf-8")).hexdigest()
    prompt = [int(token) for token in tokenizer.encode(story, add_special_tokens=True)][
        : MAX_CONTEXT_TOKENS - MAX_NEW_TOKENS
    ]
    mx.clear_cache()
    mx.synchronize()
    host_baseline = current_process_rss_bytes()
    host_samples = [host_baseline]
    benchmark_started = time.perf_counter()
    observed_logprobs_dtype: str | None = None

    def trial(*, ordinal: int, retained: bool) -> dict[str, Any]:
        nonlocal observed_logprobs_dtype
        mx.reset_peak_memory()
        mx.synchronize()
        started = time.perf_counter()
        token_times: list[float] = []
        output: list[int] = []
        for token, logprobs in generate_step(mx.array(prompt), model, max_tokens=MAX_NEW_TOKENS):
            dtype = str(logprobs.dtype).removeprefix("mlx.core.")
            if observed_logprobs_dtype is None:
                observed_logprobs_dtype = dtype
            elif observed_logprobs_dtype != dtype:
                raise ValueError("generation log-probability dtype changed between trials")
            output.append(int(token))
            token_times.append(time.perf_counter())
            if int(token) in tokenizer.eos_token_ids:
                break
        mx.synchronize()
        ended = time.perf_counter()
        current_rss = current_process_rss_bytes()
        host_samples.append(current_rss)
        end_to_end = require_finite(
            ended - started,
            name="benchmark end-to-end latency",
            strictly_positive=True,
        )
        ttft = require_finite(
            token_times[0] - started,
            name="benchmark time to first token",
            strictly_positive=True,
        )
        inter_token = [right - left for left, right in itertools.pairwise(token_times)]
        inter_token_mean = require_finite(
            statistics.fmean(inter_token),
            name="benchmark inter-token latency",
            strictly_positive=True,
        )
        throughput = require_finite(
            len(output) / end_to_end,
            name="benchmark output tokens per second",
            strictly_positive=True,
        )
        return {
            "ordinal": ordinal,
            "retained": retained,
            "output_token_count": len(output),
            "output_sha256": _token_ids_sha256(output),
            "time_to_first_token_seconds": ttft,
            "inter_token_latency_seconds": inter_token_mean,
            "end_to_end_seconds": end_to_end,
            "output_tokens_per_second": throughput,
            "host_current_rss_bytes": current_rss,
            "mlx_allocator_active_bytes": int(mx.get_active_memory()),
            "mlx_allocator_cache_bytes": int(mx.get_cache_memory()),
            "mlx_allocator_peak_bytes_since_reset": int(mx.get_peak_memory()),
        }

    warmups = [trial(ordinal=index + 1, retained=False) for index in range(warmup_count)]
    measured = [trial(ordinal=index + 1, retained=True) for index in range(repetition_count)]
    interval = require_finite(
        time.perf_counter() - benchmark_started,
        name="benchmark measurement interval",
        strictly_positive=True,
    )
    output_hashes = {record["output_sha256"] for record in warmups + measured}
    if len(output_hashes) != 1 or any(
        record["output_token_count"] != MAX_NEW_TOKENS for record in measured
    ):
        raise ValueError("benchmark workload did not produce one deterministic eight-token output")
    if observed_logprobs_dtype != "float32":
        raise ValueError("direct MLX generation did not return float32 log probabilities")

    def summary(field: str) -> dict[str, float]:
        return summarize_samples([float(record[field]) for record in measured])

    return {
        "protocol_version": "direct-mlx-metal-generation-v2",
        "measurement_order": (
            "immediately after synchronized load/runtime proof and before quality, routing, or "
            "generation-regression allocations"
        ),
        "sample_id": BENCHMARK_SAMPLE_ID,
        "content_sha256": content_sha256,
        "prompt_token_count": len(prompt),
        "prompt_token_ids_sha256": _token_ids_sha256(prompt),
        "prompt_template": PROMPT_TEMPLATE,
        "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
        "decode": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "execution_mode": "single_prompt_sequential_direct_generate_step",
        "observed_logprobs_dtype": observed_logprobs_dtype,
        "warmup_count": warmup_count,
        "warmups_excluded_from_statistics": True,
        "measured_repetition_count": repetition_count,
        "synchronization": "mx.synchronize immediately before and after each timed region",
        "allocator_peak_reset": "mx.reset_peak_memory immediately before each trial",
        "deterministic_output_sha256": next(iter(output_hashes)),
        "all_warmup_and_measured_outputs_identical": True,
        "time_to_first_token_seconds": summary("time_to_first_token_seconds"),
        "inter_token_latency_seconds": summary("inter_token_latency_seconds"),
        "end_to_end_seconds": summary("end_to_end_seconds"),
        "output_tokens_per_second": summary("output_tokens_per_second"),
        "measured_trials": measured,
        "warmup_trials": warmups,
        "host_memory": {
            "available": True,
            "collector": "mach_task_basic_info.current_resident_size",
            "scope": (
                "worker process current RSS sampled before warm-up and after each warm-up/measured "
                "trial; boundary samples are not a continuous peak"
            ),
            "baseline_bytes": host_baseline,
            "maximum_observed_bytes": max(host_samples),
            "maximum_observed_delta_bytes": max(0, max(host_samples) - host_baseline),
            "sample_count": len(host_samples),
            "measurement_interval_seconds": interval,
        },
        "device_memory": {
            "available": True,
            "collector": "MLX allocator get_active_memory/get_cache_memory/get_peak_memory",
            "scope": (
                "process MLX allocator; peak reset per trial and includes resident model "
                "parameters; does not include unrelated unified-memory consumers"
            ),
            "maximum_active_bytes": max(
                int(record["mlx_allocator_active_bytes"]) for record in measured
            ),
            "maximum_cache_bytes": max(
                int(record["mlx_allocator_cache_bytes"]) for record in measured
            ),
            "maximum_peak_bytes_since_trial_reset": max(
                int(record["mlx_allocator_peak_bytes_since_reset"]) for record in measured
            ),
        },
        "energy": {
            "available": False,
            "reason": "no validated privilege-safe Apple joule collector is available",
        },
        "hardware_utilization": {
            "available": False,
            "reason": "no validated interval collector was used",
        },
    }


def _run_worker(args: argparse.Namespace) -> None:
    """Execute one model and return raw routes only through captured stdout IPC."""

    evaluator_path = Path(__file__).resolve(strict=True)
    evaluator_sha256_at_start = file_sha256(evaluator_path)
    if evaluator_sha256_at_start != args.evaluator_sha256:
        raise ValueError("worker evaluator hash differs from the parent-start hash")
    source_audit = audit_source_snapshot(Path(args.source_snapshot))
    checkpoint_audit = audit_converted_checkpoint(Path(args.model_path), label=args.label)
    dataset_path = Path(args.dataset).expanduser().resolve(strict=True)
    if file_sha256(dataset_path) != DATASET_SHA256:
        raise ValueError("TinyStories validation file checksum mismatch")
    stories = split_stories(dataset_path.read_text(encoding="utf-8"))
    selected = select_stories(stories, PERPLEXITY_IDS)
    by_id = dict(zip(PERPLEXITY_IDS, selected, strict=True))
    runtime_versions = validate_runtime_versions()
    source_provenance = _mlx_lm_source_provenance()

    import mlx.core as mx  # type: ignore[import-not-found]
    import mlx.nn as nn  # type: ignore[import-not-found]
    from mlx.utils import tree_flatten  # type: ignore[import-not-found]
    from mlx_lm.generate import generate_step  # type: ignore[import-not-found]
    from mlx_lm.utils import compute_bits_per_weight, load  # type: ignore[import-not-found]

    if not mx.metal.is_available():
        raise RuntimeError("MLX reports no Apple Metal device")
    mx.random.seed(SEED)
    host_rss_before_load = current_process_rss_bytes()
    load_started = time.perf_counter()
    model, tokenizer, config = load(
        str(Path(args.model_path).resolve(strict=True)),
        tokenizer_config={"trust_remote_code": False},
        return_config=True,
        lazy=False,
    )
    mx.synchronize()
    load_elapsed = require_finite(
        time.perf_counter() - load_started,
        name="model load elapsed time",
        strictly_positive=True,
    )
    host_rss_after_load = current_process_rss_bytes()
    allocator_after_load = {
        "active_bytes": int(mx.get_active_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
    }
    if type(model).__module__ != "mlx_lm.models.mixtral" or type(model).__qualname__ != "Model":
        raise ValueError("MLX-LM did not resolve its built-in Mixtral Model class")
    validate_model_config_security(config, source=f"loaded {args.label}")

    named_modules = {name: module for name, module in model.named_modules()}
    runtime_quantized: dict[str, Any]
    bits = EXPECTED_CONVERSIONS[args.label]["bits"]
    if bits is None:
        if any("Quantized" in type(module).__qualname__ for module in named_modules.values()):
            raise ValueError("float32 control unexpectedly contains quantized runtime modules")
        runtime_quantized = {
            "quantized_leaf_count": 0,
            "quantized_fused_expert_projection_count": 0,
        }
    else:
        quantized_linears = [
            name
            for name, module in named_modules.items()
            if type(module).__qualname__ == "QuantizedLinear"
        ]
        quantized_embeddings = [
            name
            for name, module in named_modules.items()
            if type(module).__qualname__ == "QuantizedEmbedding"
        ]
        quantized_experts = [
            name
            for name, module in named_modules.items()
            if type(module).__qualname__ == "QuantizedSwitchLinear"
        ]
        expected_linears = _expected_runtime_quantized_linear_names()
        if set(quantized_linears) != expected_linears:
            raise ValueError(
                "runtime quantized dense-leaf names mismatch; "
                f"missing={sorted(expected_linears - set(quantized_linears))}, "
                f"extra={sorted(set(quantized_linears) - expected_linears)}"
            )
        if set(quantized_embeddings) != {"model.embed_tokens"}:
            raise ValueError("runtime quantized embedding name mismatch")
        expected_experts = _expected_runtime_quantized_expert_names()
        if set(quantized_experts) != expected_experts:
            raise ValueError(
                "runtime quantized fused-expert names mismatch; "
                f"missing={sorted(expected_experts - set(quantized_experts))}, "
                f"extra={sorted(set(quantized_experts) - expected_experts)}"
            )
        metadata = {
            name: {
                "class": (
                    f"{type(named_modules[name]).__module__}."
                    f"{type(named_modules[name]).__qualname__}"
                ),
                "bits": int(named_modules[name].bits),
                "group_size": int(named_modules[name].group_size),
                "mode": str(named_modules[name].mode),
                "packed_weight_dtype": str(named_modules[name].weight.dtype).removeprefix(
                    "mlx.core."
                ),
            }
            for name in sorted(quantized_experts)
        }
        if any(
            fact["bits"] != bits
            or fact["group_size"] != 32
            or fact["mode"] != "affine"
            or fact["packed_weight_dtype"] != "uint32"
            for fact in metadata.values()
        ):
            raise ValueError("runtime fused expert quantization metadata mismatch")
        runtime_quantized = {
            "quantized_leaf_count": 50,
            "quantized_linear_count": len(quantized_linears),
            "quantized_linear_names": sorted(quantized_linears),
            "quantized_embedding_count": len(quantized_embeddings),
            "quantized_embedding_names": sorted(quantized_embeddings),
            "quantized_fused_expert_projection_count": len(quantized_experts),
            "quantized_fused_expert_projection_names": sorted(quantized_experts),
            "fused_expert_module_metadata": metadata,
            "kernel": "mlx.core.gather_qmm via QuantizedSwitchLinear",
        }

    parameters = tree_flatten(model.parameters())
    runtime_parameter_bytes = sum(value.nbytes for _, value in parameters)
    runtime_parameter_dtypes = sorted({str(value.dtype) for _, value in parameters})
    if runtime_parameter_bytes != checkpoint_audit["tensor_bytes"]:
        raise ValueError("loaded MLX parameter bytes differ from the audited checkpoint index")

    benchmark = _benchmark_loaded_model(
        mx=mx,
        model=model,
        tokenizer=tokenizer,
        generate_step=generate_step,
        story=by_id[BENCHMARK_SAMPLE_ID],
        warmup_count=args.warmups,
        repetition_count=args.repeats,
    )

    captured: list[list[Any]] = [[] for _ in range(6)]

    class CaptureGate(nn.Module):  # type: ignore[misc]
        def __init__(self, gate: Any, sink: list[Any]):
            super().__init__()
            self.gate = gate
            self.sink = sink

        def __call__(self, inputs: Any) -> Any:
            logits = self.gate(inputs)
            self.sink.append(logits)
            return logits

    original_gates: list[tuple[Any, Any]] = []
    for layer_id, layer in enumerate(model.model.layers):
        block = layer.block_sparse_moe
        original_gates.append((block, block.gate))
        block.gate = CaptureGate(block.gate, captured[layer_id])

    quality_samples: list[dict[str, Any]] = []
    ephemeral_routes: list[dict[str, Any]] = []
    weighted_nll = 0.0
    evaluated_tokens = 0
    truncated_count = 0
    observed_logits_dtype: str | None = None
    quality_started = time.perf_counter()
    try:
        for sample_id, story in zip(PERPLEXITY_IDS, selected, strict=True):
            encoded = [int(token) for token in tokenizer.encode(story, add_special_tokens=True)]
            token_ids = encoded[:MAX_CONTEXT_TOKENS]
            if len(token_ids) < 2:
                raise ValueError(f"{sample_id} has fewer than two loss tokens")
            for layer_capture in captured:
                layer_capture.clear()
            tokens = mx.array([token_ids])
            logits = model(tokens)
            logits_dtype = str(logits.dtype).removeprefix("mlx.core.")
            if observed_logits_dtype is None:
                observed_logits_dtype = logits_dtype
            elif observed_logits_dtype != logits_dtype:
                raise ValueError("causal logits dtype changed between quality samples")
            losses = nn.losses.cross_entropy(logits[:, :-1, :], tokens[:, 1:], reduction="none")
            route_arrays: list[tuple[Any, Any]] = []
            for layer_id, layer_capture in enumerate(captured):
                if len(layer_capture) != 1:
                    raise ValueError(
                        f"expected one exact router-gate capture for {sample_id} layer {layer_id}"
                    )
                gates = layer_capture[0]
                indices = mx.argpartition(-gates, kth=1, axis=-1)[..., :2]
                scores = mx.softmax(
                    mx.take_along_axis(gates, indices, axis=-1),
                    axis=-1,
                    precise=True,
                )
                route_arrays.append((indices, scores))
            mx.eval(losses, *(item for pair in route_arrays for item in pair))
            mean_nll = require_finite(
                float(mx.mean(losses).item()),
                name=f"{sample_id} mean NLL",
                minimum=0.0,
            )
            count = len(token_ids) - 1
            weighted_nll += mean_nll * count
            evaluated_tokens += count
            truncated = len(encoded) > MAX_CONTEXT_TOKENS
            truncated_count += int(truncated)
            quality_samples.append(
                {
                    "sample_id": sample_id,
                    "content_sha256": hashlib.sha256(story.encode("utf-8")).hexdigest(),
                    "source_token_count": len(encoded),
                    "input_token_ids_sha256": _token_ids_sha256(token_ids),
                    "evaluated_token_count": count,
                    "truncated": truncated,
                    "mean_nll": mean_nll,
                }
            )
            for layer_id, (indices, _scores) in enumerate(route_arrays):
                ephemeral_routes.append(
                    {
                        "sample_id": sample_id,
                        "layer_id": layer_id,
                        "routes": indices[0].tolist(),
                    }
                )
            del tokens, logits, losses, route_arrays
            mx.clear_cache()
    finally:
        for block, gate in original_gates:
            block.gate = gate
        for layer_capture in captured:
            layer_capture.clear()
    mx.synchronize()
    quality_elapsed = require_finite(
        time.perf_counter() - quality_started,
        name="quality elapsed time",
        strictly_positive=True,
    )
    if evaluated_tokens != 5_826 or truncated_count != 2:
        raise ValueError(
            "MLX quality tokenization/truncation does not match the pinned 32-story contract"
        )
    if observed_logits_dtype != "float32":
        raise ValueError("direct MLX quality execution did not return float32 logits")
    mean_nll = require_finite(
        weighted_nll / evaluated_tokens,
        name="token-weighted mean NLL",
        minimum=0.0,
    )
    perplexity = require_finite(
        math.exp(mean_nll),
        name="quality perplexity",
        strictly_positive=True,
    )

    def generate_tokens(story: str) -> tuple[tuple[int, ...], int, bool, str]:
        encoded = [int(token) for token in tokenizer.encode(story, add_special_tokens=True)]
        prompt = encoded[: MAX_CONTEXT_TOKENS - MAX_NEW_TOKENS]
        generated: list[int] = []
        for token, _logprobs in generate_step(mx.array(prompt), model, max_tokens=MAX_NEW_TOKENS):
            generated.append(int(token))
            if int(token) in tokenizer.eos_token_ids:
                break
        mx.synchronize()
        return (
            tuple(generated),
            len(prompt),
            len(prompt) < len(encoded),
            _token_ids_sha256(prompt),
        )

    generation_records: list[dict[str, Any]] = []
    generation_hashes: dict[str, str] = {}
    for sample_id in GENERATION_IDS:
        story = by_id[sample_id]
        first, input_count, truncated, input_hash = generate_tokens(story)
        second, second_count, second_truncated, second_input_hash = generate_tokens(story)
        if (
            first != second
            or input_count != second_count
            or truncated != second_truncated
            or input_hash != second_input_hash
        ):
            raise ValueError(f"greedy generation was not deterministic for {sample_id}")
        output_hash = _token_ids_sha256(first)
        generation_hashes[sample_id] = output_hash
        generation_records.append(
            {
                "sample_id": sample_id,
                "content_sha256": hashlib.sha256(story.encode("utf-8")).hexdigest(),
                "source_token_count": len(tokenizer.encode(story, add_special_tokens=True)),
                "input_token_count": input_count,
                "input_token_ids_sha256": input_hash,
                "input_truncated": truncated,
                "generated_token_count": len(first),
                "output_sha256": output_hash,
                "repeat_output_sha256": _token_ids_sha256(second),
            }
        )
    benchmark_generation = next(
        item for item in generation_records if item["sample_id"] == BENCHMARK_SAMPLE_ID
    )
    if (
        benchmark["deterministic_output_sha256"] != generation_hashes[BENCHMARK_SAMPLE_ID]
        or benchmark_generation["generated_token_count"] != MAX_NEW_TOKENS
        or benchmark["prompt_token_ids_sha256"] != benchmark_generation["input_token_ids_sha256"]
    ):
        raise ValueError(
            "benchmark output does not match the independent two-pass generation regression"
        )

    packages, packages_sha256 = _installed_environment_projection()
    environment = {
        "hardware_label": args.hardware_label,
        "platform": platform.platform(),
        "mac_ver": platform.mac_ver()[0],
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "runtime_versions": runtime_versions,
        "mlx": runtime_versions["mlx"],
        "mlx_lm": runtime_versions["mlx-lm"],
        "mlx_metal": runtime_versions["mlx-metal"],
        "transformers": runtime_versions["transformers"],
        "safetensors": runtime_versions["safetensors"],
        "metal_available": bool(mx.metal.is_available()),
        "default_device": str(mx.default_device()),
        "device_info": mx.device_info(),
        "installed_packages": packages,
        "installed_packages_sha256": packages_sha256,
        "process_isolation": "fresh worker process for exactly one model",
        "network": "offline local paths; HF_HUB_OFFLINE=1; TRANSFORMERS_OFFLINE=1",
        **source_provenance,
    }

    route_summary = _route_trace_summary(ephemeral_routes)
    evaluator_sha256_at_end = file_sha256(evaluator_path)
    if evaluator_sha256_at_end != evaluator_sha256_at_start:
        raise ValueError("worker evaluator changed during execution")
    worker_record = {
        "label": args.label,
        "evaluator": {
            "path": "scripts/evaluate_mlx_public_moe_quant.py",
            "sha256_at_worker_start": evaluator_sha256_at_start,
            "sha256_at_worker_end": evaluator_sha256_at_end,
        },
        "source_audit_sha256": _canonical_sha256(source_audit),
        "checkpoint": checkpoint_audit,
        "load": {
            "elapsed_seconds": load_elapsed,
            "measurement_scope": (
                "timed MLX-LM materialization after the worker completed full checkpoint audit; "
                "OS file cache was not flushed or controlled and this is not cold-load latency"
            ),
            "host_current_rss_before_load_bytes": host_rss_before_load,
            "host_current_rss_after_load_bytes": host_rss_after_load,
            "mlx_allocator_after_load": allocator_after_load,
            "runtime_parameter_bytes": runtime_parameter_bytes,
            "runtime_parameter_dtypes": runtime_parameter_dtypes,
            "effective_bits_per_weight": float(compute_bits_per_weight(model)),
            "resolved_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "runtime_quantization_proof": runtime_quantized,
        },
        "quality": {
            "metric": "token_weighted_causal_nll_and_perplexity",
            "observed_logits_dtype": observed_logits_dtype,
            "sample_count": len(quality_samples),
            "evaluated_token_count": evaluated_tokens,
            "truncated_sample_count": truncated_count,
            "mean_nll": mean_nll,
            "perplexity": perplexity,
            "elapsed_seconds": quality_elapsed,
            "samples": quality_samples,
        },
        "generation": {
            "sample_count": len(generation_records),
            "decode": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
            "two_pass_determinism_verified": True,
            "samples": generation_records,
            "output_content_persisted": False,
            "output_token_ids_persisted": False,
        },
        "routing": route_summary,
        "benchmark": benchmark,
        "environment": environment,
        "_ephemeral_routes": ephemeral_routes,
    }
    print(
        json.dumps(
            worker_record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _validate_worker_evaluator_identity(
    worker: Mapping[str, Any],
    *,
    label: str,
    expected_sha256: str,
) -> None:
    evaluator = worker.get("evaluator")
    expected = {
        "path": "scripts/evaluate_mlx_public_moe_quant.py",
        "sha256_at_worker_start": expected_sha256,
        "sha256_at_worker_end": expected_sha256,
    }
    if evaluator != expected:
        raise ValueError(f"{label} worker evaluator identity differs from the parent")


def _assert_worker_contracts(
    workers: Mapping[str, Mapping[str, Any]],
    *,
    evaluator_sha256: str,
) -> None:
    baseline = workers["float32"]
    baseline_samples = baseline["quality"]["samples"]
    baseline_generation = baseline["generation"]["samples"]
    baseline_environment = baseline["environment"]
    for label, worker in workers.items():
        _validate_worker_evaluator_identity(
            worker,
            label=label,
            expected_sha256=evaluator_sha256,
        )
        if [
            (
                item["sample_id"],
                item["content_sha256"],
                item["source_token_count"],
                item["input_token_ids_sha256"],
                item["evaluated_token_count"],
                item["truncated"],
            )
            for item in worker["quality"]["samples"]
        ] != [
            (
                item["sample_id"],
                item["content_sha256"],
                item["source_token_count"],
                item["input_token_ids_sha256"],
                item["evaluated_token_count"],
                item["truncated"],
            )
            for item in baseline_samples
        ]:
            raise ValueError(f"{label} quality tokenization contract differs from the MLX control")
        if [
            (
                item["sample_id"],
                item["content_sha256"],
                item["source_token_count"],
                item["input_token_count"],
                item["input_token_ids_sha256"],
                item["input_truncated"],
            )
            for item in worker["generation"]["samples"]
        ] != [
            (
                item["sample_id"],
                item["content_sha256"],
                item["source_token_count"],
                item["input_token_count"],
                item["input_token_ids_sha256"],
                item["input_truncated"],
            )
            for item in baseline_generation
        ]:
            raise ValueError(f"{label} generation tokenization differs from the MLX control")
        if (
            worker["benchmark"]["prompt_token_ids_sha256"]
            != baseline["benchmark"]["prompt_token_ids_sha256"]
        ):
            raise ValueError(f"{label} benchmark prompt tokens differ from the MLX control")
        for key in (
            "hardware_label",
            "platform",
            "mac_ver",
            "machine",
            "python",
            "mlx",
            "mlx_lm",
            "mlx_metal",
            "runtime_versions",
            "transformers",
            "safetensors",
            "metal_available",
            "default_device",
            "device_info",
            "installed_packages_sha256",
            "mlx_lm_source_sha256",
            "mlx_quantized_layer_sha256",
        ):
            if worker["environment"][key] != baseline_environment[key]:
                raise ValueError(f"{label} environment differs from control on {key}")


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    """Audit, execute isolated workers, compare, and return final redacted evidence."""

    evaluator_path = Path(__file__).resolve(strict=True)
    evaluator_sha256_at_parent_start = file_sha256(evaluator_path)
    validate_hardware_label(args.hardware_label)
    worker_timeout = validate_worker_timeout(args.worker_timeout_seconds)
    project_root = Path(__file__).resolve(strict=True).parents[1]
    git_provenance = _git_provenance(project_root)
    if args.warmups < 1 or args.repeats < 5:
        raise ValueError("benchmark requires at least one warm-up and five measured repetitions")
    source_snapshot = Path(args.source_snapshot).expanduser().resolve(strict=True)
    source = audit_source_snapshot(source_snapshot)
    model_paths = {
        "float32": Path(args.float32_model).expanduser().resolve(strict=True),
        "mlx_affine_q4_g32": Path(args.q4_model).expanduser().resolve(strict=True),
        "mlx_affine_q8_g32": Path(args.q8_model).expanduser().resolve(strict=True),
    }
    checkpoint_audits = {
        label: audit_converted_checkpoint(path, label=label) for label, path in model_paths.items()
    }
    dataset_path = Path(args.dataset).expanduser().resolve(strict=True)
    if file_sha256(dataset_path) != DATASET_SHA256:
        raise ValueError("TinyStories validation file checksum mismatch")
    all_stories = split_stories(dataset_path.read_text(encoding="utf-8"))
    selected = select_stories(all_stories, PERPLEXITY_IDS)
    generation_stories = tuple(selected[PERPLEXITY_IDS.index(item)] for item in GENERATION_IDS)

    workers: dict[str, dict[str, Any]] = {}
    worker_order = ("float32", "mlx_affine_q4_g32", "mlx_affine_q8_g32")
    for label in worker_order:
        model_path = model_paths[label]
        command = [
            sys.executable,
            str(evaluator_path),
            "--worker",
            "--label",
            label,
            "--source-snapshot",
            str(source_snapshot),
            "--model-path",
            str(model_path),
            "--dataset",
            str(dataset_path),
            "--hardware-label",
            args.hardware_label,
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--evaluator-sha256",
            evaluator_sha256_at_parent_start,
        ]
        environment = dict(os.environ)
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=worker_timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                f"MLX worker {label} exceeded the {worker_timeout:.0f}-second timeout"
            ) from error
        if completed.returncode != 0:
            raise RuntimeError(
                f"MLX worker {label} failed with exit {completed.returncode}: "
                f"stdout={completed.stdout[-2000:]!r} stderr={completed.stderr[-4000:]!r}"
            )
        workers[label] = json.loads(completed.stdout)
    _assert_worker_contracts(
        workers,
        evaluator_sha256=evaluator_sha256_at_parent_start,
    )

    baseline = workers["float32"]
    baseline_routes = baseline.pop("_ephemeral_routes")
    baseline_generation = {
        item["sample_id"]: item["output_sha256"] for item in baseline["generation"]["samples"]
    }
    candidates: list[dict[str, Any]] = []
    for label in ("mlx_affine_q8_g32", "mlx_affine_q4_g32"):
        candidate = workers[label]
        candidate_routes = candidate.pop("_ephemeral_routes")
        routing_comparison = compare_routes(baseline_routes, candidate_routes)
        candidate_generation = {
            item["sample_id"]: item["output_sha256"] for item in candidate["generation"]["samples"]
        }
        generation_matches = {
            sample_id: candidate_generation[sample_id] == baseline_hash
            for sample_id, baseline_hash in baseline_generation.items()
        }
        candidate_perplexity = require_finite(
            float(candidate["quality"]["perplexity"]),
            name=f"{label} perplexity",
            strictly_positive=True,
        )
        baseline_perplexity = require_finite(
            float(baseline["quality"]["perplexity"]),
            name="float32 control perplexity",
            strictly_positive=True,
        )
        nll_delta = require_finite(
            float(candidate["quality"]["mean_nll"]) - float(baseline["quality"]["mean_nll"]),
            name=f"{label} mean NLL delta",
        )
        perplexity_delta = require_finite(
            candidate_perplexity - baseline_perplexity,
            name=f"{label} perplexity delta",
        )
        relative_perplexity = require_finite(
            candidate_perplexity / baseline_perplexity - 1.0,
            name=f"{label} relative perplexity change",
        )
        candidate["quality_vs_float32"] = {
            "mean_nll_delta": nll_delta,
            "perplexity_delta": perplexity_delta,
            "perplexity_relative_change": relative_perplexity,
        }
        candidate["generation_vs_float32"] = {
            "exact_match_count": sum(generation_matches.values()),
            "sample_count": len(generation_matches),
            "exact_match_rate": sum(generation_matches.values()) / len(generation_matches),
            "per_sample_exact_match": generation_matches,
        }
        candidate["routing_vs_float32"] = routing_comparison
        candidates.append(candidate)

    source_command_prefix = "$ISOLATED_PYTHON -m mlx_lm.convert"
    conversion_commands = {
        "float32": (
            f"{source_command_prefix} --hf-path $SOURCE_SNAPSHOT "
            "--mlx-path $FP32_MODEL --dtype float32"
        ),
        "mlx_affine_q4_g32": (
            f"{source_command_prefix} --hf-path $SOURCE_SNAPSHOT "
            "--mlx-path $Q4_MODEL --quantize "
            "--q-group-size 32 --q-bits 4 --q-mode affine --dtype float32"
        ),
        "mlx_affine_q8_g32": (
            f"{source_command_prefix} --hf-path $SOURCE_SNAPSHOT "
            "--mlx-path $Q8_MODEL --quantize "
            "--q-group-size 32 --q-bits 8 --q-mode affine --dtype float32"
        ),
    }
    evaluator_sha256_at_parent_end = file_sha256(evaluator_path)
    if evaluator_sha256_at_parent_end != evaluator_sha256_at_parent_start:
        raise ValueError("parent evaluator changed during execution")
    return {
        "schema_version": "public-moe-mlx-full-quantization-evidence-v3",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "evaluator": {
            "path": "scripts/evaluate_mlx_public_moe_quant.py",
            "sha256": evaluator_sha256_at_parent_start,
            "sha256_at_parent_start": evaluator_sha256_at_parent_start,
            "sha256_at_parent_end": evaluator_sha256_at_parent_end,
            "git_commit": git_provenance["commit"],
            "git_dirty": git_provenance["dirty"],
            "git_unavailable_reason": git_provenance["unavailable_reason"],
            "parent_command_template": (
                "$ISOLATED_PYTHON scripts/evaluate_mlx_public_moe_quant.py "
                "--source-snapshot $SOURCE_SNAPSHOT --float32-model $FP32_MODEL "
                "--q4-model $Q4_MODEL --q8-model $Q8_MODEL --dataset $DATASET_FILE "
                f"--hardware-label $HARDWARE_LABEL --warmups {args.warmups} "
                f"--repeats {args.repeats} --worker-timeout-seconds {worker_timeout:g} "
                "--output $ARTIFACT_OUTPUT"
            ),
            "absolute_local_paths_redacted": True,
        },
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": MODEL_LICENSE,
            "architecture": "MixtralForCausalLM / mlx_lm.models.mixtral.Model",
            "layers": 6,
            "experts_per_layer": 4,
            "experts_per_token": 2,
            "expert_weight_provenance": (
                "measured: all four source expert tensors are byte-identical within each of "
                "18 layer/projection groups"
            ),
            "router_provenance": "publisher-described as randomly initialized; not value-proven",
            "source": source,
        },
        "conversion": {
            "executed_command_templates_with_local_paths_redacted": conversion_commands,
            "command_template_semantics": (
                "argument-for-argument executed commands with only interpreter and absolute local "
                "paths replaced by named variables; output bytes are pinned below"
            ),
            "source_dtype_normalization": (
                "all three controls/candidates use mlx_lm.convert --dtype float32; direct source "
                "loading is not the control because publisher expert tensors are float16"
            ),
            "checkpoints": checkpoint_audits,
            "algorithm": "MLX affine per-group weight-only quantization",
            "calibration_required": False,
            "calibration_data": None,
            "upload_performed": False,
            "output_location": "temporary local paths outside the repository; not committed",
        },
        "dataset": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": DATASET_SPLIT,
            "license": DATASET_LICENSE,
            "file": dataset_path.name,
            "file_sha256": DATASET_SHA256,
            "full_file_story_count": len(all_stories),
            "perplexity_sample_ids": list(PERPLEXITY_IDS),
            "perplexity_selection_sha256": selection_sha256(PERPLEXITY_IDS, selected),
            "generation_sample_ids": list(GENERATION_IDS),
            "generation_selection_sha256": selection_sha256(GENERATION_IDS, generation_stories),
        },
        "protocol": {
            "seed": SEED,
            "runtime": "direct MLX on one Apple Metal device; one clean worker process per model",
            "quality": "token-weighted causal NLL over exact 32 stories",
            "tokenization": "same local tokenizer; add_special_tokens=true",
            "tokenizer_identity": {
                "class": "TokenizersBackend",
                "tokenizer_json_sha256": CONVERTED_TOKENIZER_SHA256,
                "tokenizer_config_sha256": CONVERTED_TOKENIZER_CONFIG_SHA256,
            },
            "prompt_template": PROMPT_TEMPLATE,
            "prompt_template_sha256": PROMPT_TEMPLATE_SHA256,
            "prompt_rendering": "raw story text with no prefix or suffix",
            "loss_truncation": "right to 256 tokens",
            "generation": "four exact sample IDs; greedy; 8 tokens; repeated twice",
            "routing": (
                "full aligned top-2 routes captured from exact built-in Mixtral gate outputs "
                "during the same 32 loss forwards; logits and raw routes omitted from final record"
            ),
            "benchmark": {
                "sample_id": BENCHMARK_SAMPLE_ID,
                "warmups": args.warmups,
                "measured_repetitions": args.repeats,
                "warmups_excluded": True,
                "Metal_synchronized": True,
                "worker_execution_order": list(worker_order),
                "worker_timeout_seconds": worker_timeout,
                "order_randomized": False,
                "background_load_controlled": False,
            },
            "prompt_or_output_content_persisted": False,
            "output_token_ids_persisted": False,
            "raw_router_traces_persisted": False,
        },
        "float32_control": baseline,
        "candidates": candidates,
        "worker_protocol": {
            "execution_order": list(worker_order),
            "command_template": (
                "$ISOLATED_PYTHON scripts/evaluate_mlx_public_moe_quant.py --worker "
                "--label $LABEL --source-snapshot $SOURCE_SNAPSHOT --model-path $MODEL_PATH "
                "--dataset $DATASET_FILE --hardware-label $HARDWARE_LABEL "
                f"--warmups {args.warmups} --repeats {args.repeats} "
                "--evaluator-sha256 $PARENT_EVALUATOR_SHA256"
            ),
            "result_transport": (
                "captured stdout IPC; raw route IDs decoded in parent memory and omitted from "
                "the final artifact"
            ),
            "absolute_local_paths_redacted": True,
        },
        "limitations": [
            (
                "This is one deterministic 32-story subset and one Apple M3; it is not a "
                "full-dataset, multi-seed, or cross-hardware claim."
            ),
            (
                "The publisher states that experts repeat trained TinyStories weights and that "
                "routers were randomly initialized; route drift is regression evidence, not "
                "learned-specialization evidence."
            ),
            (
                "Uniform conversion quantizes every eligible MLX leaf, including embedding, "
                "output head, routers, attention, and all fused expert projections; it does not "
                "preserve router/head precision."
            ),
            (
                "Because routers, embeddings/head, attention, and experts change together, route "
                "or quality differences cannot be attributed specifically to expert-weight "
                "quantization or any one module class."
            ),
            (
                "The q8 candidate's tiny lower perplexity is an observed single-run subset "
                "difference, not evidence of an improvement or a causal effect."
            ),
            (
                "MLX affine 8-bit is integer weight quantization with float activations, not FP8 "
                "and not a CUDA AWQ/GPTQ backend."
            ),
            (
                "Host RSS uses boundary samples and is not a continuous peak; MLX allocator "
                "memory excludes unrelated unified-memory consumers."
            ),
            (
                "Model-load timing starts after each worker fully audits and hashes its source "
                "and checkpoint; OS file cache was not flushed or controlled, so it must not be "
                "interpreted as cold-load latency."
            ),
            (
                "Generation latency is direct single-prompt MLX execution, not SGLang serving "
                "latency, concurrency, or production throughput."
            ),
            (
                "Workers ran in fixed float32, q4, q8 order; host background load, thermal state, "
                "and order effects were not randomized or controlled."
            ),
            (
                "Energy, hardware utilization, sharding, multi-device execution, export through "
                "the canonical Inkling pipeline, and public checkpoint upload remain unmeasured "
                "or unsupported."
            ),
        ],
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise FileExistsError(f"immutable output already exists: {path}") from error
    except BaseException:
        raise
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot")
    parser.add_argument("--float32-model")
    parser.add_argument("--q4-model")
    parser.add_argument("--q8-model")
    parser.add_argument("--dataset")
    parser.add_argument("--hardware-label")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--worker-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--label", choices=tuple(EXPECTED_CONVERSIONS), help=argparse.SUPPRESS)
    parser.add_argument("--model-path", help=argparse.SUPPRESS)
    parser.add_argument("--evaluator-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    required: tuple[tuple[str, str], ...] = (
        ("source_snapshot", "--source-snapshot"),
        ("dataset", "--dataset"),
        ("hardware_label", "--hardware-label"),
    )
    if args.worker:
        required += (
            ("label", "--label"),
            ("model_path", "--model-path"),
            ("evaluator_sha256", "--evaluator-sha256"),
        )
    else:
        required += (
            ("float32_model", "--float32-model"),
            ("q4_model", "--q4-model"),
            ("q8_model", "--q8-model"),
            ("output", "--output"),
        )
    missing = [flag for attribute, flag in required if getattr(args, attribute) is None]
    if missing:
        parser.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.worker:
        _run_worker(args)
        return 0
    project_root = Path(__file__).resolve().parents[1]
    output = resolve_output_path(project_root, Path(args.output))
    record = run_parent(args)
    _atomic_write_json(output, record)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": file_sha256(output),
                "schema_version": record["schema_version"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
