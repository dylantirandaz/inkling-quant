"""Exact, import-safe contract for the validated Stories15M MLX matrix.

This module intentionally has no MLX import.  Security and dependency checks run
before the optional runtime is imported, so a registry descriptor or rejected
snapshot cannot execute checkpoint-supplied Python.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from safetensors import safe_open

MODEL_ID = "ggml-org/stories15M_MOE"
MODEL_REVISION = "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
MODEL_ARCHITECTURE = "MixtralForCausalLM"
MLX_MODEL_CLASS = "mlx_lm.models.mixtral.Model"
SOURCE_WEIGHT_SHA256 = "dbfa0289f68a8dd721d10eb12d8bd82e098455682027f6f9986ba548913f9082"
CANONICAL_MLX_MODEL_CARD_BYTES = (
    b"---\nlanguage: en\ntags:\n- mlx\npipeline_tag: text-generation\nlibrary_name: mlx\n---\n"
)

EXPECTED_MLX_VERSIONS: dict[str, str] = {
    "mlx": "0.32.0",
    "mlx-lm": "0.31.3",
    "mlx-metal": "0.32.0",
    "safetensors": "0.8.0",
    "transformers": "5.12.1",
}

_DISTRIBUTION_MODULES = {
    "mlx": "mlx",
    "mlx-lm": "mlx_lm",
    "mlx-metal": "mlx",
    "safetensors": "safetensors",
    "transformers": "transformers",
}
_SOURCE_SUFFIXES = frozenset({".json", ".safetensors", ".txt"})
_BUNDLE_SUFFIXES = frozenset({".json", ".md", ".safetensors"})
_EXECUTABLE_SUFFIXES = frozenset(
    {
        ".bin",
        ".ckpt",
        ".dll",
        ".dylib",
        ".exe",
        ".joblib",
        ".pkl",
        ".pickle",
        ".pt",
        ".pth",
        ".py",
        ".pyc",
        ".pyd",
        ".so",
        ".wasm",
    }
)


@dataclass(frozen=True, slots=True)
class MLXEnvironmentStatus:
    """Installed-package and host status without importing MLX."""

    available: bool
    versions: dict[str, str]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MLXSourceContract:
    """Immutable local source facts checked before optional imports."""

    revision: str
    file_names: frozenset[str]
    config_sha256: str
    tokenizer_config_sha256: str
    tokenizer_sha256: str
    weight_sha256: str
    weight_size_bytes: int
    tensor_count: int | None


@dataclass(frozen=True, slots=True)
class MLXConversionContract:
    """Expected deterministic MLX-LM conversion bundle."""

    bits: int
    label: str
    files: Mapping[str, tuple[int, str]]
    tensor_bytes: int
    tensor_count: int

    @property
    def base_bundle_size_bytes(self) -> int:
        """Return exact bytes before the Inkling manifest is embedded."""

        return sum(size for size, _digest in self.files.values())


STORIES15M_SOURCE_CONTRACT = MLXSourceContract(
    revision=MODEL_REVISION,
    file_names=frozenset(
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
    ),
    config_sha256="e901e012953d1df93574b2cc3d7db5ed4758d52f8bd4a7dd4b647936e32261be",
    tokenizer_config_sha256=("33d29c87e41f7dd1efb0434d852730320c82970f292be452d820539bce417052"),
    tokenizer_sha256="8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1",
    weight_sha256=SOURCE_WEIGHT_SHA256,
    weight_size_bytes=72_744_704,
    tensor_count=117,
)

_COMMON_CONVERSION_FILES: dict[str, tuple[int, str]] = {
    "README.md": (81, "5ad88ec5ec365ab38e227409e147a8ddc1918d62d587183b3b8c90e9de6a2f2f"),
    "generation_config.json": (
        115,
        "295aa491adda22ab9fbdecdda9e8121e8348fd0eea0529d8802993426ab0892c",
    ),
    "tokenizer.json": (
        3_619_013,
        "ed7c4e86a6d2b2a24cf7fb6e6d96c445a97bf9a22d682f598869afc2625bfd1f",
    ),
    "tokenizer_config.json": (
        341,
        "1f6b3e8bee4075befa304524f5681f963f244e2cd9ea249d322b720433d48d78",
    ),
}

MLX_CONVERSION_CONTRACTS: dict[int, MLXConversionContract] = {
    4: MLXConversionContract(
        bits=4,
        label="mlx_affine_q4_g32",
        files={
            **_COMMON_CONVERSION_FILES,
            "config.json": (
                972,
                "1cb73e9e5c63bb78a15bb07b4189fc1e528c180ad1dd8bfb6d89cbcd8afb283f",
            ),
            "model.safetensors": (
                27_299_860,
                "610c0f80ed9e59648c652aa4e8c9227b59be1286c55cccad0d5356c064b20552",
            ),
            "model.safetensors.index.json": (
                12_793,
                "a3c277e527d0ed45ba4a690ea3a102fd75f6ab181bd87db9a2fe652936cbb6c9",
            ),
        },
        tensor_bytes=27_281_088,
        tensor_count=163,
    ),
    8: MLXConversionContract(
        bits=8,
        label="mlx_affine_q8_g32",
        files={
            **_COMMON_CONVERSION_FILES,
            "config.json": (
                972,
                "97b00d3dcda1aef2d5c98b002d3c229fc38b29983c1354f9279d23a15b02221d",
            ),
            "model.safetensors": (
                45_477_324,
                "a2a5a57afbad4aa8378fea33a6b51b0848f7cd665d3ed191bad80704d3756dec",
            ),
            "model.safetensors.index.json": (
                12_793,
                "8bbbb87885bc997cc47c5d7888e64e86174275af8069a72208abbc2b6a4aa998",
            ),
        },
        tensor_bytes=45_458_496,
        tensor_count=163,
    ),
}


def sha256_file(path: Path) -> str:
    """Hash one regular file or an already-approved safe symlink target."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mlx_environment_status() -> MLXEnvironmentStatus:
    """Check the exact validated package/host matrix without importing MLX."""

    reasons: list[str] = []
    versions: dict[str, str] = {}
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        reasons.append("mlx_metal requires macOS on Apple Silicon (Darwin arm64)")
    for distribution, expected in EXPECTED_MLX_VERSIONS.items():
        module = _DISTRIBUTION_MODULES[distribution]
        if importlib.util.find_spec(module) is None:
            reasons.append(f"missing Python package {distribution}=={expected}")
            continue
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            reasons.append(f"distribution metadata is missing for {distribution}")
            continue
        versions[distribution] = actual
        if actual != expected:
            reasons.append(f"{distribution}=={actual} does not match validated {expected}")
    return MLXEnvironmentStatus(
        available=not reasons,
        versions=dict(sorted(versions.items())),
        reasons=tuple(reasons),
    )


def validate_model_config_security(config: Mapping[str, Any], *, source: str) -> None:
    """Require MLX-LM's built-in Mixtral implementation and no code selectors."""

    if config.get("model_file") is not None:
        raise ValueError(f"{source} config.model_file must be absent or null")
    if config.get("auto_map") is not None:
        raise ValueError(f"{source} config.auto_map must be absent or null")
    nested = config.get("text_config")
    if isinstance(nested, Mapping) and (
        nested.get("model_file") is not None or nested.get("auto_map") is not None
    ):
        raise ValueError(f"{source} nested text config selects custom code")
    if config.get("model_type") != "mixtral":
        raise ValueError(f"{source} model_type must be exactly mixtral")
    if config.get("architectures") != [MODEL_ARCHITECTURE]:
        raise ValueError(f"{source} architecture must be exactly {MODEL_ARCHITECTURE}")


def validate_tokenizer_config_security(
    config: Mapping[str, Any], *, source: str, expected_class: str
) -> None:
    """Reject dynamic tokenizer helpers and require the audited built-in class."""

    if config.get("tokenizer_class") != expected_class:
        raise ValueError(f"{source} tokenizer_class must be exactly {expected_class}")
    for field in ("auto_map", "chat_template", "chat_template_type", "tool_parser_type"):
        if config.get(field) is not None:
            raise ValueError(f"{source} tokenizer field {field} must be absent or null")
    if config.get("trust_remote_code") is True:
        raise ValueError(f"{source} tokenizer must not enable remote code")


def _audit_tree(
    root: Path,
    *,
    allowed_suffixes: frozenset[str],
    allow_hf_cache_symlinks: bool,
) -> tuple[dict[str, Any], ...]:
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
            raise ValueError(f"unsupported non-file model entry: {relative}")
        suffix = candidate.suffix.lower()
        if suffix in _EXECUTABLE_SUFFIXES or suffix not in allowed_suffixes:
            raise ValueError(f"unsafe or unsupported model file: {relative}")
        if is_symlink:
            if not allow_hf_cache_symlinks:
                raise ValueError(f"converted bundle must not contain symlinks: {relative}")
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(allowed_resolution_root):
                raise ValueError(f"snapshot symlink escapes its repository cache: {relative}")
        mode = candidate.stat().st_mode
        if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            raise ValueError(f"model file must not be executable: {relative}")
        inventory.append(
            {
                "path": relative,
                "size_bytes": candidate.stat().st_size,
                "sha256": sha256_file(candidate),
                "symlink": is_symlink,
            }
        )
    if not inventory:
        raise ValueError("model tree is empty")
    return tuple(inventory)


def _json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON file {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON file {path.name} must contain an object")
    return {str(key): item for key, item in value.items()}


def audit_source_snapshot(
    snapshot: Path, *, contract: MLXSourceContract = STORIES15M_SOURCE_CONTRACT
) -> dict[str, Any]:
    """Fail closed on the exact local source before importing MLX or Transformers."""

    snapshot = snapshot.expanduser().resolve(strict=True)
    if snapshot.name != contract.revision:
        raise ValueError(f"source snapshot directory must be revision {contract.revision}")
    inventory = _audit_tree(
        snapshot,
        allowed_suffixes=_SOURCE_SUFFIXES,
        allow_hf_cache_symlinks=True,
    )
    names = {str(item["path"]) for item in inventory}
    if names != contract.file_names:
        raise ValueError(
            "source snapshot file set mismatch; "
            f"missing={sorted(contract.file_names - names)}, "
            f"extra={sorted(names - contract.file_names)}"
        )
    expected_files = {
        "config.json": contract.config_sha256,
        "tokenizer_config.json": contract.tokenizer_config_sha256,
        "tokenizer.json": contract.tokenizer_sha256,
        "model.safetensors": contract.weight_sha256,
    }
    actual = {name: sha256_file(snapshot / name) for name in expected_files}
    mismatches = [name for name, digest in expected_files.items() if actual[name] != digest]
    if mismatches:
        raise ValueError("source snapshot checksum mismatch: " + ", ".join(mismatches))
    weight = snapshot / "model.safetensors"
    if weight.stat().st_size != contract.weight_size_bytes:
        raise ValueError("source model.safetensors byte-size mismatch")
    config = _json_mapping(snapshot / "config.json")
    validate_model_config_security(config, source="source snapshot")
    tokenizer_config = _json_mapping(snapshot / "tokenizer_config.json")
    validate_tokenizer_config_security(
        tokenizer_config,
        source="source snapshot",
        expected_class="LlamaTokenizer",
    )
    if contract.tensor_count is not None:
        with safe_open(str(weight), framework="np") as handle:
            if handle.metadata() != {"format": "pt"}:
                raise ValueError("source safetensors metadata must declare format=pt")
            if len(handle.keys()) != contract.tensor_count:
                raise ValueError("source safetensors tensor count mismatch")
    return {
        "revision": contract.revision,
        "config_sha256": contract.config_sha256,
        "tokenizer_config_sha256": contract.tokenizer_config_sha256,
        "tokenizer_sha256": contract.tokenizer_sha256,
        "weight_sha256": contract.weight_sha256,
        "weight_size_bytes": contract.weight_size_bytes,
        "safetensors_tensor_count": contract.tensor_count,
        "file_count": len(inventory),
        "python_file_count": 0,
        "model_file": None,
        "auto_map": None,
        "remote_code_executed": False,
    }


def expected_quantized_linear_names() -> frozenset[str]:
    """Return the 31 exact MLX dense-linear leaves in the validated model."""

    names = {"lm_head"}
    for layer in range(6):
        prefix = f"model.layers.{layer}"
        names.add(f"{prefix}.block_sparse_moe.gate")
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            names.add(f"{prefix}.self_attn.{projection}")
    return frozenset(names)


def expected_quantized_expert_names() -> frozenset[str]:
    """Return all 18 fused expert projection leaves."""

    return frozenset(
        f"model.layers.{layer}.block_sparse_moe.switch_mlp.{projection}"
        for layer in range(6)
        for projection in ("gate_proj", "up_proj", "down_proj")
    )


def expected_quantized_leaf_names() -> frozenset[str]:
    """Return every eligible leaf: embedding, dense linears, and fused experts."""

    return frozenset(
        {"model.embed_tokens"}
        | set(expected_quantized_linear_names())
        | set(expected_quantized_expert_names())
    )


def expected_float32_only_names() -> frozenset[str]:
    """Return all 13 RMSNorm leaves excluded from affine quantization."""

    return frozenset(
        {"model.norm"}
        | {
            f"model.layers.{layer}.{kind}_layernorm"
            for layer in range(6)
            for kind in ("input", "post_attention")
        }
    )


def validate_quantized_tensor_facts(
    facts: Mapping[str, tuple[str, tuple[int, ...], int]], *, bits: int
) -> dict[str, Any]:
    """Prove that every exact eligible leaf has packed affine tensor state."""

    if bits not in MLX_CONVERSION_CONTRACTS:
        raise ValueError("MLX affine supports only the evidenced 4-bit and 8-bit formats")
    expected = expected_quantized_leaf_names()
    scale_keys = {key for key in facts if key.endswith(".scales")}
    expected_scales = {f"{name}.scales" for name in expected}
    if scale_keys != expected_scales:
        raise ValueError(
            "quantized scale-key mismatch; "
            f"missing={sorted(expected_scales - scale_keys)}, "
            f"extra={sorted(scale_keys - expected_scales)}"
        )
    expected_biases = {f"{name}.biases" for name in expected}
    bias_keys = {key for key in facts if key.endswith(".biases")}
    if bias_keys != expected_biases:
        raise ValueError("affine bias-key set does not match every quantized leaf")
    for name in expected:
        weight = facts.get(f"{name}.weight")
        scales = facts[f"{name}.scales"]
        biases = facts[f"{name}.biases"]
        if weight is None or weight[0] != "uint32":
            raise ValueError(f"quantized leaf has no uint32 packed weight: {name}")
        if scales[0] != "float32" or biases[0] != "float32":
            raise ValueError(f"quantized leaf has invalid affine metadata dtype: {name}")
    for name in expected_quantized_expert_names():
        projection = name.rpartition(".")[2]
        output_dims, input_dims = {
            "gate_proj": (768, 288),
            "up_proj": (768, 288),
            "down_proj": (288, 768),
        }[projection]
        packed = (4, output_dims, input_dims * bits // 32)
        auxiliary = (4, output_dims, input_dims // 32)
        if facts[f"{name}.weight"][:2] != ("uint32", packed):
            raise ValueError(f"fused expert packed shape mismatch: {name}")
        if facts[f"{name}.scales"][:2] != ("float32", auxiliary):
            raise ValueError(f"fused expert scale shape mismatch: {name}")
        if facts[f"{name}.biases"][:2] != ("float32", auxiliary):
            raise ValueError(f"fused expert affine-bias shape mismatch: {name}")
    return {
        "quantized_leaf_count": len(expected),
        "quantized_linear_count": len(expected_quantized_linear_names()),
        "quantized_embedding_count": 1,
        "quantized_fused_expert_projection_count": len(expected_quantized_expert_names()),
        "fused_expert_layers": 6,
        "experts_per_layer": 4,
        "bits": bits,
        "group_size": 32,
        "mode": "affine",
        "packed_weight_dtype": "uint32",
    }


def runtime_quantization_proof(model: Any, *, bits: int) -> dict[str, Any]:
    """Inspect reloaded runtime classes and exact names, not configuration labels."""

    named = {str(name): module for name, module in model.named_modules()}
    linears = {
        name for name, module in named.items() if type(module).__qualname__ == "QuantizedLinear"
    }
    embeddings = {
        name for name, module in named.items() if type(module).__qualname__ == "QuantizedEmbedding"
    }
    experts = {
        name
        for name, module in named.items()
        if type(module).__qualname__ == "QuantizedSwitchLinear"
    }
    if linears != set(expected_quantized_linear_names()):
        raise ValueError("reloaded quantized linear module-name set mismatch")
    if embeddings != {"model.embed_tokens"}:
        raise ValueError("reloaded quantized embedding module-name set mismatch")
    if experts != set(expected_quantized_expert_names()):
        raise ValueError("reloaded fused-expert module-name set mismatch")
    for name in sorted(linears | embeddings | experts):
        module = named[name]
        metadata = (
            int(module.bits),
            int(module.group_size),
            str(module.mode),
            str(module.weight.dtype).removeprefix("mlx.core."),
        )
        if metadata != (bits, 32, "affine", "uint32"):
            raise ValueError(f"reloaded quantization metadata mismatch: {name}")
    return {
        "quantized_leaf_count": 50,
        "quantized_linear_names": tuple(sorted(linears)),
        "quantized_embedding_names": tuple(sorted(embeddings)),
        "quantized_fused_expert_projection_names": tuple(sorted(experts)),
        "kernel": "mlx.core.gather_qmm via QuantizedSwitchLinear for fused experts",
    }


def audit_converted_bundle(path: Path, *, bits: int) -> dict[str, Any]:
    """Validate exact deterministic MLX-LM bytes plus optional Inkling manifest."""

    contract = MLX_CONVERSION_CONTRACTS[bits]
    path = path.expanduser().resolve(strict=True)
    inventory = _audit_tree(
        path,
        allowed_suffixes=_BUNDLE_SUFFIXES,
        allow_hf_cache_symlinks=False,
    )
    names = {str(item["path"]) for item in inventory}
    permitted = set(contract.files) | {"inkling_quant_manifest.json"}
    if names not in (set(contract.files), permitted):
        raise ValueError(
            "converted bundle file set mismatch; "
            f"missing={sorted(set(contract.files) - names)}, "
            f"extra={sorted(names - permitted)}"
        )
    facts_by_name = {str(item["path"]): item for item in inventory}
    for name, (size, digest) in contract.files.items():
        fact = facts_by_name[name]
        if fact["size_bytes"] != size or fact["sha256"] != digest:
            raise ValueError(f"converted bundle deterministic fact mismatch: {name}")
    config = _json_mapping(path / "config.json")
    validate_model_config_security(config, source=contract.label)
    if config.get("quantization") != {"bits": bits, "group_size": 32, "mode": "affine"}:
        raise ValueError("converted bundle quantization metadata mismatch")
    tokenizer = _json_mapping(path / "tokenizer_config.json")
    validate_tokenizer_config_security(
        tokenizer,
        source=contract.label,
        expected_class="TokenizersBackend",
    )
    weight = path / "model.safetensors"
    tensor_facts: dict[str, tuple[str, tuple[int, ...], int]] = {}
    with safe_open(str(weight), framework="np") as handle:
        if handle.metadata() != {"format": "mlx"}:
            raise ValueError("converted safetensors metadata must declare format=mlx")
        for key in handle.keys():  # noqa: SIM118 - safe_open is not an iterable mapping
            tensor = handle.get_tensor(key)
            tensor_facts[key] = (
                str(tensor.dtype),
                tuple(int(value) for value in tensor.shape),
                int(tensor.nbytes),
            )
    if len(tensor_facts) != contract.tensor_count:
        raise ValueError("converted safetensors tensor count mismatch")
    tensor_bytes = sum(value[2] for value in tensor_facts.values())
    if tensor_bytes != contract.tensor_bytes:
        raise ValueError("converted safetensors tensor-byte total mismatch")
    tensor_proof = validate_quantized_tensor_facts(tensor_facts, bits=bits)
    index = _json_mapping(path / "model.safetensors.index.json")
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    if not isinstance(metadata, dict) or metadata.get("total_size") != tensor_bytes:
        raise ValueError("converted index total_size mismatch")
    if not isinstance(weight_map, dict) or set(weight_map) != set(tensor_facts):
        raise ValueError("converted index does not cover the exact tensor-key set")
    if set(weight_map.values()) != {"model.safetensors"}:
        raise ValueError("converted index points outside the one safe tensor shard")
    return {
        "label": contract.label,
        "bits": bits,
        "base_bundle_size_bytes": contract.base_bundle_size_bytes,
        "bundle_size_bytes": sum(int(item["size_bytes"]) for item in inventory),
        "tensor_bytes": tensor_bytes,
        "stored_tensor_count": len(tensor_facts),
        "weight_sha256": contract.files["model.safetensors"][1],
        "index_sha256": contract.files["model.safetensors.index.json"][1],
        "config_sha256": contract.files["config.json"][1],
        "quantized_tensor_proof": tensor_proof,
        "symlink_count": 0,
        "python_file_count": 0,
    }


def directory_sha256(path: Path) -> tuple[str, tuple[str, ...]]:
    """Hash the exact safe regular-file set, binding names and contents."""

    files = tuple(
        sorted(
            candidate.relative_to(path).as_posix()
            for candidate in path.rglob("*")
            if candidate.is_file() and not candidate.is_symlink()
        )
    )
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path / name)))
    return digest.hexdigest(), files


__all__ = [
    "CANONICAL_MLX_MODEL_CARD_BYTES",
    "EXPECTED_MLX_VERSIONS",
    "MLX_CONVERSION_CONTRACTS",
    "MLX_MODEL_CLASS",
    "MODEL_ARCHITECTURE",
    "MODEL_ID",
    "MODEL_REVISION",
    "SOURCE_WEIGHT_SHA256",
    "STORIES15M_SOURCE_CONTRACT",
    "MLXEnvironmentStatus",
    "MLXSourceContract",
    "audit_converted_bundle",
    "audit_source_snapshot",
    "directory_sha256",
    "expected_float32_only_names",
    "expected_quantized_expert_names",
    "expected_quantized_leaf_names",
    "expected_quantized_linear_names",
    "mlx_environment_status",
    "runtime_quantization_proof",
    "sha256_file",
    "validate_model_config_security",
    "validate_quantized_tensor_facts",
    "validate_tokenizer_config_security",
]
