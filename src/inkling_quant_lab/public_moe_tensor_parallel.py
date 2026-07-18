"""Two-rank tensor-parallel execution of every expert block in a pinned public MoE."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import platform
import stat
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, Self, cast

import safetensors
import torch
import torch.distributed as dist
import torch.nn.functional as functional
from pydantic import BaseModel, ConfigDict, Field, model_validator
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from inkling_quant_lab.distributed import probe_gloo_capability
from inkling_quant_lab.exceptions import CapabilityError
from inkling_quant_lab.security import sensitive_literal_path

MODEL_ID = "ggml-org/stories15M_MOE"
MODEL_REVISION = "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
MODEL_WEIGHT_SHA256 = "dbfa0289f68a8dd721d10eb12d8bd82e098455682027f6f9986ba548913f9082"
MODEL_WEIGHT_SIZE_BYTES = 72_744_704
MODEL_CONFIG_SHA256 = "e901e012953d1df93574b2cc3d7db5ed4758d52f8bd4a7dd4b647936e32261be"
TOKENIZER_SHA256 = "8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1"
TOKENIZER_CONFIG_SHA256 = "33d29c87e41f7dd1efb0434d852730320c82970f292be452d820539bce417052"

_EXPECTED_SOURCE_FILES: dict[str, tuple[int, str]] = {
    "config.json": (711, MODEL_CONFIG_SHA256),
    "data.txt": (66_884, "8aba9f75ac88b48a9576935995a26b806f3d2cd75f6ca9267fe0fba79cd3f64f"),
    "generation_config.json": (
        115,
        "295aa491adda22ab9fbdecdda9e8121e8348fd0eea0529d8802993426ab0892c",
    ),
    "model.safetensors": (MODEL_WEIGHT_SIZE_BYTES, MODEL_WEIGHT_SHA256),
    "model.safetensors.index.json": (
        8_890,
        "cd4af1df494f42e9099c1bdf9ea8487c06166c8ca60f6264a617497a8151fd31",
    ),
    "moe_shakespeare15M/checkpoint-400/adapter_config.json": (
        723,
        "bc7a694ce0155d5ca6f7601f00fd61017b2f67615a5338745b5b5c97c43e89bb",
    ),
    "moe_shakespeare15M/checkpoint-400/trainer_state.json": (
        3_693,
        "91d96f6b1afb85e0f28ca7da468409d8585a0dc4384cbf84d5560a0b8ee1e122",
    ),
    "moe_shakespeare15M/checkpoint-500/adapter_config.json": (
        723,
        "bc7a694ce0155d5ca6f7601f00fd61017b2f67615a5338745b5b5c97c43e89bb",
    ),
    "moe_shakespeare15M/checkpoint-500/trainer_state.json": (
        4_416,
        "f612e4f0b86fa83474b917d064ae4997750d28487d3e7edd1697ecd26ac85ddc",
    ),
    "special_tokens_map.json": (
        411,
        "ff3b4a612c4e447acb02d40071bddd989fe0da87eb5b7fe0dbadfc4f74de7531",
    ),
    "tokenizer.json": (1_842_764, TOKENIZER_SHA256),
    "tokenizer_config.json": (686, TOKENIZER_CONFIG_SHA256),
}

_LAYER_COUNT = 6
_EXPERT_COUNT = 4
_TOP_K = 2
_HIDDEN_SIZE = 288
_INTERMEDIATE_SIZE = 768
_LOCAL_INTERMEDIATE_SIZE = _INTERMEDIATE_SIZE // 2
_INPUT_RANGES = ((0, 32), (256, 288))
_INPUT_ROWS = 64
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SCOPE = (
    "two-rank local CPU Gloo tensor parallel over every expert block in the pinned public "
    "Stories15M MoE; each rank reads only its exact intermediate-dimension safetensors slices "
    "for forward execution and all-reduces partial expert outputs; this is not an end-to-end "
    "transformer forward, generic sharded loader, performance result, or multi-host runtime"
)


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceFileEvidence(_ImmutableModel):
    """One exact, safe file in the pinned Hugging Face snapshot."""

    path: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    symlink: bool


class PublicMoeSourceEvidence(_ImmutableModel):
    """Immutable public checkpoint identity audited before distributed execution."""

    model_id: Literal["ggml-org/stories15M_MOE"] = "ggml-org/stories15M_MOE"
    revision: Literal["b6dd737497465570b5f5e962dbc9d9454ed1e0eb"] = (
        "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
    )
    location: Literal["local_huggingface_cache_snapshot"] = "local_huggingface_cache_snapshot"
    config_sha256: Literal["e901e012953d1df93574b2cc3d7db5ed4758d52f8bd4a7dd4b647936e32261be"] = (
        "e901e012953d1df93574b2cc3d7db5ed4758d52f8bd4a7dd4b647936e32261be"
    )
    tokenizer_sha256: Literal[
        "8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1"
    ] = "8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1"
    tokenizer_config_sha256: Literal[
        "33d29c87e41f7dd1efb0434d852730320c82970f292be452d820539bce417052"
    ] = "33d29c87e41f7dd1efb0434d852730320c82970f292be452d820539bce417052"
    weight_sha256: Literal["dbfa0289f68a8dd721d10eb12d8bd82e098455682027f6f9986ba548913f9082"] = (
        "dbfa0289f68a8dd721d10eb12d8bd82e098455682027f6f9986ba548913f9082"
    )
    weight_size_bytes: Literal[72744704] = 72_744_704
    safetensors_tensor_count: Literal[117] = 117
    file_inventory: tuple[SourceFileEvidence, ...]
    python_file_count: Literal[0] = 0
    native_file_count: Literal[0] = 0
    pickle_file_count: Literal[0] = 0
    remote_code_executed: Literal[False] = False

    @model_validator(mode="after")
    def exact_inventory(self) -> Self:
        observed = {item.path: (item.size_bytes, item.sha256) for item in self.file_inventory}
        if observed != _EXPECTED_SOURCE_FILES:
            raise ValueError("source evidence does not contain the exact pinned file inventory")
        if len(observed) != len(self.file_inventory):
            raise ValueError("source file inventory contains duplicate paths")
        return self


class ExpertShardEvidence(_ImmutableModel):
    """One exact rank-local expert projection slice and reconstruction proof."""

    layer_id: int = Field(ge=0, lt=_LAYER_COUNT)
    expert_id: int = Field(ge=0, lt=_EXPERT_COUNT)
    projection: Literal["w1", "w2", "w3"]
    tensor_name: str = Field(min_length=1)
    global_shape: tuple[int, int]
    local_shape: tuple[int, int]
    shard_dim: Literal[0, 1]
    shard_range: tuple[int, int]
    source_storage_dtype: Literal["float16"] = "float16"
    execution_dtype: Literal["float32"] = "float32"
    source_tensor_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_float32_sha256: str = Field(pattern=_SHA256_PATTERN)
    local_float32_sha256: str = Field(pattern=_SHA256_PATTERN)
    reconstructed_float32_sha256: str = Field(pattern=_SHA256_PATTERN)
    local_execution_bytes: Literal[442368] = 442_368

    @model_validator(mode="after")
    def exact_shard_contract(self) -> Self:
        expected_name = _expert_tensor_name(self.layer_id, self.expert_id, self.projection)
        if self.tensor_name != expected_name:
            raise ValueError("expert tensor name differs from layer/expert/projection identity")
        expected_global = (
            (_INTERMEDIATE_SIZE, _HIDDEN_SIZE)
            if self.projection in {"w1", "w3"}
            else (_HIDDEN_SIZE, _INTERMEDIATE_SIZE)
        )
        expected_dim = 0 if self.projection in {"w1", "w3"} else 1
        expected_local = list(expected_global)
        expected_local[expected_dim] = _LOCAL_INTERMEDIATE_SIZE
        if (
            self.global_shape != expected_global
            or self.shard_dim != expected_dim
            or self.local_shape != tuple(expected_local)
        ):
            raise ValueError("expert shard shape or dimension differs from the exact plan")
        start, end = self.shard_range
        if end - start != _LOCAL_INTERMEDIATE_SIZE or start not in {0, _LOCAL_INTERMEDIATE_SIZE}:
            raise ValueError("expert shard range must be one exact half of the intermediate axis")
        if self.reconstructed_float32_sha256 != self.source_float32_sha256:
            raise ValueError("collectively reconstructed float32 tensor differs from the source")
        return self


class PublicMoeRankEvidence(_ImmutableModel):
    """One rank's public expert parameter slices and output identities."""

    rank: Literal[0, 1]
    device: Literal["cpu"] = "cpu"
    intraop_threads: Literal[1] = 1
    interop_threads: Literal[1] = 1
    source_loading_api: Literal["safetensors.safe_open.get_slice"] = (
        "safetensors.safe_open.get_slice"
    )
    forward_loaded_only_rank_local_expert_slices: Literal[True] = True
    full_tensor_reconstruction_timing: Literal["after_local_projection_forward"] = (
        "after_local_projection_forward"
    )
    input_sha256: str = Field(pattern=_SHA256_PATTERN)
    local_expert_parameter_bytes: Literal[31850496] = 31_850_496
    full_float32_expert_parameter_bytes: Literal[63700992] = 63_700_992
    local_fraction: float = Field(default=0.5, ge=0.0, le=1.0, allow_inf_nan=False)
    shard_evidence: tuple[ExpertShardEvidence, ...]
    layer_output_sha256: tuple[str, ...]

    @model_validator(mode="after")
    def complete_rank_contract(self) -> Self:
        expected = {
            (layer, expert, projection)
            for layer in range(_LAYER_COUNT)
            for expert in range(_EXPERT_COUNT)
            for projection in ("w1", "w2", "w3")
        }
        observed = {
            (item.layer_id, item.expert_id, item.projection) for item in self.shard_evidence
        }
        if observed != expected or len(self.shard_evidence) != len(expected):
            raise ValueError("rank evidence must contain all 72 public expert projection shards")
        if self.local_fraction != 0.5:
            raise ValueError("rank must retain exactly half of the public expert parameters")
        expected_range = (
            (0, _LOCAL_INTERMEDIATE_SIZE)
            if self.rank == 0
            else (_LOCAL_INTERMEDIATE_SIZE, _INTERMEDIATE_SIZE)
        )
        if any(item.shard_range != expected_range for item in self.shard_evidence):
            raise ValueError("rank shard ranges do not match the process rank")
        if len(self.layer_output_sha256) != _LAYER_COUNT or any(
            len(value) != 64 for value in self.layer_output_sha256
        ):
            raise ValueError("rank must retain six SHA-256 distributed output identities")
        return self


class PublicMoeLayerEvidence(_ImmutableModel):
    """One complete public MoE layer's routed distributed forward comparison."""

    layer_id: int = Field(ge=0, lt=_LAYER_COUNT)
    selected_expert_counts: tuple[int, int, int, int]
    selected_expert_count: Literal[128] = 128
    all_experts_executed: Literal[True] = True
    reference_output_sha256: str = Field(pattern=_SHA256_PATTERN)
    distributed_output_sha256: str = Field(pattern=_SHA256_PATTERN)
    rank_max_abs_errors: tuple[float, float]
    rank_max_relative_errors: tuple[float, float]
    rank_max_tolerance_ratios: tuple[float, float]

    @model_validator(mode="after")
    def exact_layer_contract(self) -> Self:
        if sum(self.selected_expert_counts) != self.selected_expert_count:
            raise ValueError("layer expert counts do not equal rows multiplied by top-k")
        if any(count <= 0 for count in self.selected_expert_counts):
            raise ValueError("every public expert must execute at least one routed token")
        if len(set(self.rank_max_abs_errors)) > 1 or len(set(self.rank_max_relative_errors)) > 1:
            raise ValueError("both ranks must compare the same reduced output")
        if len(set(self.rank_max_tolerance_ratios)) > 1:
            raise ValueError("both ranks must retain the same combined-tolerance ratio")
        if any(not math.isfinite(value) or value < 0.0 for value in self.rank_max_abs_errors):
            raise ValueError("absolute errors must be finite and non-negative")
        if any(not math.isfinite(value) or value < 0.0 for value in self.rank_max_relative_errors):
            raise ValueError("relative errors must be finite and non-negative")
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in self.rank_max_tolerance_ratios
        ):
            raise ValueError("combined-tolerance ratios must be finite and at most one")
        return self


class PublicMoeTensorParallelHardware(_ImmutableModel):
    """Host facts attached to the executed public-model result."""

    system: str = Field(min_length=1)
    release: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    processor: str
    logical_cpu_count: int = Field(gt=0)
    physical_memory_bytes: int | None = Field(default=None, gt=0)


class PublicMoeTensorParallelSoftware(_ImmutableModel):
    """Exact interpreter, tensor, and checkpoint-reader versions."""

    python: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    torch_git_version: str | None = None
    safetensors: str = Field(min_length=1)


class PublicMoeTensorParallelResult(_ImmutableModel):
    """Executed all-layer public-MoE expert tensor-parallel evidence."""

    schema_version: Literal["public-moe-expert-tensor-parallel-v1"] = (
        "public-moe-expert-tensor-parallel-v1"
    )
    backend: Literal["gloo"] = "gloo"
    world_size: Literal[2] = 2
    rendezvous: Literal["ephemeral_file"] = "ephemeral_file"
    process_start_method: Literal["spawn"] = "spawn"
    interface: Literal["lo", "lo0"]
    source: PublicMoeSourceEvidence
    layer_count: Literal[6] = 6
    experts_per_layer: Literal[4] = 4
    top_k: Literal[2] = 2
    hidden_size: Literal[288] = 288
    intermediate_size: Literal[768] = 768
    input_kind: Literal["two_exact_source_embedding_row_ranges"] = (
        "two_exact_source_embedding_row_ranges"
    )
    input_row_ranges: tuple[tuple[int, int], tuple[int, int]] = _INPUT_RANGES
    input_row_count: Literal[64] = 64
    input_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_expert_storage_dtype: Literal["float16"] = "float16"
    execution_dtype: Literal["float32"] = "float32"
    absolute_tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    relative_tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    layers: tuple[PublicMoeLayerEvidence, ...]
    ranks: tuple[PublicMoeRankEvidence, PublicMoeRankEvidence]
    public_checkpoint_executed: Literal[True] = True
    all_public_moe_expert_blocks_tensor_parallel_validated: Literal[True] = True
    parameter_sharded_forward_validated: Literal[True] = True
    end_to_end_transformer_forward_validated: Literal[False] = False
    sharded_checkpoint_export_validated: Literal[False] = False
    distributed_training_validated: Literal[False] = False
    performance_validated: Literal[False] = False
    multi_host_validated: Literal[False] = False
    scope: str = _SCOPE
    operator_declared_hardware_label: str = Field(min_length=1)
    hardware: PublicMoeTensorParallelHardware
    software: PublicMoeTensorParallelSoftware

    @model_validator(mode="after")
    def complete_result_contract(self) -> Self:
        if self.scope != _SCOPE:
            raise ValueError("result scope differs from the exact accepted claim boundary")
        if tuple(layer.layer_id for layer in self.layers) != tuple(range(_LAYER_COUNT)):
            raise ValueError("result must retain the six public MoE layers in order")
        if tuple(rank.rank for rank in self.ranks) != (0, 1):
            raise ValueError("result must retain ranks zero and one in order")
        if any(rank.input_sha256 != self.input_sha256 for rank in self.ranks):
            raise ValueError("every rank must execute the exact declared input")
        for layer in self.layers:
            rank_hashes = tuple(rank.layer_output_sha256[layer.layer_id] for rank in self.ranks)
            if rank_hashes != (layer.distributed_output_sha256,) * 2:
                raise ValueError("rank and layer distributed output identities differ")
            if max(layer.rank_max_abs_errors) > self.absolute_tolerance:
                raise ValueError("layer absolute error exceeds the declared tolerance")
        if (
            not self.operator_declared_hardware_label
            or self.operator_declared_hardware_label
            != self.operator_declared_hardware_label.strip()
        ):
            raise ValueError("hardware label must be non-empty without boundary whitespace")
        if sensitive_literal_path(self.operator_declared_hardware_label) is not None:
            raise ValueError("hardware label must not contain credential-like material")
        return self


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Tensor) -> str:
    normalized = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(normalized.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(normalized.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _expert_tensor_name(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{projection}.weight"


def _router_tensor_name(layer: int) -> str:
    return f"model.layers.{layer}.block_sparse_moe.gate.weight"


def audit_stories15m_snapshot(snapshot: Path) -> PublicMoeSourceEvidence:
    """Fail closed unless ``snapshot`` is the exact safe public source revision."""

    root = snapshot.expanduser().resolve(strict=True)
    if not root.is_dir() or root.name != MODEL_REVISION:
        raise ValueError("source snapshot must be the exact pinned revision directory")
    allowed_resolution_root = root.parents[1].resolve(strict=True)
    inventory: list[SourceFileEvidence] = []
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root).as_posix()
        is_symlink = candidate.is_symlink()
        if is_symlink and candidate.is_dir():
            raise ValueError(f"source snapshot must not contain directory symlinks: {relative}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"source snapshot contains a non-regular file: {relative}")
        suffix = candidate.suffix.lower()
        if suffix in {".py", ".pyc", ".so", ".dylib", ".dll", ".bin", ".pt", ".pth", ".pkl"}:
            raise ValueError(
                f"source snapshot contains an unsafe executable/pickle file: {relative}"
            )
        if is_symlink:
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(allowed_resolution_root):
                raise ValueError(
                    f"source snapshot symlink escapes its repository cache: {relative}"
                )
        mode = candidate.stat().st_mode
        if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            raise ValueError(f"source snapshot contains an executable file: {relative}")
        inventory.append(
            SourceFileEvidence(
                path=relative,
                size_bytes=candidate.stat().st_size,
                sha256=_file_sha256(candidate),
                symlink=is_symlink,
            )
        )
    source = PublicMoeSourceEvidence(file_inventory=tuple(inventory))

    config_value = json.loads((root / "config.json").read_text(encoding="utf-8"))
    expected_config = {
        "model_type": "mixtral",
        "architectures": ["MixtralForCausalLM"],
        "hidden_size": _HIDDEN_SIZE,
        "intermediate_size": _INTERMEDIATE_SIZE,
        "num_hidden_layers": _LAYER_COUNT,
        "num_local_experts": _EXPERT_COUNT,
        "num_experts_per_tok": _TOP_K,
    }
    if not isinstance(config_value, dict) or any(
        config_value.get(key) != value for key, value in expected_config.items()
    ):
        raise ValueError("source config differs from the exact public Mixtral architecture")
    if config_value.get("auto_map") is not None or config_value.get("model_file") is not None:
        raise ValueError("source config selects custom model code")

    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        if checkpoint.metadata() != {"format": "pt"} or len(checkpoint.keys()) != 117:
            raise ValueError("source safetensors metadata or tensor count differs from the pin")
        embedding = checkpoint.get_slice("model.embed_tokens.weight")
        if embedding.get_shape() != [32_000, _HIDDEN_SIZE] or embedding.get_dtype() != "F16":
            raise ValueError("source embedding tensor differs from the exact workload contract")
        for layer in range(_LAYER_COUNT):
            router = checkpoint.get_slice(_router_tensor_name(layer))
            if router.get_shape() != [_EXPERT_COUNT, _HIDDEN_SIZE] or router.get_dtype() != "F32":
                raise ValueError("source router tensor differs from the exact public architecture")
            for expert in range(_EXPERT_COUNT):
                for projection in ("w1", "w2", "w3"):
                    tensor = checkpoint.get_slice(_expert_tensor_name(layer, expert, projection))
                    expected_shape = (
                        [_INTERMEDIATE_SIZE, _HIDDEN_SIZE]
                        if projection in {"w1", "w3"}
                        else [_HIDDEN_SIZE, _INTERMEDIATE_SIZE]
                    )
                    if tensor.get_shape() != expected_shape or tensor.get_dtype() != "F16":
                        raise ValueError(
                            "source expert tensor differs from the exact sharding plan"
                        )
    return source


def _workload_input(checkpoint: Any) -> Tensor:
    embedding = checkpoint.get_slice("model.embed_tokens.weight")
    return (
        torch.cat(tuple(embedding[start:end] for start, end in _INPUT_RANGES)).float().contiguous()
    )


def _routing(hidden_states: Tensor, router_weight: Tensor) -> tuple[Tensor, Tensor]:
    probabilities = torch.softmax(functional.linear(hidden_states, router_weight.float()), dim=-1)
    weights, indices = torch.topk(probabilities, _TOP_K, dim=-1)
    return weights / weights.sum(dim=-1, keepdim=True), indices


def _full_expert_forward(
    checkpoint: Any,
    *,
    layer: int,
    hidden_states: Tensor,
    top_k_weights: Tensor,
    top_k_indices: Tensor,
) -> Tensor:
    output = torch.zeros_like(hidden_states)
    expert_mask = functional.one_hot(top_k_indices, num_classes=_EXPERT_COUNT).permute(2, 1, 0)
    for expert in range(_EXPERT_COUNT):
        top_k_position, token_index = torch.where(expert_mask[expert])
        current = hidden_states[token_index]
        w1 = checkpoint.get_tensor(_expert_tensor_name(layer, expert, "w1")).float()
        w2 = checkpoint.get_tensor(_expert_tensor_name(layer, expert, "w2")).float()
        w3 = checkpoint.get_tensor(_expert_tensor_name(layer, expert, "w3")).float()
        activated = functional.silu(functional.linear(current, w1)) * functional.linear(current, w3)
        contribution = functional.linear(activated, w2)
        contribution *= top_k_weights[token_index, top_k_position, None]
        output.index_add_(0, token_index, contribution)
    return output


def _build_reference_files(snapshot: Path, directory: Path) -> tuple[Path, Path, dict[str, Any]]:
    expected_path = directory / "reference.safetensors"
    hashes_path = directory / "source-tensor-hashes.json"
    expected_tensors: dict[str, Tensor] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    layer_metadata: list[dict[str, Any]] = []
    with safe_open(snapshot / "model.safetensors", framework="pt", device="cpu") as checkpoint:
        hidden_states = _workload_input(checkpoint)
        expected_tensors["input"] = hidden_states
        for layer in range(_LAYER_COUNT):
            router = checkpoint.get_tensor(_router_tensor_name(layer)).float()
            top_k_weights, top_k_indices = _routing(hidden_states, router)
            reference = _full_expert_forward(
                checkpoint,
                layer=layer,
                hidden_states=hidden_states,
                top_k_weights=top_k_weights,
                top_k_indices=top_k_indices,
            )
            expected_tensors[f"layer_{layer}"] = reference
            counts = torch.bincount(top_k_indices.flatten(), minlength=_EXPERT_COUNT)
            layer_metadata.append(
                {
                    "layer_id": layer,
                    "selected_expert_counts": [int(value) for value in counts.tolist()],
                    "reference_output_sha256": _tensor_sha256(reference),
                }
            )
            for expert in range(_EXPERT_COUNT):
                for projection in ("w1", "w2", "w3"):
                    name = _expert_tensor_name(layer, expert, projection)
                    tensor = checkpoint.get_tensor(name)
                    source_hashes[name] = {
                        "source_tensor_sha256": _tensor_sha256(tensor),
                        "source_float32_sha256": _tensor_sha256(tensor.float()),
                    }
    save_file(expected_tensors, expected_path, metadata={"format": "pt"})
    hashes_path.write_text(
        json.dumps(source_hashes, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return (
        expected_path,
        hashes_path,
        {
            "input_sha256": _tensor_sha256(expected_tensors["input"]),
            "layers": layer_metadata,
        },
    )


def _rank_slice(checkpoint: Any, name: str, projection: str, rank: int) -> Tensor:
    start = rank * _LOCAL_INTERMEDIATE_SIZE
    end = start + _LOCAL_INTERMEDIATE_SIZE
    tensor_slice = checkpoint.get_slice(name)
    if projection in {"w1", "w3"}:
        return cast(Tensor, tensor_slice[start:end, :]).float().contiguous()
    return cast(Tensor, tensor_slice[:, start:end]).float().contiguous()


def _reconstruction_hash(local: Tensor, shard_dim: int) -> str:
    gathered = [torch.empty_like(local) for _ in range(2)]
    dist.all_gather(gathered, local)
    reconstructed = torch.cat(gathered, dim=shard_dim)
    return _tensor_sha256(reconstructed)


def _relative_error(actual: Tensor, expected: Tensor, absolute_tolerance: float) -> float:
    difference = (actual - expected).abs()
    denominator = torch.maximum(expected.abs(), torch.full_like(expected, absolute_tolerance))
    return float((difference / denominator).max().item())


def _combined_tolerance_ratio(
    actual: Tensor,
    expected: Tensor,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    difference = (actual - expected).abs()
    allowance = absolute_tolerance + relative_tolerance * expected.abs()
    return float((difference / allowance).max().item())


def _worker(
    rank: int,
    init_file: str,
    result_directory: str,
    snapshot: str,
    expected_path: str,
    hashes_path: str,
    interface: str,
    timeout_seconds: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    result_path = Path(result_directory) / f"rank-{rank}.json"
    error_path = Path(result_directory) / f"rank-{rank}.error.json"
    os.environ["GLOO_SOCKET_IFNAME"] = interface
    try:
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=timeout_seconds),
        )
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        source_hashes = json.loads(Path(hashes_path).read_text(encoding="utf-8"))
        shards: list[dict[str, Any]] = []
        layer_results: list[dict[str, Any]] = []
        with (
            safe_open(Path(snapshot) / "model.safetensors", framework="pt", device="cpu") as source,
            safe_open(expected_path, framework="pt", device="cpu") as expected_file,
        ):
            hidden_states = _workload_input(source)
            if _tensor_sha256(hidden_states) != _tensor_sha256(expected_file.get_tensor("input")):
                raise ValueError("rank input differs from the parent reference input")
            for layer in range(_LAYER_COUNT):
                router = source.get_tensor(_router_tensor_name(layer)).float()
                top_k_weights, top_k_indices = _routing(hidden_states, router)
                expert_mask = functional.one_hot(top_k_indices, num_classes=_EXPERT_COUNT).permute(
                    2, 1, 0
                )
                partial_output = torch.zeros_like(hidden_states)
                for expert in range(_EXPERT_COUNT):
                    top_k_position, token_index = torch.where(expert_mask[expert])
                    current = hidden_states[token_index]
                    local_weights: dict[str, Tensor] = {}
                    for projection in ("w1", "w2", "w3"):
                        name = _expert_tensor_name(layer, expert, projection)
                        local = _rank_slice(source, name, projection, rank)
                        local_weights[projection] = local
                    activated = functional.silu(
                        functional.linear(current, local_weights["w1"])
                    ) * functional.linear(current, local_weights["w3"])
                    contribution = functional.linear(activated, local_weights["w2"])
                    contribution *= top_k_weights[token_index, top_k_position, None]
                    partial_output.index_add_(0, token_index, contribution)
                    for projection in ("w1", "w2", "w3"):
                        name = _expert_tensor_name(layer, expert, projection)
                        local = local_weights[projection]
                        shard_dim = 0 if projection in {"w1", "w3"} else 1
                        start = rank * _LOCAL_INTERMEDIATE_SIZE
                        hashes = source_hashes[name]
                        shards.append(
                            {
                                "layer_id": layer,
                                "expert_id": expert,
                                "projection": projection,
                                "tensor_name": name,
                                "global_shape": (
                                    [_INTERMEDIATE_SIZE, _HIDDEN_SIZE]
                                    if projection in {"w1", "w3"}
                                    else [_HIDDEN_SIZE, _INTERMEDIATE_SIZE]
                                ),
                                "local_shape": list(local.shape),
                                "shard_dim": shard_dim,
                                "shard_range": [start, start + _LOCAL_INTERMEDIATE_SIZE],
                                **hashes,
                                "local_float32_sha256": _tensor_sha256(local),
                                "reconstructed_float32_sha256": _reconstruction_hash(
                                    local, shard_dim
                                ),
                            }
                        )
                dist.all_reduce(partial_output, op=dist.ReduceOp.SUM)
                expected = expected_file.get_tensor(f"layer_{layer}")
                maximum_absolute_error = float((partial_output - expected).abs().max().item())
                maximum_relative_error = _relative_error(
                    partial_output, expected, absolute_tolerance
                )
                maximum_tolerance_ratio = _combined_tolerance_ratio(
                    partial_output,
                    expected,
                    absolute_tolerance,
                    relative_tolerance,
                )
                torch.testing.assert_close(
                    partial_output,
                    expected,
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                )
                layer_results.append(
                    {
                        "layer_id": layer,
                        "selected_expert_counts": [
                            int(value)
                            for value in torch.bincount(
                                top_k_indices.flatten(), minlength=_EXPERT_COUNT
                            ).tolist()
                        ],
                        "output_sha256": _tensor_sha256(partial_output),
                        "max_abs_error": maximum_absolute_error,
                        "max_relative_error": maximum_relative_error,
                        "max_tolerance_ratio": maximum_tolerance_ratio,
                    }
                )
        payload = {
            "rank": rank,
            "input_sha256": _tensor_sha256(hidden_states),
            "shard_evidence": shards,
            "layers": layer_results,
        }
        result_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False),
            encoding="utf-8",
        )
    except BaseException as error:
        error_path.write_text(
            json.dumps(
                {"rank": rank, "error_type": type(error).__name__, "error": str(error)},
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _physical_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    total = page_size * page_count
    return total if total > 0 else None


def run_public_moe_expert_tensor_parallel(
    snapshot: Path,
    *,
    hardware_label: str,
    timeout_seconds: float = 120.0,
    absolute_tolerance: float = 1e-7,
    relative_tolerance: float = 1e-5,
) -> PublicMoeTensorParallelResult:
    """Execute every public expert block with two exact rank-local parameter halves."""

    capability = probe_gloo_capability()
    if capability.status != "available":
        raise CapabilityError(
            capability.reason or "gloo unavailable",
            component="public_moe_tensor_parallel",
        )
    if not hardware_label or hardware_label != hardware_label.strip():
        raise ValueError("hardware_label must be non-empty without boundary whitespace")
    if sensitive_literal_path(hardware_label) is not None:
        raise ValueError("hardware_label must not contain credential-like material")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")
    if (
        not math.isfinite(absolute_tolerance)
        or not math.isfinite(relative_tolerance)
        or absolute_tolerance <= 0.0
        or relative_tolerance <= 0.0
    ):
        raise ValueError("numerical tolerances must be finite and positive")
    source = audit_stories15m_snapshot(snapshot)
    resolved_snapshot = snapshot.expanduser().resolve(strict=True)
    context = multiprocessing.get_context("spawn")
    interface = "lo0" if platform.system() == "Darwin" else "lo"
    with tempfile.TemporaryDirectory(prefix="iql-public-moe-tp-") as directory_name:
        directory = Path(directory_name)
        init_file = directory / "process-group-init"
        result_directory = directory / "results"
        result_directory.mkdir()
        expected_path, hashes_path, reference = _build_reference_files(resolved_snapshot, directory)
        processes = [
            context.Process(
                target=_worker,
                args=(
                    rank,
                    str(init_file),
                    str(result_directory),
                    str(resolved_snapshot),
                    str(expected_path),
                    str(hashes_path),
                    interface,
                    timeout_seconds,
                    absolute_tolerance,
                    relative_tolerance,
                ),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        deadline = time.monotonic() + timeout_seconds
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))
        alive = [process for process in processes if process.is_alive()]
        for process in alive:
            process.terminate()
            process.join()
        if alive:
            raise CapabilityError(
                "public-MoE tensor-parallel workers timed out",
                component="public_moe_tensor_parallel",
                details={"timeout_seconds": timeout_seconds},
            )
        exit_codes = [process.exitcode for process in processes]
        if exit_codes != [0, 0]:
            failures = []
            for rank in range(2):
                error_path = result_directory / f"rank-{rank}.error.json"
                if error_path.exists():
                    failures.append(json.loads(error_path.read_text(encoding="utf-8")))
            raise CapabilityError(
                "public-MoE tensor-parallel worker failed",
                component="public_moe_tensor_parallel",
                details={"exit_codes": exit_codes, "worker_failures": failures},
            )
        payloads = [
            json.loads((result_directory / f"rank-{rank}.json").read_text(encoding="utf-8"))
            for rank in range(2)
        ]

    ranks = tuple(
        PublicMoeRankEvidence(
            rank=payload["rank"],
            input_sha256=payload["input_sha256"],
            shard_evidence=tuple(
                ExpertShardEvidence.model_validate(item) for item in payload["shard_evidence"]
            ),
            layer_output_sha256=tuple(item["output_sha256"] for item in payload["layers"]),
        )
        for payload in payloads
    )
    layers: list[PublicMoeLayerEvidence] = []
    for layer in range(_LAYER_COUNT):
        first = payloads[0]["layers"][layer]
        second = payloads[1]["layers"][layer]
        parent = reference["layers"][layer]
        if first["selected_expert_counts"] != second["selected_expert_counts"]:
            raise CapabilityError(
                "public-MoE ranks observed different router selections",
                component="public_moe_tensor_parallel",
            )
        layers.append(
            PublicMoeLayerEvidence(
                layer_id=layer,
                selected_expert_counts=tuple(first["selected_expert_counts"]),
                reference_output_sha256=parent["reference_output_sha256"],
                distributed_output_sha256=first["output_sha256"],
                rank_max_abs_errors=(first["max_abs_error"], second["max_abs_error"]),
                rank_max_relative_errors=(
                    first["max_relative_error"],
                    second["max_relative_error"],
                ),
                rank_max_tolerance_ratios=(
                    first["max_tolerance_ratio"],
                    second["max_tolerance_ratio"],
                ),
            )
        )
    logical_cpu_count = os.cpu_count()
    if logical_cpu_count is None or logical_cpu_count <= 0:
        raise CapabilityError(
            "logical CPU count is unavailable",
            component="public_moe_tensor_parallel",
        )
    git_version = getattr(torch.version, "git_version", None)
    return PublicMoeTensorParallelResult(
        interface=interface,
        source=source,
        input_sha256=reference["input_sha256"],
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        layers=tuple(layers),
        ranks=ranks,
        operator_declared_hardware_label=hardware_label,
        hardware=PublicMoeTensorParallelHardware(
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
            processor=platform.processor(),
            logical_cpu_count=logical_cpu_count,
            physical_memory_bytes=_physical_memory_bytes(),
        ),
        software=PublicMoeTensorParallelSoftware(
            python=platform.python_version(),
            torch=torch.__version__,
            torch_git_version=git_version if isinstance(git_version, str) else None,
            safetensors=safetensors.__version__,
        ),
    )


__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "MODEL_WEIGHT_SHA256",
    "ExpertShardEvidence",
    "PublicMoeLayerEvidence",
    "PublicMoeRankEvidence",
    "PublicMoeSourceEvidence",
    "PublicMoeTensorParallelResult",
    "audit_stories15m_snapshot",
    "run_public_moe_expert_tensor_parallel",
]
