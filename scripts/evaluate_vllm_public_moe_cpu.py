#!/usr/bin/env python3
"""Run a pinned public Mixtral through patched vLLM on Apple CPU, offline.

The parent process audits the immutable model/data inputs, exact vLLM source
patch, and rebuilt native extensions before starting a worker in the isolated
vLLM environment. The worker compares repeated greedy vLLM generations with a
native Transformers reference in memory. Prompt text, token IDs, generated
text, and absolute local paths are omitted from the immutable output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODEL_ID = "ggml-org/stories15M_MOE"
MODEL_REVISION = "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
MODEL_LICENSE = "MIT"
MODEL_CONFIG_SHA256 = "e901e012953d1df93574b2cc3d7db5ed4758d52f8bd4a7dd4b647936e32261be"
MODEL_TOKENIZER_CONFIG_SHA256 = "33d29c87e41f7dd1efb0434d852730320c82970f292be452d820539bce417052"
MODEL_TOKENIZER_SHA256 = "8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1"
MODEL_WEIGHT_SHA256 = "dbfa0289f68a8dd721d10eb12d8bd82e098455682027f6f9986ba548913f9082"
MODEL_WEIGHT_SIZE_BYTES = 72_744_704
MODEL_TENSOR_COUNT = 117

DATASET_ID = "hf://datasets/roneneldan/TinyStories/TinyStories-valid.txt"
DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
DATASET_SPLIT = "validation"
DATASET_SHA256 = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
STORY_SEPARATOR = "<|endoftext|>"
SAMPLE_ID = "story-000001"
PROMPT_TOKEN_COUNT = 32
GENERATED_TOKEN_COUNT = 4
REPETITIONS = 3
SEED = 271_828

VLLM_TAG = "v0.23.0"
VLLM_COMMIT = "0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665"
PATCH_RELATIVE_PATH = Path("scripts/patches/vllm-0.23.0-apple-cpu-stories15m.patch")
PATCH_SHA256 = "3ef5849f3f2f6d63fa10c2aacf224bf755fd87392771531bba82ffeff9fc4a2e"
VLLM_EXTENSION_SHA256 = "a0fa2177581f2a64a32fd0a095d5245f5551a4f5d6639181af2bbec75ced239c"
VLLM_SPINLOOP_SHA256 = "366fbae0d889893075c0216f95f3ad7c0cba5709d86b0d34c1f8034bba2bed06"

VLLM_SOURCE_HASHES = {
    "csrc/cpu/generate_cpu_attn_dispatch.py": (
        "e16671a1d24d42e4f4b9a8360e4e9299727d807e56299df07a44730a6fb188d9",
        "bfca86c453c981656213cd26f4ba3f431944431e7136c9e88230105ac7d826c1",
    ),
    "vllm/model_executor/layers/fused_moe/cpu_fused_moe.py": (
        "89b18a221302cdc705636b5c0647c55a4c0635cd2daaa2d23f2bd96c002c26ef",
        "bee528904d4f742830644244781115d7cd8363413f897b382a962ecc4cccc92f",
    ),
    "vllm/v1/attention/backends/cpu_attn.py": (
        "824ed83e61a2683dd1675fab62739f5083423ade73bc6d3d3a76c9b59aff9fc8",
        "0e78e096885c5674fab6aed52798e3c22a96a114aa3231046e465fdd3d26d075",
    ),
    "vllm/v1/sample/ops/topk_topp_sampler.py": (
        "d0edfc9a1322afadf04a7819fc360d3f8a66cfb44f77d7f190e9d257d24c416f",
        "fe10d51b04f1eb91b951a81179a4990d5d4e53093d166097a4ff5cc534c1a2c1",
    ),
}

EXPECTED_RUNTIME_VERSIONS = {
    "huggingface-hub": "1.23.0",
    "numpy": "2.3.5",
    "psutil": "7.2.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.11.0",
    "transformers": "5.14.0",
    "vllm": "0.23.1.dev0+g0fc695fc6.d20260716.cpu",
}

_MODEL_FILE_FACTS: dict[str, tuple[int, str]] = {
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
    "tokenizer.json": (1_842_764, MODEL_TOKENIZER_SHA256),
    "tokenizer_config.json": (686, MODEL_TOKENIZER_CONFIG_SHA256),
}
_MODEL_FILES = frozenset(_MODEL_FILE_FACTS)
_SAFE_MODEL_SUFFIXES = frozenset({".json", ".safetensors", ".txt"})
_UNSAFE_SUFFIXES = frozenset(
    {".bin", ".ckpt", ".dylib", ".exe", ".joblib", ".pkl", ".pickle", ".pt", ".pth", ".py", ".so"}
)
_SENSITIVE = re.compile(
    r"(?i)(authorization|bearer\s|password|secret|token\s*[=:]|hf_[a-z0-9]{8,}|sk-[a-z0-9])"
)
_WORKER_SENTINEL = "IQL_VLLM_WORKER_RESULT="


def file_sha256(path: Path) -> str:
    """Return a streaming SHA-256 for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    """Hash JSON with a stable representation and finite-number enforcement."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Hash token IDs without retaining the IDs."""

    return canonical_sha256([int(token_id) for token_id in token_ids])


def validate_hardware_label(value: str) -> str:
    """Require a non-secret, explicit hardware description."""

    if value != value.strip() or not value:
        raise ValueError("hardware label must be nonempty without surrounding whitespace")
    if len(value) > 200 or _SENSITIVE.search(value):
        raise ValueError("hardware label is too long or resembles a credential")
    return value


def validate_runtime_python(value: Path) -> Path:
    """Validate an executable while preserving its virtual-environment symlink."""

    candidate = Path(os.path.abspath(value.expanduser()))
    target = candidate.resolve(strict=True)
    if not target.is_file() or not os.access(target, os.X_OK):
        raise ValueError("runtime Python must resolve to one executable file")
    return candidate


def split_stories(text: str) -> tuple[str, ...]:
    """Split the pinned flat TinyStories file into stable, nonempty samples."""

    return tuple(part.strip() for part in text.split(STORY_SEPARATOR) if part.strip())


def select_story(text: str, sample_id: str = SAMPLE_ID) -> str:
    """Resolve one established story-NNNNNN identity."""

    if not re.fullmatch(r"story-[0-9]{6}", sample_id):
        raise ValueError(f"invalid sample ID: {sample_id}")
    stories = split_stories(text)
    index = int(sample_id.removeprefix("story-"))
    if index >= len(stories):
        raise ValueError("sample ID lies outside the pinned dataset")
    return stories[index]


def audit_dataset(path: Path) -> dict[str, Any]:
    """Validate the exact local TinyStories validation file."""

    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError("dataset must be one regular nonsymlink file")
    actual_sha256 = file_sha256(resolved)
    if actual_sha256 != DATASET_SHA256:
        raise ValueError("dataset checksum does not match the pinned validation file")
    text = resolved.read_text(encoding="utf-8")
    story = select_story(text)
    return {
        "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "file_sha256": actual_sha256,
        "file_size_bytes": resolved.stat().st_size,
        "sample_id": SAMPLE_ID,
        "sample_content_sha256": hashlib.sha256(story.encode("utf-8")).hexdigest(),
        "prompt_selection": (
            f"first {PROMPT_TOKEN_COUNT} tokenizer IDs from the selected sample, "
            "including tokenizer-configured special tokens"
        ),
    }


def _audit_model_tree(root: Path) -> list[dict[str, Any]]:
    """Inventory a Hub snapshot while allowing only in-repository blob links."""

    root = root.expanduser().resolve(strict=True)
    repository_cache_root = root.parents[1]
    inventory: list[dict[str, Any]] = []
    for candidate in sorted(root.rglob("*")):
        relative = candidate.relative_to(root).as_posix()
        is_symlink = candidate.is_symlink()
        if is_symlink and candidate.is_dir():
            raise ValueError(f"model snapshot contains a directory symlink: {relative}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"model snapshot contains a non-file entry: {relative}")
        suffix = candidate.suffix.lower()
        if suffix in _UNSAFE_SUFFIXES or suffix not in _SAFE_MODEL_SUFFIXES:
            raise ValueError(f"model snapshot contains an unsafe file: {relative}")
        resolved = candidate.resolve(strict=True)
        if is_symlink and not resolved.is_relative_to(repository_cache_root):
            raise ValueError(f"model snapshot symlink escapes repository cache: {relative}")
        mode = candidate.stat().st_mode
        if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            raise ValueError(f"model snapshot contains an executable file: {relative}")
        inventory.append(
            {
                "path": relative,
                "sha256": file_sha256(candidate),
                "size_bytes": candidate.stat().st_size,
                "symlink": is_symlink,
            }
        )
    return inventory


def audit_model_snapshot(snapshot: Path) -> dict[str, Any]:
    """Fail closed unless the source is the exact safe native Mixtral snapshot."""

    snapshot = snapshot.expanduser().resolve(strict=True)
    if snapshot.name != MODEL_REVISION:
        raise ValueError("model directory name is not the immutable revision")
    inventory = _audit_model_tree(snapshot)
    names = {entry["path"] for entry in inventory}
    if names != _MODEL_FILES:
        raise ValueError(
            "model file set mismatch; "
            f"missing={sorted(_MODEL_FILES - names)}, extra={sorted(names - _MODEL_FILES)}"
        )
    by_name = {entry["path"]: entry for entry in inventory}
    for name, (expected_size, expected_sha256) in _MODEL_FILE_FACTS.items():
        if (
            by_name[name]["size_bytes"] != expected_size
            or by_name[name]["sha256"] != expected_sha256
        ):
            raise ValueError(f"model input size/checksum mismatch: {name}")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["MixtralForCausalLM"]:
        raise ValueError("model architecture must be exactly MixtralForCausalLM")
    if config.get("model_type") != "mixtral":
        raise ValueError("model type must be exactly mixtral")
    if config.get("auto_map") is not None or config.get("model_file") is not None:
        raise ValueError("model config selects custom code")
    if config.get("hidden_size") != 288 or config.get("num_attention_heads") != 6:
        raise ValueError("model hidden/head dimensions differ from the audited contract")
    if config.get("num_hidden_layers") != 6 or config.get("num_local_experts") != 4:
        raise ValueError("model layer/expert dimensions differ from the audited contract")

    from safetensors import safe_open

    with safe_open(str(snapshot / "model.safetensors"), framework="np") as handle:
        tensor_count = len(handle.keys())
        metadata = handle.metadata()
    if tensor_count != MODEL_TENSOR_COUNT or metadata != {"format": "pt"}:
        raise ValueError("model safetensors structure or metadata mismatch")
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "architecture": "MixtralForCausalLM",
        "model_type": "mixtral",
        "hidden_size": 288,
        "attention_heads": 6,
        "attention_head_dimension": 48,
        "layers": 6,
        "experts_per_layer": 4,
        "top_k": 2,
        "weight_file": "model.safetensors",
        "weight_sha256": MODEL_WEIGHT_SHA256,
        "weight_size_bytes": MODEL_WEIGHT_SIZE_BYTES,
        "safetensors_tensor_count": tensor_count,
        "safetensors_metadata": metadata,
        "file_inventory": inventory,
        "trust_remote_code": False,
        "python_file_count": 0,
        "pickle_capable_weight_count": 0,
        "source_location": "immutable_local_huggingface_cache_snapshot",
    }


def _git(source: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        capture_output=True,
    )
    return result.stdout


def audit_vllm_source(source: Path, patch_path: Path) -> dict[str, Any]:
    """Prove the checkout, exact compatibility diff, and native binaries."""

    source = source.expanduser().resolve(strict=True)
    if not (source / ".git").is_dir():
        raise ValueError("vLLM source must be an inspectable Git checkout")
    commit = _git(source, "rev-parse", "HEAD").decode().strip()
    if commit != VLLM_COMMIT:
        raise ValueError("vLLM source commit mismatch")
    patch_bytes = patch_path.read_bytes()
    if hashlib.sha256(patch_bytes).hexdigest() != PATCH_SHA256:
        raise ValueError("checked compatibility patch checksum mismatch")
    paths = tuple(VLLM_SOURCE_HASHES)
    source_diff = _git(source, "diff", "--", *paths)
    if source_diff != patch_bytes:
        raise ValueError("vLLM tracked source diff is not the exact checked patch")
    status = _git(source, "status", "--porcelain", "--untracked-files=no").decode()
    expected_status = {f" M {path}" for path in paths}
    actual_status = {line for line in status.splitlines() if line}
    if actual_status != expected_status:
        raise ValueError("vLLM tracked working tree has changes outside the exact patch")

    source_files: list[dict[str, str]] = []
    for relative, (original_sha256, patched_sha256) in VLLM_SOURCE_HASHES.items():
        actual = file_sha256(source / relative)
        if actual != patched_sha256:
            raise ValueError(f"patched vLLM source checksum mismatch: {relative}")
        source_files.append(
            {
                "path": relative,
                "upstream_sha256": original_sha256,
                "patched_sha256": patched_sha256,
            }
        )
    extension_path = source / "vllm/_C.abi3.so"
    spinloop_path = source / "vllm/spinloop.abi3.so"
    if file_sha256(extension_path) != VLLM_EXTENSION_SHA256:
        raise ValueError("rebuilt vLLM native extension checksum mismatch")
    if file_sha256(spinloop_path) != VLLM_SPINLOOP_SHA256:
        raise ValueError("vLLM spinloop extension checksum mismatch")
    return {
        "repository": "https://github.com/vllm-project/vllm.git",
        "tag": VLLM_TAG,
        "commit": VLLM_COMMIT,
        "patch_file": PATCH_RELATIVE_PATH.as_posix(),
        "patch_sha256": PATCH_SHA256,
        "tracked_diff_exactly_matches_patch": True,
        "patched_source_files": source_files,
        "native_extensions": [
            {
                "path": "vllm/_C.abi3.so",
                "sha256": VLLM_EXTENSION_SHA256,
                "contains_head_dim_48_dispatch": True,
            },
            {
                "path": "vllm/spinloop.abi3.so",
                "sha256": VLLM_SPINLOOP_SHA256,
            },
        ],
        "patch_scopes": [
            (
                "instantiate the existing static native SiLU function without "
                "constructing a CustomOp during forward"
            ),
            (
                "instantiate and advertise the existing VEC16 CPU attention "
                "template for head dimension 48"
            ),
            (
                "select vLLM's exact eager sampler implementation on Darwin ARM "
                "instead of a torch.compile dylib with an unresolved libc++ rpath"
            ),
        ],
    }


def resolve_output_path(project_root: Path, output: Path) -> Path:
    """Require a new JSON output below the repository artifact root."""

    if output.suffix != ".json":
        raise ValueError("output must use the .json suffix")
    if ".." in output.parts:
        raise ValueError("output path cannot contain '..'")
    candidate = output if output.is_absolute() else project_root / output
    candidate = candidate.resolve(strict=False)
    artifact_root = (project_root / "artifacts").resolve(strict=False)
    if not candidate.is_relative_to(artifact_root):
        raise ValueError("output must remain below the project artifacts directory")
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"immutable output already exists: {candidate.name}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def atomic_write_json(output: Path, value: Mapping[str, Any]) -> None:
    """Publish one finite JSON record without replacing existing evidence."""

    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _worker_record(
    snapshot: Path,
    dataset: Path,
    source: Path,
    evaluator_sha256: str,
) -> dict[str, Any]:
    """Execute native Transformers and vLLM in the isolated runtime."""

    script = Path(__file__).resolve(strict=True)
    if file_sha256(script) != evaluator_sha256:
        raise RuntimeError("worker evaluator checksum differs at start")
    for name, expected in EXPECTED_RUNTIME_VERSIONS.items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"runtime package version mismatch: {name}")

    import gc

    import torch
    import vllm  # type: ignore[import-not-found]
    import vllm._C  # type: ignore[import-not-found]
    import vllm.spinloop  # type: ignore[import-not-found]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt  # type: ignore[import-not-found]
    from vllm.model_executor.layers.activation import (  # type: ignore[import-not-found]
        SiluAndMul,
    )
    from vllm.model_executor.layers.fused_moe import (  # type: ignore[import-not-found]
        cpu_fused_moe,
    )
    from vllm.model_executor.layers.fused_moe.activation import (  # type: ignore[import-not-found]
        MoEActivation,
    )
    from vllm.v1.attention.backends.cpu_attn import (  # type: ignore[import-not-found]
        CPUAttentionBackend,
    )
    from vllm.v1.sample.ops.topk_topp_sampler import (  # type: ignore[import-not-found]
        TopKTopPSampler,
    )

    module_path = Path(vllm.__file__).resolve(strict=True)
    if not module_path.is_relative_to(source.resolve(strict=True)):
        raise RuntimeError("worker imported vLLM outside the audited source checkout")
    if 48 not in CPUAttentionBackend.get_supported_head_sizes():
        raise RuntimeError("patched CPU attention backend does not advertise head dimension 48")
    if cpu_fused_moe._CPU_MOE_ACT_FN[MoEActivation.SILU] is not SiluAndMul.forward_native:
        raise RuntimeError("worker did not select the static native SiLU implementation")
    sampler = TopKTopPSampler()
    if sampler.forward.__func__ is not TopKTopPSampler.forward_native:
        raise RuntimeError("Darwin ARM worker did not select the eager native sampler")

    story = select_story(dataset.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
    )
    input_ids = tokenizer(story, add_special_tokens=True).input_ids[:PROMPT_TOKEN_COUNT]
    if len(input_ids) != PROMPT_TOKEN_COUNT:
        raise RuntimeError("selected story does not provide the required prompt token count")
    input_sha256 = token_ids_sha256(input_ids)

    torch.manual_seed(SEED)
    reference: Any = AutoModelForCausalLM.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        dtype=torch.float32,
    )
    reference.eval()
    input_tensor = torch.tensor([input_ids], dtype=torch.long)
    with torch.inference_mode():
        reference_output = reference.generate(
            input_tensor,
            do_sample=False,
            max_new_tokens=GENERATED_TOKEN_COUNT,
            pad_token_id=tokenizer.eos_token_id,
        )
    reference_ids = reference_output[0, len(input_ids) :].tolist()
    if len(reference_ids) != GENERATED_TOKEN_COUNT:
        raise RuntimeError("Transformers reference generated the wrong token count")
    reference_sha256 = token_ids_sha256(reference_ids)
    del reference, reference_output, input_tensor
    gc.collect()

    engine = LLM(
        model=str(snapshot),
        tokenizer=str(snapshot),
        trust_remote_code=False,
        load_format="safetensors",
        dtype="float32",
        tensor_parallel_size=1,
        seed=SEED,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=1,
        kv_cache_memory_bytes=67_108_864,
        enforce_eager=True,
        disable_log_stats=True,
        enable_prefix_caching=False,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=GENERATED_TOKEN_COUNT,
        seed=SEED,
        detokenize=False,
    )
    vllm_hashes: list[str] = []
    vllm_token_arrays: list[list[int]] = []
    finish_reasons: list[str | None] = []
    for _ in range(REPETITIONS):
        output = engine.generate(
            [TokensPrompt(prompt_token_ids=input_ids)],
            params,
            use_tqdm=False,
        )[0].outputs[0]
        generated_ids = list(output.token_ids)
        if len(generated_ids) != GENERATED_TOKEN_COUNT:
            raise RuntimeError("vLLM generated the wrong token count")
        vllm_token_arrays.append(generated_ids)
        vllm_hashes.append(token_ids_sha256(generated_ids))
        finish_reasons.append(output.finish_reason)
    if any(token_array != reference_ids for token_array in vllm_token_arrays):
        raise RuntimeError("repeated vLLM token arrays differ from the Transformers reference")
    if len(set(vllm_hashes)) != 1 or vllm_hashes[0] != reference_sha256:
        raise RuntimeError("repeated vLLM tokens are not deterministic/reference-identical")
    if set(finish_reasons) != {"length"}:
        raise RuntimeError("vLLM completion did not terminate at the declared length")
    architecture = engine.llm_engine.model_config.architectures
    if architecture != ["MixtralForCausalLM"]:
        raise RuntimeError("vLLM resolved an unexpected architecture")

    if file_sha256(script) != evaluator_sha256:
        raise RuntimeError("worker evaluator checksum differs at end")
    return {
        "evaluator_sha256_at_start": evaluator_sha256,
        "evaluator_sha256_at_end": evaluator_sha256,
        "runtime_versions": EXPECTED_RUNTIME_VERSIONS,
        "native_extensions_imported": ["vllm._C", "vllm.spinloop"],
        "runtime_platform": "cpu",
        "resolved_architecture": "MixtralForCausalLM",
        "dtype": "float32",
        "load_format": "safetensors",
        "trust_remote_code": False,
        "tensor_parallel_size": 1,
        "multiprocessing_enabled": False,
        "engine_mode": "V1 in-process UniProcExecutor",
        "distributed_rendezvous": "single-rank local Gloo TCP",
        "http_server_started": False,
        "zmq_transport_started": False,
        "max_model_len": 64,
        "kv_cache_memory_bytes": 67_108_864,
        "enforce_eager": True,
        "prefix_caching": False,
        "prompt_token_count": len(input_ids),
        "prompt_token_ids_sha256": input_sha256,
        "generated_token_count": GENERATED_TOKEN_COUNT,
        "sampling": {
            "method": "greedy",
            "temperature": 0.0,
            "seed": SEED,
            "detokenize": False,
        },
        "repetitions": REPETITIONS,
        "vllm_generated_token_ids_sha256": vllm_hashes,
        "transformers_generated_token_ids_sha256": reference_sha256,
        "all_vllm_repetitions_identical": True,
        "vllm_matches_transformers_exactly": True,
        "finish_reasons": finish_reasons,
        "compatibility_paths_executed": {
            "static_native_silu": True,
            "vec16_head_dimension_48_attention": True,
            "darwin_arm_eager_sampler": True,
        },
    }


def _parse_worker_stdout(stdout: str) -> dict[str, Any]:
    matches = [line for line in stdout.splitlines() if line.startswith(_WORKER_SENTINEL)]
    if len(matches) != 1:
        raise RuntimeError("vLLM worker did not emit exactly one result sentinel")
    value = json.loads(matches[0].removeprefix(_WORKER_SENTINEL))
    if not isinstance(value, dict):
        raise RuntimeError("vLLM worker result is not a JSON object")
    return value


def _validate_public_record(value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, allow_nan=False)
    forbidden = (
        "/Users/",
        "/private/",
        '"prompt_text":',
        '"generated_text":',
        '"prompt_token_ids":',
        '"generated_token_ids":',
        '"output_token_ids":',
    )
    if any(item in encoded for item in forbidden):
        raise ValueError("public evidence contains an absolute path or raw prompt/output payload")
    if _SENSITIVE.search(encoded):
        raise ValueError("public evidence resembles a credential-bearing record")


def run_parent(arguments: argparse.Namespace) -> int:
    """Audit inputs, run the isolated worker, and publish immutable evidence."""

    project_root = Path(__file__).resolve().parents[1]
    script = Path(__file__).resolve(strict=True)
    evaluator_start = file_sha256(script)
    hardware_label = validate_hardware_label(arguments.hardware_label)
    runtime_python = validate_runtime_python(arguments.runtime_python)
    patch_path = (project_root / PATCH_RELATIVE_PATH).resolve(strict=True)
    source_before = audit_vllm_source(arguments.vllm_source, patch_path)
    model = audit_model_snapshot(arguments.snapshot)
    dataset = audit_dataset(arguments.dataset)
    output = resolve_output_path(project_root, arguments.output)

    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "XDG_CACHE_HOME": str(Path(tempfile.gettempdir()) / "iql-vllm-xdg-cache"),
        }
    )
    command = [
        str(runtime_python),
        str(script),
        "--worker",
        "--snapshot",
        str(arguments.snapshot.resolve(strict=True)),
        "--dataset",
        str(arguments.dataset.resolve(strict=True)),
        "--vllm-source",
        str(arguments.vllm_source.resolve(strict=True)),
        "--evaluator-sha256",
        evaluator_start,
    ]
    worker = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=arguments.worker_timeout_seconds,
    )
    if worker.returncode != 0:
        tail = "\n".join((worker.stdout + worker.stderr).splitlines()[-25:])
        raise RuntimeError(f"vLLM worker failed with exit {worker.returncode}:\n{tail}")
    result = _parse_worker_stdout(worker.stdout)
    evaluator_end = file_sha256(script)
    if evaluator_start != evaluator_end:
        raise RuntimeError("evaluator changed during execution")
    source_after = audit_vllm_source(arguments.vllm_source, patch_path)
    if canonical_sha256(source_before) != canonical_sha256(source_after):
        raise RuntimeError("audited vLLM source/build changed during execution")

    record: dict[str, Any] = {
        "schema_version": "external-vllm-public-moe-inference-v2",
        "recorded_at": datetime.now(UTC).isoformat(),
        "status": "success",
        "claims": {
            "pinned_public_moe_weights_loaded": True,
            "vllm_generation_executed": True,
            "repeated_generation_deterministic": True,
            "generated_tokens_match_transformers_reference": True,
            "upstream_vllm_unmodified_supported": False,
            "performance_benchmark_executed": False,
            "routing_capture_executed": False,
            "loss_evaluation_executed": False,
            "quantized_checkpoint_executed": False,
            "project_runtime_promoted": False,
        },
        "evaluator": {
            "file": "scripts/evaluate_vllm_public_moe_cpu.py",
            "sha256_at_start": evaluator_start,
            "sha256_at_end": evaluator_end,
            "worker_attested_start_and_end": True,
        },
        "vllm_source_and_build": source_before,
        "model": model,
        "dataset": dataset,
        "execution": result,
        "host": {
            "hardware_label": hardware_label,
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
        },
        "offline_controls": {
            "model_and_tokenizer_paths_local": True,
            "dataset_path_local": True,
            "hf_hub_offline": True,
            "transformers_offline": True,
            "vllm_usage_stats_disabled": True,
            "remote_code_disabled": True,
            "safetensors_required": True,
            "network_download_attempted": False,
        },
        "privacy": {
            "prompt_text_persisted": False,
            "prompt_token_ids_persisted": False,
            "generated_text_persisted": False,
            "generated_token_ids_persisted": False,
            "hashes_and_counts_only": True,
        },
        "reproduction": {
            "source_patch": "git apply scripts/patches/vllm-0.23.0-apple-cpu-stories15m.patch",
            "build": (
                "VLLM_TARGET_DEVICE=cpu uv pip install --no-build-isolation "
                "--python <isolated-python> -e <vllm-source>"
            ),
            "evaluation": (
                "python scripts/evaluate_vllm_public_moe_cpu.py --runtime-python "
                "<isolated-python> --vllm-source <patched-source> --snapshot "
                "<immutable-snapshot> --dataset <pinned-validation-file> --hardware-label "
                "'<host>' --output artifacts/research-slices/<new-record>.json"
            ),
            "execution_requirement": (
                "the process must be allowed to bind vLLM's single-rank local Gloo TCP rendezvous"
            ),
        },
        "limitations": [
            (
                "This is an external direct-vLLM smoke, not an Inkling Quant Lab "
                "RuntimeBackend or governed pipeline run."
            ),
            (
                "The exact vLLM source requires the recorded compatibility patch; "
                "unmodified v0.23.0 fails on this model/host."
            ),
            (
                "The tiny yujiepan random checkpoint remains unsupported because "
                "its synthetic attention head dimension is one."
            ),
            (
                "Three identical four-token greedy generations prove deterministic "
                "execution for one prompt only; they are not a quality study."
            ),
            (
                "No latency, throughput, memory, energy, routing, loss, quantization, "
                "HTTP serving, concurrency, or multi-device claim is made."
            ),
            (
                "The model publisher repeated expert tensors and randomly initialized "
                "routers, so this does not evidence learned expert specialization."
            ),
            (
                "Although HTTP, ZMQ, and engine multiprocessing were disabled, vLLM "
                "still required one local Gloo TCP rendezvous for world size one."
            ),
            (
                "The native extension checksum is specific to the recorded "
                "Apple/compiler/PyTorch build and is not a universal binary-support "
                "claim."
            ),
        ],
    }
    _validate_public_record(record)
    atomic_write_json(output, record)
    print(
        json.dumps(
            {
                "output": output.relative_to(project_root).as_posix(),
                "record_sha256": file_sha256(output),
                "status": "success",
            },
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the strict parent/worker command surface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-python", type=Path)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hardware-label")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--evaluator-sha256", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Dispatch parent or isolated worker execution."""

    arguments = build_parser().parse_args()
    if arguments.worker:
        if not arguments.evaluator_sha256:
            raise ValueError("worker requires the parent evaluator checksum")
        result = _worker_record(
            arguments.snapshot.resolve(strict=True),
            arguments.dataset.resolve(strict=True),
            arguments.vllm_source.resolve(strict=True),
            arguments.evaluator_sha256,
        )
        print(_WORKER_SENTINEL + json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    if arguments.runtime_python is None:
        raise ValueError("parent requires --runtime-python")
    if arguments.hardware_label is None:
        raise ValueError("parent requires --hardware-label")
    if arguments.output is None:
        raise ValueError("parent requires --output")
    if not math.isfinite(arguments.worker_timeout_seconds) or not (
        30.0 <= arguments.worker_timeout_seconds <= 1800.0
    ):
        raise ValueError("worker timeout must be finite and between 30 and 1800 seconds")
    return run_parent(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
