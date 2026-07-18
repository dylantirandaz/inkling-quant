#!/usr/bin/env python3
"""Opt-in ten-update Metal preflight for the exact Stories15M routers.

The module is deliberately import-safe: checkpoint, corpus, and package audits
all finish before :func:`_run_metal` imports MLX or MLX-LM.  Persisted evidence
contains hashes and aggregate metrics, never corpus text or token IDs.
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
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import torch
import yaml
from safetensors import safe_open
from safetensors.numpy import save_file

from inkling_quant_lab.mlx_contract import (
    EXPECTED_MLX_VERSIONS,
    MLX_MODEL_CLASS,
    MODEL_ARCHITECTURE,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_WEIGHT_SHA256,
    STORIES15M_SOURCE_CONTRACT,
    audit_source_snapshot,
    expected_float32_only_names,
    expected_quantized_leaf_names,
    mlx_environment_status,
    sha256_file,
)
from inkling_quant_lab.post_training import (
    CorpusCollectionContract,
    CorpusContract,
    CorpusProvenance,
    CorpusSampleContract,
    CorpusSourceContract,
    ParentModelIdentity,
    RouterOverlayLineage,
    TrainingRunProvenance,
    TrainingSourceProvenance,
    build_router_overlay_lineage,
    state_dict_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NumpyArray: TypeAlias = np.ndarray[Any, np.dtype[Any]]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/post_training/stories15m_router_10step.yaml"
ROUTER_NAMES = tuple(f"model.layers.{layer}.block_sparse_moe.gate.weight" for layer in range(6))
EXPECTED_RUNTIME_PARAMETER_NAMES = frozenset(
    f"{name}.weight" for name in expected_quantized_leaf_names() | expected_float32_only_names()
)
_SOURCE_BUNDLE_PATHS = (
    "pyproject.toml",
    "scripts/run_mlx_router_preflight.py",
    "src/inkling_quant_lab/__init__.py",
    "src/inkling_quant_lab/mlx_contract.py",
    "src/inkling_quant_lab/models/__init__.py",
    "src/inkling_quant_lab/models/mlx_lm_mixtral.py",
    "src/inkling_quant_lab/post_training/__init__.py",
    "src/inkling_quant_lab/post_training/corpora.py",
    "src/inkling_quant_lab/post_training/lineage.py",
    "src/inkling_quant_lab/post_training/quantized_router.py",
    "src/inkling_quant_lab/post_training/router.py",
    "src/inkling_quant_lab/security.py",
    "src/inkling_quant_lab/version.py",
)
_BINARY_ARTIFACT_SUFFIXES = frozenset({".safetensors"})
_CORPUS_SNIPPET_LENGTH = 48

PINNED_CORPORA: dict[str, dict[str, Any]] = {
    "alice": {
        "dataset_id": "project-gutenberg/pg11",
        "revision": "pg11-2025-06-26",
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "provenance_url": "https://www.gutenberg.org/ebooks/11",
        "download_url": "https://www.gutenberg.org/cache/epub/11/pg11.txt",
        "local_path": "/private/tmp/manifoldmix-corpora/alice-pg11.txt",
        "size_bytes": 174_311,
        "sha256": "01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e",
        "start_marker": (
            "*** START OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
        ),
        "end_marker": (
            "*** END OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"
        ),
        "target_expert_pair": [0, 1],
    },
    "sherlock": {
        "dataset_id": "project-gutenberg/pg1661",
        "revision": "pg1661-2023-10-10",
        "title": "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "provenance_url": "https://www.gutenberg.org/ebooks/1661",
        "download_url": "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
        "local_path": "/private/tmp/manifoldmix-corpora/sherlock-pg1661.txt",
        "size_bytes": 607_606,
        "sha256": "922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0",
        "start_marker": (
            "*** START OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***"
        ),
        "end_marker": (
            "*** END OF THE PROJECT GUTENBERG EBOOK THE ADVENTURES OF SHERLOCK HOLMES ***"
        ),
        "target_expert_pair": [2, 3],
    },
}

_TOP_LEVEL_KEYS = {
    "schema_version",
    "name",
    "seed",
    "model",
    "corpora",
    "data",
    "training",
    "acceptance",
    "output_path",
    "claim_boundary",
}
_LICENSE = "Project Gutenberg License; public-domain status varies by jurisdiction"
_OVERLAY_METADATA = {
    "format": "iql_mlx_router_overlay_v1",
    "model_id": MODEL_ID,
    "revision": MODEL_REVISION,
    "parent_weights_sha256": SOURCE_WEIGHT_SHA256,
    "objective_version": "top2-domain-pair-ce-v1",
}


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Ephemeral parsed document; its text must never enter an artifact."""

    sample_id: str
    domain_id: str
    split: str
    text: str
    content_sha256: str
    split_key_sha256: str


@dataclass(frozen=True, slots=True)
class AuditedCorpus:
    """Verified source plus ephemeral document contents."""

    config: Mapping[str, Any]
    source: CorpusSourceContract
    documents: tuple[ParsedDocument, ...]


@dataclass(frozen=True, slots=True)
class TokenWindow:
    """One ephemeral, document-contained fixed-length token sequence."""

    sample_id: str
    domain_id: str
    split: str
    token_ids: tuple[int, ...]


class PreflightAcceptanceFailure(RuntimeError):
    """A measured preflight that published evidence but missed scientific gates."""

    def __init__(self, artifact_path: Path, failed_checks: Sequence[str]) -> None:
        self.artifact_path = artifact_path
        self.failed_checks = tuple(failed_checks)
        super().__init__(
            "learned-router preflight missed predeclared held-out acceptance checks; "
            f"immutable failure evidence: {artifact_path}"
        )


def _canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _source_bundle_record(paths: Sequence[str] | None = None) -> dict[str, Any]:
    """Hash the exact local implementation files used by the preflight."""

    selected_paths = paths
    if selected_paths is None:
        package_sources = tuple(
            path.relative_to(PROJECT_ROOT).as_posix()
            for path in sorted((PROJECT_ROOT / "src/inkling_quant_lab").rglob("*.py"))
        )
        selected_paths = _SOURCE_BUNDLE_PATHS + package_sources
    files: list[dict[str, Any]] = []
    for relative_name in sorted(set(selected_paths)):
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"source bundle path must be repository-relative: {relative_name}")
        candidate = PROJECT_ROOT / relative
        if candidate.is_symlink():
            raise ValueError(f"source bundle entry must not be a symlink: {relative_name}")
        path = candidate.resolve(strict=True)
        if not path.is_relative_to(PROJECT_ROOT) or not path.is_file():
            raise ValueError(
                f"source bundle entry must be a regular repository file: {relative_name}"
            )
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "hash_scheme": "sha256(canonical JSON ordered source-file facts)",
        "files": files,
        "sha256": _canonical_sha256(files),
    }


def _repository_identity() -> dict[str, str | bool | None]:
    """Record Git identity when available while remaining valid in source archives."""

    revision = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if revision.returncode != 0:
        return {"repository_revision": None, "repository_dirty": None}
    commit = revision.stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Git returned a non-canonical repository revision")
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {"repository_revision": commit, "repository_dirty": bool(status.stdout)}


def _base_environment(runtime_versions: Mapping[str, str]) -> dict[str, Any]:
    """Build import-safe host and software facts completed by the Metal worker."""

    return {
        "schema_version": "mlx-router-preflight-environment-v1",
        "software": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "runtime_versions": dict(sorted(runtime_versions.items())),
            "numpy": np.__version__,
            "pydantic": importlib.metadata.version("pydantic"),
            "pyyaml": importlib.metadata.version("PyYAML"),
            "torch": importlib.metadata.version("torch"),
        },
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mac_version": platform.mac_ver()[0],
            "logical_cpu_count": os.cpu_count(),
        },
    }


def _resolved_preflight_config(
    config: Mapping[str, Any], *, snapshot: Path, output: Path
) -> dict[str, Any]:
    """Materialize the exact serializable configuration consumed by training."""

    resolved = json.loads(json.dumps(config))
    resolved["model"]["snapshot_path"] = str(snapshot)
    resolved["output_path"] = str(output)
    return cast(dict[str, Any], resolved)


def _require_equal(observed: Any, expected: Any, field: str) -> None:
    if observed != expected:
        raise ValueError(f"{field} must be exactly {expected!r}, observed {observed!r}")


def load_preflight_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load and fail closed on the one reviewed ten-step configuration."""

    resolved = path.expanduser().resolve(strict=True)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != _TOP_LEVEL_KEYS:
        raise ValueError("preflight configuration has an unexpected top-level schema")
    config = {str(key): value for key, value in raw.items()}
    _require_equal(config["schema_version"], "mlx-router-preflight-v1", "schema_version")
    _require_equal(config["seed"], 20_260_716, "seed")

    model = config["model"]
    if not isinstance(model, dict):
        raise ValueError("model must be a mapping")
    _require_equal(
        set(model),
        {
            "model_id",
            "revision",
            "architecture",
            "snapshot_path",
            "weights_sha256",
            "device",
            "dtype",
            "trust_remote_code",
        },
        "model fields",
    )
    for field, expected in (
        ("model_id", MODEL_ID),
        ("revision", MODEL_REVISION),
        ("architecture", MODEL_ARCHITECTURE),
        ("weights_sha256", SOURCE_WEIGHT_SHA256),
        ("device", "metal"),
        ("dtype", "float32"),
        ("trust_remote_code", False),
    ):
        _require_equal(model[field], expected, f"model.{field}")

    corpora = config["corpora"]
    if not isinstance(corpora, list) or len(corpora) != 2:
        raise ValueError("corpora must contain exactly Alice and Sherlock")
    observed_domains: list[str] = []
    for corpus in corpora:
        if not isinstance(corpus, dict):
            raise ValueError("each corpus must be a mapping")
        domain = corpus.get("domain_id")
        if domain not in PINNED_CORPORA:
            raise ValueError(f"unexpected corpus domain {domain!r}")
        observed_domains.append(domain)
        expected_corpus: Mapping[str, Any] = PINNED_CORPORA[domain]
        _require_equal(
            set(corpus), set(expected_corpus) | {"domain_id", "declared_license"}, domain
        )
        for field, value in expected_corpus.items():
            _require_equal(corpus[field], value, f"corpora.{domain}.{field}")
        _require_equal(corpus["declared_license"], _LICENSE, f"corpora.{domain}.license")
    _require_equal(observed_domains, ["alice", "sherlock"], "corpus order")

    _require_equal(
        config["data"],
        {
            "parser_version": "gutenberg-paragraph-documents-v1",
            "split_method": "sha256-seed-content-modulo-v1",
            "split_modulus": 5,
            "validation_bucket": 0,
            "sequence_length": 64,
            "train_tokens_per_domain": 320,
            "validation_tokens_per_domain": 256,
        },
        "data",
    )
    _require_equal(
        config["training"],
        {
            "optimizer": "adamw",
            "steps": 10,
            "batch_size": 1,
            "learning_rate": 0.01,
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.0,
            "bias_correction": False,
            "objective_name": "domain_pair_soft_target_cross_entropy",
            "objective_version": "top2-domain-pair-ce-v1",
            "expected_router_tensor_count": 6,
            "expected_trainable_parameter_count": 6912,
        },
        "training",
    )
    _require_equal(
        config["acceptance"],
        {
            "minimum_validation_cross_entropy_reduction": 0.05,
            "minimum_validation_exact_top2_pair_accuracy": 0.60,
            "minimum_accuracy_gain_over_source_router": 0.20,
            "minimum_per_domain_cross_entropy_reduction": 0.02,
            "minimum_per_domain_exact_top2_pair_accuracy": 0.50,
            "minimum_per_domain_accuracy_gain_over_source_router": 0.15,
        },
        "acceptance",
    )
    _require_equal(
        config["claim_boundary"],
        {
            "learned_domain_supervised_routing": True,
            "causal_lm_specialization_claimed": False,
            "output_quality_retention_claimed": False,
            "raw_text_or_tokens_persisted": False,
        },
        "claim_boundary",
    )
    resolve_output_path(str(config["output_path"]))
    return config


def resolve_output_path(value: str) -> Path:
    """Constrain immutable preflight artifacts to the repository artifact root."""

    artifact_root = (PROJECT_ROOT / "artifacts").resolve()
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(artifact_root) or resolved == artifact_root:
        raise ValueError("output_path must be a child of the repository artifacts directory")
    return resolved


def parse_gutenberg_documents(
    text: str, corpus: Mapping[str, Any], *, seed: int, data: Mapping[str, Any]
) -> tuple[ParsedDocument, ...]:
    """Parse and hash document-level, deterministic disjoint train/validation splits."""

    start = str(corpus["start_marker"])
    end = str(corpus["end_marker"])
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"{corpus['domain_id']} Gutenberg body markers are not unique")
    if f"Title: {corpus['title']}" not in text or f"Author: {corpus['author']}" not in text:
        raise ValueError(f"{corpus['domain_id']} Gutenberg title/author metadata mismatch")
    if "Project Gutenberg" not in text[: text.index(start)]:
        raise ValueError("corpus header does not establish Project Gutenberg provenance")
    body = text.split(start, 1)[1].split(end, 1)[0]
    blocks = re.split(r"(?:\r?\n[ \t]*){2,}", body)
    domain = str(corpus["domain_id"])
    documents: list[ParsedDocument] = []
    for ordinal, block in enumerate(blocks):
        canonical = "\n".join(line.rstrip() for line in block.strip().splitlines()).strip()
        if len(canonical) < 40:
            continue
        content_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        sample_id = f"{domain}-document-{ordinal:05d}"
        # Repeated content must stay on one side even when it occurs at
        # multiple ordinals, so sample identity cannot enter the split key.
        split_key = hashlib.sha256(f"{seed}\0{content_digest}".encode()).hexdigest()
        bucket = int(split_key[:16], 16) % int(data["split_modulus"])
        split = "validation" if bucket == int(data["validation_bucket"]) else "train"
        documents.append(
            ParsedDocument(
                sample_id=sample_id,
                domain_id=domain,
                split=split,
                text=canonical,
                content_sha256=content_digest,
                split_key_sha256=split_key,
            )
        )
    if not documents or {item.split for item in documents} != {"train", "validation"}:
        raise ValueError(f"{domain} produced no deterministic disjoint document split")
    train_hashes = {item.content_sha256 for item in documents if item.split == "train"}
    validation_hashes = {item.content_sha256 for item in documents if item.split == "validation"}
    if train_hashes & validation_hashes:
        raise ValueError(f"{domain} train/validation document content overlaps")
    return tuple(documents)


def audit_corpus(corpus: Mapping[str, Any], config: Mapping[str, Any]) -> AuditedCorpus:
    """Verify exact local bytes and provenance without importing model runtimes."""

    path = Path(str(corpus["local_path"])).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"corpus path must be a regular non-symlink file: {path}")
    if path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
        raise ValueError(f"corpus source must not be executable: {path}")
    source = CorpusSourceContract(
        dataset_id=str(corpus["dataset_id"]),
        revision=str(corpus["revision"]),
        declared_license=str(corpus["declared_license"]),
        size_bytes=int(corpus["size_bytes"]),
        sha256=str(corpus["sha256"]),
        parser_version=str(config["data"]["parser_version"]),
    )
    text = source.verify_bytes(path.read_bytes())
    documents = parse_gutenberg_documents(
        text,
        corpus,
        seed=int(config["seed"]),
        data=config["data"],
    )
    return AuditedCorpus(config=corpus, source=source, documents=documents)


def validate_global_document_disjointness(corpora: Sequence[AuditedCorpus]) -> None:
    """Reject train/evaluation content overlap across every audited source."""

    train_hashes = {
        document.content_sha256
        for corpus in corpora
        for document in corpus.documents
        if document.split == "train"
    }
    evaluation_hashes = {
        document.content_sha256
        for corpus in corpora
        for document in corpus.documents
        if document.split == "validation"
    }
    overlap = sorted(train_hashes & evaluation_hashes)
    if overlap:
        raise ValueError(
            "cross-source training/evaluation document content overlaps: " + ", ".join(overlap)
        )


def probe_metal_subprocess(timeout_seconds: float = 30.0) -> None:
    """Contain native MLX import failures before the parent creates run state."""

    probe = (
        "import mlx.core as mx; "
        "assert mx.metal.is_available(), 'MLX reports Metal unavailable'; "
        "mx.synchronize()"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "isolated MLX/Metal capability probe failed before training "
            f"(exit code {completed.returncode})"
        )


def prove_identical_expert_tensors(tensors: Mapping[str, NumpyArray]) -> dict[str, Any]:
    """Prove that each layer's four expert triples are byte-identical."""

    layers: list[dict[str, Any]] = []
    projection_map = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    for layer in range(6):
        projection_records: list[dict[str, Any]] = []
        bundle_hashes = [hashlib.sha256() for _ in range(4)]
        for source_name, runtime_name in projection_map.items():
            names = [
                f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{source_name}.weight"
                for expert in range(4)
            ]
            if any(name not in tensors for name in names):
                raise ValueError(f"missing exact expert projection group at layer {layer}")
            arrays = [np.ascontiguousarray(tensors[name]) for name in names]
            payloads = [array.tobytes(order="C") for array in arrays]
            hashes = [hashlib.sha256(payload).hexdigest() for payload in payloads]
            if any(
                array.shape != arrays[0].shape or array.dtype != arrays[0].dtype for array in arrays
            ):
                raise ValueError(f"expert projection metadata differs at layer {layer}")
            if len(set(hashes)) != 1 or any(payload != payloads[0] for payload in payloads[1:]):
                raise ValueError(f"experts are not byte-identical at layer {layer}")
            for expert, digest in enumerate(hashes):
                bundle_hashes[expert].update(runtime_name.encode("ascii"))
                bundle_hashes[expert].update(bytes.fromhex(digest))
            projection_records.append(
                {
                    "projection": runtime_name,
                    "shape": list(arrays[0].shape),
                    "dtype": str(arrays[0].dtype),
                    "shared_sha256": hashes[0],
                    "four_experts_byte_identical": True,
                }
            )
        expert_bundle_hashes = [digest.hexdigest() for digest in bundle_hashes]
        if len(set(expert_bundle_hashes)) != 1:
            raise ValueError(f"expert bundles differ at layer {layer}")
        layers.append(
            {
                "layer": layer,
                "projections": projection_records,
                "shared_expert_bundle_sha256": expert_bundle_hashes[0],
                "four_expert_bundles_byte_identical": True,
            }
        )
    return {
        "proof": "all four expert functions have identical parameter bytes in every layer",
        "layer_count": 6,
        "projection_group_count": 18,
        "expert_count_per_layer": 4,
        "layers": layers,
        "identifiability_conclusion": (
            "with normalized top-2 mixture weights, identical experts make causal-LM output "
            "independent of router choice; causal loss cannot identify these routers"
        ),
        "causal_lm_router_specialization_identifiable": False,
    }


def audit_identical_experts(weight_path: Path) -> dict[str, Any]:
    """Read only the audited safetensors shard and establish identifiability limits."""

    tensors: dict[str, NumpyArray] = {}
    with safe_open(str(weight_path), framework="np") as handle:
        if handle.metadata() != {"format": "pt"}:
            raise ValueError("source model safetensors metadata must be format=pt")
        for layer in range(6):
            gate_name = f"model.layers.{layer}.block_sparse_moe.gate.weight"
            gate = handle.get_tensor(gate_name)
            if gate.shape != (4, 288) or str(gate.dtype) != "float32":
                raise ValueError(f"router tensor facts differ at layer {layer}")
            for expert in range(4):
                for projection in ("w1", "w2", "w3"):
                    name = (
                        f"model.layers.{layer}.block_sparse_moe.experts.{expert}."
                        f"{projection}.weight"
                    )
                    tensors[name] = handle.get_tensor(name)
    return prove_identical_expert_tensors(tensors)


def _normal_parameter_name(name: str) -> str:
    return name.replace(".block_sparse_moe.gate.gate.weight", ".block_sparse_moe.gate.weight")


def _tensor_hash(array: NumpyArray) -> str:
    normalized = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(normalized.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def load_audited_overlay(path: Path) -> dict[str, NumpyArray]:
    """Read the saved six-router payload and fail closed on every file fact."""

    arrays: dict[str, NumpyArray] = {}
    with safe_open(str(path), framework="np") as handle:
        if handle.metadata() != _OVERLAY_METADATA:
            raise ValueError("saved router overlay metadata differs from the exact contract")
        if set(handle.keys()) != set(ROUTER_NAMES):
            raise ValueError("saved router overlay must contain exactly the six routers")
        for name in ROUTER_NAMES:
            array = np.ascontiguousarray(handle.get_tensor(name))
            if array.shape != (4, 288) or str(array.dtype) != "float32":
                raise ValueError(f"saved router overlay tensor facts differ for {name}")
            arrays[name] = array
    return arrays


def _aggregate_hash(facts: Mapping[str, Mapping[str, Any]], names: Sequence[str]) -> str:
    return _canonical_sha256(
        [{"name": name, "sha256": facts[name]["sha256"]} for name in sorted(names)]
    )


def _lineage_corpus_records(
    collection: CorpusCollectionContract,
    audited_corpora: Sequence[AuditedCorpus],
) -> tuple[CorpusProvenance, ...]:
    """Project the selected corpus contract into the typed lineage schema."""

    if len(collection.corpora) != len(audited_corpora):
        raise ValueError("selected corpus collection differs from the audited corpus inventory")
    records: list[CorpusProvenance] = []
    for audited, contract in zip(audited_corpora, collection.corpora, strict=True):
        domain = str(audited.config["domain_id"])
        if contract.source != audited.source:
            raise ValueError(f"selected corpus source differs from the {domain} audit")
        for split in ("train", "validation"):
            samples = tuple(sample for sample in contract.samples if sample.split == split)
            if not samples:
                raise ValueError(f"{domain} selected corpus has no {split} samples")
            sample_facts = [sample.model_dump(mode="json") for sample in samples]
            partition_contract = {
                "source": contract.source.model_dump(mode="json"),
                "samples": sample_facts,
            }
            records.append(
                CorpusProvenance(
                    corpus_id=f"{domain}:{split}",
                    role=split,
                    dataset_id=contract.source.dataset_id,
                    revision=contract.source.revision,
                    declared_license=contract.source.declared_license,
                    source_size_bytes=contract.source.size_bytes,
                    source_sha256=contract.source.sha256,
                    encoding=contract.source.encoding,
                    parser_version=contract.source.parser_version,
                    split=split,
                    content_sha256=_canonical_sha256(partition_contract),
                    ordered_sample_ids_sha256=_canonical_sha256(
                        [sample.sample_id for sample in samples]
                    ),
                    sample_count=len(samples),
                    token_count=sum(sample.token_count for sample in samples),
                )
            )
    return tuple(records)


def _torch_state(
    arrays: Mapping[str, NumpyArray],
) -> dict[str, torch.Tensor]:
    """Create CPU tensor views solely for canonical typed lineage validation."""

    state: dict[str, torch.Tensor] = {}
    for name, array in arrays.items():
        normalized = np.ascontiguousarray(array)
        if not normalized.flags.writeable:
            normalized = normalized.copy()
        state[name] = torch.from_numpy(normalized)
    return state


def _build_typed_lineage(
    *,
    config: Mapping[str, Any],
    collection: CorpusCollectionContract,
    audited_corpora: Sequence[AuditedCorpus],
    parent_arrays: Mapping[str, NumpyArray],
    overlay_arrays: Mapping[str, NumpyArray],
    reloaded_arrays: Mapping[str, NumpyArray],
    overlay_bundle_sha256: str,
    resolved_config_sha256: str,
    source_context: Mapping[str, Any],
    environment_sha256: str,
) -> RouterOverlayLineage:
    """Build and canonical-round-trip the repository's router lineage contract."""

    parent_state = _torch_state(parent_arrays)
    overlay_state = _torch_state(overlay_arrays)
    reloaded_state = _torch_state(reloaded_arrays)
    parent_router = parent_state[ROUTER_NAMES[0]]
    parent = ParentModelIdentity(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        architecture=MODEL_ARCHITECTURE,
        resolved_class=MLX_MODEL_CLASS,
        config_sha256=STORIES15M_SOURCE_CONTRACT.config_sha256,
        tokenizer_sha256=STORIES15M_SOURCE_CONTRACT.tokenizer_sha256,
        weights_sha256=SOURCE_WEIGHT_SHA256,
        state_dict_sha256=state_dict_sha256(parent_state),
        tensor_count=len(parent_state),
        router_tensor_names=ROUTER_NAMES,
        router_shape=tuple(int(value) for value in parent_router.shape),
        router_dtype=str(parent_router.dtype),
    )
    pairs = [
        {
            "domain_id": str(corpus["domain_id"]),
            "target_expert_pair": [int(value) for value in corpus["target_expert_pair"]],
        }
        for corpus in config["corpora"]
    ]
    training = TrainingRunProvenance(
        run_id=f"{config['name']}-{resolved_config_sha256[:16]}",
        seed=int(config["seed"]),
        steps=int(config["training"]["steps"]),
        batch_size=int(config["training"]["batch_size"]),
        sequence_length=int(config["data"]["sequence_length"]),
        optimizer=str(config["training"]["optimizer"]),
        learning_rate=float(config["training"]["learning_rate"]),
        optimizer_betas=tuple(float(value) for value in config["training"]["betas"]),
        optimizer_epsilon=float(config["training"]["epsilon"]),
        weight_decay=float(config["training"]["weight_decay"]),
        bias_correction=bool(config["training"]["bias_correction"]),
        objective_name=str(config["training"]["objective_name"]),
        objective_version=str(config["training"]["objective_version"]),
        training_config_sha256=resolved_config_sha256,
        domain_pair_config_sha256=_canonical_sha256(pairs),
    )
    source = TrainingSourceProvenance(
        repository="local://inkling-quant-lab",
        repository_revision=source_context["repository_revision"],
        repository_dirty=source_context["repository_dirty"],
        entrypoint=str(source_context["entrypoint"]),
        entrypoint_sha256=str(source_context["entrypoint_sha256"]),
        source_bundle_sha256=str(source_context["source_bundle_sha256"]),
        environment_sha256=environment_sha256,
        dependency_lock_sha256=str(source_context["dependency_lock_sha256"]),
        framework="mlx",
        framework_version=EXPECTED_MLX_VERSIONS["mlx"],
    )
    lineage = build_router_overlay_lineage(
        parent=parent,
        corpus_contract_sha256=_canonical_sha256(collection.model_dump(mode="json")),
        corpus=_lineage_corpus_records(collection, audited_corpora),
        training=training,
        source=source,
        parent_state=parent_state,
        overlay_state=overlay_state,
        reloaded_state=reloaded_state,
        overlay_bundle_sha256=overlay_bundle_sha256,
        loader_name="inkling_quant_lab.models.mlx_lm_mixtral.load_exact_source",
        loader_version=f"mlx-lm-{EXPECTED_MLX_VERSIONS['mlx-lm']}",
    )
    validated = RouterOverlayLineage.model_validate_json(lineage.canonical_json())
    if validated != lineage:
        raise ValueError("typed router overlay lineage changed across canonical validation")
    return validated


def assess_held_out_acceptance(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate every predeclared held-out gate without discarding failures."""

    before_domains = before.get("per_domain")
    after_domains = after.get("per_domain")
    if not isinstance(before_domains, Mapping) or not isinstance(after_domains, Mapping):
        raise ValueError("held-out evaluations must contain per-domain measurements")
    if set(before_domains) != set(after_domains) or not before_domains:
        raise ValueError("source and learned held-out domain identities must be aligned")

    checks: list[dict[str, Any]] = []

    def add_check(
        *,
        check_id: str,
        scope: str,
        domain_id: str | None,
        metric: str,
        observed: float,
        threshold_name: str,
    ) -> None:
        threshold = float(thresholds[threshold_name])
        if not math.isfinite(observed) or not math.isfinite(threshold):
            raise ValueError(f"held-out acceptance value is not finite: {check_id}")
        checks.append(
            {
                "check_id": check_id,
                "scope": scope,
                "domain_id": domain_id,
                "metric": metric,
                "observed": observed,
                "threshold_name": threshold_name,
                "threshold": threshold,
                "comparison": "greater_than_or_equal",
                "passed": observed >= threshold,
            }
        )

    loss_reduction = float(before["mean_cross_entropy"]) - float(after["mean_cross_entropy"])
    accuracy_gain = float(after["exact_top2_pair_accuracy"]) - float(
        before["exact_top2_pair_accuracy"]
    )
    add_check(
        check_id="overall.validation_cross_entropy_reduction",
        scope="overall",
        domain_id=None,
        metric="validation_cross_entropy_reduction",
        observed=loss_reduction,
        threshold_name="minimum_validation_cross_entropy_reduction",
    )
    add_check(
        check_id="overall.validation_exact_top2_pair_accuracy",
        scope="overall",
        domain_id=None,
        metric="validation_exact_top2_pair_accuracy",
        observed=float(after["exact_top2_pair_accuracy"]),
        threshold_name="minimum_validation_exact_top2_pair_accuracy",
    )
    add_check(
        check_id="overall.accuracy_gain_over_source_router",
        scope="overall",
        domain_id=None,
        metric="validation_accuracy_gain_over_source_router",
        observed=accuracy_gain,
        threshold_name="minimum_accuracy_gain_over_source_router",
    )

    per_domain: dict[str, dict[str, float]] = {}
    for raw_domain in sorted(after_domains):
        domain = str(raw_domain)
        before_domain = before_domains[raw_domain]
        after_domain = after_domains[raw_domain]
        if not isinstance(before_domain, Mapping) or not isinstance(after_domain, Mapping):
            raise ValueError(f"held-out domain measurement is malformed: {domain}")
        domain_loss_reduction = float(before_domain["mean_cross_entropy"]) - float(
            after_domain["mean_cross_entropy"]
        )
        domain_accuracy = float(after_domain["exact_top2_pair_accuracy"])
        domain_accuracy_gain = domain_accuracy - float(before_domain["exact_top2_pair_accuracy"])
        per_domain[domain] = {
            "validation_cross_entropy_reduction": domain_loss_reduction,
            "validation_exact_top2_pair_accuracy": domain_accuracy,
            "validation_accuracy_gain_over_source_router": domain_accuracy_gain,
        }
        add_check(
            check_id=f"domain.{domain}.validation_cross_entropy_reduction",
            scope="per_domain",
            domain_id=domain,
            metric="validation_cross_entropy_reduction",
            observed=domain_loss_reduction,
            threshold_name="minimum_per_domain_cross_entropy_reduction",
        )
        add_check(
            check_id=f"domain.{domain}.validation_exact_top2_pair_accuracy",
            scope="per_domain",
            domain_id=domain,
            metric="validation_exact_top2_pair_accuracy",
            observed=domain_accuracy,
            threshold_name="minimum_per_domain_exact_top2_pair_accuracy",
        )
        add_check(
            check_id=f"domain.{domain}.accuracy_gain_over_source_router",
            scope="per_domain",
            domain_id=domain,
            metric="validation_accuracy_gain_over_source_router",
            observed=domain_accuracy_gain,
            threshold_name="minimum_per_domain_accuracy_gain_over_source_router",
        )

    failed_checks = [str(check["check_id"]) for check in checks if not check["passed"]]
    return {
        "thresholds": dict(thresholds),
        "validation_cross_entropy_reduction": loss_reduction,
        "validation_exact_top2_pair_accuracy": float(after["exact_top2_pair_accuracy"]),
        "validation_accuracy_gain_over_source_router": accuracy_gain,
        "per_domain": per_domain,
        "checks": checks,
        "failed_checks": failed_checks,
        "passed": not failed_checks,
    }


def _run_metal(
    config: Mapping[str, Any],
    audited_corpora: Sequence[AuditedCorpus],
    overlay_path: Path,
    *,
    resolved_config_sha256: str,
    source_context: Mapping[str, Any],
    base_environment: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute only after the caller has completed every static preflight audit."""

    import mlx.core as mlx_core
    import mlx.nn as mlx_nn
    import mlx.optimizers as mlx_optim
    from mlx.utils import tree_flatten as mlx_tree_flatten

    mx: Any = mlx_core
    nn: Any = mlx_nn
    optim: Any = mlx_optim
    tree_flatten: Any = mlx_tree_flatten

    from inkling_quant_lab.models.mlx_lm_mixtral import load_exact_source

    if not mx.metal.is_available():
        raise RuntimeError("MLX reports that Apple Metal is unavailable")
    device_info = json.loads(json.dumps(mx.device_info(), sort_keys=True))
    if not isinstance(device_info, dict) or not device_info:
        raise RuntimeError("MLX did not expose a concrete hardware device identity")
    environment = json.loads(json.dumps(base_environment, sort_keys=True))
    hardware_identity = {
        "host": environment["host"],
        "accelerator": {
            "metal_available": True,
            "default_device": str(mx.default_device()),
            "mlx_device_info": device_info,
        },
    }
    environment["hardware_identity"] = hardware_identity
    environment_sha256 = _canonical_sha256(environment)
    hardware_identity_sha256 = _canonical_sha256(hardware_identity)
    seed = int(config["seed"])
    mx.random.seed(seed)
    mx.reset_peak_memory()
    snapshot = Path(str(config["model"]["snapshot_path"])).expanduser().resolve(strict=True)

    load_started = time.perf_counter()
    handle, tokenizer, adapter_load_seconds = load_exact_source(snapshot, seed=seed)
    model = handle.module
    mx.synchronize()
    load_seconds = time.perf_counter() - load_started

    sequence_length = int(config["data"]["sequence_length"])

    def select_windows(
        corpus: AuditedCorpus, split: str, token_budget: int, active_tokenizer: Any
    ) -> tuple[tuple[TokenWindow, ...], tuple[CorpusSampleContract, ...]]:
        if token_budget % sequence_length:
            raise ValueError("token budget must be divisible by sequence length")
        needed = token_budget // sequence_length
        windows: list[TokenWindow] = []
        samples: list[CorpusSampleContract] = []
        documents = sorted(
            (item for item in corpus.documents if item.split == split),
            key=lambda item: item.split_key_sha256,
        )
        for document in documents:
            encoded = tuple(active_tokenizer.encode(document.text, special_tokens=False))
            if len(encoded) < sequence_length:
                continue
            token_ids = tuple(int(token) for token in encoded[:sequence_length])
            windows.append(TokenWindow(document.sample_id, document.domain_id, split, token_ids))
            samples.append(
                CorpusSampleContract.from_text(
                    sample_id=document.sample_id,
                    text=document.text,
                    split=split,  # type: ignore[arg-type]
                    labels=(document.domain_id, "domain-supervised-router"),
                    token_count=len(token_ids),
                )
            )
            if len(windows) == needed:
                break
        if len(windows) != needed:
            raise ValueError(
                f"{corpus.config['domain_id']} {split} lacks {needed} document-contained windows"
            )
        if sum(sample.token_count for sample in samples) != token_budget:
            raise ValueError("selected window token counts differ from the configured budget")
        return tuple(windows), tuple(samples)

    train_windows: dict[str, tuple[TokenWindow, ...]] = {}
    validation_windows: dict[str, tuple[TokenWindow, ...]] = {}
    corpus_contracts: list[CorpusContract] = []
    for corpus in audited_corpora:
        domain = str(corpus.config["domain_id"])
        train, train_samples = select_windows(
            corpus,
            "train",
            int(config["data"]["train_tokens_per_domain"]),
            tokenizer,
        )
        validation, validation_samples = select_windows(
            corpus,
            "validation",
            int(config["data"]["validation_tokens_per_domain"]),
            tokenizer,
        )
        train_windows[domain] = train
        validation_windows[domain] = validation
        samples = train_samples + validation_samples
        corpus_contracts.append(
            CorpusContract(
                source=corpus.source,
                samples=samples,
                token_budget=sum(item.token_count for item in samples),
            )
        )
    collection = CorpusCollectionContract(corpora=tuple(corpus_contracts))

    def parameter_state(
        active_model: Any,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, NumpyArray]]:
        mx.eval(active_model.parameters())
        facts: dict[str, dict[str, Any]] = {}
        arrays: dict[str, NumpyArray] = {}
        for raw_name, value in tree_flatten(active_model.parameters()):
            name = _normal_parameter_name(str(raw_name))
            if name in facts:
                raise ValueError(f"normalized parameter name collision: {name}")
            array = np.ascontiguousarray(np.asarray(value))
            arrays[name] = array
            facts[name] = {
                "sha256": _tensor_hash(array),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "numel": int(array.size),
            }
        if set(facts) != EXPECTED_RUNTIME_PARAMETER_NAMES:
            raise ValueError(
                "runtime parameter inventory mismatch; "
                f"missing={sorted(EXPECTED_RUNTIME_PARAMETER_NAMES - set(facts))}, "
                f"extra={sorted(set(facts) - EXPECTED_RUNTIME_PARAMETER_NAMES)}"
            )
        return facts, arrays

    parent_facts, parent_arrays = parameter_state(model)
    captured: list[list[Any]] = [[] for _ in range(6)]

    class CaptureGate(nn.Module):  # type: ignore[misc]
        def __init__(self, gate: Any, layer: int):
            super().__init__()
            self.gate = gate
            self.layer = layer

        def __call__(self, inputs: Any) -> Any:
            logits = self.gate(inputs)
            captured[self.layer].append(logits)
            return logits

    def install_capture(active_model: Any, *, trainable: bool) -> None:
        for layer, decoder in enumerate(active_model.model.layers):
            block = decoder.block_sparse_moe
            block.gate = CaptureGate(block.gate, layer)
        if trainable:
            active_model.freeze()
            for decoder in active_model.model.layers:
                decoder.block_sparse_moe.gate.gate.unfreeze()

    install_capture(model, trainable=True)
    trainable: dict[str, Any] = {
        _normal_parameter_name(str(name)): value
        for name, value in tree_flatten(model.trainable_parameters())
    }
    if tuple(sorted(trainable)) != tuple(sorted(ROUTER_NAMES)):
        raise ValueError(f"trainable inventory is not exactly six routers: {sorted(trainable)}")
    trainable_parameter_count = sum(int(value.size) for value in trainable.values())
    if trainable_parameter_count != 6_912:
        raise ValueError("six router tensors do not contain exactly 6912 parameters")

    pairs = {
        str(corpus["domain_id"]): tuple(int(value) for value in corpus["target_expert_pair"])
        for corpus in config["corpora"]
    }

    def clear_captures() -> None:
        for sink in captured:
            sink.clear()

    def objective(tokens: Any, target: Any) -> Any:
        clear_captures()
        _hidden = model.model(tokens)
        losses = []
        for layer_capture in captured:
            if len(layer_capture) != 1:
                raise ValueError("each exact Mixtral router must be captured once per forward")
            logits = layer_capture[0]
            log_probabilities = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            losses.append(-mx.mean(mx.sum(log_probabilities * target, axis=-1)))
        return mx.mean(mx.stack(losses))

    def evaluate(active_model: Any) -> dict[str, Any]:
        route_digest = hashlib.sha256()
        per_domain: dict[str, Any] = {}
        all_losses: list[float] = []
        total_correct = 0
        total_tokens = 0
        for domain in sorted(validation_windows):
            pair = pairs[domain]
            layer_records: dict[str, Any] = {}
            domain_losses: list[float] = []
            for layer in range(6):
                layer_records[str(layer)] = {
                    "losses": [],
                    "correct": 0,
                    "token_count": 0,
                    "selection_counts": [0, 0, 0, 0],
                }
            for window in validation_windows[domain]:
                clear_captures()
                tokens = mx.array([window.token_ids], dtype=mx.int32)
                hidden = active_model.model(tokens)
                layer_outputs: list[tuple[Any, Any]] = []
                for layer_capture in captured:
                    if len(layer_capture) != 1:
                        raise ValueError("router capture count changed during evaluation")
                    logits = layer_capture[0]
                    log_probabilities = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                    loss = -0.5 * mx.mean(
                        log_probabilities[..., pair[0]] + log_probabilities[..., pair[1]]
                    )
                    indices = mx.argpartition(-logits, kth=1, axis=-1)[..., :2]
                    layer_outputs.append((loss, indices))
                mx.eval(hidden, *(item for pair_items in layer_outputs for item in pair_items))
                for layer, (loss, indices) in enumerate(layer_outputs):
                    loss_value = float(loss.item())
                    if not math.isfinite(loss_value):
                        raise ValueError("evaluation router loss is not finite")
                    routes = np.asarray(indices, dtype=np.int32).reshape(-1, 2)
                    canonical_routes = np.sort(routes, axis=-1)
                    correct = int(
                        np.sum(
                            (canonical_routes[:, 0] == pair[0])
                            & (canonical_routes[:, 1] == pair[1])
                        )
                    )
                    counts = np.bincount(routes.reshape(-1), minlength=4)
                    record = layer_records[str(layer)]
                    record["losses"].append(loss_value)
                    record["correct"] += correct
                    record["token_count"] += int(routes.shape[0])
                    record["selection_counts"] = [
                        left + int(right)
                        for left, right in zip(record["selection_counts"], counts, strict=True)
                    ]
                    route_digest.update(domain.encode("ascii"))
                    route_digest.update(window.sample_id.encode("ascii"))
                    route_digest.update(bytes([layer]))
                    route_digest.update(routes.tobytes(order="C"))
            public_layers: dict[str, Any] = {}
            domain_correct = 0
            domain_token_count = 0
            for layer_key, record in layer_records.items():
                mean_loss = math.fsum(record["losses"]) / len(record["losses"])
                assignments = record["token_count"] * 2
                public_layers[layer_key] = {
                    "mean_cross_entropy": mean_loss,
                    "exact_top2_pair_accuracy": record["correct"] / record["token_count"],
                    "selection_counts": record["selection_counts"],
                    "selection_distribution": [
                        count / assignments for count in record["selection_counts"]
                    ],
                    "token_count": record["token_count"],
                }
                domain_losses.append(mean_loss)
                all_losses.append(mean_loss)
                domain_correct += int(record["correct"])
                domain_token_count += int(record["token_count"])
            total_correct += domain_correct
            total_tokens += domain_token_count
            per_domain[domain] = {
                "target_expert_pair": list(pair),
                "mean_cross_entropy": math.fsum(domain_losses) / len(domain_losses),
                "exact_top2_pair_accuracy": domain_correct / domain_token_count,
                "layers": public_layers,
            }
        return {
            "mean_cross_entropy": math.fsum(all_losses) / len(all_losses),
            "exact_top2_pair_accuracy": total_correct / total_tokens,
            "route_sha256": route_digest.hexdigest(),
            "per_domain": per_domain,
        }

    evaluation_started = time.perf_counter()
    before = evaluate(model)
    mx.synchronize()
    before_seconds = time.perf_counter() - evaluation_started

    optimizer = optim.AdamW(
        learning_rate=float(config["training"]["learning_rate"]),
        betas=[float(value) for value in config["training"]["betas"]],
        eps=float(config["training"]["epsilon"]),
        weight_decay=float(config["training"]["weight_decay"]),
        bias_correction=bool(config["training"]["bias_correction"]),
    )
    loss_and_grad = nn.value_and_grad(model, objective)
    step_records: list[dict[str, Any]] = []
    domains = ("alice", "sherlock")
    training_started = time.perf_counter()
    for step in range(int(config["training"]["steps"])):
        domain = domains[step % len(domains)]
        window = train_windows[domain][step // len(domains)]
        tokens = mx.array([window.token_ids], dtype=mx.int32)
        pair = pairs[domain]
        target_values = [0.0, 0.0, 0.0, 0.0]
        target_values[pair[0]] = 0.5
        target_values[pair[1]] = 0.5
        target = mx.array(target_values, dtype=mx.float32)
        loss, gradients = loss_and_grad(tokens, target)
        flat_gradients: dict[str, Any] = {
            _normal_parameter_name(str(name)): value for name, value in tree_flatten(gradients)
        }
        if set(flat_gradients) != set(ROUTER_NAMES):
            raise ValueError("gradient inventory is not exactly the six routers")
        norm_arrays = {
            name: mx.sqrt(mx.sum(mx.square(gradient))) for name, gradient in flat_gradients.items()
        }
        optimizer.update(model, gradients)
        mx.eval(loss, model.parameters(), optimizer.state, *norm_arrays.values())
        loss_value = float(loss.item())
        norms = {name: float(value.item()) for name, value in norm_arrays.items()}
        if not math.isfinite(loss_value) or any(
            not math.isfinite(value) or value <= 0.0 for value in norms.values()
        ):
            raise ValueError("router training produced a non-finite or zero gradient")
        step_records.append(
            {"step": step + 1, "domain_id": domain, "loss": loss_value, "gradient_l2": norms}
        )
    mx.synchronize()
    training_seconds = time.perf_counter() - training_started
    if int(optimizer.step.item()) != 10:
        raise ValueError("AdamW did not execute exactly ten updates")

    after_started = time.perf_counter()
    after = evaluate(model)
    mx.synchronize()
    after_seconds = time.perf_counter() - after_started
    after_facts, after_arrays = parameter_state(model)
    changed = sorted(
        name for name in parent_facts if parent_facts[name]["sha256"] != after_facts[name]["sha256"]
    )
    if changed != sorted(ROUTER_NAMES):
        raise ValueError(f"exact changed-tensor set is not six routers: {changed}")
    nonrouters = sorted(set(parent_facts) - set(ROUTER_NAMES))
    parent_nonrouter_hash = _aggregate_hash(parent_facts, nonrouters)
    after_nonrouter_hash = _aggregate_hash(after_facts, nonrouters)
    if parent_nonrouter_hash != after_nonrouter_hash:
        raise ValueError("one or more non-router parameter payloads changed")
    acceptance = config["acceptance"]
    acceptance_result = assess_held_out_acceptance(before, after, acceptance)
    corpus_contract = collection.model_dump(mode="json")
    training_contract = {
        "trainable_tensor_names": list(ROUTER_NAMES),
        "trainable_parameter_count": trainable_parameter_count,
        "changed_tensor_names": changed,
        "nonrouter_tensor_count": len(nonrouters),
        "parent_nonrouter_aggregate_sha256": parent_nonrouter_hash,
        "learned_nonrouter_aggregate_sha256": after_nonrouter_hash,
        "resolved_config_sha256": resolved_config_sha256,
    }
    if not acceptance_result["passed"]:
        return {
            "status": "failed_acceptance",
            "failure": {
                "failure_code": "held_out_acceptance_failed",
                "phase": "post_training_held_out_evaluation",
                "message": "ten supervised updates missed one or more predeclared held-out gates",
                "failed_checks": acceptance_result["failed_checks"],
                "overlay_exported": False,
                "lineage_created": False,
                "raw_text_or_token_ids_persisted": False,
            },
            "corpus_contract": corpus_contract,
            "corpus_contract_sha256": _canonical_sha256(corpus_contract),
            "environment": environment,
            "environment_sha256": environment_sha256,
            "hardware_identity_sha256": hardware_identity_sha256,
            "metrics": {
                "before": before,
                "after": after,
                "training_steps": step_records,
                "training_contract": training_contract,
                "timing_seconds": {
                    "load": load_seconds,
                    "adapter_reported_load": adapter_load_seconds,
                    "before_evaluation": before_seconds,
                    "training": training_seconds,
                    "after_evaluation": after_seconds,
                },
                "memory_at_failed_acceptance": {
                    "peak_mlx_bytes": int(mx.get_peak_memory()),
                    "active_mlx_bytes": int(mx.get_active_memory()),
                    "cache_mlx_bytes": int(mx.get_cache_memory()),
                },
                "acceptance": acceptance_result,
            },
        }

    overlay_arrays = {name: after_arrays[name] for name in ROUTER_NAMES}
    part = overlay_path.with_suffix(".safetensors.part")
    save_file(
        overlay_arrays,
        part,
        metadata=_OVERLAY_METADATA,
    )
    os.replace(part, overlay_path)
    persisted_overlay_arrays = load_audited_overlay(overlay_path)
    for name in ROUTER_NAMES:
        if _tensor_hash(persisted_overlay_arrays[name]) != after_facts[name]["sha256"]:
            raise ValueError(f"saved router overlay bytes differ from trained tensor: {name}")
    overlay_bundle_sha256 = sha256_file(overlay_path)

    reload_started = time.perf_counter()
    fresh_handle, fresh_tokenizer, fresh_adapter_load_seconds = load_exact_source(
        snapshot, seed=seed
    )
    fresh_model = fresh_handle.module
    for layer, decoder in enumerate(fresh_model.model.layers):
        name = ROUTER_NAMES[layer]
        decoder.block_sparse_moe.gate.load_weights(
            [("weight", mx.array(persisted_overlay_arrays[name]))],
            strict=True,
        )
    mx.eval(fresh_model.parameters())
    fresh_facts, fresh_arrays = parameter_state(fresh_model)
    for name in ROUTER_NAMES:
        if fresh_facts[name]["sha256"] != after_facts[name]["sha256"]:
            raise ValueError(f"reloaded overlay router hash differs: {name}")
    if _aggregate_hash(fresh_facts, nonrouters) != parent_nonrouter_hash:
        raise ValueError("fresh source plus overlay changed a non-router parameter")
    for corpus in audited_corpora:
        domain = str(corpus.config["domain_id"])
        fresh_train, _ = select_windows(
            corpus,
            "train",
            int(config["data"]["train_tokens_per_domain"]),
            fresh_tokenizer,
        )
        fresh_validation, _ = select_windows(
            corpus,
            "validation",
            int(config["data"]["validation_tokens_per_domain"]),
            fresh_tokenizer,
        )
        if fresh_train != train_windows[domain] or fresh_validation != validation_windows[domain]:
            raise ValueError("fresh tokenizer did not reproduce the ephemeral fixed token windows")
    clear_captures()
    install_capture(fresh_model, trainable=False)
    reloaded = evaluate(fresh_model)
    mx.synchronize()
    reload_seconds = time.perf_counter() - reload_started
    loss_delta = abs(reloaded["mean_cross_entropy"] - after["mean_cross_entropy"])
    if reloaded["route_sha256"] != after["route_sha256"] or loss_delta > 1e-7:
        raise ValueError("fresh source plus overlay did not reproduce held-out routes and loss")

    memory = {
        "peak_mlx_bytes": int(mx.get_peak_memory()),
        "active_mlx_bytes": int(mx.get_active_memory()),
        "cache_mlx_bytes": int(mx.get_cache_memory()),
    }
    lineage = _build_typed_lineage(
        config=config,
        collection=collection,
        audited_corpora=audited_corpora,
        parent_arrays=parent_arrays,
        overlay_arrays=persisted_overlay_arrays,
        reloaded_arrays=fresh_arrays,
        overlay_bundle_sha256=overlay_bundle_sha256,
        resolved_config_sha256=resolved_config_sha256,
        source_context=source_context,
        environment_sha256=environment_sha256,
    )
    return {
        "status": "complete",
        "corpus_contract": corpus_contract,
        "corpus_contract_sha256": _canonical_sha256(corpus_contract),
        "environment": environment,
        "environment_sha256": environment_sha256,
        "hardware_identity_sha256": hardware_identity_sha256,
        "metrics": {
            "before": before,
            "after": after,
            "reloaded": reloaded,
            "training_steps": step_records,
            "training_contract": training_contract,
            "timing_seconds": {
                "load": load_seconds,
                "adapter_reported_load": adapter_load_seconds,
                "before_evaluation": before_seconds,
                "training": training_seconds,
                "after_evaluation": after_seconds,
                "fresh_reload_and_evaluation": reload_seconds,
                "fresh_adapter_reported_load": fresh_adapter_load_seconds,
            },
            "memory": memory,
            "acceptance": acceptance_result,
            "reload_verification": {
                "fresh_source_plus_saved_overlay": True,
                "overlay_loaded_from_saved_safetensors": True,
                "ephemeral_token_windows_identical": True,
                "route_sha256": reloaded["route_sha256"],
                "max_loss_difference": loss_delta,
            },
        },
        "lineage": lineage.model_dump(mode="json"),
        "lineage_sha256": lineage.sha256(),
    }


def _atomic_text(path: Path, text: str) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(text, encoding="utf-8")
    os.replace(part, path)


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_text(path, json.dumps(payload, sort_keys=True, indent=2) + "\n")


def _artifact_file_facts(
    directory: Path, *, exclude: frozenset[str] = frozenset()
) -> dict[str, dict[str, Any]]:
    """Hash every regular artifact below a candidate publication directory."""

    facts: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact publication contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        if relative in exclude:
            continue
        facts[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return facts


def _make_artifact_read_only(directory: Path, *, seal_root: bool = True) -> None:
    """Seal verified files/subdirectories and optionally the rename source root."""

    directories = [directory] if seal_root else []
    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact publication contains a symlink: {path}")
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError(f"artifact publication contains a non-regular entry: {path}")
    for path in files:
        path.chmod(0o444)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555)
    for path in files:
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise RuntimeError(f"failed to make artifact file read-only: {path}")
    for path in directories:
        if stat.S_IMODE(path.stat().st_mode) != 0o555:
            raise RuntimeError(f"failed to make artifact directory read-only: {path}")


def _restore_artifact_permissions_for_cleanup(directory: Path) -> None:
    """Make an unpublished sealed temporary tree removable after failure."""

    if not directory.exists() or directory.is_symlink():
        return
    directory.chmod(0o700)
    paths = sorted(directory.rglob("*"), key=lambda item: len(item.parts))
    for path in paths:
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o700)
        elif path.is_file():
            path.chmod(0o600)


def _publish_read_only_tree(temporary: Path, output: Path) -> None:
    """Atomically publish a verified tree, then seal its rename-sensitive root."""

    _make_artifact_read_only(temporary, seal_root=False)
    os.replace(temporary, output)
    try:
        _make_artifact_read_only(output)
    except BaseException:
        _restore_artifact_permissions_for_cleanup(output)
        shutil.rmtree(output, ignore_errors=True)
        raise


def _assert_no_raw_corpus_text(directory: Path, corpora: Sequence[AuditedCorpus]) -> None:
    """Reject any exact corpus snippet in every generated non-binary artifact."""

    snippets: set[str] = set()
    short_documents: set[str] = set()
    for corpus in corpora:
        for document in corpus.documents:
            if len(document.text) < _CORPUS_SNIPPET_LENGTH:
                short_documents.add(document.text)
                continue
            snippets.update(
                document.text[offset : offset + _CORPUS_SNIPPET_LENGTH]
                for offset in range(len(document.text) - _CORPUS_SNIPPET_LENGTH + 1)
            )
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact publication contains a symlink: {path}")
        if not path.is_file() or path.suffix in _BINARY_ARTIFACT_SUFFIXES:
            continue
        try:
            persisted = path.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"unexpected non-text artifact requires an explicit audit: {path}"
            ) from error
        if any(document in persisted for document in short_documents) or any(
            persisted[offset : offset + _CORPUS_SNIPPET_LENGTH] in snippets
            for offset in range(max(0, len(persisted) - _CORPUS_SNIPPET_LENGTH + 1))
        ):
            raise ValueError(
                "raw corpus fragment reached generated artifact "
                + path.relative_to(directory).as_posix()
            )


def _write_completion_record(
    directory: Path, *, run_id: str, bindings: Mapping[str, str]
) -> dict[str, Any]:
    """Seal every prior artifact, including evidence, in one final record."""

    facts = _artifact_file_facts(directory, exclude=frozenset({"completion.json"}))
    if "evidence.json" not in facts:
        raise ValueError("completion record requires prior evidence.json")
    completion = {
        "schema_version": "mlx-router-preflight-completion-v1",
        "status": "complete",
        "run_id": run_id,
        "bindings": dict(sorted(bindings.items())),
        "file_count": len(facts),
        "files": facts,
        "ledger_scope": (
            "every regular prior artifact; completion.json excluded to avoid self-reference"
        ),
        "completion_claim": "all evidence and semantic bindings verified before atomic publication",
    }
    _atomic_json(directory / "completion.json", completion)
    return completion


def verify_completion_record(directory: Path) -> dict[str, Any]:
    """Recompute the final artifact ledger and reject missing, extra, or changed files."""

    completion_path = directory / "completion.json"
    if completion_path.is_symlink() or not completion_path.is_file():
        raise ValueError("completion.json must be a regular non-symlink file")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "status",
        "run_id",
        "bindings",
        "file_count",
        "files",
        "ledger_scope",
        "completion_claim",
    }
    if not isinstance(completion, dict) or set(completion) != expected_keys:
        raise ValueError("completion record has an unexpected schema")
    if (
        completion["schema_version"] != "mlx-router-preflight-completion-v1"
        or completion["status"] != "complete"
    ):
        raise ValueError("completion record does not declare a valid completed preflight")
    files = completion["files"]
    if not isinstance(files, dict) or completion["file_count"] != len(files):
        raise ValueError("completion file ledger count is invalid")
    for relative, fact in files.items():
        candidate = Path(relative)
        if (
            not isinstance(relative, str)
            or candidate.is_absolute()
            or ".." in candidate.parts
            or not isinstance(fact, dict)
            or set(fact) != {"sha256", "size_bytes"}
            or not isinstance(fact["size_bytes"], int)
            or fact["size_bytes"] < 0
            or not isinstance(fact["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", fact["sha256"]) is None
        ):
            raise ValueError(f"invalid completion file fact for {relative!r}")
    observed = _artifact_file_facts(directory, exclude=frozenset({"completion.json"}))
    if observed != files:
        raise ValueError("completion file ledger differs from the candidate artifact directory")
    if "evidence.json" not in observed:
        raise ValueError("completion file ledger does not bind evidence.json")
    return completion


def _failure_output_path(
    requested_output: Path,
    *,
    resolved_config_sha256: str,
    created_at: datetime | None = None,
) -> Path:
    """Derive a unique sibling so a failed attempt cannot occupy the success path."""

    timestamp = (created_at or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S%fZ")
    name = f"{requested_output.name}.failed-{timestamp}-{resolved_config_sha256[:8]}"
    candidate = requested_output.with_name(name)
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"failed preflight evidence path already exists: {candidate}")
    return candidate


def _write_failure_record(
    directory: Path,
    *,
    run_id: str,
    bindings: Mapping[str, str],
    failure: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal a measured scientific failure without calling the attempt complete."""

    facts = _artifact_file_facts(directory, exclude=frozenset({"failure.json"}))
    if not {"evidence.json", "metrics.json"}.issubset(facts):
        raise ValueError("failure record requires prior evidence.json and metrics.json")
    failed_checks = failure.get("failed_checks")
    if (
        failure.get("failure_code") != "held_out_acceptance_failed"
        or not isinstance(failed_checks, list)
        or not failed_checks
        or any(not isinstance(item, str) or not item for item in failed_checks)
    ):
        raise ValueError("failure record requires explicit held-out acceptance failures")
    record = {
        "schema_version": "mlx-router-preflight-failure-v1",
        "status": "failed",
        "run_id": run_id,
        "failure_code": "held_out_acceptance_failed",
        "failed_checks": failed_checks,
        "bindings": dict(sorted(bindings.items())),
        "file_count": len(facts),
        "files": facts,
        "ledger_scope": (
            "every regular prior artifact; failure.json excluded to avoid self-reference"
        ),
        "claim_boundary": (
            "measured negative result only; no learned-router or retention claim passed"
        ),
    }
    _atomic_json(directory / "failure.json", record)
    return record


def verify_failure_record(directory: Path) -> dict[str, Any]:
    """Recompute the failed-attempt ledger and reject missing, extra, or changed files."""

    path = directory / "failure.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("failure.json must be a regular non-symlink file")
    record = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "status",
        "run_id",
        "failure_code",
        "failed_checks",
        "bindings",
        "file_count",
        "files",
        "ledger_scope",
        "claim_boundary",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValueError("failure record has an unexpected schema")
    if (
        record["schema_version"] != "mlx-router-preflight-failure-v1"
        or record["status"] != "failed"
        or record["failure_code"] != "held_out_acceptance_failed"
    ):
        raise ValueError("failure record does not declare a valid failed preflight")
    if (
        not isinstance(record["failed_checks"], list)
        or not record["failed_checks"]
        or any(not isinstance(item, str) or not item for item in record["failed_checks"])
    ):
        raise ValueError("failure record does not retain explicit failed checks")
    files = record["files"]
    if not isinstance(files, dict) or record["file_count"] != len(files):
        raise ValueError("failure file ledger count is invalid")
    for relative, fact in files.items():
        candidate = Path(relative)
        if (
            not isinstance(relative, str)
            or candidate.is_absolute()
            or ".." in candidate.parts
            or not isinstance(fact, dict)
            or set(fact) != {"sha256", "size_bytes"}
            or not isinstance(fact["size_bytes"], int)
            or fact["size_bytes"] < 0
            or not isinstance(fact["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", fact["sha256"]) is None
        ):
            raise ValueError(f"invalid failure file fact for {relative!r}")
    observed = _artifact_file_facts(directory, exclude=frozenset({"failure.json"}))
    if observed != files:
        raise ValueError("failure file ledger differs from the candidate artifact directory")
    if not {"evidence.json", "metrics.json"}.issubset(observed):
        raise ValueError("failure file ledger does not bind evidence and metrics")
    return record


def _verify_failure_semantic_bindings(directory: Path, failure_record: Mapping[str, Any]) -> None:
    """Prove a failed attempt binds its exact inputs, environment, and observations."""

    bindings = failure_record["bindings"]
    expected_binding_keys = {
        "corpus_contract_sha256",
        "dependency_lock_sha256",
        "environment_sha256",
        "hardware_identity_sha256",
        "resolved_config_sha256",
        "source_bundle_sha256",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_binding_keys:
        raise ValueError("failure reproducibility bindings have an unexpected schema")
    if any(
        not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in bindings.values()
    ):
        raise ValueError("failure reproducibility binding is not a SHA-256 digest")

    resolved_config = yaml.safe_load(
        (directory / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    corpus_payload = json.loads((directory / "corpus_contract.json").read_text(encoding="utf-8"))
    environment = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    evidence = json.loads((directory / "evidence.json").read_text(encoding="utf-8"))
    CorpusCollectionContract.model_validate(corpus_payload)
    observed = {
        "resolved_config_sha256": _canonical_sha256(resolved_config),
        "corpus_contract_sha256": _canonical_sha256(corpus_payload),
        "environment_sha256": _canonical_sha256(environment),
        "hardware_identity_sha256": _canonical_sha256(environment["hardware_identity"]),
        "source_bundle_sha256": evidence["source_bundle"]["sha256"],
        "dependency_lock_sha256": evidence["dependency_lock_sha256"],
    }
    if observed != bindings:
        raise ValueError("persisted failed-attempt hashes differ from failure bindings")
    if evidence.get("status") != "failed" or evidence.get("bindings") != bindings:
        raise ValueError("failed evidence and failure-record bindings differ")
    acceptance = metrics.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("passed") is not False:
        raise ValueError("failed metrics must retain a failed acceptance result")
    if acceptance.get("failed_checks") != failure_record["failed_checks"]:
        raise ValueError("failed metrics and failure record disagree on failed checks")
    forbidden = {"completion.json", "lineage.json", "router_overlay.safetensors"}
    if any((directory / name).exists() for name in forbidden):
        raise ValueError("failed acceptance artifact must not contain success-only outputs")


def _verify_semantic_bindings(directory: Path, completion: Mapping[str, Any]) -> None:
    """Revalidate typed records and all cross-artifact reproducibility hashes."""

    bindings = completion["bindings"]
    expected_binding_keys = {
        "corpus_contract_sha256",
        "dependency_lock_sha256",
        "environment_sha256",
        "hardware_identity_sha256",
        "lineage_sha256",
        "resolved_config_sha256",
        "source_bundle_sha256",
        "typed_lineage_sha256",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_binding_keys:
        raise ValueError("completion reproducibility bindings have an unexpected schema")
    if any(
        not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for digest in bindings.values()
    ):
        raise ValueError("completion reproducibility binding is not a SHA-256 digest")

    resolved_config = yaml.safe_load(
        (directory / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    corpus_payload = json.loads((directory / "corpus_contract.json").read_text(encoding="utf-8"))
    environment = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
    lineage_envelope = json.loads((directory / "lineage.json").read_text(encoding="utf-8"))
    if not isinstance(lineage_envelope, dict) or set(lineage_envelope) != {
        "schema_version",
        "typed_router_overlay_lineage",
        "typed_router_overlay_lineage_sha256",
        "artifact_bindings",
    }:
        raise ValueError("persisted lineage envelope has an unexpected schema")
    if lineage_envelope["schema_version"] != "mlx-router-preflight-lineage-envelope-v1":
        raise ValueError("persisted lineage envelope has an unexpected version")
    lineage = RouterOverlayLineage.model_validate(lineage_envelope["typed_router_overlay_lineage"])
    evidence = json.loads((directory / "evidence.json").read_text(encoding="utf-8"))
    CorpusCollectionContract.model_validate(corpus_payload)

    observed = {
        "resolved_config_sha256": _canonical_sha256(resolved_config),
        "corpus_contract_sha256": _canonical_sha256(corpus_payload),
        "environment_sha256": _canonical_sha256(environment),
        "hardware_identity_sha256": _canonical_sha256(environment["hardware_identity"]),
        "lineage_sha256": _canonical_sha256(lineage_envelope),
        "typed_lineage_sha256": lineage.sha256(),
        "source_bundle_sha256": lineage.source.source_bundle_sha256,
        "dependency_lock_sha256": lineage.source.dependency_lock_sha256,
    }
    if observed != bindings:
        raise ValueError("persisted semantic artifact hashes differ from completion bindings")
    if evidence.get("bindings") != bindings:
        raise ValueError("evidence and completion reproducibility bindings differ")
    artifact_bindings = {
        name: digest
        for name, digest in bindings.items()
        if name not in {"lineage_sha256", "typed_lineage_sha256"}
    }
    if lineage_envelope["artifact_bindings"] != artifact_bindings:
        raise ValueError("lineage envelope is not bound to every input artifact contract")
    if lineage_envelope["typed_router_overlay_lineage_sha256"] != lineage.sha256():
        raise ValueError("lineage envelope typed digest differs from its typed payload")
    source_bundle = evidence.get("source_bundle")
    if (
        not isinstance(source_bundle, dict)
        or source_bundle.get("sha256") != bindings["source_bundle_sha256"]
    ):
        raise ValueError("evidence source bundle differs from the typed source lineage")
    if lineage.training.training_config_sha256 != bindings["resolved_config_sha256"]:
        raise ValueError("typed training lineage is not bound to the resolved configuration")
    if lineage.corpus_contract_sha256 != bindings["corpus_contract_sha256"]:
        raise ValueError("typed lineage is not bound to the exact corpus contract")
    if lineage.source.environment_sha256 != bindings["environment_sha256"]:
        raise ValueError("typed source lineage is not bound to the execution environment")


def _public_corpus_audit(corpus: AuditedCorpus) -> dict[str, Any]:
    return {
        "domain_id": corpus.config["domain_id"],
        "dataset_id": corpus.source.dataset_id,
        "revision": corpus.source.revision,
        "provenance_url": corpus.config["provenance_url"],
        "download_url": corpus.config["download_url"],
        "declared_license": corpus.source.declared_license,
        "local_path": corpus.config["local_path"],
        "size_bytes": corpus.source.size_bytes,
        "sha256": corpus.source.sha256,
        "parser_version": corpus.source.parser_version,
        "document_count": len(corpus.documents),
        "train_document_count": sum(item.split == "train" for item in corpus.documents),
        "validation_document_count": sum(item.split == "validation" for item in corpus.documents),
        "document_contract_sha256": _canonical_sha256(
            [
                {
                    "sample_id": item.sample_id,
                    "split": item.split,
                    "content_sha256": item.content_sha256,
                    "split_key_sha256": item.split_key_sha256,
                }
                for item in corpus.documents
            ]
        ),
        "raw_text_persisted": False,
    }


def _publish_failed_acceptance(
    temporary: Path,
    requested_output: Path,
    result: Mapping[str, Any],
    *,
    resolved_config: Mapping[str, Any],
    resolved_config_sha256: str,
    source_bundle: Mapping[str, Any],
    dependency_lock_sha256: str,
    repository: Mapping[str, Any],
    source_audit: Mapping[str, Any],
    corpora: Sequence[AuditedCorpus],
    identifiability: Mapping[str, Any],
    package_versions: Mapping[str, str],
    package_reasons: Sequence[str],
) -> Path:
    """Atomically retain a threshold failure while leaving the success path free."""

    failure = result.get("failure")
    if result.get("status") != "failed_acceptance" or not isinstance(failure, Mapping):
        raise ValueError("failed publication requires a measured acceptance failure")
    if (temporary / "router_overlay.safetensors").exists():
        raise ValueError("failed acceptance must occur before router overlay export")
    corpus_contract = CorpusCollectionContract.model_validate(result["corpus_contract"])
    corpus_payload = corpus_contract.model_dump(mode="json")
    if _canonical_sha256(corpus_payload) != result["corpus_contract_sha256"]:
        raise ValueError("failed Metal result corpus contract digest differs from its payload")
    bindings = {
        "resolved_config_sha256": resolved_config_sha256,
        "corpus_contract_sha256": str(result["corpus_contract_sha256"]),
        "environment_sha256": str(result["environment_sha256"]),
        "hardware_identity_sha256": str(result["hardware_identity_sha256"]),
        "source_bundle_sha256": str(source_bundle["sha256"]),
        "dependency_lock_sha256": dependency_lock_sha256,
    }
    _atomic_text(
        temporary / "resolved_config.yaml",
        yaml.safe_dump(dict(resolved_config), sort_keys=True),
    )
    _atomic_json(temporary / "corpus_contract.json", corpus_payload)
    _atomic_json(temporary / "environment.json", result["environment"])
    _atomic_json(temporary / "metrics.json", result["metrics"])
    file_facts = _artifact_file_facts(temporary)
    evidence = {
        "schema_version": "mlx-router-preflight-failed-evidence-v1",
        "status": "failed",
        "failure": dict(failure),
        "bindings": bindings,
        "checks": {
            "source_snapshot_audited_before_mlx_import": True,
            "corpus_bytes_and_provenance_audited_before_mlx_import": True,
            "package_matrix_audited_before_mlx_import": True,
            "identical_expert_identifiability_proved": True,
            "source_bundle_unchanged_during_execution": True,
            "exactly_ten_adamw_updates": True,
            "exactly_six_router_tensors_changed": True,
            "all_nonrouter_parameter_bytes_identical": True,
            "held_out_acceptance_passed": False,
            "router_overlay_exported": False,
            "typed_router_overlay_lineage_created": False,
            "raw_text_or_token_ids_persisted": False,
        },
        "configured_claim_boundary": resolved_config["claim_boundary"],
        "effective_claims": {
            "learned_domain_supervised_routing": False,
            "quantized_learned_router_retention": False,
            "causal_lm_specialization": False,
            "output_quality_retention": False,
        },
        "identifiability": identifiability,
        "source_audit": source_audit,
        "corpora": [_public_corpus_audit(corpus) for corpus in corpora],
        "source_bundle": source_bundle,
        "dependency_lock_sha256": dependency_lock_sha256,
        "repository": repository,
        "package_matrix": {
            "versions": dict(package_versions),
            "reasons": list(package_reasons),
        },
        "files_before_evidence_and_failure_seal": file_facts,
    }
    _atomic_json(temporary / "evidence.json", evidence)
    _assert_no_raw_corpus_text(temporary, corpora)
    failure_output = _failure_output_path(
        requested_output, resolved_config_sha256=resolved_config_sha256
    )
    failure_record = _write_failure_record(
        temporary,
        run_id=failure_output.name,
        bindings=bindings,
        failure=failure,
    )
    _assert_no_raw_corpus_text(temporary, corpora)
    persisted = verify_failure_record(temporary)
    if persisted != failure_record:
        raise ValueError("failure record changed across persisted verification")
    _verify_failure_semantic_bindings(temporary, persisted)
    if requested_output.exists():
        raise FileExistsError(f"completed preflight output already exists: {requested_output}")
    if failure_output.exists() or failure_output.is_symlink():
        raise FileExistsError(f"failed preflight output already exists: {failure_output}")
    _publish_read_only_tree(temporary, failure_output)
    return failure_output


def run_preflight(
    config_path: Path = DEFAULT_CONFIG, *, output_override: str | None = None
) -> Path:
    """Audit, execute, and atomically publish one immutable preflight directory."""

    script_path = Path(__file__).resolve(strict=True)
    config = load_preflight_config(config_path)
    if output_override is not None:
        config["output_path"] = output_override
    output = resolve_output_path(str(config["output_path"]))
    if output.exists():
        raise FileExistsError(f"completed preflight output already exists: {output}")

    snapshot = Path(str(config["model"]["snapshot_path"])).expanduser().resolve(strict=True)
    resolved_config = _resolved_preflight_config(config, snapshot=snapshot, output=output)
    resolved_config_sha256 = _canonical_sha256(resolved_config)
    source_bundle = _source_bundle_record()
    dependency_lock_candidate = PROJECT_ROOT / "uv.lock"
    if dependency_lock_candidate.is_symlink():
        raise ValueError("uv.lock must be a regular non-symlink dependency lock")
    dependency_lock = dependency_lock_candidate.resolve(strict=True)
    if not dependency_lock.is_file():
        raise ValueError("uv.lock must be a regular non-symlink dependency lock")
    dependency_lock_sha256 = sha256_file(dependency_lock)
    repository = _repository_identity()
    source_context: dict[str, Any] = {
        **repository,
        "entrypoint": script_path.relative_to(PROJECT_ROOT).as_posix(),
        "entrypoint_sha256": sha256_file(script_path),
        "source_bundle_sha256": source_bundle["sha256"],
        "dependency_lock_sha256": dependency_lock_sha256,
    }
    source_audit = audit_source_snapshot(snapshot)
    corpora = tuple(audit_corpus(item, resolved_config) for item in resolved_config["corpora"])
    validate_global_document_disjointness(corpora)
    status = mlx_environment_status()
    if not status.available:
        raise RuntimeError("exact MLX environment unavailable: " + "; ".join(status.reasons))
    if status.versions != EXPECTED_MLX_VERSIONS:
        raise RuntimeError("installed MLX package matrix differs from the pinned matrix")
    base_environment = _base_environment(status.versions)
    identifiability = audit_identical_experts(snapshot / "model.safetensors")
    probe_metal_subprocess()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        result = _run_metal(
            resolved_config,
            corpora,
            temporary / "router_overlay.safetensors",
            resolved_config_sha256=resolved_config_sha256,
            source_context=source_context,
            base_environment=base_environment,
        )
        if _source_bundle_record() != source_bundle:
            raise RuntimeError("preflight source bundle changed while the run was executing")
        if sha256_file(dependency_lock) != dependency_lock_sha256:
            raise RuntimeError("dependency lock changed while the run was executing")
        if audit_source_snapshot(snapshot) != source_audit:
            raise RuntimeError("source snapshot changed while the run was executing")

        if result.get("status") == "failed_acceptance":
            failure_output = _publish_failed_acceptance(
                temporary,
                output,
                result,
                resolved_config=resolved_config,
                resolved_config_sha256=resolved_config_sha256,
                source_bundle=source_bundle,
                dependency_lock_sha256=dependency_lock_sha256,
                repository=repository,
                source_audit=source_audit,
                corpora=corpora,
                identifiability=identifiability,
                package_versions=status.versions,
                package_reasons=status.reasons,
            )
            failure = result["failure"]
            failed_checks = failure["failed_checks"]
            raise PreflightAcceptanceFailure(failure_output, failed_checks)
        if result.get("status") != "complete":
            raise ValueError("Metal preflight returned an unknown terminal status")

        typed_lineage = RouterOverlayLineage.model_validate(result["lineage"])
        if typed_lineage.sha256() != result["lineage_sha256"]:
            raise ValueError("Metal result typed lineage digest differs from its payload")
        corpus_contract = CorpusCollectionContract.model_validate(result["corpus_contract"])
        corpus_contract_payload = corpus_contract.model_dump(mode="json")
        if _canonical_sha256(corpus_contract_payload) != result["corpus_contract_sha256"]:
            raise ValueError("Metal result corpus contract digest differs from its payload")

        artifact_bindings = {
            "resolved_config_sha256": resolved_config_sha256,
            "corpus_contract_sha256": str(result["corpus_contract_sha256"]),
            "environment_sha256": str(result["environment_sha256"]),
            "hardware_identity_sha256": str(result["hardware_identity_sha256"]),
            "source_bundle_sha256": str(source_bundle["sha256"]),
            "dependency_lock_sha256": dependency_lock_sha256,
        }
        lineage_envelope = {
            "schema_version": "mlx-router-preflight-lineage-envelope-v1",
            "typed_router_overlay_lineage": typed_lineage.model_dump(mode="json"),
            "typed_router_overlay_lineage_sha256": typed_lineage.sha256(),
            "artifact_bindings": artifact_bindings,
        }
        bindings = {
            **artifact_bindings,
            "typed_lineage_sha256": typed_lineage.sha256(),
            "lineage_sha256": _canonical_sha256(lineage_envelope),
        }
        _atomic_text(
            temporary / "resolved_config.yaml",
            yaml.safe_dump(resolved_config, sort_keys=True),
        )
        _atomic_json(temporary / "corpus_contract.json", corpus_contract_payload)
        _atomic_json(temporary / "environment.json", result["environment"])
        _atomic_json(temporary / "lineage.json", lineage_envelope)
        _atomic_json(temporary / "metrics.json", result["metrics"])
        file_facts = _artifact_file_facts(temporary)
        evidence = {
            "schema_version": "mlx-router-preflight-evidence-v1",
            "status": "complete",
            "bindings": bindings,
            "checks": {
                "source_snapshot_audited_before_mlx_import": True,
                "corpus_bytes_and_provenance_audited_before_mlx_import": True,
                "package_matrix_audited_before_mlx_import": True,
                "identical_expert_identifiability_proved": True,
                "typed_router_overlay_lineage_validated": True,
                "source_bundle_unchanged_during_execution": True,
                "exactly_ten_adamw_updates": True,
                "exactly_six_router_tensors_changed": True,
                "all_nonrouter_parameter_bytes_identical": True,
                "fresh_source_plus_overlay_reproduced_routes": True,
                "raw_text_or_token_ids_persisted": False,
            },
            "claim_boundary": resolved_config["claim_boundary"],
            "identifiability": identifiability,
            "source_audit": source_audit,
            "corpora": [_public_corpus_audit(corpus) for corpus in corpora],
            "source_bundle": source_bundle,
            "repository": repository,
            "package_matrix": {
                "versions": status.versions,
                "reasons": list(status.reasons),
            },
            "files": file_facts,
        }
        _atomic_json(temporary / "evidence.json", evidence)
        _assert_no_raw_corpus_text(temporary, corpora)
        completion = _write_completion_record(temporary, run_id=output.name, bindings=bindings)
        _assert_no_raw_corpus_text(temporary, corpora)
        persisted_completion = verify_completion_record(temporary)
        if persisted_completion != completion:
            raise ValueError("completion record changed across persisted verification")
        _verify_semantic_bindings(temporary, persisted_completion)
        if _source_bundle_record() != source_bundle:
            raise RuntimeError("preflight source bundle changed before atomic publication")
        if sha256_file(dependency_lock) != dependency_lock_sha256:
            raise RuntimeError("dependency lock changed before atomic publication")
        if audit_source_snapshot(snapshot) != source_audit:
            raise RuntimeError("source snapshot changed before atomic publication")
        if output.exists():
            raise FileExistsError(f"completed preflight output already exists: {output}")
        _publish_read_only_tree(temporary, output)
    except BaseException:
        _restore_artifact_permissions_for_cleanup(temporary)
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", help="artifact-root-contained output override")
    parser.add_argument(
        "--execute-metal",
        action="store_true",
        help="explicitly authorize this opt-in external Metal preflight",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute_metal:
        raise SystemExit("refusing to import MLX or train without explicit --execute-metal")
    try:
        output = run_preflight(args.config, output_override=args.output)
    except PreflightAcceptanceFailure as error:
        print(error.artifact_path)
        print(str(error), file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
