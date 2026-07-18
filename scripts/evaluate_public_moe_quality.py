#!/usr/bin/env python3
"""Measure pinned public-MoE perplexity on a checksum-verified TinyStories subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from inkling_quant_lab.config import ExperimentConfig, load_config
from inkling_quant_lab.hardware import probe_environment
from inkling_quant_lab.models.base import ModelBatch
from inkling_quant_lab.models.hf_causal_lm import HFCausalLMAdapter
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

_SEPARATOR = "<|endoftext|>"
_SAMPLE_PREFIX = "story-"
_MODEL_ID = "ggml-org/stories15M_MOE"
_MODEL_REVISION = "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
_DATASET_ID = "hf://datasets/roneneldan/TinyStories/TinyStories-valid.txt"
_DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
OFFICIAL_TINYSTORIES_VALID_SHA256 = (
    "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
)


def file_sha256(path: Path) -> str:
    """Hash one dataset file without retaining its contents in artifacts."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_official_dataset_sha256(value: str) -> str:
    """Accept only the pinned official TinyStories validation-file digest."""

    if value != OFFICIAL_TINYSTORIES_VALID_SHA256:
        raise ValueError(
            "public baseline quality evidence requires the official TinyStories validation "
            f"SHA-256 {OFFICIAL_TINYSTORIES_VALID_SHA256}"
        )
    return value


def validate_quality_contract(config: ExperimentConfig) -> None:
    """Require the exact checked model/data/runtime contract this evaluator implements."""

    if len(config.evaluation.suites) != 1 or config.evaluation.suites[0].type != "perplexity":
        raise ValueError("public quality config requires exactly one perplexity suite")
    suite = config.evaluation.suites[0]
    if (config.model.model_id, config.model.revision) != (_MODEL_ID, _MODEL_REVISION):
        raise ValueError("public baseline quality evidence requires the pinned Stories15M model")
    if config.model.trust_remote_code or config.security.allow_remote_code:
        raise ValueError("public baseline quality evidence requires remote code to remain disabled")
    if config.quantization.backend != "noop" or config.quantization.method != "none":
        raise ValueError("public baseline quality evidence requires the no-op quantizer")
    if (suite.dataset, suite.revision, suite.split) != (
        _DATASET_ID,
        _DATASET_REVISION,
        "validation",
    ):
        raise ValueError("public baseline quality evidence requires the pinned TinyStories split")
    if suite.prompt_template != "{text}":
        raise ValueError("raw-story baseline loss requires the exact '{text}' prompt template")
    if (
        config.runtime.backend != "torch_eager_cpu"
        or config.runtime.device != "cpu"
        or config.runtime.dtype != "float32"
        or config.runtime.device_map != "single"
        or config.runtime.sharding is not None
    ):
        raise ValueError("public baseline quality evidence requires unsharded float32 eager CPU")
    if config.security.log_prompts or config.security.log_model_outputs:
        raise ValueError("public quality evidence must not persist prompt or output content")


def split_stories(text: str) -> tuple[str, ...]:
    """Split the official text format and discard only empty separator regions."""

    return tuple(story.strip() for story in text.split(_SEPARATOR) if story.strip())


def sample_index(sample_id: str) -> int:
    """Convert a stable story identifier to its zero-based file segment index."""

    if not sample_id.startswith(_SAMPLE_PREFIX):
        raise ValueError(f"unsupported TinyStories sample ID: {sample_id}")
    suffix = sample_id.removeprefix(_SAMPLE_PREFIX)
    if len(suffix) != 6 or not suffix.isdigit():
        raise ValueError(f"unsupported TinyStories sample ID: {sample_id}")
    return int(suffix)


def select_stories(stories: tuple[str, ...], sample_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Select the exact declared file segments in configuration order."""

    indexes = tuple(sample_index(sample_id) for sample_id in sample_ids)
    if len(set(indexes)) != len(indexes):
        raise ValueError("TinyStories sample IDs must be unique")
    if not indexes or max(indexes) >= len(stories):
        raise ValueError("TinyStories sample selection is empty or outside the dataset file")
    return tuple(stories[index] for index in indexes)


def _selection_sha256(sample_ids: tuple[str, ...], stories: tuple[str, ...]) -> str:
    records = [
        {
            "sample_id": sample_id,
            "content_sha256": hashlib.sha256(story.encode("utf-8")).hexdigest(),
        }
        for sample_id, story in zip(sample_ids, stories, strict=True)
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate(
    config: ExperimentConfig,
    dataset_path: Path,
    *,
    expected_dataset_sha256: str,
    project_root: Path,
) -> dict[str, Any]:
    """Execute one provenance-complete, token-weighted causal-NLL evaluation."""

    expected_dataset_sha256 = require_official_dataset_sha256(expected_dataset_sha256)
    validate_quality_contract(config)
    actual_dataset_sha256 = file_sha256(dataset_path)
    if actual_dataset_sha256 != expected_dataset_sha256:
        raise ValueError(
            "TinyStories file checksum mismatch: "
            f"expected {expected_dataset_sha256}, received {actual_dataset_sha256}"
        )

    suite = config.evaluation.suites[0]
    stories = split_stories(dataset_path.read_text(encoding="utf-8"))
    selected = select_stories(stories, suite.sample_ids)
    runtime = TorchEagerCPURuntime()
    adapter = HFCausalLMAdapter()
    torch.manual_seed(config.seed)
    started = time.perf_counter()
    loaded = adapter.load(config, runtime)
    sample_records: list[dict[str, Any]] = []
    weighted_nll = 0.0
    token_count = 0
    truncated_count = 0
    try:
        maximum = loaded.descriptor.capabilities.max_context_length
        for sample_id, story in zip(suite.sample_ids, selected, strict=True):
            encoded = loaded.tokenizer.encode(story, special_tokens=True)
            truncated = len(encoded) > maximum
            token_ids = encoded[:maximum]
            if len(token_ids) < 2:
                raise ValueError(f"TinyStories sample {sample_id} has fewer than two tokens")
            input_ids = torch.tensor((token_ids,), dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            result = adapter.forward_loss(
                loaded,
                ModelBatch(
                    sample_ids=(sample_id,),
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ),
            )
            count = result.token_counts[0]
            mean_nll = result.negative_log_likelihoods[0]
            weighted_nll += mean_nll * count
            token_count += count
            truncated_count += int(truncated)
            sample_records.append(
                {
                    "sample_id": sample_id,
                    "content_sha256": hashlib.sha256(story.encode("utf-8")).hexdigest(),
                    "source_token_count": len(encoded),
                    "evaluated_token_count": count,
                    "truncated": truncated,
                    "mean_nll": mean_nll,
                }
            )
    finally:
        runtime.cleanup()
    if token_count <= 0:
        raise ValueError("public quality evaluation produced no causal-loss tokens")
    mean_nll = weighted_nll / token_count
    perplexity = math.exp(mean_nll)
    elapsed = time.perf_counter() - started
    return {
        "schema_version": "public-moe-quality-v1",
        "config_hash": config.config_hash(),
        "model": {
            "model_id": loaded.descriptor.model_id,
            "revision": loaded.descriptor.revision,
            "checksum": loaded.descriptor.checksum,
            "architecture": loaded.descriptor.architecture,
            "router_provenance": "randomly_initialized_by_model_publisher",
        },
        "dataset": {
            "dataset_id": suite.dataset,
            "revision": suite.revision,
            "split": suite.split,
            "file": dataset_path.name,
            "file_sha256": actual_dataset_sha256,
            "license": "cdla-sharing-1.0",
            "sample_ids": list(suite.sample_ids),
            "selection_sha256": _selection_sha256(suite.sample_ids, selected),
        },
        "protocol": {
            "metric": "token_weighted_causal_nll_and_perplexity",
            "add_special_tokens": True,
            "truncation": "right_to_model_max_context",
            "max_context_tokens": loaded.descriptor.capabilities.max_context_length,
            "seed": config.seed,
            "prompt_or_output_content_persisted": False,
        },
        "results": {
            "sample_count": len(sample_records),
            "evaluated_token_count": token_count,
            "truncated_sample_count": truncated_count,
            "mean_nll": mean_nll,
            "perplexity": perplexity,
            "elapsed_seconds_including_model_load": elapsed,
            "samples": sample_records,
        },
        "environment": probe_environment(project_root),
        "limitations": [
            "This is one deterministic subset, not a full-dataset or multi-seed quality claim.",
            "The publisher states that expert weights repeat a trained TinyStories model.",
            "The publisher states that router weights were randomly initialized; routing is not "
            "evidence of learned expert specialization.",
            "No quantized candidate participates in this baseline-only quality record.",
        ],
    }


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--expected-dataset-sha256",
        required=True,
        type=require_official_dataset_sha256,
    )
    return parser.parse_args(argv)


def main() -> None:
    """Load checked configuration, evaluate, and emit one JSON document."""

    arguments = _arguments()
    config = load_config(arguments.config)
    record = evaluate(
        config,
        arguments.dataset,
        expected_dataset_sha256=arguments.expected_dataset_sha256,
        project_root=Path.cwd().resolve(),
    )
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
