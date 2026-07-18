"""CPU calibration batches and reference expert sensitivity measurement."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, cast

from torch import nn

from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.data import LocalDataset, load_local_dataset
from inkling_quant_lab.models.base import LoadedModel, ModelAdapter, ModelBatch, ModuleInfo
from inkling_quant_lab.models.fixtures import FixedTokenizer
from inkling_quant_lab.quantization.base import CalibrationArtifact
from inkling_quant_lab.quantization.int8 import DynamicInt8Linear, _replace_module


def calibration_dataset(config: ExperimentConfig) -> LocalDataset | None:
    """Load the explicitly configured calibration set, if any."""

    calibration = config.quantization.calibration
    if calibration is None:
        return None
    dataset = load_local_dataset(calibration.dataset, calibration.revision, calibration.split)
    selected = dataset.samples[: calibration.max_samples]
    if calibration.sample_ids:
        requested = set(calibration.sample_ids)
        selected = tuple(sample for sample in selected if sample.sample_id in requested)
        missing = requested.difference(sample.sample_id for sample in selected)
        if missing:
            raise ValueError("calibration sample IDs are missing: " + ", ".join(sorted(missing)))
    return LocalDataset(
        dataset_id=dataset.dataset_id,
        revision=dataset.revision,
        split=dataset.split,
        sha256=dataset.sha256,
        samples=tuple(selected),
    )


def calibration_batches(dataset: LocalDataset, tokenizer: FixedTokenizer) -> tuple[ModelBatch, ...]:
    """Tokenize each calibration sample independently with stable IDs."""

    batches: list[ModelBatch] = []
    for sample in dataset.samples:
        text = sample.values.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"calibration sample {sample.sample_id} requires non-empty text")
        input_ids, attention_mask = tokenizer.batch_encode((text,))
        batches.append(
            ModelBatch(
                sample_ids=(sample.sample_id,),
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )
    return tuple(batches)


def _mean_nll(adapter: ModelAdapter, model: LoadedModel, batches: tuple[ModelBatch, ...]) -> float:
    weighted = 0.0
    token_count = 0
    for batch in batches:
        output = adapter.forward_loss(model, batch)
        for loss, count in zip(output.negative_log_likelihoods, output.token_counts, strict=True):
            weighted += loss * count
            token_count += count
    if token_count <= 0:
        raise ValueError("calibration produced no loss tokens")
    return weighted / token_count


def measure_expert_loss_sensitivity(
    adapter: ModelAdapter,
    model: LoadedModel,
    modules: tuple[ModuleInfo, ...],
    batches: tuple[ModelBatch, ...],
) -> dict[str, dict[str, float | int | str]]:
    """Quantize one expert linear at a time and measure calibration-loss impact."""

    baseline = _mean_nll(adapter, model, batches)
    results: dict[str, dict[str, float | int | str]] = {}
    for module_info in modules:
        if not module_info.is_expert or not module_info.class_name.endswith(".Linear"):
            continue
        candidate_module = copy.deepcopy(cast(nn.Module, model.model))
        source = candidate_module.get_submodule(module_info.name)
        if not isinstance(source, nn.Linear):
            continue
        _replace_module(candidate_module, module_info.name, DynamicInt8Linear(source))
        candidate = LoadedModel(
            model=candidate_module,
            tokenizer=model.tokenizer,
            descriptor=model.descriptor,
            load_time_seconds=0.0,
        )
        candidate_nll = _mean_nll(adapter, candidate, batches)
        impact = max(0.0, candidate_nll - baseline)
        results[module_info.name] = {
            "value": impact,
            "method": "isolated_dynamic_int8_calibration_loss_impact",
            "confidence": 1.0 if len(batches) >= 4 else len(batches) / 4.0,
            "sample_size": len(batches),
        }
    return dict(sorted(results.items()))


def make_calibration_artifact(
    config: ExperimentConfig,
    dataset: LocalDataset,
    statistics: dict[str, float],
) -> CalibrationArtifact:
    """Create a complete calibration artifact with stable content checksum."""

    calibration = config.quantization.calibration
    if calibration is None:
        raise ValueError("calibration artifact requires quantization.calibration")
    sample_ids = tuple(sample.sample_id for sample in dataset.samples)
    canonical: dict[str, Any] = {
        "dataset_sha256": dataset.sha256,
        "sample_ids": sample_ids,
        "statistics": dict(sorted(statistics.items())),
    }
    checksum = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CalibrationArtifact(
        config=calibration,
        sample_ids=sample_ids,
        statistics=dict(sorted(statistics.items())),
        checksum=checksum,
    )
