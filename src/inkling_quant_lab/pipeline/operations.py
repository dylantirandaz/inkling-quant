"""Reconstructible model, quantizer, evaluation, benchmark, and routing operations."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any, cast

import torch

from inkling_quant_lab.benchmarking.energy import select_default_energy_sensor
from inkling_quant_lab.benchmarking.latency import (
    BenchmarkResult,
    BenchmarkWorkloadProvenance,
    TrialObservation,
    run_generation_benchmark,
)
from inkling_quant_lab.bootstrap import register_builtins
from inkling_quant_lab.config import DecodeConfig, ExperimentConfig
from inkling_quant_lab.data import load_local_dataset, select_dataset_samples
from inkling_quant_lab.evaluation.base import (
    EvaluationResult,
    prompt_template_hash,
    render_prompt,
)
from inkling_quant_lab.evaluation.runner import run_evaluations
from inkling_quant_lab.exceptions import CapabilityError
from inkling_quant_lab.models.base import (
    LoadedModel,
    ModelAdapter,
    ModelBatch,
    ModelCapabilities,
    ModelDescriptor,
    ModuleInfo,
)
from inkling_quant_lab.models.fixtures import FixedTokenizer
from inkling_quant_lab.pipeline.calibration import (
    calibration_batches,
    calibration_dataset,
    make_calibration_artifact,
    measure_expert_loss_sensitivity,
)
from inkling_quant_lab.pipeline.routing_capture import (
    capture_routing_dataset,
    expert_usage_by_module,
)
from inkling_quant_lab.quantization.base import (
    CalibrationArtifact,
    PinnedSourceReloadQuantizer,
    QuantizedModel,
    Quantizer,
    SupportReport,
)
from inkling_quant_lab.quantization.policies import (
    ResolvedPrecisionPolicy,
    resolve_precision_policy,
)
from inkling_quant_lab.quantization.reference import safe_model_serialized_size_bytes
from inkling_quant_lab.registry import MODEL_ADAPTERS, QUANTIZERS, RUNTIMES
from inkling_quant_lab.routing.metrics import RoutingComparison, compare_routing
from inkling_quant_lab.routing.traces import RoutingArtifact
from inkling_quant_lab.runtimes.base import RuntimeBackend, RuntimeCapabilities


@dataclass(frozen=True, slots=True)
class Components:
    """Lazily instantiated components selected by a resolved configuration."""

    adapter: ModelAdapter
    runtime: RuntimeBackend
    quantizer: Quantizer


@dataclass(frozen=True, slots=True)
class CapabilityBundle:
    """Pre-load runtime, model, and quantizer capability reports."""

    runtime: RuntimeCapabilities
    model: ModelCapabilities
    quantizer: SupportReport


@dataclass(frozen=True, slots=True)
class StatisticsBundle:
    """Usage/sensitivity inputs with declared measurement provenance."""

    usage: dict[str, float]
    sensitivity: dict[str, float]
    sensitivity_details: dict[str, dict[str, float | int | str]]
    calibration_dataset_sha256: str | None
    calibration_sample_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON representation."""

        return {
            "usage": dict(sorted(self.usage.items())),
            "sensitivity": dict(sorted(self.sensitivity.items())),
            "sensitivity_details": dict(sorted(self.sensitivity_details.items())),
            "calibration_dataset_sha256": self.calibration_dataset_sha256,
            "calibration_sample_ids": list(self.calibration_sample_ids),
        }


def create_components(config: ExperimentConfig) -> Components:
    """Instantiate only the selected lazy registry entries."""

    register_builtins()
    return Components(
        adapter=MODEL_ADAPTERS.create(config.model.adapter),
        runtime=RUNTIMES.create(config.runtime.backend),
        quantizer=QUANTIZERS.create(config.quantization.backend),
    )


def probe_capabilities(config: ExperimentConfig, components: Components) -> CapabilityBundle:
    """Reject unavailable combinations before loading model weights."""

    runtime = components.runtime.probe(config.runtime)
    if not runtime.available or runtime.reasons:
        raise CapabilityError(
            "Selected runtime is unavailable or incompatible: " + "; ".join(runtime.reasons),
            component=config.runtime.backend,
        )
    if config.runtime.device not in runtime.devices:
        raise CapabilityError(
            f"Runtime {config.runtime.backend} does not expose device {config.runtime.device}",
            component=config.runtime.backend,
        )
    if config.runtime.dtype not in runtime.supported_dtypes:
        raise CapabilityError(
            f"Runtime {config.runtime.backend} does not support dtype {config.runtime.dtype}",
            component=config.runtime.backend,
        )
    if config.runtime.sharding is not None and not runtime.supports_sharding:
        raise CapabilityError(
            f"Runtime {config.runtime.backend} does not support sharding",
            component=config.runtime.backend,
        )
    model_capabilities = components.adapter.capabilities(config)
    if config.model.dtype not in model_capabilities.supported_dtypes:
        raise CapabilityError(
            f"Model adapter does not support dtype {config.model.dtype}",
            component=config.model.adapter,
        )
    if config.runtime.device_map not in model_capabilities.supported_device_maps:
        raise CapabilityError(
            f"Model adapter does not support device map {config.runtime.device_map}",
            component=config.model.adapter,
        )
    preliminary = ModelDescriptor(
        model_id=config.model.model_id,
        revision=config.model.revision,
        resolved_class="unresolved",
        architecture="unresolved",
        checksum="0" * 64,
        capabilities=model_capabilities,
    )
    quantizer = components.quantizer.check_support(preliminary, runtime, config.quantization)
    if not quantizer.available or not quantizer.supported:
        raise CapabilityError(
            f"Quantizer {config.quantization.backend} is unavailable or unsupported: "
            + quantizer.message(),
            component=config.quantization.backend,
            remediation=(
                quantizer.remediation
                or (
                    f"Install with `uv sync --extra {quantizer.install_extra}`."
                    if quantizer.install_extra
                    else None
                )
            ),
        )
    return CapabilityBundle(runtime=runtime, model=model_capabilities, quantizer=quantizer)


def load_baseline(config: ExperimentConfig, components: Components) -> LoadedModel:
    """Load a fresh baseline through the selected adapter/runtime."""

    return components.adapter.load(config, components.runtime)


def _routing_batches(config: ExperimentConfig, tokenizer: FixedTokenizer) -> tuple[ModelBatch, ...]:
    dataset = load_local_dataset(
        config.routing.dataset,
        config.routing.dataset_revision,
        config.routing.dataset_split,
    )
    batches: list[ModelBatch] = []
    for sample in dataset.samples:
        text = sample.values.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"routing sample {sample.sample_id} requires non-empty text")
        input_ids, attention_mask = tokenizer.batch_encode((text,))
        batches.append(
            ModelBatch(
                sample_ids=(sample.sample_id,),
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )
    return tuple(batches)


def collect_statistics(
    config: ExperimentConfig,
    components: Components,
    model: LoadedModel,
    inventory: tuple[ModuleInfo, ...],
) -> StatisticsBundle:
    """Collect complete expert usage and isolated loss-impact statistics when requested."""

    usage: dict[str, float] = {}
    sensitivity: dict[str, float] = {}
    details: dict[str, dict[str, float | int | str]] = {}
    dataset = calibration_dataset(config)
    sample_ids: tuple[str, ...] = ()
    dataset_sha256: str | None = None
    if config.quantization.policy.type != "uniform":
        if dataset is None:
            raise ValueError("expert-aware policy requires calibration data")
        tokenizer = cast(FixedTokenizer, model.tokenizer)
        batches = calibration_batches(dataset, tokenizer)
        sample_ids = tuple(batch.sample_ids[0] for batch in batches)
        dataset_sha256 = dataset.sha256
        descriptor = components.adapter.discover_moe(model)
        if descriptor is None:
            raise CapabilityError(
                "Expert-aware precision policy requires a discovered MoE model",
                component="precision_policy",
            )
        aggregate_config = config.model_copy(
            update={"routing": config.routing.model_copy(update={"mode": "aggregate"})}
        )
        artifact = capture_routing_dataset(components.adapter, model, batches, aggregate_config)
        parent_usage = expert_usage_by_module(artifact, descriptor)
        for module in inventory:
            if not module.is_expert:
                continue
            parent = next(
                (
                    name
                    for name in sorted(parent_usage, key=len, reverse=True)
                    if module.name == name or module.name.startswith(f"{name}.")
                ),
                None,
            )
            if parent is None:
                raise ValueError(f"missing routing usage for expert module {module.name}")
            usage[module.name] = float(parent_usage[parent])
        if config.quantization.policy.type in {"sensitivity_tiered", "hybrid"}:
            details = measure_expert_loss_sensitivity(components.adapter, model, inventory, batches)
            sensitivity = {
                name: float(cast(float | int, record["value"])) for name, record in details.items()
            }
    return StatisticsBundle(
        usage=dict(sorted(usage.items())),
        sensitivity=dict(sorted(sensitivity.items())),
        sensitivity_details=dict(sorted(details.items())),
        calibration_dataset_sha256=dataset_sha256,
        calibration_sample_ids=sample_ids,
    )


def resolve_policy(
    config: ExperimentConfig,
    components: Components,
    model: LoadedModel,
    inventory: tuple[ModuleInfo, ...],
    statistics: StatisticsBundle,
) -> ResolvedPrecisionPolicy:
    """Apply backend precision constraints before deterministic policy resolution."""

    runtime = components.runtime.probe(config.runtime)
    support = components.quantizer.check_support(model.descriptor, runtime, config.quantization)
    if not support.supported:
        raise CapabilityError(support.message(), component=config.quantization.backend)
    backend_precisions = set(support.supported_precisions)
    constrained = tuple(
        replace(
            module,
            supported_precisions=tuple(
                precision
                for precision in module.supported_precisions
                if precision in backend_precisions
            ),
        )
        for module in inventory
    )
    return resolve_precision_policy(
        constrained,
        config.quantization.policy,
        usage_statistics=statistics.usage,
        sensitivity_statistics=statistics.sensitivity,
    )


def build_candidate(
    config: ExperimentConfig,
    components: Components,
    baseline: LoadedModel,
    policy: ResolvedPrecisionPolicy,
    statistics: StatisticsBundle,
) -> QuantizedModel:
    """Reconstruct a candidate and include calibration provenance where applicable."""

    dataset = calibration_dataset(config)
    calibration: CalibrationArtifact | None = None
    samples: tuple[ModelBatch, ...] = ()
    if dataset is not None:
        tokenizer = cast(FixedTokenizer, baseline.tokenizer)
        samples = calibration_batches(dataset, tokenizer)
        combined_statistics = {**statistics.usage, **statistics.sensitivity}
        calibration = make_calibration_artifact(config, dataset, combined_statistics)
    backend_calibration = components.quantizer.calibrate(baseline, samples, config.quantization)
    calibration = backend_calibration or calibration
    if isinstance(components.quantizer, PinnedSourceReloadQuantizer):
        return components.quantizer.quantize_from_pinned_source(
            baseline, policy, calibration, config.quantization
        )
    return components.quantizer.quantize(baseline, policy, calibration, config.quantization)


def evaluate_model(
    config: ExperimentConfig, components: Components, model: LoadedModel
) -> tuple[EvaluationResult, ...]:
    """Run all configured evaluators."""

    return run_evaluations(components.adapter, model, config)


def _benchmark_workload(
    config: ExperimentConfig, model: LoadedModel
) -> tuple[tuple[ModelBatch, ...], DecodeConfig, BenchmarkWorkloadProvenance]:
    """Resolve the exact sequential sample workload used for every benchmark trial."""

    generation_suite = next(
        (
            suite
            for suite in config.evaluation.suites
            if suite.type in {"generation_regression", "exact_match", "behavioral_retention"}
        ),
        config.evaluation.suites[0],
    )
    dataset = load_local_dataset(
        generation_suite.dataset, generation_suite.revision, generation_suite.split
    )
    dataset = select_dataset_samples(dataset, generation_suite.sample_ids)
    tokenizer = cast(FixedTokenizer, model.tokenizer)
    batches: list[ModelBatch] = []
    for sample in dataset.samples:
        text = sample.values.get("text")
        if not isinstance(text, str):
            raise ValueError(f"benchmark sample {sample.sample_id} requires text")
        prompt = render_prompt(generation_suite.prompt_template, text)
        input_ids, attention_mask = tokenizer.batch_encode((prompt,))
        batches.append(
            ModelBatch(
                sample_ids=(sample.sample_id,),
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        )
    provenance = BenchmarkWorkloadProvenance(
        dataset_id=dataset.dataset_id,
        dataset_revision=dataset.revision,
        dataset_sha256=dataset.sha256,
        split=dataset.split,
        sample_ids=dataset.sample_ids,
        seed=config.seed,
        prompt_template_hash=prompt_template_hash(generation_suite.prompt_template),
        decode_config=generation_suite.decode.model_dump(mode="json"),
    )
    return tuple(batches), generation_suite.decode, provenance


def benchmark_model(
    config: ExperimentConfig,
    components: Components,
    model: LoadedModel,
    *,
    serialized_size_bytes: int | None = None,
) -> BenchmarkResult:
    """Measure real per-token fixture generation with warm-up excluded."""

    batches, decode, workload = _benchmark_workload(config, model)

    def trial(adapter: ModelAdapter, loaded: LoadedModel) -> TrialObservation:
        start = time.perf_counter()
        timestamps: list[float] = []
        input_tokens = 0
        for batch in batches:
            current_ids = cast(torch.Tensor, batch.input_ids).clone()
            current_mask = cast(torch.Tensor, batch.attention_mask).clone()
            input_tokens += int(current_mask.sum().item())
            for _ in range(decode.max_new_tokens):
                single = decode.model_copy(update={"max_new_tokens": 1})
                output = adapter.generate(
                    loaded,
                    ModelBatch(
                        sample_ids=batch.sample_ids,
                        input_ids=current_ids,
                        attention_mask=current_mask,
                    ),
                    single,
                )
                if output.sample_ids != batch.sample_ids or len(output.token_ids) != 1:
                    raise ValueError("benchmark adapter returned mismatched sample identity")
                if len(output.token_ids[0]) != 1:
                    raise ValueError("benchmark adapter must return exactly one generated token")
                token = output.token_ids[0][0]
                current_ids = torch.cat(
                    (current_ids, torch.tensor(((token,),), dtype=current_ids.dtype)), dim=1
                )
                current_mask = torch.cat(
                    (current_mask, torch.ones((1, 1), dtype=current_mask.dtype)), dim=1
                )
                timestamps.append(time.perf_counter())
        first = timestamps[0] - start
        intervals = tuple(right - left for left, right in pairwise(timestamps))
        return TrialObservation(
            input_tokens=input_tokens,
            output_tokens=len(timestamps),
            time_to_first_token_seconds=first,
            inter_token_latencies_seconds=intervals,
        )

    size = (
        safe_model_serialized_size_bytes(model)
        if serialized_size_bytes is None
        else serialized_size_bytes
    )
    energy_selection = (
        select_default_energy_sensor(runtime_backend=config.runtime.backend)
        if config.benchmark.measure_energy
        else None
    )
    return run_generation_benchmark(
        components.adapter,
        model,
        components.runtime,
        config.benchmark,
        trial,
        serialized_size_bytes=size,
        workload=workload,
        energy_sensor=energy_selection.sensor if energy_selection is not None else None,
        energy_unavailable_reason=(
            energy_selection.capability.reason if energy_selection is not None else None
        ),
    )


def compare_model_routing(
    config: ExperimentConfig,
    components: Components,
    baseline: LoadedModel,
    candidate: LoadedModel,
) -> tuple[RoutingArtifact, RoutingArtifact, RoutingComparison]:
    """Capture routes in the configured storage mode and compute evidenced drift metrics."""

    tokenizer = cast(FixedTokenizer, baseline.tokenizer)
    batches = _routing_batches(config, tokenizer)
    baseline_artifact = capture_routing_dataset(
        components.adapter,
        baseline,
        batches,
        config,
    )
    candidate_artifact = capture_routing_dataset(
        components.adapter,
        candidate,
        batches,
        config,
    )
    return (
        baseline_artifact,
        candidate_artifact,
        compare_routing(baseline_artifact, candidate_artifact),
    )
