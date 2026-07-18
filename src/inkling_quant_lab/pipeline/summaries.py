"""Normalize run artifacts for compatibility checks, Pareto analysis, and reports."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

from inkling_quant_lab.benchmarking.latency import BenchmarkResult
from inkling_quant_lab.comparison import (
    BenchmarkMemoryIdentity,
    BenchmarkWorkloadIdentity,
    DatasetIdentity,
    EvaluationSuiteIdentity,
    MetricValue,
    NormalizedRunSummary,
    RoutingCaptureIdentity,
)
from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.data import load_local_dataset
from inkling_quant_lab.evaluation.base import EvaluationResult
from inkling_quant_lab.evaluation.behavioral import retention_score
from inkling_quant_lab.models.base import ModelDescriptor
from inkling_quant_lab.quantization.base import QuantizationManifest
from inkling_quant_lab.quantization.policies import ResolvedPrecisionPolicy
from inkling_quant_lab.routing.metrics import RoutingComparison
from inkling_quant_lab.routing.traces import RoutingArtifact


def _available(
    value: float | int,
    *,
    unit: str | None,
    category: str,
    direction: str,
    evaluation_suite: EvaluationSuiteIdentity | None = None,
) -> MetricValue:
    return MetricValue.model_validate(
        {
            "value": float(value),
            "status": "available",
            "unit": unit,
            "category": category,
            "direction": direction,
            "evaluation_suite": (
                evaluation_suite.model_dump(mode="json") if evaluation_suite is not None else None
            ),
        }
    )


def _unavailable(
    reason: str,
    *,
    unit: str | None,
    category: str,
    direction: str,
    status: str = "unavailable",
    evaluation_suite: EvaluationSuiteIdentity | None = None,
) -> MetricValue:
    return MetricValue.model_validate(
        {
            "value": None,
            "status": status,
            "unit": unit,
            "category": category,
            "direction": direction,
            "reason": reason,
            "evaluation_suite": (
                evaluation_suite.model_dump(mode="json") if evaluation_suite is not None else None
            ),
        }
    )


def _evaluation_suite_identity(result: EvaluationResult) -> EvaluationSuiteIdentity:
    return EvaluationSuiteIdentity(
        evaluator_name=result.evaluator_name,
        evaluator_version=result.evaluator_version,
        dataset_id=result.dataset_id,
        dataset_revision=result.dataset_revision,
        split=result.split,
        dataset_sha256=result.dataset_sha256,
        sample_ids=result.sample_ids,
        seed=result.seed,
        prompt_template_hash=result.prompt_template_hash,
        decode_config=result.decode_config,
        status=result.status,
    )


def _evaluation_suite_key(suite: EvaluationSuiteIdentity) -> str:
    scientific_identity = suite.model_dump(mode="json", exclude={"status"})
    digest = hashlib.sha256(
        json.dumps(
            scientific_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return digest


def _expected_evaluation_aliases(suite: EvaluationSuiteIdentity) -> tuple[str, ...]:
    if suite.evaluator_name in {"perplexity", "forward_loss"}:
        return ("mean_nll", "perplexity", "quality")
    if suite.evaluator_name in {"generation_regression", "exact_match"}:
        return ("generation_exact_match", "quality")
    if suite.evaluator_name == "behavioral_retention":
        return (
            "behavioral_score",
            "behavioral_base_score",
            "behavioral_fine_tuned_score",
        )
    if suite.evaluator_name == "multimodal_contract":
        return ("multimodal_contract",)
    return ()


def _evaluation_alias_spec(
    name: str, suite: EvaluationSuiteIdentity
) -> tuple[str | None, str, str]:
    if name == "mean_nll":
        return "nats/token", "quality", "minimize"
    if name == "perplexity":
        return None, "quality", "minimize"
    if name == "quality":
        unit = (
            "fraction"
            if suite.evaluator_name in {"generation_regression", "exact_match"}
            else "exp(-nll)"
        )
        return unit, "quality", "maximize"
    if name in {
        "generation_exact_match",
        "behavioral_score",
        "behavioral_base_score",
        "behavioral_fine_tuned_score",
    }:
        category = "behavioral" if name.startswith("behavioral_") else "quality"
        return "fraction", category, "maximize"
    if name == "multimodal_contract":
        return "boolean", "quality", "maximize"
    raise ValueError(f"unknown evaluation alias specification: {name}")


def _evaluation_alias_priority(name: str, suite: EvaluationSuiteIdentity) -> int:
    if name == "quality" and suite.evaluator_name in {"generation_regression", "exact_match"}:
        return 10
    if name == "quality":
        return 20
    return 10


def _evaluation_metrics(
    results: tuple[EvaluationResult, ...],
) -> tuple[dict[str, MetricValue], list[str]]:
    metrics: dict[str, MetricValue] = {}
    failures: list[str] = []
    alias_candidates: dict[str, dict[str, MetricValue]] = {}
    scoped_results = sorted(
        ((_evaluation_suite_identity(result), result) for result in results),
        key=lambda item: _suite_provenance_key(item[0]),
    )
    suite_keys = tuple(_evaluation_suite_key(suite) for suite, _ in scoped_results)
    if len(set(suite_keys)) != len(suite_keys):
        raise ValueError(
            "duplicate evaluation suite scientific identities make metric provenance ambiguous"
        )

    for (suite, result), suite_key in zip(scoped_results, suite_keys, strict=True):

        def retain(
            name: str,
            metric: MetricValue,
            *,
            scope: str = suite_key,
        ) -> None:
            scoped_name = f"evaluation.{scope}.{name}"
            metrics[scoped_name] = metric
            alias_candidates.setdefault(name, {})[scope] = metric

        failures.extend(
            f"{result.evaluator_name}/{failure.sample_id or 'suite'}: {failure.message}"
            for failure in result.failures
        )
        if "mean_nll" in result.metrics:
            mean_nll = float(cast(float | int, result.metrics["mean_nll"]))
            perplexity = float(cast(float | int, result.metrics["perplexity"]))
            retain(
                "mean_nll",
                _available(
                    mean_nll,
                    unit="nats/token",
                    category="quality",
                    direction="minimize",
                    evaluation_suite=suite,
                ),
            )
            retain(
                "perplexity",
                _available(
                    perplexity,
                    unit=None,
                    category="quality",
                    direction="minimize",
                    evaluation_suite=suite,
                ),
            )
            retain(
                "quality",
                _available(
                    pow(2.718281828459045, -mean_nll),
                    unit="exp(-nll)",
                    category="quality",
                    direction="maximize",
                    evaluation_suite=suite,
                ),
            )
        if "exact_match" in result.metrics:
            exact_match = float(cast(float | int, result.metrics["exact_match"]))
            retain(
                "generation_exact_match",
                _available(
                    exact_match,
                    unit="fraction",
                    category="quality",
                    direction="maximize",
                    evaluation_suite=suite,
                ),
            )
            retain(
                "quality",
                _available(
                    exact_match,
                    unit="fraction",
                    category="quality",
                    direction="maximize",
                    evaluation_suite=suite,
                ),
            )
        if "rubric_score" in result.metrics:
            retain(
                "behavioral_score",
                _available(
                    float(cast(float | int, result.metrics["rubric_score"])),
                    unit="fraction",
                    category="behavioral",
                    direction="maximize",
                    evaluation_suite=suite,
                ),
            )
            evidence_reason = (
                "behavioral retention requires explicit per-sample base and fine-tuned "
                "rubric evidence"
            )
            base_score = result.metrics.get("base_rubric_score")
            fine_tuned_score = result.metrics.get("fine_tuned_rubric_score")
            behavioral_base_score = (
                _available(
                    float(base_score),
                    unit="fraction",
                    category="behavioral",
                    direction="maximize",
                    evaluation_suite=suite,
                )
                if isinstance(base_score, (float, int)) and not isinstance(base_score, bool)
                else _unavailable(
                    evidence_reason,
                    unit="fraction",
                    category="behavioral",
                    direction="maximize",
                    evaluation_suite=suite,
                )
            )
            retain("behavioral_base_score", behavioral_base_score)
            behavioral_fine_tuned_score = (
                _available(
                    float(fine_tuned_score),
                    unit="fraction",
                    category="behavioral",
                    direction="maximize",
                    evaluation_suite=suite,
                )
                if isinstance(fine_tuned_score, (float, int))
                and not isinstance(fine_tuned_score, bool)
                else _unavailable(
                    evidence_reason,
                    unit="fraction",
                    category="behavioral",
                    direction="maximize",
                    evaluation_suite=suite,
                )
            )
            retain("behavioral_fine_tuned_score", behavioral_fine_tuned_score)
        if result.metrics.get("contract_passed") is True:
            retain(
                "multimodal_contract",
                _available(
                    1.0,
                    unit="boolean",
                    category="quality",
                    direction="maximize",
                    evaluation_suite=suite,
                ),
            )

    suites_by_key = {
        suite_key: suite for (suite, _), suite_key in zip(scoped_results, suite_keys, strict=True)
    }
    expected_sources: dict[str, list[tuple[int, str]]] = {}
    for suite_key, suite in suites_by_key.items():
        for name in _expected_evaluation_aliases(suite):
            expected_sources.setdefault(name, []).append(
                (_evaluation_alias_priority(name, suite), suite_key)
            )
            candidates = alias_candidates.setdefault(name, {})
            if suite_key in candidates:
                continue
            unit, category, direction = _evaluation_alias_spec(name, suite)
            status = suite.status if suite.status in {"failed", "unsupported"} else "unavailable"
            missing = _unavailable(
                f"{suite.evaluator_name} suite is {suite.status} and did not produce {name}",
                unit=unit,
                category=category,
                direction=direction,
                status=status,
                evaluation_suite=suite,
            )
            metrics[f"evaluation.{suite_key}.{name}"] = missing
            candidates[suite_key] = missing

    for name, candidates in alias_candidates.items():
        sources = expected_sources.get(name, ())
        if sources:
            _, selected_key = min(sources)
            metrics[name] = candidates[selected_key]
        elif len(candidates) == 1:
            metrics[name] = next(iter(candidates.values()))
        elif name == "quality":
            metrics[name] = candidates[min(candidates)]
    return metrics, failures


def _benchmark_metrics(
    result: BenchmarkResult | None,
) -> tuple[dict[str, MetricValue], list[str]]:
    if result is None:
        reason = "benchmarking was disabled by the resolved configuration"
        specifications = (
            ("load_time_ms", "ms", "minimize"),
            ("latency_ms", "ms", "minimize"),
            ("latency_p90_ms", "ms", "minimize"),
            ("latency_mean_ms", "ms", "minimize"),
            ("latency_stdev_ms", "ms", "minimize"),
            ("latency_mean_ci95_lower_ms", "ms", "minimize"),
            ("latency_mean_ci95_upper_ms", "ms", "minimize"),
            ("time_to_first_token_ms", "ms", "minimize"),
            ("inter_token_latency_ms", "ms", "minimize"),
            ("throughput_tokens_per_second", "tokens/s", "maximize"),
            ("throughput_mean_tokens_per_second", "tokens/s", "maximize"),
            ("throughput_stdev_tokens_per_second", "tokens/s", "minimize"),
            (
                "throughput_mean_ci95_lower_tokens_per_second",
                "tokens/s",
                "maximize",
            ),
            (
                "throughput_mean_ci95_upper_tokens_per_second",
                "tokens/s",
                "maximize",
            ),
            ("serialized_size_bytes", "bytes", "minimize"),
            ("peak_host_memory_bytes", "bytes", "minimize"),
            ("benchmark_stage_worker_process_peak_rss_bytes", "bytes", "neutral"),
            ("benchmark_stage_worker_pid", "pid", "neutral"),
            (
                "benchmark_subject_artifact_worker_process_peak_rss_bytes",
                "bytes",
                "neutral",
            ),
            ("benchmark_subject_artifact_worker_pid", "pid", "neutral"),
            ("process_lifetime_peak_host_rss_bytes", "bytes", "neutral"),
            ("benchmark_interval_baseline_current_rss_bytes", "bytes", "neutral"),
            ("benchmark_interval_max_sampled_current_rss_bytes", "bytes", "neutral"),
            ("benchmark_interval_sampled_current_rss_delta_bytes", "bytes", "neutral"),
            ("benchmark_memory_sample_count", "samples", "neutral"),
            ("benchmark_memory_interval_seconds", "seconds", "neutral"),
            ("peak_device_memory_bytes", "bytes", "minimize"),
            (
                "benchmark_interval_max_sampled_current_device_memory_bytes",
                "bytes",
                "neutral",
            ),
            ("process_cpu_utilization_percent", "percent", "neutral"),
            ("energy_joules", "joules", "minimize"),
        )
        return (
            {
                name: _unavailable(
                    reason,
                    unit=unit,
                    category="resource",
                    direction=direction,
                )
                for name, unit, direction in specifications
            },
            [reason],
        )
    latency = result.latency.end_to_end_ms
    throughput = result.throughput_tokens_per_second
    metrics = {
        "load_time_ms": _available(
            result.model_load_time_ms, unit="ms", category="resource", direction="minimize"
        ),
        "latency_ms": _available(
            latency.median,
            unit="ms",
            category="resource",
            direction="minimize",
        ),
        "latency_p90_ms": _available(
            latency.p90,
            unit="ms",
            category="resource",
            direction="minimize",
        ),
        "latency_mean_ms": _available(
            latency.mean,
            unit="ms",
            category="resource",
            direction="minimize",
        ),
        "latency_stdev_ms": _available(
            latency.stdev,
            unit="ms",
            category="resource",
            direction="minimize",
        ),
        "time_to_first_token_ms": _available(
            result.latency.time_to_first_token_ms.median,
            unit="ms",
            category="resource",
            direction="minimize",
        ),
        "throughput_tokens_per_second": _available(
            throughput.median,
            unit="tokens/s",
            category="resource",
            direction="maximize",
        ),
        "throughput_mean_tokens_per_second": _available(
            throughput.mean,
            unit="tokens/s",
            category="resource",
            direction="maximize",
        ),
        "throughput_stdev_tokens_per_second": _available(
            throughput.stdev,
            unit="tokens/s",
            category="resource",
            direction="minimize",
        ),
        "serialized_size_bytes": _available(
            result.serialized_size_bytes,
            unit="bytes",
            category="resource",
            direction="minimize",
        ),
    }
    unsupported: list[str] = []
    if result.repetitions >= 2:
        latency_half_width = 1.96 * latency.stdev / math.sqrt(result.repetitions)
        throughput_half_width = 1.96 * throughput.stdev / math.sqrt(result.repetitions)
        metrics.update(
            {
                "latency_mean_ci95_lower_ms": _available(
                    latency.mean - latency_half_width,
                    unit="ms",
                    category="resource",
                    direction="minimize",
                ),
                "latency_mean_ci95_upper_ms": _available(
                    latency.mean + latency_half_width,
                    unit="ms",
                    category="resource",
                    direction="minimize",
                ),
                "throughput_mean_ci95_lower_tokens_per_second": _available(
                    throughput.mean - throughput_half_width,
                    unit="tokens/s",
                    category="resource",
                    direction="maximize",
                ),
                "throughput_mean_ci95_upper_tokens_per_second": _available(
                    throughput.mean + throughput_half_width,
                    unit="tokens/s",
                    category="resource",
                    direction="maximize",
                ),
            }
        )
    else:
        reason = "95% normal-approximation interval requires at least two repetitions"
        for name, unit, direction in (
            ("latency_mean_ci95_lower_ms", "ms", "minimize"),
            ("latency_mean_ci95_upper_ms", "ms", "minimize"),
            (
                "throughput_mean_ci95_lower_tokens_per_second",
                "tokens/s",
                "maximize",
            ),
            (
                "throughput_mean_ci95_upper_tokens_per_second",
                "tokens/s",
                "maximize",
            ),
        ):
            metrics[name] = _unavailable(
                reason,
                unit=unit,
                category="resource",
                direction=direction,
            )
        unsupported.append(reason)
    if result.latency.inter_token_latency_ms is None:
        metrics["inter_token_latency_ms"] = _unavailable(
            "generation produced one token per trial",
            unit="ms",
            category="resource",
            direction="minimize",
        )
        unsupported.append("inter-token latency unavailable")
    else:
        metrics["inter_token_latency_ms"] = _available(
            result.latency.inter_token_latency_ms.median,
            unit="ms",
            category="resource",
            direction="minimize",
        )
    host_memory = result.peak_memory
    interval_peak_kinds = {
        "benchmark_interval_peak_allocated",
        "benchmark_interval_peak_rss",
        "benchmark_stage_worker_process_peak_rss",
        "benchmark_subject_artifact_worker_process_peak_rss",
    }
    if (
        host_memory.host_available
        and host_memory.host_bytes is not None
        and host_memory.host_measurement_kind in interval_peak_kinds
    ):
        metrics["peak_host_memory_bytes"] = _available(
            host_memory.host_bytes,
            unit="bytes",
            category="resource",
            direction="minimize",
        )
    else:
        if host_memory.host_measurement_kind == "process_lifetime_high_water_rss":
            host_peak_reason = (
                "candidate-scoped peak host memory unavailable: the CPU collector exposes only "
                "process-lifetime high-water RSS, which includes allocations before this benchmark"
            )
        elif host_memory.host_measurement_kind == "benchmark_interval_sampled_current_rss":
            host_peak_reason = (
                "continuous peak host memory unavailable: current RSS is sampled only at the "
                "pre-warm-up and post-trial boundaries"
            )
        elif host_memory.host_available:
            host_peak_reason = (
                "candidate-scoped peak host memory unavailable: collector kind "
                f"{host_memory.host_measurement_kind!r} is not an interval-native peak"
            )
        else:
            host_peak_reason = "host memory measurement unavailable"
        metrics["peak_host_memory_bytes"] = _unavailable(
            host_peak_reason,
            unit="bytes",
            category="resource",
            direction="minimize",
        )
        unsupported.append(host_peak_reason)
    stage_worker_peak = (
        host_memory.host_available
        and host_memory.host_bytes is not None
        and host_memory.host_measurement_kind == "benchmark_stage_worker_process_peak_rss"
        and host_memory.host_process_isolated
        and host_memory.host_worker_pid is not None
    )
    if stage_worker_peak:
        assert host_memory.host_bytes is not None
        assert host_memory.host_worker_pid is not None
        metrics["benchmark_stage_worker_process_peak_rss_bytes"] = _available(
            host_memory.host_bytes,
            unit="bytes",
            category="resource",
            direction="neutral",
        )
        metrics["benchmark_stage_worker_pid"] = _available(
            host_memory.host_worker_pid,
            unit="pid",
            category="resource",
            direction="neutral",
        )
    else:
        reason = "fresh benchmark-stage worker peak RSS is unavailable from this collector"
        metrics["benchmark_stage_worker_process_peak_rss_bytes"] = _unavailable(
            reason,
            unit="bytes",
            category="resource",
            direction="neutral",
        )
        metrics["benchmark_stage_worker_pid"] = _unavailable(
            reason,
            unit="pid",
            category="resource",
            direction="neutral",
        )
    subject_artifact_peak = (
        host_memory.host_available
        and host_memory.host_bytes is not None
        and host_memory.host_measurement_kind
        == "benchmark_subject_artifact_worker_process_peak_rss"
        and host_memory.host_process_isolated
        and host_memory.host_worker_pid is not None
    )
    if subject_artifact_peak:
        assert host_memory.host_bytes is not None
        assert host_memory.host_worker_pid is not None
        metrics["benchmark_subject_artifact_worker_process_peak_rss_bytes"] = _available(
            host_memory.host_bytes,
            unit="bytes",
            category="resource",
            direction="neutral",
        )
        metrics["benchmark_subject_artifact_worker_pid"] = _available(
            host_memory.host_worker_pid,
            unit="pid",
            category="resource",
            direction="neutral",
        )
    else:
        reason = "fresh subject-artifact worker peak RSS is unavailable from this collector"
        metrics["benchmark_subject_artifact_worker_process_peak_rss_bytes"] = _unavailable(
            reason,
            unit="bytes",
            category="resource",
            direction="neutral",
        )
        metrics["benchmark_subject_artifact_worker_pid"] = _unavailable(
            reason,
            unit="pid",
            category="resource",
            direction="neutral",
        )
    if (
        host_memory.host_available
        and host_memory.host_bytes is not None
        and host_memory.host_measurement_kind == "process_lifetime_high_water_rss"
    ):
        metrics["process_lifetime_peak_host_rss_bytes"] = _available(
            host_memory.host_bytes,
            unit="bytes",
            category="resource",
            direction="neutral",
        )
    else:
        metrics["process_lifetime_peak_host_rss_bytes"] = _unavailable(
            "process-lifetime high-water RSS is unavailable from this memory collector",
            unit="bytes",
            category="resource",
            direction="neutral",
        )
    sampled_current_rss = (
        host_memory.host_available
        and host_memory.host_bytes is not None
        and host_memory.host_measurement_kind == "benchmark_interval_sampled_current_rss"
        and host_memory.host_baseline_bytes is not None
        and host_memory.host_max_observed_delta_bytes is not None
        and host_memory.host_sample_count is not None
        and host_memory.measurement_interval_seconds is not None
    )
    sampled_specs = (
        (
            "benchmark_interval_baseline_current_rss_bytes",
            host_memory.host_baseline_bytes,
            "bytes",
        ),
        (
            "benchmark_interval_max_sampled_current_rss_bytes",
            host_memory.host_bytes,
            "bytes",
        ),
        (
            "benchmark_interval_sampled_current_rss_delta_bytes",
            host_memory.host_max_observed_delta_bytes,
            "bytes",
        ),
        ("benchmark_memory_sample_count", host_memory.host_sample_count, "samples"),
        (
            "benchmark_memory_interval_seconds",
            host_memory.measurement_interval_seconds,
            "seconds",
        ),
    )
    if sampled_current_rss:
        for name, value, unit in sampled_specs:
            assert value is not None
            metrics[name] = _available(
                value,
                unit=unit,
                category="resource",
                direction="neutral",
            )
    else:
        reason = "sampled benchmark-interval current RSS is unavailable from this collector"
        for name, _, unit in sampled_specs:
            metrics[name] = _unavailable(
                reason,
                unit=unit,
                category="resource",
                direction="neutral",
            )
    device_memory = result.peak_memory
    device_peak_kinds = {
        "cuda_allocator_peak_since_previous_sample",
    }
    sampled_current_device_kinds = {
        "mlx_allocator_active_bytes_at_sample",
        "mps_driver_allocated_memory_at_sample",
    }
    if (
        device_memory.device_available
        and device_memory.device_bytes is not None
        and device_memory.device_measurement_kind in device_peak_kinds
    ):
        metrics["peak_device_memory_bytes"] = _available(
            device_memory.device_bytes,
            unit="bytes",
            category="resource",
            direction="minimize",
        )
    else:
        if device_memory.device_measurement_kind in sampled_current_device_kinds:
            device_peak_reason = (
                "continuous peak device memory unavailable: current device allocation is sampled "
                "only after warm-up and measured trials"
            )
        elif device_memory.device_available:
            device_peak_reason = (
                "peak device memory unavailable: collector kind "
                f"{device_memory.device_measurement_kind!r} is not an allocator-native or "
                "interval-native peak"
            )
        else:
            device_peak_reason = "device memory measurement unavailable"
        metrics["peak_device_memory_bytes"] = _unavailable(
            device_peak_reason,
            unit="bytes",
            category="resource",
            direction="minimize",
            status="unsupported",
        )
        unsupported.append(device_peak_reason)
    if (
        device_memory.device_available
        and device_memory.device_bytes is not None
        and device_memory.device_measurement_kind in sampled_current_device_kinds
    ):
        metrics["benchmark_interval_max_sampled_current_device_memory_bytes"] = _available(
            device_memory.device_bytes,
            unit="bytes",
            category="resource",
            direction="neutral",
        )
    else:
        metrics["benchmark_interval_max_sampled_current_device_memory_bytes"] = _unavailable(
            "sampled benchmark-interval current device memory is unavailable from this collector",
            unit="bytes",
            category="resource",
            direction="neutral",
        )
    utilization = result.hardware_utilization
    if utilization.status == "available" and utilization.value_percent is not None:
        metrics["process_cpu_utilization_percent"] = _available(
            utilization.value_percent,
            unit="percent",
            category="resource",
            direction="neutral",
        )
    else:
        metrics["process_cpu_utilization_percent"] = _unavailable(
            utilization.reason or "process CPU utilization unavailable",
            unit="percent",
            category="resource",
            direction="neutral",
        )
        unsupported.append(utilization.reason or "process CPU utilization unavailable")
    if result.energy.status == "available" and result.energy.joules is not None:
        metrics["energy_joules"] = _available(
            result.energy.joules,
            unit="joules",
            category="resource",
            direction="minimize",
        )
    else:
        metrics["energy_joules"] = _unavailable(
            result.energy.reason or "energy unavailable",
            unit="joules",
            category="resource",
            direction="minimize",
            status="unsupported",
        )
        unsupported.append(result.energy.reason or "energy measurement unavailable")
    return metrics, unsupported


def _routing_metrics(
    comparison: RoutingComparison | None, *, candidate: bool
) -> tuple[dict[str, MetricValue], list[str]]:
    if comparison is None:
        reason = "routing analysis is unsupported or disabled"
        return (
            {
                "routing_js_divergence": _unavailable(
                    reason,
                    unit="nats",
                    category="routing",
                    direction="minimize",
                    status="unsupported",
                ),
                "route_agreement": _unavailable(
                    reason,
                    unit="fraction",
                    category="routing",
                    direction="maximize",
                    status="unsupported",
                ),
            },
            [reason],
        )
    weighted = comparison.weighted
    metrics: dict[str, MetricValue] = {
        "routing_js_divergence": _available(
            weighted.js_divergence if candidate else 0.0,
            unit="nats",
            category="routing",
            direction="minimize",
        ),
        "routing_load_imbalance": _available(
            (weighted.candidate_load_imbalance if candidate else weighted.baseline_load_imbalance),
            unit="coefficient_of_variation",
            category="routing",
            direction="minimize",
        ),
        "routing_entropy": _available(
            weighted.candidate_entropy if candidate else weighted.baseline_entropy,
            unit="nats",
            category="routing",
            direction="neutral",
        ),
    }
    unsupported: list[str] = []
    if weighted.token_route_agreement is None:
        metrics["route_agreement"] = _unavailable(
            "aligned token traces were not available",
            unit="fraction",
            category="routing",
            direction="maximize",
        )
        unsupported.append("token route agreement unavailable")
    else:
        metrics["route_agreement"] = _available(
            1.0 if not candidate else weighted.token_route_agreement,
            unit="fraction",
            category="routing",
            direction="maximize",
        )
    if weighted.top_k_overlap is None:
        metrics["top_k_overlap"] = _unavailable(
            "aligned token traces were not available",
            unit="fraction",
            category="routing",
            direction="maximize",
        )
    else:
        metrics["top_k_overlap"] = _available(
            1.0 if not candidate else weighted.top_k_overlap,
            unit="fraction",
            category="routing",
            direction="maximize",
        )
    return metrics, unsupported


def _dataset_contract(
    results: tuple[EvaluationResult, ...],
) -> tuple[
    tuple[DatasetIdentity, ...],
    tuple[int, ...],
    tuple[str, ...],
    str,
    dict[str, Any],
    tuple[EvaluationSuiteIdentity, ...],
]:
    evaluation_suites = tuple(
        sorted(
            (_evaluation_suite_identity(result) for result in results),
            key=lambda suite: suite.canonical_json(),
        )
    )
    datasets = tuple(
        sorted(
            {
                DatasetIdentity(
                    dataset_id=result.dataset_id,
                    dataset_revision=result.dataset_revision,
                    split=result.split,
                    dataset_sha256=result.dataset_sha256,
                )
                for result in results
                if result.status in {"success", "partial"} and result.dataset_sha256 is not None
            },
            key=lambda item: (
                item.dataset_id,
                item.dataset_revision,
                item.split,
                item.dataset_sha256,
            ),
        )
    )
    logical_identities = {(item.dataset_id, item.dataset_revision, item.split) for item in datasets}
    if len(logical_identities) != len(datasets):
        raise ValueError(
            "evaluation results disagree on the exact SHA-256 for one dataset identity"
        )
    seed_set = tuple(sorted({result.seed for result in results}))
    sample_ids = tuple(sorted({sample for result in results for sample in result.sample_ids}))
    prompt_contract = [
        {
            "evaluator_name": suite.evaluator_name,
            "evaluator_version": suite.evaluator_version,
            "dataset_id": suite.dataset_id,
            "dataset_revision": suite.dataset_revision,
            "split": suite.split,
            "prompt_template_hash": suite.prompt_template_hash,
        }
        for suite in evaluation_suites
    ]
    prompt_contract_hash = hashlib.sha256(
        json.dumps(prompt_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    decode = {
        f"{index:04d}:{suite.evaluator_name}": cast(Any, suite.decode_config)
        for index, suite in enumerate(evaluation_suites)
    }
    return datasets, seed_set, sample_ids, prompt_contract_hash, decode, evaluation_suites


def _suite_provenance_key(suite: EvaluationSuiteIdentity) -> str:
    """Compare scientific inputs independently from evaluator outcome status."""

    return json.dumps(
        suite.model_dump(mode="json", exclude={"status"}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _hardware_contract(config: ExperimentConfig, environment: dict[str, Any]) -> dict[str, Any]:
    """Normalize the actual host and requested runtime placement for compatibility."""

    hardware = environment.get("hardware")
    platform = environment.get("platform")
    if not isinstance(hardware, dict) or not hardware:
        raise ValueError("environment is missing hardware provenance")
    if not isinstance(platform, dict) or not platform:
        raise ValueError("environment is missing platform provenance")
    return {
        "hardware": hardware,
        "platform": {key: platform.get(key) for key in ("system", "machine", "processor")},
        "runtime": config.runtime.model_dump(mode="json"),
    }


def _routing_dataset_contract(
    config: ExperimentConfig,
) -> tuple[DatasetIdentity | None, tuple[str, ...]]:
    """Load the exact routing dataset identity used by the capture stage."""

    if config.routing.mode == "off":
        return None, ()
    dataset = load_local_dataset(
        config.routing.dataset,
        config.routing.dataset_revision,
        config.routing.dataset_split,
    )
    return (
        DatasetIdentity(
            dataset_id=dataset.dataset_id,
            dataset_revision=dataset.revision,
            split=dataset.split,
            dataset_sha256=dataset.sha256,
        ),
        dataset.sample_ids,
    )


def _routing_capture_contract(
    config: ExperimentConfig,
    artifact: RoutingArtifact | None,
    expected_sample_ids: tuple[str, ...],
) -> RoutingCaptureIdentity | None:
    """Normalize persisted capture mode and exact retained alignment-key evidence."""

    if artifact is None:
        return None
    if artifact.mode != config.routing.mode:
        raise ValueError(
            "persisted routing capture mode differs from the resolved configuration: "
            f"configured={config.routing.mode}, captured={artifact.mode}"
        )
    observed_sample_ids = {
        sample_id
        for aggregate in artifact.aggregates.values()
        for sample_id in aggregate.sample_token_counts
    }
    if observed_sample_ids != set(expected_sample_ids):
        raise ValueError(
            "persisted routing capture sample IDs differ from the routing dataset contract"
        )
    alignment_keys = tuple(sorted(event.alignment_key for event in artifact.raw_traces))
    if len(set(alignment_keys)) != len(alignment_keys):
        raise ValueError("persisted routing traces contain duplicate token alignment keys")
    alignment_digest = (
        hashlib.sha256(
            json.dumps(alignment_keys, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if alignment_keys
        else None
    )
    return RoutingCaptureIdentity(
        configured_mode=config.routing.mode,
        captured_mode=artifact.mode,
        observed_event_count=artifact.observed_event_count,
        recorded_event_count=artifact.recorded_event_count,
        alignment_key_count=len(alignment_keys),
        alignment_key_sha256=alignment_digest,
    )


def _metric_number(metrics: dict[str, MetricValue], name: str) -> float | None:
    metric = metrics.get(name)
    if metric is None or metric.status != "available":
        return None
    return metric.value


def _add_behavioral_retention(
    baseline_metrics: dict[str, MetricValue],
    candidate_metrics: dict[str, MetricValue],
) -> None:
    """Normalize B/F/Q only from explicit evidence that matches measured F."""

    behavioral_names = {
        "behavioral_score",
        "behavioral_base_score",
        "behavioral_fine_tuned_score",
    }
    if not behavioral_names.intersection(baseline_metrics | candidate_metrics):
        return
    base_score = _metric_number(baseline_metrics, "behavioral_base_score")
    fine_tuned_score = _metric_number(baseline_metrics, "behavioral_fine_tuned_score")
    measured_fine_tuned_score = _metric_number(baseline_metrics, "behavioral_score")
    quantized_score = _metric_number(candidate_metrics, "behavioral_score")
    candidate_base_score = _metric_number(candidate_metrics, "behavioral_base_score")
    candidate_fine_tuned_score = _metric_number(candidate_metrics, "behavioral_fine_tuned_score")
    baseline_suite = (
        baseline_metrics["behavioral_score"].evaluation_suite
        if "behavioral_score" in baseline_metrics
        else None
    )
    candidate_suite = (
        candidate_metrics["behavioral_score"].evaluation_suite
        if "behavioral_score" in candidate_metrics
        else None
    )
    reason: str | None = None
    if None in {
        base_score,
        fine_tuned_score,
        measured_fine_tuned_score,
        quantized_score,
        candidate_base_score,
        candidate_fine_tuned_score,
    }:
        reason = (
            "behavioral retention is unavailable without explicit base/fine-tuned "
            "evidence and measured fine-tuned/candidate rubric scores"
        )
    elif not math.isclose(
        cast(float, base_score),
        cast(float, candidate_base_score),
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or not math.isclose(
        cast(float, fine_tuned_score),
        cast(float, candidate_fine_tuned_score),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        reason = "baseline and candidate behavioral reference evidence disagree"
    elif not math.isclose(
        cast(float, fine_tuned_score),
        cast(float, measured_fine_tuned_score),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        reason = (
            "measured baseline rubric score does not match the declared fine-tuned "
            "reference evidence"
        )
    if reason is not None:
        baseline_unavailable = _unavailable(
            reason,
            unit="fraction",
            category="behavioral",
            direction="maximize",
            evaluation_suite=baseline_suite,
        )
        candidate_unavailable = _unavailable(
            reason,
            unit="fraction",
            category="behavioral",
            direction="maximize",
            evaluation_suite=candidate_suite,
        )
        baseline_metrics["behavioral_retention"] = baseline_unavailable
        candidate_metrics["behavioral_retention"] = candidate_unavailable
        return
    baseline_metrics["behavioral_retention"] = _available(
        retention_score(
            cast(float, base_score),
            cast(float, fine_tuned_score),
            cast(float, measured_fine_tuned_score),
        ),
        unit="fraction",
        category="behavioral",
        direction="maximize",
        evaluation_suite=baseline_suite,
    )
    candidate_metrics["behavioral_retention"] = _available(
        retention_score(
            cast(float, base_score),
            cast(float, fine_tuned_score),
            cast(float, quantized_score),
        ),
        unit="fraction",
        category="behavioral",
        direction="maximize",
        evaluation_suite=candidate_suite,
    )


def build_run_summaries(
    *,
    run_id: str,
    run_directory: Path,
    config: ExperimentConfig,
    descriptor: ModelDescriptor,
    policy: ResolvedPrecisionPolicy,
    quantization_manifest: QuantizationManifest,
    baseline_evaluations: tuple[EvaluationResult, ...],
    candidate_evaluations: tuple[EvaluationResult, ...],
    baseline_benchmark: BenchmarkResult | None,
    candidate_benchmark: BenchmarkResult | None,
    routing: RoutingComparison | None,
    baseline_routing_artifact: RoutingArtifact | None,
    candidate_routing_artifact: RoutingArtifact | None,
    environment: dict[str, Any],
) -> tuple[NormalizedRunSummary, NormalizedRunSummary]:
    """Build baseline and candidate summaries with identical comparison contracts."""

    (
        datasets,
        seed_set,
        sample_ids,
        prompt_hash,
        decode_config,
        evaluation_suites,
    ) = _dataset_contract(baseline_evaluations)
    candidate_contract = _dataset_contract(candidate_evaluations)
    (
        candidate_datasets,
        candidate_seed_set,
        candidate_sample_ids,
        candidate_prompt_hash,
        candidate_decode_config,
        candidate_evaluation_suites,
    ) = candidate_contract
    if (
        candidate_datasets,
        candidate_seed_set,
        candidate_sample_ids,
        candidate_prompt_hash,
        candidate_decode_config,
        tuple(_suite_provenance_key(suite) for suite in candidate_evaluation_suites),
    ) != (
        datasets,
        seed_set,
        sample_ids,
        prompt_hash,
        decode_config,
        tuple(_suite_provenance_key(suite) for suite in evaluation_suites),
    ):
        raise ValueError("baseline and candidate evaluation contracts differ within one run")
    if seed_set != (config.seed,):
        raise ValueError(
            "evaluation result seed set differs from the resolved experiment seed: "
            f"config={config.seed}, results={seed_set}"
        )
    if config.benchmark.enabled and (baseline_benchmark is None or candidate_benchmark is None):
        raise ValueError("enabled benchmarking requires baseline and candidate results")
    if not config.benchmark.enabled and (
        baseline_benchmark is not None or candidate_benchmark is not None
    ):
        raise ValueError("disabled benchmarking cannot carry benchmark results")
    baseline_benchmark_workload = (
        BenchmarkWorkloadIdentity.model_validate(
            baseline_benchmark.workload.model_dump(mode="json")
        )
        if baseline_benchmark is not None
        else None
    )
    candidate_benchmark_workload = (
        BenchmarkWorkloadIdentity.model_validate(
            candidate_benchmark.workload.model_dump(mode="json")
        )
        if candidate_benchmark is not None
        else None
    )
    baseline_benchmark_memory = (
        BenchmarkMemoryIdentity(
            host_measurement_kind=baseline_benchmark.peak_memory.host_measurement_kind,
            host_scope=baseline_benchmark.peak_memory.host_scope,
            host_process_isolated=baseline_benchmark.peak_memory.host_process_isolated,
            device_measurement_kind=baseline_benchmark.peak_memory.device_measurement_kind,
            device_scope=baseline_benchmark.peak_memory.device_scope,
        )
        if baseline_benchmark is not None
        and (
            baseline_benchmark.peak_memory.host_available
            or baseline_benchmark.peak_memory.device_available
        )
        else None
    )
    candidate_benchmark_memory = (
        BenchmarkMemoryIdentity(
            host_measurement_kind=candidate_benchmark.peak_memory.host_measurement_kind,
            host_scope=candidate_benchmark.peak_memory.host_scope,
            host_process_isolated=candidate_benchmark.peak_memory.host_process_isolated,
            device_measurement_kind=candidate_benchmark.peak_memory.device_measurement_kind,
            device_scope=candidate_benchmark.peak_memory.device_scope,
        )
        if candidate_benchmark is not None
        and (
            candidate_benchmark.peak_memory.host_available
            or candidate_benchmark.peak_memory.device_available
        )
        else None
    )
    routing_dataset, routing_sample_ids = _routing_dataset_contract(config)
    baseline_routing_capture = _routing_capture_contract(
        config, baseline_routing_artifact, routing_sample_ids
    )
    candidate_routing_capture = _routing_capture_contract(
        config, candidate_routing_artifact, routing_sample_ids
    )
    if routing is not None and (
        baseline_routing_capture is None or candidate_routing_capture is None
    ):
        raise ValueError(
            "routing metrics require persisted baseline and candidate capture evidence"
        )
    hardware_environment = _hardware_contract(config, environment)
    baseline_metrics, baseline_failures = _evaluation_metrics(baseline_evaluations)
    candidate_metrics, candidate_failures = _evaluation_metrics(candidate_evaluations)
    baseline_benchmark_metrics, baseline_unsupported = _benchmark_metrics(baseline_benchmark)
    candidate_benchmark_metrics, candidate_unsupported = _benchmark_metrics(candidate_benchmark)
    baseline_routing_metrics, baseline_routing_unsupported = _routing_metrics(
        routing, candidate=False
    )
    candidate_routing_metrics, candidate_routing_unsupported = _routing_metrics(
        routing, candidate=True
    )
    baseline_metrics.update(baseline_benchmark_metrics)
    baseline_metrics.update(baseline_routing_metrics)
    candidate_metrics.update(candidate_benchmark_metrics)
    candidate_metrics.update(candidate_routing_metrics)
    _add_behavioral_retention(baseline_metrics, candidate_metrics)
    baseline = NormalizedRunSummary(
        schema_version="1.1",
        run_id=f"{run_id}:baseline",
        artifact_path=str(run_directory),
        model_id=descriptor.model_id,
        model_revision=descriptor.revision,
        datasets=datasets,
        seed_set=seed_set,
        sample_ids=sample_ids,
        prompt_template_hash=prompt_hash,
        decode_config=decode_config,
        evaluation_suites=evaluation_suites,
        routing_dataset=routing_dataset,
        routing_sample_ids=routing_sample_ids,
        routing_capture=baseline_routing_capture,
        benchmark_protocol_version=config.benchmark.protocol_version,
        benchmark_workload=baseline_benchmark_workload,
        benchmark_memory=baseline_benchmark_memory,
        benchmark_model_load_time_kind=(
            baseline_benchmark.model_load_time_kind if baseline_benchmark is not None else None
        ),
        benchmark_source_weight_free_load_provenance=(
            baseline_benchmark.source_weight_free_load_provenance
            if baseline_benchmark is not None
            else None
        ),
        hardware_environment=hardware_environment,
        metrics=baseline_metrics,
        environment=environment,
        resolved_config=config.canonical_dict(),
        quantization_policy={"type": "baseline", "default_precision": config.model.dtype},
        failures=tuple(baseline_failures),
        unsupported_measurements=tuple(
            sorted({*baseline_unsupported, *baseline_routing_unsupported})
        ),
        warnings=(),
    )
    candidate = NormalizedRunSummary(
        schema_version="1.1",
        run_id=f"{run_id}:candidate",
        artifact_path=str(run_directory),
        model_id=descriptor.model_id,
        model_revision=descriptor.revision,
        datasets=datasets,
        seed_set=seed_set,
        sample_ids=sample_ids,
        prompt_template_hash=prompt_hash,
        decode_config=decode_config,
        evaluation_suites=candidate_evaluation_suites,
        routing_dataset=routing_dataset,
        routing_sample_ids=routing_sample_ids,
        routing_capture=candidate_routing_capture,
        benchmark_protocol_version=config.benchmark.protocol_version,
        benchmark_workload=candidate_benchmark_workload,
        benchmark_memory=candidate_benchmark_memory,
        benchmark_model_load_time_kind=(
            candidate_benchmark.model_load_time_kind if candidate_benchmark is not None else None
        ),
        benchmark_source_weight_free_load_provenance=(
            candidate_benchmark.source_weight_free_load_provenance
            if candidate_benchmark is not None
            else None
        ),
        hardware_environment=hardware_environment,
        metrics=candidate_metrics,
        environment=environment,
        resolved_config=config.canonical_dict(),
        quantization_policy=policy.model_dump(mode="json"),
        failures=tuple(candidate_failures),
        unsupported_measurements=tuple(
            sorted({*candidate_unsupported, *candidate_routing_unsupported})
        ),
        warnings=quantization_manifest.warnings,
    )
    return baseline, candidate
