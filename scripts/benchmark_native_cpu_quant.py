#!/usr/bin/env python3
"""Reproducible native-versus-reference CPU quantized-linear benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, cast

import torch
from torch import Tensor, nn

from inkling_quant_lab.quantization.int8 import DynamicInt8Linear
from inkling_quant_lab.quantization.native_cpu import (
    NativeDynamicInt8Linear,
    NativeInt4KleidiAILinear,
    probe_native_dynamic_int8,
    probe_native_int4_kleidiai,
)
from inkling_quant_lab.quantization.reference import model_storage_bytes
from inkling_quant_lab.quantization.weight_only import PackedInt4Linear


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    interpolation = position - lower
    return ordered[lower] * (1.0 - interpolation) + ordered[upper] * interpolation


def _summary(samples: list[float]) -> dict[str, Any]:
    return {
        "mean_ms": statistics.fmean(samples),
        "median_ms": statistics.median(samples),
        "p10_ms": _percentile(samples, 0.1),
        "p90_ms": _percentile(samples, 0.9),
        "stdev_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "trial_count": len(samples),
        "trials_ms": samples,
    }


def _operation(module: nn.Module, inputs: Tensor) -> Callable[[], Tensor]:
    def run() -> Tensor:
        return cast(Tensor, module(inputs))

    return run


@torch.inference_mode()
def _benchmark(
    operations: Mapping[str, Callable[[], Tensor]],
    *,
    warmup_iterations: int,
    iterations_per_trial: int,
    trials: int,
) -> dict[str, dict[str, Any]]:
    for operation in operations.values():
        for _ in range(warmup_iterations):
            operation()
    samples: dict[str, list[float]] = {name: [] for name in operations}
    names = tuple(operations)
    for trial in range(trials):
        rotated = names[trial % len(names) :] + names[: trial % len(names)]
        for name in rotated:
            operation = operations[name]
            started = time.perf_counter_ns()
            for _ in range(iterations_per_trial):
                operation()
            elapsed_ns = time.perf_counter_ns() - started
            samples[name].append(elapsed_ns / iterations_per_trial / 1_000_000.0)
    return {name: _summary(values) for name, values in samples.items()}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-features", type=int, default=1024)
    parser.add_argument("--out-features", type=int, default=1024)
    parser.add_argument("--batch-tokens", type=int, default=8)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--iterations-per-trial", type=int, default=100)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--interop-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--quantized-engine", default="qnnpack")
    parser.add_argument("--hardware-label", default="unspecified")
    parser.add_argument("--host-isolation", default="not declared")
    return parser.parse_args()


def main() -> None:
    """Run the controlled benchmark and emit one provenance-complete JSON record."""

    arguments = _arguments()
    if (
        min(
            arguments.in_features,
            arguments.out_features,
            arguments.batch_tokens,
            arguments.iterations_per_trial,
            arguments.trials,
            arguments.threads,
            arguments.interop_threads,
        )
        <= 0
    ):
        raise ValueError("dimensions, trials, iterations, and threads must be positive")
    if arguments.warmup_iterations < 0:
        raise ValueError("warmup iterations must be non-negative")
    torch.manual_seed(arguments.seed)
    torch.set_num_threads(arguments.threads)
    torch.set_num_interop_threads(arguments.interop_threads)
    int8_capability = probe_native_dynamic_int8(arguments.quantized_engine)
    int4_capability = probe_native_int4_kleidiai()
    if not int8_capability.supported or int8_capability.implementation is None:
        raise RuntimeError("; ".join(int8_capability.reasons))
    if not int4_capability.supported:
        raise RuntimeError("; ".join(int4_capability.reasons))

    source = nn.Linear(arguments.in_features, arguments.out_features).eval()
    reference_int8 = DynamicInt8Linear(source).eval()
    native_int8 = NativeDynamicInt8Linear(source, engine=int8_capability.implementation).eval()
    reference_int4 = PackedInt4Linear(source).eval()
    native_int4 = NativeInt4KleidiAILinear(source).eval()
    inputs = torch.randn(arguments.batch_tokens, arguments.in_features)
    modules: dict[str, nn.Module] = {
        "float32": source,
        "reference_dynamic_int8": reference_int8,
        "native_dynamic_int8": native_int8,
        "reference_weight_only_int4": reference_int4,
        "native_dynamic_int4_kleidiai": native_int4,
    }
    operations = {name: _operation(module, inputs) for name, module in modules.items()}
    latency = _benchmark(
        operations,
        warmup_iterations=arguments.warmup_iterations,
        iterations_per_trial=arguments.iterations_per_trial,
        trials=arguments.trials,
    )
    with torch.inference_mode():
        float_output = source(inputs)
        errors = {
            name: float((module(inputs) - float_output).abs().max().item())
            for name, module in modules.items()
            if name != "float32"
        }
    reference_int8_median = float(latency["reference_dynamic_int8"]["median_ms"])
    native_int8_median = float(latency["native_dynamic_int8"]["median_ms"])
    reference_int4_median = float(latency["reference_weight_only_int4"]["median_ms"])
    native_int4_median = float(latency["native_dynamic_int4_kleidiai"]["median_ms"])
    record: dict[str, Any] = {
        "schema_version": "native-cpu-quant-benchmark-v1",
        "provenance": {
            "command": "scripts/benchmark_native_cpu_quant.py",
            "measured_at": datetime.now(UTC).isoformat(),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hardware_label": arguments.hardware_label,
            "host_isolation": arguments.host_isolation,
            "logical_cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "quantized_engine": int8_capability.implementation,
            "kleidiai_available": bool(cast(Any, torch.backends.kleidiai).is_available()),
            "seed": arguments.seed,
        },
        "protocol": {
            "input_shape": [arguments.batch_tokens, arguments.in_features],
            "weight_shape": [arguments.out_features, arguments.in_features],
            "dtype": "float32",
            "warmup_iterations_per_operation": arguments.warmup_iterations,
            "iterations_per_trial": arguments.iterations_per_trial,
            "trials": arguments.trials,
            "trial_order": "rotated",
            "construction_and_weight_packing_excluded": True,
        },
        "latency": latency,
        "module_state_storage_bytes": {
            name: model_storage_bytes(module) for name, module in modules.items()
        },
        "max_abs_error_vs_float32": errors,
        "speedup_vs_reference": {
            "native_dynamic_int8": reference_int8_median / native_int8_median,
            "native_dynamic_int4_kleidiai": reference_int4_median / native_int4_median,
        },
    }
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
