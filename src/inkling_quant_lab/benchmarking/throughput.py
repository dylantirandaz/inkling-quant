"""Throughput calculations shared by benchmark runners."""

from __future__ import annotations

import math

from inkling_quant_lab.exceptions import BenchmarkError


def tokens_per_second(output_tokens: int, elapsed_seconds: float) -> float:
    """Calculate generated-token throughput for one measured trial."""

    if output_tokens <= 0:
        raise BenchmarkError(
            "throughput requires at least one generated token",
            component="benchmark_throughput",
        )
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise BenchmarkError(
            "throughput requires a finite positive elapsed time",
            component="benchmark_throughput",
        )
    return output_tokens / elapsed_seconds


__all__ = ["tokens_per_second"]
