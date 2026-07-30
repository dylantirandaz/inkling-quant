"""Host-only checks for Inkling measurement latency validation."""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from inkling_quant_lab.gguf.inkling_measurement_evidence import (
    _paired_five_batch_nonnegative_metric,
)
from inkling_quant_lab.gguf.inkling_measurement_execution import summarize_latency_ms
from inkling_quant_lab.gguf.inkling_measurement_raw_evidence import (
    MeasurementFiveBatchMetricSummary,
    MeasurementResourceSampleSummary,
    MeasurementServerBatch,
    MeasurementServerCell,
    MeasurementServerCellSummary,
    MeasurementServerRequest,
    recompute_server_cell_batch_metrics,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = PROJECT_ROOT / "scripts/run_inkling_measurement_modal.py"
_HASHES = ("0" * 64, "1" * 64, "2" * 64)


def _runner_rejects_latency(ttft: float, inter_token: Sequence[float]) -> bool:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
    stream_completion = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_stream_completion"
    )
    latency_guard = next(
        node
        for node in ast.walk(stream_completion)
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Raise)
            and isinstance(statement.exc, ast.Call)
            and statement.exc.args
            and isinstance(statement.exc.args[0], ast.Constant)
            and statement.exc.args[0].value
            == "streaming inter-token latency samples are not finite and non-negative"
            for statement in node.body
        )
    )
    predicate = ast.FunctionDef(
        name="_latency_guard",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="ttft"), ast.arg(arg="inter_token")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[ast.Return(value=latency_guard.test)],
        decorator_list=[],
    )
    isolated = ast.Module(body=[predicate], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, object] = {"math": math}
    exec(compile(isolated, str(RUNNER_PATH), "exec"), namespace)
    guard = cast("Callable[[float, Sequence[float]], bool]", namespace["_latency_guard"])
    return guard(ttft, inter_token)


def test_runner_accepts_finite_zero_inter_token_delta() -> None:
    assert not _runner_rejects_latency(0.01, (0.001, 0.0, 0.002))


@pytest.mark.parametrize("invalid", [0.0, -0.001, math.nan, math.inf, -math.inf])
def test_runner_rejects_invalid_ttft(invalid: float) -> None:
    assert _runner_rejects_latency(invalid, (0.001, 0.0, 0.002))


@pytest.mark.parametrize("invalid", [-0.001, math.nan, math.inf, -math.inf])
def test_runner_rejects_invalid_inter_token_delta(invalid: float) -> None:
    assert _runner_rejects_latency(0.01, (0.001, invalid, 0.002))


def test_latency_summary_accepts_finite_zero_samples() -> None:
    summary = summarize_latency_ms((0.0, 0.0, 1.0))

    assert summary.sample_count == 3
    assert summary.minimum_ms == 0.0
    assert summary.p50_ms == 0.0
    assert summary.maximum_ms == 1.0


@pytest.mark.parametrize("invalid", [-0.001, math.nan, math.inf, -math.inf])
def test_latency_summary_rejects_invalid_samples(invalid: float) -> None:
    with pytest.raises(ValueError, match=r"finite|nonnegative"):
        summarize_latency_ms((0.0, invalid, 1.0))


def _r7_percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _server_request(
    *,
    request_started: float,
    intervals: tuple[float, ...],
) -> MeasurementServerRequest:
    first_token = request_started + 0.25
    last_token = first_token + sum(intervals)
    request_finished = last_token + 0.25
    return MeasurementServerRequest(
        request_body_sha256=_HASHES[0],
        token_ids=tuple(range(128)),
        output_sha256=_HASHES[1],
        response_sha256=_HASHES[2],
        request_started_monotonic_seconds=request_started,
        first_token_monotonic_seconds=first_token,
        last_token_monotonic_seconds=last_token,
        request_finished_monotonic_seconds=request_finished,
        wall_seconds=request_finished - request_started,
        ttft_seconds=first_token - request_started,
        prompt_n=512,
        predicted_n=128,
        prompt_ms=1000.0,
        predicted_ms=1000.0,
        prompt_tokens_per_second=512.0,
        decode_tokens_per_second=128.0,
        inter_token_latency_p50_seconds=_r7_percentile(intervals, 50.0),
        inter_token_latency_p95_seconds=_r7_percentile(intervals, 95.0),
        inter_token_latency_p99_seconds=_r7_percentile(intervals, 99.0),
        raw_inter_token_latency_seconds=intervals,
        prompt_text_recorded=False,
        output_text_recorded=False,
        request_index=1,
    )


def _server_batch(*, batch_index: int, batch_started: float) -> MeasurementServerBatch:
    intervals = (0.0,) * 126 + (1.0,)
    request = _server_request(
        request_started=batch_started + 0.25,
        intervals=intervals,
    )
    batch_finished = request.request_finished_monotonic_seconds + 0.25
    return MeasurementServerBatch(
        batch_index=batch_index,
        concurrency=1,
        batch_started_monotonic_seconds=batch_started,
        batch_finished_monotonic_seconds=batch_finished,
        batch_wall_seconds=batch_finished - batch_started,
        decode_boundary=("earliest_first_token_to_latest_last_token_127_intervals_per_request"),
        aggregate_decode_token_intervals=127,
        batch_duration_seconds=1.0,
        aggregate_decode_tokens_per_second=127.0,
        requests=(request,),
    )


def _server_cell_with_zero_percentiles() -> MeasurementServerCell:
    measured_batches = cast(
        "tuple[MeasurementServerBatch, MeasurementServerBatch, "
        "MeasurementServerBatch, MeasurementServerBatch, MeasurementServerBatch]",
        tuple(
            _server_batch(batch_index=batch_index, batch_started=10.0 * batch_index)
            for batch_index in range(1, 6)
        ),
    )
    return MeasurementServerCell(
        concurrency=1,
        single_request_warmups_completed=2,
        concurrent_batch_warmup_completed=True,
        concurrent_batch_warmup=_server_batch(batch_index=0, batch_started=1.0),
        warmup_output_token_counts=(128, 128),
        concurrent_warmup_request_count=1,
        measured_batches=measured_batches,
        measured_request_count=5,
        mean_ttft_seconds=0.25,
        mean_prompt_tokens_per_second=512.0,
        mean_decode_tokens_per_second=128.0,
        aggregate_decode_tokens_per_second_trials=(127.0,) * 5,
        mean_aggregate_decode_tokens_per_second=127.0,
        inter_token_latency_method=("r7_linear_interpolation_over_all_measured_request_intervals"),
        raw_inter_token_interval_count=5 * 127,
        inter_token_latency_p50_seconds=0.0,
        inter_token_latency_p95_seconds=0.0,
        inter_token_latency_p99_seconds=0.0,
        resource_sample_summary=MeasurementResourceSampleSummary(
            window_started_monotonic_seconds=0.5,
            window_finished_monotonic_seconds=60.0,
            sample_count=1,
            max_sampled_host_rss_bytes=1,
            max_sampled_per_gpu_memory_bytes=(1,) * 8,
            max_sampled_per_gpu_utilization_percent=(0.0,) * 8,
        ),
    )


def test_raw_request_accepts_zero_intervals_and_percentiles() -> None:
    request = _server_request(request_started=1.0, intervals=(0.0,) * 127)

    assert request.first_token_monotonic_seconds == request.last_token_monotonic_seconds
    assert request.raw_inter_token_latency_seconds == (0.0,) * 127
    assert request.inter_token_latency_p50_seconds == 0.0
    assert request.inter_token_latency_p95_seconds == 0.0
    assert request.inter_token_latency_p99_seconds == 0.0


def test_raw_request_rejects_negative_inter_token_interval() -> None:
    valid = _server_request(request_started=1.0, intervals=(0.0,) * 126 + (1.0,))
    payload = valid.model_dump(mode="python")
    payload["raw_inter_token_latency_seconds"] = (-0.25,) + (0.0,) * 125 + (1.25,)

    with pytest.raises(ValidationError, match="nonnegative"):
        MeasurementServerRequest.model_validate(payload)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_raw_request_rejects_nonfinite_inter_token_interval(invalid: float) -> None:
    valid = _server_request(request_started=1.0, intervals=(0.0,) * 126 + (1.0,))
    payload = valid.model_dump(mode="python")
    payload["raw_inter_token_latency_seconds"] = (invalid,) + (0.0,) * 126

    with pytest.raises(ValidationError):
        MeasurementServerRequest.model_validate(payload)


def test_server_cell_rollups_accept_zero_inter_token_percentiles() -> None:
    cell = _server_cell_with_zero_percentiles()
    batch_metrics = recompute_server_cell_batch_metrics(cell)
    summary = MeasurementServerCellSummary(
        concurrency=cell.concurrency,
        measured_batches=5,
        measured_requests=cell.measured_request_count,
        mean_ttft_seconds=cell.mean_ttft_seconds,
        mean_prompt_tokens_per_second=cell.mean_prompt_tokens_per_second,
        mean_decode_tokens_per_second=cell.mean_decode_tokens_per_second,
        mean_aggregate_decode_tokens_per_second=(cell.mean_aggregate_decode_tokens_per_second),
        batch_metrics=batch_metrics,
        inter_token_latency_p50_seconds=cell.inter_token_latency_p50_seconds,
        inter_token_latency_p95_seconds=cell.inter_token_latency_p95_seconds,
        inter_token_latency_p99_seconds=cell.inter_token_latency_p99_seconds,
        resource_sample_summary=cell.resource_sample_summary,
    )

    assert batch_metrics.inter_token_latency_p50_seconds.samples == (0.0,) * 5
    assert batch_metrics.inter_token_latency_p95_seconds.samples == (0.0,) * 5
    assert batch_metrics.inter_token_latency_p99_seconds.samples == (0.0,) * 5
    assert batch_metrics.mean_ttft_seconds.mean > 0.0
    assert batch_metrics.mean_request_end_to_end_latency_seconds.mean > 0.0
    assert batch_metrics.aggregate_decode_tokens_per_second.mean > 0.0
    assert summary.inter_token_latency_p50_seconds == 0.0
    assert summary.inter_token_latency_p95_seconds == 0.0
    assert summary.inter_token_latency_p99_seconds == 0.0


def test_paired_rollup_accepts_zero_inter_token_percentiles() -> None:
    batch_metrics = recompute_server_cell_batch_metrics(_server_cell_with_zero_percentiles())

    paired = _paired_five_batch_nonnegative_metric(
        batch_metrics.inter_token_latency_p50_seconds,
        batch_metrics.inter_token_latency_p50_seconds,
    )

    assert paired.bf16_samples == (0.0,) * 5
    assert paired.q3_samples == (0.0,) * 5
    assert paired.mean.bf16 == 0.0
    assert paired.mean.q3 == 0.0
    assert paired.median.bf16 == 0.0
    assert paired.median.q3 == 0.0


def test_strictly_positive_batch_metric_still_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        MeasurementFiveBatchMetricSummary(
            trial_count=5,
            samples=(0.0,) * 5,
            mean=0.0,
            median=0.0,
            sample_standard_deviation=0.0,
        )
