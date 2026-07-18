"""CPU-fast performance and scale contracts PT-001 through PT-003."""

from __future__ import annotations

import gc
import json
import statistics
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
import torch

from inkling_quant_lab.comparison import (
    DatasetIdentity,
    MetricValue,
    NormalizedRunSummary,
    ParetoObjective,
    RoutingCaptureIdentity,
    compare_summaries,
    pareto_frontier,
    pareto_points_from_summaries,
)
from inkling_quant_lab.config import (
    BenchmarkConfig,
    ExperimentConfig,
    ModelConfig,
    ReportingConfig,
    RoutingConfig,
)
from inkling_quant_lab.models.base import ModelBatch
from inkling_quant_lab.models.fixtures import FixedTokenizer
from inkling_quant_lab.models.local import LocalFixtureAdapter
from inkling_quant_lab.pipeline.routing_capture import capture_routing_dataset
from inkling_quant_lab.reporting.report import ReportArtifacts, ReportData, generate_report
from inkling_quant_lab.routing import (
    InMemoryRoutingSink,
    RoutingArtifact,
    RoutingEvent,
    RoutingMode,
)
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.integration

_TIMING_REPETITIONS = 11
_TIMED_EVENT_COUNT = 2_048
_AGGREGATE_EVENT_BUDGET_NS = 15_000

_NFR_007_BATCH_SIZE = 16
_NFR_007_SEQUENCE_LENGTH = 64
_NFR_007_DATASET_BATCHES = 8
_NFR_007_WARMUPS = 5
_NFR_007_REPETITIONS = 15
_NFR_007_AGGREGATE_OVERHEAD_LIMIT = 0.15

_TRACE_EVENT_COUNTS = (256, 512, 1_024)
_TRACE_BYTES_PER_EVENT_BOUND = 512
_TRACE_TOTAL_BYTES_BOUND = 512 * 1_024
_TRACE_SLOPE_SPREAD_BOUND = 1.25

_SCALE_CANDIDATE_COUNT = 100
_REPORT_COMPLETION_BUDGET_SECONDS = 5.0
_REPORT_PEAK_MEMORY_BOUND = 32 * 1_024 * 1_024
_REPORT_DOUBLING_MEMORY_RATIO_BOUND = 3.0


def _tiny_moe_events() -> tuple[RoutingEvent, ...]:
    config = ExperimentConfig(
        name="pt-routing-fixture",
        model=ModelConfig(
            model_id="local://fixtures/tiny-moe",
            revision="fixture-v1",
            checkpoint_format="fixture",
        ),
        routing=RoutingConfig(mode="full_trace"),
        benchmark=BenchmarkConfig(enabled=False),
        reporting=ReportingConfig(markdown=False, html=False, plots=False),
    )
    adapter = LocalFixtureAdapter()
    loaded = adapter.load(config, TorchEagerCPURuntime())
    tokenizer = cast(FixedTokenizer, loaded.tokenizer)
    input_ids, attention_mask = tokenizer.batch_encode(("alpha route", "beta expert"))
    batch = ModelBatch(
        sample_ids=("pt-route-alpha", "pt-route-beta"),
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    artifact = capture_routing_dataset(adapter, loaded, (batch,), config)
    assert artifact.raw_traces
    return artifact.raw_traces


def _expert_counts(events: tuple[RoutingEvent, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        observed = max(event.selected_expert_ids) + 1
        if event.router_probabilities is not None:
            observed = max(observed, len(event.router_probabilities))
        counts[event.layer_id] = max(counts.get(event.layer_id, 0), observed)
    return counts


def _record_trial(
    mode: RoutingMode,
    events: tuple[RoutingEvent, ...],
    expert_counts: dict[str, int],
) -> tuple[int, RoutingArtifact]:
    sink = InMemoryRoutingSink(mode, expert_counts=expert_counts)
    started = time.perf_counter_ns()
    for event in events:
        sink.record(event)
    elapsed = time.perf_counter_ns() - started
    return elapsed, sink.close()


def test_pt_001_tiny_moe_routing_instrumentation_overhead() -> None:
    """PT-001: aggregate bookkeeping stays inside its per-event development budget.

    ``capture_routing_dataset(mode="off")`` intentionally returns without executing
    inference, so the three modes cannot be timed as equivalent end-to-end calls.
    This test therefore measures the narrow storage-mode surface after normalized
    events have been emitted by the real tiny-MoE fixture. The aggregate threshold
    is a median absolute bookkeeping budget; full-trace timing is measured only as
    informational evidence, as required by TDD PT-001. This narrow test does not
    replace or weaken SPEC NFR-007's 15% representative-model end-to-end target.
    """

    source_events = _tiny_moe_events()
    repeats = (_TIMED_EVENT_COUNT + len(source_events) - 1) // len(source_events)
    events = (source_events * repeats)[:_TIMED_EVENT_COUNT]
    expert_counts = _expert_counts(events)
    modes: tuple[RoutingMode, ...] = ("off", "aggregate", "full_trace")

    for mode in modes:
        _record_trial(mode, events, expert_counts)

    timings: dict[RoutingMode, list[int]] = {mode: [] for mode in modes}
    artifacts: dict[RoutingMode, RoutingArtifact] = {}
    for repetition in range(_TIMING_REPETITIONS):
        order = modes if repetition % 2 == 0 else tuple(reversed(modes))
        for mode in order:
            elapsed, artifact = _record_trial(mode, events, expert_counts)
            timings[mode].append(elapsed)
            artifacts[mode] = artifact

    medians = {mode: statistics.median(samples) / len(events) for mode, samples in timings.items()}

    assert artifacts["off"].observed_event_count == 0
    assert artifacts["off"].raw_traces == ()
    assert artifacts["aggregate"].observed_event_count == len(events)
    assert artifacts["aggregate"].raw_traces == ()
    assert artifacts["full_trace"].observed_event_count == len(events)
    assert artifacts["full_trace"].recorded_event_count == len(events)
    assert medians["aggregate"] <= _AGGREGATE_EVENT_BUDGET_NS, medians
    # Full-trace overhead is intentionally informational until TDD establishes a budget.


@pytest.mark.no_cover
def test_nfr_007_aggregate_capture_end_to_end_overhead_is_below_15_percent() -> None:
    """NFR-007: real aggregate capture clears the representative CPU fixture target.

    Each timed direct trial executes the same eight 16x64 loss batches. Each aggregate
    trial calls the real dataset capture boundary, which creates/closes one sink and
    attaches/removes hooks once around those eight batches. Alternating trial order
    reduces drift bias; warm-ups are excluded and the ratio uses the two medians.
    Pytest-cov still executes this test but pauses tracing for its call: branch tracing
    disproportionately taxes the aggregate path's Python hooks and sink bookkeeping
    relative to the direct path's PyTorch kernels, invalidating the latency ratio.
    """

    previous_threads = torch.get_num_threads()
    garbage_collection_was_enabled = gc.isenabled()
    torch.set_num_threads(1)
    gc.disable()
    try:
        config = ExperimentConfig(
            name="nfr-007-routing-overhead",
            model=ModelConfig(
                model_id="local://fixtures/tiny-moe",
                revision="fixture-v1",
                checkpoint_format="fixture",
            ),
            routing=RoutingConfig(mode="aggregate", required=True),
            benchmark=BenchmarkConfig(enabled=False),
            reporting=ReportingConfig(markdown=False, html=False, plots=False),
        )
        adapter = LocalFixtureAdapter()
        loaded = adapter.load(config, TorchEagerCPURuntime())
        tokenizer = cast(FixedTokenizer, loaded.tokenizer)
        input_ids = torch.arange(
            _NFR_007_BATCH_SIZE * _NFR_007_SEQUENCE_LENGTH,
            dtype=torch.long,
        ).reshape(_NFR_007_BATCH_SIZE, _NFR_007_SEQUENCE_LENGTH)
        input_ids %= tokenizer.vocab_size
        attention_mask = torch.ones_like(input_ids)
        batches = tuple(
            ModelBatch(
                sample_ids=tuple(
                    f"nfr-007-batch-{batch_index}-sample-{sample_index}"
                    for sample_index in range(_NFR_007_BATCH_SIZE)
                ),
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            for batch_index in range(_NFR_007_DATASET_BATCHES)
        )

        def direct_dataset() -> None:
            for batch in batches:
                adapter.forward_loss(loaded, batch)

        def aggregate_dataset() -> RoutingArtifact:
            return capture_routing_dataset(adapter, loaded, batches, config)

        for _ in range(_NFR_007_WARMUPS):
            direct_dataset()
            aggregate_dataset()

        timings: dict[str, list[int]] = {"direct": [], "aggregate": []}
        last_artifact: RoutingArtifact | None = None
        for repetition in range(_NFR_007_REPETITIONS):
            order = ("direct", "aggregate") if repetition % 2 == 0 else ("aggregate", "direct")
            for mode in order:
                started = time.perf_counter_ns()
                if mode == "direct":
                    direct_dataset()
                else:
                    last_artifact = aggregate_dataset()
                timings[mode].append(time.perf_counter_ns() - started)

        direct_median_ns = statistics.median(timings["direct"])
        aggregate_median_ns = statistics.median(timings["aggregate"])
        overhead_ratio = aggregate_median_ns / direct_median_ns - 1.0
        expected_events = (
            _NFR_007_DATASET_BATCHES * _NFR_007_BATCH_SIZE * _NFR_007_SEQUENCE_LENGTH * 2
        )

        assert last_artifact is not None
        assert last_artifact.observed_event_count == expected_events
        assert last_artifact.recorded_event_count == 0
        assert overhead_ratio < _NFR_007_AGGREGATE_OVERHEAD_LIMIT, {
            "batch_shape": (_NFR_007_BATCH_SIZE, _NFR_007_SEQUENCE_LENGTH),
            "dataset_batches": _NFR_007_DATASET_BATCHES,
            "direct_median_ms": direct_median_ns / 1_000_000,
            "aggregate_median_ms": aggregate_median_ns / 1_000_000,
            "aggregate_overhead_percent": overhead_ratio * 100,
        }
    finally:
        if garbage_collection_was_enabled:
            gc.enable()
        torch.set_num_threads(previous_threads)


def _retained_trace_bytes(event_count: int) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        sink = InMemoryRoutingSink("full_trace", expert_counts={"moe_0": 4})
        for token_position in range(event_count):
            sink.record(
                RoutingEvent(
                    sample_id="pt-memory-sample",
                    sequence_index=0,
                    token_position=token_position,
                    layer_id="moe_0",
                    selected_expert_ids=(token_position % 4, (token_position + 1) % 4),
                    selected_weights=(0.75, 0.25),
                    router_probabilities=(0.4, 0.3, 0.2, 0.1),
                )
            )
        artifact = sink.close()
        retained_bytes, _peak_bytes = tracemalloc.get_traced_memory()
        assert artifact.recorded_event_count == event_count
        return retained_bytes
    finally:
        tracemalloc.stop()


def test_pt_002_full_trace_memory_growth_is_linear_and_bounded() -> None:
    """PT-002: retained event objects and trace containers scale approximately linearly."""

    _retained_trace_bytes(8)  # Populate one-time interpreter and source-line caches.
    baseline = _retained_trace_bytes(0)
    retained = tuple(_retained_trace_bytes(count) for count in _TRACE_EVENT_COUNTS)
    adjusted = tuple(max(1, value - baseline) for value in retained)
    slopes = tuple(
        used_bytes / event_count
        for used_bytes, event_count in zip(adjusted, _TRACE_EVENT_COUNTS, strict=True)
    )

    assert max(slopes) <= _TRACE_BYTES_PER_EVENT_BOUND, slopes
    assert max(slopes) / min(slopes) <= _TRACE_SLOPE_SPREAD_BOUND, slopes
    assert retained[-1] <= _TRACE_TOTAL_BYTES_BOUND, retained


def _synthetic_summary(index: int) -> NormalizedRunSummary:
    run_id = "baseline" if index == 0 else f"candidate-{index:03d}"
    return NormalizedRunSummary(
        run_id=run_id,
        artifact_path=f"artifacts/{run_id}",
        model_id="fixture://pt-scale-tiny-moe",
        model_revision="fixture-v1",
        datasets=(
            DatasetIdentity(
                dataset_id="fixture://pt-scale-evaluation",
                dataset_revision="fixture-data-v1",
                split="evaluation",
                dataset_sha256="a" * 64,
            ),
        ),
        seed_set=(17,),
        sample_ids=("pt-sample-a", "pt-sample-b"),
        prompt_template_hash="a" * 64,
        decode_config={"do_sample": False, "max_new_tokens": 4},
        routing_dataset=DatasetIdentity(
            dataset_id="fixture://pt-scale-routing",
            dataset_revision="fixture-data-v1",
            split="routing",
            dataset_sha256="b" * 64,
        ),
        routing_sample_ids=("pt-route-a", "pt-route-b"),
        routing_capture=RoutingCaptureIdentity(
            configured_mode="full_trace",
            captured_mode="full_trace",
            observed_event_count=4,
            recorded_event_count=4,
            alignment_key_count=4,
            alignment_key_sha256="c" * 64,
        ),
        benchmark_protocol_version="cpu-v1",
        hardware_environment={
            "hardware": {"device": "cpu"},
            "runtime": {"backend": "torch_eager_cpu", "device": "cpu"},
        },
        metrics={
            "quality": MetricValue(
                value=0.8 + index / 1_000,
                category="quality",
                direction="maximize",
            ),
            "serialized_size_bytes": MetricValue(
                value=1_000_000.0 + index * 1_000,
                unit="bytes",
                category="resource",
                direction="minimize",
            ),
            "latency_ms": MetricValue(
                value=10.0 + index / 100,
                unit="ms",
                category="resource",
                direction="minimize",
            ),
            "throughput_tokens_per_second": MetricValue(
                value=20.0 + index / 10,
                unit="tokens/s",
                category="resource",
                direction="maximize",
            ),
            "route_agreement": MetricValue(
                value=0.95 - index / 10_000,
                category="routing",
                direction="maximize",
            ),
        },
        environment={"python": "3.11", "hardware": {"device": "cpu"}},
        resolved_config={"seed": 17, "candidate_index": index},
        quantization_policy={"default_precision": "int8"},
    )


def _report_data(candidate_count: int) -> ReportData:
    runs = tuple(_synthetic_summary(index) for index in range(candidate_count + 1))
    baseline = runs[0]
    comparisons = tuple(compare_summaries(baseline, candidate) for candidate in runs[1:])
    objectives = (
        ParetoObjective(metric="quality", direction="maximize"),
        ParetoObjective(metric="serialized_size_bytes", direction="minimize"),
    )
    pareto = pareto_frontier(pareto_points_from_summaries(runs, objectives), objectives)
    return ReportData(
        runs=runs,
        comparisons=comparisons,
        pareto=pareto,
        interpretations=("Synthetic scale input; no model-quality claim is made.",),
        reproduction_commands=("uv run pytest tests/integration/test_performance_scale.py",),
        metadata={"synthetic_candidate_count": candidate_count},
    )


@dataclass(frozen=True, slots=True)
class _ReportMeasurement:
    data: ReportData
    artifacts: ReportArtifacts
    elapsed_seconds: float
    peak_bytes: int


def _measure_report(candidate_count: int, destination: Path) -> _ReportMeasurement:
    gc.collect()
    tracemalloc.start()
    try:
        started = time.perf_counter()
        data = _report_data(candidate_count)
        artifacts = generate_report(data, destination)
        elapsed = time.perf_counter() - started
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        return _ReportMeasurement(data, artifacts, elapsed, peak_bytes)
    finally:
        tracemalloc.stop()


def test_pt_003_report_generation_scales_to_one_hundred_candidates(tmp_path: Path) -> None:
    """PT-003: complete reports handle 100 candidates without quadratic retained memory.

    The synthetic quality/size values form a trade-off, so the documented O(n²)
    Pareto comparison executes while its persisted membership payload remains O(n).
    Doubling candidates from 50 to 100 must stay below a threefold peak-memory
    increase, below the fourfold signature of quadratic retained state.
    """

    smaller = _measure_report(_SCALE_CANDIDATE_COUNT // 2, tmp_path / "fifty")
    measured = _measure_report(_SCALE_CANDIDATE_COUNT, tmp_path / "one-hundred")

    assert len(measured.data.runs) == _SCALE_CANDIDATE_COUNT + 1
    assert len(measured.data.comparisons) == _SCALE_CANDIDATE_COUNT
    assert measured.data.pareto is not None
    assert len(measured.data.pareto.memberships) == _SCALE_CANDIDATE_COUNT + 1
    assert measured.elapsed_seconds <= _REPORT_COMPLETION_BUDGET_SECONDS, measured.elapsed_seconds
    assert measured.peak_bytes <= _REPORT_PEAK_MEMORY_BOUND, measured.peak_bytes
    assert measured.peak_bytes / smaller.peak_bytes <= _REPORT_DOUBLING_MEMORY_RATIO_BOUND, (
        smaller.peak_bytes,
        measured.peak_bytes,
    )

    payload = json.loads(measured.artifacts.machine_readable_path.read_text(encoding="utf-8"))
    markdown = measured.artifacts.markdown_path.read_text(encoding="utf-8")
    assert len(payload["runs"]) == _SCALE_CANDIDATE_COUNT + 1
    assert "candidate-100" in markdown
    assert len(measured.artifacts.plot_paths) == 3
    assert all(path.is_file() for path in measured.artifacts.plot_source_paths)
