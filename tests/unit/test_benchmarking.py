"""Deterministic CPU benchmark tests using the fake runtime contract."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, cast

import pytest
from pydantic import ValidationError

from inkling_quant_lab.benchmarking import (
    BenchmarkWorkloadProvenance,
    TrialObservation,
    run_generation_benchmark,
    summarize_distribution,
    tokens_per_second,
)
from inkling_quant_lab.benchmarking.energy import (
    EnergyDomainProvenance,
    EnergySensor,
    EnergySensorProvenance,
    EnergySession,
    finish_energy_measurement,
)
from inkling_quant_lab.benchmarking.memory import aggregate_peak_memory, current_process_rss_bytes
from inkling_quant_lab.benchmarking.utilization import (
    UtilizationSample,
    UtilizationSensor,
    begin_hardware_utilization,
    finish_hardware_utilization,
)
from inkling_quant_lab.config import (
    BenchmarkConfig,
    DecodeConfig,
    EvaluationConfig,
    EvaluationSuiteConfig,
    ExperimentConfig,
    ModelConfig,
)
from inkling_quant_lab.data import load_local_dataset
from inkling_quant_lab.evaluation.base import prompt_template_hash
from inkling_quant_lab.exceptions import BenchmarkError
from inkling_quant_lab.models.base import (
    LoadedModel,
    ModelAdapter,
    ModelCapabilities,
    ModelDescriptor,
)
from inkling_quant_lab.models.fixtures import FixedTokenizer
from inkling_quant_lab.pipeline.operations import _benchmark_workload
from inkling_quant_lab.runtimes.base import MemorySnapshot
from inkling_quant_lab.runtimes.fake import FakeRuntime
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.unit


def _loaded_model(*, load_time_seconds: float = 0.25) -> LoadedModel:
    capabilities = ModelCapabilities(
        supports_text=True,
        supports_images=False,
        supports_audio=False,
        is_moe=False,
        supports_router_logits=False,
        supports_token_level_routes=False,
        supported_dtypes=("float32",),
        supported_device_maps=("single",),
        max_context_length=32,
    )
    descriptor = ModelDescriptor(
        model_id="local://benchmark-fixture",
        revision="fixture-v1",
        resolved_class="tests.BenchmarkFixture",
        architecture="benchmark_fixture",
        checksum="abc123",
        capabilities=capabilities,
    )
    return LoadedModel(
        model=object(),
        tokenizer=object(),
        descriptor=descriptor,
        load_time_seconds=load_time_seconds,
    )


def _adapter() -> ModelAdapter:
    return cast(ModelAdapter, object())


def _workload() -> BenchmarkWorkloadProvenance:
    return BenchmarkWorkloadProvenance(
        dataset_id="fixture://benchmark",
        dataset_revision="fixture-v1",
        dataset_sha256="a" * 64,
        split="test",
        sample_ids=("sample-1",),
        seed=7,
        prompt_template_hash="b" * 64,
        decode_config={"max_new_tokens": 3, "do_sample": False},
    )


def _observation(
    *,
    ttft: float = 0.1,
    intervals: tuple[float, ...] = (0.2, 0.3),
) -> TrialObservation:
    return TrialObservation(
        input_tokens=5,
        output_tokens=len(intervals) + 1,
        time_to_first_token_seconds=ttft,
        inter_token_latencies_seconds=intervals,
    )


def test_warmup_is_excluded_and_complete_trial_metrics_are_retained() -> None:
    """TC-BENCH-001: warm-up timing is excluded while memory scope is explicit."""

    runtime = FakeRuntime(
        host_memory_bytes=999,
        device_memory_bytes=999,
        timing_values=[0.0, 100.0, 200.0, 201.0, 300.0, 302.0, 400.0, 404.0],
    )
    observations = iter(
        (
            _observation(ttft=10.0, intervals=(20.0, 30.0)),
            _observation(ttft=0.1, intervals=(0.2, 0.3)),
            _observation(ttft=0.2, intervals=(0.4, 0.5)),
            _observation(ttft=0.4, intervals=(0.5, 0.6)),
        )
    )
    memory = iter(((1000, 2000), (1500, 1800), (1200, 2500)))
    call_index = 0

    def trial(adapter: ModelAdapter, model: LoadedModel) -> TrialObservation:
        nonlocal call_index
        assert adapter is benchmark_adapter
        assert model is loaded
        observation = next(observations)
        if call_index > 0:
            runtime.host_memory_bytes, runtime.device_memory_bytes = next(memory)
        call_index += 1
        return observation

    benchmark_adapter = _adapter()
    loaded = _loaded_model()
    result = run_generation_benchmark(
        benchmark_adapter,
        loaded,
        runtime,
        BenchmarkConfig(warmup_iterations=1, repetitions=3),
        trial,
        serialized_size_bytes=12_345,
        workload=_workload(),
        clock=runtime.clock,
    )

    assert result.model_load_time_ms == 250.0
    assert result.serialized_size_bytes == 12_345
    assert result.workload == _workload()
    assert result.warmup_iterations == 1
    assert len(result.trials) == 3
    assert [item.end_to_end_ms for item in result.trials] == [1000.0, 2000.0, 4000.0]
    assert [item.time_to_first_token_ms for item in result.trials] == [100.0, 200.0, 400.0]
    assert [item.inter_token_latency_ms for item in result.trials] == [250.0, 450.0, 550.0]
    assert [item.tokens_per_second for item in result.trials] == [3.0, 1.5, 0.75]
    assert result.latency.end_to_end_ms.median == 2000.0
    assert result.latency.end_to_end_ms.p10 == pytest.approx(1200.0)
    assert result.latency.end_to_end_ms.p90 == pytest.approx(3600.0)
    assert result.latency.end_to_end_ms.mean == pytest.approx(7000.0 / 3.0)
    assert result.latency.end_to_end_ms.stdev == pytest.approx(math.sqrt(14_000_000.0 / 9.0))
    assert result.peak_memory.host_bytes == 1500
    assert result.peak_memory.device_bytes == 2500
    assert result.peak_memory.host_measurement_kind == "synthetic_test_reading"
    assert result.peak_memory.host_scope == "configured fake-runtime sampling boundary"
    assert result.peak_memory.host_baseline_bytes == 999
    assert result.peak_memory.host_max_observed_delta_bytes == 501
    assert result.peak_memory.host_sample_count == 5
    assert result.peak_memory.measurement_interval_seconds is not None
    assert result.peak_memory.measurement_interval_seconds > 0.0
    assert result.trials[0].host_memory_bytes_at_post_trial_sample == 1000
    assert runtime.events.count("synchronize") == 8
    assert runtime.events.count("memory_snapshot") == 5
    assert result.energy.status == "unavailable"
    assert "disabled" in (result.energy.reason or "")


def test_synchronization_immediately_brackets_each_timed_region() -> None:
    """TC-BENCH-002: synchronize before start-clock and after the trial."""

    runtime = FakeRuntime(host_memory_bytes=100, device_memory_bytes=None)
    clock_values: Iterator[float] = iter((0.0, 1.0))

    def clock() -> float:
        value = next(clock_values)
        runtime.events.append(f"clock:{value:g}")
        return value

    def trial(adapter: ModelAdapter, model: LoadedModel) -> TrialObservation:
        del adapter, model
        runtime.events.append("trial")
        return _observation()

    run_generation_benchmark(
        _adapter(),
        _loaded_model(),
        runtime,
        BenchmarkConfig(warmup_iterations=0, repetitions=1),
        trial,
        serialized_size_bytes=1,
        workload=_workload(),
        clock=clock,
    )

    assert runtime.events == [
        "memory_snapshot",
        "execution_context",
        "synchronize",
        "clock:0",
        "trial",
        "synchronize",
        "clock:1",
        "memory_snapshot",
    ]


def test_post_trial_synchronization_occurs_even_when_trial_raises() -> None:
    runtime = FakeRuntime(timing_values=[0.0])

    def failing_trial(adapter: ModelAdapter, model: LoadedModel) -> TrialObservation:
        del adapter, model
        runtime.events.append("trial")
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected failure"):
        run_generation_benchmark(
            _adapter(),
            _loaded_model(),
            runtime,
            BenchmarkConfig(warmup_iterations=0, repetitions=1),
            failing_trial,
            serialized_size_bytes=1,
            workload=_workload(),
            clock=runtime.clock,
        )

    assert runtime.events == [
        "memory_snapshot",
        "execution_context",
        "synchronize",
        "trial",
        "synchronize",
    ]


def test_known_distribution_statistics() -> None:
    """TC-BENCH-003: use linear percentiles and population standard deviation."""

    result = summarize_distribution((1.0, 2.0, 3.0, 4.0, 5.0))

    assert result.sample_count == 5
    assert result.median == 3.0
    assert result.p10 == pytest.approx(1.4)
    assert result.p90 == pytest.approx(4.6)
    assert result.mean == 3.0
    assert result.stdev == pytest.approx(math.sqrt(2.0))
    assert summarize_distribution((7.0,)).stdev == 0.0


def test_requested_energy_without_sensor_is_explicitly_unavailable() -> None:
    """TC-BENCH-004: an unsupported energy sensor does not fail the benchmark."""

    runtime = FakeRuntime(timing_values=[0.0, 1.0])
    result = run_generation_benchmark(
        _adapter(),
        _loaded_model(),
        runtime,
        BenchmarkConfig(warmup_iterations=0, repetitions=1, measure_energy=True),
        lambda adapter, model: _observation(),
        serialized_size_bytes=1,
        workload=_workload(),
        clock=runtime.clock,
        energy_sensor=None,
    )

    assert result.energy.status == "unavailable"
    assert result.energy.joules is None
    assert result.energy.sensor_name is None
    assert result.energy.reason == "energy sensor unavailable for this runtime"


class _FakeEnergySensor(EnergySensor):
    name = "fake-joules"
    observed_wraparounds = 0
    provenance = EnergySensorProvenance(
        sensor_name=name,
        measurement_kind="synthetic_cumulative_joules",
        scope="configured fake benchmark interval",
        domains=(
            EnergyDomainProvenance(
                domain_id="fake-domain",
                domain_name="fake-domain",
                counter_path="synthetic://fake-domain",
                max_energy_range_uj=1,
            ),
        ),
        counter_wraparound_handling="not applicable to this monotonic fake",
    )

    def __init__(self, readings: tuple[float, ...]) -> None:
        self._readings = iter(readings)

    def read_joules(self) -> float:
        return next(self._readings)


def test_available_energy_records_delta_sensor_and_interval() -> None:
    runtime = FakeRuntime(timing_values=[0.0, 2.0])
    energy_times = iter((10.0, 12.0))
    result = run_generation_benchmark(
        _adapter(),
        _loaded_model(),
        runtime,
        BenchmarkConfig(warmup_iterations=0, repetitions=1, measure_energy=True),
        lambda adapter, model: _observation(ttft=0.2, intervals=(0.3, 0.4)),
        serialized_size_bytes=1,
        workload=_workload(),
        clock=runtime.clock,
        energy_sensor=_FakeEnergySensor((10.0, 12.5)),
        energy_clock=lambda: next(energy_times),
    )

    assert result.energy.status == "available"
    assert result.energy.joules == 2.5
    assert result.energy.sensor_name == "fake-joules"
    assert result.energy.sampling_interval_seconds == 2.0
    assert result.energy.observed_counter_wraparounds == 0
    assert result.energy.provenance == _FakeEnergySensor.provenance


class _FakeUtilizationSensor(UtilizationSensor):
    name = "fake-process-cpu"
    logical_cpu_count = 2

    def __init__(self) -> None:
        self._samples = iter(
            (
                UtilizationSample(wall_seconds=10.0, process_cpu_seconds=3.0),
                UtilizationSample(wall_seconds=12.0, process_cpu_seconds=4.0),
            )
        )

    def sample(self) -> UtilizationSample:
        return next(self._samples)


def test_cpu_hardware_utilization_has_declared_measurement_interval() -> None:
    runtime = FakeRuntime(timing_values=[0.0, 1.0])
    result = run_generation_benchmark(
        _adapter(),
        _loaded_model(),
        runtime,
        BenchmarkConfig(warmup_iterations=0, repetitions=1),
        lambda adapter, model: _observation(),
        serialized_size_bytes=1,
        workload=_workload(),
        clock=runtime.clock,
        utilization_sensor=_FakeUtilizationSensor(),
    )

    assert result.hardware_utilization.status == "available"
    assert result.hardware_utilization.metric == "normalized_process_cpu_percent"
    assert result.hardware_utilization.value_percent == pytest.approx(25.0)
    assert result.hardware_utilization.process_cpu_seconds == pytest.approx(1.0)
    assert result.hardware_utilization.sampling_interval_seconds == pytest.approx(2.0)
    assert result.hardware_utilization.logical_cpu_count == 2
    assert "warm-up and measured trials" in result.hardware_utilization.scope


def test_unavailable_hardware_utilization_never_fabricates_zero() -> None:
    measurement = finish_hardware_utilization(begin_hardware_utilization(None))

    assert measurement.status == "unavailable"
    assert measurement.value_percent is None
    assert measurement.process_cpu_seconds is None
    assert measurement.sampling_interval_seconds is None
    assert measurement.reason


def test_energy_rejects_invalid_or_decreasing_cumulative_readings() -> None:
    runtime = FakeRuntime(timing_values=[0.0, 1.0])
    with pytest.raises(BenchmarkError, match="invalid cumulative reading"):
        run_generation_benchmark(
            _adapter(),
            _loaded_model(),
            runtime,
            BenchmarkConfig(warmup_iterations=0, repetitions=1, measure_energy=True),
            lambda adapter, model: _observation(),
            serialized_size_bytes=1,
            workload=_workload(),
            clock=runtime.clock,
            energy_sensor=_FakeEnergySensor((float("nan"),)),
        )

    runtime.timing_values = [0.0, 1.0]
    with pytest.raises(BenchmarkError, match="decreased"):
        run_generation_benchmark(
            _adapter(),
            _loaded_model(),
            runtime,
            BenchmarkConfig(warmup_iterations=0, repetitions=1, measure_energy=True),
            lambda adapter, model: _observation(),
            serialized_size_bytes=1,
            workload=_workload(),
            clock=runtime.clock,
            energy_sensor=_FakeEnergySensor((2.0, 1.0)),
        )

    with pytest.raises(BenchmarkError, match="sampling interval"):
        finish_energy_measurement(
            EnergySession(None, None, "unavailable"),
            sampling_interval_seconds=-1.0,
        )


def test_single_output_token_keeps_inter_token_latency_unavailable() -> None:
    runtime = FakeRuntime(timing_values=[0.0, 1.0], device_memory_bytes=None)
    result = run_generation_benchmark(
        _adapter(),
        _loaded_model(),
        runtime,
        BenchmarkConfig(warmup_iterations=0, repetitions=1),
        lambda adapter, model: _observation(ttft=0.1, intervals=()),
        serialized_size_bytes=1,
        workload=_workload(),
        clock=runtime.clock,
    )

    assert result.trials[0].output_tokens == 1
    assert result.trials[0].inter_token_latency_ms is None
    assert result.latency.inter_token_latency_ms is None
    assert result.peak_memory.device_available is False
    assert result.peak_memory.device_bytes is None


def test_unsynchronized_configuration_does_not_add_boundaries() -> None:
    runtime = FakeRuntime(timing_values=[0.0, 1.0])
    run_generation_benchmark(
        _adapter(),
        _loaded_model(),
        runtime,
        BenchmarkConfig(warmup_iterations=0, repetitions=1, synchronize=False),
        lambda adapter, model: _observation(),
        serialized_size_bytes=1,
        workload=_workload(),
        clock=runtime.clock,
    )

    assert "synchronize" not in runtime.events


def test_token_timeline_cannot_exceed_end_to_end_measurement() -> None:
    runtime = FakeRuntime(timing_values=[0.0, 0.5])

    with pytest.raises(BenchmarkError, match="token timeline exceeds"):
        run_generation_benchmark(
            _adapter(),
            _loaded_model(),
            runtime,
            BenchmarkConfig(warmup_iterations=0, repetitions=1),
            lambda adapter, model: _observation(ttft=0.2, intervals=(0.2, 0.2)),
            serialized_size_bytes=1,
            workload=_workload(),
            clock=runtime.clock,
        )


def test_non_positive_clock_duration_is_rejected() -> None:
    runtime = FakeRuntime(timing_values=[1.0, 1.0])

    with pytest.raises(BenchmarkError, match="non-positive"):
        run_generation_benchmark(
            _adapter(),
            _loaded_model(),
            runtime,
            BenchmarkConfig(warmup_iterations=0, repetitions=1),
            lambda adapter, model: _observation(),
            serialized_size_bytes=1,
            workload=_workload(),
            clock=runtime.clock,
        )


def test_trial_observation_validates_interval_count_and_values() -> None:
    with pytest.raises(ValidationError, match="output_tokens - 1"):
        TrialObservation(
            input_tokens=1,
            output_tokens=3,
            time_to_first_token_seconds=0.1,
            inter_token_latencies_seconds=(0.2,),
        )
    with pytest.raises(ValidationError, match="finite and non-negative"):
        TrialObservation(
            input_tokens=1,
            output_tokens=2,
            time_to_first_token_seconds=0.1,
            inter_token_latencies_seconds=(-0.2,),
        )


def test_benchmark_workload_requires_exact_nonempty_unique_sample_identity() -> None:
    payload = _workload().model_dump(mode="json")
    payload["sample_ids"] = []
    with pytest.raises(ValidationError, match="non-empty stable sample_ids"):
        BenchmarkWorkloadProvenance.model_validate(payload)
    payload["sample_ids"] = ["same", "same"]
    with pytest.raises(ValidationError, match="must be unique"):
        BenchmarkWorkloadProvenance.model_validate(payload)


def test_benchmark_workload_honors_configured_samples_prompt_and_decode() -> None:
    suite = EvaluationSuiteConfig(
        type="generation_regression",
        dataset="local://fixtures/generation-prompts",
        revision="fixture-data-v1",
        split="evaluation",
        sample_ids=("eval-gen-002", "eval-gen-001"),
        prompt_template="answer {text}",
        decode=DecodeConfig(max_new_tokens=2),
    )
    config = ExperimentConfig(
        name="benchmark-provenance",
        seed=29,
        model=ModelConfig(model_id="local://fixtures/tiny-dense", checkpoint_format="fixture"),
        evaluation=EvaluationConfig(suites=(suite,)),
    )
    model = _loaded_model()
    tokenizer = FixedTokenizer()
    model.tokenizer = tokenizer

    batches, decode, workload = _benchmark_workload(config, model)

    assert tuple(batch.sample_ids[0] for batch in batches) == suite.sample_ids
    assert [tokenizer.decode(cast(Any, batch.input_ids)[0].tolist()) for batch in batches] == [
        "answer red blue",
        "answer alpha beta",
    ]
    assert decode == suite.decode
    dataset = load_local_dataset(suite.dataset, suite.revision, suite.split)
    assert workload.dataset_sha256 == dataset.sha256
    assert workload.sample_ids == suite.sample_ids
    assert workload.seed == 29
    assert workload.prompt_template_hash == prompt_template_hash("answer {text}")
    assert workload.decode_config == suite.decode.model_dump(mode="json")


@pytest.mark.parametrize(
    ("config", "serialized_size", "load_time", "message"),
    (
        (BenchmarkConfig(enabled=False), 1, 0.1, "disabled"),
        (BenchmarkConfig(), -1, 0.1, "serialized checkpoint size"),
        (BenchmarkConfig(), 1, 0.0, "model load time"),
        (BenchmarkConfig(), 1, float("nan"), "model load time"),
    ),
)
def test_benchmark_rejects_invalid_top_level_inputs(
    config: BenchmarkConfig,
    serialized_size: int,
    load_time: float,
    message: str,
) -> None:
    with pytest.raises(BenchmarkError, match=message):
        run_generation_benchmark(
            _adapter(),
            _loaded_model(load_time_seconds=load_time),
            FakeRuntime(),
            config,
            lambda adapter, model: _observation(),
            serialized_size_bytes=serialized_size,
            workload=_workload(),
        )


@pytest.mark.parametrize(
    ("output_tokens", "elapsed_seconds"),
    ((0, 1.0), (1, 0.0), (1, float("inf"))),
)
def test_throughput_rejects_invalid_measurements(
    output_tokens: int, elapsed_seconds: float
) -> None:
    with pytest.raises(BenchmarkError):
        tokens_per_second(output_tokens, elapsed_seconds)


def test_memory_aggregation_rejects_inconsistent_available_value() -> None:
    with pytest.raises(BenchmarkError, match="returned no value"):
        aggregate_peak_memory(
            (
                MemorySnapshot(
                    host_bytes=None,
                    device_bytes=None,
                    host_available=True,
                    device_available=False,
                ),
            )
        )

    with pytest.raises(BenchmarkError, match="negative device memory"):
        aggregate_peak_memory(
            (
                MemorySnapshot(
                    host_bytes=1,
                    device_bytes=-1,
                    host_available=True,
                    device_available=True,
                    host_measurement_kind="synthetic_test_reading",
                    host_scope="test",
                    device_measurement_kind="synthetic_test_reading",
                    device_scope="test",
                ),
            )
        )

    with pytest.raises(BenchmarkError, match="requires measurement kind and scope"):
        aggregate_peak_memory(
            (
                MemorySnapshot(
                    host_bytes=1,
                    device_bytes=None,
                    host_available=True,
                    device_available=False,
                ),
            )
        )


def test_dependency_free_current_rss_reader_and_cpu_snapshot_are_truthfully_labeled() -> None:
    rss = current_process_rss_bytes()
    snapshot = TorchEagerCPURuntime().memory_snapshot()

    assert rss > 0
    assert snapshot.host_available is True
    assert snapshot.host_bytes is not None and snapshot.host_bytes > 0
    assert snapshot.host_measurement_kind == "process_current_rss"
    assert "instantaneous" in (snapshot.host_scope or "")
    with pytest.raises(OSError, match="unsupported"):
        current_process_rss_bytes("UnsupportedOS")


def test_sampled_current_rss_interval_records_baseline_and_never_invents_a_decrease() -> None:
    def snapshot(value: int) -> MemorySnapshot:
        return MemorySnapshot(
            host_bytes=value,
            device_bytes=None,
            host_available=True,
            device_available=False,
            host_measurement_kind="process_current_rss",
            host_scope="instantaneous process RSS",
        )

    measurement = aggregate_peak_memory(
        (snapshot(900), snapshot(800)),
        baseline=snapshot(1_000),
        measurement_interval_seconds=0.25,
    )

    assert measurement.host_bytes == 1_000
    assert measurement.host_baseline_bytes == 1_000
    assert measurement.host_max_observed_delta_bytes == 0
    assert measurement.host_sample_count == 3
    assert measurement.measurement_interval_seconds == 0.25
    assert measurement.host_measurement_kind == "benchmark_interval_sampled_current_rss"
    assert "not a continuous" in (measurement.host_scope or "")


@pytest.mark.parametrize("values", [(), (1.0, float("nan"))])
def test_statistics_reject_empty_or_non_finite_values(values: tuple[float, ...]) -> None:
    with pytest.raises(BenchmarkError):
        summarize_distribution(values)
