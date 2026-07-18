"""Truthful normalized reporting for device-memory collector semantics."""

from __future__ import annotations

import pytest

from inkling_quant_lab.benchmarking.energy import EnergyMeasurement
from inkling_quant_lab.benchmarking.latency import (
    BenchmarkResult,
    BenchmarkTrial,
    BenchmarkWorkloadProvenance,
    DistributionStatistics,
    LatencyStatistics,
)
from inkling_quant_lab.benchmarking.memory import PeakMemoryMeasurement
from inkling_quant_lab.benchmarking.utilization import HardwareUtilizationMeasurement
from inkling_quant_lab.pipeline.summaries import _benchmark_metrics

pytestmark = pytest.mark.unit


def _statistics(value: float) -> DistributionStatistics:
    return DistributionStatistics(
        sample_count=1,
        median=value,
        p10=value,
        p90=value,
        mean=value,
        stdev=0.0,
    )


def _benchmark_result(device_measurement_kind: str) -> BenchmarkResult:
    distribution = _statistics(1.0)
    return BenchmarkResult(
        protocol_version="device-memory-semantics-v1",
        model_id="fixture://tiny-moe",
        model_revision="fixture-v1",
        model_checksum="a" * 64,
        workload=BenchmarkWorkloadProvenance(
            dataset_id="fixture://generation",
            dataset_revision="fixture-v1",
            dataset_sha256="b" * 64,
            split="test",
            sample_ids=("sample-1",),
            seed=17,
            prompt_template_hash="c" * 64,
            decode_config={"do_sample": False, "max_new_tokens": 1},
        ),
        model_load_time_ms=1.0,
        model_load_time_kind="cold_model_load",
        warmup_iterations=0,
        repetitions=1,
        trials=(
            BenchmarkTrial(
                index=0,
                input_tokens=1,
                output_tokens=1,
                time_to_first_token_ms=1.0,
                end_to_end_ms=1.0,
                tokens_per_second=1_000.0,
                device_memory_bytes_at_post_trial_sample=4_096,
            ),
        ),
        latency=LatencyStatistics(
            time_to_first_token_ms=distribution,
            inter_token_latency_ms=None,
            end_to_end_ms=distribution,
        ),
        throughput_tokens_per_second=_statistics(1_000.0),
        peak_memory=PeakMemoryMeasurement(
            host_bytes=None,
            device_bytes=4_096,
            host_available=False,
            device_available=True,
            device_measurement_kind=device_measurement_kind,
            device_scope="fixture device-memory collector scope",
        ),
        serialized_size_bytes=1_024,
        hardware_utilization=HardwareUtilizationMeasurement(
            status="unavailable",
            reason="fixture utilization unavailable",
        ),
        energy=EnergyMeasurement(
            status="unavailable",
            reason="fixture energy unavailable",
        ),
    )


@pytest.mark.parametrize(
    "measurement_kind",
    (
        "mps_driver_allocated_memory_at_sample",
        "mlx_allocator_active_bytes_at_sample",
    ),
)
def test_boundary_current_device_memory_is_not_promoted_to_peak(
    measurement_kind: str,
) -> None:
    metrics, _ = _benchmark_metrics(_benchmark_result(measurement_kind))

    peak = metrics["peak_device_memory_bytes"]
    assert peak.status == "unsupported"
    assert peak.value is None
    assert "sampled only" in (peak.reason or "")

    sampled = metrics["benchmark_interval_max_sampled_current_device_memory_bytes"]
    assert sampled.status == "available"
    assert sampled.value == 4_096
    assert sampled.direction == "neutral"


def test_allocator_native_device_peak_retains_peak_metric() -> None:
    metrics, _ = _benchmark_metrics(_benchmark_result("cuda_allocator_peak_since_previous_sample"))

    peak = metrics["peak_device_memory_bytes"]
    assert peak.status == "available"
    assert peak.value == 4_096
    assert peak.direction == "minimize"

    sampled = metrics["benchmark_interval_max_sampled_current_device_memory_bytes"]
    assert sampled.status == "unavailable"
    assert sampled.value is None


def test_unknown_device_collector_is_not_assumed_to_be_a_peak() -> None:
    metrics, _ = _benchmark_metrics(_benchmark_result("unclassified_device_reading"))

    peak = metrics["peak_device_memory_bytes"]
    assert peak.status == "unsupported"
    assert peak.value is None
    assert "not an allocator-native or interval-native peak" in (peak.reason or "")
    assert (
        metrics["benchmark_interval_max_sampled_current_device_memory_bytes"].status
        == "unavailable"
    )


def test_source_weight_free_subject_artifact_worker_is_a_distinct_host_peak() -> None:
    result = _benchmark_result("cuda_allocator_peak_since_previous_sample").model_copy(
        update={
            "peak_memory": PeakMemoryMeasurement(
                host_bytes=512_000_000,
                device_bytes=None,
                host_available=True,
                device_available=False,
                host_measurement_kind="benchmark_subject_artifact_worker_process_peak_rss",
                host_scope="fresh worker loading exactly one persisted benchmark subject",
                host_process_isolated=True,
                host_worker_pid=41_337,
            )
        }
    )

    metrics, _ = _benchmark_metrics(result)

    assert metrics["peak_host_memory_bytes"].status == "available"
    assert metrics["peak_host_memory_bytes"].value == 512_000_000
    subject_peak = metrics["benchmark_subject_artifact_worker_process_peak_rss_bytes"]
    assert subject_peak.status == "available"
    assert subject_peak.value == 512_000_000
    assert subject_peak.direction == "neutral"
    assert metrics["benchmark_subject_artifact_worker_pid"].value == 41_337
    assert metrics["benchmark_stage_worker_process_peak_rss_bytes"].status == "unavailable"
    assert metrics["benchmark_stage_worker_pid"].status == "unavailable"


def test_isolated_host_memory_rejects_an_unrecognized_worker_kind() -> None:
    with pytest.raises(ValueError, match="governed worker"):
        PeakMemoryMeasurement(
            host_bytes=1,
            device_bytes=None,
            host_available=True,
            device_available=False,
            host_measurement_kind="unclassified_worker_peak",
            host_scope="unknown",
            host_process_isolated=True,
            host_worker_pid=1,
        )
