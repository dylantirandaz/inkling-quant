"""Contracts for process-isolated benchmark peak-RSS evidence."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inkling_quant_lab.benchmarking.energy import EnergyMeasurement
from inkling_quant_lab.benchmarking.latency import (
    BenchmarkResult,
    BenchmarkTrial,
    BenchmarkWorkloadProvenance,
    DistributionStatistics,
    LatencyStatistics,
)
from inkling_quant_lab.benchmarking.memory import (
    PeakMemoryMeasurement,
    process_peak_rss_bytes,
)
from inkling_quant_lab.benchmarking.utilization import HardwareUtilizationMeasurement
from inkling_quant_lab.config import ExperimentConfig, load_config
from inkling_quant_lab.exceptions import BenchmarkError
from inkling_quant_lab.pipeline.benchmark_worker import (
    _execute_isolated_request,
    run_isolated_benchmark,
)
from inkling_quant_lab.pipeline.operations import (
    build_candidate,
    collect_statistics,
    create_components,
    load_baseline,
    resolve_policy,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _distribution(value: float) -> DistributionStatistics:
    return DistributionStatistics(
        sample_count=1,
        median=value,
        p10=value,
        p90=value,
        mean=value,
        stdev=0.0,
    )


def _mock_benchmark_result(*, load_kind: str) -> BenchmarkResult:
    distribution = _distribution(1.0)
    return BenchmarkResult(
        protocol_version="source-weight-free-worker-test-v1",
        model_id="ggml-org/stories15M_MOE",
        model_revision="b6dd737497465570b5f5e962dbc9d9454ed1e0eb",
        model_checksum="a" * 64,
        workload=BenchmarkWorkloadProvenance(
            dataset_id="local://fixtures/generation-prompts",
            dataset_revision="fixture-data-v1",
            dataset_sha256="b" * 64,
            split="benchmark",
            sample_ids=("sample-1",),
            seed=20260716,
            prompt_template_hash="c" * 64,
            decode_config={"do_sample": False, "max_new_tokens": 1},
        ),
        model_load_time_ms=1.0,
        model_load_time_kind=load_kind,
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
            ),
        ),
        latency=LatencyStatistics(
            time_to_first_token_ms=distribution,
            inter_token_latency_ms=None,
            end_to_end_ms=distribution,
        ),
        throughput_tokens_per_second=_distribution(1_000.0),
        peak_memory=PeakMemoryMeasurement(
            host_bytes=None,
            device_bytes=None,
            host_available=False,
            device_available=False,
        ),
        serialized_size_bytes=1,
        hardware_utilization=HardwareUtilizationMeasurement(
            status="unavailable", reason="test sensor unavailable"
        ),
        energy=EnergyMeasurement(status="unavailable", reason="test sensor unavailable"),
    )


def _source_weight_free_config() -> ExperimentConfig:
    return load_config(
        PROJECT_ROOT / "configs/experiments/hf_stories15m_native_int8_source_weight_free_peak.yaml"
    )


class _CleanupRuntime:
    def __init__(self) -> None:
        self.cleaned = False

    def cleanup(self) -> None:
        self.cleaned = True


class _SourceWeightFreeAdapter:
    def load_empty_export_shell(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("mock helper should intercept source-weight-free shell loading")


def _fast_config(
    config_factory: Callable[..., ExperimentConfig], tmp_path: Path
) -> ExperimentConfig:
    config = config_factory(model_id="local://fixtures/tiny-dense", routing_mode="off")
    return config.model_copy(
        update={
            "benchmark": config.benchmark.model_copy(
                update={
                    "protocol_version": "isolated-worker-v1",
                    "warmup_iterations": 0,
                    "repetitions": 1,
                    "host_memory_mode": "isolated_stage_worker_peak_rss",
                }
            ),
            "output": config.output.model_copy(update={"root": str(tmp_path / "artifacts")}),
            "reporting": config.reporting.model_copy(
                update={"markdown": False, "html": False, "plots": False}
            ),
        }
    )


def _persist_candidate_inputs(config: ExperimentConfig, run_directory: Path) -> None:
    components = create_components(config)
    try:
        baseline = load_baseline(config, components)
        inventory = tuple(components.adapter.enumerate_modules(baseline))
        statistics = collect_statistics(config, components, baseline, inventory)
        policy = resolve_policy(config, components, baseline, inventory, statistics)
        candidate = build_candidate(config, components, baseline, policy, statistics)

        statistics_path = run_directory / "metrics/statistics/statistics.json"
        policy_path = run_directory / "checkpoints/policy/resolved_policy.json"
        manifest_path = run_directory / "checkpoints/candidate/quantization_manifest.json"
        for path in (statistics_path, policy_path, manifest_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        statistics_path.write_text(
            json.dumps(statistics.as_dict(), sort_keys=True), encoding="utf-8"
        )
        policy_path.write_text(policy.model_dump_json(), encoding="utf-8")
        manifest_path.write_text(candidate.manifest.model_dump_json(), encoding="utf-8")
    finally:
        components.runtime.cleanup()


def test_process_peak_rss_reads_current_process_high_water_mark() -> None:
    assert process_peak_rss_bytes() > 0


def test_process_peak_rss_parses_linux_vmhwm_in_kibibytes(tmp_path: Path) -> None:
    status = tmp_path / "status"
    status.write_text(
        "Name:\tpython\nVmPeak:\t999999 kB\nVmHWM:\t12345 kB\nVmRSS:\t2345 kB\n",
        encoding="ascii",
    )

    assert process_peak_rss_bytes(system="Linux", linux_status_path=status) == 12_345 * 1024


@pytest.mark.parametrize(
    "contents",
    (
        "Name:\tpython\nVmRSS:\t123 kB\n",
        "Name:\tpython\nVmHWM:\tnot-a-number kB\n",
        "Name:\tpython\nVmHWM:\t0 kB\n",
        "Name:\tpython\nVmHWM:\t123 bytes\n",
    ),
)
def test_process_peak_rss_rejects_invalid_linux_status(tmp_path: Path, contents: str) -> None:
    status = tmp_path / "status"
    status.write_text(contents, encoding="ascii")

    with pytest.raises((OSError, ValueError)):
        process_peak_rss_bytes(system="Linux", linux_status_path=status)


def test_process_peak_rss_rejects_unsupported_platform(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="unsupported"):
        process_peak_rss_bytes(system="Plan9", linux_status_path=tmp_path / "unused")


def test_baseline_and_candidate_use_distinct_isolated_workers(
    config_factory: Callable[..., ExperimentConfig], tmp_path: Path
) -> None:
    config = _fast_config(config_factory, tmp_path)
    project_root = Path(__file__).resolve().parents[2]
    baseline_directory = tmp_path / "baseline"
    candidate_directory = tmp_path / "candidate"
    baseline_directory.mkdir()
    candidate_directory.mkdir()
    _persist_candidate_inputs(config, candidate_directory)

    baseline = run_isolated_benchmark(
        config,
        baseline_directory,
        candidate=False,
        project_root=project_root,
        timeout_seconds=30.0,
    )
    candidate = run_isolated_benchmark(
        config,
        candidate_directory,
        candidate=True,
        project_root=project_root,
        timeout_seconds=30.0,
    )

    parent_pid = os.getpid()
    baseline_memory = baseline.peak_memory
    candidate_memory = candidate.peak_memory
    assert baseline_memory.host_worker_pid not in {None, parent_pid}
    assert candidate_memory.host_worker_pid not in {None, parent_pid}
    assert baseline_memory.host_worker_pid != candidate_memory.host_worker_pid
    for result in (baseline, candidate):
        memory = result.peak_memory
        assert result.repetitions == 1
        assert len(result.trials) == 1
        assert memory.host_available is True
        assert memory.host_bytes is not None and memory.host_bytes > 0
        assert memory.host_measurement_kind == "benchmark_stage_worker_process_peak_rss"
        assert memory.host_process_isolated is True
        assert memory.host_scope is not None
        assert "warm" in memory.host_scope
        assert "trial" in memory.host_scope
        assert "post-read runtime cleanup" in memory.host_scope
        assert "result serialization/IPC" in memory.host_scope
    assert "model load" in (baseline_memory.host_scope or "")
    assert "candidate reconstruction" in (candidate_memory.host_scope or "")


def test_isolated_worker_failure_is_typed_and_does_not_hang(
    config_factory: Callable[..., ExperimentConfig], tmp_path: Path
) -> None:
    config = _fast_config(config_factory, tmp_path)
    invalid = config.model_copy(
        update={
            "model": config.model.model_copy(update={"model_id": "local://fixtures/not-a-model"})
        }
    )
    run_directory = tmp_path / "failed"
    run_directory.mkdir()

    with pytest.raises(BenchmarkError, match="worker"):
        run_isolated_benchmark(
            invalid,
            run_directory,
            candidate=False,
            project_root=Path(__file__).resolve().parents[2],
            timeout_seconds=30.0,
        )


def test_subject_artifact_candidate_worker_never_loads_or_rebuilds_source_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inkling_quant_lab.pipeline.benchmark_worker as worker

    config = _source_weight_free_config()
    runtime = _CleanupRuntime()
    components = SimpleNamespace(
        adapter=_SourceWeightFreeAdapter(), runtime=runtime, quantizer=object()
    )
    candidate_model = object()
    provenance = {
        "schema_version": "source-weight-free-reload-v1",
        "source_weights_loaded": False,
    }
    quantized = SimpleNamespace(
        loaded=candidate_model,
        manifest=SimpleNamespace(serialized_size_bytes=123_456),
    )
    observed: dict[str, Any] = {}

    monkeypatch.setattr(worker.os, "chdir", lambda _path: None)
    monkeypatch.setattr(worker, "create_components", lambda _config: components)
    monkeypatch.setattr(worker, "probe_capabilities", lambda *_args: None)
    monkeypatch.setattr(
        worker,
        "load_governed_candidate_artifact",
        lambda *args: (
            quantized,
            SimpleNamespace(as_dict=lambda: provenance),
        ),
    )
    monkeypatch.setattr(
        worker,
        "load_baseline",
        lambda *_args: pytest.fail("candidate worker loaded float source weights"),
    )
    monkeypatch.setattr(
        worker,
        "build_candidate",
        lambda *_args: pytest.fail("candidate worker reconstructed the candidate"),
    )

    def fake_benchmark(
        _config: ExperimentConfig,
        _components: Any,
        model: object,
        *,
        serialized_size_bytes: int | None,
    ) -> BenchmarkResult:
        observed.update(model=model, serialized_size_bytes=serialized_size_bytes)
        return _mock_benchmark_result(load_kind="candidate_source_weight_free_export_load")

    monkeypatch.setattr(worker, "benchmark_model", fake_benchmark)
    monkeypatch.setattr(worker, "process_peak_rss_bytes", lambda: 654_321)
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    result = _execute_isolated_request(
        config,
        run_directory,
        candidate=True,
        project_root=tmp_path,
    )

    assert observed == {
        "model": candidate_model,
        "serialized_size_bytes": 123_456,
    }
    assert result.source_weight_free_load_provenance == provenance
    assert result.peak_memory.host_bytes == 654_321
    assert (
        result.peak_memory.host_measurement_kind
        == "benchmark_subject_artifact_worker_process_peak_rss"
    )
    assert runtime.cleaned is True


def test_subject_artifact_baseline_worker_uses_governed_size_without_candidate_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inkling_quant_lab.pipeline.benchmark_worker as worker

    config = _source_weight_free_config()
    runtime = _CleanupRuntime()
    components = SimpleNamespace(
        adapter=_SourceWeightFreeAdapter(), runtime=runtime, quantizer=object()
    )
    descriptor = object()
    baseline_model = SimpleNamespace(descriptor=descriptor)
    observed: dict[str, Any] = {}

    monkeypatch.setattr(worker.os, "chdir", lambda _path: None)
    monkeypatch.setattr(worker, "create_components", lambda _config: components)
    monkeypatch.setattr(worker, "probe_capabilities", lambda *_args: None)
    monkeypatch.setattr(
        worker,
        "load_verified_baseline_descriptor",
        lambda *_args: SimpleNamespace(
            descriptor=descriptor,
            serialized_size_bytes=987_654,
        ),
    )
    monkeypatch.setattr(worker, "load_baseline", lambda *_args: baseline_model)
    monkeypatch.setattr(
        worker,
        "load_governed_candidate_artifact",
        lambda *_args: pytest.fail("baseline worker loaded the candidate artifact"),
    )
    monkeypatch.setattr(
        worker,
        "build_candidate",
        lambda *_args: pytest.fail("baseline worker reconstructed the candidate"),
    )

    def fake_benchmark(
        _config: ExperimentConfig,
        _components: Any,
        model: object,
        *,
        serialized_size_bytes: int | None,
    ) -> BenchmarkResult:
        observed.update(model=model, serialized_size_bytes=serialized_size_bytes)
        return _mock_benchmark_result(load_kind="cold_model_load")

    monkeypatch.setattr(worker, "benchmark_model", fake_benchmark)
    monkeypatch.setattr(worker, "process_peak_rss_bytes", lambda: 765_432)
    run_directory = tmp_path / "run"
    run_directory.mkdir()

    result = _execute_isolated_request(
        config,
        run_directory,
        candidate=False,
        project_root=tmp_path,
    )

    assert observed == {
        "model": baseline_model,
        "serialized_size_bytes": 987_654,
    }
    assert result.source_weight_free_load_provenance is None
    assert result.peak_memory.host_bytes == 765_432
    assert runtime.cleaned is True
