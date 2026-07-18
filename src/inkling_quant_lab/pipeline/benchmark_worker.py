"""Fresh-process benchmark execution for exact governed stage peak RSS."""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Protocol, cast

import torch

from inkling_quant_lab.benchmarking.latency import BenchmarkResult
from inkling_quant_lab.benchmarking.memory import PeakMemoryMeasurement, process_peak_rss_bytes
from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.exceptions import BenchmarkError
from inkling_quant_lab.models.base import SourceWeightFreeModelAdapter
from inkling_quant_lab.pipeline.candidate_artifact import (
    load_governed_candidate_artifact,
    load_verified_baseline_descriptor,
)
from inkling_quant_lab.pipeline.operations import (
    StatisticsBundle,
    benchmark_model,
    build_candidate,
    create_components,
    load_baseline,
    probe_capabilities,
)
from inkling_quant_lab.quantization.base import QuantizationManifest
from inkling_quant_lab.quantization.policies import ResolvedPrecisionPolicy

_STAGE_PEAK_KIND = "benchmark_stage_worker_process_peak_rss"
_STAGE_PEAK_SCOPE = (
    "OS process high-water RSS through the final governed benchmark boundary, sampled "
    "immediately after measured trials; includes interpreter startup, imports, baseline model "
    "load or candidate reconstruction, serialized-size accounting, warm-up, and measured "
    "trials; excludes the parent pipeline, all prior stages, post-read runtime cleanup, result "
    "serialization/IPC, and process-exit work and is not steady-state-only memory"
)
_SUBJECT_ARTIFACT_PEAK_KIND = "benchmark_subject_artifact_worker_process_peak_rss"
_SUBJECT_ARTIFACT_PEAK_SCOPE = (
    "OS process high-water RSS through the final governed benchmark boundary in a fresh "
    "spawned worker loading exactly one persisted benchmark subject; includes interpreter and "
    "import startup, artifact integrity validation, pinned architecture and tokenizer "
    "construction, baseline source-weight loading or candidate exported-weight loading, native "
    "prepacking, warm-up, and measured trials; the candidate path does not load float source "
    "weights, quantize, or create an export; excludes the parent pipeline, all prior stages, "
    "post-read runtime cleanup, result serialization/IPC, and process-exit work and is an "
    "artifact-load-plus-inference process peak rather than steady-state-only, tensor-attributable, "
    "or final-through-exit residency"
)


class _WorkerProcess(Protocol):
    @property
    def exitcode(self) -> int | None: ...

    def is_alive(self) -> bool: ...

    def join(self, timeout: float | None = None) -> None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


def _load_statistics(run_directory: Path) -> StatisticsBundle:
    path = run_directory / "metrics/statistics/statistics.json"
    if not path.is_file():
        return StatisticsBundle(
            usage={},
            sensitivity={},
            sensitivity_details={},
            calibration_dataset_sha256=None,
            calibration_sample_ids=(),
        )
    data = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    return StatisticsBundle(
        usage={str(key): float(value) for key, value in data["usage"].items()},
        sensitivity={str(key): float(value) for key, value in data["sensitivity"].items()},
        sensitivity_details=cast(
            dict[str, dict[str, float | int | str]], data["sensitivity_details"]
        ),
        calibration_dataset_sha256=cast(str | None, data["calibration_dataset_sha256"]),
        calibration_sample_ids=tuple(str(item) for item in data["calibration_sample_ids"]),
    )


def _isolated_peak_memory(
    result: BenchmarkResult,
    peak_bytes: int,
    *,
    mode: str,
) -> PeakMemoryMeasurement:
    if mode == "isolated_stage_worker_peak_rss":
        kind, scope = _STAGE_PEAK_KIND, _STAGE_PEAK_SCOPE
    elif mode == "isolated_subject_artifact_peak_rss":
        kind, scope = _SUBJECT_ARTIFACT_PEAK_KIND, _SUBJECT_ARTIFACT_PEAK_SCOPE
    else:
        raise BenchmarkError(
            f"unsupported isolated benchmark memory mode: {mode}",
            component="benchmark_worker",
        )
    payload = result.peak_memory.model_dump(mode="json")
    payload.update(
        {
            "host_bytes": peak_bytes,
            "host_available": True,
            "host_measurement_kind": kind,
            "host_scope": scope,
            "host_baseline_bytes": None,
            "host_max_observed_delta_bytes": None,
            "host_sample_count": None,
            "measurement_interval_seconds": None,
            "host_process_isolated": True,
            "host_worker_pid": os.getpid(),
        }
    )
    return PeakMemoryMeasurement.model_validate(payload)


def _execute_isolated_request(
    config: ExperimentConfig,
    run_directory: Path,
    *,
    candidate: bool,
    project_root: Path,
) -> BenchmarkResult:
    if config.benchmark.host_memory_mode not in {
        "isolated_stage_worker_peak_rss",
        "isolated_subject_artifact_peak_rss",
    }:
        raise BenchmarkError(
            "isolated benchmark worker requires a governed isolated host-memory mode",
            component="benchmark_worker",
        )
    if not run_directory.is_dir() or not project_root.is_dir():
        raise BenchmarkError(
            "isolated benchmark worker received a missing run directory or project root",
            component="benchmark_worker",
        )

    os.chdir(project_root)
    torch.manual_seed(config.seed)
    components = create_components(config)
    try:
        probe_capabilities(config, components)
        serialized_size: int | None = None
        source_weight_free_provenance: dict[str, Any] | None = None
        if config.benchmark.host_memory_mode == "isolated_subject_artifact_peak_rss":
            if candidate:
                if not isinstance(components.adapter, SourceWeightFreeModelAdapter):
                    raise BenchmarkError(
                        "configured adapter cannot construct a source-weight-free export shell",
                        component="benchmark_worker",
                    )
                quantized, provenance = load_governed_candidate_artifact(
                    run_directory,
                    config,
                    components.adapter,
                    components.runtime,
                )
                model = quantized.loaded
                serialized_size = quantized.manifest.serialized_size_bytes
                source_weight_free_provenance = provenance.as_dict()
            else:
                persisted = load_verified_baseline_descriptor(run_directory, config)
                model = load_baseline(config, components)
                if model.descriptor != persisted.descriptor:
                    raise BenchmarkError(
                        "loaded baseline identity differs from the governed baseline descriptor",
                        component="benchmark_worker",
                    )
                serialized_size = persisted.serialized_size_bytes
        else:
            baseline = load_baseline(config, components)
            model = baseline
            if candidate:
                policy_path = run_directory / "checkpoints/policy/resolved_policy.json"
                manifest_path = run_directory / "checkpoints/candidate/quantization_manifest.json"
                policy = ResolvedPrecisionPolicy.model_validate_json(
                    policy_path.read_text(encoding="utf-8")
                )
                persisted_manifest = QuantizationManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
                quantized = build_candidate(
                    config,
                    components,
                    baseline,
                    policy,
                    _load_statistics(run_directory),
                )
                if quantized.manifest != persisted_manifest:
                    raise BenchmarkError(
                        "isolated candidate reconstruction differs from the persisted "
                        "quantization manifest",
                        component="benchmark_worker",
                    )
                model = quantized.loaded
                serialized_size = persisted_manifest.serialized_size_bytes

        result = benchmark_model(
            config,
            components,
            model,
            serialized_size_bytes=serialized_size,
        )
        peak_bytes = process_peak_rss_bytes()
        return result.model_copy(
            update={
                "peak_memory": _isolated_peak_memory(
                    result,
                    peak_bytes,
                    mode=config.benchmark.host_memory_mode,
                ),
                "source_weight_free_load_provenance": source_weight_free_provenance,
            }
        )
    finally:
        components.runtime.cleanup()


def _worker_entry(
    sender: Connection,
    config_json: str,
    run_directory: str,
    candidate: bool,
    project_root: str,
) -> None:
    try:
        config = ExperimentConfig.model_validate_json(config_json)
        result = _execute_isolated_request(
            config,
            Path(run_directory).resolve(),
            candidate=candidate,
            project_root=Path(project_root).resolve(),
        )
        sender.send({"status": "success", "result": result.model_dump_json()})
    except Exception as error:
        sender.send(
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
            }
        )
    finally:
        sender.close()


def _terminate_worker(process: _WorkerProcess) -> None:
    if not process.is_alive():
        process.join()
        return
    process.terminate()
    process.join(timeout=5.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=5.0)


def run_isolated_benchmark(
    config: ExperimentConfig,
    run_directory: str | Path,
    *,
    candidate: bool,
    project_root: str | Path,
    timeout_seconds: float | None = None,
) -> BenchmarkResult:
    """Execute exactly one benchmark subject in a new spawn worker and reap it."""

    timeout = (
        config.benchmark.worker_timeout_seconds if timeout_seconds is None else timeout_seconds
    )
    if timeout <= 0.0:
        raise BenchmarkError(
            "isolated benchmark worker timeout must be positive",
            component="benchmark_worker",
        )
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker_entry,
        args=(
            sender,
            config.canonical_json(),
            str(Path(run_directory).resolve()),
            candidate,
            str(Path(project_root).resolve()),
        ),
        name="inkling-quant-benchmark-stage",
    )
    process.start()
    sender.close()
    deadline = time.monotonic() + timeout
    payload: dict[str, Any] | None = None
    try:
        while payload is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _terminate_worker(process)
                raise BenchmarkError(
                    f"isolated benchmark worker exceeded {timeout:g} seconds",
                    component="benchmark_worker",
                    details={"candidate": candidate},
                )
            if receiver.poll(min(0.05, remaining)):
                try:
                    payload = cast(dict[str, Any], receiver.recv())
                except EOFError as error:
                    process.join(timeout=1.0)
                    raise BenchmarkError(
                        "isolated benchmark worker exited without a result",
                        component="benchmark_worker",
                        details={"exit_code": process.exitcode, "candidate": candidate},
                    ) from error
            elif not process.is_alive():
                process.join()
                raise BenchmarkError(
                    "isolated benchmark worker exited without a result",
                    component="benchmark_worker",
                    details={"exit_code": process.exitcode, "candidate": candidate},
                )
        process.join(timeout=5.0)
        if process.is_alive():
            _terminate_worker(process)
            raise BenchmarkError(
                "isolated benchmark worker did not exit after publishing its result",
                component="benchmark_worker",
                details={"candidate": candidate},
            )
        if process.exitcode != 0:
            raise BenchmarkError(
                "isolated benchmark worker exited unsuccessfully",
                component="benchmark_worker",
                details={"exit_code": process.exitcode, "candidate": candidate},
            )
        if payload.get("status") != "success":
            raise BenchmarkError(
                "isolated benchmark worker failed: "
                f"{payload.get('error_type', 'Error')}: {payload.get('message', 'unknown error')}",
                component="benchmark_worker",
                details={"candidate": candidate},
            )
        raw_result = payload.get("result")
        if not isinstance(raw_result, str):
            raise BenchmarkError(
                "isolated benchmark worker returned a malformed result",
                component="benchmark_worker",
            )
        return BenchmarkResult.model_validate_json(raw_result)
    finally:
        receiver.close()
        if process.is_alive():
            _terminate_worker(process)


__all__ = ["run_isolated_benchmark"]
