"""End-to-end evidence for independently scoped benchmark-stage peak RSS."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from inkling_quant_lab.config import load_config
from inkling_quant_lab.manifests import RunStatus, load_manifest
from inkling_quant_lab.pipeline.resume import verify_successful_stage_outputs
from inkling_quant_lab.pipeline.runner import run_experiment

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_baseline_and_candidate_benchmarks_use_distinct_peak_rss_workers(
    tmp_path: Path,
) -> None:
    config = load_config(
        PROJECT_ROOT / "configs/experiments/tiny_moe_int8.yaml",
        (
            f"output.root={json.dumps(str(tmp_path / 'artifacts'))}",
            'benchmark.protocol_version="cpu-isolated-stage-peak-v1"',
            "benchmark.warmup_iterations=0",
            "benchmark.repetitions=1",
            'benchmark.host_memory_mode="isolated_stage_worker_peak_rss"',
            "benchmark.worker_timeout_seconds=60.0",
        ),
    )
    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="isolated-stage-peak",
    )

    manifest = load_manifest(run_directory)
    assert manifest.status is RunStatus.SUCCESS
    verify_successful_stage_outputs(run_directory, manifest)

    results = {
        kind: _read_json(run_directory / f"metrics/benchmark_{kind}/benchmark.json")
        for kind in ("baseline", "candidate")
    }
    parent_pid = os.getpid()
    worker_pids: set[int] = set()
    for kind, result in results.items():
        memory = result["peak_memory"]
        assert memory["host_measurement_kind"] == "benchmark_stage_worker_process_peak_rss"
        assert memory["host_process_isolated"] is True
        assert memory["host_available"] is True
        assert memory["host_bytes"] > 0
        assert memory["host_worker_pid"] != parent_pid
        assert "prior stages" in memory["host_scope"]
        assert "not steady-state-only" in memory["host_scope"]
        assert "post-read runtime cleanup" in memory["host_scope"]
        assert "result serialization/IPC" in memory["host_scope"]
        assert memory["host_baseline_bytes"] is None
        assert memory["host_sample_count"] is None
        assert all(
            memory["host_bytes"] >= trial["host_memory_bytes_at_post_trial_sample"]
            for trial in result["trials"]
        )
        worker_pids.add(memory["host_worker_pid"])
        expected_load_kind = "cold_model_load" if kind == "baseline" else "candidate_reconstruction"
        assert result["model_load_time_kind"] == expected_load_kind
    assert len(worker_pids) == 2

    for kind in ("baseline", "candidate"):
        summary = _read_json(run_directory / f"reports/{kind}_summary.json")
        metrics = summary["metrics"]
        peak = metrics["peak_host_memory_bytes"]
        stage_peak = metrics["benchmark_stage_worker_process_peak_rss_bytes"]
        worker_pid = metrics["benchmark_stage_worker_pid"]
        assert peak["status"] == "available"
        assert stage_peak["status"] == "available"
        assert peak["value"] == stage_peak["value"]
        assert worker_pid["status"] == "available"
        assert int(worker_pid["value"]) in worker_pids
