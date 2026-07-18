"""Opt-in governed public-MoE native-INT8 isolated-peak pipeline contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import pytest

from inkling_quant_lab.config import ExperimentConfig, load_config
from inkling_quant_lab.manifests import RunStatus, StageStatus, load_manifest
from inkling_quant_lab.pipeline.resume import (
    verify_successful_stage_fingerprints,
    verify_successful_stage_outputs,
)
from inkling_quant_lab.pipeline.runner import run_experiment
from inkling_quant_lab.quantization.reference import load_export_recipe

pytestmark = [pytest.mark.integration, pytest.mark.model_public, pytest.mark.slow]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs/experiments/hf_stories15m_native_int8_isolated_peak.yaml"
MODEL_ID = "ggml-org/stories15M_MOE"
MODEL_REVISION = "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
_RUN_PUBLIC_NATIVE_ISOLATED_PEAK = os.environ.get("IQL_RUN_PUBLIC_NATIVE_ISOLATED_PEAK") == "1"
_OPT_IN = pytest.mark.skipif(
    not _RUN_PUBLIC_NATIVE_ISOLATED_PEAK,
    reason=(
        "set IQL_RUN_PUBLIC_NATIVE_ISOLATED_PEAK=1 after caching the exact pinned "
        "Stories15M revision"
    ),
)
_REQUIRED_CACHED_FILES = (
    "config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _read_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return cast(list[dict[str, Any]], value)


def _require_exact_offline_cache(config: ExperimentConfig) -> Path:
    """Require every file consumed by the validated adapter without network access."""

    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        pytest.fail("the opt-in public-model test requires the Hugging Face integration extra")

    cached_paths: list[Path] = []
    for filename in _REQUIRED_CACHED_FILES:
        cached = try_to_load_from_cache(
            config.model.model_id,
            filename,
            revision=config.model.revision,
        )
        if not isinstance(cached, str):
            pytest.fail(
                f"exact pinned cache is missing {filename}; populate it before enabling the test"
            )
        path = Path(cached)
        if not path.is_file():
            pytest.fail(f"cached pinned file is not readable: {filename}")
        cached_paths.append(path)

    snapshot_roots = {path.parent for path in cached_paths}
    assert len(snapshot_roots) == 1
    snapshot = snapshot_roots.pop()
    assert snapshot.name == MODEL_REVISION
    assert not any(
        path.suffix.lower() in {".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle"}
        for path in cached_paths
    )
    return snapshot


def _assert_evaluations_are_aligned(run_directory: Path) -> None:
    baseline = _read_records(run_directory / "metrics/evaluation_baseline/results.json")
    candidate = _read_records(run_directory / "metrics/evaluation_candidate/results.json")
    assert len(baseline) == len(candidate) == 2

    identity_fields = (
        "evaluator_name",
        "evaluator_version",
        "dataset_id",
        "dataset_revision",
        "split",
        "dataset_sha256",
        "sample_ids",
        "sample_count",
        "seed",
        "prompt_template_hash",
        "decode_config",
    )
    for baseline_result, candidate_result in zip(baseline, candidate, strict=True):
        assert baseline_result["status"] == candidate_result["status"] == "success"
        assert baseline_result["failures"] == candidate_result["failures"] == []
        assert baseline_result["sample_count"] == len(baseline_result["sample_ids"])
        assert baseline_result["sample_count"] > 0
        assert {field: baseline_result[field] for field in identity_fields} == {
            field: candidate_result[field] for field in identity_fields
        }


def _assert_routing_is_aligned(run_directory: Path) -> None:
    summaries = {
        kind: _read_object(run_directory / f"reports/{kind}_summary.json")
        for kind in ("baseline", "candidate")
    }
    captures = {kind: summary["routing_capture"] for kind, summary in summaries.items()}
    for capture in captures.values():
        assert capture["configured_mode"] == capture["captured_mode"] == "full_trace"
        assert capture["recorded_event_count"] == capture["observed_event_count"]
        assert capture["alignment_key_count"] == capture["recorded_event_count"]
        assert capture["alignment_key_count"] > 0
        assert len(capture["alignment_key_sha256"]) == 64
    assert (
        captures["baseline"]["alignment_key_sha256"]
        == captures["candidate"]["alignment_key_sha256"]
    )
    assert (
        captures["baseline"]["alignment_key_count"] == captures["candidate"]["alignment_key_count"]
    )

    comparison = _read_object(run_directory / "routing/comparison.json")
    assert comparison["macro"]["layer_count"] == 6
    assert comparison["macro"]["token_weight"] > 0
    assert comparison["macro"]["token_route_agreement"] is not None
    assert comparison["macro"]["top_k_overlap"] is not None
    assert len(comparison["per_layer"]) == 6
    for layer in comparison["per_layer"]:
        assert layer["expert_count"] == 4
        assert layer["aligned_token_count"] > 0
        assert layer["baseline_token_count"] == layer["candidate_token_count"]
        assert layer["baseline_assignment_count"] == layer["candidate_assignment_count"]

    for kind in ("baseline", "candidate"):
        assert (run_directory / f"routing/{kind}/traces.jsonl").is_file()
    for metric in ("route_agreement", "top_k_overlap"):
        assert summaries["candidate"]["metrics"][metric]["status"] == "available"


@_OPT_IN
def test_public_native_int8_pipeline_records_isolated_peak_rss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the exact cached public checkpoint through the governed CPU pipeline."""

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    config = load_config(
        CONFIG,
        (
            f"output.root={json.dumps(str(tmp_path / 'artifacts'))}",
            "benchmark.warmup_iterations=0",
            "benchmark.repetitions=1",
        ),
    )
    assert config.model.model_id == MODEL_ID
    assert config.model.revision == MODEL_REVISION
    assert config.model.local_files_only is True
    _require_exact_offline_cache(config)

    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="public-native-int8-isolated-peak",
    )

    manifest = load_manifest(run_directory)
    assert manifest.status is RunStatus.SUCCESS
    assert len(manifest.stages) == 14
    assert all(stage.status is StageStatus.SUCCESS for stage in manifest.stages.values())
    assert manifest.model.id == MODEL_ID
    assert manifest.model.revision == MODEL_REVISION
    verify_successful_stage_outputs(run_directory, manifest)
    verify_successful_stage_fingerprints(config, manifest)
    assert _read_object(run_directory / "status.json")["status"] == "success"

    quantization = _read_object(run_directory / "checkpoints/candidate/quantization_manifest.json")
    assert quantization["backend"] == "torch_native_dynamic_int8"
    assert quantization["method"] == "native_dynamic_w8a8"
    assert quantization["quantization_parameters"]["kernel"] == "quantized::linear_dynamic"
    assert quantization["quantization_parameters"]["weight_bits"] == 8
    precision_map = cast(dict[str, str], quantization["module_precision_map"])
    selected = {name for name, precision in precision_map.items() if precision == "int8"}
    expected = {
        f"model.layers.{layer}.self_attn.{projection}_proj"
        for layer in range(6)
        for projection in ("q", "k", "v", "o")
    }
    assert len(selected) == 24
    assert selected == expected
    assert set(quantization["excluded_modules"]) == {
        name for name, precision in precision_map.items() if precision != "int8"
    }

    inventory_records = _read_records(run_directory / "metrics/inventory/modules.json")
    inventory = {record["name"]: record for record in inventory_records}
    assert selected <= inventory.keys()
    for name in selected:
        module = inventory[name]
        assert module["class_name"] == "torch.nn.modules.linear.Linear"
        assert "int8" in module["supported_precisions"]
        assert not any(
            module[role] for role in ("is_embedding", "is_expert", "is_output_head", "is_router")
        )

    export = run_directory / "checkpoints/candidate/candidate"
    recipe = load_export_recipe(export)
    assert recipe["schema_version"] == "2.0"
    assert recipe["model"]["model_id"] == MODEL_ID
    assert recipe["model"]["revision"] == MODEL_REVISION
    assert recipe["quantization"] == quantization
    assert recipe["reload"]["format"] == "safetensors"
    assert set(path.name for path in export.iterdir()) == {"metadata.json", "model.safetensors"}
    assert all(path.is_file() and not path.is_symlink() for path in export.iterdir())
    assert quantization["serialized_size_bytes"] == sum(
        path.stat().st_size for path in export.iterdir()
    )

    benchmarks = {
        kind: _read_object(run_directory / f"metrics/benchmark_{kind}/benchmark.json")
        for kind in ("baseline", "candidate")
    }
    parent_pid = os.getpid()
    worker_pids: set[int] = set()
    for kind, benchmark in benchmarks.items():
        assert benchmark["warmup_iterations"] == 0
        assert benchmark["repetitions"] == 1
        assert len(benchmark["trials"]) == 1
        memory = benchmark["peak_memory"]
        assert memory["host_measurement_kind"] == "benchmark_stage_worker_process_peak_rss"
        assert memory["host_process_isolated"] is True
        assert memory["host_available"] is True
        assert memory["host_bytes"] > 0
        assert memory["host_worker_pid"] != parent_pid
        assert (
            memory["host_bytes"] >= benchmark["trials"][0]["host_memory_bytes_at_post_trial_sample"]
        )
        assert "prior stages" in memory["host_scope"]
        assert "not steady-state-only" in memory["host_scope"]
        worker_pids.add(memory["host_worker_pid"])
        expected_load_kind = "cold_model_load" if kind == "baseline" else "candidate_reconstruction"
        assert benchmark["model_load_time_kind"] == expected_load_kind
    assert len(worker_pids) == 2

    for kind, benchmark in benchmarks.items():
        summary = _read_object(run_directory / f"reports/{kind}_summary.json")
        metrics = summary["metrics"]
        assert metrics["peak_host_memory_bytes"]["value"] == benchmark["peak_memory"]["host_bytes"]
        assert (
            int(metrics["benchmark_stage_worker_pid"]["value"])
            == benchmark["peak_memory"]["host_worker_pid"]
        )

    _assert_evaluations_are_aligned(run_directory)
    _assert_routing_is_aligned(run_directory)
