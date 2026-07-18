"""Opt-in governed CPU GPTQ contract for the exact cached Stories15M model."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
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
from inkling_quant_lab.pipeline.stages import STAGE_ORDER, stage_definitions
from inkling_quant_lab.public_moe_tensor_parallel import (
    MODEL_ID,
    MODEL_REVISION,
    audit_stories15m_snapshot,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.backend_gptq,
    pytest.mark.model_public,
    pytest.mark.slow,
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs/experiments/hf_stories15m_gptq_cpu_pilot.yaml"
_RUN_GPTQ_CPU = os.environ.get("IQL_RUN_STORIES15M_GPTQ_CPU") == "1"
_OPT_IN = pytest.mark.skipif(
    not _RUN_GPTQ_CPU,
    reason=(
        "set IQL_RUN_STORIES15M_GPTQ_CPU=1 after caching the exact pinned Stories15M "
        "snapshot and installing the exact GPTQ CPU dependency matrix"
    ),
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_QLINEAR_NAMES_SHA256 = "1fc95a1405d0cbbad4abad8c8ed64e61080c939061df15e9ad85e4545a75b176"
_EXACT_DEPENDENCIES = {
    "accelerate": "1.14.0",
    "defuser": "0.0.23",
    "gptqmodel": "5.8.0",
    "huggingface-hub": "1.23.0",
    "kernels": "0.14.1",
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "transformers": "5.12.1",
}
_TEXT_ARTIFACT_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".md", ".txt", ".yaml"})


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _read_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return cast(list[dict[str, Any]], value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _names_sha256(names: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def _require_exact_dependencies() -> None:
    observed: dict[str, str] = {}
    missing: list[str] = []
    for distribution in _EXACT_DEPENDENCIES:
        try:
            observed[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing or observed != _EXACT_DEPENDENCIES:
        pytest.skip(
            "exact GPTQ CPU dependency matrix is unavailable: "
            f"missing={missing}, observed={observed}, required={_EXACT_DEPENDENCIES}"
        )


def _require_exact_snapshot(config: ExperimentConfig) -> Path:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        pytest.skip("the exact GPTQ CPU matrix requires huggingface-hub")
    cached = try_to_load_from_cache(
        config.model.model_id,
        "config.json",
        revision=config.model.revision,
    )
    if not isinstance(cached, str):
        pytest.skip("the exact pinned Stories15M snapshot is not cached")
    snapshot = Path(cached).parent
    try:
        audit_stories15m_snapshot(snapshot)
    except (OSError, RuntimeError, ValueError) as error:
        pytest.skip(f"cached Stories15M snapshot does not satisfy the exact audit: {error}")
    return snapshot


def _assert_evaluation_is_finite(run_directory: Path) -> None:
    baseline = _read_records(run_directory / "metrics/evaluation_baseline/results.json")
    candidate = _read_records(run_directory / "metrics/evaluation_candidate/results.json")
    assert len(baseline) == len(candidate) == 1
    identity_fields = (
        "dataset_id",
        "dataset_revision",
        "dataset_sha256",
        "decode_config",
        "evaluator_name",
        "evaluator_version",
        "prompt_template_hash",
        "sample_count",
        "sample_ids",
        "seed",
        "split",
    )
    assert {field: baseline[0][field] for field in identity_fields} == {
        field: candidate[0][field] for field in identity_fields
    }
    for result in (baseline[0], candidate[0]):
        assert result["status"] == "success"
        assert result["failures"] == []
        metrics = cast(dict[str, Any], result["metrics"])
        assert metrics["evaluated_tokens"] == 17
        assert metrics["successful_samples"] == 4
        assert math.isfinite(float(metrics["mean_nll"]))
        assert math.isfinite(float(metrics["perplexity"]))
        assert float(metrics["mean_nll"]) > 0.0
        assert float(metrics["perplexity"]) > 0.0


def _assert_prompt_privacy(run_directory: Path) -> None:
    fixture_paths = (
        PROJECT_ROOT / "src/inkling_quant_lab/fixture_data/gptq-calibration.jsonl",
        PROJECT_ROOT / "src/inkling_quant_lab/fixture_data/tiny-corpus.jsonl",
    )
    private_texts = {
        cast(str, json.loads(line)["text"])
        for fixture in fixture_paths
        for line in fixture.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for artifact in run_directory.rglob("*"):
        if not artifact.is_file() or artifact.suffix.lower() not in _TEXT_ARTIFACT_SUFFIXES:
            continue
        persisted = artifact.read_text(encoding="utf-8")
        for private_text in private_texts:
            assert private_text not in persisted, artifact

    governed = "\n".join(
        (
            (run_directory / "events.jsonl").read_text(encoding="utf-8"),
            (run_directory / "checkpoints/candidate/quantization_manifest.json").read_text(
                encoding="utf-8"
            ),
            (
                run_directory / "checkpoints/candidate/candidate/inkling_quant_manifest.json"
            ).read_text(encoding="utf-8"),
            (run_directory / "checkpoints/candidate/candidate/inkling_quant_reload.json").read_text(
                encoding="utf-8"
            ),
        )
    )
    assert '"input_ids"' not in governed
    assert '"attention_mask"' not in governed


@_OPT_IN
def test_cached_stories15m_gptq_cpu_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute conversion, strict export reload, and finite candidate evaluation offline."""

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    config = load_config(
        CONFIG,
        (f"output.root={json.dumps(str(tmp_path / 'artifacts'))}",),
    )
    assert config.model.adapter == "hf_causal_lm_linear_mixtral"
    assert config.model.model_id == MODEL_ID
    assert config.model.revision == MODEL_REVISION
    assert config.model.local_files_only is True
    assert config.runtime.backend == "torch_eager_cpu"
    assert config.runtime.device == "cpu"
    assert config.runtime.dtype == "float32"
    assert config.quantization.backend == config.quantization.method == "gptq"
    assert config.quantization.parameters["device"] == "cpu"
    assert config.security.log_prompts is False
    assert config.security.log_model_outputs is False
    _require_exact_dependencies()
    snapshot = _require_exact_snapshot(config)
    assert snapshot.name == MODEL_REVISION

    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="stories15m-gptq-cpu-system",
    )

    manifest = load_manifest(run_directory)
    assert manifest.status is RunStatus.SUCCESS
    assert len(manifest.stages) == len(STAGE_ORDER)
    assert set(manifest.stages) == set(STAGE_ORDER)
    expected_required = {
        definition.name for definition in stage_definitions(config) if definition.required
    }
    assert {name for name, stage in manifest.stages.items() if stage.required} == expected_required
    assert all(manifest.stages[name].status is StageStatus.SUCCESS for name in expected_required)
    assert {
        name
        for name, stage in manifest.stages.items()
        if stage.status is StageStatus.SKIPPED_NOT_REQUIRED
    } == {"benchmark_baseline", "benchmark_candidate", "compare_routing"}
    assert manifest.model.id == MODEL_ID
    assert manifest.model.revision == MODEL_REVISION
    verify_successful_stage_outputs(run_directory, manifest)
    verify_successful_stage_fingerprints(config, manifest)
    assert _read_object(run_directory / "status.json")["status"] == "success"

    resolved_policy = _read_object(run_directory / "checkpoints/policy/resolved_policy.json")
    assignments = cast(dict[str, dict[str, Any]], resolved_policy["assignments"])
    assert len(assignments) == 118
    assert sum(item["precision"] == "int4" for item in assignments.values()) == 24
    assert sum(item["precision"] == "float32" for item in assignments.values()) == 94

    quantization = _read_object(run_directory / "checkpoints/candidate/quantization_manifest.json")
    assert quantization["backend"] == quantization["method"] == "gptq"
    assert quantization["backend_version"] == "5.8.0"
    precision_map = cast(dict[str, str], quantization["module_precision_map"])
    selected = {name for name, precision in precision_map.items() if precision == "int4"}
    expected_selected = {
        f"model.layers.{layer}.self_attn.{projection}_proj"
        for layer in range(6)
        for projection in ("q", "k", "v", "o")
    }
    assert precision_map == {
        name: cast(str, assignment["precision"]) for name, assignment in assignments.items()
    }
    assert len(precision_map) == 118
    assert selected == expected_selected
    assert sum(precision == "float32" for precision in precision_map.values()) == 94
    assert len(quantization["excluded_modules"]) == 94
    assert quantization["calibration_sample_count"] == 4
    assert quantization["calibration_token_count"] == 74
    assert quantization["calibration_sample_ids"] == [
        "gptq-calibration-001",
        "gptq-calibration-002",
        "gptq-calibration-003",
        "gptq-calibration-004",
    ]
    assert _SHA256.fullmatch(cast(str, quantization["calibration_checksum"]))

    parameters = cast(dict[str, Any], quantization["quantization_parameters"])
    assert parameters["quantized_module_count"] == 24
    assert parameters["quantized_module_names_sha256"] == _QLINEAR_NAMES_SHA256
    assert _names_sha256(selected) == _QLINEAR_NAMES_SHA256
    assert parameters["runtime_exclusion_count"] == 94
    assert parameters["excluded_state_tensor_count"] == 95
    assert parameters["excluded_expanded_linear_count"] == 0
    assert parameters["excluded_state_verified_equal_to_source"] is True
    assert parameters["cpu_eager_post_init_module_count"] == 24
    assert parameters["runtime_qzero_conversion_count"] == 24
    assert parameters["runtime_qzero_format"] == "gptq_v2"
    assert {
        distribution: parameters[f"{distribution.replace('-', '_')}_version"]
        for distribution in _EXACT_DEPENDENCIES
    } == _EXACT_DEPENDENCIES
    for key in ("candidate_state_sha256", "excluded_state_sha256", "software_matrix_sha256"):
        assert _SHA256.fullmatch(cast(str, parameters[key]))
    assert parameters["candidate_state_sha256"] != parameters["excluded_state_sha256"]

    export = run_directory / "checkpoints/candidate/candidate"
    recipe_path = export / "inkling_quant_reload.json"
    assert recipe_path.is_file() and not recipe_path.is_symlink()
    recipe = _read_object(recipe_path)
    assert recipe["adapter"] == "inkling_quant_lab_gptqmodel_cpu"
    assert recipe["loader"] == (
        "inkling_quant_lab.quantization.optional.reload_gptqmodel_cpu_export"
    )
    assert recipe["model_id"] == MODEL_ID
    assert recipe["revision"] == MODEL_REVISION
    assert recipe["source_model_checksum"] == quantization["source_model_checksum"]
    assert recipe["source_dtype"] == "float32"
    assert recipe["checkpoint_qzero_format"] == "gptq_v1"
    assert recipe["runtime_qzero_format"] == "gptq_v2"
    assert recipe["selected_module_count"] == 24
    assert recipe["selected_module_names"] == sorted(expected_selected)
    assert recipe["selected_module_names_sha256"] == _QLINEAR_NAMES_SHA256
    assert recipe["excluded_module_count"] == 94
    assert recipe["candidate_state_sha256"] == parameters["candidate_state_sha256"]
    assert recipe["excluded_state_sha256"] == parameters["excluded_state_sha256"]
    assert recipe["software_matrix_sha256"] == parameters["software_matrix_sha256"]
    assert recipe["software_versions"] == _EXACT_DEPENDENCIES
    assert recipe["strict_state_assignment"] is True
    assert recipe["policy_preserving_reload_supported"] is True
    assert recipe["validation_required_on_reload"] is True
    assert recipe["local_files_only"] is True
    assert recipe["trust_remote_code"] is False
    assert recipe["use_safetensors"] is True
    weights = export / cast(str, recipe["weight_file"])
    assert weights.is_file() and not weights.is_symlink()
    assert recipe["weight_file_sha256"] == _file_sha256(weights)
    assert recipe["weight_file_size_bytes"] == weights.stat().st_size
    exported_config = export / cast(str, recipe["config_file"])
    assert exported_config.is_file() and not exported_config.is_symlink()
    assert recipe["config_file_sha256"] == _file_sha256(exported_config)
    assert recipe["config_file_size_bytes"] == exported_config.stat().st_size
    assert parameters["export_config_sha256"] == recipe["config_file_sha256"]
    assert parameters["export_config_size_bytes"] == recipe["config_file_size_bytes"]
    assert _read_object(export / "inkling_quant_manifest.json") == quantization

    _assert_evaluation_is_finite(run_directory)
    candidate_summary = _read_object(run_directory / "reports/candidate_summary.json")
    assert candidate_summary["benchmark_model_load_time_kind"] is None
    load_metric = cast(dict[str, Any], candidate_summary["metrics"])["load_time_ms"]
    assert load_metric["status"] == "unavailable"
    assert load_metric["reason"] == "benchmarking was disabled by the resolved configuration"
    _assert_prompt_privacy(run_directory)
