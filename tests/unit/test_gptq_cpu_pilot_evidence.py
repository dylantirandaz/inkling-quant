"""Contracts for the curated public Stories15M GPTQModel CPU pilot evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, cast

import pytest

from inkling_quant_lab.hardware import project_source_provenance

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "docs/experiments/stories15m-gptq-cpu-pilot.json"
EVIDENCE_SHA256 = "c6ca320bd8c793818fed97da6263ae3f5a3d36bd87b4a7c28ae45cbf0e88eb35"
CANONICAL_RUN_ID = "hf-stories15m-gptq-cpu-pilot-20260717T083523430551Z-cd45e8d4-032a76"
SOURCE_BOUND_RUN_IDS = (
    "hf-stories15m-gptq-cpu-pilot-20260717T081224283945Z-cd45e8d4-ef8dda",
    "hf-stories15m-gptq-cpu-pilot-20260717T080806076188Z-cd45e8d4-3fc140",
)
ADAPTER_ISOLATED_RUN_ID = "hf-stories15m-gptq-cpu-pilot-20260717T075858584773Z-cd45e8d4-89c2db"
EARLIER_CLEAN_RUN_ID = "hf-stories15m-gptq-cpu-pilot-20260717T075034920658Z-058c709a-5b8c99"
SECONDARY_RUN_ID = "hf-stories15m-gptq-cpu-pilot-20260717T073445204598Z-058c709a-f89df0"
RAW_ARTIFACT_VALIDATION_ENV = "IQL_VALIDATE_RAW_GPTQ_PILOT_ARTIFACTS"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record() -> dict[str, Any]:
    raw = EVIDENCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EVIDENCE_SHA256
    assert b"/Users/" not in raw
    assert b"dylantirandaz" not in raw
    assert b"Once upon" not in raw
    return cast(dict[str, Any], json.loads(raw))


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)


def _json_records(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, list)
    assert all(isinstance(item, dict) for item in value)
    return cast(list[dict[str, Any]], value)


def test_evidence_binds_clean_canonical_run_model_software_and_policy() -> None:
    record = _record()
    run = record["canonical_run"]
    model = record["model"]
    software = record["software_and_hardware"]
    provenance = record["project_source_provenance"]
    calibration = record["calibration"]
    policy = record["policy_and_realization"]
    state = record["state_integrity"]
    reload = record["export_and_reload"]

    assert record["schema_version"] == "curated-gptqmodel-cpu-public-moe-pilot-v1"
    assert record["status"] == "success"
    assert run["run_id"] == CANONICAL_RUN_ID
    assert run["run_kind"] == "fresh_uninterrupted_adapter_isolated_source_bound"
    assert run["config_hash"] == (
        "cd45e8d47f4b8ab13a3bafba965989c644d887e6484695f26df4d2724fd6e9b5"
    )
    assert run["model_adapter"] == "hf_causal_lm_linear_mixtral"
    assert run["adapter_isolation_explicit"] is True
    assert run["deterministic_project_source_manifest_recorded"] is True
    assert run["required_stage_count"] == run["successful_required_stage_count"] == 11
    assert run["required_stage_attempts_are_all_one"] is True
    assert run["failed_stage_count"] == 0
    assert set(run["skipped_not_required_stages"]) == {
        "benchmark_baseline",
        "benchmark_candidate",
        "compare_routing",
    }

    assert model["model_id"] == "ggml-org/stories15M_MOE"
    assert model["revision"] == "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
    assert model["loaded_baseline_float32_state_sha256"] == (
        "368fc0265ea7f6b86ef2103ab33e5d929c592521b029f024100d7910b576ed51"
    )
    assert model["source_checkpoint"] == {
        "repository_filename": "model.safetensors",
        "sha256": "dbfa0289f68a8dd721d10eb12d8bd82e098455682027f6f9986ba548913f9082",
        "size_bytes": 72_744_704,
        "tensor_count": 117,
        "dtype_counts": {"F16": 111, "F32": 6},
    }
    assert model["local_files_only"] is True
    assert model["trust_remote_code"] is False
    assert model["pickle_weights_allowed"] is False

    expected_matrix = {
        "accelerate": "1.14.0",
        "defuser": "0.0.23",
        "gptqmodel": "5.8.0",
        "huggingface-hub": "1.23.0",
        "kernels": "0.14.1",
        "safetensors": "0.8.0",
        "torch": "2.13.0",
        "transformers": "5.12.1",
    }
    assert software["exact_gptq_cpu_packages"] == expected_matrix
    matrix_payload = json.dumps(expected_matrix, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(matrix_payload.encode()).hexdigest() == software["software_matrix_sha256"]
    assert software["execution_device"] == "cpu"
    assert software["available_devices"] == ["cpu", "mps"]
    assert software["cuda_available"] is False
    assert software["cuda_device_count"] == 0
    assert software["mps_built"] is True
    assert software["mps_available_to_run"] is True

    assert provenance["kind"] == "filesystem_sha256_manifest_v1"
    assert provenance["tree_sha256"] == (
        "b5099bafdf044eef327d5ad6b41db732a9fba7c0b600f45d6abf344eaec714f0"
    )
    assert provenance["file_count"] == 94
    assert provenance["first_file"] == {
        "path": "pyproject.toml",
        "sha256": "eaebb09534ff40fd6401af4a6769f5d6a8a95f8e811a66218ec62b40777370d6",
        "size_bytes": 3272,
    }
    assert provenance["last_file"] == {
        "path": "uv.lock",
        "sha256": "4e2d482b5be0fb7e7479eb5f668cc431798630cfbaf34bd21f2fa49dbfc382d8",
        "size_bytes": 661_141,
    }
    assert provenance["symlinks_permitted"] is False
    assert provenance["importable_sourceless_bytecode_permitted"] is False
    assert provenance["recorded_environment_manifest_equals_current_project_source_manifest"]
    assert provenance["full_file_manifest_persisted_in_environment"] is True
    assert provenance["git_provenance_available"] is False

    assert calibration["sample_count"] == 4
    assert calibration["token_count"] == 74
    assert calibration["sample_ids"] == [
        "gptq-calibration-001",
        "gptq-calibration-002",
        "gptq-calibration-003",
        "gptq-calibration-004",
    ]
    assert calibration["disjoint_from_evaluation_by_dataset_and_sample_id"] is True

    assert policy["resolved_assignment_count"] == 118
    assert policy["int4_assignment_count"] == policy["selected_module_count"] == 24
    assert policy["float32_assignment_count"] == policy["runtime_exclusion_count"] == 94
    assert policy["expert_projection_modules_quantized"] == 0
    assert policy["expert_projection_modules_retained_float32"] == 72
    assert policy["router_modules_retained_float32"] == 6
    expected_selected = {
        f"model.layers.{layer}.self_attn.{projection}_proj"
        for layer in range(6)
        for projection in ("q", "k", "v", "o")
    }
    selected_digest = hashlib.sha256(
        "\n".join(sorted(expected_selected)).encode("utf-8")
    ).hexdigest()
    assert selected_digest == policy["selected_module_names_sha256"]
    assert policy["quantization"] == {
        "backend": "gptq",
        "backend_version": "5.8.0",
        "execution_backend": "torch",
        "bits": 4,
        "group_size": 32,
        "symmetric": True,
        "desc_act": False,
        "act_group_aware": True,
        "damp_percent": 0.1,
        "batch_size": 1,
        "python_packing": True,
    }

    assert state["candidate_state_sha256"] == (
        "89bd85450717005aad377ec08ba313ad004da4ebd44d0ecd3c6bae9fd0c2cca7"
    )
    assert state["excluded_state_sha256"] == (
        "1340b3deb923033cbefc9cae30203893b6e0193106c921cef58a552334858231"
    )
    assert state["excluded_state_verified_equal_to_source"] is True
    assert state["candidate_state_sha256"] != reload["weight_file_sha256"]
    assert state["runtime_qzero_conversion_count"] == state["quantized_module_count"] == 24
    assert state["runtime_qzero_format"] == "gptq_v2"
    assert state["export_config_sha256"] == (
        "4da4c2b5f5b96586c8be675d32548d8b14e6e9147c0ffa27f9ebcc1f46eba3ca"
    )
    assert state["export_config_size_bytes"] == 7_899
    assert reload["checkpoint_qzero_format"] == "gptq_v1"
    assert reload["runtime_qzero_format"] == "gptq_v2"
    assert reload["config_file"] == "config.json"
    assert reload["config_file_sha256"] == state["export_config_sha256"]
    assert reload["config_file_size_bytes"] == state["export_config_size_bytes"]
    assert reload["config_binding_matches_quantization_manifest_reload_recipe_and_actual_file"]
    assert reload["exported_bundle_file_count"] == 9
    assert reload["exported_bundle_size_bytes"] == 142_322_660
    assert reload["strict_state_assignment"] is True
    assert reload["strict_behavior_affecting_config_binding"] is True
    assert reload["candidate_evaluation_used_persisted_export_reload"] is True
    assert reload["successful_strict_export_reload_count_in_canonical_run"] == 1
    assert reload["reloaded_state_sha256_matches_quantized_candidate"] is True
    assert reload["reloaded_excluded_state_sha256_matches_source"] is True


def test_evidence_preserves_negative_quality_and_serialized_size_results() -> None:
    record = _record()
    quality = record["pilot_quality"]
    size = record["serialized_size_result"]

    baseline = quality["baseline"]
    candidate = quality["candidate"]
    delta = quality["candidate_minus_baseline"]
    assert quality["sample_count"] == quality["successful_sample_count"] == 4
    assert quality["evaluated_token_count"] == 17
    assert math.isclose(
        candidate["mean_nll"] - baseline["mean_nll"],
        delta["mean_nll"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        candidate["perplexity"] - baseline["perplexity"],
        delta["perplexity"],
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    assert math.isclose(
        delta["perplexity"] / baseline["perplexity"],
        delta["perplexity_relative_delta"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert candidate["mean_nll"] > baseline["mean_nll"]
    assert candidate["perplexity"] > baseline["perplexity"]
    assert quality["observed_direction"] == "candidate_worse"
    assert quality["representative_quality_dataset"] is False
    assert quality["noninferiority_test_performed"] is False
    assert quality["quality_preservation_claim_supported"] is False

    assert (
        size["candidate_weight_file_size_bytes"] - size["source_weight_file_size_bytes"]
        == size["candidate_minus_source_weight_file_bytes"]
        == 65_926_528
    )
    assert math.isclose(
        size["candidate_minus_source_weight_file_bytes"] / size["source_weight_file_size_bytes"],
        size["candidate_relative_size_delta"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert (
        size["candidate_bundle_size_bytes"] - size["source_weight_file_size_bytes"]
        == size["candidate_bundle_minus_source_weight_file_bytes"]
        == 69_577_956
    )
    assert math.isclose(
        size["candidate_bundle_minus_source_weight_file_bytes"]
        / size["source_weight_file_size_bytes"],
        size["candidate_bundle_relative_size_delta"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert size["serialized_size_reduction_observed"] is False
    assert size["observed_direction"] == "candidate_larger"


def test_evidence_keeps_history_and_unavailable_claims_explicit() -> None:
    record = _record()
    source_bound = record["superseded_source_bound_validation_history"]
    adapter_isolated = record["earlier_adapter_isolated_validation_history"]
    earlier = record["earlier_clean_pre_adapter_isolation_history"]
    history = record["secondary_failed_reload_and_resume_history"]
    unavailable = record["unavailable_measurements_and_claims"]
    claims = record["claims"]

    assert tuple(item["run_id"] for item in source_bound) == SOURCE_BOUND_RUN_IDS
    assert [item["source_tree_sha256"] for item in source_bound] == [
        "304def974cb56a9567893a324c871bad4afa219ce55016758a84df669de6dad7",
        "2f0b91408e5e837ed3d63c19da7b11a1bbdec7d587fbbe4ff8c204a95a6103a8",
    ]
    assert all(item["canonical_claim_source"] is False for item in source_bound)
    assert all(item["source_file_count"] == 93 for item in source_bound)
    assert all(item["strict_exported_config_binding"] is False for item in source_bound)
    assert all(item["required_stage_attempts_are_all_one"] is True for item in source_bound)
    assert all(item["scientific_outputs_match_canonical"] is True for item in source_bound)

    assert adapter_isolated["run_id"] == ADAPTER_ISOLATED_RUN_ID
    assert adapter_isolated["canonical_claim_source"] is False
    assert (
        adapter_isolated["status"]
        == "clean_adapter_isolated_validation_superseded_by_source_binding"
    )
    assert adapter_isolated["model_adapter"] == "hf_causal_lm_linear_mixtral"
    assert adapter_isolated["adapter_isolation_explicit"] is True
    assert adapter_isolated["deterministic_project_source_manifest_recorded"] is False
    assert (
        adapter_isolated["required_stage_count"]
        == adapter_isolated["successful_required_stage_count"]
        == 11
    )
    assert adapter_isolated["required_stage_attempts_are_all_one"] is True
    assert adapter_isolated["failed_stage_count"] == 0
    assert all(adapter_isolated["scientific_outputs_match_canonical"].values())

    assert earlier["run_id"] == EARLIER_CLEAN_RUN_ID
    assert earlier["canonical_claim_source"] is False
    assert earlier["status"] == "clean_validation_superseded_by_adapter_isolation"
    assert earlier["model_adapter"] == "hf_causal_lm"
    assert earlier["required_stage_count"] == earlier["successful_required_stage_count"] == 11
    assert earlier["required_stage_attempts_are_all_one"] is True
    assert earlier["failed_stage_count"] == 0
    assert all(earlier["scientific_outputs_match_canonical"].values())

    assert history["run_id"] == SECONDARY_RUN_ID
    assert history["canonical_claim_source"] is False
    assert history["status"] == "diagnostic_only"
    assert history["quantize_stage_attempt"] == 1
    assert history["candidate_evaluation_attempts"] == 2
    assert history["first_reload"]["code"] == "QUANTIZATION_ERROR"
    assert history["first_reload"]["component"] == "gptq"
    assert history["first_reload"]["message"] == (
        "CPU GPTQ reload did not restore global compatibility bindings"
    )
    assert (
        history["resumed_reload"]["candidate_evaluation_sha256"]
        == record["artifact_bindings"]["candidate_evaluation"]["sha256"]
    )

    assert unavailable["routing"]["status"] == "not_measured"
    assert unavailable["benchmark"]["status"] == "not_measured"
    assert unavailable["energy"]["status"] == "unavailable"
    assert unavailable["cuda"]["status"] == "unavailable_on_run_host"
    assert unavailable["fp8"]["status"] == "not_measured"
    assert unavailable["awq"]["status"] == "not_measured"
    assert unavailable["awq"]["awq_pack_shim_applied"] is False

    true_claims = (
        "real_public_moe_checkpoint_loaded_exactly",
        "real_gptqmodel_5_8_cpu_conversion_executed",
        "fresh_uninterrupted_governed_run_completed",
        "explicit_linear_mixtral_adapter_isolation",
        "deterministic_project_source_manifest_recorded",
        "current_project_source_matches_recorded_manifest",
        "strict_persisted_export_reload_executed",
        "strict_behavior_affecting_export_config_bound",
        "reloaded_candidate_evaluated",
        "exact_attention_int4_policy_realized",
        "protected_float32_state_preserved",
    )
    false_claims = (
        "serialized_size_reduction_observed",
        "representative_quality_preservation",
        "expert_weights_quantized",
        "routing_drift_measured",
        "benchmark_executed",
        "latency_or_throughput_claim",
        "peak_memory_claim",
        "energy_joules_claim",
        "cuda_execution_claim",
        "fp8_execution_claim",
        "awq_execution_claim",
        "universal_gptq_support_claim",
        "clean_checkout_provenance",
    )
    assert all(claims[name] is True for name in true_claims)
    assert all(claims[name] is False for name in false_claims)


def test_checked_in_inputs_match_curated_bindings() -> None:
    record = _record()
    for binding in record["configuration"]["checked_in_files"].values():
        path = PROJECT_ROOT / binding["path"]
        assert _sha256(path) == binding["sha256"]
        assert path.stat().st_size == binding["size_bytes"]
    calibration = record["calibration"]
    calibration_path = PROJECT_ROOT / calibration["fixture_path"]
    assert _sha256(calibration_path) == calibration["fixture_sha256"]
    assert calibration_path.stat().st_size == calibration["fixture_size_bytes"]


def test_raw_artifacts_match_curated_bindings_when_explicitly_requested() -> None:
    if os.environ.get(RAW_ARTIFACT_VALIDATION_ENV) != "1":
        pytest.skip(f"set {RAW_ARTIFACT_VALIDATION_ENV}=1 to validate ignored historical artifacts")

    record = _record()
    run_root = PROJECT_ROOT / record["canonical_run"]["artifact_root"]
    assert run_root.is_dir(), f"raw governed GPTQ artifact root is missing: {run_root}"

    for binding in record["artifact_bindings"].values():
        path = PROJECT_ROOT / binding["path"]
        assert path.is_file()
        assert _sha256(path) == binding["sha256"]
        assert path.stat().st_size == binding["size_bytes"]

    environment = _json_object(run_root / "environment.json")
    recorded_source = cast(dict[str, Any], environment["project_source"])
    assert project_source_provenance(PROJECT_ROOT) == recorded_source
    source_files = cast(list[dict[str, Any]], recorded_source["files"])
    source_paths = [str(item["path"]) for item in source_files]
    assert len(source_files) == recorded_source["file_count"] == 94
    assert source_paths == sorted(source_paths)
    assert len(source_paths) == len(set(source_paths))
    canonical_source = json.dumps(
        source_files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    assert (
        hashlib.sha256(b"inkling-quant-project-source-v1\0" + canonical_source).hexdigest()
        == recorded_source["tree_sha256"]
    )

    manifest = _json_object(run_root / "manifest.json")
    assert manifest["run_id"] == CANONICAL_RUN_ID
    assert manifest["status"] == "success"
    assert manifest["environment"]["project_source"] == recorded_source
    assert manifest["environment"]["hardware"]["device"] == "cpu"
    assert manifest["environment"]["hardware"]["available_devices"] == ["cpu", "mps"]
    required = [stage for stage in manifest["stages"].values() if stage["required"]]
    assert len(required) == 11
    assert all(stage["status"] == "success" and stage["attempt"] == 1 for stage in required)
    assert manifest["stages"]["evaluate_candidate"]["attempt"] == 1

    quantization = _json_object(run_root / "checkpoints/candidate/quantization_manifest.json")
    policy = _json_object(run_root / "checkpoints/policy/resolved_policy.json")
    recipe = _json_object(run_root / "checkpoints/candidate/candidate/inkling_quant_reload.json")
    assignments = cast(dict[str, dict[str, Any]], policy["assignments"])
    assert len(assignments) == 118
    assert sum(value["precision"] == "int4" for value in assignments.values()) == 24
    assert sum(value["precision"] == "float32" for value in assignments.values()) == 94
    assert (
        quantization["quantization_parameters"]["candidate_state_sha256"]
        == record["state_integrity"]["candidate_state_sha256"]
    )
    assert (
        quantization["quantization_parameters"]["excluded_state_sha256"]
        == record["state_integrity"]["excluded_state_sha256"]
    )
    assert recipe["candidate_state_sha256"] == record["state_integrity"]["candidate_state_sha256"]
    assert recipe["weight_file_sha256"] == record["export_and_reload"]["weight_file_sha256"]
    config_path = run_root / "checkpoints/candidate/candidate/config.json"
    config_digest = _sha256(config_path)
    config_size = config_path.stat().st_size
    quantization_parameters = quantization["quantization_parameters"]
    assert config_digest == quantization_parameters["export_config_sha256"]
    assert config_size == quantization_parameters["export_config_size_bytes"]
    assert config_digest == recipe["config_file_sha256"]
    assert config_size == recipe["config_file_size_bytes"]
    assert recipe["config_file"] == config_path.name
    assert config_digest == record["state_integrity"]["export_config_sha256"]
    assert config_size == record["state_integrity"]["export_config_size_bytes"]
    assert config_digest == record["export_and_reload"]["config_file_sha256"]
    assert config_size == record["export_and_reload"]["config_file_size_bytes"]
    assert recipe["strict_state_assignment"] is True
    assert recipe["validation_required_on_reload"] is True
    assert recipe["local_files_only"] is True
    assert recipe["trust_remote_code"] is False

    export_root = config_path.parent
    embedded_manifest = export_root / "inkling_quant_manifest.json"
    assert (
        embedded_manifest.read_bytes()
        == (run_root / "checkpoints/candidate/quantization_manifest.json").read_bytes()
    )
    exported_files = [path for path in export_root.iterdir() if path.is_file()]
    assert len(exported_files) == record["export_and_reload"]["exported_bundle_file_count"] == 9
    assert (
        sum(path.stat().st_size for path in exported_files)
        == record["export_and_reload"]["exported_bundle_size_bytes"]
        == 142_322_660
    )

    baseline = _json_records(run_root / "metrics/evaluation_baseline/results.json")[0]
    candidate = _json_records(run_root / "metrics/evaluation_candidate/results.json")[0]
    assert baseline["metrics"] == {
        "evaluated_tokens": 17,
        "mean_nll": record["pilot_quality"]["baseline"]["mean_nll"],
        "perplexity": record["pilot_quality"]["baseline"]["perplexity"],
        "successful_samples": 4,
    }
    assert candidate["metrics"] == {
        "evaluated_tokens": 17,
        "mean_nll": record["pilot_quality"]["candidate"]["mean_nll"],
        "perplexity": record["pilot_quality"]["candidate"]["perplexity"],
        "successful_samples": 4,
    }


def test_adapter_isolated_validation_artifacts_match_when_present() -> None:
    record = _record()
    prior = record["earlier_adapter_isolated_validation_history"]
    run_root = PROJECT_ROOT / prior["artifact_root"]
    if not run_root.exists():
        pytest.skip("prior clean validation artifacts are not required in source archives")

    relative_paths = {
        "resolved_config": "resolved_config.yaml",
        "environment": "environment.json",
        "manifest": "manifest.json",
        "status": "status.json",
        "completion": "completion.json",
        "events": "events.jsonl",
        "quantization_manifest": "checkpoints/candidate/quantization_manifest.json",
        "exported_weights": "checkpoints/candidate/candidate/model.safetensors",
        "baseline_evaluation": "metrics/evaluation_baseline/results.json",
        "candidate_evaluation": "metrics/evaluation_candidate/results.json",
    }
    for name, relative in relative_paths.items():
        binding = prior["artifact_bindings"][name]
        path = run_root / relative
        assert _sha256(path) == binding["sha256"]
        assert path.stat().st_size == binding["size_bytes"]

    manifest = _json_object(run_root / "manifest.json")
    assert manifest["run_id"] == ADAPTER_ISOLATED_RUN_ID
    assert manifest["status"] == "success"
    required = [stage for stage in manifest["stages"].values() if stage["required"]]
    assert len(required) == 11
    assert all(stage["status"] == "success" and stage["attempt"] == 1 for stage in required)
    resolved = (run_root / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "  adapter: hf_causal_lm_linear_mixtral\n" in resolved
    assert "project_source" not in _json_object(run_root / "environment.json")


def test_earlier_generic_adapter_validation_artifacts_match_when_present() -> None:
    record = _record()
    earlier = record["earlier_clean_pre_adapter_isolation_history"]
    run_root = PROJECT_ROOT / earlier["artifact_root"]
    if not run_root.exists():
        pytest.skip("earlier clean validation artifacts are not required in source archives")

    relative_paths = {
        "resolved_config": "resolved_config.yaml",
        "manifest": "manifest.json",
        "status": "status.json",
        "completion": "completion.json",
        "events": "events.jsonl",
        "quantization_manifest": "checkpoints/candidate/quantization_manifest.json",
        "exported_weights": "checkpoints/candidate/candidate/model.safetensors",
        "baseline_evaluation": "metrics/evaluation_baseline/results.json",
        "candidate_evaluation": "metrics/evaluation_candidate/results.json",
    }
    for name, relative in relative_paths.items():
        binding = earlier["artifact_bindings"][name]
        path = run_root / relative
        assert _sha256(path) == binding["sha256"]
        assert path.stat().st_size == binding["size_bytes"]

    manifest = _json_object(run_root / "manifest.json")
    assert manifest["run_id"] == EARLIER_CLEAN_RUN_ID
    assert manifest["status"] == "success"
    required = [stage for stage in manifest["stages"].values() if stage["required"]]
    assert len(required) == 11
    assert all(stage["status"] == "success" and stage["attempt"] == 1 for stage in required)
    resolved = (run_root / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "  adapter: hf_causal_lm\n" in resolved
    assert "hf_causal_lm_linear_mixtral" not in resolved


def test_superseded_source_bound_artifacts_match_when_present() -> None:
    record = _record()
    history = record["superseded_source_bound_validation_history"]
    for entry in history:
        run_root = PROJECT_ROOT / entry["artifact_root"]
        if not run_root.exists():
            pytest.skip("superseded source-bound artifacts are not required in source archives")

        digests = entry["artifact_sha256"]
        expected_paths = {
            "environment": "environment.json",
            "manifest": "manifest.json",
            "status": "status.json",
            "quantization_manifest": "checkpoints/candidate/quantization_manifest.json",
            "reload_recipe": "checkpoints/candidate/candidate/inkling_quant_reload.json",
        }
        for name, relative in expected_paths.items():
            assert _sha256(run_root / relative) == digests[name]

        environment = _json_object(run_root / "environment.json")
        source = environment["project_source"]
        assert source["tree_sha256"] == entry["source_tree_sha256"]
        assert source["file_count"] == entry["source_file_count"] == 93
        manifest = _json_object(run_root / "manifest.json")
        assert manifest["run_id"] == entry["run_id"]
        assert manifest["status"] == "success"
        assert manifest["environment"]["project_source"] == source
        required = [stage for stage in manifest["stages"].values() if stage["required"]]
        assert len(required) == 11
        assert all(stage["status"] == "success" and stage["attempt"] == 1 for stage in required)
        quantization = _json_object(run_root / "checkpoints/candidate/quantization_manifest.json")
        recipe = _json_object(
            run_root / "checkpoints/candidate/candidate/inkling_quant_reload.json"
        )
        assert "export_config_sha256" not in quantization["quantization_parameters"]
        assert "config_file_sha256" not in recipe


def test_secondary_failed_reload_artifacts_match_when_present() -> None:
    record = _record()
    history = record["secondary_failed_reload_and_resume_history"]
    run_root = PROJECT_ROOT / history["artifact_root"]
    if not run_root.exists():
        pytest.skip("secondary diagnostic run artifacts are not required in source archives")

    expected = {
        "manifest.json": history["artifact_bindings"]["manifest_sha256"],
        "status.json": history["artifact_bindings"]["status_sha256"],
        "events.jsonl": history["artifact_bindings"]["events_sha256"],
        "failures/evaluate_candidate-attempt-1.json": history["first_reload"][
            "failure_artifact_sha256"
        ],
    }
    for relative, digest in expected.items():
        assert _sha256(run_root / relative) == digest

    failure = _json_object(run_root / "failures/evaluate_candidate-attempt-1.json")
    assert failure["code"] == history["first_reload"]["code"]
    assert failure["component"] == history["first_reload"]["component"]
    assert failure["message"] == history["first_reload"]["message"]
    manifest = _json_object(run_root / "manifest.json")
    assert manifest["stages"]["quantize"]["attempt"] == 1
    assert manifest["stages"]["evaluate_candidate"]["attempt"] == 2
