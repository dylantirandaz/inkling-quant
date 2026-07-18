from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "docs/experiments/stories15m-native-int8-source-weight-free-peak-m3.json"
EVIDENCE_SHA256 = "e3ad3763ea733484f4ac5a3f623faa3a9eaa3fb113468364cb7f790b7fc0d4db"


def _record() -> dict[str, Any]:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EVIDENCE_SHA256
    assert b"/Users/" not in payload
    assert b"dylantirandaz" not in payload
    return cast(dict[str, Any], json.loads(payload))


def _assert_sha256(value: str) -> None:
    assert len(value) == 64
    assert int(value, 16) >= 0


def test_source_weight_free_evidence_binds_exact_successful_execution() -> None:
    record = _record()
    run = record["run"]
    model = record["model"]
    quantization = record["quantization"]
    reload = record["source_weight_free_reload"]
    binding = reload["external_manifest_binding"]

    assert record["status"] == "success"
    assert run["config_hash"] == (
        "325b9cd40b5f9efce28268730f09e1fdc0165bd6765f16d8b8918e96a229ce4e"
    )
    assert run["required_stage_count"] == run["successful_stage_count"] == 14
    assert run["manifest_output_count"] == 36
    assert run["manifest_output_hash_verification_passed"] is True
    assert model["model_id"] == "ggml-org/stories15M_MOE"
    assert model["revision"] == "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
    assert model["source_model_checksum"] == (
        "93e36334ff1be21096ca5f59c6b4d8bdfb212c8854b815583110755df75d6ed9"
    )
    assert quantization["backend"] == "torch_native_dynamic_int8"
    assert quantization["kernel"] == "quantized::linear_dynamic"
    assert quantization["quantized_engine"] == "qnnpack"

    selected = set(quantization["selected_modules"])
    expected = {
        f"model.layers.{layer}.self_attn.{projection}_proj"
        for layer in range(6)
        for projection in ("q", "k", "v", "o")
    }
    assert selected == expected
    assert len(selected) == quantization["selected_int8_module_count"] == 24
    assert quantization["float32_assignment_count"] == 46
    assert quantization["fused_expert_slice_count"] == 24
    assert quantization["fused_expert_slices_quantized"] is False

    assert reload["schema_version"] == "source-weight-free-reload-v1"
    assert reload["reload_adapter"] == "hf_causal_lm_source_weight_free_v1"
    assert reload["model_load_time_kind"] == "candidate_source_weight_free_export_load"
    assert reload["source_weights_loaded"] is False
    assert reload["source_metadata_required"] is True
    assert reload["fully_source_free"] is False
    assert reload["self_contained_export"] is False
    assert reload["strict_load"] is reload["assign"] is True
    assert reload["missing_keys"] == reload["unexpected_keys"] == []
    assert reload["meta_tensor_names"] == []
    assert reload["native_wrapper_count"] == 24
    assert reload["tensor_count"] == 81
    assert set(reload["published_export_file_names"]) == {
        "metadata.json",
        "model.safetensors",
    }

    assert binding["quantize_stage_output_count"] == 3
    assert all(
        value is True for key, value in binding.items() if key != "quantize_stage_output_count"
    )
    assert reload["metadata_sha256"] == record["raw_artifact_sha256"]["candidate_metadata.json"]
    assert reload["tensor_sha256"] == record["raw_artifact_sha256"]["candidate_model.safetensors"]

    expected_source_metadata = {
        "config.json": "e901e012953d1df93574b2cc3d7db5ed4758d52f8bd4a7dd4b647936e32261be",
        "special_tokens_map.json": (
            "ff3b4a612c4e447acb02d40071bddd989fe0da87eb5b7fe0dbadfc4f74de7531"
        ),
        "tokenizer.json": "8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1",
        "tokenizer_config.json": (
            "33d29c87e41f7dd1efb0434d852730320c82970f292be452d820539bce417052"
        ),
    }
    assert reload["source_metadata_file_sha256"] == expected_source_metadata


def test_source_weight_free_evidence_preserves_memory_size_and_timing_arithmetic() -> None:
    record = _record()
    memory = record["benchmark"]["memory"]
    timing = record["benchmark"]["timing"]
    quantization = record["quantization"]

    assert memory["measurement_kind"] == ("benchmark_subject_artifact_worker_process_peak_rss")
    assert memory["process_isolated"] is True
    assert memory["baseline_worker_pid"] != memory["candidate_worker_pid"]
    assert memory["baseline_peak_rss_bytes"] == 630_423_552
    assert memory["candidate_peak_rss_bytes"] == 552_615_936
    assert (
        memory["candidate_peak_rss_bytes"] - memory["baseline_peak_rss_bytes"]
        == memory["absolute_delta_bytes"]
        == -77_807_616
    )
    assert math.isclose(
        memory["absolute_delta_bytes"] / memory["baseline_peak_rss_bytes"],
        memory["relative_delta"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert memory["observed_candidate_direction"] == "lower"
    assert memory["comparison_direction"] == "minimize"
    assert memory["same_measurement_kind_and_scope"] is True
    assert "exactly one persisted benchmark subject" in memory["measurement_scope"]
    assert "does not load float source weights" in memory["measurement_scope"]
    assert memory["steady_state_only"] is False
    assert memory["tensor_attributable"] is False
    assert memory["deployment_residency"] is False
    assert memory["final_through_exit"] is False

    assert (
        quantization["candidate_tensor_bytes"] + quantization["candidate_metadata_bytes"]
        == quantization["candidate_serialized_size_bytes"]
        == 139_505_866
    )
    assert (
        quantization["candidate_serialized_size_bytes"]
        - quantization["baseline_serialized_size_bytes"]
        == quantization["serialized_size_delta_bytes"]
        == -5_935_276
    )
    assert math.isclose(
        quantization["serialized_size_delta_bytes"]
        / quantization["baseline_serialized_size_bytes"],
        quantization["serialized_size_relative_delta"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    assert math.isclose(
        (timing["candidate_median_end_to_end_ms"] - timing["baseline_median_end_to_end_ms"])
        / timing["baseline_median_end_to_end_ms"],
        timing["candidate_relative_latency_delta"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert timing["candidate_median_end_to_end_ms"] > timing["baseline_median_end_to_end_ms"]
    assert (
        timing["candidate_median_throughput_tokens_per_second"]
        < timing["baseline_median_throughput_tokens_per_second"]
    )
    assert timing["speedup_observed"] is False
    assert timing["baseline_load_time_kind"] != timing["candidate_load_time_kind"]
    assert timing["load_time_comparison_status"] == "unavailable"
    assert timing["load_time_values_are_like_for_like"] is False


def test_source_weight_free_evidence_keeps_claim_boundary_explicit() -> None:
    record = _record()
    claims = record["claims"]
    generation = record["evaluation"]["generation_fixture"]

    true_claims = (
        "real_public_model_executed",
        "full_governed_pipeline_completed",
        "optimized_native_int8_kernel_executed",
        "isolated_subject_artifact_process_peak_rss_measured",
        "distinct_baseline_and_candidate_workers",
        "source_weight_free_candidate_export_load",
        "published_bundle_reloaded_and_executed",
        "same_scope_subject_process_peak_reduction_observed",
        "serialized_size_reduction_observed",
        "safe_candidate_bundle_published",
    )
    assert all(claims[name] is True for name in true_claims)

    false_claims = (
        "fully_source_free_candidate_loader",
        "self_contained_candidate_export",
        "steady_state_memory_reduction",
        "tensor_attributable_memory_reduction",
        "candidate_attributable_deployment_residency",
        "speedup",
        "model_load_speedup",
        "representative_quality_preservation",
        "representative_quality_improvement",
        "generation_output_retention",
        "expert_weights_quantized",
        "learned_router_preservation",
        "peak_device_memory",
        "energy_joules",
        "cuda_execution",
        "awq_execution",
        "gptq_execution",
        "fp8_execution",
        "clean_checkout_provenance",
    )
    assert all(claims[name] is False for name in false_claims)

    assert generation["baseline_exact_match"] == generation["candidate_exact_match"] == 0.0
    assert generation["baseline_output_token_hash"] != generation["candidate_output_token_hash"]
    assert generation["outputs_are_identical"] is False
    assert record["evaluation"]["perplexity_fixture"]["evaluated_tokens"] == 17
    assert record["evaluation"]["perplexity_fixture"]["representative_quality_dataset"] is False
    assert record["quantization"]["fused_expert_slices_quantized"] is False


def test_source_weight_free_evidence_pins_raw_and_provenance_hashes() -> None:
    record = _record()
    hashes = record["raw_artifact_sha256"]
    reload = record["source_weight_free_reload"]
    audit = record["artifact_audit"]

    assert hashes["manifest.json"] == (
        "3cacfaa779f7b01f50424a54f5bd7b01fe90d7004cf4d0a6edac04cb495d719e"
    )
    assert hashes["benchmark_candidate.json"] == (
        "b340afc8c85cc3ec60ec42831ce9f0168876b74f0426c7131d5fb10e8aaa9431"
    )
    for value in hashes.values():
        _assert_sha256(value)
    for value in reload["source_metadata_file_sha256"].values():
        _assert_sha256(value)
    for name in (
        "candidate_state_checksum",
        "bundle_sha256",
        "metadata_sha256",
        "tensor_sha256",
    ):
        _assert_sha256(reload[name])

    assert audit["no_credential_or_private_key_pattern_found"] is True
    assert audit["raw_artifacts_contain_local_absolute_paths"] is True
    assert audit["compact_record_contains_local_absolute_paths"] is False
    assert audit["candidate_bundle_contains_derived_model_weights"] is True
