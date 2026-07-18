from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "docs/experiments/stories15m-native-int8-isolated-peak-m3.json"
EVIDENCE_SHA256 = "193250369a98577c00e1f2f75dc1dc9a4e6c613c2f9171f452ac70536d929986"


def _record() -> dict[str, Any]:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EVIDENCE_SHA256
    assert b"/Users/" not in payload
    assert b"dylantirandaz" not in payload
    return cast(dict[str, Any], json.loads(payload))


def test_public_isolated_peak_evidence_binds_successful_exact_execution() -> None:
    record = _record()
    run = record["run"]
    model = record["model"]
    quantization = record["quantization"]

    assert record["status"] == "success"
    assert run["config_hash"] == (
        "27c9aa827b7bca3f6fb2c2165f90e82321c28117979da075ca5659ca39c7e776"
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
    assert quantization["fused_expert_slice_count"] == 24
    assert quantization["fused_expert_slices_quantized"] is False


def test_public_isolated_peak_evidence_preserves_raw_memory_and_size_arithmetic() -> None:
    record = _record()
    memory = record["benchmark"]["memory"]
    quantization = record["quantization"]

    assert memory["measurement_kind"] == "benchmark_stage_worker_process_peak_rss"
    assert memory["process_isolated"] is True
    assert memory["baseline_worker_pid"] != memory["candidate_worker_pid"]
    assert memory["baseline_peak_rss_bytes"] == 948_174_848
    assert memory["candidate_peak_rss_bytes"] == 1_199_210_496
    assert (
        memory["candidate_peak_rss_bytes"] - memory["baseline_peak_rss_bytes"]
        == memory["absolute_delta_bytes"]
        == 251_035_648
    )
    assert math.isclose(
        memory["absolute_delta_bytes"] / memory["baseline_peak_rss_bytes"],
        memory["relative_delta"],
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert memory["direction"] == "neutral"
    assert memory["candidate_reconstruction_can_retain_source_and_candidate"] is True
    assert memory["device_memory_available"] is False
    assert memory["energy_available"] is False

    assert (
        quantization["candidate_serialized_size_bytes"]
        - quantization["baseline_serialized_size_bytes"]
        == quantization["serialized_size_delta_bytes"]
        == -5_935_276
    )
    assert (
        quantization["candidate_serialized_size_bytes"]
        < quantization["baseline_serialized_size_bytes"]
    )


def test_public_isolated_peak_evidence_keeps_quality_reload_and_speed_claims_false() -> None:
    record = _record()
    claims = record["claims"]
    generation = record["evaluation"]["generation_fixture"]
    timing = record["benchmark"]["timing"]
    export = record["export"]

    assert claims["real_public_model_executed"] is True
    assert claims["optimized_native_int8_kernel_executed"] is True
    assert claims["isolated_os_process_peak_rss_measured"] is True
    assert claims["serialized_size_reduction_observed"] is True
    assert claims["safe_candidate_bundle_published"] is True

    false_claims = (
        "live_memory_reduction",
        "candidate_attributable_deployment_residency",
        "source_free_candidate_loader",
        "published_bundle_reloaded_and_executed",
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
    assert timing["candidate_median_end_to_end_ms"] > timing["baseline_median_end_to_end_ms"]
    assert timing["speedup_observed"] is False
    assert timing["baseline_load_time_kind"] != timing["candidate_load_time_kind"]
    assert timing["load_time_values_are_like_for_like"] is False
    assert export["published"] is True
    assert export["published_bundle_reloaded_and_executed"] is False
    assert export["metadata_reload_adapter"] != export["source_adapter"]


def test_public_isolated_peak_evidence_pins_raw_hashes_and_audit_defects() -> None:
    record = _record()
    hashes = record["raw_artifact_sha256"]
    audit = record["artifact_audit"]

    assert hashes["manifest.json"] == (
        "415e3d459956800d9dedbe8bf298287934685f6f1d439f7b538600a467e22869"
    )
    assert hashes["candidate_model.safetensors"] == (
        "8538aedc9121f9da9791351e4ab9a5ed0cc747b5c3d5eda0fe47bea5836be3ce"
    )
    assert all(len(value) == 64 for value in hashes.values())
    assert audit["no_credential_or_private_key_pattern_found"] is True
    assert audit["raw_artifacts_contain_local_absolute_paths"] is True
    assert audit["compact_record_contains_local_absolute_paths"] is False
    assert audit["manifest_warning_list_empty_despite_component_warnings"] is True
    assert audit["generated_report_duplicates_native_warning"] is True
