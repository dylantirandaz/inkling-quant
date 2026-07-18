"""Contracts for the curated Stories15M confirmatory quality evidence."""

from __future__ import annotations

import hashlib
import json
import math
import stat
from pathlib import Path
from typing import Any, cast

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "docs/experiments/stories15m-native-int8-confirmatory-256-m3.json"
EVIDENCE_SHA256 = "a8e1e9751a29f38ae1d0678a845329afd81a6213d952a0e4a8d9c09cb431d40f"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, Any]:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EVIDENCE_SHA256
    assert b"/Users/" not in payload
    assert b"dylantirandaz" not in payload
    return cast(dict[str, Any], json.loads(payload))


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        mapping = cast(dict[str, object], value)
        return set(mapping).union(*(_all_keys(item) for item in mapping.values()))
    if isinstance(value, list):
        return set().union(*(_all_keys(item) for item in value))
    return set()


def test_evidence_is_content_redacted_and_binds_the_sealed_pair() -> None:
    record = _record()
    protocol = record["protocol"]
    confirmation = record["confirmation"]
    audit = record["artifact_audit"]

    assert record["status"] == "confirmed"
    assert confirmation["confirmatory_pass"] is True
    assert confirmation["confirmatory_claim_ready"] is True
    assert confirmation["claim_producing_path"] is True
    assert confirmation["clean_execution_count"] == 2
    assert confirmation["required_clean_execution_count"] == 2
    assert confirmation["statistical_sample_count"] == 256
    assert confirmation["deterministic_repeat_is_additional_statistical_sample"] is False
    assert confirmation["within_execution_noninferiority_passed"] == {
        "1": True,
        "2": True,
    }
    assert len(confirmation["attempts"]) == 2
    assert {attempt["ordinal"] for attempt in confirmation["attempts"]} == {1, 2}
    assert protocol["bound_file_count"] == 98
    assert protocol["preregistration"]["outcomes_fields_are_historical_lock_time_values"] is True

    assert audit["compact_record_contains_local_absolute_paths"] is False
    assert audit["compact_record_contains_prompt_or_output_content"] is False
    assert audit["raw_prompt_text_or_output_token_ids_persisted"] is False
    assert not {"prompt", "text", "output_ids", "token_ids"}.intersection(_all_keys(record))

    preregistration = PROJECT_ROOT / protocol["preregistration"]["path"]
    assert _sha256(preregistration) == protocol["preregistration"]["sha256"]
    assert preregistration.stat().st_size == protocol["preregistration"]["size_bytes"]


def test_primary_result_preserves_the_preregistered_estimand_and_arithmetic() -> None:
    record = _record()
    sampling = record["dataset_and_sampling"]
    result = record["primary_noninferiority_result"]

    assert sampling["finite_population_story_count"] == 21_958
    assert sampling["finite_population_evaluated_token_count"] == 4_441_967
    assert sampling["selected_story_count"] == 256
    assert sampling["selected_evaluated_token_count"] == 52_038
    assert sampling["selected_truncated_story_count"] == 50
    assert sum(item["population_story_count"] for item in sampling["strata"]) == 21_958
    assert sum(item["population_evaluated_token_count"] for item in sampling["strata"]) == 4_441_967
    assert sum(item["selected_story_count"] for item in sampling["strata"]) == 256
    assert sum(item["selected_evaluated_token_count"] for item in sampling["strata"]) == 52_038

    assert result["estimator"] == (
        "stratified_horvitz_thompson_nll_numerator_over_known_token_denominator"
    )
    assert result["sampling_unit"] == "paired_story_cluster"
    assert result["bootstrap_seed"] == 20_260_715
    assert result["bootstrap_replicates"] == 100_000
    assert result["coverage_interpretation"] == (
        "nominal_approximate_not_exact_finite_population_coverage"
    )
    assert math.isclose(
        result["candidate_mean_nll_ht"] - result["baseline_mean_nll_ht"],
        result["candidate_minus_baseline_mean_nll_ht"],
        rel_tol=0.0,
        abs_tol=5e-16,
    )
    assert math.isclose(
        math.expm1(result["candidate_minus_baseline_mean_nll_ht"]),
        result["relative_perplexity_change_point"],
        rel_tol=0.0,
        abs_tol=1e-16,
    )
    assert math.isclose(
        math.expm1(result["nominal_one_sided_95_upper_bound_nll"]),
        result["nominal_one_sided_95_upper_bound_relative_perplexity"],
        rel_tol=0.0,
        abs_tol=1e-16,
    )
    assert math.isclose(
        math.log1p(result["noninferiority_margin_relative_perplexity"]),
        result["noninferiority_margin_nll"],
        rel_tol=0.0,
        abs_tol=1e-16,
    )
    assert math.isclose(
        result["noninferiority_margin_nll"] - result["nominal_one_sided_95_upper_bound_nll"],
        result["margin_slack_nll"],
        rel_tol=0.0,
        abs_tol=1e-16,
    )
    assert result["nominal_one_sided_95_upper_bound_nll"] < result["noninferiority_margin_nll"]
    assert result["passed"] is True


def test_evidence_retains_per_layer_routing_and_quantization_boundaries() -> None:
    record = _record()
    quantization = record["quantization"]
    generation = record["descriptive_generation"]
    routing = record["descriptive_routing"]
    claims = record["claims"]

    assert quantization["backend"] == "torch_native_dynamic_int8"
    assert quantization["kernel"] == "quantized::linear_dynamic"
    assert quantization["quantized_engine"] == "qnnpack"
    assert quantization["quantized_attention_linear_count"] == 24
    assert quantization["fused_expert_slices_quantized"] == 0
    assert quantization["fused_expert_slices_float32"] == 24
    assert (
        quantization["candidate_runtime_tensor_storage_bytes"]
        - quantization["baseline_runtime_tensor_storage_bytes"]
        == quantization["runtime_tensor_storage_delta_bytes"]
        == -5_944_320
    )
    assert math.isclose(
        quantization["runtime_tensor_storage_delta_bytes"]
        / quantization["baseline_runtime_tensor_storage_bytes"],
        quantization["runtime_tensor_storage_relative_delta"],
        rel_tol=0.0,
        abs_tol=1e-16,
    )

    assert generation["inference_role"] == routing["inference_role"] == "descriptive_only"
    assert generation["sample_count"] == 16
    assert generation["exact_output_hash_match_count"] == 15
    assert generation["exact_output_hash_match_rate"] == 0.9375
    assert generation["differing_sample_ids"] == ["story-004416"]
    assert generation["output_content_persisted"] is False

    assert routing["quality_forward_count_without_hooks"] == 256
    assert routing["routing_reforward_count_with_hooks"] == 16
    assert (
        routing["expected_layer_token_event_count"]
        == routing["observed_layer_token_event_count"]
        == routing["recorded_layer_token_event_count"]
        == 19_764
    )
    assert routing["reforward_token_counts_and_mean_nll_match_hook_free_exactly"] is True
    assert routing["model_state_unchanged"] is True
    assert routing["hooks_removed_before_generation"] is True
    assert len(routing["per_layer"]) == routing["macro"]["layer_count"] == 6
    assert len({item["layer_id"] for item in routing["per_layer"]}) == 6
    ranked = [
        item["layer_id"]
        for item in sorted(
            routing["per_layer"], key=lambda item: item["js_divergence"], reverse=True
        )
    ]
    assert routing["per_layer_drift_ranking"] == ranked

    assert claims["prospective_finite_holdout_noninferiority_confirmed"] is True
    assert claims["optimized_native_dynamic_int8_kernel_executed"] is True
    for name in (
        "all_moe_expert_weights_quantized",
        "learned_router_specialization_preserved",
        "generation_or_routing_inferential_claim",
        "universal_model_quality_preservation",
        "causal_effect_claim",
        "latency_benchmark_claim",
        "energy_joules_claim",
        "cuda_execution_claim",
        "awq_execution_claim",
        "gptq_execution_claim",
        "fp8_execution_claim",
    ):
        assert claims[name] is False


def test_raw_sealed_artifacts_match_curated_bindings_when_present() -> None:
    record = _record()
    confirmation = record["confirmation"]
    aggregate_path = PROJECT_ROOT / confirmation["aggregate"]["path"]
    if not aggregate_path.exists():
        pytest.skip(
            "raw research-slice artifacts are intentionally not required in source archives"
        )

    assert _sha256(aggregate_path) == confirmation["aggregate"]["sha256"]
    assert aggregate_path.stat().st_size == confirmation["aggregate"]["size_bytes"]
    assert stat.S_IMODE(aggregate_path.stat().st_mode) == 0o444
    assert stat.S_IMODE(aggregate_path.parent.stat().st_mode) == 0o555

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["status"] == "confirmed"
    assert aggregate["confirmatory_pass"] is True
    assert aggregate["confirmatory_claim_ready"] is True
    assert aggregate["reasons"] == []
    assert (
        aggregate["shared_scientific_projection_sha256"]
        == confirmation["shared_scientific_projection_sha256"]
    )
    assert (
        aggregate["shared_environment_projection_sha256"]
        == confirmation["shared_environment_projection_sha256"]
    )

    for attempt in confirmation["attempts"]:
        path = PROJECT_ROOT / attempt["record_path"]
        assert _sha256(path) == attempt["record_sha256"]
        assert path.stat().st_size == attempt["record_size_bytes"]
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o555

    first = json.loads((PROJECT_ROOT / confirmation["attempts"][0]["record_path"]).read_text())
    result = record["primary_noninferiority_result"]
    raw_result = first["protocol"]["confirmatory"]["noninferiority_result"]
    assert (
        raw_result["candidate_minus_baseline_mean_nll_ht"]
        == result["candidate_minus_baseline_mean_nll_ht"]
    )
    assert (
        raw_result["upper_confidence_bound_nll"] == result["nominal_one_sided_95_upper_bound_nll"]
    )
    assert (
        raw_result["relative_perplexity_change"]["upper_confidence_bound"]
        == result["nominal_one_sided_95_upper_bound_relative_perplexity"]
    )
    assert (
        first["candidates"][0]["routing"]["comparison"]["macro"]
        == record["descriptive_routing"]["macro"]
    )
