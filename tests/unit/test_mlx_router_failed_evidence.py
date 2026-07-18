from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = PROJECT_ROOT / "docs/experiments/stories15m-router-domain-pair-10step-failed-m3.json"
EVIDENCE_SHA256 = "516bbd2c90b1326e83ba878a3c076ed86d9e9b0d307119e5562a9491fb29921d"


def _record() -> dict[str, Any]:
    payload = EVIDENCE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EVIDENCE_SHA256
    return cast(dict[str, Any], json.loads(payload))


def test_failed_router_evidence_is_a_sealed_negative_result() -> None:
    record = _record()

    assert record["status"] == "failed"
    assert record["failure_code"] == "held_out_acceptance_failed"
    assert record["source_artifact"]["semantic_binding_verification_passed"] is True
    assert record["source_artifact"]["read_only_directory_mode"] == "0555"
    assert record["source_artifact"]["read_only_file_mode"] == "0444"
    assert record["acceptance"]["passed"] is False
    assert record["acceptance"]["passed_check_count"] == 7
    assert record["acceptance"]["total_check_count"] == 9
    assert [item["check_id"] for item in record["acceptance"]["failed_checks"]] == [
        "overall.validation_exact_top2_pair_accuracy",
        "domain.alice.validation_exact_top2_pair_accuracy",
    ]
    assert record["artifact_outcome"]["configured_success_path_absent"] is True
    assert record["artifact_outcome"]["router_overlay_exported"] is False
    assert record["artifact_outcome"]["typed_router_lineage_created"] is False


def test_failed_router_evidence_retains_exact_execution_and_metric_arithmetic() -> None:
    record = _record()
    execution = record["execution"]
    metrics = record["metrics"]
    control = metrics["source_router_control"]
    trained = metrics["ten_step_router"]
    deltas = metrics["deltas"]

    assert execution["device"] == "Device(gpu, 0)"
    assert execution["device_name"] == "Apple M3"
    assert execution["mlx"] == "0.32.0"
    assert execution["mlx_lm"] == "0.31.3"
    assert execution["optimizer_steps"] == 10
    assert execution["changed_router_tensor_count"] == 6
    assert execution["unchanged_nonrouter_tensor_count"] == 57
    assert math.isclose(
        control["mean_cross_entropy"] - trained["mean_cross_entropy"],
        deltas["validation_cross_entropy_reduction"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        trained["exact_top2_pair_accuracy"] - control["exact_top2_pair_accuracy"],
        deltas["validation_accuracy_gain_over_source_router"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert metrics["mlx_allocator_at_failed_acceptance"]["peak_bytes"] > 0


def test_failed_router_evidence_suppresses_every_unearned_claim() -> None:
    record = _record()
    claims = record["claims"]

    assert claims["real_public_model_executed_on_apple_metal"] is True
    assert claims["exactly_ten_supervised_router_updates_executed"] is True
    assert claims["exactly_six_router_tensors_changed"] is True
    assert claims["all_nonrouter_parameter_bytes_unchanged"] is True
    assert claims["learned_domain_supervised_routing_accepted"] is False
    assert claims["quantized_learned_router_retention"] is False
    assert claims["causal_lm_specialization"] is False
    assert claims["output_quality_retention"] is False
