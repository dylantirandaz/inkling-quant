from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORD = PROJECT_ROOT / "docs/experiments/stories15m-moe-registered-mlx-pipeline-m3.json"
RECORD_SHA256 = "707ae70dc8f466ca75b5c659800014cef1c8797bf408bd906ca66142285ef1b4"


def _walk(value: Any) -> tuple[Any, ...]:
    if isinstance(value, dict):
        return tuple(item for nested in value.values() for item in _walk(nested))
    if isinstance(value, list):
        return tuple(item for nested in value for item in _walk(nested))
    return (value,)


def test_registered_mlx_acceptance_record_is_pinned_redacted_and_complete() -> None:
    raw = RECORD.read_bytes()
    evidence = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == RECORD_SHA256
    assert evidence["schema_version"] == "registered-mlx-public-moe-pipeline-acceptance-v1"
    assert set(evidence["runs"]) == {"q4", "q8"}
    assert evidence["model"]["revision"] == ("b6dd737497465570b5f5e962dbc9d9454ed1e0eb")
    assert evidence["environment"]["packages"] == {
        "mlx": "0.32.0",
        "mlx-lm": "0.31.3",
        "mlx-metal": "0.32.0",
        "safetensors": "0.8.0",
        "transformers": "5.12.1",
    }

    required_stages = {
        "resolve_configuration",
        "probe_runtime",
        "load_baseline",
        "inventory_modules",
        "collect_statistics",
        "resolve_precision_policy",
        "quantize",
        "evaluate_baseline",
        "evaluate_candidate",
        "benchmark_baseline",
        "benchmark_candidate",
        "compare_routing",
        "generate_reports",
        "finalize_manifest",
    }
    for label, bits in (("q4", 4), ("q8", 8)):
        run = evidence["runs"][label]
        assert run["run_kind"] == "fresh"
        assert set(run["stages"]) == required_stages
        assert run["stages"]["quantize"] == {"status": "success", "attempt": 1}
        for stage in ("benchmark_baseline", "benchmark_candidate"):
            assert run["stages"][stage] == {
                "status": "skipped_not_required",
                "attempt": 0,
            }
        assert run["bundle"]["bits"] == bits
        assert run["bundle"]["quantized_leaf_count"] == 50
        assert run["bundle"]["quantized_fused_expert_projection_count"] == 18
        assert set(run["hashes"]) == {
            "config_sha256",
            "resolved_config_sha256",
            "environment_sha256",
            "manifest_sha256",
            "completion_sha256",
            "quantization_manifest_sha256",
            "packed_weight_sha256",
            "baseline_evaluation_sha256",
            "candidate_evaluation_sha256",
            "routing_comparison_sha256",
        }
        assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in run["hashes"].values())

    claims = evidence["claims"]
    assert claims["q4_fresh_governed_run_completed"] is True
    assert claims["q8_fresh_governed_run_completed"] is True
    assert claims["safe_export_reload_execution_validated"] is True
    for unsupported in (
        "pipeline_benchmarks_executed",
        "latency_or_throughput_claim",
        "peak_memory_claim",
        "energy_or_utilization_claim",
        "representative_quality_claim",
        "expert_aware_or_per_expert_precision_claim",
        "generic_mlx_or_other_model_support_claim",
        "learned_router_specialization_claim",
        "raw_prompts_persisted",
        "raw_model_outputs_persisted",
    ):
        assert claims[unsupported] is False

    assert evidence["secondary_resume_evidence"]["primary_acceptance_run"] is False
    assert evidence["secondary_resume_evidence"]["quantize_attempt"] == 3
    assert all(math.isfinite(value) for value in _walk(evidence) if isinstance(value, float))

    text = raw.decode("utf-8")
    assert "/Users/" not in text
    assert "/private/tmp" not in text
    assert "Once upon" not in text
    assert not re.search(r"(?i)(authorization|bearer\s|hf_[a-z0-9]{8,}|sk-[a-z0-9])", text)
