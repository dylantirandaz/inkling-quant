"""Pin the checked isolated benchmark-stage peak-RSS acceptance record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECORD = PROJECT_ROOT / "docs/experiments/tiny-moe-native-int8-isolated-peak-m3.json"
RECORD_SHA256 = "a5c1ec04185c913cdc6c19704a242a83096cd566a692a9dbb3c48c6ba67fb06a"


def test_isolated_peak_record_is_pinned_redacted_and_narrow() -> None:
    raw = RECORD.read_bytes()
    evidence = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == RECORD_SHA256
    assert evidence["schema_version"] == "isolated-benchmark-stage-peak-rss-v1"
    assert evidence["run"]["status"] == "success"
    assert evidence["run"]["successful_stage_count"] == 14
    assert evidence["run"]["declared_output_count"] == 34
    assert evidence["benchmark_contract"]["measurement_kind"] == (
        "benchmark_stage_worker_process_peak_rss"
    )
    assert "execution-time bytes" in evidence["hash_provenance"]

    baseline = evidence["measurements"]["baseline"]
    candidate = evidence["measurements"]["candidate"]
    assert baseline["worker_pid"] != candidate["worker_pid"]
    assert baseline["peak_rss_bytes"] > 0
    assert candidate["peak_rss_bytes"] > 0
    assert evidence["measurements"]["candidate_minus_baseline_peak_rss_bytes"] == (
        candidate["peak_rss_bytes"] - baseline["peak_rss_bytes"]
    )

    claims = evidence["claims"]
    for supported in (
        "baseline_and_candidate_used_distinct_processes",
        "os_process_high_water_rss_through_read_point_measured",
        "each_peak_is_scoped_to_one_governed_benchmark_stage",
        "parent_pipeline_and_prior_stages_excluded",
        "optimized_native_int8_candidate_executed",
        "all_manifest_outputs_verified",
    ):
        assert claims[supported] is True
    for unsupported in (
        "steady_state_only_peak_claim",
        "candidate_attributable_memory_claim",
        "deployed_checkpoint_loader_peak_claim",
        "final_process_through_exit_peak_claim",
        "post_read_cleanup_or_result_transport_included",
        "public_model_memory_claim",
        "speedup_claim",
        "energy_claim",
    ):
        assert claims[unsupported] is False

    text = raw.decode("utf-8")
    assert "/Users/" not in text
    assert "/private/tmp" not in text
    assert "hf_" not in text
    assert "sk-" not in text
