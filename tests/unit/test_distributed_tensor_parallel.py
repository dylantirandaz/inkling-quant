"""Validation contracts for checked local CPU tensor-parallel evidence."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from queue import Empty
from typing import Any

import pytest
from pydantic import ValidationError

from inkling_quant_lab import distributed
from inkling_quant_lab.distributed import (
    LocalCPUTensorParallelResult,
    run_local_cpu_tensor_parallel_smoke,
)
from inkling_quant_lab.exceptions import CapabilityError

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = PROJECT_ROOT / "docs/experiments/cpu-tensor-parallel-tiny-mlp-torch-2.13.json"
CANONICAL_RESULT_PATH = (
    PROJECT_ROOT / "docs/experiments/cpu-tensor-parallel-tiny-mlp-torch-2.13.result.canonical.json"
)


def _result_payload() -> dict[str, Any]:
    record: dict[str, Any] = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    return deepcopy(record["result"])


def test_checked_tensor_parallel_result_satisfies_strict_evidence_model() -> None:
    result = LocalCPUTensorParallelResult.model_validate(_result_payload())

    assert result.parameter_sharding_executed is True
    assert result.public_model_sharding_validated is False
    assert result.rank_results[0].tensor_parallel_output_sha256 == (
        result.rank_results[1].tensor_parallel_output_sha256
    )


def test_checked_tensor_parallel_evidence_hashes_match_retained_files() -> None:
    record: dict[str, Any] = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    for source in record["provenance"]["executed_implementation_files"]:
        source_path = PROJECT_ROOT / source["path"]
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source["sha256"]
    canonical = CANONICAL_RESULT_PATH.read_bytes()
    assert (
        hashlib.sha256(canonical).hexdigest()
        == (record["provenance"]["canonical_result"]["sha256"])
    )
    assert json.loads(canonical) == record["result"]


def test_tensor_parallel_result_rejects_error_beyond_parent_tolerance() -> None:
    payload = _result_payload()
    payload["rank_results"][0]["max_abs_error"] = 1.0

    with pytest.raises(ValidationError, match="absolute error exceeds"):
        LocalCPUTensorParallelResult.model_validate(payload)


def test_tensor_parallel_result_rejects_different_reference_outputs() -> None:
    payload = _result_payload()
    payload["rank_results"][1]["reference_output_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="different reference outputs"):
        LocalCPUTensorParallelResult.model_validate(payload)


def test_tensor_parallel_result_rejects_changed_model_shard_metadata() -> None:
    payload = _result_payload()
    shard = payload["rank_results"][1]["parameter_shards"][0]
    shard["global_shape"] = [14, 8]
    shard["global_numel"] = 112

    with pytest.raises(ValidationError, match="exact model plan"):
        LocalCPUTensorParallelResult.model_validate(payload)


def test_tensor_parallel_result_rejects_incomplete_global_partition() -> None:
    payload = _result_payload()
    payload["rank_results"][0]["parameter_shards"][0]["shard_range"] = [1, 7]

    with pytest.raises(ValidationError, match="global origin"):
        LocalCPUTensorParallelResult.model_validate(payload)


def test_tensor_parallel_result_rejects_credential_like_hardware_label() -> None:
    payload = _result_payload()
    payload["operator_declared_hardware_label"] = "Authorization: Bearer abc123"

    with pytest.raises(ValidationError, match="credential-like"):
        LocalCPUTensorParallelResult.model_validate(payload)
    with pytest.raises(ValueError, match="credential-like"):
        run_local_cpu_tensor_parallel_smoke(
            hardware_label="Authorization: Bearer abc123",
        )


@pytest.mark.parametrize(
    "override",
    (
        {"timeout_seconds": float("inf")},
        {"absolute_tolerance": float("nan")},
        {"relative_tolerance": float("inf")},
        {"seed": 1.5},
    ),
)
def test_tensor_parallel_runner_rejects_non_finite_protocol_values(
    override: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match=r"finite and positive|non-negative integer"):
        run_local_cpu_tensor_parallel_smoke(
            hardware_label="operator-declared unit-test CPU",
            **override,
        )


def test_tensor_parallel_result_rejects_non_finite_errors() -> None:
    payload = _result_payload()
    payload["rank_results"][0]["max_relative_error"] = float("nan")

    with pytest.raises(ValidationError, match="finite number"):
        LocalCPUTensorParallelResult.model_validate(payload)


@pytest.mark.parametrize("invalid_seed", (True, 1.5))
def test_tensor_parallel_result_requires_strict_integer_seed(
    invalid_seed: bool | float,
) -> None:
    payload = _result_payload()
    payload["seed"] = invalid_seed

    with pytest.raises(ValidationError, match="valid integer"):
        LocalCPUTensorParallelResult.model_validate(payload)


class _AlreadyFailedQueue:
    def get_nowait(self) -> dict[str, Any]:
        raise Empty

    def get(self, *, timeout: float) -> dict[str, Any]:
        raise AssertionError(f"failed workers must not trigger a {timeout}-second queue wait")


def test_failed_tensor_parallel_workers_are_reported_without_blocking_queue_wait() -> None:
    with pytest.raises(CapabilityError, match="worker failed") as captured:
        distributed._collect_tensor_parallel_payloads(
            _AlreadyFailedQueue(),
            exit_codes=[-6, -6],
            world_size=2,
            timeout_seconds=30.0,
        )

    assert captured.value.details["exit_codes"] == [-6, -6]
