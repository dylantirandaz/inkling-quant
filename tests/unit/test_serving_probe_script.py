"""Contracts for the content-redacting optimized-serving probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.probe_serving_endpoint as serving_probe
from scripts.probe_serving_endpoint import (
    generation_request,
    normalize_generation,
    normalize_model_info,
    normalize_server_environment,
    text_sha256,
    validate_base_url,
)

pytestmark = pytest.mark.unit


def _server_environment() -> dict[str, Any]:
    return {
        "captured_at_utc": "2026-07-16T04:00:00Z",
        "platform": {"system": "Darwin", "release": "24.3.0", "machine": "arm64"},
        "hardware": {
            "description": "Apple M3, 10 GPU cores",
            "logical_cpu_count": 8,
            "accelerator": "Apple Metal",
        },
        "python": {"version": "3.12.7", "implementation": "CPython"},
        "software": {
            "mlx": "0.32.0",
            "mlx-lm": "0.31.3",
            "sglang": "1.0",
            "torch": "2.11.0",
            "transformers": "5.12.1",
        },
        "capture_method": "versions queried inside the serving virtual environment",
        "ignored_secret": "must not survive normalization",
    }


def test_model_info_requires_exact_declared_model() -> None:
    normalized = normalize_model_info(
        {
            "model_path": "org/model",
            "model_type": "mixtral",
            "architectures": ["MixtralForCausalLM"],
            "is_generation": True,
            "weight_version": "default",
        },
        "org/model",
    )

    assert normalized["model_type"] == "mixtral"
    assert normalized["architectures"] == ["MixtralForCausalLM"]
    with pytest.raises(ValueError, match="model mismatch"):
        normalize_model_info({"model_path": "other"}, "org/model")


def test_generation_normalization_redacts_text_and_token_ids() -> None:
    output = "generated content"
    normalized = normalize_generation(
        {
            "text": output,
            "output_ids": [1, 2, 3],
            "meta_info": {
                "prompt_tokens": 4,
                "completion_tokens": 3,
                "e2e_latency": 0.25,
                "cached_tokens": 0,
                "finish_reason": {"type": "length"},
            },
        },
        0.3,
    )

    assert normalized["output_sha256"] == text_sha256(output)
    assert output not in str(normalized)
    assert "output_ids" not in normalized
    assert normalized["completion_tokens"] == 3


def test_generation_rejects_inconsistent_token_metadata() -> None:
    with pytest.raises(ValueError, match="did not match"):
        normalize_generation(
            {
                "text": "result",
                "output_ids": [1],
                "meta_info": {
                    "prompt_tokens": 1,
                    "completion_tokens": 2,
                    "e2e_latency": 0.1,
                },
            },
            0.1,
        )


@pytest.mark.parametrize("server_latency", (float("nan"), float("inf"), float("-inf")))
def test_generation_rejects_non_finite_server_latency(server_latency: float) -> None:
    with pytest.raises(ValueError, match="valid server latency"):
        normalize_generation(
            {
                "text": "result",
                "output_ids": [1],
                "meta_info": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "e2e_latency": server_latency,
                    "finish_reason": {"type": "length"},
                },
            },
            0.1,
        )


def test_generation_request_applies_declared_sampling_seed() -> None:
    request = generation_request("prompt", seed=17, max_new_tokens=4)

    assert request["sampling_params"] == {
        "temperature": 0,
        "max_new_tokens": 4,
        "sampling_seed": 17,
    }


def test_remote_endpoint_requires_explicit_authorization() -> None:
    assert (
        validate_base_url("http://127.0.0.1:30000/", allow_remote_endpoint=False)
        == "http://127.0.0.1:30000"
    )


def test_server_environment_requires_matching_runtime_provenance() -> None:
    payload = _server_environment()

    normalized = normalize_server_environment(
        payload,
        backend="sglang",
        backend_version="1.0",
        runtime="mlx_on_apple_metal",
    )

    assert normalized["hardware"]["description"] == "Apple M3, 10 GPU cores"
    assert normalized["software"]["mlx-lm"] == "0.31.3"
    assert "ignored_secret" not in normalized
    payload["software"].pop("mlx-lm")
    with pytest.raises(ValueError, match="missing required software versions: mlx-lm"):
        normalize_server_environment(
            payload,
            backend="sglang",
            backend_version="1.0",
            runtime="mlx_on_apple_metal",
        )
    payload = _server_environment()
    payload["captured_at_utc"] = "2026-07-16T05:00:00+01:00"
    with pytest.raises(ValueError, match="must use UTC"):
        normalize_server_environment(
            payload,
            backend="sglang",
            backend_version="1.0",
            runtime="mlx_on_apple_metal",
        )
    payload = _server_environment()
    payload["software"]["authorization"] = "Bearer secret"
    with pytest.raises(ValueError, match="unreviewed package keys: authorization"):
        normalize_server_environment(
            payload,
            backend="sglang",
            backend_version="1.0",
            runtime="mlx_on_apple_metal",
        )
    payload = _server_environment()
    payload["software"]["torch"] = "Bearer secret"
    with pytest.raises(ValueError, match="invalid version strings: torch"):
        normalize_server_environment(
            payload,
            backend="sglang",
            backend_version="1.0",
            runtime="mlx_on_apple_metal",
        )
    payload = _server_environment()
    payload["capture_method"] = "env TOKEN=hf_abcdefgh1234 python capture.py"
    with pytest.raises(ValueError, match=r"credential-like material at provenance.capture_method"):
        normalize_server_environment(
            payload,
            backend="sglang",
            backend_version="1.0",
            runtime="mlx_on_apple_metal",
        )
    with pytest.raises(ValueError, match="allow-remote-endpoint"):
        validate_base_url("https://serving.example", allow_remote_endpoint=False)
    assert (
        validate_base_url("https://serving.example", allow_remote_endpoint=True)
        == "https://serving.example"
    )


def test_probe_excludes_warmup_and_redacts_content(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def request(
        url: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout_seconds: float,
    ) -> tuple[dict[str, Any], float]:
        del method, timeout_seconds
        calls.append((url, payload))
        if url.endswith("/model_info"):
            return {
                "model_path": "org/model",
                "model_type": "mixtral",
                "architectures": ["MixtralForCausalLM"],
                "is_generation": True,
                "weight_version": "default",
            }, 0.01
        index = len(calls)
        return {
            "text": f"private-output-{index}",
            "output_ids": [1, 2],
            "meta_info": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "e2e_latency": index / 100,
                "finish_reason": {"type": "length"},
                "cached_tokens": 0,
            },
        }, index / 100

    monkeypatch.setattr(serving_probe, "_request_json", request)
    record = serving_probe.probe(
        base_url="http://localhost:30000",
        prompt="private-prompt",
        model_id="org/model",
        model_revision="a" * 40,
        model_checksum="b" * 64,
        backend="sglang",
        backend_version="1.0",
        backend_revision="c" * 40,
        runtime="mlx",
        seed=17,
        max_new_tokens=2,
        warmup_trials=1,
        measured_trials=2,
        timeout_seconds=1.0,
        server_environment=_server_environment(),
    )

    assert len(calls) == 4
    assert len(record["results"]["trials"]) == 2
    measured_payload = calls[1][1]
    assert measured_payload is not None
    assert measured_payload["sampling_params"]["sampling_seed"] == 17
    assert "private-prompt" not in json.dumps(record)
    assert "private-output" not in json.dumps(record)
    assert record["protocol"]["probe_record_contains_prompt_or_output_content"] is False
    assert record["schema_version"] == "optimized-serving-smoke-v2"
    assert record["server_environment"]["software"]["sglang"] == "1.0"


def test_checked_sglang_evidence_uses_distinct_curated_schema() -> None:
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (root / "docs/experiments/sglang-mlx-stories15m-smoke.json").read_text(encoding="utf-8")
    )

    assert record["schema_version"] == "sglang-mlx-serving-evidence-v1"
    assert (
        record["model_revision_resolution"]["sglang_revision_forwarded_to_mlx_weight_loader"]
        is False
    )
    assert record["model_code_safety"]["general_remote_code_guard_established"] is False
    assert record["model_code_safety"]["python_files_in_resolved_snapshot"] == []
    assert record["protocol"]["checked_record_contains_prompt_or_output_content"] is False
