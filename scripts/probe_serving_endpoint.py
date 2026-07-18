#!/usr/bin/env python3
"""Probe an already-running generation server without persisting text content."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import re
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from inkling_quant_lab.security import sensitive_literal_path

_SUPPORTED_BACKENDS = frozenset({"sglang", "vllm"})
_SAFE_SOFTWARE_PACKAGES = frozenset(
    {
        "accelerate",
        "flashinfer-python",
        "mlx",
        "mlx-lm",
        "numpy",
        "safetensors",
        "sglang",
        "tokenizers",
        "torch",
        "transformers",
        "triton",
        "vllm",
        "xgrammar",
    }
)


def text_sha256(value: str) -> str:
    """Return a stable content identifier without retaining the content."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_model_info(payload: dict[str, Any], expected_model_id: str) -> dict[str, Any]:
    """Validate and retain the non-content model identity exposed by SGLang."""

    model_path = payload.get("model_path")
    if model_path != expected_model_id:
        raise ValueError(
            f"serving endpoint model mismatch: expected {expected_model_id}, received {model_path}"
        )
    architectures = payload.get("architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or not all(isinstance(item, str) for item in architectures)
    ):
        raise ValueError("serving endpoint did not report a valid architecture list")
    model_type = payload.get("model_type")
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("serving endpoint did not report a model type")
    if payload.get("is_generation") is not True:
        raise ValueError("serving endpoint did not identify a generation model")
    return {
        "model_path": model_path,
        "model_type": model_type,
        "architectures": architectures,
        "is_generation": payload.get("is_generation") is True,
        "weight_version": payload.get("weight_version"),
    }


def normalize_generation(payload: dict[str, Any], elapsed_seconds: float) -> dict[str, Any]:
    """Validate a generation response and remove generated content from the record."""

    output = payload.get("text")
    output_ids = payload.get("output_ids")
    metadata = payload.get("meta_info")
    if not isinstance(output, str):
        raise ValueError("generation response did not contain text")
    if not isinstance(output_ids, list) or not all(type(item) is int for item in output_ids):
        raise ValueError("generation response did not contain integer output token IDs")
    if not isinstance(metadata, dict):
        raise ValueError("generation response did not contain metadata")
    completion_tokens = metadata.get("completion_tokens")
    prompt_tokens = metadata.get("prompt_tokens")
    if type(completion_tokens) is not int or completion_tokens != len(output_ids):
        raise ValueError("completion token metadata did not match returned token IDs")
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        raise ValueError("generation response did not report a positive prompt token count")
    server_latency = metadata.get("e2e_latency")
    if isinstance(server_latency, bool) or not isinstance(server_latency, (int, float)):
        raise ValueError("generation response did not report a valid server latency")
    try:
        normalized_server_latency = float(server_latency)
    except OverflowError as error:
        raise ValueError("generation response did not report a valid server latency") from error
    if not math.isfinite(normalized_server_latency) or normalized_server_latency < 0:
        raise ValueError("generation response did not report a valid server latency")
    finish_reason = metadata.get("finish_reason")
    finish_type = finish_reason.get("type") if isinstance(finish_reason, dict) else None
    if not isinstance(finish_type, str) or not finish_type:
        raise ValueError("generation response did not report a finish reason")
    cached_tokens = metadata.get("cached_tokens")
    if cached_tokens is not None and (type(cached_tokens) is not int or cached_tokens < 0):
        raise ValueError("generation response reported an invalid cached token count")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
        raise ValueError("client elapsed time must be finite and non-negative")
    return {
        "output_sha256": text_sha256(output),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_type,
        "server_e2e_seconds": normalized_server_latency,
        "client_elapsed_seconds": elapsed_seconds,
        "cached_tokens": cached_tokens,
    }


def normalize_server_environment(
    payload: dict[str, Any], *, backend: str, backend_version: str, runtime: str
) -> dict[str, Any]:
    """Retain only required, non-secret server hardware/software provenance."""

    captured_at = payload.get("captured_at_utc")
    if not isinstance(captured_at, str) or not captured_at:
        raise ValueError("server environment must declare captured_at_utc")
    try:
        captured_timestamp = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("server environment captured_at_utc must be ISO 8601") from error
    if captured_timestamp.tzinfo is None or captured_timestamp.utcoffset() != timedelta(0):
        raise ValueError("server environment captured_at_utc must use UTC")

    platform_payload = payload.get("platform")
    if not isinstance(platform_payload, dict):
        raise ValueError("server environment must declare platform metadata")
    normalized_platform: dict[str, str] = {}
    for field in ("system", "release", "machine"):
        value = platform_payload.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"server environment platform.{field} must be non-empty")
        normalized_platform[field] = value

    hardware_payload = payload.get("hardware")
    if not isinstance(hardware_payload, dict):
        raise ValueError("server environment must declare hardware metadata")
    description = hardware_payload.get("description")
    if not isinstance(description, str) or not description:
        raise ValueError("server environment hardware.description must be non-empty")
    normalized_hardware: dict[str, Any] = {"description": description}
    for field in ("accelerator", "accelerator_details"):
        value = hardware_payload.get(field)
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"server environment hardware.{field} must be non-empty")
            normalized_hardware[field] = value
    for field in ("logical_cpu_count", "memory_bytes", "accelerator_memory_bytes"):
        value = hardware_payload.get(field)
        if value is not None:
            if type(value) is not int or value <= 0:
                raise ValueError(f"server environment hardware.{field} must be positive")
            normalized_hardware[field] = value

    python_payload = payload.get("python")
    if not isinstance(python_payload, dict):
        raise ValueError("server environment must declare Python metadata")
    python_version = python_payload.get("version")
    if not isinstance(python_version, str) or not python_version:
        raise ValueError("server environment python.version must be non-empty")
    normalized_python = {"version": python_version}
    implementation = python_payload.get("implementation")
    if implementation is not None:
        if not isinstance(implementation, str) or not implementation:
            raise ValueError("server environment python.implementation must be non-empty")
        normalized_python["implementation"] = implementation

    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            "serving probe supports only explicitly reviewed backends: "
            + ", ".join(sorted(_SUPPORTED_BACKENDS))
        )
    software_payload = payload.get("software")
    if (
        not isinstance(software_payload, dict)
        or not software_payload
        or not all(
            isinstance(package, str) and package and isinstance(version, str) and version
            for package, version in software_payload.items()
        )
    ):
        raise ValueError("server environment software versions must be non-empty strings")
    unsafe_package_keys = sorted(set(software_payload).difference(_SAFE_SOFTWARE_PACKAGES))
    if unsafe_package_keys:
        raise ValueError(
            "server environment software contains unreviewed package keys: "
            + ", ".join(unsafe_package_keys)
        )
    invalid_versions = sorted(
        package
        for package, version in software_payload.items()
        if re.fullmatch(r"[0-9][0-9A-Za-z.!+_-]*", version) is None
    )
    if invalid_versions:
        raise ValueError(
            "server environment software contains invalid version strings: "
            + ", ".join(invalid_versions)
        )
    required_packages = {backend}
    identity = f"{backend} {runtime}".lower()
    if backend.lower() in {"sglang", "vllm"}:
        required_packages.update(("torch", "transformers"))
    if "mlx" in identity:
        required_packages.update(("mlx", "mlx-lm"))
    missing = sorted(required_packages.difference(software_payload))
    if missing:
        raise ValueError(
            "server environment is missing required software versions: " + ", ".join(missing)
        )
    if software_payload[backend] != backend_version:
        raise ValueError("server environment backend version does not match the probe declaration")

    capture_method = payload.get("capture_method")
    if not isinstance(capture_method, str) or not capture_method:
        raise ValueError("server environment capture_method must be non-empty")
    normalized = {
        "captured_at_utc": captured_at,
        "platform": normalized_platform,
        "hardware": normalized_hardware,
        "python": normalized_python,
        "software": dict(sorted(software_payload.items())),
        "provenance": {
            "scope": "serving_process_and_host_declared_by_operator",
            "capture_method": capture_method,
            "endpoint_reported_environment": False,
        },
    }
    secret_path = sensitive_literal_path(normalized)
    if secret_path is not None:
        raise ValueError(
            "server environment contains credential-like material at " + ".".join(secret_path)
        )
    return normalized


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def validate_base_url(base_url: str, *, allow_remote_endpoint: bool) -> str:
    """Require an explicit opt-in before any prompt leaves the loopback host."""

    normalized = base_url.rstrip("/")
    parsed = urllib.parse.urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("serving base URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("serving base URL must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("serving base URL must not contain a query or fragment")
    hostname = parsed.hostname
    loopback = hostname == "localhost"
    if not loopback:
        try:
            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = False
    if not loopback and not allow_remote_endpoint:
        raise ValueError(
            "non-loopback serving endpoints require --allow-remote-endpoint because prompt "
            "content will leave this host"
        )
    return normalized


def generation_request(prompt: str, *, seed: int, max_new_tokens: int) -> dict[str, Any]:
    """Build the deterministic native SGLang generation request."""

    return {
        "text": prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "sampling_seed": seed,
        },
    }


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout_seconds: float,
) -> tuple[dict[str, Any], float]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status != 200:
                raise ValueError(f"serving endpoint returned HTTP {response.status}")
            parsed = json.loads(response.read())
    except urllib.error.URLError as error:
        raise RuntimeError(f"serving endpoint request failed: {error}") from error
    elapsed = time.perf_counter() - started
    if not isinstance(parsed, dict):
        raise ValueError("serving endpoint response was not a JSON object")
    return parsed, elapsed


def probe(
    *,
    base_url: str,
    prompt: str,
    model_id: str,
    model_revision: str,
    model_checksum: str,
    backend: str,
    backend_version: str,
    backend_revision: str,
    runtime: str,
    seed: int,
    max_new_tokens: int,
    warmup_trials: int,
    measured_trials: int,
    timeout_seconds: float,
    server_environment: dict[str, Any],
    allow_remote_endpoint: bool = False,
    endpoint_model_path: str | None = None,
) -> dict[str, Any]:
    """Run warm-up and measured greedy generations against a local endpoint."""

    if warmup_trials < 0 or measured_trials < 1 or max_new_tokens < 1:
        raise ValueError("trial counts and max_new_tokens must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    if not model_id or not backend or not backend_version or not runtime:
        raise ValueError("model and backend identity fields must be non-empty")
    if not re.fullmatch(r"[0-9a-f]{40}", model_revision):
        raise ValueError("model_revision must be an immutable 40-character commit")
    if not re.fullmatch(r"[0-9a-f]{64}", model_checksum):
        raise ValueError("model_checksum must be a lowercase SHA-256 digest")
    if not re.fullmatch(r"[0-9a-f]{40}", backend_revision):
        raise ValueError("backend_revision must be an immutable 40-character commit")
    normalized_environment = normalize_server_environment(
        server_environment,
        backend=backend,
        backend_version=backend_version,
        runtime=runtime,
    )
    normalized_base = validate_base_url(base_url, allow_remote_endpoint=allow_remote_endpoint)
    model_payload, _ = _request_json(
        f"{normalized_base}/model_info", timeout_seconds=timeout_seconds
    )
    model_info = normalize_model_info(model_payload, endpoint_model_path or model_id)
    request_payload = generation_request(prompt, seed=seed, max_new_tokens=max_new_tokens)
    for _ in range(warmup_trials):
        _request_json(
            f"{normalized_base}/generate",
            method="POST",
            payload=request_payload,
            timeout_seconds=timeout_seconds,
        )
    trials: list[dict[str, Any]] = []
    for _ in range(measured_trials):
        response, elapsed = _request_json(
            f"{normalized_base}/generate",
            method="POST",
            payload=request_payload,
            timeout_seconds=timeout_seconds,
        )
        trials.append(normalize_generation(response, elapsed))

    client_latencies = [float(trial["client_elapsed_seconds"]) for trial in trials]
    server_latencies = [float(trial["server_e2e_seconds"]) for trial in trials]
    return {
        "schema_version": "optimized-serving-smoke-v2",
        "observed_at_utc": _utc_now(),
        "backend": {
            "name": backend,
            "version": backend_version,
            "source_revision": backend_revision,
            "runtime": runtime,
            "identity_reported_by_endpoint": False,
        },
        "model": {
            "model_id": model_id,
            "declared_revision": model_revision,
            "declared_checksum": model_checksum,
            "revision_reported_by_endpoint": False,
            **model_info,
        },
        "protocol": {
            "seed": seed,
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "warmup_trials": warmup_trials,
            "measured_trials": measured_trials,
            "prompt_sha256": text_sha256(prompt),
            "probe_record_contains_prompt_or_output_content": False,
            "server_request_logging_verified": False,
            "cache_policy": "server_configuration",
        },
        "server_environment": normalized_environment,
        "results": {
            "client_elapsed_seconds": {
                "median": statistics.median(client_latencies),
                "minimum": min(client_latencies),
                "maximum": max(client_latencies),
            },
            "server_e2e_seconds": {
                "median": statistics.median(server_latencies),
                "minimum": min(server_latencies),
                "maximum": max(server_latencies),
            },
            "unique_output_sha256_count": len({str(trial["output_sha256"]) for trial in trials}),
            "trials": trials,
        },
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:43440")
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="Read prompt text from this untracked file; omit to read it from standard input.",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--endpoint-model-path",
        help=(
            "Exact /model_info path when the server was launched from an immutable local snapshot."
        ),
    )
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-checksum", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--backend-version", required=True)
    parser.add_argument("--backend-revision", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument(
        "--server-environment-file",
        type=Path,
        required=True,
        help=(
            "JSON with captured_at_utc, platform, hardware, Python, and serving-process "
            "software versions; do not include environment variables or secrets."
        ),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup-trials", type=int, default=1)
    parser.add_argument("--measured-trials", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--allow-remote-endpoint",
        action="store_true",
        help="Explicitly authorize sending prompt content to a non-loopback endpoint.",
    )
    return parser.parse_args()


def main() -> None:
    """Probe the configured endpoint and emit a content-redacted JSON record."""

    arguments = _arguments()
    prompt = (
        arguments.prompt_file.read_text(encoding="utf-8")
        if arguments.prompt_file is not None
        else sys.stdin.read()
    )
    if not prompt:
        raise ValueError("prompt input must not be empty")
    environment_payload = json.loads(arguments.server_environment_file.read_text(encoding="utf-8"))
    if not isinstance(environment_payload, dict):
        raise ValueError("server environment file must contain a JSON object")
    record = probe(
        base_url=arguments.base_url,
        prompt=prompt,
        model_id=arguments.model_id,
        model_revision=arguments.model_revision,
        model_checksum=arguments.model_checksum,
        backend=arguments.backend,
        backend_version=arguments.backend_version,
        backend_revision=arguments.backend_revision,
        runtime=arguments.runtime,
        seed=arguments.seed,
        max_new_tokens=arguments.max_new_tokens,
        warmup_trials=arguments.warmup_trials,
        measured_trials=arguments.measured_trials,
        timeout_seconds=arguments.timeout_seconds,
        server_environment=environment_payload,
        allow_remote_endpoint=arguments.allow_remote_endpoint,
        endpoint_model_path=arguments.endpoint_model_path,
    )
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
