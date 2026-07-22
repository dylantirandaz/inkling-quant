#!/usr/bin/env python3
"""Verify two provisional public-MoE quality records without retaining raw outcomes.

The evaluator deliberately emits one provisional record per clean process.  This
module is the separate, fail-closed acceptance step: it validates ordinals one
and two, compares explicit scientific and environment projections, and writes a
small content-redacted aggregate.  A failed comparison is evidence too, so the
CLI publishes the failed aggregate before returning a non-zero exit status.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

_RAW_SCHEMA = "public-moe-native-linear-quality-v2"
_AGGREGATE_SCHEMA = "public-moe-native-linear-quality-repeat-verification-v1"
_COMPARISON_SCHEMA = "public-moe-native-linear-quality-repeat-comparison-v1"
_PROTOCOL_ID = "stories15m-native-int8-confirmatory-quality-v1"
_PROVISIONAL_STATUS = "provisional_until_2_clean_executions"
_NONINFERIORITY_SCOPE = "within_execution_statistical_gate_only_not_overall_confirmatory_claim"
_UPPER_BOUND_LABEL = "nominal_one_sided_95_percent_bootstrap_upper_bound"
_COVERAGE_INTERPRETATION = (
    "nominal_approximate_coverage_under_the_predeclared_stratified_"
    "finite_population_bootstrap_design"
)
_SHA256_LENGTH = 64
_PREREGISTRATION = Path(
    "docs/experiments/stories15m-native-int8-confirmatory-256-preregistration.json"
)
_ATTEMPT_ROOT = Path("artifacts/research-slices/stories15m-native-int8-confirmatory-256")
_AGGREGATE_PATH = _ATTEMPT_ROOT / "repeat-verification.json"
_BASELINE_CONFIG_SHA256 = "e6db2959221babf9aaba1f529a20349fe3462d4309b6929caf3a7d3331d668f2"
_CANDIDATE_CONFIG_SHA256 = "d6ee2054d6db801c56596042df7e9eca130109ab922ee46735b61953da8d7912"
_DATASET_SHA256 = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
_DATASET_SIZE_BYTES = 19_447_282
_DATASET_STORY_COUNT = 21_990
_HARDWARE_LABEL = "Apple M3 MacBook Air (8 cores, 16 GB)"
_OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONHASHSEED": "0",
    "PYTHONPYCACHEPREFIX": "/dev/null",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
}
_SOURCE_IMPORT_CONTRACT_STATIC = {
    "python_flags": ["-s", "-P", "-B"],
    "sys_flags": {
        "no_user_site": True,
        "safe_path": True,
        "dont_write_bytecode": True,
    },
    "sys_pycache_prefix": "/dev/null",
    "forbidden_ambient_import_environment_absent": [
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUSERBASE",
    ],
    "environment": {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": "/dev/null",
        "PYTHONNOUSERSITE": "1",
    },
    "local_project_imports_deferred_until_after_entry_guard": True,
}
_RUNNER = Path("scripts/run_confirmatory_quality_attempt.py")
_EVALUATOR = Path("scripts/evaluate_public_moe_native_quality.py")
_STATISTICAL_MODULE = Path("src/inkling_quant_lab/evaluation/noninferiority.py")
_REPEAT_VERIFIER = Path("scripts/verify_confirmatory_quality_repeats.py")
_BASELINE_CONFIG = Path(
    "configs/experiments/hf_stories15m_tinystories_native_confirmatory_256.yaml"
)
_CANDIDATE_CONFIG = Path(
    "configs/experiments/hf_stories15m_tinystories_native_int8_confirmatory_256.yaml"
)
_PROTOCOL_CONFIG = Path("configs/evaluations/stories15m_native_int8_confirmatory_256.yaml")
_REQUIRED_BOUND_FILES = (
    Path("scripts/__init__.py"),
    _RUNNER,
    _EVALUATOR,
    _STATISTICAL_MODULE,
    _REPEAT_VERIFIER,
    _BASELINE_CONFIG,
    _CANDIDATE_CONFIG,
    _PROTOCOL_CONFIG,
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("configs/models/hf_stories15m_moe.yaml"),
    Path("configs/quantization/native_dynamic_int8.yaml"),
)
_SOURCE_CLOSURE_ROOT = Path("src/inkling_quant_lab")
_ATTEMPT_FILES = {
    "start.json",
    "completion.json",
    "record.json",
    "stdout.log",
    "stderr.log",
}
_COMPLETION_CLAIM_BOUNDARY = (
    "attempt execution status only; the within-execution statistical decision is not "
    "an overall confirmatory claim"
)
_TOP_LEVEL_KEYS = {
    "schema_version",
    "created_at_utc",
    "baseline_config",
    "baseline_config_hash",
    "candidate_configs",
    "model",
    "dataset",
    "protocol",
    "inventory",
    "baseline",
    "candidates",
    "environment",
    "elapsed_seconds_including_model_load_and_all_candidates",
    "limitations",
    "confirmatory_status",
}
_STATUS_KEYS = {
    "status",
    "confirmatory_claim_ready",
    "required_clean_process_executions",
    "execution_ordinal",
    "within_execution_noninferiority_passed",
    "within_execution_decision_is_overall_confirmatory_claim",
    "pair_verification_required",
}
_CONFIRMATORY_KEYS = {
    "definition",
    "definition_sha256",
    "selection",
    "noninferiority_result",
    "noninferiority_result_scope",
    "upper_bound_label",
    "coverage_interpretation",
}
_TORCH_EXECUTION_KEYS = {
    "intraop_threads",
    "interop_threads",
    "quantized_engine",
    "supported_quantized_engines",
    "kleidiai_available",
}


class RepeatVerificationError(ValueError):
    """A raw record is malformed or violates the frozen repeat contract."""


@dataclass(frozen=True, slots=True)
class _RawInput:
    requested: Path
    resolved: Path
    display_path: str
    size_bytes: int | None
    sha256: str | None
    record: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True, slots=True)
class _ValidatedRecord:
    ordinal: Literal[1, 2]
    within_execution_passed: bool
    scientific_projection_sha256: str
    environment_projection_sha256: str


@dataclass(frozen=True, slots=True)
class _JsonEvidence:
    path: Path
    display_path: str
    size_bytes: int | None
    sha256: str | None
    value: dict[str, Any] | None
    error: str | None


@dataclass(frozen=True, slots=True)
class _PreregistrationBindings:
    protocol_definition_sha256: str
    baseline_config_sha256: str
    candidate_config_sha256: str
    dataset: dict[str, Any]
    environment: dict[str, Any]
    file_sha256: dict[str, str]


@dataclass(frozen=True, slots=True)
class _AttemptEnvelope:
    ordinal: Literal[1, 2]
    directory: str
    start: _JsonEvidence
    completion: _JsonEvidence
    record: _JsonEvidence
    stdout: _JsonEvidence
    stderr: _JsonEvidence


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _reject_json_constant(value: str) -> object:
    raise RepeatVerificationError(f"non-finite JSON constant is forbidden: {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RepeatVerificationError(f"duplicate JSON object key is forbidden: {key}")
        result[key] = value
    return result


def _require_finite_json(value: object, *, path: str = "record") -> None:
    if isinstance(value, str) and any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise RepeatVerificationError(f"invalid Unicode surrogate at {path}")
    if isinstance(value, float) and not math.isfinite(value):
        raise RepeatVerificationError(f"non-finite number at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise RepeatVerificationError(f"non-string object key at {path}")
            if any(0xD800 <= ord(character) <= 0xDFFF for character in key):
                raise RepeatVerificationError(f"invalid Unicode surrogate in key at {path}")
            _require_finite_json(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, path=f"{path}[{index}]")


def _parse_strict_json(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RepeatVerificationError("raw record is not valid UTF-8") from error
    try:
        parsed = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except json.JSONDecodeError as error:
        raise RepeatVerificationError(f"raw record is not valid JSON: {error.msg}") from error
    if not isinstance(parsed, dict):
        raise RepeatVerificationError("raw record JSON root must be an object")
    _require_finite_json(parsed)
    return parsed


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _path_has_symlink(path: Path, *, project_root: Path) -> bool:
    try:
        relative = path.relative_to(project_root)
    except ValueError:
        return True
    current = project_root
    if current.is_symlink():
        return True
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _read_evidence_file(
    path: Path,
    *,
    project_root: Path,
    parse_json: bool,
) -> _JsonEvidence:
    display = _display_path(path, project_root)
    if _path_has_symlink(path, project_root=project_root):
        return _JsonEvidence(path, display, None, None, None, "path contains a symlink")
    if not path.is_file():
        return _JsonEvidence(path, display, None, None, None, "not a regular file")
    try:
        payload = path.read_bytes()
    except OSError:
        return _JsonEvidence(path, display, None, None, None, "file cannot be read")
    digest = hashlib.sha256(payload).hexdigest()
    if not parse_json:
        return _JsonEvidence(path, display, len(payload), digest, None, None)
    try:
        value = _parse_strict_json(payload)
    except RepeatVerificationError as error:
        return _JsonEvidence(path, display, len(payload), digest, None, str(error))
    return _JsonEvidence(path, display, len(payload), digest, value, None)


def _fixed_requested_path(
    requested: Path,
    *,
    project_root: Path,
    expected_relative: Path,
    kind: Literal["file", "directory"],
) -> tuple[Path, str | None]:
    if requested.is_absolute():
        candidate = requested.expanduser().absolute()
    else:
        candidate = (project_root / requested).absolute()
    expected = (project_root / expected_relative).absolute()
    if ".." in requested.parts or candidate != expected:
        return expected, f"requested_path_is_not_fixed:{expected_relative.as_posix()}"
    if _path_has_symlink(expected, project_root=project_root):
        return expected, f"fixed_{kind}_path_contains_a_symlink:{expected_relative.as_posix()}"
    exists_as_kind = expected.is_file() if kind == "file" else expected.is_dir()
    if not exists_as_kind:
        return expected, f"fixed_{kind}_is_missing:{expected_relative.as_posix()}"
    return expected, None


def _read_raw_input(requested: Path, *, project_root: Path) -> _RawInput:
    candidate = requested if requested.is_absolute() else project_root / requested
    absolute = candidate.absolute()
    resolved = absolute.resolve(strict=False)
    display = _display_path(absolute, project_root)
    if absolute.is_symlink():
        return _RawInput(requested, resolved, display, None, None, None, "input is a symlink")
    if not absolute.is_file():
        return _RawInput(
            requested,
            resolved,
            display,
            None,
            None,
            None,
            "input is not a regular file",
        )
    try:
        payload = absolute.read_bytes()
    except OSError:
        return _RawInput(requested, resolved, display, None, None, None, "input cannot be read")
    digest = hashlib.sha256(payload).hexdigest()
    try:
        record = _parse_strict_json(payload)
    except RepeatVerificationError as error:
        return _RawInput(requested, resolved, display, len(payload), digest, None, str(error))
    return _RawInput(requested, resolved, display, len(payload), digest, record, None)


def _mapping(value: object, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RepeatVerificationError(f"{path} must be an object")
    return value


def _list(value: object, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise RepeatVerificationError(f"{path} must be an array")
    return value


def _field(mapping: Mapping[str, Any], key: str, *, path: str) -> Any:
    if key not in mapping:
        raise RepeatVerificationError(f"missing required field: {path}.{key}")
    return mapping[key]


def _string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise RepeatVerificationError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RepeatVerificationError(f"{path} must be an integer")
    return value


def _boolean(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise RepeatVerificationError(f"{path} must be a boolean")
    return value


def _sha256(value: object, *, path: str) -> str:
    digest = _string(value, path=path)
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise RepeatVerificationError(f"{path} must be a lowercase SHA-256 digest")
    return digest


def _number(value: object, *, path: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RepeatVerificationError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        qualifier = "nonnegative finite" if nonnegative else "finite"
        raise RepeatVerificationError(f"{path} must be a {qualifier} number")
    return number


def _exact_mapping(
    value: object,
    expected: set[str],
    *,
    path: str,
) -> dict[str, Any]:
    mapping = _mapping(value, path=path)
    if set(mapping) != expected:
        raise RepeatVerificationError(f"{path} has missing or unknown fields")
    return mapping


def _timestamp(value: object, *, path: str) -> str:
    text = _string(value, path=path)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise RepeatVerificationError(f"{path} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise RepeatVerificationError(f"{path} must include a timezone")
    return text


def _required_mapping(root: Mapping[str, Any], *parts: str) -> dict[str, Any]:
    current: object = root
    traversed: list[str] = []
    for part in parts:
        traversed.append(part)
        current = _field(
            _mapping(current, path=".".join(traversed[:-1]) or "record"),
            part,
            path=".".join(traversed[:-1]) or "record",
        )
    return _mapping(current, path=".".join(parts))


def _required_list(root: Mapping[str, Any], *parts: str) -> list[Any]:
    current: object = root
    traversed: list[str] = []
    for part in parts:
        traversed.append(part)
        current = _field(
            _mapping(current, path=".".join(traversed[:-1]) or "record"),
            part,
            path=".".join(traversed[:-1]) or "record",
        )
    return _list(current, path=".".join(parts))


def _validate_required_scientific_fields(record: dict[str, Any]) -> None:
    _timestamp(record["created_at_utc"], path="created_at_utc")
    _string(record["baseline_config"], path="baseline_config")
    baseline_config_hash = _sha256(record["baseline_config_hash"], path="baseline_config_hash")
    candidate_configs = _list(record["candidate_configs"], path="candidate_configs")
    candidates = _list(record["candidates"], path="candidates")
    if len(candidate_configs) != 1 or len(candidates) != 1:
        raise RepeatVerificationError(
            "exactly one candidate config and candidate result are required"
        )
    candidate_config = _mapping(candidate_configs[0], path="candidate_configs[0]")
    candidate = _mapping(candidates[0], path="candidates[0]")
    candidate_name = _string(
        _field(candidate_config, "name", path="candidate_configs[0]"),
        path="candidate_configs[0].name",
    )
    candidate_hash = _sha256(
        _field(candidate_config, "config_hash", path="candidate_configs[0]"),
        path="candidate_configs[0].config_hash",
    )
    if _field(candidate, "config_name", path="candidates[0]") != candidate_name:
        raise RepeatVerificationError("candidate config name does not match candidate result")
    if _field(candidate, "config_hash", path="candidates[0]") != candidate_hash:
        raise RepeatVerificationError("candidate config hash does not match candidate result")

    for name in ("model", "dataset", "inventory", "protocol", "baseline", "environment"):
        _mapping(record[name], path=name)
    _list(record["limitations"], path="limitations")
    _string(_field(record["model"], "model_id", path="model"), path="model.model_id")
    _string(_field(record["model"], "revision", path="model"), path="model.revision")
    _sha256(
        _field(record["model"], "loaded_float32_state_sha256", path="model"),
        path="model.loaded_float32_state_sha256",
    )
    _sha256(
        _field(record["dataset"], "file_sha256", path="dataset"),
        path="dataset.file_sha256",
    )

    baseline = _mapping(record["baseline"], path="baseline")
    _field(baseline, "load_time_seconds", path="baseline")
    _field(baseline, "measurement_elapsed_seconds", path="baseline")
    _integer(
        _field(baseline, "runtime_tensor_storage_bytes", path="baseline"),
        path="baseline.runtime_tensor_storage_bytes",
    )
    baseline_quality = _required_mapping(baseline, "quality")
    if not _required_list(baseline_quality, "samples"):
        raise RepeatVerificationError("baseline.quality.samples must not be empty")
    baseline_generation = _required_mapping(baseline, "generation")
    if not _required_list(baseline_generation, "samples"):
        raise RepeatVerificationError("baseline.generation.samples must not be empty")
    _required_mapping(baseline, "routing")
    _required_mapping(baseline, "routing_observer_proof")

    for name in ("config_name", "backend", "method"):
        _string(_field(candidate, name, path="candidates[0]"), path=f"candidates[0].{name}")
    _sha256(
        _field(candidate, "config_hash", path="candidates[0]"), path="candidates[0].config_hash"
    )
    _required_mapping(candidate, "support")
    quantization = _required_mapping(candidate, "quantization")
    _field(quantization, "quantization_elapsed_seconds", path="candidates[0].quantization")
    _sha256(
        _field(quantization, "candidate_state_sha256", path="candidates[0].quantization"),
        path="candidates[0].quantization.candidate_state_sha256",
    )
    for name in ("runtime_tensor_storage_bytes", "safe_bundle_serialized_size_bytes"):
        _integer(
            _field(quantization, name, path="candidates[0].quantization"),
            path=f"candidates[0].quantization.{name}",
        )
    _required_mapping(quantization, "manifest")
    candidate_quality = _required_mapping(candidate, "quality")
    if not _required_list(candidate_quality, "samples"):
        raise RepeatVerificationError("candidates[0].quality.samples must not be empty")
    generation_retention = _required_mapping(candidate, "generation_retention")
    if not _required_list(generation_retention, "samples"):
        raise RepeatVerificationError(
            "candidates[0].generation_retention.samples must not be empty"
        )
    _required_mapping(candidate, "routing")
    _required_mapping(candidate, "routing_observer_proof")
    _required_mapping(candidate, "noninferiority")
    _field(candidate, "measurement_elapsed_seconds", path="candidates[0]")

    confirmatory = _required_mapping(record["protocol"], "confirmatory")
    if set(confirmatory) != _CONFIRMATORY_KEYS:
        raise RepeatVerificationError("protocol.confirmatory has missing or unknown fields")
    definition = _required_mapping(confirmatory, "definition")
    definition_digest = _sha256(
        _field(confirmatory, "definition_sha256", path="protocol.confirmatory"),
        path="protocol.confirmatory.definition_sha256",
    )
    if _canonical_sha256(definition) != definition_digest:
        raise RepeatVerificationError(
            "protocol definition digest does not match its canonical JSON"
        )
    if _field(definition, "protocol_id", path="protocol confirmatory definition") != _PROTOCOL_ID:
        raise RepeatVerificationError("protocol definition identity is invalid")
    selection = _required_mapping(confirmatory, "selection")
    for name in (
        "quality_input_token_ids_manifest_sha256",
        "routing_input_token_ids_manifest_sha256",
        "generation_prompt_token_ids_manifest_sha256",
    ):
        _sha256(
            _field(selection, name, path="protocol.confirmatory.selection"),
            path=f"protocol.confirmatory.selection.{name}",
        )
    if (
        _field(confirmatory, "noninferiority_result_scope", path="protocol.confirmatory")
        != _NONINFERIORITY_SCOPE
    ):
        raise RepeatVerificationError("protocol.confirmatory noninferiority scope is invalid")
    if (
        _field(confirmatory, "upper_bound_label", path="protocol.confirmatory")
        != _UPPER_BOUND_LABEL
    ):
        raise RepeatVerificationError("protocol.confirmatory upper-bound label is invalid")
    if (
        _field(confirmatory, "coverage_interpretation", path="protocol.confirmatory")
        != _COVERAGE_INTERPRETATION
    ):
        raise RepeatVerificationError("protocol.confirmatory coverage interpretation is invalid")
    protocol_result = _required_mapping(confirmatory, "noninferiority_result")
    if protocol_result != candidate["noninferiority"]:
        raise RepeatVerificationError("candidate and protocol noninferiority results differ")

    execution_contract = _required_mapping(definition, "execution_contract")
    if (
        _field(
            execution_contract,
            "baseline_resolved_config_sha256",
            path="protocol definition execution_contract",
        )
        != baseline_config_hash
    ):
        raise RepeatVerificationError(
            "baseline config hash is not bound by the protocol definition"
        )
    if (
        _field(
            execution_contract,
            "candidate_resolved_config_sha256",
            path="protocol definition execution_contract",
        )
        != candidate_hash
    ):
        raise RepeatVerificationError(
            "candidate config hash is not bound by the protocol definition"
        )
    model_definition = _required_mapping(definition, "model")
    dataset_definition = _required_mapping(definition, "dataset")
    for key in ("model_id", "revision", "source_weight_sha256", "source_weight_size_bytes"):
        if _field(model_definition, key, path="protocol definition model") != _field(
            record["model"], key, path="model"
        ):
            raise RepeatVerificationError(f"protocol model binding differs for {key}")
    for key in ("dataset_id", "revision", "split", "file_sha256"):
        record_key = "dataset_id" if key == "dataset_id" else key
        if _field(dataset_definition, key, path="protocol definition dataset") != _field(
            record["dataset"], record_key, path="dataset"
        ):
            raise RepeatVerificationError(f"protocol dataset binding differs for {key}")


def scientific_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Return the raw record minus only declared nondeterministic timing/status fields."""

    projection = copy.deepcopy(record)
    projection.pop("created_at_utc")
    projection.pop("environment")
    projection.pop("confirmatory_status")
    projection.pop("elapsed_seconds_including_model_load_and_all_candidates")
    baseline = _mapping(projection["baseline"], path="baseline")
    baseline.pop("load_time_seconds")
    baseline.pop("measurement_elapsed_seconds")
    for candidate_value in _list(projection["candidates"], path="candidates"):
        candidate = _mapping(candidate_value, path="candidates[]")
        candidate.pop("measurement_elapsed_seconds")
        _mapping(candidate["quantization"], path="candidates[].quantization").pop(
            "quantization_elapsed_seconds"
        )
    return projection


def environment_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Select the environment identity required to agree across clean processes."""

    environment = _mapping(record["environment"], path="environment")
    python = _required_mapping(environment, "python")
    python_projection = {
        "version": _string(
            _field(python, "version", path="environment.python"),
            path="environment.python.version",
        ),
        "implementation": _string(
            _field(python, "implementation", path="environment.python"),
            path="environment.python.implementation",
        ),
    }
    platform = _required_mapping(environment, "platform")
    hardware = _required_mapping(environment, "hardware")
    packages = _required_mapping(environment, "packages")
    software = _required_mapping(environment, "software")
    runtime = _required_mapping(environment, "runtime_capability")
    label = _string(
        _field(environment, "operator_declared_hardware_label", path="environment"),
        path="environment.operator_declared_hardware_label",
    )
    label_scope = _string(
        _field(environment, "operator_hardware_label_scope", path="environment"),
        path="environment.operator_hardware_label_scope",
    )
    torch_execution = _required_mapping(environment, "torch_cpu_execution")
    if set(torch_execution) != _TORCH_EXECUTION_KEYS:
        raise RepeatVerificationError(
            "environment.torch_cpu_execution has missing or unknown fields"
        )
    _integer(
        _field(torch_execution, "intraop_threads", path="environment.torch_cpu_execution"),
        path="environment.torch_cpu_execution.intraop_threads",
    )
    _integer(
        _field(torch_execution, "interop_threads", path="environment.torch_cpu_execution"),
        path="environment.torch_cpu_execution.interop_threads",
    )
    if (
        _field(torch_execution, "quantized_engine", path="environment.torch_cpu_execution")
        != "qnnpack"
    ):
        raise RepeatVerificationError("environment current quantized engine must be qnnpack")
    supported = _list(
        _field(
            torch_execution,
            "supported_quantized_engines",
            path="environment.torch_cpu_execution",
        ),
        path="environment.torch_cpu_execution.supported_quantized_engines",
    )
    if "qnnpack" not in supported or any(not isinstance(item, str) for item in supported):
        raise RepeatVerificationError(
            "environment supported quantized engines must include qnnpack"
        )
    _boolean(
        _field(torch_execution, "kleidiai_available", path="environment.torch_cpu_execution"),
        path="environment.torch_cpu_execution.kleidiai_available",
    )
    return {
        "python": python_projection,
        "platform": copy.deepcopy(platform),
        "hardware": copy.deepcopy(hardware),
        "packages": copy.deepcopy(packages),
        "software": copy.deepcopy(software),
        "runtime_capability": copy.deepcopy(runtime),
        "operator_declared_hardware_label": label,
        "operator_hardware_label_scope": label_scope,
        "torch_cpu_execution": copy.deepcopy(torch_execution),
    }


def _validate_record(record: dict[str, Any]) -> _ValidatedRecord:
    if set(record) != _TOP_LEVEL_KEYS:
        raise RepeatVerificationError("raw v2 record has missing or unknown top-level fields")
    if record["schema_version"] != _RAW_SCHEMA:
        raise RepeatVerificationError(f"schema_version must be {_RAW_SCHEMA}")
    _validate_required_scientific_fields(record)
    status = _mapping(record["confirmatory_status"], path="confirmatory_status")
    if set(status) != _STATUS_KEYS:
        raise RepeatVerificationError("confirmatory_status has missing or unknown fields")
    if status["status"] != _PROVISIONAL_STATUS:
        raise RepeatVerificationError("raw record status must remain provisional")
    if _boolean(
        status["confirmatory_claim_ready"],
        path="confirmatory_status.confirmatory_claim_ready",
    ):
        raise RepeatVerificationError("raw record must not claim confirmatory readiness")
    if (
        _integer(
            status["required_clean_process_executions"],
            path="confirmatory_status.required_clean_process_executions",
        )
        != 2
    ):
        raise RepeatVerificationError("raw record must require exactly two clean executions")
    ordinal = _integer(status["execution_ordinal"], path="confirmatory_status.execution_ordinal")
    if ordinal not in (1, 2):
        raise RepeatVerificationError("execution ordinal must be 1 or 2")
    passed = _boolean(
        status["within_execution_noninferiority_passed"],
        path="confirmatory_status.within_execution_noninferiority_passed",
    )
    if _boolean(
        status["within_execution_decision_is_overall_confirmatory_claim"],
        path="confirmatory_status.within_execution_decision_is_overall_confirmatory_claim",
    ):
        raise RepeatVerificationError("within-execution decision must not be an overall claim")
    if not _boolean(
        status["pair_verification_required"],
        path="confirmatory_status.pair_verification_required",
    ):
        raise RepeatVerificationError("raw record must require pair verification")
    candidate = _mapping(_list(record["candidates"], path="candidates")[0], path="candidates[0]")
    candidate_passed = _boolean(
        _required_mapping(candidate, "noninferiority")["passed"],
        path="candidates[0].noninferiority.passed",
    )
    if candidate_passed != passed:
        raise RepeatVerificationError("status and candidate noninferiority decisions differ")
    scientific_hash = _canonical_sha256(scientific_projection(record))
    environment_hash = _canonical_sha256(environment_projection(record))
    return _ValidatedRecord(
        ordinal=cast(Literal[1, 2], ordinal),
        within_execution_passed=passed,
        scientific_projection_sha256=scientific_hash,
        environment_projection_sha256=environment_hash,
    )


def _claim_boundary() -> dict[str, object]:
    return {
        "inference": (
            "nominal_one_sided_95_percent_moment_matched_paired_story_bootstrap_noninferiority"
        ),
        "coverage_qualification": ("nominal_bootstrap_bound_not_exact_finite_population_coverage"),
        "decision_rule": "both_within_execution_upper_bounds_strictly_less_than_margin",
        "boundaries": [
            "pinned_Stories15M_MOE_revision_and_TinyStories_finite_holdout_only",
            "attention_and_other_unprotected_concrete_linear_leaves_only",
            "fused_experts_routers_embeddings_normalization_and_output_head_remain_float32",
            "generation_and_routing_are_descriptive_only",
            "publisher_randomized_router_does_not_measure_learned_expert_specialization",
            "model_seed_hardware_and_cross_domain_uncertainty_are_not_covered",
            "no_latency_energy_cuda_awq_gptq_fp8_or_causal_claim",
        ],
    }


def _regular_file_identity(path: Path, *, field: str) -> tuple[str, int]:
    if path.is_symlink():
        raise RepeatVerificationError(f"{field} resolved target must not be a symlink")
    try:
        with path.open("rb") as handle:
            file_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise RepeatVerificationError(f"{field} resolved target must be regular")
            digest = hashlib.sha256()
            while block := handle.read(1024 * 1024):
                digest.update(block)
    except OSError as error:
        raise RepeatVerificationError(f"{field} resolved target cannot be read") from error
    return digest.hexdigest(), file_stat.st_size


def _validate_environment_contract(value: object, *, project_root: Path) -> dict[str, Any]:
    environment = _exact_mapping(
        value,
        {
            "hardware_label",
            "python",
            "platform",
            "software",
            "torch_cpu",
            "offline_environment",
            "python_executable",
            "virtual_environment",
        },
        path="preregistration.environment_contract",
    )
    if environment["hardware_label"] != _HARDWARE_LABEL:
        raise RepeatVerificationError("preregistration hardware label is invalid")
    python = _exact_mapping(
        environment["python"],
        {"implementation", "version"},
        path="preregistration.environment_contract.python",
    )
    _string(
        python["implementation"], path="preregistration.environment_contract.python.implementation"
    )
    _string(python["version"], path="preregistration.environment_contract.python.version")
    platform = _exact_mapping(
        environment["platform"],
        {"system", "release", "machine"},
        path="preregistration.environment_contract.platform",
    )
    for name in ("system", "release", "machine"):
        _string(platform[name], path=f"preregistration.environment_contract.platform.{name}")
    software = _mapping(
        environment["software"], path="preregistration.environment_contract.software"
    )
    if not software:
        raise RepeatVerificationError("preregistration software contract must not be empty")
    for name, version in software.items():
        _string(name, path="preregistration.environment_contract.software key")
        _string(version, path=f"preregistration.environment_contract.software.{name}")
    torch_cpu = _exact_mapping(
        environment["torch_cpu"],
        {"intraop_threads", "interop_threads", "required_supported_engine"},
        path="preregistration.environment_contract.torch_cpu",
    )
    for name in ("intraop_threads", "interop_threads"):
        if (
            _integer(torch_cpu[name], path=f"preregistration.environment_contract.torch_cpu.{name}")
            <= 0
        ):
            raise RepeatVerificationError("preregistered torch thread counts must be positive")
    if torch_cpu["required_supported_engine"] != "qnnpack":
        raise RepeatVerificationError("preregistered required quantized engine must be qnnpack")
    if environment["offline_environment"] != _OFFLINE_ENVIRONMENT:
        raise RepeatVerificationError("preregistered offline environment is invalid")
    executable = _exact_mapping(
        environment["python_executable"],
        {"invocation_path", "resolved_path", "sha256", "size_bytes"},
        path="preregistration.environment_contract.python_executable",
    )
    invocation_text = _string(
        executable["invocation_path"],
        path="preregistration.environment_contract.python_executable.invocation_path",
    )
    expected_invocation = (project_root / ".venv/bin/python").absolute()
    if invocation_text != str(expected_invocation):
        raise RepeatVerificationError(
            "preregistered Python invocation must be the project .venv/bin/python"
        )
    for directory in (project_root / ".venv", project_root / ".venv/bin"):
        if directory.is_symlink() or not directory.is_dir():
            raise RepeatVerificationError(
                "project virtual-environment path components must be real directories"
            )
    if not (expected_invocation.is_symlink() or expected_invocation.is_file()):
        raise RepeatVerificationError("project Python invocation path is missing")
    try:
        resolved_invocation = expected_invocation.resolve(strict=True)
    except OSError as error:
        raise RepeatVerificationError("project Python invocation cannot be resolved") from error
    resolved_text = _string(
        executable["resolved_path"],
        path="preregistration.environment_contract.python_executable.resolved_path",
    )
    if resolved_text != str(resolved_invocation):
        raise RepeatVerificationError("preregistered Python resolution is invalid")
    executable_sha256 = _sha256(
        executable["sha256"],
        path="preregistration.environment_contract.python_executable.sha256",
    )
    executable_size = _integer(
        executable["size_bytes"],
        path="preregistration.environment_contract.python_executable.size_bytes",
    )
    actual_executable_sha256, actual_executable_size = _regular_file_identity(
        resolved_invocation,
        field="preregistered Python executable",
    )
    if executable_size <= 0:
        raise RepeatVerificationError("preregistered Python executable size must be positive")
    if executable_sha256 != actual_executable_sha256 or executable_size != actual_executable_size:
        raise RepeatVerificationError(
            "preregistered Python executable identity differs from its resolved target"
        )
    virtual_environment = _exact_mapping(
        environment["virtual_environment"],
        {"pyvenv_cfg_path", "sha256", "size_bytes"},
        path="preregistration.environment_contract.virtual_environment",
    )
    if virtual_environment["pyvenv_cfg_path"] != ".venv/pyvenv.cfg":
        raise RepeatVerificationError("preregistered virtual environment path is invalid")
    pyvenv_sha256 = _sha256(
        virtual_environment["sha256"],
        path="preregistration.environment_contract.virtual_environment.sha256",
    )
    pyvenv_size = _integer(
        virtual_environment["size_bytes"],
        path="preregistration.environment_contract.virtual_environment.size_bytes",
    )
    pyvenv_path = (project_root / ".venv/pyvenv.cfg").absolute()
    if _path_has_symlink(pyvenv_path, project_root=project_root):
        raise RepeatVerificationError("project pyvenv.cfg path must not contain a symlink")
    actual_pyvenv_sha256, actual_pyvenv_size = _regular_file_identity(
        pyvenv_path,
        field="preregistered pyvenv.cfg",
    )
    if pyvenv_size <= 0:
        raise RepeatVerificationError("preregistered virtual environment size must be positive")
    if pyvenv_sha256 != actual_pyvenv_sha256 or pyvenv_size != actual_pyvenv_size:
        raise RepeatVerificationError(
            "preregistered virtual environment identity differs from pyvenv.cfg"
        )
    return environment


def _required_bound_paths(project_root: Path) -> set[str]:
    source_root = (project_root / _SOURCE_CLOSURE_ROOT).absolute()
    if _path_has_symlink(source_root, project_root=project_root) or not source_root.is_dir():
        raise RepeatVerificationError("bound Python source closure root is invalid")
    discovered: set[str] = set()
    try:
        paths = tuple(source_root.rglob("*"))
    except OSError as error:
        raise RepeatVerificationError("bound Python source closure cannot be read") from error
    for path in paths:
        if _path_has_symlink(path, project_root=project_root):
            raise RepeatVerificationError("bound Python source closure contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RepeatVerificationError(
                "bound Python source closure contains a non-regular entry"
            )
        if path.suffix == ".py":
            discovered.add(path.relative_to(project_root).as_posix())
    if not discovered:
        raise RepeatVerificationError("bound Python source closure contains no Python files")
    return {path.as_posix() for path in _REQUIRED_BOUND_FILES} | discovered


def _validate_preregistration(
    preregistration: _JsonEvidence,
    *,
    project_root: Path,
) -> _PreregistrationBindings:
    if preregistration.error is not None or preregistration.value is None:
        raise RepeatVerificationError(preregistration.error or "preregistration is unavailable")
    top = _exact_mapping(
        preregistration.value,
        {
            "schema_version",
            "status",
            "protocol_id",
            "created_at_utc",
            "outcomes",
            "bindings",
            "environment_contract",
            "planned_attempts",
            "primary_analysis",
            "claim_boundary",
        },
        path="preregistration",
    )
    if top["schema_version"] != "confirmatory-quality-preregistration-v1":
        raise RepeatVerificationError("preregistration schema identity is invalid")
    if top["status"] != "locked_before_execution" or top["protocol_id"] != _PROTOCOL_ID:
        raise RepeatVerificationError("preregistration lock identity is invalid")
    _timestamp(top["created_at_utc"], path="preregistration.created_at_utc")
    outcomes = _exact_mapping(
        top["outcomes"],
        {"holdout_model_forward_executed", "holdout_outcomes_inspected"},
        path="preregistration.outcomes",
    )
    if outcomes != {
        "holdout_model_forward_executed": False,
        "holdout_outcomes_inspected": False,
    }:
        raise RepeatVerificationError("preregistration is not outcome-blind")

    bindings = _exact_mapping(
        top["bindings"],
        {"files", "resolved_config_sha256", "protocol_definition_sha256", "dataset"},
        path="preregistration.bindings",
    )
    files = _mapping(bindings["files"], path="preregistration.bindings.files")
    required_paths = _required_bound_paths(project_root)
    missing_required = sorted(required_paths - set(files))
    if missing_required:
        raise RepeatVerificationError("preregistration omits required bound project files")
    unexpected = sorted(set(files) - required_paths)
    if unexpected:
        raise RepeatVerificationError("preregistration contains non-runtime bound project files")
    validated_files: dict[str, str] = {}
    for relative_text in sorted(required_paths):
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != relative_text:
            raise RepeatVerificationError("preregistration contains an unsafe bound-file path")
        declared = _sha256(
            files[relative_text], path=f"preregistration.bindings.files.{relative_text}"
        )
        current_path = (project_root / relative_text).absolute()
        current = _read_evidence_file(
            current_path,
            project_root=project_root,
            parse_json=False,
        )
        if current.error is not None or current.sha256 is None:
            raise RepeatVerificationError(f"bound project file is invalid: {relative_text}")
        if current.sha256 != declared:
            raise RepeatVerificationError(f"bound project file is stale: {relative_text}")
        validated_files[relative_text] = declared

    resolved = _exact_mapping(
        bindings["resolved_config_sha256"],
        {"baseline", "candidate"},
        path="preregistration.bindings.resolved_config_sha256",
    )
    baseline_hash = _sha256(
        resolved["baseline"],
        path="preregistration.bindings.resolved_config_sha256.baseline",
    )
    candidate_hash = _sha256(
        resolved["candidate"],
        path="preregistration.bindings.resolved_config_sha256.candidate",
    )
    if baseline_hash != _BASELINE_CONFIG_SHA256 or candidate_hash != _CANDIDATE_CONFIG_SHA256:
        raise RepeatVerificationError("preregistered resolved config identities are invalid")
    protocol_hash = _sha256(
        bindings["protocol_definition_sha256"],
        path="preregistration.bindings.protocol_definition_sha256",
    )
    dataset = _exact_mapping(
        bindings["dataset"],
        {"file_sha256", "size_bytes", "story_count"},
        path="preregistration.bindings.dataset",
    )
    _sha256(dataset["file_sha256"], path="preregistration.bindings.dataset.file_sha256")
    _integer(dataset["size_bytes"], path="preregistration.bindings.dataset.size_bytes")
    _integer(dataset["story_count"], path="preregistration.bindings.dataset.story_count")
    if dataset != {
        "file_sha256": _DATASET_SHA256,
        "size_bytes": _DATASET_SIZE_BYTES,
        "story_count": _DATASET_STORY_COUNT,
    }:
        raise RepeatVerificationError("preregistered dataset identity is invalid")
    environment = _validate_environment_contract(
        top["environment_contract"], project_root=project_root
    )

    expected_attempts = [
        {
            "ordinal": ordinal,
            "directory": (_ATTEMPT_ROOT / f"attempt-{ordinal}").as_posix(),
            "record_path": (_ATTEMPT_ROOT / f"attempt-{ordinal}" / "record.json").as_posix(),
        }
        for ordinal in (1, 2)
    ]
    if top["planned_attempts"] != expected_attempts:
        raise RepeatVerificationError("preregistered attempt paths are invalid")
    primary = _mapping(top["primary_analysis"], path="preregistration.primary_analysis")
    expected_repeat = {
        "required_clean_process_executions": 2,
        "input_records": [item["record_path"] for item in expected_attempts],
        "aggregate_path": _AGGREGATE_PATH.as_posix(),
        "single_execution_status": _PROVISIONAL_STATUS,
        "promotion_rule": (
            "both_within_execution_gates_pass_and_scientific_environment_projections_match_exactly"
        ),
        "repeat_is_additional_statistical_sample": False,
    }
    if (
        _field(primary, "repeat_verification", path="preregistration.primary_analysis")
        != expected_repeat
    ):
        raise RepeatVerificationError("preregistered repeat-verification rule is invalid")
    claim_boundary = _list(top["claim_boundary"], path="preregistration.claim_boundary")
    if not claim_boundary or any(not isinstance(item, str) or not item for item in claim_boundary):
        raise RepeatVerificationError("preregistered claim boundary must contain text entries")
    return _PreregistrationBindings(
        protocol_definition_sha256=protocol_hash,
        baseline_config_sha256=baseline_hash,
        candidate_config_sha256=candidate_hash,
        dataset=copy.deepcopy(dataset),
        environment=copy.deepcopy(environment),
        file_sha256=validated_files,
    )


def _require_evidence(evidence: _JsonEvidence, *, path: str) -> dict[str, Any]:
    if evidence.error is not None or evidence.value is None:
        raise RepeatVerificationError(f"{path} is invalid: {evidence.error or 'unavailable'}")
    if evidence.sha256 is None or evidence.size_bytes is None:
        raise RepeatVerificationError(f"{path} is missing its file identity")
    return evidence.value


def _validate_file_fact(
    value: object,
    *,
    expected_name: str,
    evidence: _JsonEvidence,
    path: str,
) -> None:
    fact = _exact_mapping(value, {"path", "sha256", "size_bytes"}, path=path)
    if fact["path"] != expected_name:
        raise RepeatVerificationError(f"{path}.path is invalid")
    digest = _sha256(fact["sha256"], path=f"{path}.sha256")
    size = _integer(fact["size_bytes"], path=f"{path}.size_bytes")
    if evidence.error is not None or digest != evidence.sha256 or size != evidence.size_bytes:
        raise RepeatVerificationError(f"{path} does not match the referenced file bytes")


def _validate_source_import_contract(
    value: object,
    *,
    bindings: _PreregistrationBindings,
    ordinal: Literal[1, 2],
) -> None:
    expected_keys = set(_SOURCE_IMPORT_CONTRACT_STATIC) | {"verified_local_modules"}
    contract = _exact_mapping(
        value,
        expected_keys,
        path=f"attempt-{ordinal}.start.source_import_contract",
    )
    for key, expected in _SOURCE_IMPORT_CONTRACT_STATIC.items():
        if contract[key] != expected:
            raise RepeatVerificationError(
                f"attempt-{ordinal} source import contract differs for {key}"
            )
    modules = _mapping(
        contract["verified_local_modules"],
        path=f"attempt-{ordinal}.start.source_import_contract.verified_local_modules",
    )
    if not {"inkling_quant_lab", "scripts"}.issubset(modules):
        raise RepeatVerificationError(
            f"attempt-{ordinal} source import contract omits required local packages"
        )
    for name, raw_fact in modules.items():
        if name not in {"inkling_quant_lab", "scripts"} and not name.startswith(
            ("inkling_quant_lab.", "scripts.")
        ):
            raise RepeatVerificationError(
                f"attempt-{ordinal} source import contract contains an unknown module"
            )
        fact = _exact_mapping(
            raw_fact,
            {"source_path", "source_sha256", "cached_path", "cached_exists"},
            path=f"attempt-{ordinal}.start.source_import_contract.verified_local_modules.{name}",
        )
        source_path = _string(
            fact["source_path"],
            path=f"attempt-{ordinal}.source_import_contract.{name}.source_path",
        )
        relative_source = Path(source_path)
        if (
            relative_source.is_absolute()
            or ".." in relative_source.parts
            or relative_source.suffix != ".py"
            or relative_source.as_posix() != source_path
        ):
            raise RepeatVerificationError(f"attempt-{ordinal} local module source path is unsafe")
        source_sha256 = _sha256(
            fact["source_sha256"],
            path=f"attempt-{ordinal}.source_import_contract.{name}.source_sha256",
        )
        if bindings.file_sha256.get(source_path) != source_sha256:
            raise RepeatVerificationError(
                f"attempt-{ordinal} local module source is absent or stale in preregistration"
            )
        cached_path = _string(
            fact["cached_path"],
            path=f"attempt-{ordinal}.source_import_contract.{name}.cached_path",
        )
        cached = Path(cached_path)
        if (
            not cached.is_absolute()
            or ".." in cached.parts
            or cached == Path("/dev/null")
            or not cached.is_relative_to(Path("/dev/null"))
            or fact["cached_exists"] is not False
        ):
            raise RepeatVerificationError(
                f"attempt-{ordinal} local module bytecode redirection is invalid"
            )


def _validate_start_record(
    evidence: _JsonEvidence,
    *,
    ordinal: Literal[1, 2],
    project_root: Path,
    preregistration: _JsonEvidence,
    bindings: _PreregistrationBindings,
) -> dict[str, Any]:
    top = _exact_mapping(
        _require_evidence(evidence, path=f"attempt-{ordinal}/start.json"),
        {
            "schema_version",
            "status",
            "created_at_utc",
            "ordinal",
            "attempt_directory",
            "planned_record_path",
            "command_argv",
            "shell",
            "working_directory",
            "preregistration",
            "dataset",
            "file_sha256",
            "resolved_config_sha256",
            "protocol_definition_sha256",
            "environment_contract",
            "selected_child_environment",
            "source_import_contract",
            "stdout_path",
            "stderr_path",
            "outcome_fields_present",
            "preflight_materialization_failure",
        },
        path=f"attempt-{ordinal}.start",
    )
    if top["schema_version"] != "confirmatory-quality-attempt-start-v1":
        raise RepeatVerificationError(f"attempt-{ordinal} start schema is invalid")
    if top["status"] != "started_before_preflight_and_model_execution":
        raise RepeatVerificationError(f"attempt-{ordinal} start status is invalid")
    _timestamp(top["created_at_utc"], path=f"attempt-{ordinal}.start.created_at_utc")
    if _integer(top["ordinal"], path=f"attempt-{ordinal}.start.ordinal") != ordinal:
        raise RepeatVerificationError(f"attempt-{ordinal} start ordinal is invalid")
    relative_directory = _ATTEMPT_ROOT / f"attempt-{ordinal}"
    relative_record = relative_directory / "record.json"
    if top["attempt_directory"] != relative_directory.as_posix():
        raise RepeatVerificationError(f"attempt-{ordinal} start directory is invalid")
    if top["planned_record_path"] != relative_record.as_posix():
        raise RepeatVerificationError(f"attempt-{ordinal} planned record path is invalid")
    if top["shell"] is not False or top["outcome_fields_present"] is not False:
        raise RepeatVerificationError(f"attempt-{ordinal} start execution boundary is invalid")
    if top["working_directory"] != str(project_root):
        raise RepeatVerificationError(f"attempt-{ordinal} working directory is invalid")
    if top["preflight_materialization_failure"] is not None:
        raise RepeatVerificationError(f"attempt-{ordinal} preflight materialization failed")
    if top["stdout_path"] != "stdout.log" or top["stderr_path"] != "stderr.log":
        raise RepeatVerificationError(f"attempt-{ordinal} log paths are invalid")

    prereg_fact = _exact_mapping(
        top["preregistration"],
        {"path", "sha256", "size_bytes"},
        path=f"attempt-{ordinal}.start.preregistration",
    )
    if prereg_fact["path"] != _PREREGISTRATION.as_posix():
        raise RepeatVerificationError(f"attempt-{ordinal} preregistration path is invalid")
    if (
        _sha256(prereg_fact["sha256"], path=f"attempt-{ordinal}.start.preregistration.sha256")
        != preregistration.sha256
        or _integer(
            prereg_fact["size_bytes"],
            path=f"attempt-{ordinal}.start.preregistration.size_bytes",
        )
        != preregistration.size_bytes
    ):
        raise RepeatVerificationError(
            f"attempt-{ordinal} does not bind the current preregistration bytes"
        )

    dataset = _exact_mapping(
        top["dataset"],
        {"path", "sha256", "size_bytes", "story_count"},
        path=f"attempt-{ordinal}.start.dataset",
    )
    _string(dataset["path"], path=f"attempt-{ordinal}.start.dataset.path")
    if not Path(dataset["path"]).is_absolute():
        raise RepeatVerificationError(f"attempt-{ordinal} dataset path must be absolute")
    if {
        "file_sha256": dataset["sha256"],
        "size_bytes": dataset["size_bytes"],
        "story_count": dataset["story_count"],
    } != bindings.dataset:
        raise RepeatVerificationError(
            f"attempt-{ordinal} dataset identity differs from preregistration"
        )

    files = _mapping(top["file_sha256"], path=f"attempt-{ordinal}.start.file_sha256")
    if files != bindings.file_sha256:
        raise RepeatVerificationError(
            f"attempt-{ordinal} bound-file hashes differ from preregistration"
        )
    resolved = _exact_mapping(
        top["resolved_config_sha256"],
        {"baseline", "candidate"},
        path=f"attempt-{ordinal}.start.resolved_config_sha256",
    )
    if resolved != {
        "baseline": bindings.baseline_config_sha256,
        "candidate": bindings.candidate_config_sha256,
    }:
        raise RepeatVerificationError(
            f"attempt-{ordinal} resolved configs differ from preregistration"
        )
    if top["protocol_definition_sha256"] != bindings.protocol_definition_sha256:
        raise RepeatVerificationError(f"attempt-{ordinal} protocol differs from preregistration")
    if top["environment_contract"] != bindings.environment:
        raise RepeatVerificationError(f"attempt-{ordinal} environment differs from preregistration")
    if top["selected_child_environment"] != _OFFLINE_ENVIRONMENT:
        raise RepeatVerificationError(f"attempt-{ordinal} child environment is invalid")
    _validate_source_import_contract(
        top["source_import_contract"], bindings=bindings, ordinal=ordinal
    )

    command = _list(top["command_argv"], path=f"attempt-{ordinal}.start.command_argv")
    if len(command) != 19 or any(not isinstance(item, str) for item in command):
        raise RepeatVerificationError(f"attempt-{ordinal} command argv schema is invalid")
    interpreter = _string(command[0], path=f"attempt-{ordinal}.start.command_argv[0]")
    if interpreter != bindings.environment["python_executable"]["invocation_path"]:
        raise RepeatVerificationError(
            f"attempt-{ordinal} command interpreter differs from preregistration"
        )
    expected_command = [
        interpreter,
        "-s",
        "-P",
        "-B",
        _EVALUATOR.as_posix(),
        _BASELINE_CONFIG.as_posix(),
        _CANDIDATE_CONFIG.as_posix(),
        "--dataset",
        dataset["path"],
        "--expected-dataset-sha256",
        _DATASET_SHA256,
        "--hardware-label",
        _HARDWARE_LABEL,
        "--confirmatory-protocol",
        _PROTOCOL_CONFIG.as_posix(),
        "--execution-ordinal",
        str(ordinal),
        "--output",
        relative_record.as_posix(),
    ]
    if command != expected_command:
        raise RepeatVerificationError(f"attempt-{ordinal} command does not match the fixed launch")
    return top


def _validate_completion_record(
    evidence: _JsonEvidence,
    *,
    ordinal: Literal[1, 2],
    start: _JsonEvidence,
    record: _JsonEvidence,
    stdout: _JsonEvidence,
    stderr: _JsonEvidence,
) -> dict[str, Any]:
    top = _exact_mapping(
        _require_evidence(evidence, path=f"attempt-{ordinal}/completion.json"),
        {
            "schema_version",
            "status",
            "created_at_utc",
            "ordinal",
            "attempt_directory",
            "subprocess_launched",
            "preflight_complete",
            "subprocess_returncode",
            "runner_exit_code",
            "elapsed_seconds",
            "failure",
            "stdout",
            "stderr",
            "start",
            "record",
            "confirmatory_claim_ready",
            "claim_boundary",
        },
        path=f"attempt-{ordinal}.completion",
    )
    if top["schema_version"] != "confirmatory-quality-attempt-completion-v1":
        raise RepeatVerificationError(f"attempt-{ordinal} completion schema is invalid")
    if top["status"] != "attempt_execution_complete":
        raise RepeatVerificationError(f"attempt-{ordinal} did not complete successfully")
    _timestamp(top["created_at_utc"], path=f"attempt-{ordinal}.completion.created_at_utc")
    if _integer(top["ordinal"], path=f"attempt-{ordinal}.completion.ordinal") != ordinal:
        raise RepeatVerificationError(f"attempt-{ordinal} completion ordinal is invalid")
    relative_directory = _ATTEMPT_ROOT / f"attempt-{ordinal}"
    if top["attempt_directory"] != relative_directory.as_posix():
        raise RepeatVerificationError(f"attempt-{ordinal} completion directory is invalid")
    if (
        top["subprocess_launched"] is not True
        or top["preflight_complete"] is not True
        or top["subprocess_returncode"] != 0
        or top["runner_exit_code"] != 0
        or top["failure"] is not None
    ):
        raise RepeatVerificationError(f"attempt-{ordinal} completion success fields are invalid")
    _number(
        top["elapsed_seconds"],
        path=f"attempt-{ordinal}.completion.elapsed_seconds",
        nonnegative=True,
    )
    if top["confirmatory_claim_ready"] is not False:
        raise RepeatVerificationError(f"attempt-{ordinal} completion made an early claim")
    if top["claim_boundary"] != _COMPLETION_CLAIM_BOUNDARY:
        raise RepeatVerificationError(f"attempt-{ordinal} completion claim boundary is invalid")
    _validate_file_fact(
        top["start"],
        expected_name="start.json",
        evidence=start,
        path=f"attempt-{ordinal}.completion.start",
    )
    _validate_file_fact(
        top["stdout"],
        expected_name="stdout.log",
        evidence=stdout,
        path=f"attempt-{ordinal}.completion.stdout",
    )
    _validate_file_fact(
        top["stderr"],
        expected_name="stderr.log",
        evidence=stderr,
        path=f"attempt-{ordinal}.completion.stderr",
    )
    record_fact = _exact_mapping(
        top["record"],
        {
            "exists",
            "is_regular_non_symlink",
            "size_bytes",
            "sha256",
            "schema_version",
            "execution_ordinal",
            "provisional_status",
            "confirmatory_claim_ready",
            "valid",
            "validation_error",
        },
        path=f"attempt-{ordinal}.completion.record",
    )
    if (
        record_fact["exists"] is not True
        or record_fact["is_regular_non_symlink"] is not True
        or record_fact["schema_version"] != _RAW_SCHEMA
        or record_fact["execution_ordinal"] != ordinal
        or record_fact["provisional_status"] != _PROVISIONAL_STATUS
        or record_fact["confirmatory_claim_ready"] is not False
        or record_fact["valid"] is not True
        or record_fact["validation_error"] is not None
    ):
        raise RepeatVerificationError(f"attempt-{ordinal} completion record status is invalid")
    if (
        _sha256(record_fact["sha256"], path=f"attempt-{ordinal}.completion.record.sha256")
        != record.sha256
        or _integer(
            record_fact["size_bytes"], path=f"attempt-{ordinal}.completion.record.size_bytes"
        )
        != record.size_bytes
        or record.error is not None
    ):
        raise RepeatVerificationError(f"attempt-{ordinal} completion does not match record bytes")
    return top


def _read_attempt_envelope(
    *,
    ordinal: Literal[1, 2],
    project_root: Path,
) -> tuple[_AttemptEnvelope, list[str]]:
    relative = _ATTEMPT_ROOT / f"attempt-{ordinal}"
    directory = (project_root / relative).absolute()
    reasons: list[str] = []
    if _path_has_symlink(directory, project_root=project_root) or not directory.is_dir():
        reasons.append(f"attempt_{ordinal}_fixed_directory_is_invalid")
    else:
        if stat.S_IMODE(directory.stat().st_mode) != 0o555:
            reasons.append(f"attempt_{ordinal}_directory_mode_is_not_0555")
        try:
            names = {entry.name for entry in directory.iterdir()}
        except OSError:
            names = set()
            reasons.append(f"attempt_{ordinal}_directory_cannot_be_read")
        if names != _ATTEMPT_FILES:
            reasons.append(f"attempt_{ordinal}_directory_entries_are_not_exact")
    start = _read_evidence_file(
        directory / "start.json", project_root=project_root, parse_json=True
    )
    completion = _read_evidence_file(
        directory / "completion.json", project_root=project_root, parse_json=True
    )
    record = _read_evidence_file(
        directory / "record.json", project_root=project_root, parse_json=True
    )
    stdout = _read_evidence_file(
        directory / "stdout.log", project_root=project_root, parse_json=False
    )
    stderr = _read_evidence_file(
        directory / "stderr.log", project_root=project_root, parse_json=False
    )
    for name, evidence in (
        ("start", start),
        ("completion", completion),
        ("record", record),
        ("stdout", stdout),
        ("stderr", stderr),
    ):
        if evidence.error is None and stat.S_IMODE(evidence.path.stat().st_mode) != 0o444:
            reasons.append(f"attempt_{ordinal}_{name}_mode_is_not_0444")
    return (
        _AttemptEnvelope(
            ordinal=ordinal,
            directory=relative.as_posix(),
            start=start,
            completion=completion,
            record=record,
            stdout=stdout,
            stderr=stderr,
        ),
        reasons,
    )


def _validate_record_envelope_identity(
    record: dict[str, Any],
    *,
    ordinal: Literal[1, 2],
    bindings: _PreregistrationBindings,
) -> None:
    validated = _validate_record(record)
    if validated.ordinal != ordinal:
        raise RepeatVerificationError(f"attempt-{ordinal} raw record ordinal is invalid")
    candidate_config = _mapping(
        _list(record["candidate_configs"], path="candidate_configs")[0],
        path="candidate_configs[0]",
    )
    confirmatory = _required_mapping(record["protocol"], "confirmatory")
    definition = _required_mapping(confirmatory, "definition")
    if record["baseline_config_hash"] != bindings.baseline_config_sha256:
        raise RepeatVerificationError(
            f"attempt-{ordinal} baseline config differs from preregistration"
        )
    if candidate_config["config_hash"] != bindings.candidate_config_sha256:
        raise RepeatVerificationError(
            f"attempt-{ordinal} candidate config differs from preregistration"
        )
    if confirmatory["definition_sha256"] != bindings.protocol_definition_sha256:
        raise RepeatVerificationError(
            f"attempt-{ordinal} protocol definition differs from preregistration"
        )
    dataset = _mapping(record["dataset"], path="dataset")
    if (
        dataset.get("file_sha256") != bindings.dataset["file_sha256"]
        or dataset.get("full_file_story_count") != bindings.dataset["story_count"]
    ):
        raise RepeatVerificationError(f"attempt-{ordinal} dataset differs from preregistration")

    execution_contract = _required_mapping(definition, "execution_contract")
    candidate = _mapping(_list(record["candidates"], path="candidates")[0], path="candidates[0]")
    for record_key, definition_key in (
        ("backend", "candidate_backend"),
        ("method", "candidate_method"),
    ):
        if candidate.get(record_key) != execution_contract.get(definition_key):
            raise RepeatVerificationError(
                f"attempt-{ordinal} candidate {record_key} differs from preregistered protocol"
            )
    if execution_contract.get("quantized_engine") != "qnnpack":
        raise RepeatVerificationError(f"attempt-{ordinal} protocol quantized engine is invalid")

    expected_environment = bindings.environment
    environment = _mapping(record["environment"], path="environment")
    if (
        environment.get("operator_declared_hardware_label")
        != expected_environment["hardware_label"]
    ):
        raise RepeatVerificationError(
            f"attempt-{ordinal} hardware label differs from preregistration"
        )
    raw_python = _required_mapping(environment, "python")
    if {
        "implementation": raw_python.get("implementation"),
        "version": raw_python.get("version"),
    } != expected_environment["python"]:
        raise RepeatVerificationError(
            f"attempt-{ordinal} Python identity differs from preregistration"
        )
    raw_platform = _required_mapping(environment, "platform")
    if {
        name: raw_platform.get(name) for name in ("system", "release", "machine")
    } != expected_environment["platform"]:
        raise RepeatVerificationError(f"attempt-{ordinal} platform differs from preregistration")
    raw_software = _required_mapping(environment, "software")
    if any(
        raw_software.get(name) != version
        for name, version in expected_environment["software"].items()
    ):
        raise RepeatVerificationError(f"attempt-{ordinal} software differs from preregistration")
    raw_torch = _required_mapping(environment, "torch_cpu_execution")
    expected_torch = expected_environment["torch_cpu"]
    if (
        raw_torch.get("intraop_threads") != expected_torch["intraop_threads"]
        or raw_torch.get("interop_threads") != expected_torch["interop_threads"]
        or raw_torch.get("quantized_engine") != "qnnpack"
        or "qnnpack" not in raw_torch.get("supported_quantized_engines", [])
    ):
        raise RepeatVerificationError(
            f"attempt-{ordinal} torch execution differs from preregistration"
        )


def _file_summary(evidence: _JsonEvidence) -> dict[str, object]:
    return {
        "path": evidence.display_path,
        "sha256": evidence.sha256,
        "size_bytes": evidence.size_bytes,
    }


def _compare_raw_inputs(raw_inputs: tuple[_RawInput, _RawInput]) -> dict[str, object]:
    reasons: list[str] = []
    if raw_inputs[0].resolved == raw_inputs[1].resolved:
        reasons.append("raw_input_paths_are_not_distinct")
    if raw_inputs[0].sha256 is not None and raw_inputs[0].sha256 == raw_inputs[1].sha256:
        reasons.append("raw_input_content_hashes_are_not_distinct")

    validated: list[_ValidatedRecord | None] = []
    for index, raw in enumerate(raw_inputs, start=1):
        if raw.error is not None or raw.record is None:
            reasons.append(f"input_{index}_invalid:{raw.error or 'record unavailable'}")
            validated.append(None)
            continue
        try:
            validated.append(_validate_record(raw.record))
        except (KeyError, RepeatVerificationError) as error:
            reasons.append(f"input_{index}_invalid:{error}")
            validated.append(None)

    valid_records = tuple(item for item in validated if item is not None)
    if len(valid_records) == 2:
        ordinals = {item.ordinal for item in valid_records}
        if ordinals != {1, 2}:
            reasons.append("execution_ordinals_must_be_exactly_1_and_2")
        for item in valid_records:
            if not item.within_execution_passed:
                reasons.append(f"execution_{item.ordinal}_noninferiority_failed")
        if (
            valid_records[0].scientific_projection_sha256
            != valid_records[1].scientific_projection_sha256
        ):
            reasons.append("scientific_projections_differ")
        if (
            valid_records[0].environment_projection_sha256
            != valid_records[1].environment_projection_sha256
        ):
            reasons.append("environment_projections_differ")

    ordered_indexes = list(range(2))
    if len(valid_records) == 2 and {item.ordinal for item in valid_records} == {1, 2}:
        ordered_indexes.sort(key=lambda index: valid_records[index].ordinal)
    raw_records: list[dict[str, object]] = []
    scientific_by_execution: dict[str, str] = {}
    environment_by_execution: dict[str, str] = {}
    passed_by_execution: dict[str, bool] = {}
    for index in ordered_indexes:
        raw = raw_inputs[index]
        validated_item = validated[index]
        raw_records.append(
            {
                "input_index": index + 1,
                "execution_ordinal": None if validated_item is None else validated_item.ordinal,
                "path": raw.display_path,
                "sha256": raw.sha256,
                "size_bytes": raw.size_bytes,
            }
        )
        if validated_item is not None:
            key = str(validated_item.ordinal)
            scientific_by_execution[key] = validated_item.scientific_projection_sha256
            environment_by_execution[key] = validated_item.environment_projection_sha256
            passed_by_execution[key] = validated_item.within_execution_passed

    shared_scientific = None
    shared_environment = None
    if len(valid_records) == 2:
        if (
            valid_records[0].scientific_projection_sha256
            == valid_records[1].scientific_projection_sha256
        ):
            shared_scientific = valid_records[0].scientific_projection_sha256
        if (
            valid_records[0].environment_projection_sha256
            == valid_records[1].environment_projection_sha256
        ):
            shared_environment = valid_records[0].environment_projection_sha256

    matched = not reasons
    return {
        "schema_version": _COMPARISON_SCHEMA,
        "status": "matched_non_claiming" if matched else "failed",
        "pair_comparison_passed": matched,
        "confirmatory_pass": False,
        "confirmatory_claim_ready": False,
        "claim_producing_path": False,
        "reasons": reasons,
        "raw_inputs": raw_records,
        "scientific_projection_sha256_by_execution": scientific_by_execution,
        "environment_projection_sha256_by_execution": environment_by_execution,
        "within_execution_noninferiority_passed": passed_by_execution,
        "shared_scientific_projection_sha256": shared_scientific,
        "shared_environment_projection_sha256": shared_environment,
        "claim_boundary_if_provenance_verified": _claim_boundary(),
        "raw_prompt_text_or_output_token_ids_persisted": False,
    }


def verify_repeat_records(
    first: Path,
    second: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Compare arbitrary raw records without producing a confirmatory claim."""

    root = Path.cwd().resolve() if project_root is None else project_root.resolve()
    return _compare_raw_inputs(
        (
            _read_raw_input(first, project_root=root),
            _read_raw_input(second, project_root=root),
        )
    )


def _raw_input_from_evidence(evidence: _JsonEvidence) -> _RawInput:
    return _RawInput(
        requested=evidence.path,
        resolved=evidence.path,
        display_path=evidence.display_path,
        size_bytes=evidence.size_bytes,
        sha256=evidence.sha256,
        record=evidence.value,
        error=evidence.error,
    )


def verify_confirmatory_attempts(
    first_attempt: Path,
    second_attempt: Path,
    preregistration_path: Path,
    *,
    project_root: Path | None = None,
) -> dict[str, object]:
    """Verify only the two fixed sealed attempts against the checked preregistration."""

    root = (Path.cwd() if project_root is None else project_root).expanduser().absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("project root must be a real non-symlink directory")
    reasons: list[str] = []
    attempt_root = (root / _ATTEMPT_ROOT).absolute()
    if _path_has_symlink(attempt_root, project_root=root) or not attempt_root.is_dir():
        reasons.append("fixed_attempt_root_is_invalid")
    else:
        try:
            attempt_root_entries = {entry.name for entry in attempt_root.iterdir()}
        except OSError:
            attempt_root_entries = set()
            reasons.append("fixed_attempt_root_cannot_be_read")
        if attempt_root_entries != {"attempt-1", "attempt-2"}:
            reasons.append("fixed_attempt_root_entries_are_not_exact")
    for requested, expected, kind in (
        (first_attempt, _ATTEMPT_ROOT / "attempt-1", "directory"),
        (second_attempt, _ATTEMPT_ROOT / "attempt-2", "directory"),
        (preregistration_path, _PREREGISTRATION, "file"),
    ):
        _, error = _fixed_requested_path(
            requested,
            project_root=root,
            expected_relative=expected,
            kind=cast(Literal["file", "directory"], kind),
        )
        if error is not None:
            reasons.append(error)

    preregistration = _read_evidence_file(
        (root / _PREREGISTRATION).absolute(),
        project_root=root,
        parse_json=True,
    )
    bindings: _PreregistrationBindings | None = None
    try:
        bindings = _validate_preregistration(preregistration, project_root=root)
    except (KeyError, RepeatVerificationError) as error:
        reasons.append(f"preregistration_invalid:{error}")

    first, first_reasons = _read_attempt_envelope(ordinal=1, project_root=root)
    second, second_reasons = _read_attempt_envelope(ordinal=2, project_root=root)
    reasons.extend(first_reasons)
    reasons.extend(second_reasons)
    starts: list[dict[str, Any] | None] = []
    completions: list[dict[str, Any] | None] = []
    for envelope in (first, second):
        ordinal = envelope.ordinal
        start_value: dict[str, Any] | None = None
        completion_value: dict[str, Any] | None = None
        if bindings is None:
            reasons.append(f"attempt_{ordinal}_cannot_bind_without_valid_preregistration")
        else:
            try:
                start_value = _validate_start_record(
                    envelope.start,
                    ordinal=ordinal,
                    project_root=root,
                    preregistration=preregistration,
                    bindings=bindings,
                )
            except (KeyError, RepeatVerificationError) as error:
                reasons.append(f"attempt_{ordinal}_start_invalid:{error}")
            try:
                completion_value = _validate_completion_record(
                    envelope.completion,
                    ordinal=ordinal,
                    start=envelope.start,
                    record=envelope.record,
                    stdout=envelope.stdout,
                    stderr=envelope.stderr,
                )
            except (KeyError, RepeatVerificationError) as error:
                reasons.append(f"attempt_{ordinal}_completion_invalid:{error}")
            if envelope.record.value is not None and envelope.record.error is None:
                try:
                    _validate_record_envelope_identity(
                        envelope.record.value,
                        ordinal=ordinal,
                        bindings=bindings,
                    )
                except (KeyError, RepeatVerificationError, TypeError) as error:
                    reasons.append(f"attempt_{ordinal}_record_identity_invalid:{error}")
            else:
                reasons.append(f"attempt_{ordinal}_record_is_invalid")
        starts.append(start_value)
        completions.append(completion_value)

    if bindings is not None and preregistration.value is not None:
        preregistered_at = datetime.fromisoformat(str(preregistration.value["created_at_utc"]))
        for index, (start_value, completion_value, envelope) in enumerate(
            zip(starts, completions, (first, second), strict=True), start=1
        ):
            if start_value is not None:
                started_at = datetime.fromisoformat(str(start_value["created_at_utc"]))
                if started_at < preregistered_at:
                    reasons.append(f"attempt_{index}_started_before_preregistration")
                if completion_value is not None:
                    completed_at = datetime.fromisoformat(str(completion_value["created_at_utc"]))
                    if completed_at < started_at:
                        reasons.append(f"attempt_{index}_completed_before_it_started")
                    if envelope.record.value is not None:
                        raw_created = envelope.record.value.get("created_at_utc")
                        if isinstance(raw_created, str):
                            try:
                                recorded_at = datetime.fromisoformat(raw_created)
                            except ValueError:
                                pass
                            else:
                                if recorded_at.tzinfo is not None and (
                                    recorded_at < started_at or recorded_at > completed_at
                                ):
                                    reasons.append(
                                        f"attempt_{index}_record_timestamp_outside_envelope"
                                    )
        if starts[0] is not None and starts[1] is not None:
            first_command = _list(starts[0]["command_argv"], path="attempt-1 command")
            second_command = _list(starts[1]["command_argv"], path="attempt-2 command")
            if first_command[0] != second_command[0]:
                reasons.append("attempt_interpreter_paths_differ")
            if starts[0]["dataset"] != starts[1]["dataset"]:
                reasons.append("attempt_dataset_envelopes_differ")
        if completions[0] is not None and starts[1] is not None:
            first_completed_at = datetime.fromisoformat(str(completions[0]["created_at_utc"]))
            second_started_at = datetime.fromisoformat(str(starts[1]["created_at_utc"]))
            if first_completed_at > second_started_at:
                reasons.append("attempt_2_started_before_attempt_1_completed")

    comparison = _compare_raw_inputs(
        (
            _raw_input_from_evidence(first.record),
            _raw_input_from_evidence(second.record),
        )
    )
    comparison_reasons = cast(list[str], comparison["reasons"])
    reasons.extend(f"record_pair:{reason}" for reason in comparison_reasons)
    confirmed = not reasons and comparison["pair_comparison_passed"] is True
    attempt_summaries = []
    for envelope in (first, second):
        attempt_summaries.append(
            {
                "ordinal": envelope.ordinal,
                "directory": envelope.directory,
                "start": _file_summary(envelope.start),
                "completion": _file_summary(envelope.completion),
                "record": _file_summary(envelope.record),
                "stdout": _file_summary(envelope.stdout),
                "stderr": _file_summary(envelope.stderr),
            }
        )
    binding_summary: dict[str, object] = {
        "protocol_definition_sha256": None,
        "baseline_config_sha256": None,
        "candidate_config_sha256": None,
        "dataset_identity_sha256": None,
        "environment_contract_sha256": None,
        "bound_file_manifest_sha256": None,
        "bound_file_count": None,
    }
    if bindings is not None:
        binding_summary = {
            "protocol_definition_sha256": bindings.protocol_definition_sha256,
            "baseline_config_sha256": bindings.baseline_config_sha256,
            "candidate_config_sha256": bindings.candidate_config_sha256,
            "dataset_identity_sha256": _canonical_sha256(bindings.dataset),
            "environment_contract_sha256": _canonical_sha256(bindings.environment),
            "bound_file_manifest_sha256": _canonical_sha256(bindings.file_sha256),
            "bound_file_count": len(bindings.file_sha256),
        }
    return {
        "schema_version": _AGGREGATE_SCHEMA,
        "status": "confirmed" if confirmed else "failed",
        "confirmatory_pass": confirmed,
        "confirmatory_claim_ready": confirmed,
        "claim_producing_path": True,
        "verification_scope": "fixed_sealed_attempt_envelopes_and_checked_preregistration",
        "aggregate_path": _AGGREGATE_PATH.as_posix(),
        "reasons": reasons,
        "preregistration": _file_summary(preregistration),
        "preregistration_sha256": preregistration.sha256,
        "attempts": attempt_summaries,
        "bindings": binding_summary,
        "scientific_projection_sha256_by_execution": comparison[
            "scientific_projection_sha256_by_execution"
        ],
        "environment_projection_sha256_by_execution": comparison[
            "environment_projection_sha256_by_execution"
        ],
        "within_execution_noninferiority_passed": comparison[
            "within_execution_noninferiority_passed"
        ],
        "shared_scientific_projection_sha256": comparison["shared_scientific_projection_sha256"],
        "shared_environment_projection_sha256": comparison["shared_environment_projection_sha256"],
        "claim": _claim_boundary(),
        "raw_prompt_text_or_output_token_ids_persisted": False,
    }


def resolve_output_path(project_root: Path, requested: Path) -> Path:
    """Require a new project-relative JSON path beneath ``artifacts/``."""

    root = project_root.expanduser().resolve()
    if requested.is_absolute():
        raise ValueError("output must be project-relative")
    if requested.suffix != ".json":
        raise ValueError("output must use the .json suffix")
    if ".." in requested.parts:
        raise ValueError("output path cannot contain '..'")
    artifact_requested = root / "artifacts"
    if artifact_requested.is_symlink():
        raise ValueError("project artifacts directory must not be a symlink")
    artifact_root = artifact_requested.resolve(strict=False)
    candidate = (root / requested).resolve(strict=False)
    if candidate == artifact_root or not candidate.is_relative_to(artifact_root):
        raise ValueError("output must remain below the project artifacts directory")
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(f"immutable output already exists: {candidate.name}")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    current = candidate.parent
    while current != artifact_root:
        if current.is_symlink():
            raise ValueError("output parent directories must not be symlinks")
        current = current.parent
    if artifact_root.is_symlink():
        raise ValueError("project artifacts directory must not be a symlink")
    if candidate.resolve(strict=False) != candidate:
        raise ValueError("output path changed after parent creation")
    return candidate


def resolve_claim_output_path(project_root: Path, requested: Path) -> Path:
    """Resolve only the preregistered immutable repeat-verification path."""

    if requested != _AGGREGATE_PATH:
        raise ValueError(f"claim output must be exactly the preregistered path: {_AGGREGATE_PATH}")
    return resolve_output_path(project_root, requested)


def atomic_write_json(output: Path, value: Mapping[str, object]) -> None:
    """Publish one finite aggregate atomically without replacing prior evidence."""

    payload = (_canonical_json(value) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def seal_claim_artifact_root(project_root: Path, output: Path) -> None:
    """Seal the fixed aggregate and its attempt root after exclusive publication."""

    expected_output = (project_root / _AGGREGATE_PATH).absolute()
    attempt_root = (project_root / _ATTEMPT_ROOT).absolute()
    if output != expected_output or output.is_symlink() or not output.is_file():
        raise ValueError("published claim aggregate path is invalid")
    if stat.S_IMODE(output.stat().st_mode) != 0o444:
        raise RuntimeError("published claim aggregate mode is not 0444")
    expected_entries = {"attempt-1", "attempt-2", _AGGREGATE_PATH.name}
    if {entry.name for entry in attempt_root.iterdir()} != expected_entries:
        raise RuntimeError("attempt root entries changed during aggregate publication")
    attempt_root.chmod(0o555)
    if stat.S_IMODE(attempt_root.stat().st_mode) != 0o555:
        raise RuntimeError("published attempt root mode is not 0555")


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("first_attempt", type=Path)
    parser.add_argument("second_attempt", type=Path)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Verify two records, retain the aggregate, and signal failure with the exit status."""

    arguments = _arguments(argv)
    project_root = Path.cwd().absolute()
    output = resolve_claim_output_path(project_root, arguments.output)
    aggregate = verify_confirmatory_attempts(
        arguments.first_attempt,
        arguments.second_attempt,
        arguments.preregistration,
        project_root=project_root,
    )
    atomic_write_json(output, aggregate)
    seal_claim_artifact_root(project_root, output)
    print(output)
    return 0 if aggregate["status"] == "confirmed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
