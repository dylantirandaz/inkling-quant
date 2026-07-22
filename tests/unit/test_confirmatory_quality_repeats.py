"""Fail-closed contracts for the two-record confirmatory quality verifier."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import suppress
from pathlib import Path
from typing import Any

import pytest

from scripts import verify_confirmatory_quality_repeats as verifier

pytestmark = pytest.mark.unit

_BASELINE_HASH = "e6db2959221babf9aaba1f529a20349fe3462d4309b6929caf3a7d3331d668f2"
_CANDIDATE_HASH = "d6ee2054d6db801c56596042df7e9eca130109ab922ee46735b61953da8d7912"
_MODEL_STATE_HASH = "3" * 64
_CANDIDATE_STATE_HASH = "4" * 64
_DATASET_HASH = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
_SOURCE_WEIGHT_HASH = "6" * 64
_CONTENT_HASH = "7" * 64
_OUTPUT_HASH = "8" * 64
_TOKEN_MANIFESTS = {
    "quality_input_token_ids_manifest_sha256": "9" * 64,
    "routing_input_token_ids_manifest_sha256": "a" * 64,
    "generation_prompt_token_ids_manifest_sha256": "b" * 64,
}


@pytest.fixture(autouse=True)
def _restore_synthetic_permissions(tmp_path: Path) -> Iterator[None]:
    yield
    for path in sorted(tmp_path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            continue
        with suppress(FileNotFoundError):
            path.chmod(0o755 if path.is_dir() else 0o644)


def _definition() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "protocol_id": "stories15m-native-int8-confirmatory-quality-v1",
        "model": {
            "model_id": "ggml-org/stories15M_MOE",
            "revision": "b6dd737497465570b5f5e962dbc9d9454ed1e0eb",
            "source_weight_sha256": _SOURCE_WEIGHT_HASH,
            "source_weight_size_bytes": 72_744_704,
        },
        "dataset": {
            "dataset_id": "hf://datasets/roneneldan/TinyStories/TinyStories-valid.txt",
            "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
            "split": "validation",
            "file_sha256": _DATASET_HASH,
        },
        "execution_contract": {
            "baseline_resolved_config_sha256": _BASELINE_HASH,
            "candidate_resolved_config_sha256": _CANDIDATE_HASH,
            "candidate_backend": "torch_native_dynamic_int8",
            "candidate_method": "native_dynamic_w8a8",
            "quantized_engine": "qnnpack",
        },
        "holdout_selection": {**_TOKEN_MANIFESTS, "quality_sample_count": 256},
        "generation_and_routing": {"sample_count": 16},
        "noninferiority": {
            "confidence_level": 0.95,
            "margin_nats_per_token": 0.004987541511039074,
            "bootstrap": {
                "method": "paired_story_cluster_within_length_strata_srswor_moment_matching",
                "replicates": 100_000,
            },
        },
        "repeatability": {"required_clean_process_executions": 2},
        "claims": {"routing_and_generation_inferential": False},
    }


def _noninferiority(*, passed: bool) -> dict[str, Any]:
    return {
        "schema_version": "paired-stratified-ht-noninferiority-v1",
        "passed": passed,
        "confidence": 0.95,
        "margin_nll": 0.004987541511039074,
        "candidate_minus_baseline_mean_nll_ht": 0.001,
        "upper_confidence_bound_nll": 0.002,
        "bootstrap_distribution_sha256": "c" * 64,
        "method": {
            "bootstrap": "within_stratum_paired_empirical_resampling_with_srswor_moment_matching"
        },
    }


def _record(
    ordinal: int,
    *,
    passed: bool = True,
    timing_offset: float = 0.0,
    executable: str | None = None,
) -> dict[str, Any]:
    definition = _definition()
    noninferiority = _noninferiority(passed=passed)
    sample = {
        "sample_id": "story-000147",
        "content_sha256": _CONTENT_HASH,
        "evaluated_token_count": 17,
        "length_stratum": 0,
        "mean_nll": 1.25,
    }
    observer = {
        "schema_version": "routing-observer-proof-v1",
        "quality_forward_count": 256,
        "routing_reforward_count": 16,
        "model_state_unchanged": True,
        "reforward_observations": [sample],
    }
    return {
        "schema_version": "public-moe-native-linear-quality-v2",
        "created_at_utc": f"2026-07-16T00:00:0{ordinal}+00:00",
        "baseline_config": "confirmatory-baseline",
        "baseline_config_hash": _BASELINE_HASH,
        "candidate_configs": [{"name": "confirmatory-int8", "config_hash": _CANDIDATE_HASH}],
        "model": {
            "model_id": definition["model"]["model_id"],
            "revision": definition["model"]["revision"],
            "resolved_class": "transformers.MixtralForCausalLM",
            "architecture": "MixtralForCausalLM",
            "loaded_float32_state_sha256": _MODEL_STATE_HASH,
            "source_weight_file": "model.safetensors",
            "source_weight_sha256": _SOURCE_WEIGHT_HASH,
            "source_weight_size_bytes": 72_744_704,
            "license": "MIT",
            "expert_weight_provenance": "publisher",
            "router_provenance": "publisher_randomized",
        },
        "dataset": {
            "dataset_id": definition["dataset"]["dataset_id"],
            "revision": definition["dataset"]["revision"],
            "split": definition["dataset"]["split"],
            "file": "TinyStories-valid.txt",
            "file_sha256": _DATASET_HASH,
            "full_file_story_count": 21_990,
            "license": "cdla-sharing-1.0",
            "perplexity_sample_ids": ["story-000147"],
            "perplexity_selection_sha256": "d" * 64,
            "generation_sample_ids": ["story-000147"],
            "generation_selection_sha256": "e" * 64,
        },
        "protocol": {
            "quality_metric": "token_weighted_causal_nll_and_perplexity",
            "loss_special_tokens": True,
            "loss_truncation": "right_to_model_max_context",
            "max_context_tokens": 256,
            "generation_decode": {
                "max_new_tokens": 8,
                "do_sample": False,
                "temperature": 1.0,
                "top_k": None,
            },
            "generation_input_truncation": "right_to_max_context_minus_max_new_tokens",
            "generation_reference": "same_loaded_float32_baseline",
            "routing_capture": (
                "all_256_quality_loss_forwards_hook_free_then_full_trace_on_exact_16_"
                "sample_reforward"
            ),
            "routing_comparison": "stable_sample_id_token_position_and_layer",
            "seed": 20260715,
            "prompt_or_output_content_persisted": False,
            "quantization_scope": "concrete_unprotected_nn.Linear_leaves_only",
            "generation_and_routing_inference_claim": "descriptive_only",
            "confirmatory": {
                "definition": definition,
                "definition_sha256": verifier._canonical_sha256(definition),
                "selection": {
                    **_TOKEN_MANIFESTS,
                    "quality_sample_ids": ["story-000147"],
                    "source_text_or_token_ids_persisted": False,
                },
                "noninferiority_result": copy.deepcopy(noninferiority),
                "noninferiority_result_scope": (
                    "within_execution_statistical_gate_only_not_overall_confirmatory_claim"
                ),
                "upper_bound_label": "nominal_one_sided_95_percent_bootstrap_upper_bound",
                "coverage_interpretation": (
                    "nominal_approximate_coverage_under_the_predeclared_stratified_"
                    "finite_population_bootstrap_design"
                ),
            },
        },
        "inventory": {
            "entry_count": 76,
            "addressable_linear_count": 24,
            "addressable_linear_names": ["model.layers.0.self_attn.q_proj"],
        },
        "baseline": {
            "runtime_tensor_storage_bytes": 145_000_000,
            "load_time_seconds": 1.0 + timing_offset,
            "measurement_elapsed_seconds": 2.0 + timing_offset,
            "quality": {
                "sample_count": 256,
                "evaluated_token_count": 52_038,
                "mean_nll": 1.25,
                "perplexity": 3.4903429574618414,
                "samples": [copy.deepcopy(sample)],
            },
            "generation": {
                "sample_count": 16,
                "decode": {"max_new_tokens": 8, "do_sample": False},
                "samples": [
                    {
                        "sample_id": "story-000147",
                        "content_sha256": _CONTENT_HASH,
                        "output_sha256": _OUTPUT_HASH,
                    }
                ],
                "output_content_persisted": False,
            },
            "routing": {
                "observed_event_count": 19_764,
                "recorded_event_count": 19_764,
                "batch_count": 16,
                "alignment_sha256": "f" * 64,
            },
            "routing_observer_proof": copy.deepcopy(observer),
        },
        "candidates": [
            {
                "config_name": "confirmatory-int8",
                "config_hash": _CANDIDATE_HASH,
                "backend": "torch_native_dynamic_int8",
                "method": "native_dynamic_w8a8",
                "support": {"available": True, "supported": True},
                "quantization": {
                    "scope": "attention_and_other_unprotected_concrete_nn.Linear_leaves_only",
                    "target_precision": "int8",
                    "quantized_module_count": 24,
                    "quantized_modules": ["model.layers.0.self_attn.q_proj"],
                    "float32_module_count": 52,
                    "fused_expert_slices_quantized": 0,
                    "fused_expert_slices_float32": 24,
                    "quantization_elapsed_seconds": 3.0 + timing_offset,
                    "runtime_tensor_storage_bytes": 139_000_000,
                    "runtime_tensor_storage_ratio_vs_float32": 0.9586206896551724,
                    "safe_bundle_serialized_size_bytes": 139_000_000,
                    "candidate_state_sha256": _CANDIDATE_STATE_HASH,
                    "manifest": {
                        "backend": "torch_native_dynamic_int8",
                        "parameters": {"quantized_engine": "qnnpack"},
                    },
                },
                "quality": {
                    "sample_count": 256,
                    "evaluated_token_count": 52_038,
                    "mean_nll": 1.251,
                    "perplexity": 3.493835046200138,
                    "samples": [{**sample, "mean_nll": 1.251}],
                    "mean_nll_delta_vs_float32": 0.001,
                    "perplexity_delta_vs_float32": 0.0034920887382965,
                    "perplexity_relative_change": 0.001,
                },
                "generation_retention": {
                    "sample_count": 16,
                    "exact_match_count": 15,
                    "exact_match_rate": 0.9375,
                    "samples": [
                        {
                            "sample_id": "story-000147",
                            "exact_match": True,
                            "baseline_output_sha256": _OUTPUT_HASH,
                            "candidate_output_sha256": _OUTPUT_HASH,
                        }
                    ],
                    "output_content_persisted": False,
                },
                "routing": {
                    "alignment_sha256": "f" * 64,
                    "observed_event_count": 19_764,
                    "recorded_event_count": 19_764,
                    "comparison": {"top1_agreement": 0.99},
                },
                "routing_observer_proof": copy.deepcopy(observer),
                "noninferiority": noninferiority,
                "measurement_elapsed_seconds": 4.0 + timing_offset,
            }
        ],
        "environment": {
            "python": {
                "version": "3.12.13",
                "implementation": "CPython",
                "executable": executable or f"/venv-{ordinal}/bin/python",
            },
            "platform": {"system": "Darwin", "release": "24.3", "machine": "arm64"},
            "hardware": {"logical_cpu_count": 8, "device": "cpu"},
            "packages": {"torch": "2.13.0", "transformers": "5.12.1"},
            "software": {"numpy": "2.2.6", "torch": "2.13.0"},
            "runtime_capability": {"available": True, "devices": ["cpu"]},
            "operator_declared_hardware_label": "Apple M3 MacBook Air (8 cores, 16 GB)",
            "operator_hardware_label_scope": "explicit CLI metadata",
            "torch_cpu_execution": {
                "intraop_threads": 8,
                "interop_threads": 8,
                "quantized_engine": "qnnpack",
                "supported_quantized_engines": ["qnnpack", "none"],
                "kleidiai_available": True,
            },
            "energy": {"available": False},
            "distributed": {"gloo": {"available": True}},
            "git": {"commit": None, "dirty": None},
        },
        "elapsed_seconds_including_model_load_and_all_candidates": 5.0 + timing_offset,
        "limitations": ["finite pinned holdout only"],
        "confirmatory_status": {
            "status": "provisional_until_2_clean_executions",
            "confirmatory_claim_ready": False,
            "required_clean_process_executions": 2,
            "execution_ordinal": ordinal,
            "within_execution_noninferiority_passed": passed,
            "within_execution_decision_is_overall_confirmatory_claim": False,
            "pair_verification_required": True,
        },
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def _pair(tmp_path: Path, *, passed: bool = True) -> tuple[Path, Path]:
    first = tmp_path / "execution-1.json"
    second = tmp_path / "execution-2.json"
    _write(first, _record(1, passed=passed))
    _write(second, _record(2, passed=passed, timing_offset=100.0))
    return first, second


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fact(path: Path) -> dict[str, Any]:
    return {"path": path.name, "sha256": _sha256_file(path), "size_bytes": path.stat().st_size}


def _environment_contract(project_root: Path) -> dict[str, Any]:
    invocation = (project_root / ".venv/bin/python").absolute()
    resolved = invocation.resolve(strict=True)
    pyvenv = project_root / ".venv/pyvenv.cfg"
    return {
        "hardware_label": "Apple M3 MacBook Air (8 cores, 16 GB)",
        "python": {"implementation": "CPython", "version": "3.12.13"},
        "platform": {"system": "Darwin", "release": "24.3", "machine": "arm64"},
        "software": {"numpy": "2.2.6", "torch": "2.13.0"},
        "torch_cpu": {
            "intraop_threads": 8,
            "interop_threads": 8,
            "required_supported_engine": "qnnpack",
        },
        "offline_environment": copy.deepcopy(verifier._OFFLINE_ENVIRONMENT),
        "python_executable": {
            "invocation_path": str(invocation),
            "resolved_path": str(resolved),
            "sha256": _sha256_file(resolved),
            "size_bytes": resolved.stat().st_size,
        },
        "virtual_environment": {
            "pyvenv_cfg_path": ".venv/pyvenv.cfg",
            "sha256": _sha256_file(pyvenv),
            "size_bytes": pyvenv.stat().st_size,
        },
    }


def _materialize_bound_files(project_root: Path) -> dict[str, str]:
    relative_paths = (
        *verifier._REQUIRED_BOUND_FILES,
        Path("src/inkling_quant_lab/__init__.py"),
        Path("src/inkling_quant_lab/config.py"),
    )
    result: dict[str, str] = {}
    for relative in relative_paths:
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic bound source: {relative.as_posix()}\n", encoding="utf-8")
        result[relative.as_posix()] = _sha256_file(path)
    target = project_root / "synthetic-python-target"
    target.write_bytes(b"synthetic resolved Python executable\n")
    target.chmod(0o755)
    bin_directory = project_root / ".venv/bin"
    bin_directory.mkdir(parents=True)
    (bin_directory / "python").symlink_to(Path("../../synthetic-python-target"))
    (project_root / ".venv/pyvenv.cfg").write_text(
        "home = /synthetic\ninclude-system-site-packages = false\n",
        encoding="utf-8",
    )
    return result


def _source_import_contract(file_sha256: dict[str, str]) -> dict[str, Any]:
    contract = copy.deepcopy(verifier._SOURCE_IMPORT_CONTRACT_STATIC)
    contract["verified_local_modules"] = {
        "inkling_quant_lab": {
            "source_path": "src/inkling_quant_lab/__init__.py",
            "source_sha256": file_sha256["src/inkling_quant_lab/__init__.py"],
            "cached_path": "/dev/null/synthetic/inkling_quant_lab/__init__.pyc",
            "cached_exists": False,
        },
        "scripts": {
            "source_path": "scripts/__init__.py",
            "source_sha256": file_sha256["scripts/__init__.py"],
            "cached_path": "/dev/null/synthetic/scripts/__init__.pyc",
            "cached_exists": False,
        },
    }
    return contract


def _preregistration(project_root: Path, file_sha256: dict[str, str]) -> Path:
    path = project_root / verifier._PREREGISTRATION
    path.parent.mkdir(parents=True, exist_ok=True)
    attempt_records = [
        (verifier._ATTEMPT_ROOT / f"attempt-{ordinal}" / "record.json").as_posix()
        for ordinal in (1, 2)
    ]
    value = {
        "schema_version": "confirmatory-quality-preregistration-v1",
        "status": "locked_before_execution",
        "protocol_id": "stories15m-native-int8-confirmatory-quality-v1",
        "created_at_utc": "2026-07-15T23:59:00+00:00",
        "outcomes": {
            "holdout_model_forward_executed": False,
            "holdout_outcomes_inspected": False,
        },
        "bindings": {
            "files": file_sha256,
            "resolved_config_sha256": {
                "baseline": _BASELINE_HASH,
                "candidate": _CANDIDATE_HASH,
            },
            "protocol_definition_sha256": verifier._canonical_sha256(_definition()),
            "dataset": {
                "file_sha256": _DATASET_HASH,
                "size_bytes": 19_447_282,
                "story_count": 21_990,
            },
        },
        "environment_contract": _environment_contract(project_root),
        "planned_attempts": [
            {
                "ordinal": ordinal,
                "directory": (verifier._ATTEMPT_ROOT / f"attempt-{ordinal}").as_posix(),
                "record_path": attempt_records[ordinal - 1],
            }
            for ordinal in (1, 2)
        ],
        "primary_analysis": {
            "estimand": "synthetic_fixture_only",
            "repeat_verification": {
                "required_clean_process_executions": 2,
                "input_records": attempt_records,
                "aggregate_path": verifier._AGGREGATE_PATH.as_posix(),
                "single_execution_status": "provisional_until_2_clean_executions",
                "promotion_rule": (
                    "both_within_execution_gates_pass_and_scientific_environment_"
                    "projections_match_exactly"
                ),
                "repeat_is_additional_statistical_sample": False,
            },
        },
        "claim_boundary": ["synthetic fixture; no real outcome"],
    }
    _write(path, value)
    return path


def _completion_record_fact(record_path: Path, ordinal: int) -> dict[str, Any]:
    return {
        "exists": True,
        "is_regular_non_symlink": True,
        "size_bytes": record_path.stat().st_size,
        "sha256": _sha256_file(record_path),
        "schema_version": "public-moe-native-linear-quality-v2",
        "execution_ordinal": ordinal,
        "provisional_status": "provisional_until_2_clean_executions",
        "confirmatory_claim_ready": False,
        "valid": True,
        "validation_error": None,
    }


def _attempt_directory(
    project_root: Path,
    preregistration: Path,
    file_sha256: dict[str, str],
    ordinal: int,
    *,
    passed: bool = True,
) -> Path:
    directory = project_root / verifier._ATTEMPT_ROOT / f"attempt-{ordinal}"
    directory.mkdir(parents=True)
    record_path = directory / "record.json"
    start_time = "2026-07-16T00:00:00+00:00" if ordinal == 1 else "2026-07-16T00:00:06+00:00"
    record_time = "2026-07-16T00:00:01+00:00" if ordinal == 1 else "2026-07-16T00:00:07+00:00"
    completion_time = "2026-07-16T00:00:05+00:00" if ordinal == 1 else "2026-07-16T00:00:11+00:00"
    record = _record(ordinal, passed=passed, timing_offset=float(ordinal * 100))
    record["created_at_utc"] = record_time
    _write(record_path, record)
    stdout_path = directory / "stdout.log"
    stderr_path = directory / "stderr.log"
    stdout_path.write_bytes(b"synthetic stdout\n")
    stderr_path.write_bytes(b"")
    relative_directory = verifier._ATTEMPT_ROOT / f"attempt-{ordinal}"
    relative_record = relative_directory / "record.json"
    dataset_path = str((project_root / "inputs/TinyStories-valid.txt").absolute())
    start = {
        "schema_version": "confirmatory-quality-attempt-start-v1",
        "status": "started_before_preflight_and_model_execution",
        "created_at_utc": start_time,
        "ordinal": ordinal,
        "attempt_directory": relative_directory.as_posix(),
        "planned_record_path": relative_record.as_posix(),
        "command_argv": [
            str((project_root / ".venv/bin/python").absolute()),
            "-s",
            "-P",
            "-B",
            verifier._EVALUATOR.as_posix(),
            verifier._BASELINE_CONFIG.as_posix(),
            verifier._CANDIDATE_CONFIG.as_posix(),
            "--dataset",
            dataset_path,
            "--expected-dataset-sha256",
            _DATASET_HASH,
            "--hardware-label",
            "Apple M3 MacBook Air (8 cores, 16 GB)",
            "--confirmatory-protocol",
            verifier._PROTOCOL_CONFIG.as_posix(),
            "--execution-ordinal",
            str(ordinal),
            "--output",
            relative_record.as_posix(),
        ],
        "shell": False,
        "working_directory": str(project_root),
        "preregistration": {
            "path": verifier._PREREGISTRATION.as_posix(),
            "sha256": _sha256_file(preregistration),
            "size_bytes": preregistration.stat().st_size,
        },
        "dataset": {
            "path": dataset_path,
            "sha256": _DATASET_HASH,
            "size_bytes": 19_447_282,
            "story_count": 21_990,
        },
        "file_sha256": file_sha256,
        "resolved_config_sha256": {
            "baseline": _BASELINE_HASH,
            "candidate": _CANDIDATE_HASH,
        },
        "protocol_definition_sha256": verifier._canonical_sha256(_definition()),
        "environment_contract": _environment_contract(project_root),
        "selected_child_environment": copy.deepcopy(verifier._OFFLINE_ENVIRONMENT),
        "source_import_contract": _source_import_contract(file_sha256),
        "stdout_path": "stdout.log",
        "stderr_path": "stderr.log",
        "outcome_fields_present": False,
        "preflight_materialization_failure": None,
    }
    start_path = directory / "start.json"
    _write(start_path, start)
    completion = {
        "schema_version": "confirmatory-quality-attempt-completion-v1",
        "status": "attempt_execution_complete",
        "created_at_utc": completion_time,
        "ordinal": ordinal,
        "attempt_directory": relative_directory.as_posix(),
        "subprocess_launched": True,
        "preflight_complete": True,
        "subprocess_returncode": 0,
        "runner_exit_code": 0,
        "elapsed_seconds": 10.0,
        "failure": None,
        "stdout": _fact(stdout_path),
        "stderr": _fact(stderr_path),
        "start": _fact(start_path),
        "record": _completion_record_fact(record_path, ordinal),
        "confirmatory_claim_ready": False,
        "claim_boundary": verifier._COMPLETION_CLAIM_BOUNDARY,
    }
    _write(directory / "completion.json", completion)
    for path in directory.iterdir():
        path.chmod(0o444)
    directory.chmod(0o555)
    return directory


def _sealed_pair(project_root: Path, *, passed: bool = True) -> tuple[Path, Path, Path]:
    file_sha256 = _materialize_bound_files(project_root)
    preregistration = _preregistration(project_root, file_sha256)
    first = _attempt_directory(project_root, preregistration, file_sha256, 1, passed=passed)
    second = _attempt_directory(project_root, preregistration, file_sha256, 2, passed=passed)
    return first, second, preregistration


def _refresh_completion_record_fact(directory: Path, ordinal: int) -> None:
    completion_path = directory / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["record"] = _completion_record_fact(directory / "record.json", ordinal)
    _rewrite_sealed_json(completion_path, completion)


def _rewrite_sealed_json(path: Path, value: object) -> None:
    path.chmod(0o644)
    _write(path, value)
    path.chmod(0o444)


def _rewrite_sealed_bytes(path: Path, value: bytes) -> None:
    path.chmod(0o644)
    path.write_bytes(value)
    path.chmod(0o444)


def test_matching_raw_pair_is_explicitly_non_claiming_and_content_redacted(tmp_path: Path) -> None:
    first, second = _pair(tmp_path)

    aggregate = verifier.verify_repeat_records(first, second, project_root=tmp_path)

    assert aggregate["status"] == "matched_non_claiming"
    assert aggregate["pair_comparison_passed"] is True
    assert aggregate["confirmatory_pass"] is False
    assert aggregate["confirmatory_claim_ready"] is False
    assert aggregate["claim_producing_path"] is False
    assert aggregate["reasons"] == []
    assert aggregate["shared_scientific_projection_sha256"]
    assert aggregate["shared_environment_projection_sha256"]
    assert [item["execution_ordinal"] for item in aggregate["raw_inputs"]] == [1, 2]
    assert aggregate["raw_inputs"][0]["sha256"] != aggregate["raw_inputs"][1]["sha256"]
    assert aggregate["claim_boundary_if_provenance_verified"]["coverage_qualification"] == (
        "nominal_bootstrap_bound_not_exact_finite_population_coverage"
    )
    serialized = json.dumps(aggregate, sort_keys=True)
    assert "story-000147" not in serialized
    assert "mean_nll" not in serialized
    assert _OUTPUT_HASH not in serialized


def test_timing_and_python_executable_are_the_only_ignored_runtime_fields(
    tmp_path: Path,
) -> None:
    first_record = _record(1, timing_offset=0.0, executable="/one/python")
    second_record = _record(2, timing_offset=999.0, executable="/two/python")

    assert verifier.scientific_projection(first_record) == verifier.scientific_projection(
        second_record
    )
    assert verifier.environment_projection(first_record) == verifier.environment_projection(
        second_record
    )

    del second_record["environment"]["python"]["executable"]
    assert verifier.environment_projection(first_record) == verifier.environment_projection(
        second_record
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda record: record["protocol"]["confirmatory"]["selection"].__setitem__(
            "quality_input_token_ids_manifest_sha256", "0" * 64
        ),
        lambda record: record["inventory"].__setitem__("entry_count", 77),
        lambda record: record["baseline"]["quality"]["samples"][0].__setitem__("mean_nll", 1.26),
        lambda record: record["candidates"][0]["quantization"].__setitem__(
            "candidate_state_sha256", "0" * 64
        ),
        lambda record: record["candidates"][0]["routing_observer_proof"].__setitem__(
            "model_state_unchanged", False
        ),
    ),
)
def test_any_scientific_projection_mismatch_fails(
    tmp_path: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    first, second = _pair(tmp_path)
    changed = _record(2, timing_offset=100.0)
    mutate(changed)
    _write(second, changed)

    aggregate = verifier.verify_repeat_records(first, second, project_root=tmp_path)

    assert aggregate["status"] == "failed"
    assert aggregate["confirmatory_pass"] is False
    assert "scientific_projections_differ" in aggregate["reasons"]
    assert aggregate["shared_scientific_projection_sha256"] is None


def test_environment_mismatch_fails(tmp_path: Path) -> None:
    first, second = _pair(tmp_path)
    changed = _record(2, timing_offset=100.0)
    changed["environment"]["hardware"]["logical_cpu_count"] = 10
    _write(second, changed)

    aggregate = verifier.verify_repeat_records(first, second, project_root=tmp_path)

    assert "environment_projections_differ" in aggregate["reasons"]
    assert aggregate["shared_environment_projection_sha256"] is None


def test_two_matching_failed_gates_are_retained_as_failed_aggregate(tmp_path: Path) -> None:
    first, second = _pair(tmp_path, passed=False)

    aggregate = verifier.verify_repeat_records(first, second, project_root=tmp_path)

    assert aggregate["status"] == "failed"
    assert aggregate["confirmatory_pass"] is False
    assert aggregate["shared_scientific_projection_sha256"]
    assert "execution_1_noninferiority_failed" in aggregate["reasons"]
    assert "execution_2_noninferiority_failed" in aggregate["reasons"]


def test_ordinals_paths_and_content_must_each_be_distinct(tmp_path: Path) -> None:
    first, second = _pair(tmp_path)
    duplicate_ordinal = _record(1, timing_offset=100.0)
    _write(second, duplicate_ordinal)
    ordinal_result = verifier.verify_repeat_records(first, second, project_root=tmp_path)
    assert "execution_ordinals_must_be_exactly_1_and_2" in ordinal_result["reasons"]

    same_path = verifier.verify_repeat_records(first, first, project_root=tmp_path)
    assert "raw_input_paths_are_not_distinct" in same_path["reasons"]
    assert "raw_input_content_hashes_are_not_distinct" in same_path["reasons"]

    _write(second, _record(1))
    same_content = verifier.verify_repeat_records(first, second, project_root=tmp_path)
    assert "raw_input_content_hashes_are_not_distinct" in same_content["reasons"]


@pytest.mark.parametrize(
    ("change", "reason_fragment"),
    (
        (
            lambda record: record.pop("model"),
            "missing or unknown top-level fields",
        ),
        (
            lambda record: record["confirmatory_status"].__setitem__(
                "confirmatory_claim_ready", True
            ),
            "must not claim confirmatory readiness",
        ),
        (
            lambda record: record.__setitem__("schema_version", "wrong"),
            "schema_version must be",
        ),
        (
            lambda record: record["environment"]["torch_cpu_execution"].__setitem__(
                "quantized_engine", "fbgemm"
            ),
            "current quantized engine must be qnnpack",
        ),
    ),
)
def test_missing_or_invalid_contract_fields_fail_strictly(
    tmp_path: Path,
    change: Callable[[dict[str, Any]], object],
    reason_fragment: str,
) -> None:
    first, second = _pair(tmp_path)
    changed = _record(2, timing_offset=100.0)
    change(changed)
    _write(second, changed)

    aggregate = verifier.verify_repeat_records(first, second, project_root=tmp_path)

    assert aggregate["status"] == "failed"
    assert any(reason_fragment in reason for reason in aggregate["reasons"])


@pytest.mark.parametrize(
    "invalid_payload",
    (
        b'{"schema_version": NaN}',
        b'{"schema_version": 1e400}',
        b'{"schema_version": 1, "schema_version": 2}',
        b'{"schema_version": "\\ud800"}',
        b"[]",
        b"not-json",
    ),
)
def test_invalid_duplicate_or_nonfinite_json_is_rejected_and_retained(
    tmp_path: Path, invalid_payload: bytes
) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    _rewrite_sealed_bytes(second / "record.json", invalid_payload)
    output = verifier._AGGREGATE_PATH

    previous = Path.cwd()
    try:
        os.chdir(tmp_path)
        exit_code = verifier.main(
            [
                str(first),
                str(second),
                "--preregistration",
                str(preregistration),
                "--output",
                str(output),
            ]
        )
    finally:
        os.chdir(previous)

    aggregate = json.loads((tmp_path / output).read_text(encoding="utf-8"))
    assert exit_code == 1
    assert aggregate["status"] == "failed"
    assert stat.S_IMODE((tmp_path / output).stat().st_mode) == 0o444
    assert stat.S_IMODE((tmp_path / verifier._ATTEMPT_ROOT).stat().st_mode) == 0o555
    assert any("input_2_invalid" in reason for reason in aggregate["reasons"])


def test_output_path_and_atomic_publication_are_safe(tmp_path: Path) -> None:
    output = verifier.resolve_output_path(tmp_path, Path("artifacts/repeats/result.json"))
    verifier.atomic_write_json(output, {"finite": 1.0})
    original = output.read_bytes()
    assert stat.S_IMODE(output.stat().st_mode) == 0o444

    with pytest.raises(FileExistsError, match="immutable output"):
        verifier.resolve_output_path(tmp_path, Path("artifacts/repeats/result.json"))
    with pytest.raises(FileExistsError):
        verifier.atomic_write_json(output, {"replacement": True})
    assert output.read_bytes() == original
    with pytest.raises(ValueError, match="project-relative"):
        verifier.resolve_output_path(tmp_path, tmp_path / "artifacts/absolute.json")
    with pytest.raises(ValueError, match="cannot contain"):
        verifier.resolve_output_path(tmp_path, Path("artifacts/../escape.json"))
    with pytest.raises(ValueError, match="below the project artifacts"):
        verifier.resolve_output_path(tmp_path, Path("docs/result.json"))
    with pytest.raises(ValueError, match=r"\.json suffix"):
        verifier.resolve_output_path(tmp_path, Path("artifacts/repeats/result.txt"))
    with pytest.raises(ValueError, match="Out of range float values"):
        verifier.atomic_write_json(tmp_path / "nan.json", {"bad": float("nan")})

    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "artifacts" / "escape"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="below the project artifacts"):
        verifier.resolve_output_path(tmp_path, Path("artifacts/escape/result.json"))

    with pytest.raises(ValueError, match="exactly the preregistered path"):
        verifier.resolve_claim_output_path(tmp_path, Path("artifacts/other.json"))


def test_fixed_attempt_envelopes_and_preregistration_produce_the_only_claim(
    tmp_path: Path,
) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    ignored_documentation = {
        "AGENTS.md",
        "SPEC.md",
        "SDD.md",
        "TDD.md",
        "docs/adr/ADR-025-prospective-tinystories-int8-noninferiority.md",
    }

    aggregate = verifier.verify_confirmatory_attempts(
        first, second, preregistration, project_root=tmp_path
    )

    assert aggregate["status"] == "confirmed"
    assert aggregate["confirmatory_pass"] is True
    assert aggregate["confirmatory_claim_ready"] is True
    assert aggregate["claim_producing_path"] is True
    assert aggregate["preregistration_sha256"] == _sha256_file(preregistration)
    bound_files = json.loads(preregistration.read_text(encoding="utf-8"))["bindings"]["files"]
    assert ignored_documentation.isdisjoint(bound_files)
    assert all(not (tmp_path / relative).exists() for relative in ignored_documentation)
    assert aggregate["bindings"]["bound_file_count"] == len(bound_files)
    serialized = json.dumps(aggregate, sort_keys=True)
    assert "story-000147" not in serialized
    assert "mean_nll" not in serialized
    assert _OUTPUT_HASH not in serialized


def test_project_venv_python_symlink_binds_its_resolved_target(tmp_path: Path) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    invocation = tmp_path / ".venv/bin/python"

    aggregate = verifier.verify_confirmatory_attempts(
        first, second, preregistration, project_root=tmp_path
    )

    assert invocation.is_symlink()
    assert aggregate["status"] == "confirmed"
    declared = json.loads(preregistration.read_text(encoding="utf-8"))["environment_contract"][
        "python_executable"
    ]
    assert declared["invocation_path"] == str(invocation)
    assert declared["resolved_path"] == str(invocation.resolve(strict=True))


def test_swapped_or_arbitrary_attempt_paths_can_never_produce_a_claim(tmp_path: Path) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)

    aggregate = verifier.verify_confirmatory_attempts(
        second, first, preregistration, project_root=tmp_path
    )

    assert aggregate["status"] == "failed"
    assert aggregate["confirmatory_claim_ready"] is False
    assert sum("requested_path_is_not_fixed" in reason for reason in aggregate["reasons"]) == 2


@pytest.mark.parametrize("target", ("start", "completion", "record", "preregistration"))
def test_tampered_envelope_or_preregistration_fails_closed(tmp_path: Path, target: str) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    if target == "start":
        path = first / "start.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["shell"] = True
        _rewrite_sealed_json(path, value)
    elif target == "completion":
        path = first / "completion.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["preflight_complete"] = False
        _rewrite_sealed_json(path, value)
    elif target == "record":
        path = second / "record.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["baseline_config_hash"] = "0" * 64
        _rewrite_sealed_json(path, value)
    else:
        value = json.loads(preregistration.read_text(encoding="utf-8"))
        value["outcomes"]["holdout_outcomes_inspected"] = True
        _write(preregistration, value)

    aggregate = verifier.verify_confirmatory_attempts(
        first, second, preregistration, project_root=tmp_path
    )

    assert aggregate["status"] == "failed"
    assert aggregate["confirmatory_claim_ready"] is False


def test_non_runtime_document_binding_cannot_enter_a_future_preregistration(
    tmp_path: Path,
) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    value = json.loads(preregistration.read_text(encoding="utf-8"))
    value["bindings"]["files"]["AGENTS.md"] = "0" * 64
    _write(preregistration, value)

    aggregate = verifier.verify_confirmatory_attempts(
        first, second, preregistration, project_root=tmp_path
    )

    assert not (tmp_path / "AGENTS.md").exists()
    assert aggregate["status"] == "failed"
    assert any("non-runtime bound project files" in reason for reason in aggregate["reasons"])


def test_symlinked_envelope_file_fails_closed(tmp_path: Path) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    start = first / "start.json"
    moved = tmp_path / "moved-start.json"
    first.chmod(0o755)
    start.rename(moved)
    start.symlink_to(moved)
    first.chmod(0o555)

    aggregate = verifier.verify_confirmatory_attempts(
        first, second, preregistration, project_root=tmp_path
    )

    assert aggregate["status"] == "failed"
    assert any("start_invalid" in reason for reason in aggregate["reasons"])


def test_unbound_local_import_provenance_fails_closed(tmp_path: Path) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    start_path = first / "start.json"
    start = json.loads(start_path.read_text(encoding="utf-8"))
    start["source_import_contract"]["verified_local_modules"]["scripts"]["source_sha256"] = "0" * 64
    _rewrite_sealed_json(start_path, start)
    completion_path = first / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["start"] = _fact(start_path)
    _rewrite_sealed_json(completion_path, completion)

    aggregate = verifier.verify_confirmatory_attempts(
        first, second, preregistration, project_root=tmp_path
    )

    assert aggregate["status"] == "failed"
    assert any("source is absent or stale" in reason for reason in aggregate["reasons"])


def test_stale_extra_bound_source_fails_preregistration(tmp_path: Path) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    (tmp_path / "src/inkling_quant_lab/config.py").write_text(
        "tampered imported source\n", encoding="utf-8"
    )

    aggregate = verifier.verify_confirmatory_attempts(
        first, second, preregistration, project_root=tmp_path
    )

    assert aggregate["status"] == "failed"
    assert any("bound project file is stale" in reason for reason in aggregate["reasons"])


@pytest.mark.parametrize("target", ("directory", "record"))
def test_writable_attempt_mode_fails_closed(tmp_path: Path, target: str) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    if target == "directory":
        second.chmod(0o755)
        expected_reason = "attempt_2_directory_mode_is_not_0555"
    else:
        (second / "record.json").chmod(0o644)
        expected_reason = "attempt_2_record_mode_is_not_0444"

    aggregate = verifier.verify_confirmatory_attempts(
        first, second, preregistration, project_root=tmp_path
    )

    assert aggregate["status"] == "failed"
    assert expected_reason in aggregate["reasons"]


def test_extra_attempt_root_entry_fails_closed(tmp_path: Path) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    (tmp_path / verifier._ATTEMPT_ROOT / "attempt-3").mkdir()

    aggregate = verifier.verify_confirmatory_attempts(
        first, second, preregistration, project_root=tmp_path
    )

    assert aggregate["status"] == "failed"
    assert "fixed_attempt_root_entries_are_not_exact" in aggregate["reasons"]


def test_identity_mismatch_fails_even_when_completion_hash_is_refreshed(tmp_path: Path) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    record_path = second / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["dataset"]["full_file_story_count"] = 21_989
    _rewrite_sealed_json(record_path, record)
    _refresh_completion_record_fact(second, 2)

    aggregate = verifier.verify_confirmatory_attempts(
        first, second, preregistration, project_root=tmp_path
    )

    assert aggregate["status"] == "failed"
    assert any("dataset differs from preregistration" in reason for reason in aggregate["reasons"])


@pytest.mark.parametrize("impossible_order", ("start_before_prereg", "completion_before_start"))
def test_impossible_envelope_timestamp_order_fails(tmp_path: Path, impossible_order: str) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    if impossible_order == "start_before_prereg":
        start_path = first / "start.json"
        start = json.loads(start_path.read_text(encoding="utf-8"))
        start["created_at_utc"] = "2026-07-15T23:58:00+00:00"
        _rewrite_sealed_json(start_path, start)
        completion_path = first / "completion.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["start"] = _fact(start_path)
        _rewrite_sealed_json(completion_path, completion)
    else:
        completion_path = first / "completion.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["created_at_utc"] = "2026-07-15T23:59:30+00:00"
        _rewrite_sealed_json(completion_path, completion)

    aggregate = verifier.verify_confirmatory_attempts(
        first, second, preregistration, project_root=tmp_path
    )

    assert aggregate["status"] == "failed"
    assert any(
        fragment in reason
        for reason in aggregate["reasons"]
        for fragment in ("started_before_preregistration", "completed_before_it_started")
    )


def test_raw_record_timestamp_outside_envelope_fails(tmp_path: Path) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    record_path = first / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["created_at_utc"] = "2026-07-16T00:00:12+00:00"
    _rewrite_sealed_json(record_path, record)
    _refresh_completion_record_fact(first, 1)

    aggregate = verifier.verify_confirmatory_attempts(
        first, second, preregistration, project_root=tmp_path
    )

    assert aggregate["status"] == "failed"
    assert "attempt_1_record_timestamp_outside_envelope" in aggregate["reasons"]


def test_attempt_two_cannot_start_before_attempt_one_completes(tmp_path: Path) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    start_path = second / "start.json"
    start = json.loads(start_path.read_text(encoding="utf-8"))
    start["created_at_utc"] = "2026-07-16T00:00:04+00:00"
    _rewrite_sealed_json(start_path, start)
    completion_path = second / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["start"] = _fact(start_path)
    _rewrite_sealed_json(completion_path, completion)

    aggregate = verifier.verify_confirmatory_attempts(
        first, second, preregistration, project_root=tmp_path
    )

    assert aggregate["status"] == "failed"
    assert "attempt_2_started_before_attempt_1_completed" in aggregate["reasons"]


def test_confirmed_cli_writes_immutable_aggregate_and_returns_zero(tmp_path: Path) -> None:
    first, second, preregistration = _sealed_pair(tmp_path)
    output = verifier._AGGREGATE_PATH

    previous = Path.cwd()
    try:
        os.chdir(tmp_path)
        exit_code = verifier.main(
            [
                str(first),
                str(second),
                "--preregistration",
                str(preregistration),
                "--output",
                str(output),
            ]
        )
    finally:
        os.chdir(previous)

    aggregate = json.loads((tmp_path / output).read_text(encoding="utf-8"))
    assert exit_code == 0
    assert aggregate["status"] == "confirmed"
    assert aggregate["confirmatory_pass"] is True
    assert stat.S_IMODE((tmp_path / output).stat().st_mode) == 0o444
    assert stat.S_IMODE((tmp_path / verifier._ATTEMPT_ROOT).stat().st_mode) == 0o555
