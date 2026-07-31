from __future__ import annotations

import hashlib
import math
import statistics
from typing import Any, Literal

import pytest
from pydantic import ValidationError

from inkling_quant_lab.gguf.inkling_measurement_control import (
    MEASUREMENT_CONTROL_RECORD_MAX_BYTES,
    MEASUREMENT_PLANNED_STAGES,
    MEASUREMENT_RUNTIME_COMMANDS,
    MeasurementAppliedPatch,
    MeasurementBenchCase,
    MeasurementBenchCaseRollup,
    MeasurementLlamaBenchCaseIdentity,
    MeasurementLlamaBenchWorkloadIdentity,
    MeasurementPairedBytes,
    MeasurementPairedFiveBatchMetricSummary,
    MeasurementPairedFiveBatchNonnegativeMetricSummary,
    MeasurementPairedGpuBytes,
    MeasurementPairedGpuUtilization,
    MeasurementPairedNonnegativeValue,
    MeasurementPairedPositiveValue,
    MeasurementPairedRepeatedLoadDurations,
    MeasurementPerformanceRollup,
    MeasurementPrePatchExecutable,
    MeasurementQualityRollup,
    MeasurementQualitySuite,
    MeasurementRuntimeCommand,
    MeasurementRuntimeCommandClosure,
    MeasurementRuntimeDependency,
    MeasurementRuntimeIdentity,
    MeasurementRuntimeRegularFile,
    MeasurementRuntimeSymlink,
    MeasurementServerCellRollup,
    MeasurementServerWorkloadIdentity,
    MeasurementSuccessTerminalReceipt,
    MeasurementSuiteQuality,
    MeasurementSupportingRecordKind,
    MeasurementSupportingRecordReference,
    build_measurement_supporting_record_reference,
    build_measurement_terminal_receipt_reference,
    canonical_measurement_json_bytes,
    measurement_absolute_evidence_path,
    measurement_llama_bench_dataset_bytes,
    measurement_performance_rollup_sha256,
    measurement_quality_rollup_sha256,
    measurement_runtime_manifest_sha256,
    measurement_server_prompt_source_text,
    parse_measurement_terminal_receipt,
    strict_measurement_json_object,
    validate_absolute_evidence_path,
    validate_measurement_supporting_record_reference,
    validate_measurement_terminal_receipt_reference,
    validate_repository_relative_path,
)

RUN_ID = "inkling-measurement-control-contracts"
EVIDENCE_ROOT = "/measurement-evidence"


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _paired_positive(bf16: float, q3: float) -> MeasurementPairedPositiveValue:
    return MeasurementPairedPositiveValue(
        bf16=bf16,
        q3=q3,
        q3_to_bf16_ratio=q3 / bf16,
    )


def _paired_nonnegative(
    bf16: float,
    q3: float,
) -> MeasurementPairedNonnegativeValue:
    return MeasurementPairedNonnegativeValue(bf16=bf16, q3=q3)


def _runtime_regular_files() -> tuple[MeasurementRuntimeRegularFile, ...]:
    values = (
        ("libllama-perplexity-impl.so", "perplexity-dso", 37),
        ("llama-bench", "llama-bench", 31),
        ("llama-cli", "llama-cli", 29),
        ("llama-perplexity", "llama-perplexity", 41),
        ("llama-server-real", "llama-server", 43),
    )
    return tuple(
        MeasurementRuntimeRegularFile(
            relative_path=path,
            sha256=_digest(identity),
            size_bytes=size_bytes,
        )
        for path, identity, size_bytes in values
    )


def _runtime_symlinks() -> tuple[MeasurementRuntimeSymlink, ...]:
    return (
        MeasurementRuntimeSymlink(
            relative_path="llama-server",
            raw_target="llama-server-real",
            resolved_relative_path="llama-server-real",
        ),
    )


def _runtime_dependencies(
    command: MeasurementRuntimeCommand,
) -> tuple[MeasurementRuntimeDependency, ...]:
    dependencies = [
        MeasurementRuntimeDependency(
            classification="system",
            soname="libc.so.6",
            resolved_path="/lib/x86_64-linux-gnu/libc.so.6",
            sha256=_digest("libc"),
            size_bytes=1_024,
        ),
        MeasurementRuntimeDependency(
            classification="cuda",
            soname="libcuda.so.1",
            resolved_path="/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            sha256=_digest("libcuda"),
            size_bytes=2_048,
        ),
    ]
    if command == "llama-perplexity":
        dependencies.append(
            MeasurementRuntimeDependency(
                classification="project_owned",
                soname="libllama-perplexity-impl.so",
                resolved_path="/opt/llama.cpp/build/bin/libllama-perplexity-impl.so",
                sha256=_digest("perplexity-dso"),
                size_bytes=37,
            )
        )
    dependencies.append(
        MeasurementRuntimeDependency(
            classification="virtual",
            soname="linux-vdso.so.1",
        )
    )
    return tuple(sorted(dependencies, key=lambda item: (item.soname, item.resolved_path or "")))


def _runtime_commands() -> tuple[
    MeasurementRuntimeCommandClosure,
    MeasurementRuntimeCommandClosure,
    MeasurementRuntimeCommandClosure,
    MeasurementRuntimeCommandClosure,
]:
    sizes = {
        "llama-cli": 29,
        "llama-server": 43,
        "llama-bench": 31,
        "llama-perplexity": 41,
    }
    commands: tuple[MeasurementRuntimeCommand, ...] = (
        "llama-cli",
        "llama-server",
        "llama-bench",
        "llama-perplexity",
    )
    closures = tuple(
        MeasurementRuntimeCommandClosure(
            command=command,
            binary_path=f"/opt/llama.cpp/build/bin/{command}",
            binary_manifest_path=command,
            binary_sha256=_digest(command),
            binary_size_bytes=sizes[command],
            ldd_output_sha256=_digest(f"{command}-ldd"),
            dependencies=_runtime_dependencies(command),
        )
        for command in commands
    )
    return (closures[0], closures[1], closures[2], closures[3])


def _runtime_identity(**overrides: Any) -> MeasurementRuntimeIdentity:
    regular_files = _runtime_regular_files()
    symlinks = _runtime_symlinks()
    values: dict[str, Any] = {
        "repository": "https://github.com/danielhanchen/llama.cpp.git",
        "repository_commit": "a015409e6c27b84f60d688823d4c0126a11571fd",
        "cuda_image": "nvidia/cuda:13.1.2-devel-ubuntu24.04",
        "cuda_image_digest": (
            "sha256:952e42d23230610a2714c8484f38e9c934ed68e6f9c9c7fac62dcd5f98858a6e"
        ),
        "platform": "linux/amd64",
        "patches_applied_in_order": (
            MeasurementAppliedPatch(
                path="patches/inkling-smoke-a015409.patch",
                sha256="005f1f342511fc3fc843bdcc7be814ed8a60e67033b733eb7e7e4af53925be04",
                size_bytes=48_409,
            ),
            MeasurementAppliedPatch(
                path="patches/inkling-measurement-a015409.patch",
                sha256="2758f3dda9a6d954793705f750feb517193cf77545cc7f92c68ad44edd12c29f",
                size_bytes=8_660,
            ),
        ),
        "base_pre_measurement_patch_executables": (
            MeasurementPrePatchExecutable(
                name="llama-cli",
                path="/opt/llama.cpp/build/bin/llama-cli",
                sha256="098d8b9c6e57f25b846c5b5b43ded5bb1194cbb3d1ce985f17bbd09c87a82dbc",
                size_bytes=1_246_680,
            ),
            MeasurementPrePatchExecutable(
                name="llama-server",
                path="/opt/llama.cpp/build/bin/llama-server",
                sha256="e960cfe4dcb2f7e541fc0b15bf97a4c1f6feb5fc304267796ef2bdd004cd1b93",
                size_bytes=17_920,
            ),
            MeasurementPrePatchExecutable(
                name="llama-bench",
                path="/opt/llama.cpp/build/bin/llama-bench",
                sha256="e0844ac337c419ebd8b6cee4902ba13e210a067d6fe47cb652429c71ae97382b",
                size_bytes=17_920,
            ),
            MeasurementPrePatchExecutable(
                name="llama-perplexity",
                path="/opt/llama.cpp/build/bin/llama-perplexity",
                sha256="d04051888a157ee50a7d6286cffcc78da3a9ca5295c79aa99ea2d92672ebf733",
                size_bytes=15_968,
            ),
        ),
        "cmake_generator": "Ninja",
        "effective_cmake_definitions": (
            "CMAKE_BUILD_TYPE=Release",
            "BUILD_SHARED_LIBS=ON",
            "GGML_CUDA=ON",
            "GGML_NATIVE=OFF",
            "LLAMA_CURL=OFF",
            "LLAMA_BUILD_UI=OFF",
            "LLAMA_USE_PREBUILT_UI=OFF",
            "CMAKE_CUDA_ARCHITECTURES=103",
            "CMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/iql-cuda-driver-link",
            "CMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE",
        ),
        "build_targets": MEASUREMENT_RUNTIME_COMMANDS,
        "build_shared_libs": True,
        "cmake_version": "cmake version 3.31.6",
        "cxx_compiler_version": "g++ 13.3.0",
        "cuda_compiler_version": "Cuda compilation tools, release 13.1",
        "build_bin_root": "/opt/llama.cpp/build/bin",
        "regular_files": regular_files,
        "symlinks": symlinks,
        "commands": _runtime_commands(),
        "manifest_sha256": measurement_runtime_manifest_sha256(
            build_bin_root="/opt/llama.cpp/build/bin",
            regular_files=regular_files,
            symlinks=symlinks,
        ),
    }
    values.update(overrides)
    return MeasurementRuntimeIdentity(**values)


def _llama_bench_workload() -> MeasurementLlamaBenchWorkloadIdentity:
    content = measurement_llama_bench_dataset_bytes()
    prompt_template = (
        b"inkling-llama-bench-prompt-template-v1\0"
        b"c_stdlib_rand_default_seed_1_without_srand\0"
        b"first_token_optional_model_bos_else_rand_mod_vocab\0"
        b"remaining_tokens_rand_mod_vocab"
    )
    return MeasurementLlamaBenchWorkloadIdentity(
        schema_version="inkling-llama-bench-workload-v1",
        dataset_id="llama.cpp/llama-bench-synthetic-token-workload",
        dataset_revision="a015409e6c27b84f60d688823d4c0126a11571fd",
        split="benchmark",
        content_sha256=hashlib.sha256(content).hexdigest(),
        content_size_bytes=len(content),
        ordered_sample_ids=("pp512", "pp2048", "tg128"),
        seed=1,
        seed_protocol="c_stdlib_rand_default_seed_1_without_srand",
        prompt_template_sha256=hashlib.sha256(prompt_template).hexdigest(),
        prompt_template_protocol=(
            "c_stdlib_rand_default_seed_1_without_srand_with_optional_bos_first_token"
        ),
        execution_mode="single_process_single_model_load_ordered_cases",
        cases=(
            MeasurementLlamaBenchCaseIdentity(
                sample_id="pp512",
                prompt_tokens=512,
                generation_tokens=0,
            ),
            MeasurementLlamaBenchCaseIdentity(
                sample_id="pp2048",
                prompt_tokens=2_048,
                generation_tokens=0,
            ),
            MeasurementLlamaBenchCaseIdentity(
                sample_id="tg128",
                prompt_tokens=0,
                generation_tokens=128,
            ),
        ),
    )


def _server_workload() -> MeasurementServerWorkloadIdentity:
    content = measurement_server_prompt_source_text().encode()
    prompt_template = (
        b"inkling-server-prompt-template-v1\0"
        b"repeat_utf8_literal=matched Inkling measurement input \\x20\0"
        b"repeat_count=2048\0"
        b"tokenize_add_special=false\0"
        b"tokenize_parse_special=false\0"
        b"take_first_token_ids=512"
    )
    return MeasurementServerWorkloadIdentity(
        schema_version="inkling-server-benchmark-workload-v1",
        dataset_id="inkling-quant-lab/synthetic-server-benchmark",
        dataset_revision="inkling-server-benchmark-v1",
        split="benchmark",
        content_sha256=hashlib.sha256(content).hexdigest(),
        content_size_bytes=len(content),
        ordered_sample_ids=("server_prompt_0001",),
        seed=42,
        prompt_template_sha256=hashlib.sha256(prompt_template).hexdigest(),
        prompt_template_protocol=(
            "repeat_utf8_literal_2048_then_tokenize_without_special_tokens_then_take_first_512_ids"
        ),
        prompt_tokens=512,
        output_tokens=128,
        temperature=0.0,
        streaming=True,
        cache_prompt=False,
        return_tokens=True,
        ignore_eos=True,
        execution_mode="llama_server_streaming_completion_concurrency_1_2_4",
    )


def _positive_five_batch(
    bf16: tuple[float, float, float, float, float] = (1.0, 2.0, 3.0, 4.0, 5.0),
    q3: tuple[float, float, float, float, float] = (2.0, 3.0, 4.0, 5.0, 6.0),
) -> MeasurementPairedFiveBatchMetricSummary:
    return MeasurementPairedFiveBatchMetricSummary(
        trial_count_per_subject=5,
        bf16_samples=bf16,
        q3_samples=q3,
        mean=_paired_positive(statistics.fmean(bf16), statistics.fmean(q3)),
        median=_paired_positive(statistics.median(bf16), statistics.median(q3)),
        sample_standard_deviation=_paired_nonnegative(
            statistics.stdev(bf16),
            statistics.stdev(q3),
        ),
    )


def _nonnegative_five_batch(
    bf16: tuple[float, float, float, float, float],
    q3: tuple[float, float, float, float, float],
) -> MeasurementPairedFiveBatchNonnegativeMetricSummary:
    return MeasurementPairedFiveBatchNonnegativeMetricSummary(
        trial_count_per_subject=5,
        bf16_samples=bf16,
        q3_samples=q3,
        mean=_paired_nonnegative(statistics.fmean(bf16), statistics.fmean(q3)),
        median=_paired_nonnegative(statistics.median(bf16), statistics.median(q3)),
        sample_standard_deviation=_paired_nonnegative(
            statistics.stdev(bf16),
            statistics.stdev(q3),
        ),
    )


def _server_cell(concurrency: Literal[1, 2, 4]) -> MeasurementServerCellRollup:
    return MeasurementServerCellRollup(
        concurrency=concurrency,
        measured_batches_per_subject=5,
        measured_requests_per_subject=5 * concurrency,
        request_end_to_end_latency_seconds=_positive_five_batch(),
        ttft_seconds=_positive_five_batch(),
        prompt_tokens_per_second=_positive_five_batch(),
        decode_tokens_per_second=_positive_five_batch(),
        aggregate_decode_tokens_per_second=_positive_five_batch(),
        inter_token_latency_p50_seconds=_nonnegative_five_batch(
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        inter_token_latency_p95_seconds=_nonnegative_five_batch(
            (0.1, 0.1, 0.1, 0.1, 0.1),
            (0.2, 0.2, 0.2, 0.2, 0.2),
        ),
        inter_token_latency_p99_seconds=_nonnegative_five_batch(
            (0.2, 0.2, 0.2, 0.2, 0.2),
            (0.3, 0.3, 0.3, 0.3, 0.3),
        ),
        bf16_resource_sample_count=5,
        q3_resource_sample_count=5,
        max_sampled_host_rss_bytes=MeasurementPairedBytes(bf16=1_000, q3=800),
        max_sampled_per_gpu_memory_bytes=MeasurementPairedGpuBytes(
            bf16=(1_000,) * 8,
            q3=(800,) * 8,
        ),
        max_sampled_per_gpu_utilization_percent=MeasurementPairedGpuUtilization(
            bf16=(90.0,) * 8,
            q3=(85.0,) * 8,
        ),
    )


def _load_durations() -> MeasurementPairedRepeatedLoadDurations:
    bf16 = (3.0, 4.0, 5.0)
    q3 = (2.0, 3.0, 4.0)
    return MeasurementPairedRepeatedLoadDurations(
        trial_count_per_subject=3,
        bf16_durations_seconds=bf16,
        q3_durations_seconds=q3,
        median_seconds=_paired_positive(statistics.median(bf16), statistics.median(q3)),
        sample_standard_deviation_seconds=_paired_nonnegative(
            statistics.stdev(bf16),
            statistics.stdev(q3),
        ),
    )


def _quality_rollup(**overrides: Any) -> MeasurementQualityRollup:
    suite_order: tuple[MeasurementQualitySuite, ...] = (
        "text",
        "math",
        "code",
        "multilingual",
        "instruction",
        "vision",
        "audio",
        "post_training",
    )
    suites = tuple(
        MeasurementSuiteQuality(
            suite=suite,
            item_count=8,
            bf16_accuracy=0.875,
            q3_accuracy=0.875,
            accuracy_loss=0.0,
            bf16_floor_passed=True,
            q3_non_inferiority_passed=True,
        )
        for suite in suite_order
    )
    values: dict[str, Any] = {
        "paired_token_positions": 16_320,
        "bf16_diagnostic_items_scored": 64,
        "q3_diagnostic_items_scored": 64,
        "bf16_mean_nll": 2.0,
        "q3_mean_nll": 2.05,
        "mean_nll_delta": 0.05,
        "bf16_perplexity": math.exp(2.0),
        "q3_perplexity": math.exp(2.05),
        "bf16_overall_accuracy": 0.875,
        "q3_overall_accuracy": 0.875,
        "overall_accuracy_loss": 0.0,
        "suites": suites,
        "all_items_scored": True,
        "all_suites_interpretable": True,
        "mean_nll_gate_passed": True,
        "overall_accuracy_gate_passed": True,
        "bf16_overall_floor_passed": True,
        "bf16_suite_floors_passed": True,
        "suite_accuracy_gates_passed": True,
        "non_inferiority_passed": True,
    }
    values.update(overrides)
    return MeasurementQualityRollup(**values)


def _performance_rollup(**overrides: Any) -> MeasurementPerformanceRollup:
    case_order: tuple[MeasurementBenchCase, ...] = ("pp512", "pp2048", "tg128")
    concurrency_order: tuple[Literal[1, 2, 4], ...] = (1, 2, 4)
    bench_cases = tuple(
        MeasurementBenchCaseRollup(
            case=case,
            repetitions_per_subject=5,
            average_tokens_per_second=_paired_positive(100.0, 120.0),
            median_tokens_per_second=_paired_positive(100.0, 120.0),
            standard_deviation_tokens_per_second=_paired_nonnegative(0.0, 0.0),
        )
        for case in case_order
    )
    values: dict[str, Any] = {
        "llama_bench_workload_identity": _llama_bench_workload(),
        "server_workload_identity": _server_workload(),
        "text_checkpoint_size_bytes": MeasurementPairedBytes(bf16=100, q3=60),
        "multimodal_projector_size_bytes": MeasurementPairedBytes(bf16=20, q3=20),
        "executable_gguf_bundle_size_bytes": MeasurementPairedBytes(bf16=120, q3=80),
        "load_pair_repetitions_per_subject": 3,
        "workload_load_pair_trial_index": 3,
        "cold_server_load_trials": _load_durations(),
        "warm_server_load_trials": _load_durations(),
        "warm_load_protocol": (
            "second_same_artifact_process_after_cold_termination_without_requested_"
            "cache_conditioning_or_eviction"
        ),
        "requested_telemetry_sampling_interval_seconds": 1.0,
        "bench_cases": bench_cases,
        "server_cells": tuple(_server_cell(concurrency) for concurrency in concurrency_order),
        "single_request_warmups_per_subject": 2,
        "concurrent_batch_warmups_per_cell_per_subject": 1,
        "bench_cases_share_one_model_load": True,
        "server_quality_and_performance_share_one_model_load": True,
        "warmups_excluded_from_measurement": True,
        "raw_trials_recorded": True,
        "matched_runtime_hardware_workload": True,
        "all_metrics_complete": True,
        "equivalent_trials_valid": True,
        "comparison_complete": True,
        "speedup_claim_allowed": True,
    }
    values.update(overrides)
    return MeasurementPerformanceRollup(**values)


def _supporting_records() -> tuple[
    MeasurementSupportingRecordReference,
    MeasurementSupportingRecordReference,
    MeasurementSupportingRecordReference,
]:
    kinds: tuple[MeasurementSupportingRecordKind, ...] = (
        "bf16_subject",
        "q3_subject",
        "comparison",
    )
    records = tuple(
        build_measurement_supporting_record_reference(
            canonical_measurement_json_bytes({"kind": kind}),
            run_id=RUN_ID,
            kind=kind,
        )
        for kind in kinds
    )
    return (records[0], records[1], records[2])


def _success_receipt(**overrides: Any) -> MeasurementSuccessTerminalReceipt:
    runtime = _runtime_identity()
    quality = _quality_rollup()
    performance = _performance_rollup()
    values: dict[str, Any] = {
        "run_id": RUN_ID,
        "control_plane_sha256": _digest("control-plane"),
        "reviewed_config_file_sha256": _digest("reviewed-config"),
        "resolved_config_sha256": _digest("resolved-config"),
        "launch_intent_sha256": _digest("launch-intent"),
        "post_spawn_acceptance_sha256": _digest("post-spawn-acceptance"),
        "call_id": "fc-MeasurementControlContracts",
        "attempt_claim_sha256": _digest("attempt-claim"),
        "completed_at_utc": "2026-07-30T18:00:00.000000Z",
        "completed_stages": MEASUREMENT_PLANNED_STAGES,
        "runtime_identity": runtime,
        "runtime_manifest_sha256": runtime.manifest_sha256,
        "hardware_identity_sha256": _digest("hardware-identity"),
        "model_id": "thinkingmachines/Inkling",
        "model_revision": "86b4d430ab871652a707666b89203a866888c5e5",
        "protocol_sha256": _digest("protocol"),
        "workload_sha256": _digest("workload"),
        "supporting_records": _supporting_records(),
        "quality_rollup": quality,
        "performance_rollup": performance,
        "quality_rollup_sha256": measurement_quality_rollup_sha256(quality),
        "performance_rollup_sha256": measurement_performance_rollup_sha256(performance),
        "quality_retention_passed": True,
        "performance_comparison_complete": True,
        "speedup_claim_allowed": True,
    }
    values.update(overrides)
    return MeasurementSuccessTerminalReceipt(**values)


def test_strict_control_json_rejects_ambiguous_or_unbounded_records() -> None:
    assert strict_measurement_json_object(b'{"a":1}\n') == {"a": 1}
    assert strict_measurement_json_object('{"a":1}\n') == {"a": 1}

    with pytest.raises(TypeError, match="bytes or text"):
        strict_measurement_json_object(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be UTF-8"):
        strict_measurement_json_object(b"\xff")
    with pytest.raises(ValueError, match="exceeds its size limit"):
        strict_measurement_json_object(
            b'{"value":"' + b"x" * MEASUREMENT_CONTROL_RECORD_MAX_BYTES + b'"}'
        )
    with pytest.raises(ValueError, match="duplicate key"):
        strict_measurement_json_object(b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="non-finite"):
        strict_measurement_json_object(b'{"value":NaN}')
    with pytest.raises(ValueError, match="JSON is invalid"):
        strict_measurement_json_object(b"{")
    with pytest.raises(ValueError, match="root must be an object"):
        strict_measurement_json_object(b"[]")


@pytest.mark.parametrize(
    "path",
    (
        "",
        "/absolute",
        "nested\\windows",
        "nested/../escape",
        "nested/./alias",
    ),
)
def test_repository_relative_paths_reject_noncanonical_aliases(path: str) -> None:
    with pytest.raises(ValueError, match="canonical and repository-relative"):
        validate_repository_relative_path(path)


@pytest.mark.parametrize(
    "path",
    (
        "/",
        "relative/path",
        "//double-root",
        "/nested\\windows",
        "/nested/../escape",
        "/nested/./alias",
    ),
)
def test_absolute_evidence_paths_reject_noncanonical_aliases(path: str) -> None:
    with pytest.raises(ValueError, match="canonical absolute POSIX path"):
        validate_absolute_evidence_path(path)


def test_absolute_evidence_path_join_preserves_exact_root_binding() -> None:
    assert (
        measurement_absolute_evidence_path(
            EVIDENCE_ROOT,
            "runs/example/receipt.json",
        )
        == f"{EVIDENCE_ROOT}/runs/example/receipt.json"
    )
    with pytest.raises(ValueError, match="canonical and repository-relative"):
        measurement_absolute_evidence_path(EVIDENCE_ROOT, "../outside.json")


def test_runtime_identity_roundtrips_exact_manifest_and_dependency_closure() -> None:
    runtime = _runtime_identity()

    assert (
        MeasurementRuntimeIdentity.model_validate_json(
            runtime.model_dump_json(),
            strict=True,
        )
        == runtime
    )
    assert runtime.commands[1].binary_manifest_path == "llama-server"
    assert runtime.symlinks[0].resolved_relative_path == "llama-server-real"
    assert runtime.commands[-1].dependencies[-2].classification == "project_owned"


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        (
            {
                "patches_applied_in_order": tuple(
                    reversed(_runtime_identity().patches_applied_in_order)
                )
            },
            "pinned application order",
        ),
        ({"effective_cmake_definitions": ("GGML_CUDA=OFF",)}, "CMake definitions"),
        ({"regular_files": tuple(reversed(_runtime_regular_files()))}, "sorted and unique"),
        ({"manifest_sha256": _digest("tampered-manifest")}, "manifest hash"),
    ),
)
def test_runtime_identity_rejects_tampered_build_bindings(
    overrides: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        _runtime_identity(**overrides)


def test_runtime_identity_rejects_unbound_project_dependency_and_stub_cuda() -> None:
    commands = list(_runtime_commands())
    perplexity = commands[-1]
    changed_dependencies = tuple(
        dependency
        if dependency.classification != "project_owned"
        else MeasurementRuntimeDependency(
            classification="project_owned",
            soname=dependency.soname,
            resolved_path="/opt/llama.cpp/build/bin/untracked.so",
            sha256=dependency.sha256,
            size_bytes=dependency.size_bytes,
        )
        for dependency in perplexity.dependencies
    )
    commands[-1] = MeasurementRuntimeCommandClosure(
        **{
            **perplexity.model_dump(),
            "dependencies": changed_dependencies,
        }
    )
    with pytest.raises(ValidationError, match="absent from the manifest"):
        _runtime_identity(commands=tuple(commands))

    with pytest.raises(ValidationError, match="stub library"):
        MeasurementRuntimeDependency(
            classification="cuda",
            soname="libcuda.so.1",
            resolved_path="/usr/local/cuda/lib64/stubs/libcuda.so.1",
            sha256=_digest("stub"),
            size_bytes=1,
        )


def test_runtime_identity_rejects_missing_or_misbound_runtime_inventory() -> None:
    runtime = _runtime_identity()

    with pytest.raises(ValidationError, match="pre-measurement executable identities"):
        _runtime_identity(
            base_pre_measurement_patch_executables=tuple(
                reversed(runtime.base_pre_measurement_patch_executables)
            )
        )
    with pytest.raises(ValidationError, match="must not be empty"):
        _runtime_identity(regular_files=())
    with pytest.raises(ValidationError, match="must resolve to an inventoried regular file"):
        _runtime_identity(
            symlinks=(
                MeasurementRuntimeSymlink(
                    relative_path="llama-server",
                    raw_target="missing-server",
                    resolved_relative_path="missing-server",
                ),
            )
        )
    with pytest.raises(ValidationError, match="closures must use the pinned order"):
        _runtime_identity(commands=tuple(reversed(runtime.commands)))

    missing_cli = tuple(item for item in runtime.regular_files if item.relative_path != "llama-cli")
    with pytest.raises(ValidationError, match="binary is absent from the manifest"):
        _runtime_identity(regular_files=missing_cli)

    changed_cli = tuple(
        MeasurementRuntimeRegularFile(
            relative_path=item.relative_path,
            sha256=_digest("changed-cli") if item.relative_path == "llama-cli" else item.sha256,
            size_bytes=item.size_bytes,
        )
        for item in runtime.regular_files
    )
    with pytest.raises(ValidationError, match="binary differs from its manifest identity"):
        _runtime_identity(regular_files=changed_cli)


def test_runtime_identity_rejects_symlink_and_project_dso_binding_tamper() -> None:
    runtime = _runtime_identity()
    with pytest.raises(ValidationError, match="symlink inventory must be sorted and unique"):
        _runtime_identity(
            symlinks=(
                MeasurementRuntimeSymlink(
                    relative_path="zz-link",
                    raw_target="llama-cli",
                    resolved_relative_path="llama-cli",
                ),
                runtime.symlinks[0],
            )
        )
    with pytest.raises(ValidationError, match="both regular files and symlinks"):
        _runtime_identity(
            symlinks=(
                MeasurementRuntimeSymlink(
                    relative_path="llama-cli",
                    raw_target="llama-cli",
                    resolved_relative_path="llama-cli",
                ),
            )
        )

    perplexity = runtime.commands[-1]
    project_dependency = next(
        item for item in perplexity.dependencies if item.classification == "project_owned"
    )

    def commands_with(
        dependency: MeasurementRuntimeDependency | None,
    ) -> tuple[MeasurementRuntimeCommandClosure, ...]:
        dependencies = tuple(
            item for item in perplexity.dependencies if item.classification != "project_owned"
        )
        if dependency is not None:
            dependencies = tuple(
                sorted(
                    (*dependencies, dependency),
                    key=lambda item: (item.soname, item.resolved_path or ""),
                )
            )
        changed = MeasurementRuntimeCommandClosure(
            **{
                **perplexity.model_dump(),
                "dependencies": dependencies,
            }
        )
        return (*runtime.commands[:-1], changed)

    outside = MeasurementRuntimeDependency(
        classification="project_owned",
        soname=project_dependency.soname,
        resolved_path="/opt/other/libllama-perplexity-impl.so",
        sha256=project_dependency.sha256,
        size_bytes=project_dependency.size_bytes,
    )
    with pytest.raises(ValidationError, match="outside build/bin"):
        _runtime_identity(commands=commands_with(outside))

    mismatched = MeasurementRuntimeDependency(
        classification="project_owned",
        soname=project_dependency.soname,
        resolved_path=project_dependency.resolved_path,
        sha256=_digest("changed-perplexity-dso"),
        size_bytes=project_dependency.size_bytes,
    )
    with pytest.raises(ValidationError, match="differs from its manifest identity"):
        _runtime_identity(commands=commands_with(mismatched))

    with pytest.raises(ValidationError, match="omits its patched implementation DSO"):
        _runtime_identity(commands=commands_with(None))


def test_runtime_dependency_and_command_contracts_reject_ambiguous_closures() -> None:
    with pytest.raises(ValidationError, match="must not claim file identity"):
        MeasurementRuntimeDependency(
            classification="virtual",
            soname="linux-vdso.so.1",
            resolved_path="/virtual",
            sha256=_digest("virtual"),
            size_bytes=1,
        )
    with pytest.raises(ValidationError, match="require complete identity"):
        MeasurementRuntimeDependency(
            classification="system",
            soname="libc.so.6",
            resolved_path="/lib/libc.so.6",
        )
    with pytest.raises(ValidationError, match="must not be empty"):
        MeasurementRuntimeCommandClosure(
            command="llama-cli",
            binary_path="/opt/llama.cpp/build/bin/llama-cli",
            binary_manifest_path="llama-cli",
            binary_sha256=_digest("llama-cli"),
            binary_size_bytes=29,
            ldd_output_sha256=_digest("llama-cli-ldd"),
            dependencies=(),
        )
    duplicate = MeasurementRuntimeDependency(
        classification="virtual",
        soname="linux-vdso.so.1",
    )
    with pytest.raises(ValidationError, match="sorted and unique"):
        MeasurementRuntimeCommandClosure(
            command="llama-cli",
            binary_path="/opt/llama.cpp/build/bin/llama-cli",
            binary_manifest_path="llama-cli",
            binary_sha256=_digest("llama-cli"),
            binary_size_bytes=29,
            ldd_output_sha256=_digest("llama-cli-ldd"),
            dependencies=(duplicate, duplicate),
        )


def test_workload_identities_reject_content_protocol_and_order_tamper() -> None:
    bench = _llama_bench_workload()
    bench_values = bench.model_dump()
    for bench_field_name, bench_value, bench_message in (
        ("content_sha256", _digest("wrong-bench-content"), "content identity"),
        ("prompt_template_sha256", _digest("wrong-bench-template"), "prompt-template hash"),
        ("ordered_sample_ids", ("pp2048", "pp512", "tg128"), "Input should be"),
    ):
        changed = dict(bench_values)
        changed[bench_field_name] = bench_value
        with pytest.raises(ValidationError, match=bench_message):
            MeasurementLlamaBenchWorkloadIdentity(**changed)

    server = _server_workload()
    server_values = server.model_dump()
    for server_field_name, server_value, server_message in (
        ("content_sha256", _digest("wrong-server-content"), "content identity"),
        ("temperature", 0.5, "exactly 0.0"),
        ("prompt_template_sha256", _digest("wrong-server-template"), "prompt-template hash"),
    ):
        changed = dict(server_values)
        changed[server_field_name] = server_value
        with pytest.raises(ValidationError, match=server_message):
            MeasurementServerWorkloadIdentity(**changed)


def test_supporting_record_reference_binds_exact_canonical_bytes() -> None:
    payload = canonical_measurement_json_bytes({"subject": "bf16"})
    reference = build_measurement_supporting_record_reference(
        payload,
        run_id=RUN_ID,
        kind="bf16_subject",
    )

    assert (
        validate_measurement_supporting_record_reference(payload, expected=reference) == reference
    )
    with pytest.raises(ValueError, match="differs from exact bytes"):
        validate_measurement_supporting_record_reference(
            canonical_measurement_json_bytes({"subject": "q3"}),
            expected=reference,
        )
    with pytest.raises(ValueError, match="not canonical"):
        build_measurement_supporting_record_reference(
            b'{ "subject": "bf16" }\n',
            run_id=RUN_ID,
            kind="bf16_subject",
        )
    with pytest.raises(TypeError, match="must be bytes"):
        build_measurement_supporting_record_reference(
            '{"subject":"bf16"}\n',  # type: ignore[arg-type]
            run_id=RUN_ID,
            kind="bf16_subject",
        )
    with pytest.raises(ValueError, match="exceeds its size limit"):
        build_measurement_supporting_record_reference(
            b'{"value":"' + b"x" * MEASUREMENT_CONTROL_RECORD_MAX_BYTES + b'"}',
            run_id=RUN_ID,
            kind="bf16_subject",
        )
    with pytest.raises(ValidationError, match="path differs"):
        MeasurementSupportingRecordReference(
            run_id=reference.run_id,
            kind=reference.kind,
            relative_path="records/wrong.json",
            content_sha256=reference.content_sha256,
            size_bytes=reference.size_bytes,
        )


def test_quality_rollup_roundtrips_and_rejects_derived_gate_tamper() -> None:
    quality = _quality_rollup()

    assert MeasurementQualityRollup.model_validate_json(quality.model_dump_json()) == quality
    assert measurement_quality_rollup_sha256(quality) == measurement_quality_rollup_sha256(quality)
    with pytest.raises(ValidationError, match="mean_nll_gate_passed"):
        _quality_rollup(mean_nll_gate_passed=False)
    with pytest.raises(ValidationError, match="perplexity differs"):
        _quality_rollup(q3_perplexity=1.0)


def test_quality_contract_rejects_suite_and_overall_derivation_tamper() -> None:
    with pytest.raises(ValidationError, match="exact multiple of one eighth"):
        MeasurementSuiteQuality(
            suite="text",
            item_count=8,
            bf16_accuracy=0.8,
            q3_accuracy=0.875,
            accuracy_loss=-0.075,
            bf16_floor_passed=True,
            q3_non_inferiority_passed=True,
        )
    with pytest.raises(ValidationError, match="accuracy loss differs"):
        MeasurementSuiteQuality(
            suite="text",
            item_count=8,
            bf16_accuracy=0.875,
            q3_accuracy=0.875,
            accuracy_loss=0.125,
            bf16_floor_passed=True,
            q3_non_inferiority_passed=True,
        )
    with pytest.raises(ValidationError, match="quality suites must use the checked order"):
        _quality_rollup(suites=tuple(reversed(_quality_rollup().suites)))
    with pytest.raises(ValidationError, match="mean NLL delta"):
        _quality_rollup(mean_nll_delta=0.0)
    with pytest.raises(ValidationError, match="non-inferiority result"):
        _quality_rollup(non_inferiority_passed=False)


def test_performance_rollup_accepts_zero_inter_token_latency_and_binds_sizes() -> None:
    performance = _performance_rollup()

    assert (
        MeasurementPerformanceRollup.model_validate_json(performance.model_dump_json())
        == performance
    )
    assert performance.server_cells[0].inter_token_latency_p50_seconds.bf16_samples == (0.0,) * 5
    with pytest.raises(ValidationError, match="text plus projector"):
        _performance_rollup(
            executable_gguf_bundle_size_bytes=MeasurementPairedBytes(bf16=121, q3=80)
        )
    with pytest.raises(ValidationError, match="server concurrency cells"):
        _performance_rollup(server_cells=tuple(reversed(_performance_rollup().server_cells)))


def test_performance_contract_rejects_statistics_and_percentile_tamper() -> None:
    with pytest.raises(ValidationError, match="statistics differ"):
        MeasurementPairedFiveBatchMetricSummary(
            trial_count_per_subject=5,
            bf16_samples=(1.0, 2.0, 3.0, 4.0, 5.0),
            q3_samples=(2.0, 3.0, 4.0, 5.0, 6.0),
            mean=_paired_positive(4.0, 4.0),
            median=_paired_positive(3.0, 4.0),
            sample_standard_deviation=_paired_nonnegative(
                statistics.stdev((1.0, 2.0, 3.0, 4.0, 5.0)),
                statistics.stdev((2.0, 3.0, 4.0, 5.0, 6.0)),
            ),
        )

    cell = _server_cell(1)
    with pytest.raises(ValidationError, match="request count differs"):
        MeasurementServerCellRollup(
            **{
                **cell.model_dump(),
                "measured_requests_per_subject": 4,
            }
        )
    with pytest.raises(ValidationError, match="percentiles are not ordered"):
        MeasurementServerCellRollup(
            **{
                **cell.model_dump(),
                "inter_token_latency_p50_seconds": _nonnegative_five_batch(
                    (0.3,) * 5,
                    (0.3,) * 5,
                ),
            }
        )
    with pytest.raises(ValidationError, match=r"exactly 1\.0 seconds"):
        _performance_rollup(requested_telemetry_sampling_interval_seconds=0.5)


def test_success_terminal_receipt_roundtrips_and_binds_rollups_and_records() -> None:
    receipt = _success_receipt()
    payload = canonical_measurement_json_bytes(receipt.model_dump(mode="json"))

    assert (
        parse_measurement_terminal_receipt(
            payload,
            run_id=RUN_ID,
            outcome="success",
        )
        == receipt
    )
    reference = build_measurement_terminal_receipt_reference(
        payload,
        evidence_root=EVIDENCE_ROOT,
        run_id=RUN_ID,
        outcome="success",
    )
    assert (
        validate_measurement_terminal_receipt_reference(
            payload,
            evidence_root=EVIDENCE_ROOT,
            expected=reference,
        )
        == reference
    )

    with pytest.raises(ValidationError, match="quality rollup hash"):
        _success_receipt(quality_rollup_sha256=_digest("tampered-quality-rollup"))
    with pytest.raises(ValidationError, match="supporting records are incomplete"):
        _success_receipt(supporting_records=tuple(reversed(_success_receipt().supporting_records)))


def test_success_terminal_receipt_rejects_cross_record_binding_tamper() -> None:
    receipt = _success_receipt()

    with pytest.raises(ValidationError, match="complete every checked stage"):
        _success_receipt(completed_stages=MEASUREMENT_PLANNED_STAGES[:-1])
    with pytest.raises(ValidationError, match="runtime manifest hash"):
        _success_receipt(runtime_manifest_sha256=_digest("wrong-runtime-manifest"))
    with pytest.raises(ValidationError, match="quality result differs"):
        _success_receipt(quality_retention_passed=False)
    other_run_records = tuple(
        build_measurement_supporting_record_reference(
            canonical_measurement_json_bytes({"kind": reference.kind}),
            run_id="another-measurement-run",
            kind=reference.kind,
        )
        for reference in receipt.supporting_records
    )
    with pytest.raises(ValidationError, match="belongs to another run"):
        _success_receipt(supporting_records=other_run_records)
