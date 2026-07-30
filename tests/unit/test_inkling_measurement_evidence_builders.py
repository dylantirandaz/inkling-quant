"""Host-side contracts for matched measurement evidence builders.

These tests validate immutable record construction and retained CUDA-audit
parsing. They do not load a model, execute inference, or contact Modal.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Sequence
from typing import Literal, cast

import pytest

from inkling_quant_lab.gguf.inkling_matched_execution import ExactCudaPlacementPolicy
from inkling_quant_lab.gguf.inkling_measurement_control import (
    MeasurementLlamaBenchCaseIdentity,
    MeasurementLlamaBenchWorkloadIdentity,
    MeasurementServerWorkloadIdentity,
    measurement_llama_bench_dataset_bytes,
    measurement_performance_rollup_sha256,
    measurement_quality_rollup_sha256,
    measurement_server_prompt_source_text,
)
from inkling_quant_lab.gguf.inkling_measurement_evidence import (
    build_measurement_performance_rollup,
    build_measurement_placement_summaries,
    build_measurement_quality_rollup,
    canonical_measurement_evidence_json_bytes,
    measurement_subject_performance_projection_sha256,
    measurement_subject_quality_projection_sha256,
)
from inkling_quant_lab.gguf.inkling_measurement_raw_evidence import (
    CAPTURED_TOOL_LOG_DELIMITER,
    MEASUREMENT_BENCH_CASE_ORDER,
    MEASUREMENT_PRINTED_PERPLEXITY_ABSOLUTE_TOLERANCE,
    MEASUREMENT_QUALITY_SUITE_ORDER,
    MEASUREMENT_RAW_WORKLOAD_ORDER,
    MEASUREMENT_SERVER_CONCURRENCY_ORDER,
    MeasurementAttemptBindings,
    MeasurementBackendAuditEvidence,
    MeasurementBackendAuditWorkload,
    MeasurementBenchCaseSummary,
    MeasurementDiagnosticSuiteSummary,
    MeasurementFiveBatchMetricSummary,
    MeasurementFiveBatchNonnegativeMetricSummary,
    MeasurementRepeatedLoadDurations,
    MeasurementResourceSampleSummary,
    MeasurementServerCellBatchMetrics,
    MeasurementServerCellSummary,
    MeasurementSubjectPerformanceSummary,
    MeasurementSubjectQualitySummary,
    MeasurementTokenNllSummary,
    canonical_measurement_raw_json_bytes,
    parse_backend_audit_evidence,
)

pytestmark = pytest.mark.unit

Subject = Literal["bf16", "q3"]
BenchCase = Literal["pp512", "pp2048", "tg128"]
RawWorkload = Literal["perplexity", "server_quality_and_performance", "llama_bench"]
CaptureMode = Literal["captured_stdout_stderr", "combined_server_log"]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _quality_summary(
    subject: Subject,
    *,
    mean_nll: float,
    suite_order: Sequence[str] = MEASUREMENT_QUALITY_SUITE_ORDER,
) -> MeasurementSubjectQualitySummary:
    suites = tuple(
        MeasurementDiagnosticSuiteSummary(
            suite=cast(
                "Literal['text', 'math', 'code', 'multilingual', 'instruction', "
                "'vision', 'audio', 'post_training']",
                suite,
            ),
            item_count=8,
            correct_items=7,
            accuracy=0.875,
        )
        for suite in suite_order
    )
    return MeasurementSubjectQualitySummary(
        subject=subject,
        token_nll=MeasurementTokenNllSummary(
            scored_tokens=16_320,
            mean_nll=mean_nll,
            computed_perplexity=math.exp(mean_nll),
        ),
        printed_perplexity=math.exp(mean_nll),
        printed_perplexity_uncertainty=0.01,
        printed_perplexity_absolute_tolerance=(MEASUREMENT_PRINTED_PERPLEXITY_ABSOLUTE_TOLERANCE),
        diagnostic_items=64,
        correct_items=56,
        overall_accuracy=0.875,
        suites=cast(
            "tuple[MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary, "
            "MeasurementDiagnosticSuiteSummary]",
            suites,
        ),
    )


def _five_positive(
    values: tuple[float, float, float, float, float],
) -> MeasurementFiveBatchMetricSummary:
    return MeasurementFiveBatchMetricSummary(
        trial_count=5,
        samples=values,
        mean=statistics.fmean(values),
        median=statistics.median(values),
        sample_standard_deviation=statistics.stdev(values),
    )


def _five_nonnegative(
    values: tuple[float, float, float, float, float],
) -> MeasurementFiveBatchNonnegativeMetricSummary:
    return MeasurementFiveBatchNonnegativeMetricSummary(
        trial_count=5,
        samples=values,
        mean=statistics.fmean(values),
        median=statistics.median(values),
        sample_standard_deviation=statistics.stdev(values),
    )


def _load_durations(
    values: tuple[float, ...],
) -> MeasurementRepeatedLoadDurations:
    return MeasurementRepeatedLoadDurations(
        trial_count=len(values),
        durations_seconds=values,
        median_seconds=statistics.median(values),
        sample_standard_deviation_seconds=statistics.stdev(values),
    )


def _server_cell(
    concurrency: Literal[1, 2, 4],
    *,
    q3: bool,
) -> MeasurementServerCellSummary:
    latency_scale = 0.8 if q3 else 1.0
    rate_scale = 1.25 if q3 else 1.0
    end_to_end = _five_positive(
        cast(
            "tuple[float, float, float, float, float]",
            tuple(latency_scale * value for value in (1.0, 1.1, 1.2, 1.3, 1.4)),
        )
    )
    ttft = _five_positive(
        cast(
            "tuple[float, float, float, float, float]",
            tuple(latency_scale * value for value in (0.10, 0.11, 0.12, 0.13, 0.14)),
        )
    )
    prompt_rate = _five_positive(
        cast(
            "tuple[float, float, float, float, float]",
            tuple(rate_scale * value for value in (100.0, 101.0, 102.0, 103.0, 104.0)),
        )
    )
    decode_rate = _five_positive(
        cast(
            "tuple[float, float, float, float, float]",
            tuple(rate_scale * value for value in (20.0, 21.0, 22.0, 23.0, 24.0)),
        )
    )
    aggregate_rate = _five_positive(
        cast(
            "tuple[float, float, float, float, float]",
            tuple(rate_scale * concurrency * value for value in (20.0, 21.0, 22.0, 23.0, 24.0)),
        )
    )
    # Zero is a valid percentile when retained token timestamps are equal.
    p50 = _five_nonnegative((0.0, 0.0, 0.0, 0.0, 0.0))
    p95 = _five_nonnegative((0.001, 0.0011, 0.0012, 0.0013, 0.0014))
    p99 = _five_nonnegative((0.002, 0.0021, 0.0022, 0.0023, 0.0024))
    batch_metrics = MeasurementServerCellBatchMetrics(
        mean_request_end_to_end_latency_seconds=end_to_end,
        mean_ttft_seconds=ttft,
        mean_prompt_tokens_per_second=prompt_rate,
        mean_decode_tokens_per_second=decode_rate,
        aggregate_decode_tokens_per_second=aggregate_rate,
        inter_token_latency_p50_seconds=p50,
        inter_token_latency_p95_seconds=p95,
        inter_token_latency_p99_seconds=p99,
    )
    return MeasurementServerCellSummary(
        concurrency=concurrency,
        measured_batches=5,
        measured_requests=5 * concurrency,
        mean_ttft_seconds=ttft.mean,
        mean_prompt_tokens_per_second=prompt_rate.mean,
        mean_decode_tokens_per_second=decode_rate.mean,
        mean_aggregate_decode_tokens_per_second=aggregate_rate.mean,
        batch_metrics=batch_metrics,
        inter_token_latency_p50_seconds=0.0,
        inter_token_latency_p95_seconds=0.0012,
        inter_token_latency_p99_seconds=0.0022,
        resource_sample_summary=MeasurementResourceSampleSummary(
            window_started_monotonic_seconds=10.0,
            window_finished_monotonic_seconds=20.0,
            sample_count=10,
            max_sampled_host_rss_bytes=4_000_000_000,
            max_sampled_per_gpu_memory_bytes=(
                20_000_000_000,
                20_000_000_001,
                20_000_000_002,
                20_000_000_003,
                20_000_000_004,
                20_000_000_005,
                20_000_000_006,
                20_000_000_007,
            ),
            max_sampled_per_gpu_utilization_percent=(
                80.0,
                81.0,
                82.0,
                83.0,
                84.0,
                85.0,
                86.0,
                87.0,
            ),
        ),
    )


def _performance_summary(
    subject: Subject,
    *,
    load_pair_repetitions: Literal[2, 3] = 3,
) -> MeasurementSubjectPerformanceSummary:
    q3 = subject == "q3"
    throughput_scale = 1.25 if q3 else 1.0
    cold_values = (10.0, 9.5, 9.0) if load_pair_repetitions == 3 else (10.0, 9.5)
    warm_values = (8.0, 7.5, 7.0) if load_pair_repetitions == 3 else (8.0, 7.5)
    workload_trial_index = load_pair_repetitions
    bench_cases = tuple(
        MeasurementBenchCaseSummary(
            case=cast("BenchCase", case),
            sample_count=5,
            average_tokens_per_second=throughput_scale * (100.0 + 10.0 * index),
            median_tokens_per_second=throughput_scale * (99.0 + 10.0 * index),
            standard_deviation_tokens_per_second=1.0 + index,
        )
        for index, case in enumerate(MEASUREMENT_BENCH_CASE_ORDER)
    )
    server_cells = tuple(
        _server_cell(
            cast("Literal[1, 2, 4]", concurrency),
            q3=q3,
        )
        for concurrency in MEASUREMENT_SERVER_CONCURRENCY_ORDER
    )
    text_size = 500 if q3 else 1_000
    return MeasurementSubjectPerformanceSummary(
        subject=subject,
        text_checkpoint_size_bytes=text_size,
        multimodal_projector_size_bytes=100,
        executable_gguf_bundle_size_bytes=text_size + 100,
        load_pair_repetitions=load_pair_repetitions,
        workload_load_pair_trial_index=workload_trial_index,
        cold_server_load_trials=_load_durations(cold_values),
        warm_server_load_trials=_load_durations(warm_values),
        cold_server_process_load_seconds=cold_values[workload_trial_index - 1],
        warm_server_process_load_seconds=warm_values[workload_trial_index - 1],
        bench_cases=cast(
            "tuple[MeasurementBenchCaseSummary, "
            "MeasurementBenchCaseSummary, "
            "MeasurementBenchCaseSummary]",
            bench_cases,
        ),
        server_cells=cast(
            "tuple[MeasurementServerCellSummary, "
            "MeasurementServerCellSummary, "
            "MeasurementServerCellSummary]",
            server_cells,
        ),
    )


def _llama_bench_workload_identity() -> MeasurementLlamaBenchWorkloadIdentity:
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
                prompt_tokens=2048,
                generation_tokens=0,
            ),
            MeasurementLlamaBenchCaseIdentity(
                sample_id="tg128",
                prompt_tokens=0,
                generation_tokens=128,
            ),
        ),
    )


def _server_workload_identity() -> MeasurementServerWorkloadIdentity:
    content = measurement_server_prompt_source_text().encode("utf-8")
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


def _placement_policy() -> ExactCudaPlacementPolicy:
    return ExactCudaPlacementPolicy(
        schema_version="iql-exact-cuda-placement-policy-v1",
        gpu_count=8,
        tensor_split=(1, 1, 1, 1, 1, 1, 1, 1),
        split_mode="layer",
        text_graph_policy="at_least_one_all_expected_cuda",
        vision_graph_policy="cuda0_only",
        audio_graph_policy="cuda0_only",
    )


def _identity_line(
    *,
    graph_uid: int,
    graph_owner: str,
    backend_index: int,
    compute: int,
) -> str:
    return (
        f"IQL_SMOKE_BACKEND_IDENTITY_V2 graph_uid={graph_uid} "
        f"graph_owner={graph_owner} backend_index={backend_index} "
        f"backend_name=CUDA{backend_index} device_name=CUDA{backend_index} "
        f"device_type=gpu compute={compute}"
    )


def _graph_block(
    *,
    graph_uid: int,
    graph_owner: str,
    backend_indices: Sequence[int],
    first_compute: int,
) -> tuple[str, ...]:
    identities = tuple(
        _identity_line(
            graph_uid=graph_uid,
            graph_owner=graph_owner,
            backend_index=backend_index,
            compute=first_compute + offset,
        )
        for offset, backend_index in enumerate(backend_indices)
    )
    compute = sum(first_compute + offset for offset, _index in enumerate(backend_indices))
    graph = (
        f"IQL_SMOKE_BACKEND_GRAPH_V2 graph_uid={graph_uid} "
        f"graph_owner={graph_owner} phase=post_assignment_pre_split "
        f"scope=non_view_compute compute={compute} gpu={compute} "
        "cpu=0 accel=0 other=0 unassigned=0"
    )
    return (*identities, graph)


def _exact_cuda_audit_log() -> str:
    return "\n".join(
        (
            *_graph_block(
                graph_uid=1,
                graph_owner="text",
                backend_indices=tuple(range(8)),
                first_compute=10,
            ),
            *_graph_block(
                graph_uid=2,
                graph_owner="vision",
                backend_indices=(0,),
                first_compute=20,
            ),
            *_graph_block(
                graph_uid=3,
                graph_owner="audio",
                backend_indices=(0,),
                first_compute=30,
            ),
        )
    )


def _bindings() -> MeasurementAttemptBindings:
    return MeasurementAttemptBindings(
        run_id="measurement-evidence-builders",
        subject="bf16",
        reviewed_config_file_sha256=_digest("reviewed-config"),
        resolved_config_sha256=_digest("resolved-config"),
        protocol_sha256=_digest("protocol"),
        workload_sha256=_digest("workload"),
        launch_intent_sha256=_digest("launch-intent"),
        post_spawn_acceptance_sha256=_digest("post-spawn-acceptance"),
        call_id="fc-MeasurementEvidenceBuilders",
        attempt_claim_sha256=_digest("attempt-claim"),
    )


def _backend_audit(*, cpu_text_graph: bool = False) -> MeasurementBackendAuditEvidence:
    workloads = []
    for process_id, workload in enumerate(MEASUREMENT_RAW_WORKLOAD_ORDER, start=1):
        audit_log = _exact_cuda_audit_log()
        if cpu_text_graph and workload == "perplexity":
            audit_log = audit_log.replace(
                "scope=non_view_compute compute=108 gpu=108 cpu=0",
                "scope=non_view_compute compute=108 gpu=107 cpu=1",
                1,
            )
        if workload == "server_quality_and_performance":
            log = audit_log
            capture_mode = "combined_server_log"
            delimiter = None
        else:
            log = f"{audit_log}{CAPTURED_TOOL_LOG_DELIMITER}captured stdout\n"
            capture_mode = "captured_stdout_stderr"
            delimiter = CAPTURED_TOOL_LOG_DELIMITER
        payload = log.encode("utf-8")
        workloads.append(
            MeasurementBackendAuditWorkload(
                workload=cast("RawWorkload", workload),
                process_id=process_id,
                command=(f"/opt/llama.cpp/build/bin/{workload}",),
                capture_mode=cast("CaptureMode", capture_mode),
                stdout_stderr_delimiter=delimiter,
                log=log,
                log_size_bytes=len(payload),
                log_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    return MeasurementBackendAuditEvidence(
        schema_version="inkling-measurement-backend-audit-v1",
        bindings=_bindings(),
        workloads=cast(
            "tuple[MeasurementBackendAuditWorkload, "
            "MeasurementBackendAuditWorkload, "
            "MeasurementBackendAuditWorkload]",
            tuple(workloads),
        ),
    )


def test_quality_rollup_builds_complete_matched_non_inferiority_evidence() -> None:
    bf16 = _quality_summary("bf16", mean_nll=1.0)
    q3 = _quality_summary("q3", mean_nll=1.05)

    rollup = build_measurement_quality_rollup(
        bf16,
        q3,
        paired_inputs_validated=True,
    )

    assert rollup.mean_nll_delta == pytest.approx(0.05)
    assert rollup.non_inferiority_passed is True
    assert tuple(item.suite for item in rollup.suites) == MEASUREMENT_QUALITY_SUITE_ORDER
    assert measurement_quality_rollup_sha256(rollup) == (measurement_quality_rollup_sha256(rollup))
    assert measurement_subject_quality_projection_sha256(bf16) != (
        measurement_subject_quality_projection_sha256(q3)
    )


def test_quality_rollup_rejects_unproved_pairing_and_suite_order_drift() -> None:
    bf16 = _quality_summary("bf16", mean_nll=1.0)
    q3 = _quality_summary("q3", mean_nll=1.05)

    with pytest.raises(ValueError, match="validated paired inputs"):
        build_measurement_quality_rollup(
            bf16,
            q3,
            paired_inputs_validated=cast("Literal[True]", False),
        )

    reordered_q3 = _quality_summary(
        "q3",
        mean_nll=1.05,
        suite_order=(
            "math",
            "text",
            *MEASUREMENT_QUALITY_SUITE_ORDER[2:],
        ),
    )
    with pytest.raises(ValueError, match="different suite order"):
        build_measurement_quality_rollup(
            bf16,
            reordered_q3,
            paired_inputs_validated=True,
        )


def test_performance_rollup_keeps_full_distributions_and_zero_latency_samples() -> None:
    bf16 = _performance_summary("bf16")
    q3 = _performance_summary("q3")

    rollup = build_measurement_performance_rollup(
        bf16,
        q3,
        llama_bench_workload_identity=_llama_bench_workload_identity(),
        server_workload_identity=_server_workload_identity(),
        equivalent_trials_validated=True,
    )

    assert rollup.speedup_claim_allowed is True
    assert tuple(item.case for item in rollup.bench_cases) == MEASUREMENT_BENCH_CASE_ORDER
    assert tuple(item.concurrency for item in rollup.server_cells) == (
        MEASUREMENT_SERVER_CONCURRENCY_ORDER
    )
    assert rollup.server_cells[0].inter_token_latency_p50_seconds.bf16_samples == (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    assert rollup.server_cells[0].inter_token_latency_p50_seconds.mean.bf16 == 0.0
    assert measurement_performance_rollup_sha256(rollup) == (
        measurement_performance_rollup_sha256(rollup)
    )
    assert measurement_subject_performance_projection_sha256(bf16) != (
        measurement_subject_performance_projection_sha256(q3)
    )


def test_performance_rollup_rejects_unproved_or_mismatched_load_trials() -> None:
    bf16 = _performance_summary("bf16")
    q3 = _performance_summary("q3")
    identities = (_llama_bench_workload_identity(), _server_workload_identity())

    with pytest.raises(ValueError, match="validated equivalent trials"):
        build_measurement_performance_rollup(
            bf16,
            q3,
            llama_bench_workload_identity=identities[0],
            server_workload_identity=identities[1],
            equivalent_trials_validated=cast("Literal[True]", False),
        )

    with pytest.raises(ValueError, match="different load-pair trials"):
        build_measurement_performance_rollup(
            bf16,
            _performance_summary("q3", load_pair_repetitions=2),
            llama_bench_workload_identity=identities[0],
            server_workload_identity=identities[1],
            equivalent_trials_validated=True,
        )


def test_placement_builder_replays_all_retained_logs_and_rejects_cpu_graphs() -> None:
    audit = _backend_audit()
    audit_bytes = canonical_measurement_raw_json_bytes(audit.model_dump(mode="json"))
    audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
    parsed_audit = parse_backend_audit_evidence(audit_bytes)

    summaries = build_measurement_placement_summaries(
        parsed_audit,
        backend_audit_content_sha256=audit_sha256,
        policy=_placement_policy(),
    )

    assert tuple(item.workload for item in summaries) == MEASUREMENT_RAW_WORKLOAD_ORDER
    assert all(item.observed_graphs == 3 for item in summaries)
    assert all(item.compute_operations == 158 for item in summaries)
    assert all(item.cuda_operations == 158 for item in summaries)
    assert all(item.cpu_operations == 0 for item in summaries)
    assert all(len(item.cuda_identities) == 8 for item in summaries)
    assert summaries[0].backend_audit_content_sha256 == audit_sha256
    assert (
        summaries[0].command_sha256
        == hashlib.sha256(
            canonical_measurement_evidence_json_bytes(list(audit.workloads[0].command))
        ).hexdigest()
    )

    with pytest.raises(ValueError, match="cpu"):
        build_measurement_placement_summaries(
            _backend_audit(cpu_text_graph=True),
            backend_audit_content_sha256=_digest("cpu-audit"),
            policy=_placement_policy(),
        )
