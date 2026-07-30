from __future__ import annotations

import hashlib
import json
import math
import statistics
from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError

from inkling_quant_lab.gguf.inkling_measurement_control import (
    MEASUREMENT_PLANNED_STAGES,
    MeasurementAttemptClaim,
    MeasurementControlPlaneProvenance,
    MeasurementDeploymentIdentity,
    MeasurementFailureTerminalReceipt,
    MeasurementLaunchConfirmationChallenge,
    MeasurementLaunchIntent,
    MeasurementLlamaBenchCaseIdentity,
    MeasurementLlamaBenchWorkloadIdentity,
    MeasurementPostSpawnAcceptance,
    MeasurementReviewedInputs,
    MeasurementServerWorkloadIdentity,
    build_measurement_attempt_claim,
    build_measurement_control_plane_provenance,
    build_measurement_launch_intent,
    build_measurement_post_spawn_acceptance,
    build_measurement_supporting_record_reference,
    build_measurement_terminal_receipt_reference,
    canonical_measurement_json_bytes,
    claim_measurement_attempt,
    measurement_app_name,
    measurement_attempt_claim_path,
    measurement_deployment_tag,
    measurement_launch_intent_path,
    measurement_llama_bench_dataset_bytes,
    measurement_post_spawn_acceptance_path,
    measurement_server_prompt_source_text,
    parse_measurement_terminal_receipt,
    validate_measurement_attempt_claim,
    validate_measurement_control_plane_provenance,
    validate_measurement_launch_intent,
    validate_measurement_post_spawn_acceptance,
    validate_measurement_terminal_receipt_reference,
)
from inkling_quant_lab.gguf.inkling_measurement_evidence import (
    MEASUREMENT_PLACEMENT_WORKLOAD_ORDER,
    MEASUREMENT_RAW_BLOB_KIND_ORDER,
    MeasurementComparisonCompactRecord,
    MeasurementCudaIdentitySummary,
    MeasurementEvidenceSubject,
    MeasurementExecutableArtifactIdentity,
    MeasurementPlacementSummary,
    MeasurementPlacementWorkload,
    MeasurementRawBlobKind,
    MeasurementRawBlobReference,
    MeasurementSubjectCompactRecord,
    build_measurement_performance_rollup,
    build_measurement_quality_rollup,
    build_measurement_raw_blob_reference,
    canonical_measurement_evidence_json_bytes,
    measurement_raw_blob_path,
    measurement_subject_performance_projection_sha256,
    measurement_subject_quality_projection_sha256,
    parse_measurement_comparison_compact_record,
    parse_measurement_subject_compact_record,
    validate_measurement_comparison_links,
    validate_measurement_raw_blob_reference,
)
from inkling_quant_lab.gguf.inkling_measurement_execution import (
    PINNED_LLAMA_CPP_BUILD_COMMIT,
    LlamaBenchCommandSpec,
    LlamaPerplexityCommandSpec,
    LlamaServerCommandSpec,
    bind_exact_cuda_topology,
    build_llama_bench_command,
    build_llama_perplexity_command,
    build_llama_server_command,
)
from inkling_quant_lab.gguf.inkling_measurement_raw_evidence import (
    CAPTURED_TOOL_LOG_DELIMITER,
    MEASUREMENT_QUALITY_SUITE_ORDER,
    MEASUREMENT_REMOTE_CORPUS_PATH,
    MeasurementAttemptBindings,
    MeasurementBackendAuditEvidence,
    MeasurementBackendAuditWorkload,
    MeasurementColdCacheConditioning,
    MeasurementColdServerLoad,
    MeasurementCudaPeerEdge,
    MeasurementCudaPeerTopology,
    MeasurementCudaRuntimeDeviceProbe,
    MeasurementCudaRuntimePreflight,
    MeasurementDiagnosticItem,
    MeasurementDiagnosticTimings,
    MeasurementDiagnosticTrial,
    MeasurementHardwareGpu,
    MeasurementHardwareIdentity,
    MeasurementLlamaBenchCase,
    MeasurementLlamaBenchTrials,
    MeasurementPerplexityTrial,
    MeasurementRawTrialsEvidence,
    MeasurementRepeatedLoadDurations,
    MeasurementResourceSampleSummary,
    MeasurementServerBatch,
    MeasurementServerCell,
    MeasurementServerLoadObservation,
    MeasurementServerLoadPairTrial,
    MeasurementServerRequest,
    MeasurementServerSingleWarmup,
    MeasurementServerTrials,
    MeasurementStagedArtifact,
    MeasurementSubjectPerformanceSummary,
    MeasurementSubjectQualitySummary,
    MeasurementSubjectStaging,
    canonical_measurement_raw_json_bytes,
    parse_backend_audit_evidence,
    parse_raw_trials_evidence,
    parse_token_nll_raw_evidence,
    recompute_subject_performance_summary,
    recompute_subject_quality_summary,
)
from inkling_quant_lab.gguf.inkling_smoke import (
    parse_artifact_load_evidence,
    parse_loader_offload_evidence,
)

RUN_ID = "inkling-measurement-records-test"
COMMIT_SHA = "a" * 40
TREE_SHA = "b" * 40
EVIDENCE_ROOT = "/measurement-evidence"
CUDA_LOGICAL_DEVICES: tuple[
    Literal["cuda:0"],
    Literal["cuda:1"],
    Literal["cuda:2"],
    Literal["cuda:3"],
    Literal["cuda:4"],
    Literal["cuda:5"],
    Literal["cuda:6"],
    Literal["cuda:7"],
] = (
    "cuda:0",
    "cuda:1",
    "cuda:2",
    "cuda:3",
    "cuda:4",
    "cuda:5",
    "cuda:6",
    "cuda:7",
)

REVIEWED_PATHS = {
    "measurement_config": "configs/experiments/inkling_q3_k_m_measurement_modal.yaml",
    "diagnostic_dataset": "configs/experiments/inkling_quality_diagnostic_v1.jsonl",
    "corpus_reference": "configs/experiments/inkling_wikitext2_raw_test_reference.json",
    "corpus_materializer": "scripts/materialize_inkling_measurement_corpus.py",
    "bf16_subject_reference": "configs/experiments/inkling_bf16_subject_reference.json",
    "q3_verified_export_reference": "configs/experiments/inkling_q3_k_m_verified_export.json",
    "source_adoption_reference": "configs/experiments/inkling_q3_k_m_source_adoption.json",
}
CONTROL_FILES = {path: f"{role}: reviewed\n".encode() for role, path in REVIEWED_PATHS.items()}


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _control_plane() -> MeasurementControlPlaneProvenance:
    return build_measurement_control_plane_provenance(
        reviewed_commit_sha=COMMIT_SHA,
        reviewed_tree_sha=TREE_SHA,
        files=CONTROL_FILES,
        required_paths=tuple(CONTROL_FILES),
    )


def _reviewed_inputs() -> MeasurementReviewedInputs:
    control_plane = _control_plane()
    by_path = {item.path: item for item in control_plane.files}
    return MeasurementReviewedInputs(
        control_plane=control_plane,
        measurement_config=by_path[REVIEWED_PATHS["measurement_config"]],
        resolved_config_sha256=_digest("resolved-config"),
        diagnostic_dataset=by_path[REVIEWED_PATHS["diagnostic_dataset"]],
        corpus_reference=by_path[REVIEWED_PATHS["corpus_reference"]],
        corpus_materializer=by_path[REVIEWED_PATHS["corpus_materializer"]],
        bf16_subject_reference=by_path[REVIEWED_PATHS["bf16_subject_reference"]],
        q3_verified_export_reference=by_path[REVIEWED_PATHS["q3_verified_export_reference"]],
        source_adoption_reference=by_path[REVIEWED_PATHS["source_adoption_reference"]],
    )


def _deployment() -> MeasurementDeploymentIdentity:
    control_hash = _reviewed_inputs().control_plane.control_plane_sha256
    return MeasurementDeploymentIdentity(
        deployed_at_utc="2026-07-30T12:10:00.000000Z",
        control_plane_sha256=control_hash,
        app_name=measurement_app_name(control_hash),
        deployment_version=1,
        deployment_tag=measurement_deployment_tag(control_hash),
        function_id="fu-MeasurementRecords",
        attempt_registry_id="di-MeasurementRecords",
        attempt_registry_created_at_utc="2026-07-30T12:00:00.000000Z",
        evidence_volume_id="vo-MeasurementRecords",
    )


def _launch_challenge() -> MeasurementLaunchConfirmationChallenge:
    return MeasurementLaunchConfirmationChallenge(
        created_at_utc="2026-07-30T12:20:00.000000Z",
        expires_at_utc="2026-07-30T12:30:00.000000Z",
        authorization_nonce=_digest("authorization-nonce"),
        run_id=RUN_ID,
        reviewed_inputs=_reviewed_inputs(),
        deployment=_deployment(),
    )


def _launch_intent() -> MeasurementLaunchIntent:
    challenge = _launch_challenge()
    return build_measurement_launch_intent(
        challenge,
        confirmation=challenge.confirmation_text(),
        authorized_at_utc="2026-07-30T12:21:00.000000Z",
    )


def _acceptance() -> MeasurementPostSpawnAcceptance:
    return build_measurement_post_spawn_acceptance(
        _launch_intent(),
        accepted_at_utc="2026-07-30T12:22:00.000000Z",
        call_id="fc-MeasurementRecords",
    )


def _attempt_claim() -> MeasurementAttemptClaim:
    return build_measurement_attempt_claim(
        _launch_intent(),
        _acceptance(),
        claimed_at_utc="2026-07-30T12:23:00.000000Z",
        input_id="in-MeasurementRecords:0-0",
        task_id="ta-MeasurementRecords",
    )


def _failure_values() -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "control_plane_sha256": _digest("control-plane"),
        "reviewed_config_file_sha256": _digest("reviewed-config"),
        "resolved_config_sha256": _digest("resolved-config"),
        "launch_intent_sha256": _digest("launch-intent"),
        "post_spawn_acceptance_sha256": _digest("post-spawn-acceptance"),
        "call_id": "fc-MeasurementRecords",
        "attempt_claim_sha256": _digest("attempt-claim"),
        "completed_at_utc": "2026-07-30T13:00:00.000000Z",
        "completed_stages": (),
        "failed_stage": "verify_references",
        "failed_subject": None,
        "error_code": "reference_validation_failed",
        "error_summary_sha256": _digest("safe-error-summary"),
        "supporting_records": (),
    }


def _failure_receipt(**overrides: Any) -> MeasurementFailureTerminalReceipt:
    values = _failure_values()
    values.update(overrides)
    return MeasurementFailureTerminalReceipt(**values)


def _bindings(subject: MeasurementEvidenceSubject) -> MeasurementAttemptBindings:
    return MeasurementAttemptBindings(
        run_id=RUN_ID,
        subject=subject,
        reviewed_config_file_sha256=_digest("reviewed-config"),
        resolved_config_sha256=_digest("resolved-config"),
        protocol_sha256=_digest("protocol"),
        workload_sha256=_digest("workload"),
        launch_intent_sha256=_digest("launch-intent"),
        post_spawn_acceptance_sha256=_digest("post-spawn-acceptance"),
        call_id="fc-MeasurementRecords",
        attempt_claim_sha256=_digest("attempt-claim"),
    )


def _backend_audit(
    subject: MeasurementEvidenceSubject,
    *,
    suffix: str = "",
) -> MeasurementBackendAuditEvidence:
    workloads = []
    for process_id, workload in enumerate(MEASUREMENT_PLACEMENT_WORKLOAD_ORDER, start=1):
        typed_workload = cast("MeasurementPlacementWorkload", workload)
        capture_mode: Literal["captured_stdout_stderr", "combined_server_log"]
        if workload == "server_quality_and_performance":
            log = f"complete server CUDA audit{suffix}\n"
            capture_mode = "combined_server_log"
            delimiter = None
        else:
            log = f"captured stderr{suffix}{CAPTURED_TOOL_LOG_DELIMITER}captured stdout\n"
            capture_mode = "captured_stdout_stderr"
            delimiter = CAPTURED_TOOL_LOG_DELIMITER
        payload = log.encode()
        workloads.append(
            MeasurementBackendAuditWorkload(
                workload=typed_workload,
                process_id=process_id,
                command=(f"/opt/llama.cpp/{workload}",),
                capture_mode=capture_mode,
                stdout_stderr_delimiter=delimiter,
                log=log,
                log_size_bytes=len(payload),
                log_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    exact_workloads = cast(
        "tuple["
        "MeasurementBackendAuditWorkload, "
        "MeasurementBackendAuditWorkload, "
        "MeasurementBackendAuditWorkload"
        "]",
        tuple(workloads),
    )
    return MeasurementBackendAuditEvidence(
        schema_version="inkling-measurement-backend-audit-v1",
        bindings=_bindings(subject),
        workloads=exact_workloads,
    )


def _raw_reference(
    subject: MeasurementEvidenceSubject,
    kind: MeasurementRawBlobKind,
) -> MeasurementRawBlobReference:
    if kind == "backend_audit":
        payload = canonical_measurement_raw_json_bytes(
            _backend_audit(subject).model_dump(mode="json")
        )
        return build_measurement_raw_blob_reference(
            payload,
            run_id=RUN_ID,
            subject=subject,
            kind=kind,
        )
    content_hash = _digest(f"{subject}-{kind}")
    raw_format: Literal["json", "jsonl"] = "json" if kind == "raw_trials" else "jsonl"
    record_count = 16_320 if kind == "token_nll" else 1
    return MeasurementRawBlobReference(
        run_id=RUN_ID,
        subject=subject,
        kind=kind,
        format=raw_format,
        relative_path=measurement_raw_blob_path(
            RUN_ID,
            subject=subject,
            kind=kind,
            content_sha256=content_hash,
        ),
        content_sha256=content_hash,
        size_bytes=128,
        record_count=record_count,
    )


def _artifact_inventory(
    subject: MeasurementEvidenceSubject,
) -> tuple[MeasurementExecutableArtifactIdentity, ...]:
    artifacts = []
    for ordinal in range(50):
        role: Literal["text_shard", "multimodal_projector"]
        if ordinal < 49:
            label = "BF16" if subject == "bf16" else "Q3_K_M"
            name = f"inkling-{label}-{ordinal + 1:05d}-of-00049.gguf"
            role = "text_shard"
        else:
            name = "mmproj-BF16.gguf"
            role = "multimodal_projector"
        artifacts.append(
            MeasurementExecutableArtifactIdentity(
                ordinal=ordinal,
                role=role,
                source_path=f"/source/{subject}/{name}",
                staged_path=f"/cache/inkling-measurement-subject/{subject}/{name}",
                sha256=_digest(f"{subject}-{name}"),
                size_bytes=ordinal + 1,
            )
        )
    return tuple(artifacts)


def _placement(
    workload: MeasurementPlacementWorkload,
    *,
    backend_audit_content_sha256: str,
) -> MeasurementPlacementSummary:
    identities = tuple(
        MeasurementCudaIdentitySummary(
            ordinal=ordinal,
            backend_name=f"CUDA{ordinal}",
            device_name=f"CUDA{ordinal}",
            compute_operations=1,
        )
        for ordinal in range(8)
    )
    exact_identities = cast(
        "tuple["
        "MeasurementCudaIdentitySummary, "
        "MeasurementCudaIdentitySummary, "
        "MeasurementCudaIdentitySummary, "
        "MeasurementCudaIdentitySummary, "
        "MeasurementCudaIdentitySummary, "
        "MeasurementCudaIdentitySummary, "
        "MeasurementCudaIdentitySummary, "
        "MeasurementCudaIdentitySummary"
        "]",
        identities,
    )
    return MeasurementPlacementSummary(
        workload=workload,
        backend_audit_content_sha256=backend_audit_content_sha256,
        audit_log_sha256=_digest(f"{workload}-audit-log"),
        command_sha256=_digest(f"{workload}-command"),
        placement_policy_sha256=_digest("placement-policy"),
        observed_graphs=1,
        compute_operations=8,
        cuda_operations=8,
        cpu_operations=0,
        accelerator_operations=0,
        other_operations=0,
        unassigned_operations=0,
        cuda_identities=exact_identities,
    )


def _subject_record(
    subject: MeasurementEvidenceSubject,
    **overrides: Any,
) -> MeasurementSubjectCompactRecord:
    raw_blobs = tuple(
        _raw_reference(subject, cast("MeasurementRawBlobKind", kind))
        for kind in MEASUREMENT_RAW_BLOB_KIND_ORDER
    )
    backend_hash = raw_blobs[-1].content_sha256
    values: dict[str, Any] = {
        "run_id": RUN_ID,
        "subject": subject,
        "control_plane_sha256": _digest("control-plane"),
        "reviewed_config_file_sha256": _digest("reviewed-config"),
        "resolved_config_sha256": _digest("resolved-config"),
        "launch_intent_sha256": _digest("launch-intent"),
        "post_spawn_acceptance_sha256": _digest("post-spawn-acceptance"),
        "call_id": "fc-MeasurementRecords",
        "attempt_claim_sha256": _digest("attempt-claim"),
        "runtime_manifest_sha256": _digest("runtime-manifest"),
        "hardware_identity_sha256": _digest("hardware-identity"),
        "model_id": "thinkingmachines/Inkling",
        "model_revision": "86b4d430ab871652a707666b89203a866888c5e5",
        "artifact_inventory": _artifact_inventory(subject),
        "protocol_sha256": _digest("protocol"),
        "workload_sha256": _digest("workload"),
        "raw_blobs": raw_blobs,
        "placement_summaries": tuple(
            _placement(
                cast("MeasurementPlacementWorkload", workload),
                backend_audit_content_sha256=backend_hash,
            )
            for workload in MEASUREMENT_PLACEMENT_WORKLOAD_ORDER
        ),
        "quality_projection_sha256": _digest(f"{subject}-quality"),
        "performance_projection_sha256": _digest(f"{subject}-performance"),
    }
    values.update(overrides)
    return MeasurementSubjectCompactRecord(**values)


def _comparison_record(
    bf16: MeasurementSubjectCompactRecord,
    q3: MeasurementSubjectCompactRecord,
    **overrides: Any,
) -> MeasurementComparisonCompactRecord:
    values: dict[str, Any] = {
        field_name: getattr(bf16, field_name)
        for field_name in (
            "run_id",
            "control_plane_sha256",
            "reviewed_config_file_sha256",
            "resolved_config_sha256",
            "launch_intent_sha256",
            "post_spawn_acceptance_sha256",
            "call_id",
            "attempt_claim_sha256",
            "runtime_manifest_sha256",
            "hardware_identity_sha256",
            "model_id",
            "model_revision",
            "protocol_sha256",
            "workload_sha256",
        )
    }
    values.update(
        {
            "subject_records": (
                build_measurement_supporting_record_reference(
                    bf16.canonical_bytes(),
                    run_id=RUN_ID,
                    kind="bf16_subject",
                ),
                build_measurement_supporting_record_reference(
                    q3.canonical_bytes(),
                    run_id=RUN_ID,
                    kind="q3_subject",
                ),
            ),
            "raw_blobs": (*bf16.raw_blobs, *q3.raw_blobs),
            "token_nll_pairing_sha256": _digest("token-nll-pairing"),
            "diagnostic_pairing_sha256": _digest("diagnostic-pairing"),
            "performance_pairing_sha256": _digest("performance-pairing"),
            "quality_rollup_sha256": _digest("quality-rollup"),
            "performance_rollup_sha256": _digest("performance-rollup"),
        }
    )
    values.update(overrides)
    return MeasurementComparisonCompactRecord(**values)


def _token_nll_payload() -> bytes:
    return b"".join(
        canonical_measurement_raw_json_bytes(
            {
                "chunk_index": chunk_index,
                "nll": 1.0,
                "token_id": 1000 + ordinal,
                "token_index": chunk_index * 512 + 257 + within_chunk,
            }
        )
        for ordinal in range(16_320)
        for chunk_index, within_chunk in (divmod(ordinal, 255),)
    )


def _raw_staging() -> MeasurementSubjectStaging:
    suffixes = [
        *(f"/bf16/inkling-BF16-{ordinal:05d}-of-00049.gguf" for ordinal in range(1, 50)),
        "/convert_text_bf16.success.json",
        "/mmproj/mmproj-BF16.gguf",
        "/chat_template.jinja",
        "/processor_config.json",
        "/special_tokens_map.json",
        "/tiktoken/tokenizer.model",
        "/tokenizer.json",
        "/tokenizer_config.json",
    ]
    staging_root = "/cache/inkling-measurement-subject/bf16"
    artifacts = tuple(
        MeasurementStagedArtifact(
            source_path=f"/source{suffix}",
            staged_path=f"{staging_root}{suffix}",
            sha256=_digest(f"staged-{suffix}"),
            size_bytes=ordinal + 1,
            source_passes=1,
        )
        for ordinal, suffix in enumerate(suffixes)
    )
    required_bytes = sum(item.size_bytes for item in artifacts)
    inventory = {
        "artifacts": [item.model_dump(mode="json") for item in artifacts],
        "required_bytes": required_bytes,
        "required_headroom_bytes": 137_438_953_472,
    }
    return MeasurementSubjectStaging(
        schema_version="inkling-measurement-subject-staging-v1",
        subject="bf16",
        source_volume_read_only=True,
        copy_and_hash_same_source_pass=True,
        source_passes_per_artifact=1,
        staging_root=staging_root,
        artifact_count=57,
        required_bytes=required_bytes,
        required_headroom_bytes=137_438_953_472,
        free_bytes_before_staging=required_bytes + 137_438_953_472,
        artifacts=artifacts,
        inventory_sha256=hashlib.sha256(
            canonical_measurement_raw_json_bytes(inventory)
        ).hexdigest(),
    )


def _gpu_uuid(ordinal: int) -> str:
    return f"GPU-{ordinal + 1:08x}-0000-0000-0000-{ordinal + 1:012x}"


def _raw_hardware_identity() -> MeasurementHardwareIdentity:
    gpus = tuple(
        MeasurementHardwareGpu(
            cuda_ordinal=ordinal,
            uuid=_gpu_uuid(ordinal),
            name="NVIDIA B300",
            memory_total_mib=274_113,
            driver_version="580.1",
            compute_capability="10.3",
        )
        for ordinal in range(8)
    )
    exact_gpus = cast(
        "tuple["
        "MeasurementHardwareGpu, "
        "MeasurementHardwareGpu, "
        "MeasurementHardwareGpu, "
        "MeasurementHardwareGpu, "
        "MeasurementHardwareGpu, "
        "MeasurementHardwareGpu, "
        "MeasurementHardwareGpu, "
        "MeasurementHardwareGpu"
        "]",
        gpus,
    )
    edges = tuple(
        MeasurementCudaPeerEdge(
            source_cuda_ordinal=source,
            source_uuid=_gpu_uuid(source),
            destination_cuda_ordinal=destination,
            destination_uuid=_gpu_uuid(destination),
            can_access_peer=True,
            performance_rank=0,
            access_supported=True,
            native_atomic_supported=True,
            cuda_array_access_supported=True,
            only_partial_native_atomic_supported=False,
        )
        for source in range(8)
        for destination in range(8)
        if source != destination
    )
    peer_topology = MeasurementCudaPeerTopology(
        schema_version="inkling-matched-cuda-peer-topology-v1",
        protocol="cuda-driver-p2p-attributes-v1",
        cuda_driver_api_version=13_100,
        gpu_uuids=tuple(_gpu_uuid(ordinal) for ordinal in range(8)),
        edges=edges,
    )
    probes = tuple(
        MeasurementCudaRuntimeDeviceProbe(
            cuda_ordinal=ordinal,
            logical_device=CUDA_LOGICAL_DEVICES[ordinal],
            allocation_size_bytes=16,
            memset_byte_value=ordinal + 1,
            copied_payload_hex=f"{ordinal + 1:02x}" * 16,
            cuda_set_device_result=0,
            cuda_malloc_result=0,
            cuda_memset_result=0,
            cuda_synchronize_after_memset_result=0,
            cuda_memcpy_device_to_host_result=0,
            cuda_synchronize_after_copy_result=0,
            payload_verified=True,
            cuda_free_result=0,
        )
        for ordinal in range(8)
    )
    exact_probes = cast(
        "tuple["
        "MeasurementCudaRuntimeDeviceProbe, "
        "MeasurementCudaRuntimeDeviceProbe, "
        "MeasurementCudaRuntimeDeviceProbe, "
        "MeasurementCudaRuntimeDeviceProbe, "
        "MeasurementCudaRuntimeDeviceProbe, "
        "MeasurementCudaRuntimeDeviceProbe, "
        "MeasurementCudaRuntimeDeviceProbe, "
        "MeasurementCudaRuntimeDeviceProbe"
        "]",
        probes,
    )
    runtime = MeasurementCudaRuntimePreflight(
        schema_version="inkling-measurement-cuda-runtime-preflight-v1",
        protocol="libcudart-set-malloc-memset-sync-d2h-sync-free-v1",
        libcudart_soname="libcudart.so.13",
        libcudart_path="/usr/local/cuda/lib64/libcudart.so.13",
        libcudart_sha256=_digest("libcudart"),
        libcudart_size_bytes=1024,
        execution_process="short-lived-subprocess",
        child_process_exit_code=0,
        cuda_get_device_count_result=0,
        observed_device_count=8,
        probes=exact_probes,
        all_devices_usable=True,
    )
    without_hash = {
        "schema_version": "inkling-measurement-hardware-identity-v1",
        "backend": "CUDA",
        "logical_devices": [f"cuda:{ordinal}" for ordinal in range(8)],
        "gpus": [item.model_dump(mode="json") for item in gpus],
        "peer_topology": peer_topology.model_dump(mode="json"),
        "cuda_driver_path": "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "cuda_runtime_preflight": runtime.model_dump(mode="json"),
        "precision": "model-native-subject-precision",
        "gpu_layers": "all",
        "cpu_moe_layers": 0,
        "cpu_fallback": False,
    }
    return MeasurementHardwareIdentity(
        schema_version="inkling-measurement-hardware-identity-v1",
        backend="CUDA",
        logical_devices=CUDA_LOGICAL_DEVICES,
        gpus=exact_gpus,
        peer_topology=peer_topology,
        cuda_driver_path="/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        cuda_runtime_preflight=runtime,
        precision="model-native-subject-precision",
        gpu_layers="all",
        cpu_moe_layers=0,
        cpu_fallback=False,
        identity_sha256=hashlib.sha256(
            canonical_measurement_raw_json_bytes(without_hash)
        ).hexdigest(),
    )


def _measurement_commands(
    staging: MeasurementSubjectStaging,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    topology = bind_exact_cuda_topology(
        tuple(f"CUDA{ordinal}" for ordinal in range(8)),
        (1,) * 8,
    )
    model_path = staging.artifacts[0].staged_path
    projector_path = next(
        item.staged_path
        for item in staging.artifacts
        if item.source_path.endswith("/mmproj/mmproj-BF16.gguf")
    )
    perplexity = build_llama_perplexity_command(
        LlamaPerplexityCommandSpec(
            model_path=model_path,
            corpus_path=MEASUREMENT_REMOTE_CORPUS_PATH,
            context_size=512,
            batch_size=512,
            ubatch_size=512,
            chunks=64,
            topology=topology,
        )
    )
    server = build_llama_server_command(
        LlamaServerCommandSpec(
            model_path=model_path,
            projector_path=projector_path,
            context_size=8192,
            batch_size=2048,
            ubatch_size=512,
            parallel_slots=4,
            port=18_080,
            topology=topology,
        )
    )
    bench = build_llama_bench_command(
        LlamaBenchCommandSpec(
            model_path=model_path,
            repetitions=5,
            batch_size=2048,
            ubatch_size=512,
            threads=16,
            topology=topology,
        )
    )
    return perplexity, server, bench


def _loader_log(staging: MeasurementSubjectStaging) -> str:
    first_shard = staging.artifacts[0].staged_path
    projector = next(
        item.staged_path
        for item in staging.artifacts
        if item.source_path.endswith("/mmproj/mmproj-BF16.gguf")
    )
    return "\n".join(
        (
            "ggml_cuda_init: found 8 CUDA devices (Total VRAM: 2140 GiB):",
            "load_tensors: offloading output layer to GPU",
            "load_tensors: offloaded 128/128 layers to GPU",
            f"llama_model_loader: loaded meta data with 57 key-value pairs from {first_shard}",
            "llama_model_loader: additional 48 GGUFs metadata loaded.",
            f"srv load_model: loaded multimodal model, '{projector}'",
            "IQL_SMOKE_TEXT_SHARDS_V1 expected=49 opened=49 contexts=49 tensors=1200",
            "IQL_SMOKE_TEXT_LOAD_V2 opened=49 accounted=49 tensors=1200 "
            "bytes=451035400288 size_done=451035400288 "
            "size_data=451035400288 mmap=1",
            "IQL_SMOKE_PROJECTOR_TENSORS_V1 modality=vision projector=inkling "
            "tensors=10 bytes=90000000",
            "IQL_SMOKE_PROJECTOR_TENSORS_V1 modality=audio projector=inkling "
            "tensors=12 bytes=93264288",
            "IQL_SMOKE_PROJECTOR_READY_V1 opened=1 vision=1 audio=1 "
            "vision_type=inkling audio_type=inkling n_embd=7168",
        )
    )


def _raw_diagnostics() -> tuple[MeasurementDiagnosticItem, ...]:
    items = []
    ordinal = 0
    for suite in MEASUREMENT_QUALITY_SUITE_ORDER:
        for item_index in range(1, 9):
            started = 30.0 + ordinal * 0.2
            finished = started + 0.1
            modality: Literal["text", "image", "audio"] = (
                "image" if suite == "vision" else "audio" if suite == "audio" else "text"
            )
            items.append(
                MeasurementDiagnosticItem(
                    item_id=f"{suite}_{item_index:02d}",
                    suite=suite,
                    modality=modality,
                    request_body_sha256=_digest(f"diagnostic-request-{ordinal}"),
                    prompt_sha256=_digest(f"diagnostic-prompt-{ordinal}"),
                    fixture_sha256=(
                        _digest(f"diagnostic-fixture-{ordinal}") if modality != "text" else None
                    ),
                    fixture_size_bytes=1 if modality != "text" else None,
                    seed=42,
                    temperature=0.0,
                    max_new_tokens=8,
                    scorer_kind="exact_text",
                    score=True,
                    trials=(
                        MeasurementDiagnosticTrial(
                            trial_index=1,
                            request_started_monotonic_seconds=started,
                            request_finished_monotonic_seconds=finished,
                            request_wall_seconds=finished - started,
                            token_ids=(ordinal + 1,),
                            output_sha256=_digest(f"diagnostic-output-{ordinal}"),
                            response_sha256=_digest(f"diagnostic-response-{ordinal}"),
                            normalization_succeeded=True,
                            normalized_sha256=_digest(f"diagnostic-normalized-{ordinal}"),
                            score=True,
                            timings=MeasurementDiagnosticTimings(
                                prompt_n=1,
                                predicted_n=1,
                                prompt_ms=1.0,
                                predicted_ms=1.0,
                                prompt_per_second=1000.0,
                                predicted_per_second=1000.0,
                            ),
                        ),
                    ),
                    prompt_text_recorded=False,
                    output_text_recorded=False,
                )
            )
            ordinal += 1
    return tuple(items)


def _server_response_values(
    *,
    request_body_sha256: str,
    label: str,
    started: float,
) -> dict[str, Any]:
    intervals = (0.01,) * 127
    first = started + 0.1
    last = first + sum(intervals)
    finished = last + 0.01
    return {
        "request_body_sha256": request_body_sha256,
        "token_ids": tuple(range(128)),
        "output_sha256": _digest(f"{label}-output"),
        "response_sha256": _digest(f"{label}-response"),
        "request_started_monotonic_seconds": started,
        "first_token_monotonic_seconds": first,
        "last_token_monotonic_seconds": last,
        "request_finished_monotonic_seconds": finished,
        "wall_seconds": finished - started,
        "ttft_seconds": first - started,
        "prompt_n": 512,
        "predicted_n": 128,
        "prompt_ms": 512.0,
        "predicted_ms": 128.0,
        "prompt_tokens_per_second": 1000.0,
        "decode_tokens_per_second": 1000.0,
        "inter_token_latency_p50_seconds": 0.01,
        "inter_token_latency_p95_seconds": 0.01,
        "inter_token_latency_p99_seconds": 0.01,
        "raw_inter_token_latency_seconds": intervals,
        "prompt_text_recorded": False,
        "output_text_recorded": False,
    }


def _server_batch(
    *,
    concurrency: Literal[1, 2, 4],
    batch_index: int,
    base: float,
    request_body_sha256: str,
) -> MeasurementServerBatch:
    requests = tuple(
        MeasurementServerRequest(
            request_index=request_index,
            **_server_response_values(
                request_body_sha256=request_body_sha256,
                label=f"c{concurrency}-b{batch_index}-r{request_index}",
                started=base + 0.01,
            ),
        )
        for request_index in range(1, concurrency + 1)
    )
    finished = max(item.request_finished_monotonic_seconds for item in requests) + 0.01
    decode_start = min(item.first_token_monotonic_seconds for item in requests)
    decode_finish = max(item.last_token_monotonic_seconds for item in requests)
    duration = decode_finish - decode_start
    return MeasurementServerBatch(
        batch_index=batch_index,
        concurrency=concurrency,
        batch_started_monotonic_seconds=base,
        batch_finished_monotonic_seconds=finished,
        batch_wall_seconds=finished - base,
        decode_boundary=("earliest_first_token_to_latest_last_token_127_intervals_per_request"),
        aggregate_decode_token_intervals=127 * concurrency,
        batch_duration_seconds=duration,
        aggregate_decode_tokens_per_second=127 * concurrency / duration,
        requests=requests,
    )


def _server_cell(
    *,
    concurrency: Literal[1, 2, 4],
    base: float,
    request_body_sha256: str,
) -> MeasurementServerCell:
    warmup = _server_batch(
        concurrency=concurrency,
        batch_index=0,
        base=base,
        request_body_sha256=request_body_sha256,
    )
    measured = tuple(
        _server_batch(
            concurrency=concurrency,
            batch_index=batch_index,
            base=base + 2.0 * batch_index,
            request_body_sha256=request_body_sha256,
        )
        for batch_index in range(1, 6)
    )
    exact_measured = cast(
        "tuple["
        "MeasurementServerBatch, "
        "MeasurementServerBatch, "
        "MeasurementServerBatch, "
        "MeasurementServerBatch, "
        "MeasurementServerBatch"
        "]",
        measured,
    )
    requests = tuple(request for batch in measured for request in batch.requests)
    intervals = tuple(
        interval for request in requests for interval in request.raw_inter_token_latency_seconds
    )
    resource_summary = MeasurementResourceSampleSummary(
        window_started_monotonic_seconds=measured[0].batch_started_monotonic_seconds,
        window_finished_monotonic_seconds=measured[-1].batch_finished_monotonic_seconds,
        sample_count=2,
        max_sampled_host_rss_bytes=1024,
        max_sampled_per_gpu_memory_bytes=(2048,) * 8,
        max_sampled_per_gpu_utilization_percent=(50.0,) * 8,
    )
    return MeasurementServerCell(
        concurrency=concurrency,
        single_request_warmups_completed=2 if concurrency == 1 else 0,
        concurrent_batch_warmup_completed=True,
        concurrent_batch_warmup=warmup,
        warmup_output_token_counts=(128, 128) if concurrency == 1 else (),
        concurrent_warmup_request_count=concurrency,
        measured_batches=exact_measured,
        measured_request_count=5 * concurrency,
        mean_ttft_seconds=statistics.fmean(item.ttft_seconds for item in requests),
        mean_prompt_tokens_per_second=statistics.fmean(
            item.prompt_tokens_per_second for item in requests
        ),
        mean_decode_tokens_per_second=statistics.fmean(
            item.decode_tokens_per_second for item in requests
        ),
        aggregate_decode_tokens_per_second_trials=cast(
            "tuple[float, float, float, float, float]",
            tuple(item.aggregate_decode_tokens_per_second for item in measured),
        ),
        mean_aggregate_decode_tokens_per_second=statistics.fmean(
            item.aggregate_decode_tokens_per_second for item in measured
        ),
        inter_token_latency_method=("r7_linear_interpolation_over_all_measured_request_intervals"),
        raw_inter_token_interval_count=len(intervals),
        inter_token_latency_p50_seconds=0.01,
        inter_token_latency_p95_seconds=0.01,
        inter_token_latency_p99_seconds=0.01,
        resource_sample_summary=resource_summary,
    )


def _load_observation(
    *,
    command: tuple[str, ...],
    process_id: int,
    started: float,
    ready: float,
    finished: float,
    log: str,
    loader_offload: Any,
    artifact_load: Any,
) -> MeasurementServerLoadObservation:
    log_bytes = log.encode()
    return MeasurementServerLoadObservation(
        command=command,
        process_id=process_id,
        process_started_monotonic_seconds=started,
        server_ready_monotonic_seconds=ready,
        process_finished_monotonic_seconds=finished,
        process_load_seconds=ready - started,
        log=log,
        log_size_bytes=len(log_bytes),
        log_sha256=hashlib.sha256(log_bytes).hexdigest(),
        loader_offload=loader_offload,
        artifact_load=artifact_load,
    )


def _raw_server(
    *,
    staging: MeasurementSubjectStaging,
    hardware: MeasurementHardwareIdentity,
    command: tuple[str, ...],
) -> MeasurementServerTrials:
    log = _loader_log(staging)
    first_shard = staging.artifacts[0].staged_path
    projector = next(
        item.staged_path
        for item in staging.artifacts
        if item.source_path.endswith("/mmproj/mmproj-BF16.gguf")
    )
    loader_offload = parse_loader_offload_evidence(log, expected_gpu_count=8)
    artifact_load = parse_artifact_load_evidence(
        log,
        expected_first_shard_path=first_shard,
        expected_projector_path=projector,
    )
    gguf_artifacts = tuple(item for item in staging.artifacts if item.staged_path.endswith(".gguf"))
    advised_bytes = sum(item.size_bytes for item in gguf_artifacts)
    pair_boundaries = (
        (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
        (17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0),
        (24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 100.0),
    )
    load_pairs = []
    for trial_index, (
        conditioned,
        cold_started,
        cold_ready,
        cold_finished,
        warm_started,
        warm_ready,
        warm_finished,
    ) in enumerate(pair_boundaries, start=1):
        conditioning = MeasurementColdCacheConditioning(
            schema_version="inkling-measurement-cold-cache-conditioning-v1",
            method=("file_level_posix_fadvise_posix_fadv_dontneed_on_all_staged_gguf_files"),
            advice="POSIX_FADV_DONTNEED",
            staged_paths=tuple(item.staged_path for item in gguf_artifacts),
            artifact_count=50,
            advised_bytes=advised_bytes,
            completed_monotonic_seconds=conditioned,
            all_advice_calls_succeeded=True,
            global_cache_flush_claimed=False,
        )
        cold = _load_observation(
            command=command,
            process_id=100 + 2 * trial_index,
            started=cold_started,
            ready=cold_ready,
            finished=cold_finished,
            log=log,
            loader_offload=loader_offload,
            artifact_load=artifact_load,
        )
        warm = _load_observation(
            command=command,
            process_id=101 + 2 * trial_index,
            started=warm_started,
            ready=warm_ready,
            finished=warm_finished,
            log=log,
            loader_offload=loader_offload,
            artifact_load=artifact_load,
        )
        load_pairs.append(
            MeasurementServerLoadPairTrial(
                trial_index=trial_index,
                cold_cache_conditioning=conditioning,
                cold=cold,
                warm=warm,
                warm_load_is_next_model_load_after_cold=True,
                explicit_cache_conditioning_or_eviction_requested_between_loads=False,
                cold_to_warm_restart_gap_seconds=warm_started - cold_finished,
            )
        )
    selected = load_pairs[-1]
    log_bytes = log.encode()
    cold_load = MeasurementColdServerLoad(
        schema_version="inkling-measurement-cold-server-load-v1",
        command=command,
        process_id=selected.cold.process_id,
        process_started_monotonic_seconds=selected.cold.process_started_monotonic_seconds,
        server_ready_monotonic_seconds=selected.cold.server_ready_monotonic_seconds,
        process_finished_monotonic_seconds=selected.cold.process_finished_monotonic_seconds,
        cold_server_process_load_seconds=selected.cold.process_load_seconds,
        hardware_identity_sha256=hardware.identity_sha256,
        readiness_only=True,
        generation_requests_executed=0,
        log=log,
        log_size_bytes=len(log_bytes),
        log_sha256=hashlib.sha256(log_bytes).hexdigest(),
        loader_offload=loader_offload,
        artifact_load=artifact_load,
    )
    prompt_token_ids = tuple(range(512))
    prompt_token_ids_sha256 = hashlib.sha256(
        canonical_measurement_raw_json_bytes(list(prompt_token_ids))
    ).hexdigest()
    request_body_sha256 = hashlib.sha256(
        canonical_measurement_raw_json_bytes(
            {
                "prompt": list(prompt_token_ids),
                "seed": 42,
                "temperature": 0.0,
                "n_predict": 128,
                "stream": True,
                "cache_prompt": False,
                "return_tokens": True,
                "ignore_eos": True,
            }
        )
    ).hexdigest()
    single_warmups = tuple(
        MeasurementServerSingleWarmup(
            warmup_index=cast("Literal[1, 2]", index),
            **_server_response_values(
                request_body_sha256=request_body_sha256,
                label=f"single-warmup-{index}",
                started=42.8 + 2.0 * index,
            ),
        )
        for index in (1, 2)
    )
    exact_single_warmups = cast(
        "tuple[MeasurementServerSingleWarmup, MeasurementServerSingleWarmup]",
        single_warmups,
    )
    cells = (
        _server_cell(
            concurrency=1,
            base=49.0,
            request_body_sha256=request_body_sha256,
        ),
        _server_cell(
            concurrency=2,
            base=62.0,
            request_body_sha256=request_body_sha256,
        ),
        _server_cell(
            concurrency=4,
            base=75.0,
            request_body_sha256=request_body_sha256,
        ),
    )
    durations = MeasurementRepeatedLoadDurations(
        trial_count=3,
        durations_seconds=(1.0, 1.0, 1.0),
        median_seconds=1.0,
        sample_standard_deviation_seconds=0.0,
    )
    return MeasurementServerTrials(
        schema_version="inkling-measurement-subject-server-v1",
        subject="bf16",
        load_pair_repetitions=3,
        load_pair_trial_scope=(
            "process_start_to_readiness_for_ordered_same_artifact_cold_then_warm_pairs"
        ),
        load_pair_trials=tuple(load_pairs),
        cold_server_load_trials=durations,
        warm_server_load_trials=durations,
        workload_load_pair_trial_index=3,
        cold_cache_conditioning=selected.cold_cache_conditioning,
        cold_load=cold_load,
        warm_load_is_next_model_load_after_cold=True,
        explicit_cache_conditioning_or_eviction_requested_between_server_loads=False,
        cold_to_warm_restart_gap_seconds=(
            selected.warm.process_started_monotonic_seconds
            - selected.cold.process_finished_monotonic_seconds
        ),
        command=command,
        process_id=selected.warm.process_id,
        process_started_monotonic_seconds=selected.warm.process_started_monotonic_seconds,
        server_ready_monotonic_seconds=selected.warm.server_ready_monotonic_seconds,
        process_finished_monotonic_seconds=selected.warm.process_finished_monotonic_seconds,
        warm_server_process_load_seconds=selected.warm.process_load_seconds,
        vocab_size=200_064,
        diagnostic_items_completed_before_performance=64,
        diagnostic_repetitions=1,
        single_request_warmups=exact_single_warmups,
        prompt_token_ids=prompt_token_ids,
        prompt_token_ids_sha256=prompt_token_ids_sha256,
        prompt_token_count=512,
        output_tokens=128,
        seed=42,
        temperature=0.0,
        streaming=True,
        cache_prompt=False,
        return_tokens=True,
        ignore_eos=True,
        request_body_sha256=request_body_sha256,
        concurrency=cells,
        log_sha256=selected.warm.log_sha256,
        log_size_bytes=selected.warm.log_size_bytes,
        prompt_text_recorded=False,
        output_text_recorded=False,
    )


def _raw_trials(token_nll_payload: bytes) -> MeasurementRawTrialsEvidence:
    staging = _raw_staging()
    hardware = _raw_hardware_identity()
    perplexity_command, server_command, bench_command = _measurement_commands(staging)
    bench_specs: tuple[
        tuple[Literal["pp512"], Literal[512], Literal[0]],
        tuple[Literal["pp2048"], Literal[2048], Literal[0]],
        tuple[Literal["tg128"], Literal[0], Literal[128]],
    ] = (
        ("pp512", 512, 0),
        ("pp2048", 2048, 0),
        ("tg128", 0, 128),
    )
    bench_cases = tuple(
        MeasurementLlamaBenchCase(
            case=case,
            build_commit=PINNED_LLAMA_CPP_BUILD_COMMIT,
            test_time_utc="2026-07-30T12:00:00Z",
            model_path=staging.artifacts[0].staged_path,
            model_type="Inkling",
            model_size_bytes=1024,
            model_parameter_count=1,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            sample_nanoseconds=(1_000_000_000,) * 5,
            sample_tokens_per_second=(float(prompt_tokens + generated_tokens),) * 5,
            average_nanoseconds=1_000_000_000,
            standard_deviation_nanoseconds=0,
            average_tokens_per_second=float(prompt_tokens + generated_tokens),
            standard_deviation_tokens_per_second=0.0,
            gpu_info=", ".join("NVIDIA B300" for _ in range(8)),
            backends="CUDA",
        )
        for case, prompt_tokens, generated_tokens in bench_specs
    )
    exact_bench_cases = cast(
        "tuple[MeasurementLlamaBenchCase, MeasurementLlamaBenchCase, MeasurementLlamaBenchCase]",
        bench_cases,
    )
    return MeasurementRawTrialsEvidence(
        schema_version="inkling-measurement-raw-trials-v1",
        bindings=_bindings("bf16"),
        hardware_identity=hardware,
        staging=staging,
        perplexity=MeasurementPerplexityTrial(
            process_id=1,
            process_started_monotonic_seconds=1.0,
            process_finished_monotonic_seconds=2.0,
            elapsed_seconds=1.0,
            command=perplexity_command,
            corpus_reference_sha256=(
                "5dfc8c426a1509c28d119857f437365c90a4bd57e229705d60e6fd3c1c65b95d"
            ),
            corpus_sha256=("173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"),
            corpus_size_bytes=1_290_590,
            perplexity=math.e,
            uncertainty=0.0,
            token_nll_sha256=hashlib.sha256(token_nll_payload).hexdigest(),
            token_nll_size_bytes=len(token_nll_payload),
            stdout_sha256=_digest("perplexity-stdout"),
            stdout_size_bytes=1,
            stderr_sha256=_digest("perplexity-stderr"),
            stderr_size_bytes=1,
        ),
        diagnostics=_raw_diagnostics(),
        llama_bench=MeasurementLlamaBenchTrials(
            process_id=200,
            process_started_monotonic_seconds=201.0,
            process_finished_monotonic_seconds=202.0,
            elapsed_seconds=1.0,
            command=bench_command,
            cases=exact_bench_cases,
            stdout_sha256=_digest("bench-stdout"),
            stdout_size_bytes=1,
            stderr_sha256=_digest("bench-stderr"),
            stderr_size_bytes=1,
            warmup_enabled=True,
            single_model_load=True,
        ),
        server=_raw_server(
            staging=staging,
            hardware=hardware,
            command=server_command,
        ),
        prompt_text_recorded=False,
        output_text_recorded=False,
    )


def _quality_as_q3(
    source: MeasurementSubjectQualitySummary,
) -> MeasurementSubjectQualitySummary:
    return MeasurementSubjectQualitySummary(
        subject="q3",
        token_nll=source.token_nll,
        printed_perplexity=source.printed_perplexity,
        printed_perplexity_uncertainty=source.printed_perplexity_uncertainty,
        printed_perplexity_absolute_tolerance=(source.printed_perplexity_absolute_tolerance),
        diagnostic_items=source.diagnostic_items,
        correct_items=source.correct_items,
        overall_accuracy=source.overall_accuracy,
        suites=source.suites,
    )


def _performance_as_q3(
    source: MeasurementSubjectPerformanceSummary,
) -> MeasurementSubjectPerformanceSummary:
    return MeasurementSubjectPerformanceSummary(
        subject="q3",
        text_checkpoint_size_bytes=source.text_checkpoint_size_bytes,
        multimodal_projector_size_bytes=source.multimodal_projector_size_bytes,
        executable_gguf_bundle_size_bytes=source.executable_gguf_bundle_size_bytes,
        load_pair_repetitions=source.load_pair_repetitions,
        workload_load_pair_trial_index=source.workload_load_pair_trial_index,
        cold_server_load_trials=source.cold_server_load_trials,
        warm_server_load_trials=source.warm_server_load_trials,
        cold_server_process_load_seconds=source.cold_server_process_load_seconds,
        warm_server_process_load_seconds=source.warm_server_process_load_seconds,
        bench_cases=source.bench_cases,
        server_cells=source.server_cells,
    )


def _bench_workload_identity() -> MeasurementLlamaBenchWorkloadIdentity:
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


def test_control_plane_and_remote_control_records_roundtrip_canonically() -> None:
    control_plane = _control_plane()
    assert (
        validate_measurement_control_plane_provenance(
            control_plane.canonical_bytes(),
            reviewed_commit_sha=COMMIT_SHA,
            reviewed_tree_sha=TREE_SHA,
            files=CONTROL_FILES,
            required_paths=tuple(CONTROL_FILES),
        )
        == control_plane
    )

    intent = _launch_intent()
    assert (
        validate_measurement_launch_intent(
            intent.canonical_bytes(),
            expected=intent,
            intent_sha256=intent.intent_sha256(),
            evidence_path=measurement_launch_intent_path(RUN_ID, intent.intent_sha256()),
        )
        == intent
    )

    acceptance = _acceptance()
    assert (
        validate_measurement_post_spawn_acceptance(
            acceptance.canonical_bytes(),
            expected=acceptance,
            acceptance_sha256=acceptance.acceptance_sha256(),
            evidence_path=measurement_post_spawn_acceptance_path(
                RUN_ID,
                intent.intent_sha256(),
            ),
        )
        == acceptance
    )

    claim = _attempt_claim()
    assert (
        validate_measurement_attempt_claim(
            claim.canonical_bytes(),
            expected=claim,
            claim_sha256=claim.claim_sha256(),
            evidence_path=measurement_attempt_claim_path(RUN_ID, claim.claim_sha256()),
        )
        == claim
    )


def test_control_records_reject_tamper_noncanonical_bytes_and_second_claim() -> None:
    control_plane = _control_plane()
    changed_files = dict(CONTROL_FILES)
    changed_files[REVIEWED_PATHS["measurement_config"]] += b"tampered: true\n"
    with pytest.raises(ValueError, match="differs from deployed bytes"):
        validate_measurement_control_plane_provenance(
            control_plane.canonical_bytes(),
            reviewed_commit_sha=COMMIT_SHA,
            reviewed_tree_sha=TREE_SHA,
            files=changed_files,
            required_paths=tuple(changed_files),
        )

    intent = _launch_intent()
    pretty = json.dumps(intent.model_dump(mode="json"), indent=2).encode()
    with pytest.raises(ValueError, match="not canonical"):
        validate_measurement_launch_intent(
            pretty,
            expected=intent,
            intent_sha256=intent.intent_sha256(),
            evidence_path=measurement_launch_intent_path(RUN_ID, intent.intent_sha256()),
        )

    class Registry:
        def __init__(self) -> None:
            self.values: dict[str, bytes] = {}

        def put(self, key: Any, value: Any, *, skip_if_exists: bool = False) -> bool:
            if skip_if_exists and key in self.values:
                return False
            self.values[key] = value
            return True

    registry = Registry()
    claim = _attempt_claim()
    assert claim_measurement_attempt(registry, claim) == claim.claim_sha256()
    with pytest.raises(RuntimeError, match="already been consumed"):
        claim_measurement_attempt(registry, claim)


def test_failure_terminal_receipt_roundtrip_and_content_addressed_reference() -> None:
    receipt = _failure_receipt()
    payload = canonical_measurement_json_bytes(receipt.model_dump(mode="json"))
    assert (
        parse_measurement_terminal_receipt(
            payload,
            run_id=RUN_ID,
            outcome="failure",
        )
        == receipt
    )

    reference = build_measurement_terminal_receipt_reference(
        payload,
        evidence_root=EVIDENCE_ROOT,
        run_id=RUN_ID,
        outcome="failure",
    )
    assert reference.absolute_path == f"{EVIDENCE_ROOT}/{reference.relative_path}"
    assert (
        validate_measurement_terminal_receipt_reference(
            payload,
            evidence_root=EVIDENCE_ROOT,
            expected=reference,
        )
        == reference
    )

    tampered = receipt.model_dump(mode="json")
    tampered["error_summary_sha256"] = _digest("different-safe-error-summary")
    with pytest.raises(ValueError, match="differs from exact receipt bytes"):
        validate_measurement_terminal_receipt_reference(
            canonical_measurement_json_bytes(tampered),
            evidence_root=EVIDENCE_ROOT,
            expected=reference,
        )


def test_failure_terminal_receipt_rejects_invalid_stage_transitions() -> None:
    with pytest.raises(ValidationError, match="exact checked prefix"):
        _failure_receipt(
            completed_stages=("stage_and_rehash_bf16",),
            failed_stage="measure_bf16_quality",
            failed_subject="bf16",
        )
    with pytest.raises(ValidationError, match="immediately follow"):
        _failure_receipt(failed_stage="stage_and_rehash_bf16", failed_subject="bf16")
    with pytest.raises(ValidationError, match="failed subject"):
        _failure_receipt(failed_subject="bf16")
    with pytest.raises(ValidationError, match="every measurement stage"):
        _failure_receipt(
            completed_stages=MEASUREMENT_PLANNED_STAGES,
            failed_stage="compare_and_publish",
        )

    payload = canonical_measurement_json_bytes(_failure_receipt().model_dump(mode="json"))
    with pytest.raises(ValueError, match="schema is invalid"):
        parse_measurement_terminal_receipt(payload, run_id=RUN_ID, outcome="success")


def test_backend_audit_raw_blob_roundtrip_and_retained_log_hash_tamper() -> None:
    audit = _backend_audit("bf16")
    payload = canonical_measurement_raw_json_bytes(audit.model_dump(mode="json"))
    assert parse_backend_audit_evidence(payload) == audit

    reference = build_measurement_raw_blob_reference(
        payload,
        run_id=RUN_ID,
        subject="bf16",
        kind="backend_audit",
    )
    assert validate_measurement_raw_blob_reference(payload, expected=reference) == reference

    changed = _backend_audit("bf16", suffix="-changed")
    changed_payload = canonical_measurement_raw_json_bytes(changed.model_dump(mode="json"))
    with pytest.raises(ValueError, match="differs from its exact bytes"):
        validate_measurement_raw_blob_reference(changed_payload, expected=reference)

    stale_hash = audit.model_dump(mode="json")
    stale_hash["workloads"][0]["log"] += "tampered"
    with pytest.raises(ValidationError, match=r"log size|log SHA-256"):
        parse_backend_audit_evidence(canonical_measurement_raw_json_bytes(stale_hash))


@pytest.mark.parametrize("subject", ("bf16", "q3"))
def test_subject_compact_record_roundtrip_is_canonical_and_scope_bound(
    subject: MeasurementEvidenceSubject,
) -> None:
    record = _subject_record(subject)
    payload = record.canonical_bytes()
    assert record.content_sha256() == hashlib.sha256(payload).hexdigest()
    assert (
        parse_measurement_subject_compact_record(
            payload,
            run_id=RUN_ID,
            subject=subject,
        )
        == record
    )

    with pytest.raises(ValueError, match="not canonical"):
        parse_measurement_subject_compact_record(
            json.dumps(record.model_dump(mode="json"), indent=2).encode(),
            run_id=RUN_ID,
            subject=subject,
        )
    other_subject: MeasurementEvidenceSubject = "q3" if subject == "bf16" else "bf16"
    with pytest.raises(ValueError, match="wrong run or subject"):
        parse_measurement_subject_compact_record(
            payload,
            run_id=RUN_ID,
            subject=other_subject,
        )


def test_comparison_roundtrip_links_exact_subject_bytes_and_rejects_tamper() -> None:
    bf16 = _subject_record("bf16")
    q3 = _subject_record("q3")
    comparison = _comparison_record(bf16, q3)
    payload = comparison.canonical_bytes()

    assert parse_measurement_comparison_compact_record(payload, run_id=RUN_ID) == comparison
    assert validate_measurement_comparison_links(comparison, bf16=bf16, q3=q3) == comparison

    tampered_q3 = _subject_record(
        "q3",
        quality_projection_sha256=_digest("tampered-q3-quality"),
    )
    with pytest.raises(ValueError, match="exact compact records"):
        validate_measurement_comparison_links(
            comparison,
            bf16=bf16,
            q3=tampered_q3,
        )

    with pytest.raises(ValidationError, match="incomplete or out of order"):
        _comparison_record(
            bf16,
            q3,
            raw_blobs=tuple(reversed(comparison.raw_blobs)),
        )


def test_compact_record_canonical_encoder_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        canonical_measurement_evidence_json_bytes({"latency_seconds": float("nan")})


def test_raw_trials_recompute_quality_performance_and_paired_rollups() -> None:
    token_nll_payload = _token_nll_payload()
    token_nll = parse_token_nll_raw_evidence(token_nll_payload)
    raw_trials = _raw_trials(token_nll_payload)
    raw_trials_payload = canonical_measurement_raw_json_bytes(raw_trials.model_dump(mode="json"))
    parsed_raw_trials = parse_raw_trials_evidence(raw_trials_payload)
    assert parsed_raw_trials == raw_trials

    raw_payloads: tuple[
        tuple[Literal["token_nll"], bytes],
        tuple[Literal["raw_trials"], bytes],
    ] = (
        ("token_nll", token_nll_payload),
        ("raw_trials", raw_trials_payload),
    )
    for kind, payload in raw_payloads:
        reference = build_measurement_raw_blob_reference(
            payload,
            run_id=RUN_ID,
            subject="bf16",
            kind=kind,
        )
        assert validate_measurement_raw_blob_reference(payload, expected=reference) == reference

    bf16_quality = recompute_subject_quality_summary(token_nll, parsed_raw_trials)
    q3_quality = _quality_as_q3(bf16_quality)
    quality_rollup = build_measurement_quality_rollup(
        bf16_quality,
        q3_quality,
        paired_inputs_validated=True,
    )
    assert quality_rollup.non_inferiority_passed is True
    assert quality_rollup.paired_token_positions == 16_320
    assert measurement_subject_quality_projection_sha256(
        bf16_quality
    ) != measurement_subject_quality_projection_sha256(q3_quality)

    bf16_performance = recompute_subject_performance_summary(parsed_raw_trials)
    q3_performance = _performance_as_q3(bf16_performance)
    performance_rollup = build_measurement_performance_rollup(
        bf16_performance,
        q3_performance,
        llama_bench_workload_identity=_bench_workload_identity(),
        server_workload_identity=_server_workload_identity(),
        equivalent_trials_validated=True,
    )
    assert performance_rollup.comparison_complete is True
    assert performance_rollup.speedup_claim_allowed is True
    assert measurement_subject_performance_projection_sha256(
        bf16_performance
    ) != measurement_subject_performance_projection_sha256(q3_performance)

    tampered = raw_trials.model_dump(mode="json")
    tampered["hardware_identity"]["identity_sha256"] = _digest("stale-hardware-identity")
    with pytest.raises(ValidationError, match="identity SHA-256 differs"):
        parse_raw_trials_evidence(canonical_measurement_raw_json_bytes(tampered))
