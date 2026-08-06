"""Run one reviewed Inkling measurement data path on Modal.

This is the paid GPU data plane.  It is deployed and called only by the
matching reviewed measurement or diagnostic manager.  The deployment mode
exposes either the matched BF16-versus-Q3 measurement or the isolated BF16
prompt-interface diagnostic.  No model computation has a CPU substitute.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, TypeGuard, cast

import modal
from pydantic import BaseModel

LOCAL_PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
LOCAL_SRC_ROOT: Final = LOCAL_PROJECT_ROOT / "src"
REMOTE_PROJECT_ROOT: Final = Path("/root/iql_project")
REMOTE_PROVENANCE_PATH: Final = Path("/root/iql-measurement-control-provenance.json")
REMOTE_DIAGNOSTIC_PROVENANCE_PATH: Final = Path(
    "/root/iql-bf16-interface-diagnostic-control-provenance.json"
)
LLAMA_CPP_ROOT: Final = Path("/opt/llama.cpp")
BUILD_BIN_ROOT: Final = LLAMA_CPP_ROOT / "build/bin"
EVIDENCE_ROOT: Final = Path("/evidence")
SUBJECT_STAGING_ROOT: Final = Path("/cache/inkling-measurement-subject")
DIAGNOSTIC_STAGING_ROOT: Final = Path("/cache/inkling-bf16-interface-diagnostic")
SUBJECT_STAGING_HEADROOM_BYTES: Final = 128 * 1024 * 1024 * 1024
BASE_PATCH_REMOTE: Final = REMOTE_PROJECT_ROOT / "patches/inkling-smoke-a015409.patch"
MEASUREMENT_PATCH_REMOTE: Final = REMOTE_PROJECT_ROOT / "patches/inkling-measurement-a015409.patch"

PINNED_LLAMA_CPP_COMMIT: Final = "a015409e6c27b84f60d688823d4c0126a11571fd"
PINNED_CUDA_IMAGE: Final = (
    "nvidia/cuda:13.1.2-devel-ubuntu24.04@"
    "sha256:952e42d23230610a2714c8484f38e9c934ed68e6f9c9c7fac62dcd5f98858a6e"
)
BUILD_TARGETS: Final = (
    "llama-cli",
    "llama-server",
    "llama-bench",
    "llama-perplexity",
)
EFFECTIVE_CMAKE_DEFINITIONS: Final = (
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
)
COMMAND_BINARIES: Final = {name: f"/opt/llama.cpp/build/bin/{name}" for name in BUILD_TARGETS}
SUITES: Final = (
    "text",
    "math",
    "code",
    "multilingual",
    "instruction",
    "vision",
    "audio",
    "post_training",
)
BENCH_CASES: Final = ("pp512", "pp2048", "tg128")
SERVER_CONCURRENCIES: Final = (1, 2, 4)
PLANNED_STAGES: Final = (
    "verify_references",
    "verify_cuda_preflight",
    "stage_and_rehash_bf16",
    "measure_bf16_quality",
    "measure_bf16_performance",
    "release_bf16",
    "stage_and_rehash_q3",
    "measure_q3_quality",
    "measure_q3_performance",
    "release_q3",
    "compare_and_publish",
)
FUNCTION_TIMEOUT_SECONDS: Final = 86_400
PUBLICATION_RESERVE_SECONDS: Final = 600
ACCEPTANCE_TIMEOUT_SECONDS: Final = 120.0
SERVER_READY_TIMEOUT_SECONDS: Final = 3_600.0
REQUEST_TIMEOUT_SECONDS: Final = 900.0
MAX_HTTP_BYTES: Final = 32 * 1024 * 1024
MAX_LOG_BYTES: Final = 128 * 1024 * 1024
DIAGNOSTIC_SERVER_PORT: Final = 19_183
DIAGNOSTIC_FORCED_LOGIT_BIAS: Final = 1_000_000_000.0
DIAGNOSTIC_CONTENT_TEXT_MARKER: Final = "<|content_text|>"
DIAGNOSTIC_END_MESSAGE_MARKER: Final = "<|end_message|>"
DIAGNOSTIC_END_SAMPLING_MARKER: Final = "<|content_model_end_sampling|>"
RUN_ID_RE: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
CUDA_RUNTIME_SONAME_RE: Final = re.compile(r"^libcudart\.so(?:\.[0-9]+)*$")
CUDA_RUNTIME_PREFLIGHT_ALLOCATION_BYTES: Final = 16
CUDA_MEMCPY_DEVICE_TO_HOST: Final = 2
CUDA_RUNTIME_PREFLIGHT_CHILD_MODE: Final = "__iql_cuda_runtime_preflight_child_v1__"
CUDA_RUNTIME_PREFLIGHT_CHILD_TIMEOUT_SECONDS: Final = 300
CUDA_RUNTIME_PREFLIGHT_CHILD_MAX_OUTPUT_BYTES: Final = 1024 * 1024

if str(LOCAL_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC_ROOT))

from inkling_quant_lab.gguf.inkling_bf16_interface_diagnostic import (  # noqa: E402
    DIAGNOSTIC_ATTEMPT_REGISTRY_NAME,
    DIAGNOSTIC_COMPARISON_TOKEN_ID,
    DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
    DIAGNOSTIC_EOS_TOKEN_ID,
    DIAGNOSTIC_FUNCTION_NAME,
    DIAGNOSTIC_PLANNED_STAGES,
    DIAGNOSTIC_STAGE,
    DiagnosticAttemptClaim,
    DiagnosticControlPlaneProvenance,
    DiagnosticEogEvidence,
    DiagnosticEogTokenProbe,
    DiagnosticFailureTerminalReceipt,
    DiagnosticLaunchIntent,
    DiagnosticPostSpawnAcceptance,
    DiagnosticPrivateRawEvidence,
    DiagnosticPrivateRawReference,
    DiagnosticPrivateTrial,
    DiagnosticServerLoadClaim,
    DiagnosticStageName,
    DiagnosticSuccessTerminalReceipt,
    DiagnosticTerminalReceiptReference,
    InklingBF16InterfaceDiagnosticBundle,
    build_diagnostic_attempt_claim,
    build_diagnostic_post_spawn_acceptance,
    build_diagnostic_private_raw_reference,
    build_diagnostic_rollup,
    build_diagnostic_server_command,
    build_diagnostic_server_load_claim,
    build_diagnostic_terminal_receipt_reference,
    canonical_diagnostic_json_bytes,
    claim_diagnostic_attempt,
    diagnostic_app_name,
    diagnostic_attempt_claim_path,
    diagnostic_launch_intent_path,
    diagnostic_post_spawn_acceptance_path,
    diagnostic_protocol_sha256,
    diagnostic_rollup_sha256,
    diagnostic_runtime_identity_sha256,
    diagnostic_server_load_claim_path,
    diagnostic_workload_sha256,
    load_bf16_interface_diagnostic_bundle,
    parse_diagnostic_private_raw_evidence,
    parse_diagnostic_terminal_receipt,
    strict_diagnostic_json_object,
    validate_diagnostic_attempt_claim,
    validate_diagnostic_control_plane_provenance,
    validate_diagnostic_launch_intent,
    validate_diagnostic_post_spawn_acceptance,
    validate_diagnostic_private_raw_reference,
    validate_diagnostic_private_trials,
    validate_diagnostic_server_load_claim,
    validate_diagnostic_terminal_receipt_reference,
)
from inkling_quant_lab.gguf.inkling_matched_execution import (  # noqa: E402
    ExactCudaPlacementPolicy,
    build_matched_cuda_placement_policy,
    build_matched_server_environment,
    enumerate_matched_cuda_peer_topology,
    order_matched_nvidia_smi_identity_by_cuda_uuid,
    parse_exact_cuda_backend_audit,
    parse_exact_text_cuda_backend_audit,
    parse_matched_nvidia_smi_identity_csv,
    parse_matched_nvidia_smi_monitor_csv,
)
from inkling_quant_lab.gguf.inkling_measurement import (  # noqa: E402
    CORPUS_MATERIALIZER_RELATIVE_PATH,
    CORPUS_REFERENCE_RELATIVE_PATH,
    MATERIALIZED_CORPUS_PATH,
    MEASUREMENT_MEDIA_MARKER,
    InklingMeasurementBundle,
    build_diagnostic_fixture_bytes,
    load_measurement_bundle,
    measurement_protocol_sha256,
    measurement_workload_sha256,
)
from inkling_quant_lab.gguf.inkling_measurement_control import (  # noqa: E402
    MEASUREMENT_ATTEMPT_REGISTRY_NAME,
    MEASUREMENT_FUNCTION_NAME,
    MEASUREMENT_PLANNED_STAGES,
    MEASUREMENT_RUNTIME_COMMANDS,
    MeasurementAppliedPatch,
    MeasurementControlPlaneProvenance,
    MeasurementExecutionResources,
    MeasurementFailureTerminalReceipt,
    MeasurementLaunchIntent,
    MeasurementPerformanceRollup,
    MeasurementPostSpawnAcceptance,
    MeasurementPrePatchExecutable,
    MeasurementQualityRollup,
    MeasurementRuntimeCommandClosure,
    MeasurementRuntimeDependency,
    MeasurementRuntimeIdentity,
    MeasurementRuntimeRegularFile,
    MeasurementRuntimeSymlink,
    MeasurementStage,
    MeasurementSuccessTerminalReceipt,
    MeasurementSupportingRecordReference,
    MeasurementTerminalReceiptReference,
    build_measurement_attempt_claim,
    build_measurement_post_spawn_acceptance,
    build_measurement_supporting_record_reference,
    build_measurement_terminal_receipt_reference,
    canonical_measurement_json_bytes,
    claim_measurement_attempt,
    measurement_app_name,
    measurement_attempt_claim_path,
    measurement_launch_intent_path,
    measurement_performance_rollup_sha256,
    measurement_post_spawn_acceptance_path,
    measurement_quality_rollup_sha256,
    measurement_runtime_manifest_sha256,
    measurement_server_prompt_source_text,
    parse_measurement_terminal_receipt,
    strict_measurement_json_object,
    validate_measurement_attempt_claim,
    validate_measurement_control_plane_provenance,
    validate_measurement_launch_intent,
    validate_measurement_post_spawn_acceptance,
    validate_measurement_supporting_record_reference,
    validate_measurement_terminal_receipt_reference,
)
from inkling_quant_lab.gguf.inkling_measurement_evidence import (  # noqa: E402
    MeasurementComparisonCompactRecord,
    MeasurementExecutableArtifactIdentity,
    MeasurementRawBlobReference,
    MeasurementSubjectCompactRecord,
    build_measurement_performance_rollup,
    build_measurement_placement_summaries,
    build_measurement_quality_rollup,
    build_measurement_raw_blob_reference,
    measurement_subject_performance_projection_sha256,
    measurement_subject_quality_projection_sha256,
    parse_measurement_comparison_compact_record,
    parse_measurement_subject_compact_record,
    validate_measurement_comparison_links,
    validate_measurement_raw_blob_reference,
)
from inkling_quant_lab.gguf.inkling_measurement_execution import (  # noqa: E402
    DiagnosticScoreEvidence,
    LlamaBenchCommandSpec,
    LlamaPerplexityCommandSpec,
    LlamaServerCommandSpec,
    bind_exact_cuda_topology,
    build_llama_bench_command,
    build_llama_perplexity_command,
    build_llama_server_command,
    evaluate_diagnostic_response,
    extract_llama_perplexity_machine_failure,
    parse_llama_bench_jsonl,
    parse_llama_perplexity_final,
    summarize_latency_ms,
)
from inkling_quant_lab.gguf.inkling_measurement_raw_evidence import (  # noqa: E402
    CAPTURED_TOOL_LOG_DELIMITER,
    MeasurementAttemptBindings,
    MeasurementBackendAuditEvidence,
    MeasurementCudaRuntimePreflight,
    MeasurementDiagnosticTimings,
    MeasurementHardwareIdentity,
    MeasurementLlamaBenchTrials,
    MeasurementPairingProjectionHashes,
    MeasurementPerplexityTrial,
    MeasurementRawTrialsEvidence,
    MeasurementResourceSampleSummary,
    MeasurementServerTrials,
    MeasurementSubjectPerformanceSummary,
    MeasurementSubjectQualitySummary,
    canonical_measurement_raw_json_bytes,
    parse_backend_audit_evidence,
    parse_raw_trials_evidence,
    parse_resource_telemetry_evidence,
    parse_token_nll_raw_evidence,
    recompute_pairing_projection_hashes,
    recompute_subject_performance_summary,
    recompute_subject_quality_summary,
    validate_measurement_diagnostic_evidence,
    validate_measurement_raw_evidence_links,
    validate_pairing_projection_hashes,
)
from inkling_quant_lab.gguf.inkling_smoke import (  # noqa: E402
    BackendCpuPlacementError,
    TextArtifactLoadEvidence,
    parse_artifact_load_evidence,
    parse_loader_offload_evidence,
    parse_text_artifact_load_evidence,
)

SERVER_AUDIT_ENVIRONMENT: Final = {
    # These names are the protocol exposed by the pinned base instrumentation
    # patch.  They enable placement evidence; this runner is not a smoke path.
    "IQL_SMOKE_BACKEND_AUDIT": "1",
    "LLAMA_MEDIA_MARKER": MEASUREMENT_MEDIA_MARKER,
}
DIAGNOSTIC_SERVER_AUDIT_ENVIRONMENT: Final = {
    "IQL_SMOKE_BACKEND_AUDIT": "1",
}


@dataclass(frozen=True)
class _FileHash:
    """Internal stable-file hash before it is assigned a receipt role."""

    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class InvocationBinding:
    intent: MeasurementLaunchIntent
    acceptance: MeasurementPostSpawnAcceptance
    claim_sha256: str
    call_id: str


@dataclass(frozen=True)
class DiagnosticInvocationBinding:
    intent: DiagnosticLaunchIntent
    acceptance: DiagnosticPostSpawnAcceptance
    claim: DiagnosticAttemptClaim
    claim_sha256: str
    call_id: str
    input_id: str
    execution_task_id: str


@dataclass(frozen=True)
class MeasurementEvidenceBindings:
    run_id: str
    subject: Literal["bf16", "q3"]
    reviewed_config_file_sha256: str
    resolved_config_sha256: str
    protocol_sha256: str
    workload_sha256: str
    launch_intent_sha256: str
    post_spawn_acceptance_sha256: str
    call_id: str
    attempt_claim_sha256: str

    def fields(self) -> dict[str, str]:
        return {
            "run_id": self.run_id,
            "subject": self.subject,
            "reviewed_config_file_sha256": self.reviewed_config_file_sha256,
            "resolved_config_sha256": self.resolved_config_sha256,
            "protocol_sha256": self.protocol_sha256,
            "workload_sha256": self.workload_sha256,
            "launch_intent_sha256": self.launch_intent_sha256,
            "post_spawn_acceptance_sha256": (self.post_spawn_acceptance_sha256),
            "call_id": self.call_id,
            "attempt_claim_sha256": self.attempt_claim_sha256,
        }


@dataclass(frozen=True)
class SubjectSpec:
    name: Literal["bf16", "q3"]
    model_path: str
    shard_paths: tuple[str, ...]
    projector_path: str
    artifacts: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True)
class DiagnosticSubjectSpec:
    model_path: str
    shard_paths: tuple[str, ...]
    artifacts: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True)
class DiagnosticSourceEvidence:
    source_config_sha256: str
    asset_manifest_sha256: str
    expected_server_chat_template: str


@dataclass(frozen=True)
class DiagnosticCompletionResult:
    content: str
    tokens: tuple[int, ...]
    tokens_evaluated: int
    tokens_predicted: int
    stop: bool
    stop_type: Literal["eos", "limit"]
    truncated: bool
    response_sha256: str
    duration_seconds: float
    timings: MeasurementDiagnosticTimings


@dataclass(frozen=True)
class DiagnosticServerResult:
    trials: tuple[DiagnosticPrivateTrial, ...]
    eog: DiagnosticEogEvidence
    text_artifact_load: TextArtifactLoadEvidence
    command: tuple[str, ...]
    command_sha256: str
    server_log_sha256: str
    server_log_size_bytes: int
    server_log_text: str
    server_process_id: int
    resource_sample_summary: MeasurementResourceSampleSummary
    started_at_utc: str
    completed_at_utc: str


@dataclass
class DiagnosticServerProcess:
    process: subprocess.Popen[bytes]
    command: tuple[str, ...]
    log_path: Path
    started_at_utc: str
    started_monotonic: float
    monitor: ResourceMonitor


@dataclass
class ServerProcess:
    process: subprocess.Popen[bytes]
    command: tuple[str, ...]
    log_path: Path
    started_monotonic: float
    ready_monotonic: float
    monitor: ResourceMonitor


@dataclass(frozen=True)
class PerplexityMeasurementResult:
    evidence: dict[str, Any]
    token_nll_payload: bytes
    stdout: str
    stderr: str


@dataclass(frozen=True)
class QualityMeasurementResult:
    evidence: dict[str, Any]
    perplexity: PerplexityMeasurementResult


@dataclass(frozen=True)
class ServerMeasurementResult:
    diagnostics: tuple[dict[str, Any], ...]
    evidence: dict[str, Any]
    log: str


@dataclass(frozen=True)
class BenchmarkMeasurementResult:
    evidence: dict[str, Any]
    stdout: str
    stderr: str


@dataclass(frozen=True)
class PerformanceMeasurementResult:
    evidence: dict[str, Any]
    benchmark: BenchmarkMeasurementResult


@dataclass(frozen=True)
class SubjectMeasurementResult:
    source_subject: SubjectSpec
    staged_subject: SubjectSpec
    staging: dict[str, Any]
    quality: QualityMeasurementResult
    performance: PerformanceMeasurementResult
    server: ServerMeasurementResult


@dataclass(frozen=True)
class PublishedSubjectEvidence:
    record: MeasurementSubjectCompactRecord
    payload: bytes
    reference: MeasurementSupportingRecordReference
    raw_payloads: tuple[tuple[str, bytes], ...]
    raw_references: tuple[
        MeasurementRawBlobReference,
        MeasurementRawBlobReference,
        MeasurementRawBlobReference,
        MeasurementRawBlobReference,
    ]
    quality: MeasurementSubjectQualitySummary
    performance: MeasurementSubjectPerformanceSummary
    pairing: MeasurementPairingProjectionHashes


class PublicationCollisionError(RuntimeError):
    """An immutable evidence path contains different or unsafe bytes."""


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _canonical_bytes(value: Mapping[str, Any] | BaseModel) -> bytes:
    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else dict(value)
    return canonical_measurement_json_bytes(payload)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_file_identity(
    path: Path,
    *,
    displayed_path: str | None = None,
    allow_empty: bool = False,
) -> _FileHash:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (before.st_size <= 0 and not allow_empty):
            raise RuntimeError("runtime identity requires a regular file of valid size")
        while chunk := os.read(descriptor, 16 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise RuntimeError("runtime file changed while it was hashed")
    return _FileHash(
        path=displayed_path or path.as_posix(),
        sha256=digest.hexdigest(),
        size_bytes=before.st_size,
    )


def _cuda_runtime_function(
    library: Any,
    name: str,
    argtypes: list[object],
) -> Any:
    try:
        function = getattr(library, name)
    except AttributeError as error:
        raise RuntimeError(f"libcudart lacks required function {name}") from error
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function


def _cuda_runtime_error_text(cuda_get_error_string: Any, result: int) -> str:
    raw = cuda_get_error_string(result)
    if raw is None:
        return "error text unavailable"
    try:
        return cast("bytes", raw).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return "non-UTF-8 CUDA error text"


def _require_cuda_runtime_success(
    function: Any,
    *arguments: object,
    label: str,
    cuda_get_error_string: Any,
) -> int:
    result = int(function(*arguments))
    if result != 0:
        message = _cuda_runtime_error_text(cuda_get_error_string, result)
        raise RuntimeError(
            f"CUDA Runtime preflight {label} failed with cudaError {result}: {message}"
        )
    return result


def _probe_cuda_runtime_device(
    *,
    cuda_ordinal: int,
    cuda_get_error_string: Any,
    cuda_set_device: Any,
    cuda_malloc: Any,
    cuda_memset: Any,
    cuda_device_synchronize: Any,
    cuda_memcpy: Any,
    cuda_free: Any,
) -> dict[str, Any]:
    allocation_size = CUDA_RUNTIME_PREFLIGHT_ALLOCATION_BYTES
    memset_byte = cuda_ordinal + 1
    expected_payload = bytes([memset_byte]) * allocation_size
    host_buffer = (ctypes.c_ubyte * allocation_size)()
    device_pointer = ctypes.c_void_p()
    allocated = False
    operation_error: BaseException | None = None
    free_error: BaseException | None = None

    try:
        set_device_result = _require_cuda_runtime_success(
            cuda_set_device,
            cuda_ordinal,
            label=f"cuda:{cuda_ordinal} cudaSetDevice",
            cuda_get_error_string=cuda_get_error_string,
        )
        malloc_result = _require_cuda_runtime_success(
            cuda_malloc,
            ctypes.byref(device_pointer),
            allocation_size,
            label=f"cuda:{cuda_ordinal} cudaMalloc",
            cuda_get_error_string=cuda_get_error_string,
        )
        if device_pointer.value in (None, 0):
            raise RuntimeError(
                f"CUDA Runtime preflight cuda:{cuda_ordinal} cudaMalloc returned a null pointer"
            )
        allocated = True
        memset_result = _require_cuda_runtime_success(
            cuda_memset,
            device_pointer,
            memset_byte,
            allocation_size,
            label=f"cuda:{cuda_ordinal} cudaMemset",
            cuda_get_error_string=cuda_get_error_string,
        )
        synchronize_after_memset_result = _require_cuda_runtime_success(
            cuda_device_synchronize,
            label=f"cuda:{cuda_ordinal} synchronization after cudaMemset",
            cuda_get_error_string=cuda_get_error_string,
        )
        memcpy_result = _require_cuda_runtime_success(
            cuda_memcpy,
            ctypes.cast(host_buffer, ctypes.c_void_p),
            device_pointer,
            allocation_size,
            CUDA_MEMCPY_DEVICE_TO_HOST,
            label=f"cuda:{cuda_ordinal} cudaMemcpy device to host",
            cuda_get_error_string=cuda_get_error_string,
        )
        synchronize_after_copy_result = _require_cuda_runtime_success(
            cuda_device_synchronize,
            label=f"cuda:{cuda_ordinal} synchronization after device-to-host copy",
            cuda_get_error_string=cuda_get_error_string,
        )
        copied_payload = bytes(host_buffer)
        if copied_payload != expected_payload:
            raise RuntimeError(
                f"CUDA Runtime preflight cuda:{cuda_ordinal} copied "
                f"{copied_payload.hex()}, expected {expected_payload.hex()}"
            )
    except BaseException as error:
        operation_error = error

    free_result: int | None = None
    if allocated:
        try:
            free_result = _require_cuda_runtime_success(
                cuda_free,
                device_pointer,
                label=f"cuda:{cuda_ordinal} cudaFree",
                cuda_get_error_string=cuda_get_error_string,
            )
        except BaseException as error:
            free_error = error

    if operation_error is not None:
        if free_error is not None:
            operation_error.add_note(f"CUDA allocation cleanup also failed: {free_error}")
        raise operation_error
    if free_error is not None:
        raise free_error
    if free_result is None:
        raise RuntimeError(
            f"CUDA Runtime preflight cuda:{cuda_ordinal} did not free its allocation"
        )

    return {
        "cuda_ordinal": cuda_ordinal,
        "logical_device": f"cuda:{cuda_ordinal}",
        "allocation_size_bytes": allocation_size,
        "memset_byte_value": memset_byte,
        "copied_payload_hex": copied_payload.hex(),
        "cuda_set_device_result": set_device_result,
        "cuda_malloc_result": malloc_result,
        "cuda_memset_result": memset_result,
        "cuda_synchronize_after_memset_result": synchronize_after_memset_result,
        "cuda_memcpy_device_to_host_result": memcpy_result,
        "cuda_synchronize_after_copy_result": synchronize_after_copy_result,
        "payload_verified": True,
        "cuda_free_result": free_result,
    }


def _run_cuda_runtime_preflight_in_child(
    dependency: MeasurementRuntimeDependency,
) -> dict[str, Any]:
    assert dependency.resolved_path is not None
    assert dependency.sha256 is not None
    assert dependency.size_bytes is not None
    observed = _stable_file_identity(Path(dependency.resolved_path))
    if observed.sha256 != dependency.sha256 or observed.size_bytes != dependency.size_bytes:
        raise RuntimeError("libcudart differs from the identity passed to the child process")
    try:
        library = ctypes.CDLL(dependency.resolved_path)
    except OSError as error:
        raise RuntimeError("the exact runtime libcudart could not be loaded") from error

    cuda_get_error_string = _cuda_runtime_function(
        library,
        "cudaGetErrorString",
        [ctypes.c_int],
    )
    cuda_get_error_string.restype = ctypes.c_char_p
    cuda_get_device_count = _cuda_runtime_function(
        library,
        "cudaGetDeviceCount",
        [ctypes.POINTER(ctypes.c_int)],
    )
    cuda_set_device = _cuda_runtime_function(library, "cudaSetDevice", [ctypes.c_int])
    cuda_malloc = _cuda_runtime_function(
        library,
        "cudaMalloc",
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
    )
    cuda_memset = _cuda_runtime_function(
        library,
        "cudaMemset",
        [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t],
    )
    cuda_device_synchronize = _cuda_runtime_function(
        library,
        "cudaDeviceSynchronize",
        [],
    )
    cuda_memcpy = _cuda_runtime_function(
        library,
        "cudaMemcpy",
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int],
    )
    cuda_free = _cuda_runtime_function(library, "cudaFree", [ctypes.c_void_p])

    device_count = ctypes.c_int()
    device_count_result = _require_cuda_runtime_success(
        cuda_get_device_count,
        ctypes.byref(device_count),
        label="cudaGetDeviceCount",
        cuda_get_error_string=cuda_get_error_string,
    )
    if device_count.value != 8:
        raise RuntimeError(
            "CUDA Runtime preflight requires exactly 8 devices, "
            f"but libcudart reported {device_count.value}"
        )

    probes = [
        _probe_cuda_runtime_device(
            cuda_ordinal=cuda_ordinal,
            cuda_get_error_string=cuda_get_error_string,
            cuda_set_device=cuda_set_device,
            cuda_malloc=cuda_malloc,
            cuda_memset=cuda_memset,
            cuda_device_synchronize=cuda_device_synchronize,
            cuda_memcpy=cuda_memcpy,
            cuda_free=cuda_free,
        )
        for cuda_ordinal in range(8)
    ]
    return {
        "schema_version": "inkling-measurement-cuda-runtime-preflight-v1",
        "protocol": "libcudart-set-malloc-memset-sync-d2h-sync-free-v1",
        "libcudart_soname": dependency.soname,
        "libcudart_path": dependency.resolved_path,
        "libcudart_sha256": dependency.sha256,
        "libcudart_size_bytes": dependency.size_bytes,
        "execution_process": "short-lived-subprocess",
        "child_process_exit_code": 0,
        "cuda_get_device_count_result": device_count_result,
        "observed_device_count": device_count.value,
        "probes": probes,
        "all_devices_usable": True,
    }


def _cuda_runtime_preflight_child(arguments: Sequence[str]) -> int:
    if len(arguments) != 4:
        raise RuntimeError("CUDA Runtime preflight child received invalid arguments")
    soname, resolved_path, sha256, size_text = arguments
    if re.fullmatch(r"[1-9][0-9]*", size_text) is None:
        raise RuntimeError("CUDA Runtime preflight child received an invalid library size")
    dependency = MeasurementRuntimeDependency(
        classification="cuda",
        soname=soname,
        resolved_path=resolved_path,
        sha256=sha256,
        size_bytes=int(size_text),
    )
    evidence = MeasurementCudaRuntimePreflight.model_validate_json(
        canonical_measurement_raw_json_bytes(_run_cuda_runtime_preflight_in_child(dependency)),
        strict=True,
    )
    sys.stdout.buffer.write(canonical_measurement_raw_json_bytes(evidence.model_dump(mode="json")))
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__" and sys.argv[1:2] == [CUDA_RUNTIME_PREFLIGHT_CHILD_MODE]:
    raise SystemExit(_cuda_runtime_preflight_child(sys.argv[2:]))


def _remaining_work_timeout(
    work_deadline: float,
    maximum_seconds: float,
    *,
    label: str,
) -> float:
    remaining = work_deadline - time.monotonic()
    if remaining <= 0.0:
        raise TimeoutError(f"{label} cannot start after the measurement work deadline")
    return min(maximum_seconds, remaining)


def _safe_relative_path(value: str) -> PurePosixPath:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError("evidence path is not canonical relative POSIX")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("evidence path is not canonical relative POSIX")
    if path.as_posix() != value:
        raise ValueError("evidence path is not canonical relative POSIX")
    return path


def _canonical_absolute_directory_parts(path: Path, *, label: str) -> PurePosixPath:
    text = path.as_posix()
    posix = PurePosixPath(text)
    if (
        not posix.is_absolute()
        or text == "/"
        or "\\" in text
        or "\x00" in text
        or text.startswith("//")
        or posix.as_posix() != text
    ):
        raise RuntimeError(f"{label} is not one canonical absolute POSIX directory")
    return posix


def _require_canonical_directory_components(
    path: Path,
    *,
    label: str,
) -> Path:
    """Reject a non-directory or symbolic-link component in one absolute path."""

    posix = _canonical_absolute_directory_parts(path, label=label)
    current = Path("/")
    for part in posix.parts[1:]:
        current /= part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise RuntimeError(f"{label} contains a symbolic-link component")
        if not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"{label} contains a non-directory component")
    if current != path or current.resolve(strict=True) != path:
        raise RuntimeError(f"{label} does not resolve to its exact absolute path")
    return current


def _resolved_modal_mount_root(path: Path, *, label: str) -> Path:
    """Resolve one platform-owned mount alias without trusting descendants."""

    _canonical_absolute_directory_parts(path, label=label)
    try:
        mount_info = path.lstat()
        resolved = path.resolve(strict=True)
        resolved_info = resolved.lstat()
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise RuntimeError(f"{label} is missing or unsafe") from error
    if not (
        stat.S_ISDIR(mount_info.st_mode) or stat.S_ISLNK(mount_info.st_mode)
    ) or not stat.S_ISDIR(resolved_info.st_mode):
        raise RuntimeError(f"{label} is missing or unsafe")
    return _require_canonical_directory_components(
        resolved,
        label=f"resolved {label}",
    )


def _evidence_path_binding(relative: str) -> tuple[Path, Path]:
    path = _safe_relative_path(relative)
    try:
        root = _resolved_modal_mount_root(
            EVIDENCE_ROOT,
            label="evidence root",
        )
    except RuntimeError as error:
        raise PublicationCollisionError(str(error)) from error
    candidate = root
    for index, part in enumerate(path.parts):
        candidate /= part
        if os.path.lexists(candidate):
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise PublicationCollisionError("evidence path contains a symbolic-link component")
            if index < len(path.parts) - 1 and not stat.S_ISDIR(info.st_mode):
                raise PublicationCollisionError("evidence path parent is not a directory")
    if not candidate.is_relative_to(root):
        raise PublicationCollisionError("evidence path escaped its mount root")
    return root, candidate


def _evidence_path(relative: str) -> Path:
    return _evidence_path_binding(relative)[1]


def _create_safe_evidence_parent(path: Path, *, root: Path) -> None:
    parent = path.parent
    if not parent.is_relative_to(root):
        raise PublicationCollisionError("evidence parent escaped its mount root")
    current = root
    for part in parent.relative_to(root).parts:
        current /= part
        with suppress(FileExistsError):
            current.mkdir(mode=0o700)
        try:
            info = current.lstat()
        except OSError as error:
            raise PublicationCollisionError("evidence path has an unreadable ancestor") from error
        if not stat.S_ISDIR(info.st_mode):
            raise PublicationCollisionError(
                "evidence path has a symbolic-link or non-directory ancestor"
            )


def _read_regular_bytes(path: Path, *, maximum_bytes: int | None = None) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError("expected one regular file")
        if maximum_bytes is not None and info.st_size > maximum_bytes:
            raise RuntimeError("regular file exceeds its byte limit")
        chunks: list[bytes] = []
        remaining = info.st_size
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("regular file ended before its declared size")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RuntimeError("regular file grew during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Publish through the Volume v1 same-directory rename contract."""

    if source.parent != destination.parent:
        raise RuntimeError("immutable evidence rename must remain in one directory")
    try:
        os.lstat(destination)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(destination)
    os.rename(source, destination)


def _write_once(relative: str, payload: bytes) -> None:
    root, path = _evidence_path_binding(relative)
    _create_safe_evidence_parent(path, root=root)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(16)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            _rename_noreplace(temporary, path)
        except FileExistsError:
            if _read_regular_bytes(path, maximum_bytes=len(payload)) != payload:
                raise PublicationCollisionError(
                    "immutable evidence exists with different bytes"
                ) from None
        if _read_regular_bytes(path, maximum_bytes=len(payload)) != payload:
            raise RuntimeError("immutable evidence failed immediate readback")
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        with suppress(OSError):
            temporary.unlink()


def _commit_and_verify(payloads: Mapping[str, bytes]) -> None:
    if not payloads:
        raise ValueError("publication requires at least one payload")
    ordered = tuple(sorted(payloads.items()))
    for relative, payload in ordered:
        _write_once(relative, payload)
    evidence_volume.commit()
    evidence_volume.reload()
    for relative, payload in ordered:
        if (
            _read_regular_bytes(
                _evidence_path(relative),
                maximum_bytes=len(payload),
            )
            != payload
        ):
            raise RuntimeError("committed evidence differs from exact published bytes")


def _supporting_reference(
    *,
    run_id: str,
    kind: Literal["bf16_subject", "q3_subject", "comparison"],
    payload: bytes,
) -> MeasurementSupportingRecordReference:
    return build_measurement_supporting_record_reference(
        payload,
        run_id=run_id,
        kind=kind,
    )


def _validate_supporting_reference(
    reference: MeasurementSupportingRecordReference,
    payload: bytes,
) -> None:
    validate_measurement_supporting_record_reference(payload, expected=reference)


def _load_local_measurement_deployment() -> tuple[InklingMeasurementBundle, str, Path]:
    bundle = load_measurement_bundle(LOCAL_PROJECT_ROOT)
    control_sha = os.environ.get("IQL_MEASUREMENT_CONTROL_PLANE_SHA256")
    provenance_text = os.environ.get("IQL_MEASUREMENT_CONTROL_PLANE_PROVENANCE_PATH")
    if (
        not isinstance(control_sha, str)
        or SHA256_RE.fullmatch(control_sha) is None
        or not provenance_text
    ):
        raise RuntimeError(
            "deploy this paid runner only through manage_inkling_measurement_modal.py"
        )
    provenance_path = Path(provenance_text)
    payload = provenance_path.read_bytes()
    strict_measurement_json_object(payload)
    provenance = MeasurementControlPlaneProvenance.model_validate_json(
        payload,
        strict=True,
    )
    if payload != provenance.canonical_bytes():
        raise RuntimeError("local measurement provenance is not canonical")
    files = {item.path: (LOCAL_PROJECT_ROOT / item.path).read_bytes() for item in provenance.files}
    validate_measurement_control_plane_provenance(
        payload,
        reviewed_commit_sha=provenance.reviewed_commit_sha,
        reviewed_tree_sha=provenance.reviewed_tree_sha,
        files=files,
        required_paths=tuple(item.path for item in provenance.files),
    )
    if provenance.control_plane_sha256 != control_sha:
        raise RuntimeError("local deployment control-plane identity drifted")
    return bundle, control_sha, provenance_path


def _load_local_diagnostic_deployment() -> tuple[
    InklingBF16InterfaceDiagnosticBundle,
    str,
    Path,
]:
    bundle = load_bf16_interface_diagnostic_bundle(LOCAL_PROJECT_ROOT)
    control_sha = os.environ.get("IQL_BF16_DIAGNOSTIC_CONTROL_PLANE_SHA256")
    provenance_text = os.environ.get("IQL_BF16_DIAGNOSTIC_CONTROL_PLANE_PROVENANCE_PATH")
    if (
        not isinstance(control_sha, str)
        or SHA256_RE.fullmatch(control_sha) is None
        or not provenance_text
    ):
        raise RuntimeError(
            "deploy this paid runner diagnostic only through "
            "manage_inkling_bf16_interface_diagnostic_modal.py"
        )
    provenance_path = Path(provenance_text)
    payload = provenance_path.read_bytes()
    strict_diagnostic_json_object(
        payload,
        maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
    )
    provenance = DiagnosticControlPlaneProvenance.model_validate_json(
        payload,
        strict=True,
    )
    if payload != provenance.canonical_bytes():
        raise RuntimeError("local diagnostic provenance is not canonical")
    files = {item.path: (LOCAL_PROJECT_ROOT / item.path).read_bytes() for item in provenance.files}
    observed = validate_diagnostic_control_plane_provenance(
        payload,
        reviewed_commit_sha=provenance.reviewed_commit_sha,
        reviewed_tree_sha=provenance.reviewed_tree_sha,
        files=files,
        required_paths=tuple(item.path for item in provenance.files),
    )
    if observed.control_plane_sha256 != control_sha:
        raise RuntimeError("local diagnostic control-plane identity drifted")
    return bundle, control_sha, provenance_path


_MEASUREMENT_CONTROL_SHA = os.environ.get("IQL_MEASUREMENT_CONTROL_PLANE_SHA256")
_DIAGNOSTIC_CONTROL_SHA = os.environ.get("IQL_BF16_DIAGNOSTIC_CONTROL_PLANE_SHA256")
if bool(_MEASUREMENT_CONTROL_SHA) == bool(_DIAGNOSTIC_CONTROL_SHA):
    raise RuntimeError("select exactly one paid runner deployment mode")
_DIAGNOSTIC_MODE: Final = bool(_DIAGNOSTIC_CONTROL_SHA)

_LOCAL_BUNDLE: InklingMeasurementBundle | None
_LOCAL_DIAGNOSTIC_BUNDLE: InklingBF16InterfaceDiagnosticBundle | None
_CONTROL_SHA256: str
_LOCAL_PROVENANCE: Path

if modal.is_local():
    if _DIAGNOSTIC_MODE:
        (
            _LOCAL_DIAGNOSTIC_BUNDLE,
            _CONTROL_SHA256,
            _LOCAL_PROVENANCE,
        ) = _load_local_diagnostic_deployment()
        _LOCAL_BUNDLE = None
    else:
        _LOCAL_BUNDLE, _CONTROL_SHA256, _LOCAL_PROVENANCE = _load_local_measurement_deployment()
        _LOCAL_DIAGNOSTIC_BUNDLE = None
else:
    if _DIAGNOSTIC_MODE:
        _LOCAL_DIAGNOSTIC_BUNDLE = load_bf16_interface_diagnostic_bundle(REMOTE_PROJECT_ROOT)
        _LOCAL_BUNDLE = None
        _CONTROL_SHA256 = os.environ["IQL_BF16_DIAGNOSTIC_CONTROL_PLANE_SHA256"]
        _LOCAL_PROVENANCE = REMOTE_DIAGNOSTIC_PROVENANCE_PATH
    else:
        _LOCAL_BUNDLE = load_measurement_bundle(REMOTE_PROJECT_ROOT)
        _LOCAL_DIAGNOSTIC_BUNDLE = None
        _CONTROL_SHA256 = os.environ["IQL_MEASUREMENT_CONTROL_PLANE_SHA256"]
        _LOCAL_PROVENANCE = REMOTE_PROVENANCE_PATH

app = modal.App(
    diagnostic_app_name(_CONTROL_SHA256)
    if _DIAGNOSTIC_MODE
    else measurement_app_name(_CONTROL_SHA256)
)

_evidence_create_if_missing: bool
if _LOCAL_DIAGNOSTIC_BUNDLE is not None:
    _storage = _LOCAL_DIAGNOSTIC_BUNDLE.config.storage
    _bf16_volume_name = _storage.bf16_volume
    _bf16_volume_version = _storage.bf16_volume_version
    _bf16_run_subpath = _storage.bf16_run_subpath
    _source_volume_name = _storage.source_volume
    _source_volume_version = _storage.source_volume_version
    _source_run_subpath = _storage.source_run_subpath
    _evidence_volume_name = _storage.evidence_volume
    _evidence_volume_version = _storage.evidence_volume_version
    _evidence_create_if_missing = _storage.evidence_create_if_missing
    final_volume: modal.Volume | None = None
elif _LOCAL_BUNDLE is not None:
    _matched_storage = _LOCAL_BUNDLE.matched.config.storage
    _bf16_volume_name = _matched_storage.bf16_volume
    _bf16_volume_version = _matched_storage.bf16_volume_version
    _bf16_run_subpath = _matched_storage.bf16_run_subpath
    _source_volume_name = _matched_storage.source_volume
    _source_volume_version = _matched_storage.source_volume_version
    _source_run_subpath = _matched_storage.source_run_subpath
    _evidence_volume_name = _LOCAL_BUNDLE.config.storage.evidence_volume
    _evidence_volume_version = _LOCAL_BUNDLE.config.storage.evidence_volume_version
    _evidence_create_if_missing = False
    final_volume = modal.Volume.from_name(
        _matched_storage.final_volume,
        environment_name="inkling-quant",
        create_if_missing=False,
        version=_matched_storage.final_volume_version,
    ).with_mount_options(
        sub_path=_matched_storage.final_run_subpath,
        read_only=True,
    )
else:
    raise RuntimeError("paid runner deployment lacks its validated bundle")

baseline_volume = modal.Volume.from_name(
    _bf16_volume_name,
    environment_name="inkling-quant",
    create_if_missing=False,
    version=_bf16_volume_version,
).with_mount_options(
    sub_path=_bf16_run_subpath,
    read_only=True,
)
source_volume = modal.Volume.from_name(
    _source_volume_name,
    environment_name="inkling-quant",
    create_if_missing=False,
    version=_source_volume_version,
).with_mount_options(
    sub_path=_source_run_subpath,
    read_only=True,
)
evidence_volume = modal.Volume.from_name(
    _evidence_volume_name,
    environment_name="inkling-quant",
    create_if_missing=_evidence_create_if_missing,
    version=_evidence_volume_version,
)

_FUNCTION_VOLUMES: dict[
    str | PurePosixPath,
    modal.Volume | modal.CloudBucketMount,
] = {
    "/baseline": baseline_volume,
    "/source": source_volume,
    "/evidence": evidence_volume,
}
if final_volume is not None:
    _FUNCTION_VOLUMES["/final"] = final_volume

measurement_image = (
    modal.Image.from_registry(PINNED_CUDA_IMAGE, add_python="3.12")
    .apt_install(
        "build-essential",
        "ca-certificates",
        "cmake",
        "git",
        "ninja-build",
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "src/inkling_quant_lab"),
        str(REMOTE_PROJECT_ROOT / "src/inkling_quant_lab"),
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc", "**/*.pyo"],
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "configs/experiments"),
        str(REMOTE_PROJECT_ROOT / "configs/experiments"),
        copy=True,
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "scripts"),
        str(REMOTE_PROJECT_ROOT / "scripts"),
        copy=True,
        ignore=["**/__pycache__/**", "**/*.pyc", "**/*.pyo"],
    )
    .add_local_dir(
        str(LOCAL_PROJECT_ROOT / "patches"),
        str(REMOTE_PROJECT_ROOT / "patches"),
        copy=True,
    )
    .add_local_file(
        str(_LOCAL_PROVENANCE),
        str(REMOTE_DIAGNOSTIC_PROVENANCE_PATH if _DIAGNOSTIC_MODE else REMOTE_PROVENANCE_PATH),
        copy=True,
    )
    .run_commands(
        *(
            ()
            if _DIAGNOSTIC_MODE
            else (
                (
                    f"python {REMOTE_PROJECT_ROOT / CORPUS_MATERIALIZER_RELATIVE_PATH} "
                    f"--reference {REMOTE_PROJECT_ROOT / CORPUS_REFERENCE_RELATIVE_PATH} "
                    f"--output {MATERIALIZED_CORPUS_PATH}"
                ),
            )
        ),
        f"git init {LLAMA_CPP_ROOT}",
        (
            f"git -C {LLAMA_CPP_ROOT} remote add origin "
            "https://github.com/danielhanchen/llama.cpp.git"
        ),
        f"git -C {LLAMA_CPP_ROOT} fetch --depth 1 origin {PINNED_LLAMA_CPP_COMMIT}",
        f"git -C {LLAMA_CPP_ROOT} checkout --detach FETCH_HEAD",
        f"git -C {LLAMA_CPP_ROOT} apply --check {BASE_PATCH_REMOTE}",
        f"git -C {LLAMA_CPP_ROOT} apply {BASE_PATCH_REMOTE}",
        "python -m pip install --no-cache-dir pydantic==2.13.4 PyYAML==6.0.3",
        "mkdir -p /opt/iql-cuda-driver-link",
        ("ln -s /usr/local/cuda/lib64/stubs/libcuda.so /opt/iql-cuda-driver-link/libcuda.so.1"),
        (
            f"cmake -S {LLAMA_CPP_ROOT} -B {LLAMA_CPP_ROOT}/build -G Ninja "
            "-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON "
            "-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF "
            "-DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF "
            "-DCMAKE_CUDA_ARCHITECTURES=103 "
            "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/iql-cuda-driver-link "
            "-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=FALSE"
        ),
        (f"cmake --build {LLAMA_CPP_ROOT}/build --parallel 16 --target " + " ".join(BUILD_TARGETS)),
        (
            'python -c "import hashlib,pathlib; '
            "p=pathlib.Path('/opt/llama.cpp/build/bin/llama-cli'); "
            "b=p.read_bytes(); "
            "assert len(b)==1246680 and "
            "hashlib.sha256(b).hexdigest()=="
            "'098d8b9c6e57f25b846c5b5b43ded5bb1194cbb3d1ce985f17bbd09c87a82dbc'\""
        ),
        (
            'python -c "import hashlib,pathlib; '
            "p=pathlib.Path('/opt/llama.cpp/build/bin/llama-server'); "
            "b=p.read_bytes(); "
            "assert len(b)==17920 and "
            "hashlib.sha256(b).hexdigest()=="
            "'e960cfe4dcb2f7e541fc0b15bf97a4c1f6feb5fc304267796ef2bdd004cd1b93'\""
        ),
        (
            'python -c "import hashlib,pathlib; '
            "p=pathlib.Path('/opt/llama.cpp/build/bin/llama-bench'); "
            "b=p.read_bytes(); "
            "assert len(b)==17920 and "
            "hashlib.sha256(b).hexdigest()=="
            "'e0844ac337c419ebd8b6cee4902ba13e210a067d6fe47cb652429c71ae97382b'\""
        ),
        (
            'python -c "import hashlib,pathlib; '
            "p=pathlib.Path('/opt/llama.cpp/build/bin/llama-perplexity'); "
            "b=p.read_bytes(); "
            "assert len(b)==15968 and "
            "hashlib.sha256(b).hexdigest()=="
            "'d04051888a157ee50a7d6286cffcc78da3a9ca5295c79aa99ea2d92672ebf733'\""
        ),
        f"git -C {LLAMA_CPP_ROOT} apply --check {MEASUREMENT_PATCH_REMOTE}",
        f"git -C {LLAMA_CPP_ROOT} apply {MEASUREMENT_PATCH_REMOTE}",
        # common_init prefixes INFO/ERROR logs. Machine records use level-NONE LOG.
        (
            "python -c 'from pathlib import Path; "
            'source=Path("/opt/llama.cpp/tools/perplexity/perplexity.cpp").read_text('
            'encoding="utf-8"); '
            'required=("LOG(\\"IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=%s",'
            '"LOG(\\"IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=measurement_failed",'
            '"LOG(\\"IQL_MEASUREMENT_TOKEN_NLL_V1 count=%d",'
            '"LOG(\\"Final estimate: PPL = %.4lf +/- %.5lf"); '
            'forbidden=("LOG_ERR(\\"IQL_MEASUREMENT_PERPLEXITY_ERROR_V1",'
            '"LOG_INF(\\"IQL_MEASUREMENT_TOKEN_NLL_V1",'
            '"LOG_INF(\\"Final estimate: PPL = %.4lf +/- %.5lf"); '
            "assert all(source.count(item)==1 for item in required) and "
            "not any(item in source for item in forbidden), "
            '"measurement protocol source is not exactly unprefixed"\''
        ),
        (
            f"cmake --build {LLAMA_CPP_ROOT}/build --clean-first --parallel 16 --target "
            + " ".join(BUILD_TARGETS)
        ),
        (
            "python -c 'from pathlib import Path; "
            'binary=Path("/opt/llama.cpp/build/bin/libllama-perplexity-impl.so").read_bytes(); '
            'markers=(b"IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=%s",'
            'b"IQL_MEASUREMENT_PERPLEXITY_ERROR_V1 code=measurement_failed",'
            'b"IQL_MEASUREMENT_TOKEN_NLL_V1 count=%d"); '
            "assert all(marker in binary for marker in markers), "
            '"patched perplexity DSO lacks the measurement protocol"\''
        ),
        "unlink /opt/iql-cuda-driver-link/libcuda.so.1",
        "rmdir /opt/iql-cuda-driver-link",
        (f'test "$(git -C {LLAMA_CPP_ROOT} rev-parse HEAD)" = "{PINNED_LLAMA_CPP_COMMIT}"'),
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_PROJECT_ROOT / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            (
                "IQL_BF16_DIAGNOSTIC_CONTROL_PLANE_SHA256"
                if _DIAGNOSTIC_MODE
                else "IQL_MEASUREMENT_CONTROL_PLANE_SHA256"
            ): _CONTROL_SHA256,
        }
    )
)


def _materialized_corpus_identity(bundle: InklingMeasurementBundle) -> _FileHash:
    reference = bundle.corpus
    if (
        reference.materialized_path != MATERIALIZED_CORPUS_PATH
        or reference.materializer_path != CORPUS_MATERIALIZER_RELATIVE_PATH
    ):
        raise RuntimeError("materialized corpus path differs from the reviewed reference")
    identity = _stable_file_identity(
        Path(MATERIALIZED_CORPUS_PATH),
        displayed_path=MATERIALIZED_CORPUS_PATH,
    )
    if (
        identity.sha256 != reference.corpus_sha256
        or identity.size_bytes != reference.corpus_size_bytes
    ):
        raise RuntimeError("materialized corpus bytes differ from the reviewed reference")
    return identity


def _build_bin_inventory() -> tuple[
    tuple[MeasurementRuntimeRegularFile, ...],
    tuple[MeasurementRuntimeSymlink, ...],
]:
    root = BUILD_BIN_ROOT.resolve(strict=True)
    regular: list[MeasurementRuntimeRegularFile] = []
    symlinks: list[MeasurementRuntimeSymlink] = []
    seen: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        # os.walk lists symlinked directories separately; record them, then stop
        # descent.  Every symlink must resolve to one regular build/bin file.
        for name in tuple(directory_names):
            candidate = directory_path / name
            if candidate.is_symlink():
                file_names.append(name)
                directory_names.remove(name)
        for name in file_names:
            candidate = directory_path / name
            relative = candidate.relative_to(root).as_posix()
            if relative in seen:
                raise RuntimeError("build/bin inventory contains a duplicate path")
            seen.add(relative)
            info = candidate.lstat()
            if stat.S_ISLNK(info.st_mode):
                raw_target = os.readlink(candidate)
                resolved = candidate.resolve(strict=True)
                if not resolved.is_relative_to(root) or not resolved.is_file():
                    raise RuntimeError(
                        "build/bin symlink does not resolve to a regular in-tree file"
                    )
                symlinks.append(
                    MeasurementRuntimeSymlink(
                        relative_path=relative,
                        raw_target=raw_target,
                        resolved_relative_path=resolved.relative_to(root).as_posix(),
                    )
                )
            elif stat.S_ISREG(info.st_mode):
                identity = _stable_file_identity(
                    candidate,
                    displayed_path=relative,
                    allow_empty=True,
                )
                regular.append(
                    MeasurementRuntimeRegularFile(
                        relative_path=identity.path,
                        sha256=identity.sha256,
                        size_bytes=identity.size_bytes,
                    )
                )
            else:
                raise RuntimeError("build/bin contains a non-file, non-symlink entry")
    regular.sort(key=lambda item: item.relative_path)
    symlinks.sort(key=lambda item: item.relative_path)
    regular_paths = {item.relative_path for item in regular}
    if not regular or any(item.resolved_relative_path not in regular_paths for item in symlinks):
        raise RuntimeError("build/bin symlink inventory is not closed over its files")
    return tuple(regular), tuple(symlinks)


def _parse_ldd(
    command: str,
    *,
    manifest_by_path: Mapping[str, MeasurementRuntimeRegularFile],
    symlink_by_path: Mapping[str, MeasurementRuntimeSymlink],
) -> MeasurementRuntimeCommandClosure:
    binary_path = Path(COMMAND_BINARIES[command])
    binary_relative = binary_path.relative_to(BUILD_BIN_ROOT).as_posix()
    binary_manifest_identity_path = (
        symlink_by_path[binary_relative].resolved_relative_path
        if binary_relative in symlink_by_path
        else binary_relative
    )
    binary_identity = manifest_by_path.get(binary_manifest_identity_path)
    if binary_identity is None:
        raise RuntimeError(f"{command} is absent from the post-patch manifest")
    observed_binary = _stable_file_identity(binary_path.resolve(strict=True))
    if (
        observed_binary.sha256 != binary_identity.sha256
        or observed_binary.size_bytes != binary_identity.size_bytes
    ):
        raise RuntimeError(f"{command} differs from the post-patch manifest")
    result = subprocess.run(
        ["ldd", binary_path.as_posix()],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        shell=False,
    )
    dependencies: list[MeasurementRuntimeDependency] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=> not found" in line:
            raise RuntimeError(f"{command} has an unresolved dynamic dependency")
        if "=>" in line:
            soname, remainder = (piece.strip() for piece in line.split("=>", 1))
            resolved_text = remainder.split(" (", 1)[0].strip()
        else:
            resolved_text = line.split(" (", 1)[0].strip()
            soname = PurePosixPath(resolved_text).name
        if not resolved_text.startswith("/"):
            dependencies.append(
                MeasurementRuntimeDependency(
                    classification="virtual",
                    soname=soname,
                    resolved_path=None,
                    sha256=None,
                    size_bytes=None,
                )
            )
            continue
        resolved = Path(resolved_text).resolve(strict=True)
        identity = _stable_file_identity(resolved)
        if resolved.is_relative_to(BUILD_BIN_ROOT.resolve(strict=True)):
            relative = resolved.relative_to(BUILD_BIN_ROOT.resolve(strict=True)).as_posix()
            expected = manifest_by_path.get(relative)
            if expected is None or (
                identity.sha256,
                identity.size_bytes,
            ) != (expected.sha256, expected.size_bytes):
                raise RuntimeError("project-owned DSO differs from build/bin manifest")
            classification: Literal["project_owned", "system", "cuda", "virtual"] = "project_owned"
        elif "cuda" in soname.lower() or "nvidia" in resolved.as_posix().lower():
            classification = "cuda"
        else:
            classification = "system"
        dependencies.append(
            MeasurementRuntimeDependency(
                classification=classification,
                soname=soname,
                resolved_path=resolved.as_posix(),
                sha256=identity.sha256,
                size_bytes=identity.size_bytes,
            )
        )
    dependencies.sort(
        key=lambda item: (
            item.soname,
            "" if item.resolved_path is None else item.resolved_path,
        )
    )
    identities = tuple(
        (
            item.soname,
            "" if item.resolved_path is None else item.resolved_path,
        )
        for item in dependencies
    )
    if len(set(identities)) != len(identities):
        raise RuntimeError(f"{command} dynamic dependency closure contains duplicates")
    if command == "llama-perplexity" and not any(
        item.classification == "project_owned" and "libllama-perplexity-impl" in item.soname
        for item in dependencies
    ):
        raise RuntimeError("llama-perplexity closure lacks its patched implementation DSO")
    return MeasurementRuntimeCommandClosure(
        command=cast(
            "Literal['llama-cli', 'llama-server', 'llama-bench', 'llama-perplexity']",
            command,
        ),
        binary_path=binary_path.as_posix(),
        binary_manifest_path=binary_relative,
        binary_sha256=binary_identity.sha256,
        binary_size_bytes=binary_identity.size_bytes,
        ldd_output_sha256=_sha256_bytes(result.stdout.encode("utf-8")),
        dependencies=tuple(dependencies),
    )


def _tool_version(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        shell=False,
    )
    text = (result.stdout + result.stderr).strip()
    if not text:
        raise RuntimeError(f"{command[0]} did not report its version")
    return text


def _runtime_identity(
    bundle: InklingMeasurementBundle | InklingBF16InterfaceDiagnosticBundle,
) -> MeasurementRuntimeIdentity:
    if isinstance(bundle, InklingBF16InterfaceDiagnosticBundle):
        base_runtime = bundle.config.runtime
        measurement_patch_path = bundle.config.runtime_measurement_patch.path
        measurement_patch_sha256 = bundle.config.runtime_measurement_patch.sha256
    else:
        base_runtime = bundle.config.base_runtime
        measurement_patch_path = bundle.config.measurement_patch.path
        measurement_patch_sha256 = bundle.config.measurement_patch.sha256
    commit = subprocess.run(
        ["git", "-C", LLAMA_CPP_ROOT.as_posix(), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        shell=False,
    ).stdout.strip()
    if commit != PINNED_LLAMA_CPP_COMMIT:
        raise RuntimeError("llama.cpp checkout differs from the pinned commit")
    manifest, symlinks = _build_bin_inventory()
    manifest_by_path = {item.relative_path: item for item in manifest}
    symlink_by_path = {item.relative_path: item for item in symlinks}
    commands = tuple(
        _parse_ldd(
            name,
            manifest_by_path=manifest_by_path,
            symlink_by_path=symlink_by_path,
        )
        for name in BUILD_TARGETS
    )
    patch_hashes = (
        _stable_file_identity(
            BASE_PATCH_REMOTE,
            displayed_path=base_runtime.instrumentation_patch_path,
        ),
        _stable_file_identity(
            MEASUREMENT_PATCH_REMOTE,
            displayed_path=measurement_patch_path,
        ),
    )
    if (
        patch_hashes[0].sha256 != base_runtime.instrumentation_patch_sha256
        or patch_hashes[1].sha256 != measurement_patch_sha256
    ):
        raise RuntimeError("runtime patch identity differs from the reviewed configuration")
    patches = tuple(
        MeasurementAppliedPatch(
            path=item.path,
            sha256=item.sha256,
            size_bytes=item.size_bytes,
        )
        for item in patch_hashes
    )
    base_pre_patch = tuple(
        MeasurementPrePatchExecutable(
            name=item.name,
            path=item.path,
            sha256=item.sha256,
            size_bytes=item.size_bytes,
        )
        for item in base_runtime.binaries
    )
    manifest_sha256 = measurement_runtime_manifest_sha256(
        build_bin_root=BUILD_BIN_ROOT.as_posix(),
        regular_files=manifest,
        symlinks=symlinks,
    )
    return MeasurementRuntimeIdentity(
        repository=base_runtime.repository,
        repository_commit=PINNED_LLAMA_CPP_COMMIT,
        cuda_image=base_runtime.cuda_image,
        cuda_image_digest=base_runtime.cuda_image_digest,
        platform=base_runtime.platform,
        patches_applied_in_order=cast(
            "tuple[MeasurementAppliedPatch, MeasurementAppliedPatch]",
            patches,
        ),
        base_pre_measurement_patch_executables=cast(
            "tuple[MeasurementPrePatchExecutable, MeasurementPrePatchExecutable, "
            "MeasurementPrePatchExecutable, MeasurementPrePatchExecutable]",
            base_pre_patch,
        ),
        cmake_generator="Ninja",
        effective_cmake_definitions=EFFECTIVE_CMAKE_DEFINITIONS,
        build_targets=MEASUREMENT_RUNTIME_COMMANDS,
        build_shared_libs=True,
        cmake_version=_tool_version(("cmake", "--version")),
        cxx_compiler_version=_tool_version(("c++", "--version")),
        cuda_compiler_version=_tool_version(("nvcc", "--version")),
        build_bin_root="/opt/llama.cpp/build/bin",
        regular_files=manifest,
        symlinks=symlinks,
        commands=cast(
            "tuple[MeasurementRuntimeCommandClosure, "
            "MeasurementRuntimeCommandClosure, MeasurementRuntimeCommandClosure, "
            "MeasurementRuntimeCommandClosure]",
            commands,
        ),
        manifest_sha256=manifest_sha256,
    )


def _real_cuda_driver_path(runtime: MeasurementRuntimeIdentity) -> str:
    candidates = {
        dependency.resolved_path
        for command in runtime.commands
        for dependency in command.dependencies
        if dependency.soname == "libcuda.so.1"
        and dependency.classification == "cuda"
        and dependency.resolved_path is not None
    }
    if len(candidates) != 1:
        raise RuntimeError("runtime closure does not bind one real libcuda.so.1")
    path = candidates.pop()
    assert path is not None
    forbidden = (
        PurePosixPath("/usr/local/cuda/lib64/stubs"),
        PurePosixPath("/opt/iql-cuda-driver-link"),
    )
    pure = PurePosixPath(path)
    if any(root == pure or root in pure.parents for root in forbidden):
        raise RuntimeError("runtime resolved libcuda to a build-time stub")
    return path


def _real_cuda_runtime_dependency(
    runtime: MeasurementRuntimeIdentity,
) -> MeasurementRuntimeDependency:
    candidates: dict[
        tuple[str, str, int, str],
        MeasurementRuntimeDependency,
    ] = {}
    for command in runtime.commands:
        for dependency in command.dependencies:
            if (
                dependency.classification != "cuda"
                or CUDA_RUNTIME_SONAME_RE.fullmatch(dependency.soname) is None
                or dependency.resolved_path is None
                or dependency.sha256 is None
                or dependency.size_bytes is None
            ):
                continue
            key = (
                dependency.resolved_path,
                dependency.sha256,
                dependency.size_bytes,
                dependency.soname,
            )
            candidates[key] = dependency
    if len(candidates) != 1:
        raise RuntimeError("runtime closure does not bind one exact libcudart")
    dependency = next(iter(candidates.values()))
    assert dependency.resolved_path is not None
    assert dependency.sha256 is not None
    assert dependency.size_bytes is not None
    observed = _stable_file_identity(Path(dependency.resolved_path))
    if observed.sha256 != dependency.sha256 or observed.size_bytes != dependency.size_bytes:
        raise RuntimeError("libcudart changed after the runtime closure was recorded")
    return dependency


def _run_cuda_runtime_preflight(
    runtime: MeasurementRuntimeIdentity,
) -> dict[str, Any]:
    dependency = _real_cuda_runtime_dependency(runtime)
    assert dependency.resolved_path is not None
    assert dependency.sha256 is not None
    assert dependency.size_bytes is not None
    script_path = Path(__file__).resolve(strict=True)
    command = [
        sys.executable,
        script_path.as_posix(),
        CUDA_RUNTIME_PREFLIGHT_CHILD_MODE,
        dependency.soname,
        dependency.resolved_path,
        dependency.sha256,
        str(dependency.size_bytes),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=CUDA_RUNTIME_PREFLIGHT_CHILD_TIMEOUT_SECONDS,
            shell=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("CUDA Runtime preflight child process timed out") from error
    if (
        len(result.stdout) > CUDA_RUNTIME_PREFLIGHT_CHILD_MAX_OUTPUT_BYTES
        or len(result.stderr) > CUDA_RUNTIME_PREFLIGHT_CHILD_MAX_OUTPUT_BYTES
    ):
        raise RuntimeError("CUDA Runtime preflight child process output exceeded its limit")
    if result.returncode != 0:
        detail = result.stderr[-4096:].decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"CUDA Runtime preflight child process exited with code {result.returncode}{suffix}"
        )
    try:
        strict_measurement_json_object(result.stdout)
        evidence = MeasurementCudaRuntimePreflight.model_validate_json(
            result.stdout,
            strict=True,
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "CUDA Runtime preflight child returned invalid typed evidence"
        ) from error
    canonical = canonical_measurement_raw_json_bytes(evidence.model_dump(mode="json"))
    if result.stdout != canonical:
        raise RuntimeError("CUDA Runtime preflight child evidence is not canonical JSON")
    if (
        evidence.libcudart_soname != dependency.soname
        or evidence.libcudart_path != dependency.resolved_path
        or evidence.libcudart_sha256 != dependency.sha256
        or evidence.libcudart_size_bytes != dependency.size_bytes
    ):
        raise RuntimeError("CUDA Runtime preflight child used a different libcudart")
    return evidence.model_dump(mode="json")


def _validate_remote_provenance(expected_sha256: str) -> MeasurementControlPlaneProvenance:
    payload = _read_regular_bytes(REMOTE_PROVENANCE_PATH)
    strict_measurement_json_object(payload)
    provenance = MeasurementControlPlaneProvenance.model_validate_json(
        payload,
        strict=True,
    )
    files = {
        item.path: _read_regular_bytes(
            REMOTE_PROJECT_ROOT / item.path,
            maximum_bytes=item.size_bytes,
        )
        for item in provenance.files
    }
    observed = validate_measurement_control_plane_provenance(
        payload,
        reviewed_commit_sha=provenance.reviewed_commit_sha,
        reviewed_tree_sha=provenance.reviewed_tree_sha,
        files=files,
        required_paths=tuple(item.path for item in provenance.files),
    )
    if observed.control_plane_sha256 != expected_sha256:
        raise RuntimeError("deployed control plane differs from authorization")
    return observed


def _invocation_ids() -> tuple[str, str, str]:
    call_id = modal.current_function_call_id()
    input_id = modal.current_input_id()
    task_id = os.environ.get("MODAL_TASK_ID")
    if (
        not isinstance(call_id, str)
        or re.fullmatch(r"fc-[A-Za-z0-9]+", call_id) is None
        or not isinstance(input_id, str)
        or re.fullmatch(r"in-[A-Za-z0-9]+(?::[0-9]+-[0-9]+)?", input_id) is None
        or not isinstance(task_id, str)
        or re.fullmatch(r"ta-[A-Za-z0-9]+", task_id) is None
    ):
        raise RuntimeError("Modal invocation identity is unavailable")
    return call_id, input_id, task_id


def _load_intent(run_id: str, intent_sha256: str) -> MeasurementLaunchIntent:
    relative = measurement_launch_intent_path(run_id, intent_sha256)
    payload = _read_regular_bytes(_evidence_path(relative))
    strict_measurement_json_object(payload)
    intent = MeasurementLaunchIntent.model_validate_json(payload, strict=True)
    validate_measurement_launch_intent(
        payload,
        expected=intent,
        intent_sha256=intent_sha256,
        evidence_path=relative,
    )
    return intent


def _wait_for_acceptance(
    intent: MeasurementLaunchIntent,
    *,
    call_id: str,
) -> MeasurementPostSpawnAcceptance:
    relative = measurement_post_spawn_acceptance_path(
        intent.run_id,
        intent.intent_sha256(),
    )
    deadline = time.monotonic() + ACCEPTANCE_TIMEOUT_SECONDS
    while True:
        evidence_volume.reload()
        try:
            payload = _read_regular_bytes(_evidence_path(relative))
            break
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise RuntimeError("post-spawn acceptance was not published in time") from None
            time.sleep(0.25)
    strict_measurement_json_object(payload)
    raw = MeasurementPostSpawnAcceptance.model_validate_json(payload, strict=True)
    expected = build_measurement_post_spawn_acceptance(
        intent,
        accepted_at_utc=raw.accepted_at_utc,
        call_id=call_id,
    )
    validate_measurement_post_spawn_acceptance(
        payload,
        expected=expected,
        acceptance_sha256=raw.acceptance_sha256(),
        evidence_path=relative,
    )
    return raw


def _claim_attempt(
    intent: MeasurementLaunchIntent,
    acceptance: MeasurementPostSpawnAcceptance,
    invocation: tuple[str, str, str],
) -> InvocationBinding:
    call_id, input_id, task_id = invocation
    registry = modal.Dict.from_id(intent.deployment.attempt_registry_id)
    registry.hydrate()
    info = registry.info()
    created = cast(object, info.created_at)
    if isinstance(created, datetime):
        if created.tzinfo is None:
            raise RuntimeError("attempt registry time has no time zone")
        created_at = created.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    elif isinstance(created, (int, float)) and not isinstance(created, bool):
        created_at = datetime.fromtimestamp(float(created), UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        raise RuntimeError("attempt registry creation time is unavailable")
    if (
        registry.object_id != intent.deployment.attempt_registry_id
        or info.name != MEASUREMENT_ATTEMPT_REGISTRY_NAME
        or created_at != intent.deployment.attempt_registry_created_at_utc
    ):
        raise RuntimeError("sealed attempt registry identity changed")
    claim = build_measurement_attempt_claim(
        intent,
        acceptance,
        claimed_at_utc=_utc_now(),
        input_id=input_id,
        task_id=task_id,
    )
    claim_sha = claim_measurement_attempt(registry, claim)
    relative = measurement_attempt_claim_path(intent.run_id, claim_sha)
    _commit_and_verify({relative: claim.canonical_bytes()})
    validate_measurement_attempt_claim(
        _read_regular_bytes(_evidence_path(relative)),
        expected=claim,
        claim_sha256=claim_sha,
        evidence_path=relative,
    )
    return InvocationBinding(
        intent=intent,
        acceptance=acceptance,
        claim_sha256=claim_sha,
        call_id=call_id,
    )


def _subject_specs(bundle: InklingMeasurementBundle) -> tuple[SubjectSpec, SubjectSpec]:
    paths = bundle.matched.paths
    tokenizer_pairs = tuple(
        (path, artifact.sha256, artifact.size_bytes)
        for path, artifact in zip(
            paths.tokenizer_assets,
            bundle.matched.config.tokenizer_assets,
            strict=True,
        )
    )
    projector = bundle.matched.q3.projector
    shared_projector = (
        paths.shared_projector,
        projector.sha256,
        projector.size_bytes,
    )
    bf16_artifacts = (
        *(
            (path, artifact.sha256, artifact.size_bytes)
            for path, artifact in zip(
                paths.bf16_shards,
                bundle.matched.bf16.bf16_shards,
                strict=True,
            )
        ),
        (
            paths.bf16_conversion_receipt,
            bundle.matched.bf16.conversion_receipt.sha256,
            bundle.matched.bf16.conversion_receipt.size_bytes,
        ),
        shared_projector,
        *tokenizer_pairs,
    )
    q3_artifacts = (
        *(
            (path, artifact.sha256, artifact.size_bytes)
            for path, artifact in zip(
                paths.q3_shards,
                bundle.matched.q3.q3_shards,
                strict=True,
            )
        ),
        shared_projector,
        (
            paths.q3_export_manifest,
            bundle.matched.q3.export_manifest.sha256,
            bundle.matched.q3.export_manifest.size_bytes,
        ),
        (
            paths.q3_verify_receipt,
            bundle.matched.q3.verify_receipt.sha256,
            bundle.matched.q3.verify_receipt.size_bytes,
        ),
        (
            paths.q3_quantize_receipt,
            bundle.matched.q3.quantize_receipt.sha256,
            bundle.matched.q3.quantize_receipt.size_bytes,
        ),
        (
            paths.projector_conversion_receipt,
            bundle.matched.q3.mmproj_receipt.sha256,
            bundle.matched.q3.mmproj_receipt.size_bytes,
        ),
        *tokenizer_pairs,
    )
    return (
        SubjectSpec(
            name="bf16",
            model_path=paths.bf16_shards[0],
            shard_paths=paths.bf16_shards,
            projector_path=paths.shared_projector,
            artifacts=bf16_artifacts,
        ),
        SubjectSpec(
            name="q3",
            model_path=paths.q3_shards[0],
            shard_paths=paths.q3_shards,
            projector_path=paths.shared_projector,
            artifacts=q3_artifacts,
        ),
    )


def _measurement_evidence_bindings(
    binding: InvocationBinding,
    *,
    bundle: InklingMeasurementBundle,
    subject: Literal["bf16", "q3"],
) -> MeasurementEvidenceBindings:
    reviewed = binding.intent.reviewed_inputs
    reviewed_config_sha256 = reviewed.measurement_config.sha256
    resolved_config_sha256 = bundle.config.config_hash()
    if (
        reviewed.resolved_config_sha256 != resolved_config_sha256
        or binding.acceptance.reviewed_config_file_sha256 != reviewed_config_sha256
        or binding.acceptance.resolved_config_sha256 != resolved_config_sha256
        or binding.acceptance.control_plane_sha256 != reviewed.control_plane.control_plane_sha256
    ):
        raise RuntimeError("measurement attempt config bindings differ from the reviewed inputs")
    values = MeasurementEvidenceBindings(
        run_id=binding.intent.run_id,
        subject=subject,
        reviewed_config_file_sha256=reviewed_config_sha256,
        resolved_config_sha256=resolved_config_sha256,
        protocol_sha256=measurement_protocol_sha256(bundle.config),
        workload_sha256=measurement_workload_sha256(bundle.config),
        launch_intent_sha256=binding.intent.intent_sha256(),
        post_spawn_acceptance_sha256=binding.acceptance.acceptance_sha256(),
        call_id=binding.call_id,
        attempt_claim_sha256=binding.claim_sha256,
    )
    for name, value in values.fields().items():
        if name in {"run_id", "subject", "call_id"}:
            continue
        if SHA256_RE.fullmatch(value) is None:
            raise RuntimeError(f"measurement evidence binding {name} is not SHA-256")
    return values


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise RuntimeError("staged artifact write made no progress")
        written += count


def _stage_file_once(
    *,
    source_path: str,
    resolved_source_path: Path,
    staged_path: Path,
    expected_sha256: str,
    expected_size_bytes: int,
    work_deadline: float,
) -> dict[str, Any]:
    """Copy and verify one mounted artifact in the same source-file pass."""

    source_descriptor = os.open(
        resolved_source_path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = staged_path.parent / f".{staged_path.name}.{secrets.token_hex(16)}.tmp"
    target_descriptor = -1
    digest = hashlib.sha256()
    try:
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size_bytes:
            raise RuntimeError("mounted subject artifact has the wrong type or size")
        target_descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        remaining = expected_size_bytes
        while remaining:
            _remaining_work_timeout(
                work_deadline,
                1.0,
                label="subject artifact staging",
            )
            chunk = os.read(source_descriptor, min(16 * 1024 * 1024, remaining))
            if not chunk:
                raise RuntimeError("mounted subject artifact ended before its reviewed size")
            digest.update(chunk)
            _write_all(target_descriptor, chunk)
            remaining -= len(chunk)
        if os.read(source_descriptor, 1):
            raise RuntimeError("mounted subject artifact exceeds its reviewed size")
        os.fsync(target_descriptor)
        target_info = os.fstat(target_descriptor)
        after = os.fstat(source_descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if tuple(getattr(before, field) for field in stable_fields) != tuple(
            getattr(after, field) for field in stable_fields
        ):
            raise RuntimeError("mounted subject artifact changed during staging")
        if not stat.S_ISREG(target_info.st_mode) or target_info.st_size != expected_size_bytes:
            raise RuntimeError("local staged artifact has the wrong type or size")
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise RuntimeError("mounted subject artifact differs from its reviewed hash")
    finally:
        os.close(source_descriptor)
        if target_descriptor >= 0:
            os.close(target_descriptor)
    try:
        if os.path.lexists(staged_path):
            raise RuntimeError("local staged artifact path already exists")
        os.rename(temporary, staged_path)
    finally:
        with suppress(OSError):
            temporary.unlink()
    return {
        "source_path": source_path,
        "staged_path": staged_path.as_posix(),
        "sha256": expected_sha256,
        "size_bytes": expected_size_bytes,
        "source_passes": 1,
    }


def _prepare_staging_root(bundle: InklingMeasurementBundle) -> Path:
    execution = bundle.config.execution
    if (
        execution.subject_staging_root != SUBJECT_STAGING_ROOT.as_posix()
        or execution.subject_staging_headroom_mib * 1024 * 1024 != SUBJECT_STAGING_HEADROOM_BYTES
        or not execution.release_staged_subject_before_next
    ):
        raise RuntimeError("subject staging configuration differs from the fixed protocol")
    parent = SUBJECT_STAGING_ROOT.parent
    if not os.path.lexists(parent):
        os.mkdir(parent, 0o700)
    _require_canonical_directory_components(
        parent,
        label="subject staging parent",
    )
    if not os.path.lexists(SUBJECT_STAGING_ROOT):
        os.mkdir(SUBJECT_STAGING_ROOT, 0o700)
    return _require_canonical_directory_components(
        SUBJECT_STAGING_ROOT,
        label="subject staging root",
    )


def _stage_subject(
    subject: SubjectSpec,
    *,
    bundle: InklingMeasurementBundle,
    work_deadline: float,
) -> tuple[SubjectSpec, dict[str, Any]]:
    """Stage one complete subject locally while hashing each source file once."""

    root = _prepare_staging_root(bundle)
    subject_root = root / subject.name
    if os.path.lexists(subject_root):
        raise RuntimeError("subject staging directory already exists")
    source_paths = tuple(path for path, _, _ in subject.artifacts)
    if len(source_paths) != len(set(source_paths)):
        raise RuntimeError("subject artifact inventory contains duplicate paths")
    allowed_roots = tuple(
        Path(path)
        for path in (
            bundle.matched.config.storage.bf16_mount_path,
            bundle.matched.config.storage.final_mount_path,
            bundle.matched.config.storage.source_mount_path,
        )
    )
    resolved_roots = {
        root: _resolved_modal_mount_root(
            root,
            label=f"{root.as_posix()} subject mount",
        )
        for root in allowed_roots
    }
    resolved_sources: dict[str, Path] = {}
    for source_path in source_paths:
        source = Path(source_path)
        matching_roots = tuple(
            root for root in allowed_roots if source == root or source.is_relative_to(root)
        )
        if (
            not source.is_absolute()
            or "\\" in source_path
            or "\x00" in source_path
            or PurePosixPath(source_path).as_posix() != source_path
            or any(part in {"", ".", ".."} for part in PurePosixPath(source_path).parts[1:])
            or len(matching_roots) != 1
        ):
            raise RuntimeError("subject artifact path is outside reviewed read-only mounts")
        mount_root = matching_roots[0]
        suffix = source.relative_to(mount_root)
        resolved_source = resolved_roots[mount_root].joinpath(*suffix.parts)
        if not resolved_source.is_relative_to(resolved_roots[mount_root]):
            raise RuntimeError("subject artifact path escaped its resolved read-only mount")
        _require_canonical_directory_components(
            resolved_source.parent,
            label="subject artifact parent",
        )
        resolved_sources[source_path] = resolved_source
    required_bytes = sum(size for _, _, size in subject.artifacts)
    filesystem = os.statvfs(root)
    free_bytes = filesystem.f_bavail * filesystem.f_frsize
    if free_bytes < required_bytes + SUBJECT_STAGING_HEADROOM_BYTES:
        raise RuntimeError(
            "local ephemeral disk lacks subject bytes plus the required 128 GiB headroom"
        )
    observed: list[dict[str, Any]] = []
    staged_by_source: dict[str, str] = {}
    subject_root.mkdir(mode=0o700)
    try:
        for source_path, expected_hash, expected_size in subject.artifacts:
            _remaining_work_timeout(
                work_deadline,
                1.0,
                label=f"{subject.name} subject staging",
            )
            staged_path = subject_root / Path(source_path).relative_to("/")
            item = _stage_file_once(
                source_path=source_path,
                resolved_source_path=resolved_sources[source_path],
                staged_path=staged_path,
                expected_sha256=expected_hash,
                expected_size_bytes=expected_size,
                work_deadline=work_deadline,
            )
            observed.append(item)
            staged_by_source[source_path] = staged_path.as_posix()
    except BaseException:
        with suppress(OSError):
            shutil.rmtree(subject_root)
        raise
    model_path = staged_by_source.get(subject.model_path)
    projector_path = staged_by_source.get(subject.projector_path)
    shard_paths = tuple(staged_by_source.get(path, "") for path in subject.shard_paths)
    if model_path is None or projector_path is None or any(not path for path in shard_paths):
        shutil.rmtree(subject_root)
        raise RuntimeError("staged subject does not bind every executable model artifact")
    inventory_payload = {
        "artifacts": observed,
        "required_bytes": required_bytes,
        "required_headroom_bytes": SUBJECT_STAGING_HEADROOM_BYTES,
    }
    staged = SubjectSpec(
        name=subject.name,
        model_path=model_path,
        shard_paths=shard_paths,
        projector_path=projector_path,
        artifacts=tuple(
            (staged_by_source[path], sha256, size_bytes)
            for path, sha256, size_bytes in subject.artifacts
        ),
    )
    return staged, {
        "schema_version": "inkling-measurement-subject-staging-v1",
        "subject": subject.name,
        "source_volume_read_only": True,
        "copy_and_hash_same_source_pass": True,
        "source_passes_per_artifact": 1,
        "staging_root": subject_root.as_posix(),
        "artifact_count": len(observed),
        "required_bytes": required_bytes,
        "required_headroom_bytes": SUBJECT_STAGING_HEADROOM_BYTES,
        "free_bytes_before_staging": free_bytes,
        "artifacts": observed,
        "inventory_sha256": _sha256_bytes(canonical_measurement_json_bytes(inventory_payload)),
    }


def _release_staged_subject(subject: SubjectSpec) -> None:
    root = _require_canonical_directory_components(
        SUBJECT_STAGING_ROOT,
        label="subject staging root",
    )
    subject_root = root / subject.name
    if (
        not subject_root.exists()
        or subject_root.is_symlink()
        or subject_root.resolve(strict=True) != subject_root
        or subject_root.parent != root
    ):
        raise RuntimeError("staged subject directory is not safe to release")
    shutil.rmtree(subject_root)
    if os.path.lexists(subject_root):
        raise RuntimeError("staged subject directory remained after release")


def _observe_hardware(runtime: MeasurementRuntimeIdentity) -> dict[str, Any]:
    identity_csv = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        shell=False,
    ).stdout
    unordered = parse_matched_nvidia_smi_identity_csv(identity_csv)
    cuda_driver = _real_cuda_driver_path(runtime)
    topology = enumerate_matched_cuda_peer_topology(
        cuda_driver,
        nvidia_smi_gpus=unordered,
    )
    ordered = order_matched_nvidia_smi_identity_by_cuda_uuid(
        unordered,
        cuda_gpu_uuids=topology.gpu_uuids,
    )
    if tuple(item.cuda_ordinal for item in ordered) != tuple(range(8)):
        raise RuntimeError("allocation is not exact CUDA ordinals zero through seven")
    if any(
        item.name != "NVIDIA B300 SXM6 AC"
        or item.compute_capability != "10.3"
        or item.memory_total_mib != 275_040
        for item in ordered
    ):
        raise RuntimeError("allocation does not provide the exact eight-B300 cell")
    cuda_runtime_preflight = _run_cuda_runtime_preflight(runtime)
    payload = {
        "schema_version": "inkling-measurement-hardware-identity-v1",
        "backend": "CUDA",
        "logical_devices": [f"cuda:{index}" for index in range(8)],
        "gpus": [item.model_dump(mode="json") for item in ordered],
        "peer_topology": topology.model_dump(mode="json"),
        "cuda_driver_path": cuda_driver,
        "cuda_runtime_preflight": cuda_runtime_preflight,
        "precision": "model-native-subject-precision",
        "gpu_layers": "all",
        "cpu_moe_layers": 0,
        "cpu_fallback": False,
    }
    payload["identity_sha256"] = _sha256_bytes(canonical_measurement_json_bytes(payload))
    return payload


class ResourceMonitor:
    """Collect raw host-process and per-GPU samples for one child process."""

    def __init__(self, pid: int, expected_uuids: Sequence[str]) -> None:
        self._pid = pid
        self._expected_uuids = tuple(expected_uuids)
        self._samples: list[dict[str, Any]] = []
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    @property
    def process_id(self) -> int:
        return self._pid

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise RuntimeError("resource monitor did not stop")
        if self._error is not None:
            raise RuntimeError("resource monitor failed") from self._error
        if not self._samples:
            raise RuntimeError("resource monitor produced no complete samples")
        max_sampled_rss = max(int(item["host_rss_bytes"]) for item in self._samples)
        max_sampled_memory_by_gpu = [
            max(
                int(cast("list[dict[str, Any]]", item["gpus"])[index]["memory_used_mib"])
                * 1024
                * 1024
                for item in self._samples
            )
            for index in range(8)
        ]
        max_sampled_utilization_by_gpu = [
            [
                int(cast("list[dict[str, Any]]", item["gpus"])[index]["utilization_percent"])
                for item in self._samples
            ]
            for index in range(8)
        ]
        return {
            "schema_version": "inkling-measurement-resource-telemetry-v1",
            "requested_sampling_interval_seconds": 1.0,
            "sample_count": len(self._samples),
            "samples": self._samples,
            "max_sampled_host_rss_bytes": max_sampled_rss,
            "max_sampled_per_gpu_memory_bytes": max_sampled_memory_by_gpu,
            "max_sampled_per_gpu_utilization_percent": [
                max(values) for values in max_sampled_utilization_by_gpu
            ],
        }

    def _host_rss(self) -> int | None:
        status_path = Path(f"/proc/{self._pid}/status")
        try:
            text = status_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        matches = re.findall(r"^VmRSS:\s+([0-9]+)\s+kB$", text, flags=re.MULTILINE)
        if len(matches) == 1:
            return int(matches[0]) * 1024
        states = re.findall(
            r"^State:\s+([A-Za-z])\s+\([^)]+\)$",
            text,
            flags=re.MULTILINE,
        )
        if len(states) == 1 and states[0] in {"Z", "X"}:
            return None
        raise RuntimeError("child process RSS is unavailable")

    def _sample(self) -> bool:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            shell=False,
        ).stdout
        gpu_samples = parse_matched_nvidia_smi_monitor_csv(
            output,
            expected_uuids=self._expected_uuids,
        )
        host_rss = self._host_rss()
        if host_rss is None:
            return False
        self._samples.append(
            {
                "requested_sampling_interval_seconds": 1.0,
                "sampled_at_monotonic_seconds": time.monotonic(),
                "host_rss_bytes": host_rss,
                "gpus": [
                    {
                        "cuda_ordinal": cuda_ordinal,
                        **item.model_dump(mode="json"),
                    }
                    for cuda_ordinal, item in enumerate(gpu_samples)
                ],
            }
        )
        return True

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._sample():
                    return
                self._stop.wait(1.0)
        except BaseException as error:
            self._error = error


def _telemetry_window(
    telemetry: Mapping[str, Any],
    *,
    started_monotonic: float,
    finished_monotonic: float,
) -> dict[str, Any]:
    samples_raw = telemetry.get("samples")
    if not isinstance(samples_raw, list):
        raise RuntimeError("resource telemetry does not contain raw samples")
    samples = [
        item
        for item in samples_raw
        if isinstance(item, Mapping)
        and isinstance(item.get("sampled_at_monotonic_seconds"), (int, float))
        and started_monotonic <= float(item["sampled_at_monotonic_seconds"]) <= finished_monotonic
    ]
    if not samples:
        raise RuntimeError("measurement cell has no in-window resource sample")
    max_sampled_host_rss = max(int(item["host_rss_bytes"]) for item in samples)
    memory = tuple(
        max(
            int(cast("list[dict[str, Any]]", item["gpus"])[index]["memory_used_mib"]) * 1024 * 1024
            for item in samples
        )
        for index in range(8)
    )
    utilization = tuple(
        float(
            max(
                int(cast("list[dict[str, Any]]", item["gpus"])[index]["utilization_percent"])
                for item in samples
            )
        )
        for index in range(8)
    )
    if max_sampled_host_rss <= 0 or any(value <= 0 for value in memory):
        raise RuntimeError("measurement cell sampled resource maxima are not positive")
    if any(value < 0.0 or value > 100.0 for value in utilization):
        raise RuntimeError("GPU utilization is outside its valid percentage range")
    return {
        "window_started_monotonic_seconds": started_monotonic,
        "window_finished_monotonic_seconds": finished_monotonic,
        "sample_count": len(samples),
        "max_sampled_host_rss_bytes": max_sampled_host_rss,
        "max_sampled_per_gpu_memory_bytes": list(memory),
        "max_sampled_per_gpu_utilization_percent": list(utilization),
    }


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    process_id: int
    started_monotonic_seconds: float
    finished_monotonic_seconds: float
    stdout: str
    stderr: str
    elapsed_seconds: float
    telemetry: dict[str, Any]


def _run_captured(
    command: Sequence[str],
    *,
    expected_uuids: Sequence[str],
    timeout: float,
    work_deadline: float,
    environment: Mapping[str, str] | None = None,
) -> CommandResult:
    started = time.monotonic()
    exact_command = tuple(command)
    process = subprocess.Popen(
        exact_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=None if environment is None else dict(environment),
        shell=False,
    )
    monitor = ResourceMonitor(process.pid, expected_uuids)
    monitor.start()
    try:
        stdout_bytes, stderr_bytes = process.communicate(
            timeout=_remaining_work_timeout(
                work_deadline,
                timeout,
                label=PurePosixPath(exact_command[0]).name,
            )
        )
        finished = time.monotonic()
    except BaseException:
        with suppress(ProcessLookupError):
            process.kill()
        process.wait(timeout=30)
        raise
    finally:
        telemetry = monitor.stop()
    if len(stdout_bytes) > MAX_LOG_BYTES or len(stderr_bytes) > MAX_LOG_BYTES:
        raise RuntimeError("command output exceeded its evidence bound")
    stdout = stdout_bytes.decode("utf-8", errors="strict")
    stderr = stderr_bytes.decode("utf-8", errors="strict")
    if process.returncode != 0:
        failure = f"{PurePosixPath(command[0]).name} returned nonzero"
        marker = extract_llama_perplexity_machine_failure(stdout, stderr)
        if marker is not None:
            failure = f"{failure}: {marker}"
        raise RuntimeError(failure)
    return CommandResult(
        command=exact_command,
        process_id=process.pid,
        started_monotonic_seconds=started,
        finished_monotonic_seconds=finished,
        stdout=stdout,
        stderr=stderr,
        elapsed_seconds=finished - started,
        telemetry=telemetry,
    )


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    def reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("HTTP JSON contains a duplicate key")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"HTTP JSON contains non-finite constant {value}")

    parsed = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_pairs,
        parse_constant=reject_constant,
    )
    if type(parsed) is not dict:
        raise RuntimeError("HTTP response root is not one JSON object")
    return parsed


def _http_json(
    port: int,
    method: str,
    endpoint: str,
    body: Mapping[str, Any] | None,
    *,
    timeout: float,
    work_deadline: float,
) -> tuple[dict[str, Any], str]:
    encoded = None if body is None else canonical_measurement_json_bytes(body)
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{endpoint}",
        data=encoded,
        method=method,
        headers={} if encoded is None else {"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=_remaining_work_timeout(
                work_deadline,
                timeout,
                label=f"{method} {endpoint}",
            ),
        ) as response:
            payload = response.read(MAX_HTTP_BYTES + 1)
            if len(payload) > MAX_HTTP_BYTES:
                raise RuntimeError("HTTP response exceeds its evidence bound")
            if response.status != 200:
                raise RuntimeError("HTTP endpoint returned non-200")
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP endpoint returned {error.code}") from error
    return _strict_json_object(payload), _sha256_bytes(payload)


def _wait_server_ready(
    port: int,
    process: subprocess.Popen[bytes],
    *,
    work_deadline: float,
) -> float:
    deadline = min(
        time.monotonic() + SERVER_READY_TIMEOUT_SECONDS,
        work_deadline,
    )
    while True:
        if process.poll() is not None:
            raise RuntimeError("llama-server exited before becoming healthy")
        try:
            payload, _ = _http_json(
                port,
                "GET",
                "/health",
                None,
                timeout=10,
                work_deadline=work_deadline,
            )
            if payload.get("status") == "ok":
                return time.monotonic()
        except (OSError, RuntimeError, ValueError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("llama-server did not become healthy in time")
        time.sleep(0.5)


def _start_server(
    *,
    subject: SubjectSpec,
    projector_path: str,
    port: int,
    parallel_slots: int,
    expected_uuids: Sequence[str],
    work_deadline: float,
) -> ServerProcess:
    topology = bind_exact_cuda_topology(
        tuple(f"CUDA{index}" for index in range(8)),
        (1,) * 8,
    )
    spec = LlamaServerCommandSpec(
        model_path=subject.model_path,
        projector_path=projector_path,
        context_size=8192,
        batch_size=2048,
        ubatch_size=512,
        parallel_slots=parallel_slots,
        port=port,
        topology=topology,
    )
    command = build_llama_server_command(spec)
    log_path = Path(f"/tmp/iql-measurement-{subject.name}-{port}.log")
    with suppress(FileNotFoundError):
        log_path.unlink()
    descriptor = os.open(
        log_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    environment = build_matched_server_environment(
        os.environ,
        audit_environment=SERVER_AUDIT_ENVIRONMENT,
    )
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            stdout=descriptor,
            stderr=subprocess.STDOUT,
            env=environment,
            shell=False,
        )
    finally:
        os.close(descriptor)
    monitor = ResourceMonitor(process.pid, expected_uuids)
    monitor.start()
    try:
        ready = _wait_server_ready(
            port,
            process,
            work_deadline=work_deadline,
        )
    except BaseException as error:
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=30)
        except BaseException as cleanup_error:
            error.add_note(
                f"server readiness process cleanup also failed: {type(cleanup_error).__name__}"
            )
        try:
            monitor.stop()
        except BaseException as cleanup_error:
            error.add_note(
                "server readiness resource-monitor cleanup also failed: "
                f"{type(cleanup_error).__name__}"
            )
        try:
            log_path.unlink()
        except FileNotFoundError:
            pass
        except BaseException as cleanup_error:
            error.add_note(
                f"server readiness log cleanup also failed: {type(cleanup_error).__name__}"
            )
        raise
    return ServerProcess(
        process=process,
        command=tuple(command),
        log_path=log_path,
        started_monotonic=started,
        ready_monotonic=ready,
        monitor=monitor,
    )


def _stop_server(server: ServerProcess) -> tuple[str, dict[str, Any], float]:
    if server.process.poll() is None:
        server.process.terminate()
        try:
            server.process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait(timeout=30)
    telemetry = server.monitor.stop()
    log_bytes = _read_regular_bytes(server.log_path, maximum_bytes=MAX_LOG_BYTES)
    server.log_path.unlink()
    if server.process.returncode not in (0, -15):
        raise RuntimeError("llama-server cleanup observed an unexpected exit")
    return (
        log_bytes.decode("utf-8", errors="strict"),
        telemetry,
        time.monotonic(),
    )


def _server_contract(port: int, *, work_deadline: float) -> tuple[int, str]:
    props, _ = _http_json(
        port,
        "GET",
        "/props",
        None,
        timeout=30,
        work_deadline=work_deadline,
    )
    modalities = props.get("modalities")
    marker = props.get("media_marker")
    build_info = props.get("build_info")
    if (
        not isinstance(modalities, Mapping)
        or modalities.get("vision") is not True
        or modalities.get("audio") is not True
        or not isinstance(marker, str)
        or marker != MEASUREMENT_MEDIA_MARKER
        or not isinstance(build_info, str)
        or PINNED_LLAMA_CPP_COMMIT[:7] not in build_info
    ):
        raise RuntimeError("llama-server does not expose the pinned multimodal build")
    models, _ = _http_json(
        port,
        "GET",
        "/v1/models",
        None,
        timeout=30,
        work_deadline=work_deadline,
    )
    data = models.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], Mapping):
        raise RuntimeError("llama-server model metadata has the wrong shape")
    meta = data[0].get("meta")
    if not isinstance(meta, Mapping) or type(meta.get("n_vocab")) is not int:
        raise RuntimeError("llama-server model metadata lacks a vocabulary size")
    vocab_size = int(meta["n_vocab"])
    if vocab_size <= 0:
        raise RuntimeError("llama-server vocabulary size is invalid")
    return vocab_size, marker


def _validate_completion(
    payload: Mapping[str, Any],
    *,
    vocab_size: int,
) -> tuple[str, tuple[int, ...], dict[str, Any]]:
    if payload.get("error") is not None:
        raise RuntimeError("completion returned an error")
    content = payload.get("content")
    tokens = payload.get("tokens")
    timings = payload.get("timings")
    if (
        not isinstance(content, str)
        or not isinstance(tokens, list)
        or not tokens
        or any(type(token) is not int or not 0 <= token < vocab_size for token in tokens)
        or not isinstance(timings, Mapping)
    ):
        raise RuntimeError("completion response violates its output contract")
    numeric_timings: dict[str, Any] = {}
    for field in (
        "prompt_n",
        "predicted_n",
        "prompt_ms",
        "predicted_ms",
        "prompt_per_second",
        "predicted_per_second",
    ):
        value = timings.get(field)
        if field.endswith("_n"):
            if type(value) is not int or value <= 0:
                raise RuntimeError("completion timing count is invalid")
            numeric_timings[field] = value
        else:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise RuntimeError("completion timing value is invalid")
            numeric_timings[field] = float(value)
    if numeric_timings["predicted_n"] != len(tokens):
        raise RuntimeError("completion timing and returned token counts differ")
    return content, tuple(int(token) for token in tokens), numeric_timings


def _run_diagnostics(
    *,
    port: int,
    marker: str,
    vocab_size: int,
    bundle: InklingMeasurementBundle,
    work_deadline: float,
) -> tuple[dict[str, Any], ...]:
    results: list[dict[str, Any]] = []
    for item in bundle.diagnostic_items:
        fixture = build_diagnostic_fixture_bytes(item.fixture)
        prompt_text = f"{bundle.config.quality.prompt_template}\n{item.prompt}"
        prompt: object = prompt_text
        if fixture is not None:
            prompt = {
                "prompt_string": f"{marker}\n{prompt_text}",
                "multimodal_data": [base64.b64encode(fixture).decode("ascii")],
            }
        request = {
            "prompt": prompt,
            "seed": item.seed,
            "temperature": item.temperature,
            "n_predict": item.max_new_tokens,
            "stream": False,
            "cache_prompt": False,
            "return_tokens": True,
            "timings_per_token": True,
        }
        request_body_sha256 = _sha256_bytes(canonical_measurement_json_bytes(request))
        request_started = time.monotonic()
        payload, response_sha = _http_json(
            port,
            "POST",
            "/completion",
            request,
            timeout=REQUEST_TIMEOUT_SECONDS,
            work_deadline=work_deadline,
        )
        request_finished = time.monotonic()
        content, token_ids, timings = _validate_completion(
            payload,
            vocab_size=vocab_size,
        )
        score_evidence = evaluate_diagnostic_response(
            content,
            scorer_kind=item.scorer.kind,
            expected=item.scorer.expected,
        )
        trial = {
            "trial_index": 1,
            "request_started_monotonic_seconds": request_started,
            "request_finished_monotonic_seconds": request_finished,
            "request_wall_seconds": request_finished - request_started,
            "token_ids": list(token_ids),
            "output_sha256": _sha256_bytes(content.encode("utf-8")),
            "response_sha256": response_sha,
            "normalization_succeeded": score_evidence.normalization_succeeded,
            "normalized_sha256": score_evidence.normalized_sha256,
            "score": score_evidence.score,
            "timings": timings,
        }
        results.append(
            {
                "item_id": item.item_id,
                "suite": item.suite,
                "modality": item.modality,
                "request_body_sha256": request_body_sha256,
                "prompt_sha256": _sha256_bytes(prompt_text.encode("utf-8")),
                "fixture_sha256": (None if fixture is None else _sha256_bytes(fixture)),
                "fixture_size_bytes": None if fixture is None else len(fixture),
                "seed": item.seed,
                "temperature": item.temperature,
                "max_new_tokens": item.max_new_tokens,
                "scorer_kind": item.scorer.kind,
                "score": trial["score"],
                "trials": [trial],
                "prompt_text_recorded": False,
                "output_text_recorded": False,
            }
        )
    if len(results) != 64:
        raise RuntimeError("diagnostic run did not produce exactly 64 complete items")
    return tuple(results)


def _parse_token_nll(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular_bytes(path, maximum_bytes=16 * 1024 * 1024)
    lines = payload.splitlines()
    if len(lines) != 16_321:
        raise RuntimeError("token NLL file does not contain its exact header and rows")
    header = _strict_json_object(lines[0])
    if header != {
        "schema_version": "iql-token-nll-v1",
        "n_ctx": 512,
        "n_chunks": 64,
        "scored_tokens": 16_320,
    }:
        raise RuntimeError("token NLL header differs from the reviewed protocol")
    rows: list[dict[str, Any]] = []
    for ordinal, line in enumerate(lines[1:]):
        row = _strict_json_object(line)
        if set(row) != {"chunk_index", "token_index", "token_id", "nll"}:
            raise RuntimeError("token NLL row fields differ from the protocol")
        chunk = ordinal // 255
        within = ordinal % 255
        expected_token_index = chunk * 512 + 257 + within
        nll = row["nll"]
        if (
            type(row["chunk_index"]) is not int
            or row["chunk_index"] != chunk
            or type(row["token_index"]) is not int
            or row["token_index"] != expected_token_index
            or type(row["token_id"]) is not int
            or row["token_id"] < 0
            or isinstance(nll, bool)
            or not isinstance(nll, (int, float))
            or not math.isfinite(nll)
            or nll < 0
        ):
            raise RuntimeError("token NLL row is invalid or out of exact order")
        rows.append(
            {
                "chunk_index": chunk,
                "token_index": expected_token_index,
                "token_id": row["token_id"],
                "nll": float(nll),
            }
        )
    canonical_rows = b"".join(canonical_measurement_raw_json_bytes(row) for row in rows)
    if canonical_rows.count(b"\n") != 16_320:
        raise RuntimeError("canonical token NLL evidence has the wrong record count")
    path.unlink()
    mean_nll = sum(float(row["nll"]) for row in rows) / len(rows)
    return (
        {
            "source_schema_version": "iql-token-nll-v1",
            "scored_tokens": 16_320,
            "mean_nll": mean_nll,
            "source_file_sha256": _sha256_bytes(payload),
            "source_file_size_bytes": len(payload),
            "canonical_raw_sha256": _sha256_bytes(canonical_rows),
            "canonical_raw_size_bytes": len(canonical_rows),
        },
        canonical_rows,
    )


def _perplexity_measurement(
    *,
    subject: SubjectSpec,
    bundle: InklingMeasurementBundle,
    corpus_identity: _FileHash,
    expected_uuids: Sequence[str],
    work_deadline: float,
) -> PerplexityMeasurementResult:
    topology = bind_exact_cuda_topology(
        tuple(f"CUDA{index}" for index in range(8)),
        (1,) * 8,
    )
    placement_policy = build_matched_cuda_placement_policy(bundle.matched.config)
    if (
        corpus_identity.path != bundle.corpus.materialized_path
        or corpus_identity.sha256 != bundle.corpus.corpus_sha256
        or corpus_identity.size_bytes != bundle.corpus.corpus_size_bytes
    ):
        raise RuntimeError("perplexity corpus identity differs from the reviewed reference")

    token_nll_path = Path(f"/tmp/iql-{subject.name}-token-nll-{secrets.token_hex(8)}.jsonl")
    with suppress(FileNotFoundError):
        token_nll_path.unlink()
    ppl_spec = LlamaPerplexityCommandSpec(
        model_path=subject.model_path,
        corpus_path=corpus_identity.path,
        context_size=512,
        batch_size=512,
        ubatch_size=512,
        chunks=64,
        topology=topology,
    )
    ppl_command = build_llama_perplexity_command(ppl_spec)
    environment = build_matched_server_environment(
        os.environ,
        audit_environment={
            **SERVER_AUDIT_ENVIRONMENT,
            "IQL_MEASUREMENT_TOKEN_NLL_PATH": token_nll_path.as_posix(),
        },
    )
    ppl_run = _run_captured(
        ppl_command,
        expected_uuids=expected_uuids,
        timeout=10_800,
        work_deadline=work_deadline,
        environment=environment,
    )
    combined = ppl_run.stdout + CAPTURED_TOOL_LOG_DELIMITER + ppl_run.stderr
    ppl = parse_llama_perplexity_final(combined)
    ppl_placement = parse_exact_text_cuda_backend_audit(
        combined,
        policy=placement_policy,
    )
    if combined.count("IQL_MEASUREMENT_TOKEN_NLL_V1 count=16320") != 1:
        raise RuntimeError("perplexity process did not confirm exact token-NLL emission")
    token_nll, token_nll_payload = _parse_token_nll(token_nll_path)

    return PerplexityMeasurementResult(
        evidence={
            "command": list(ppl_run.command),
            "corpus_reference_sha256": bundle.corpus.reference_sha256,
            "corpus_sha256": corpus_identity.sha256,
            "corpus_size_bytes": corpus_identity.size_bytes,
            "process_id": ppl_run.process_id,
            "process_started_monotonic_seconds": ppl_run.started_monotonic_seconds,
            "process_finished_monotonic_seconds": (ppl_run.finished_monotonic_seconds),
            "perplexity": ppl.perplexity,
            "uncertainty": ppl.uncertainty,
            "token_nll_sha256": _sha256_bytes(token_nll_payload),
            "token_nll_size_bytes": len(token_nll_payload),
            "elapsed_seconds": ppl_run.elapsed_seconds,
            "stdout_sha256": _sha256_bytes(ppl_run.stdout.encode("utf-8")),
            "stdout_size_bytes": len(ppl_run.stdout.encode("utf-8")),
            "stderr_sha256": _sha256_bytes(ppl_run.stderr.encode("utf-8")),
            "stderr_size_bytes": len(ppl_run.stderr.encode("utf-8")),
            "telemetry": ppl_run.telemetry,
            "placement_audit": ppl_placement.model_dump(mode="json"),
            "token_nll": token_nll,
        },
        token_nll_payload=token_nll_payload,
        stdout=ppl_run.stdout,
        stderr=ppl_run.stderr,
    )


def _quality_measurement(
    *,
    subject: SubjectSpec,
    perplexity: PerplexityMeasurementResult,
    diagnostics: Sequence[Mapping[str, Any]],
    vocab_size: int,
) -> QualityMeasurementResult:
    if len(diagnostics) != 64:
        raise RuntimeError("quality result requires exactly 64 diagnostic items")
    suite_scores = {
        suite: sum(int(item["score"]) for item in diagnostics if item["suite"] == suite) / 8.0
        for suite in SUITES
    }
    if any(sum(1 for item in diagnostics if item["suite"] == suite) != 8 for suite in SUITES):
        raise RuntimeError("quality result requires exactly eight items per suite")
    overall = sum(int(item["score"]) for item in diagnostics) / 64.0
    return QualityMeasurementResult(
        evidence={
            "schema_version": "inkling-measurement-subject-quality-v1",
            "subject": subject.name,
            "perplexity": perplexity.evidence,
            "diagnostics": list(diagnostics),
            "diagnostic_item_count": len(diagnostics),
            "diagnostic_repetitions": 1,
            "suite_accuracy": suite_scores,
            "overall_accuracy": overall,
            "vocab_size": vocab_size,
            "prompt_text_recorded": False,
            "output_text_recorded": False,
        },
        perplexity=perplexity,
    )


def _exact_prompt_tokens(port: int, *, work_deadline: float) -> tuple[int, ...]:
    # Tokenize once, then pass the first exact 512 IDs directly to /completion.
    # This avoids relying on text whose token count could vary by subject.
    payload, _ = _http_json(
        port,
        "POST",
        "/tokenize",
        {
            "content": measurement_server_prompt_source_text(),
            "add_special": False,
            "parse_special": False,
            "with_pieces": False,
        },
        timeout=60,
        work_deadline=work_deadline,
    )
    tokens = payload.get("tokens")
    if (
        not isinstance(tokens, list)
        or len(tokens) < 512
        or any(type(token) is not int or token < 0 for token in tokens)
    ):
        raise RuntimeError("tokenizer did not produce a usable exact prompt")
    return tuple(int(token) for token in tokens[:512])


def _stream_completion(
    *,
    port: int,
    prompt_tokens: Sequence[int],
    vocab_size: int,
    barrier: threading.Barrier | None,
    work_deadline: float,
) -> dict[str, Any]:
    if len(prompt_tokens) != 512:
        raise ValueError("streaming benchmark prompt must contain exactly 512 token IDs")
    body = canonical_measurement_raw_json_bytes(
        {
            "prompt": list(prompt_tokens),
            "seed": 42,
            "temperature": 0.0,
            "n_predict": 128,
            "stream": True,
            "cache_prompt": False,
            "return_tokens": True,
            "ignore_eos": True,
        }
    )
    request_body_sha256 = _sha256_bytes(body)
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    if barrier is not None:
        barrier.wait(
            timeout=_remaining_work_timeout(
                work_deadline,
                30,
                label="concurrent request barrier",
            )
        )
    started = time.monotonic()
    token_ids: list[int] = []
    token_times: list[float] = []
    content_hasher = hashlib.sha256()
    response_hasher = hashlib.sha256()
    response_bytes = 0
    final_timings: Mapping[str, Any] | None = None
    terminal_seen = False
    with urllib.request.urlopen(
        request,
        timeout=_remaining_work_timeout(
            work_deadline,
            REQUEST_TIMEOUT_SECONDS,
            label="streaming completion",
        ),
    ) as response:
        if response.status != 200:
            raise RuntimeError("streaming completion returned non-200")
        while True:
            line = response.readline(MAX_HTTP_BYTES + 1)
            if not line:
                break
            response_bytes += len(line)
            if response_bytes > MAX_HTTP_BYTES:
                raise RuntimeError("streaming response exceeded its evidence bound")
            response_hasher.update(line)
            stripped = line.strip()
            if not stripped or stripped.startswith(b":"):
                continue
            if not stripped.startswith(b"data:"):
                raise RuntimeError("streaming completion returned a non-SSE data line")
            event_bytes = stripped[5:].strip()
            if event_bytes == b"[DONE]":
                raise RuntimeError("streaming completion returned an unexpected DONE event")
            if terminal_seen:
                raise RuntimeError("streaming completion returned an event after terminal")
            event = _strict_json_object(event_bytes)
            if event.get("error") is not None:
                raise RuntimeError("streaming completion returned an error")
            event_tokens = event.get("tokens")
            content = event.get("content")
            stop = event.get("stop")
            if stop is True:
                if not isinstance(event_tokens, list) or event_tokens or content != "":
                    raise RuntimeError("terminal streaming event has payload content")
                timings = event.get("timings")
                if not isinstance(timings, Mapping):
                    raise RuntimeError("terminal streaming event lacks timings")
                final_timings = timings
                terminal_seen = True
                continue
            if (
                stop is not False
                or not isinstance(event_tokens, list)
                or len(event_tokens) != 1
                or type(event_tokens[0]) is not int
                or not 0 <= event_tokens[0] < vocab_size
                or not isinstance(content, str)
            ):
                raise RuntimeError("nonterminal streaming token event is invalid")
            token_ids.append(int(event_tokens[0]))
            token_times.append(time.monotonic())
            content_hasher.update(content.encode("utf-8"))
    finished = time.monotonic()
    if len(token_ids) != 128 or len(token_times) != 128 or final_timings is None:
        raise RuntimeError("streaming completion did not return exactly 128 token events")
    ttft = token_times[0] - started
    inter_token = tuple(later - earlier for earlier, later in pairwise(token_times))
    if (
        ttft <= 0
        or not math.isfinite(ttft)
        or any(value < 0 or not math.isfinite(value) for value in inter_token)
    ):
        raise RuntimeError("streaming inter-token latency samples are not finite and non-negative")
    latency = summarize_latency_ms(tuple(value * 1000.0 for value in inter_token))
    prompt_n = final_timings.get("prompt_n")
    predicted_n = final_timings.get("predicted_n")
    prompt_ms = final_timings.get("prompt_ms")
    predicted_ms = final_timings.get("predicted_ms")
    prompt_tps = final_timings.get("prompt_per_second")
    decode_tps = final_timings.get("predicted_per_second")
    if (
        type(prompt_n) is not int
        or prompt_n != 512
        or type(predicted_n) is not int
        or predicted_n != 128
        or isinstance(prompt_ms, bool)
        or not isinstance(prompt_ms, (int, float))
        or not math.isfinite(prompt_ms)
        or prompt_ms <= 0
        or isinstance(predicted_ms, bool)
        or not isinstance(predicted_ms, (int, float))
        or not math.isfinite(predicted_ms)
        or predicted_ms <= 0
        or isinstance(prompt_tps, bool)
        or not isinstance(prompt_tps, (int, float))
        or not math.isfinite(prompt_tps)
        or prompt_tps <= 0
        or isinstance(decode_tps, bool)
        or not isinstance(decode_tps, (int, float))
        or not math.isfinite(decode_tps)
        or decode_tps <= 0
    ):
        raise RuntimeError("streaming completion timings differ from the workload")
    return {
        "request_body_sha256": request_body_sha256,
        "token_ids": token_ids,
        "output_sha256": content_hasher.hexdigest(),
        "response_sha256": response_hasher.hexdigest(),
        "request_started_monotonic_seconds": started,
        "first_token_monotonic_seconds": token_times[0],
        "last_token_monotonic_seconds": token_times[-1],
        "request_finished_monotonic_seconds": finished,
        "wall_seconds": finished - started,
        "ttft_seconds": ttft,
        "prompt_n": prompt_n,
        "predicted_n": predicted_n,
        "prompt_ms": float(prompt_ms),
        "predicted_ms": float(predicted_ms),
        "prompt_tokens_per_second": float(prompt_tps),
        "decode_tokens_per_second": float(decode_tps),
        "inter_token_latency_p50_seconds": latency.p50_ms / 1000.0,
        "inter_token_latency_p95_seconds": latency.p95_ms / 1000.0,
        "inter_token_latency_p99_seconds": latency.p99_ms / 1000.0,
        "raw_inter_token_latency_seconds": list(inter_token),
        "prompt_text_recorded": False,
        "output_text_recorded": False,
    }


def _run_stream_batch(
    *,
    port: int,
    prompt_tokens: Sequence[int],
    vocab_size: int,
    concurrency: int,
    work_deadline: float,
) -> dict[str, Any]:
    barrier = threading.Barrier(concurrency)
    batch_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _stream_completion,
                port=port,
                prompt_tokens=prompt_tokens,
                vocab_size=vocab_size,
                barrier=barrier,
                work_deadline=work_deadline,
            )
            for _ in range(concurrency)
        ]
        requests = tuple(
            {
                "request_index": request_index,
                **future.result(),
            }
            for request_index, future in enumerate(futures, start=1)
        )
    batch_finished = time.monotonic()
    decode_started = min(float(item["first_token_monotonic_seconds"]) for item in requests)
    decode_finished = max(float(item["last_token_monotonic_seconds"]) for item in requests)
    decode_window = decode_finished - decode_started
    if decode_window <= 0.0 or not math.isfinite(decode_window):
        raise RuntimeError("concurrent decode window is not finite and positive")
    return {
        "concurrency": concurrency,
        "batch_started_monotonic_seconds": batch_started,
        "batch_finished_monotonic_seconds": batch_finished,
        "batch_wall_seconds": batch_finished - batch_started,
        "decode_boundary": ("earliest_first_token_to_latest_last_token_127_intervals_per_request"),
        "aggregate_decode_token_intervals": 127 * concurrency,
        "batch_duration_seconds": decode_window,
        "aggregate_decode_tokens_per_second": 127.0 * concurrency / decode_window,
        "requests": list(requests),
    }


def _benchmark_cases(
    *,
    subject: SubjectSpec,
    bundle: InklingMeasurementBundle,
    expected_uuids: Sequence[str],
    work_deadline: float,
) -> BenchmarkMeasurementResult:
    topology = bind_exact_cuda_topology(
        tuple(f"CUDA{index}" for index in range(8)),
        (1,) * 8,
    )
    spec = LlamaBenchCommandSpec(
        model_path=subject.model_path,
        repetitions=5,
        batch_size=2_048,
        ubatch_size=512,
        threads=16,
        topology=topology,
    )
    command = build_llama_bench_command(spec)
    environment = build_matched_server_environment(
        os.environ,
        audit_environment=SERVER_AUDIT_ENVIRONMENT,
    )
    run = _run_captured(
        command,
        expected_uuids=expected_uuids,
        timeout=7_200,
        work_deadline=work_deadline,
        environment=environment,
    )
    parsed = parse_llama_bench_jsonl(run.stdout, spec=spec)
    records = [
        {
            "case": name,
            **asdict(result),
        }
        for name, result in zip(BENCH_CASES, parsed, strict=True)
    ]
    if tuple(item["case"] for item in records) != BENCH_CASES:
        raise RuntimeError("llama-bench cases are incomplete or out of order")
    combined = run.stdout + CAPTURED_TOOL_LOG_DELIMITER + run.stderr
    placement = parse_exact_text_cuda_backend_audit(
        combined,
        policy=build_matched_cuda_placement_policy(bundle.matched.config),
    )
    return BenchmarkMeasurementResult(
        evidence={
            "command": list(run.command),
            "process_id": run.process_id,
            "process_started_monotonic_seconds": run.started_monotonic_seconds,
            "process_finished_monotonic_seconds": run.finished_monotonic_seconds,
            "cases": records,
            "stdout_sha256": _sha256_bytes(run.stdout.encode("utf-8")),
            "stdout_size_bytes": len(run.stdout.encode("utf-8")),
            "stderr_sha256": _sha256_bytes(run.stderr.encode("utf-8")),
            "stderr_size_bytes": len(run.stderr.encode("utf-8")),
            "elapsed_seconds": run.elapsed_seconds,
            "telemetry": run.telemetry,
            "placement_audit": placement.model_dump(mode="json"),
            "warmup_enabled": True,
            "single_model_load": True,
        },
        stdout=run.stdout,
        stderr=run.stderr,
    )


def _condition_cold_server_files(subject: SubjectSpec) -> dict[str, Any]:
    """Apply the reviewed file-level cache advice to all executable GGUF files."""

    gguf_artifacts = tuple(
        (path, size_bytes) for path, _, size_bytes in subject.artifacts if path.endswith(".gguf")
    )
    expected_paths = (*subject.shard_paths, subject.projector_path)
    if (
        len(gguf_artifacts) != 50
        or tuple(path for path, _ in gguf_artifacts) != expected_paths
        or len(set(expected_paths)) != 50
    ):
        raise RuntimeError("cold-cache conditioning requires 49 ordered shards then one projector")
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if not callable(fadvise) or type(dontneed) is not int:
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable in the Modal runtime")
    advised_bytes = 0
    for path_text, expected_size in gguf_artifacts:
        path = Path(path_text)
        if (
            not path.is_absolute()
            or "\\" in path_text
            or PurePosixPath(path_text).as_posix() != path_text
        ):
            raise RuntimeError("cold-cache conditioning path is not canonical POSIX")
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != expected_size:
                raise RuntimeError("cold-cache conditioning artifact identity changed")
            fadvise(descriptor, 0, 0, dontneed)
        finally:
            os.close(descriptor)
        advised_bytes += expected_size
    return {
        "schema_version": "inkling-measurement-cold-cache-conditioning-v1",
        "method": ("file_level_posix_fadvise_posix_fadv_dontneed_on_all_staged_gguf_files"),
        "advice": "POSIX_FADV_DONTNEED",
        "staged_paths": list(expected_paths),
        "artifact_count": 50,
        "advised_bytes": advised_bytes,
        "completed_monotonic_seconds": time.monotonic(),
        "all_advice_calls_succeeded": True,
        "global_cache_flush_claimed": False,
    }


def _server_load_observation(
    server: ServerProcess,
    *,
    subject: SubjectSpec,
    log: str,
    process_finished_monotonic_seconds: float,
) -> dict[str, Any]:
    """Bind one server readiness interval to its log and loader evidence."""

    log_bytes = log.encode("utf-8", errors="strict")
    if not (server.started_monotonic < server.ready_monotonic < process_finished_monotonic_seconds):
        raise RuntimeError("server load process boundaries are not strictly increasing")
    if not log_bytes:
        raise RuntimeError("server load produced an empty log")
    loader_offload = parse_loader_offload_evidence(
        log,
        expected_gpu_count=8,
    )
    artifact_load = parse_artifact_load_evidence(
        log,
        expected_first_shard_path=subject.model_path,
        expected_projector_path=subject.projector_path,
    )
    return {
        "command": list(server.command),
        "process_id": server.process.pid,
        "process_started_monotonic_seconds": server.started_monotonic,
        "server_ready_monotonic_seconds": server.ready_monotonic,
        "process_finished_monotonic_seconds": process_finished_monotonic_seconds,
        "process_load_seconds": server.ready_monotonic - server.started_monotonic,
        "log": log,
        "log_size_bytes": len(log_bytes),
        "log_sha256": _sha256_bytes(log_bytes),
        "loader_offload": loader_offload.model_dump(mode="json"),
        "artifact_load": artifact_load.model_dump(mode="json"),
    }


def _server_load_pair_trial(
    *,
    trial_index: int,
    cold_cache_conditioning: Mapping[str, Any],
    cold: Mapping[str, Any],
    warm: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one ordered cache-conditioned cold then unconditioned warm pair."""

    restart_gap = float(warm["process_started_monotonic_seconds"]) - float(
        cold["process_finished_monotonic_seconds"]
    )
    if restart_gap < 0.0 or not math.isfinite(restart_gap):
        raise RuntimeError("cold-to-warm server restart gap is invalid")
    return {
        "trial_index": trial_index,
        "cold_cache_conditioning": dict(cold_cache_conditioning),
        "cold": dict(cold),
        "warm": dict(warm),
        "warm_load_is_next_model_load_after_cold": True,
        "explicit_cache_conditioning_or_eviction_requested_between_loads": False,
        "cold_to_warm_restart_gap_seconds": restart_gap,
    }


def _repeated_load_summary(
    load_pair_trials: Sequence[Mapping[str, Any]],
    *,
    temperature: Literal["cold", "warm"],
) -> dict[str, Any]:
    """Derive the exact ordered duration summary from retained load trials."""

    durations = tuple(
        float(cast("Mapping[str, Any]", trial[temperature])["process_load_seconds"])
        for trial in load_pair_trials
    )
    if len(durations) < 2:
        raise RuntimeError("repeated server load summary requires at least two trials")
    return {
        "trial_count": len(durations),
        "durations_seconds": list(durations),
        "median_seconds": float(statistics.median(durations)),
        "sample_standard_deviation_seconds": float(statistics.stdev(durations)),
    }


def _server_measurement(
    *,
    subject: SubjectSpec,
    bundle: InklingMeasurementBundle,
    expected_uuids: Sequence[str],
    hardware_identity_sha256: str,
    port: int,
    work_deadline: float,
) -> ServerMeasurementResult:
    if bundle.config.quality.diagnostic_repetitions != 1:
        raise RuntimeError("diagnostics must run exactly once per subject")
    if SHA256_RE.fullmatch(hardware_identity_sha256) is None:
        raise RuntimeError("cold load lacks the accepted hardware identity")
    load_pair_repetitions = bundle.config.performance.server.load_pair_repetitions
    if load_pair_repetitions != 3:
        raise RuntimeError("server protocol requires exactly three cold/warm load pairs")

    load_pair_trials: list[dict[str, Any]] = []
    cold_cache_conditioning: dict[str, Any] | None = None
    cold_load: dict[str, Any] | None = None
    selected_cold_observation: dict[str, Any] | None = None
    server: ServerProcess | None = None
    for trial_index in range(1, load_pair_repetitions + 1):
        conditioning = _condition_cold_server_files(subject)
        if load_pair_trials and float(conditioning["completed_monotonic_seconds"]) <= float(
            cast("Mapping[str, Any]", load_pair_trials[-1]["warm"])[
                "process_finished_monotonic_seconds"
            ]
        ):
            raise RuntimeError("next cold conditioning did not follow prior warm termination")
        cold_server = _start_server(
            subject=subject,
            projector_path=subject.projector_path,
            port=port,
            parallel_slots=4,
            expected_uuids=expected_uuids,
            work_deadline=work_deadline,
        )
        cold_log, _, cold_finished = _stop_server(cold_server)

        # This is deliberately the first model-load operation after the cold
        # process terminates. No cache advice is issued between pair members.
        warm_server = _start_server(
            subject=subject,
            projector_path=subject.projector_path,
            port=port,
            parallel_slots=4,
            expected_uuids=expected_uuids,
            work_deadline=work_deadline,
        )
        warm_stopped = False
        try:
            if warm_server.command != cold_server.command:
                raise RuntimeError("cold and warm server commands differ")
            if warm_server.process.pid == cold_server.process.pid:
                raise RuntimeError("cold and warm server processes are not distinct")
            cold_observation = _server_load_observation(
                cold_server,
                subject=subject,
                log=cold_log,
                process_finished_monotonic_seconds=cold_finished,
            )
            if trial_index == load_pair_repetitions:
                cold_log_bytes = cold_log.encode("utf-8", errors="strict")
                cold_cache_conditioning = conditioning
                selected_cold_observation = cold_observation
                cold_load = {
                    "schema_version": "inkling-measurement-cold-server-load-v1",
                    "command": list(cold_server.command),
                    "process_id": cold_server.process.pid,
                    "process_started_monotonic_seconds": cold_server.started_monotonic,
                    "server_ready_monotonic_seconds": cold_server.ready_monotonic,
                    "process_finished_monotonic_seconds": cold_finished,
                    "cold_server_process_load_seconds": (
                        cold_server.ready_monotonic - cold_server.started_monotonic
                    ),
                    "hardware_identity_sha256": hardware_identity_sha256,
                    "readiness_only": True,
                    "generation_requests_executed": 0,
                    "log": cold_log,
                    "log_size_bytes": len(cold_log_bytes),
                    "log_sha256": _sha256_bytes(cold_log_bytes),
                    "loader_offload": cold_observation["loader_offload"],
                    "artifact_load": cold_observation["artifact_load"],
                }
                server = warm_server
                break

            warm_log, _, warm_finished = _stop_server(warm_server)
            warm_stopped = True
            warm_observation = _server_load_observation(
                warm_server,
                subject=subject,
                log=warm_log,
                process_finished_monotonic_seconds=warm_finished,
            )
            load_pair_trials.append(
                _server_load_pair_trial(
                    trial_index=trial_index,
                    cold_cache_conditioning=conditioning,
                    cold=cold_observation,
                    warm=warm_observation,
                )
            )
        except BaseException:
            if not warm_stopped:
                with suppress(BaseException):
                    _stop_server(warm_server)
            raise
    if (
        server is None
        or cold_cache_conditioning is None
        or cold_load is None
        or selected_cold_observation is None
    ):
        raise RuntimeError("final warm workload server was not retained")

    benchmark_error: BaseException | None = None
    diagnostics: tuple[dict[str, Any], ...] = ()
    server_trials: list[dict[str, Any]] = []
    cell_windows: list[tuple[int, float, float]] = []
    single_warmups: list[dict[str, Any]] = []
    prompt_tokens: tuple[int, ...] = ()
    vocab_size = 0
    try:
        vocab_size, marker = _server_contract(
            port,
            work_deadline=work_deadline,
        )
        diagnostics = _run_diagnostics(
            port=port,
            marker=marker,
            vocab_size=vocab_size,
            bundle=bundle,
            work_deadline=work_deadline,
        )
        prompt_tokens = _exact_prompt_tokens(
            port,
            work_deadline=work_deadline,
        )
        # Two single-request warmups are required before concurrency-specific
        # warmups and are never included in measured trial summaries.
        single_warmups = [
            {
                "warmup_index": index,
                **_stream_completion(
                    port=port,
                    prompt_tokens=prompt_tokens,
                    vocab_size=vocab_size,
                    barrier=None,
                    work_deadline=work_deadline,
                ),
            }
            for index in range(1, 3)
        ]
        for concurrency in (1, 2, 4):
            concurrent_warmup = {
                "batch_index": 0,
                **_run_stream_batch(
                    port=port,
                    prompt_tokens=prompt_tokens,
                    vocab_size=vocab_size,
                    concurrency=concurrency,
                    work_deadline=work_deadline,
                ),
            }
            measured_started = time.monotonic()
            measured = [
                {
                    "batch_index": batch_index,
                    **_run_stream_batch(
                        port=port,
                        prompt_tokens=prompt_tokens,
                        vocab_size=vocab_size,
                        concurrency=concurrency,
                        work_deadline=work_deadline,
                    ),
                }
                for batch_index in range(1, 6)
            ]
            measured_finished = time.monotonic()
            cell_windows.append((concurrency, measured_started, measured_finished))
            flattened = [
                request
                for batch in measured
                for request in cast("list[dict[str, Any]]", batch["requests"])
            ]
            raw_intervals = [
                float(interval)
                for request in flattened
                for interval in cast(
                    "list[float]",
                    request["raw_inter_token_latency_seconds"],
                )
            ]
            if len(raw_intervals) != 5 * concurrency * 127:
                raise RuntimeError("server cell did not retain every measured inter-token interval")
            latency = summarize_latency_ms(tuple(value * 1000.0 for value in raw_intervals))
            aggregate_decode = [
                float(batch["aggregate_decode_tokens_per_second"]) for batch in measured
            ]
            if len(aggregate_decode) != 5 or any(
                value <= 0.0 or not math.isfinite(value) for value in aggregate_decode
            ):
                raise RuntimeError("server cell aggregate decode trials are invalid")
            server_trials.append(
                {
                    "concurrency": concurrency,
                    "single_request_warmups_completed": (
                        len(single_warmups) if concurrency == 1 else 0
                    ),
                    "concurrent_batch_warmup_completed": True,
                    "concurrent_batch_warmup": concurrent_warmup,
                    "warmup_output_token_counts": (
                        [len(cast("list[int]", item["token_ids"])) for item in single_warmups]
                        if concurrency == 1
                        else []
                    ),
                    "concurrent_warmup_request_count": len(
                        cast("list[dict[str, Any]]", concurrent_warmup["requests"])
                    ),
                    "measured_batches": measured,
                    "measured_request_count": len(flattened),
                    "mean_ttft_seconds": sum(float(item["ttft_seconds"]) for item in flattened)
                    / len(flattened),
                    "mean_prompt_tokens_per_second": sum(
                        float(item["prompt_tokens_per_second"]) for item in flattened
                    )
                    / len(flattened),
                    "mean_decode_tokens_per_second": sum(
                        float(item["decode_tokens_per_second"]) for item in flattened
                    )
                    / len(flattened),
                    "aggregate_decode_tokens_per_second_trials": aggregate_decode,
                    "mean_aggregate_decode_tokens_per_second": (
                        sum(aggregate_decode) / len(aggregate_decode)
                    ),
                    "inter_token_latency_method": (
                        "r7_linear_interpolation_over_all_measured_request_intervals"
                    ),
                    "raw_inter_token_interval_count": len(raw_intervals),
                    "inter_token_latency_p50_seconds": latency.p50_ms / 1000.0,
                    "inter_token_latency_p95_seconds": latency.p95_ms / 1000.0,
                    "inter_token_latency_p99_seconds": latency.p99_ms / 1000.0,
                }
            )
    except BaseException as error:
        benchmark_error = error
    server_log, telemetry, process_finished = _stop_server(server)
    if benchmark_error is not None:
        raise benchmark_error
    warm_observation = _server_load_observation(
        server,
        subject=subject,
        log=server_log,
        process_finished_monotonic_seconds=process_finished,
    )
    selected_load_pair = _server_load_pair_trial(
        trial_index=load_pair_repetitions,
        cold_cache_conditioning=cold_cache_conditioning,
        cold=selected_cold_observation,
        warm=warm_observation,
    )
    load_pair_trials.append(selected_load_pair)
    if len(load_pair_trials) != load_pair_repetitions:
        raise RuntimeError("server load-pair trials are incomplete")
    cold_server_load_trials = _repeated_load_summary(
        load_pair_trials,
        temperature="cold",
    )
    warm_server_load_trials = _repeated_load_summary(
        load_pair_trials,
        temperature="warm",
    )
    if tuple(item["concurrency"] for item in server_trials) != SERVER_CONCURRENCIES:
        raise RuntimeError("server benchmark concurrency cells are incomplete")
    placement = parse_exact_cuda_backend_audit(
        server_log,
        policy=build_matched_cuda_placement_policy(bundle.matched.config),
    )
    for cell, (concurrency, started, finished) in zip(
        server_trials,
        cell_windows,
        strict=True,
    ):
        if cell["concurrency"] != concurrency:
            raise RuntimeError("server resource window order differs from cells")
        cell["resource_sample_summary"] = _telemetry_window(
            telemetry,
            started_monotonic=started,
            finished_monotonic=finished,
        )
    return ServerMeasurementResult(
        diagnostics=diagnostics,
        evidence={
            "schema_version": "inkling-measurement-subject-server-v1",
            "subject": subject.name,
            "load_pair_repetitions": load_pair_repetitions,
            "load_pair_trial_scope": (
                "process_start_to_readiness_for_ordered_same_artifact_cold_then_warm_pairs"
            ),
            "load_pair_trials": load_pair_trials,
            "cold_server_load_trials": cold_server_load_trials,
            "warm_server_load_trials": warm_server_load_trials,
            "workload_load_pair_trial_index": load_pair_repetitions,
            "cold_cache_conditioning": cold_cache_conditioning,
            "cold_load": cold_load,
            "warm_load_is_next_model_load_after_cold": True,
            "explicit_cache_conditioning_or_eviction_requested_between_server_loads": (False),
            "cold_to_warm_restart_gap_seconds": (
                selected_load_pair["cold_to_warm_restart_gap_seconds"]
            ),
            "command": list(server.command),
            "process_id": server.process.pid,
            "process_started_monotonic_seconds": server.started_monotonic,
            "server_ready_monotonic_seconds": server.ready_monotonic,
            "process_finished_monotonic_seconds": process_finished,
            "warm_server_process_load_seconds": (server.ready_monotonic - server.started_monotonic),
            "vocab_size": vocab_size,
            "diagnostic_items_completed_before_performance": len(diagnostics),
            "diagnostic_repetitions": 1,
            "single_request_warmups": single_warmups,
            "prompt_token_ids": list(prompt_tokens),
            "prompt_token_ids_sha256": _sha256_bytes(
                canonical_measurement_json_bytes(list(prompt_tokens))
            ),
            "prompt_token_count": len(prompt_tokens),
            "output_tokens": 128,
            "seed": 42,
            "temperature": 0.0,
            "streaming": True,
            "cache_prompt": False,
            "return_tokens": True,
            "ignore_eos": True,
            "request_body_sha256": _sha256_bytes(
                canonical_measurement_raw_json_bytes(
                    {
                        "prompt": list(prompt_tokens),
                        "seed": 42,
                        "temperature": 0.0,
                        "n_predict": 128,
                        "stream": True,
                        "cache_prompt": False,
                        "return_tokens": True,
                        "ignore_eos": True,
                    }
                )
            ),
            "concurrency": server_trials,
            "telemetry": telemetry,
            "log_sha256": _sha256_bytes(server_log.encode("utf-8")),
            "log_size_bytes": len(server_log.encode("utf-8")),
            "placement_audit": placement.model_dump(mode="json"),
            "prompt_text_recorded": False,
            "output_text_recorded": False,
        },
        log=server_log,
    )


def _performance_measurement(
    *,
    subject: SubjectSpec,
    bundle: InklingMeasurementBundle,
    expected_uuids: Sequence[str],
    server_evidence: Mapping[str, Any],
    work_deadline: float,
) -> PerformanceMeasurementResult:
    bench = _benchmark_cases(
        subject=subject,
        bundle=bundle,
        expected_uuids=expected_uuids,
        work_deadline=work_deadline,
    )
    return PerformanceMeasurementResult(
        evidence={
            "schema_version": "inkling-measurement-subject-performance-v1",
            "subject": subject.name,
            "llama_bench": bench.evidence,
            "server": dict(server_evidence),
            "prompt_text_recorded": False,
            "output_text_recorded": False,
        },
        benchmark=bench,
    )


def _project_model_fields(
    model_type: type[BaseModel],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Select exactly one evidence model's declared fields from a raw result."""

    missing = tuple(name for name in model_type.model_fields if name not in source)
    if missing:
        raise RuntimeError(f"{model_type.__name__} source lacks fields: {', '.join(missing)}")
    return {name: source[name] for name in model_type.model_fields}


def _raw_attempt_bindings(
    bindings: MeasurementEvidenceBindings,
) -> MeasurementAttemptBindings:
    return MeasurementAttemptBindings.model_validate(bindings.fields())


def _resource_telemetry_payload(
    measurement: SubjectMeasurementResult,
    *,
    bindings: MeasurementAttemptBindings,
) -> bytes:
    sources = (
        (
            "perplexity",
            measurement.quality.perplexity.evidence,
        ),
        (
            "server_quality_and_performance",
            measurement.server.evidence,
        ),
        (
            "llama_bench",
            measurement.performance.benchmark.evidence,
        ),
    )
    rows: list[bytes] = []
    for workload, evidence in sources:
        process_id = evidence.get("process_id")
        telemetry = evidence.get("telemetry")
        if type(process_id) is not int or not isinstance(telemetry, Mapping):
            raise RuntimeError(f"{workload} lacks process-bound resource telemetry")
        samples = telemetry.get("samples")
        if not isinstance(samples, list) or not samples:
            raise RuntimeError(f"{workload} lacks raw resource samples")
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                raise RuntimeError(f"{workload} resource sample is not an object")
            rows.append(
                canonical_measurement_raw_json_bytes(
                    {
                        "schema_version": ("inkling-measurement-resource-telemetry-row-v1"),
                        "bindings": bindings.model_dump(mode="json"),
                        "workload": workload,
                        "process_id": process_id,
                        "sample_index": sample_index,
                        "requested_sampling_interval_seconds": (
                            sample["requested_sampling_interval_seconds"]
                        ),
                        "sampled_at_monotonic_seconds": (sample["sampled_at_monotonic_seconds"]),
                        "host_rss_bytes": sample["host_rss_bytes"],
                        "gpus": sample["gpus"],
                    }
                )
            )
    payload = b"".join(rows)
    parse_resource_telemetry_evidence(payload)
    return payload


def _captured_audit_log(stdout: str, stderr: str) -> str:
    return stdout + CAPTURED_TOOL_LOG_DELIMITER + stderr


def _backend_audit_payload(
    measurement: SubjectMeasurementResult,
    *,
    bindings: MeasurementAttemptBindings,
) -> bytes:
    perplexity_log = _captured_audit_log(
        measurement.quality.perplexity.stdout,
        measurement.quality.perplexity.stderr,
    )
    benchmark_log = _captured_audit_log(
        measurement.performance.benchmark.stdout,
        measurement.performance.benchmark.stderr,
    )
    sources = (
        (
            "perplexity",
            measurement.quality.perplexity.evidence,
            perplexity_log,
            "captured_stdout_stderr",
            CAPTURED_TOOL_LOG_DELIMITER,
        ),
        (
            "server_quality_and_performance",
            measurement.server.evidence,
            measurement.server.log,
            "combined_server_log",
            None,
        ),
        (
            "llama_bench",
            measurement.performance.benchmark.evidence,
            benchmark_log,
            "captured_stdout_stderr",
            CAPTURED_TOOL_LOG_DELIMITER,
        ),
    )
    workloads: list[dict[str, Any]] = []
    for workload, evidence, log, capture_mode, delimiter in sources:
        process_id = evidence.get("process_id")
        command = evidence.get("command")
        if type(process_id) is not int or not isinstance(command, list):
            raise RuntimeError(f"{workload} lacks its exact process command")
        log_bytes = log.encode("utf-8", errors="strict")
        workloads.append(
            {
                "workload": workload,
                "process_id": process_id,
                "command": command,
                "capture_mode": capture_mode,
                "stdout_stderr_delimiter": delimiter,
                "log": log,
                "log_size_bytes": len(log_bytes),
                "log_sha256": _sha256_bytes(log_bytes),
            }
        )
    raw_evidence = {
        "schema_version": "inkling-measurement-backend-audit-v1",
        "bindings": bindings.model_dump(mode="json"),
        "workloads": workloads,
    }
    audit_evidence = MeasurementBackendAuditEvidence.model_validate_json(
        canonical_measurement_raw_json_bytes(raw_evidence),
        strict=True,
    )
    payload = canonical_measurement_raw_json_bytes(audit_evidence.model_dump(mode="json"))
    parse_backend_audit_evidence(payload)
    return payload


def _raw_trials_payload(
    measurement: SubjectMeasurementResult,
    *,
    bindings: MeasurementAttemptBindings,
    hardware: Mapping[str, Any],
) -> bytes:
    raw_evidence = {
        "schema_version": "inkling-measurement-raw-trials-v1",
        "bindings": bindings.model_dump(mode="json"),
        "hardware_identity": dict(hardware),
        "staging": measurement.staging,
        "perplexity": _project_model_fields(
            MeasurementPerplexityTrial,
            measurement.quality.perplexity.evidence,
        ),
        "diagnostics": list(measurement.server.diagnostics),
        "llama_bench": _project_model_fields(
            MeasurementLlamaBenchTrials,
            measurement.performance.benchmark.evidence,
        ),
        "server": _project_model_fields(
            MeasurementServerTrials,
            measurement.server.evidence,
        ),
        "prompt_text_recorded": False,
        "output_text_recorded": False,
    }
    evidence = MeasurementRawTrialsEvidence.model_validate_json(
        canonical_measurement_raw_json_bytes(raw_evidence),
        strict=True,
    )
    payload = canonical_measurement_raw_json_bytes(evidence.model_dump(mode="json"))
    parse_raw_trials_evidence(payload)
    return payload


def _executable_artifact_inventory(
    raw_trials: MeasurementRawTrialsEvidence,
) -> tuple[MeasurementExecutableArtifactIdentity, ...]:
    staging = raw_trials.staging.artifacts
    projector = tuple(
        item for item in staging if PurePosixPath(item.source_path).name == "mmproj-BF16.gguf"
    )
    if len(projector) != 1:
        raise RuntimeError("staged subject lacks one exact executable projector")
    executable = (*staging[:49], projector[0])
    return tuple(
        MeasurementExecutableArtifactIdentity(
            ordinal=ordinal,
            role=("text_shard" if ordinal < 49 else "multimodal_projector"),
            source_path=item.source_path,
            staged_path=item.staged_path,
            sha256=item.sha256,
            size_bytes=item.size_bytes,
        )
        for ordinal, item in enumerate(executable)
    )


def _publish_subject_evidence(
    measurement: SubjectMeasurementResult,
    *,
    bundle: InklingMeasurementBundle,
    binding: InvocationBinding,
    runtime: MeasurementRuntimeIdentity,
    hardware: Mapping[str, Any],
) -> PublishedSubjectEvidence:
    subject = measurement.source_subject.name
    bindings = _measurement_evidence_bindings(
        binding,
        bundle=bundle,
        subject=subject,
    )
    raw_bindings = _raw_attempt_bindings(bindings)
    token_nll_payload = measurement.quality.perplexity.token_nll_payload
    raw_trials_payload = _raw_trials_payload(
        measurement,
        bindings=raw_bindings,
        hardware=hardware,
    )
    telemetry_payload = _resource_telemetry_payload(
        measurement,
        bindings=raw_bindings,
    )
    backend_payload = _backend_audit_payload(
        measurement,
        bindings=raw_bindings,
    )
    token_nll = parse_token_nll_raw_evidence(token_nll_payload)
    raw_trials = parse_raw_trials_evidence(raw_trials_payload)
    telemetry = parse_resource_telemetry_evidence(telemetry_payload)
    backend_audit = parse_backend_audit_evidence(backend_payload)
    validate_measurement_raw_evidence_links(
        token_nll,
        raw_trials,
        telemetry,
        backend_audit,
    )
    validate_measurement_diagnostic_evidence(
        bundle.diagnostic_items,
        prompt_template=bundle.config.quality.prompt_template,
        raw_trials=raw_trials,
    )
    quality = recompute_subject_quality_summary(token_nll, raw_trials)
    performance = recompute_subject_performance_summary(raw_trials)
    initial_pairing = recompute_pairing_projection_hashes(token_nll, raw_trials)

    raw_payload_by_kind = (
        ("token_nll", token_nll_payload),
        ("raw_trials", raw_trials_payload),
        ("resource_telemetry", telemetry_payload),
        ("backend_audit", backend_payload),
    )
    raw_references = cast(
        "tuple[MeasurementRawBlobReference, MeasurementRawBlobReference, "
        "MeasurementRawBlobReference, MeasurementRawBlobReference]",
        tuple(
            build_measurement_raw_blob_reference(
                payload,
                run_id=binding.intent.run_id,
                subject=subject,
                kind=cast("Any", kind),
            )
            for kind, payload in raw_payload_by_kind
        ),
    )
    for raw_reference, (_, payload) in zip(
        raw_references,
        raw_payload_by_kind,
        strict=True,
    ):
        validate_measurement_raw_blob_reference(payload, expected=raw_reference)

    placements = build_measurement_placement_summaries(
        backend_audit,
        backend_audit_content_sha256=raw_references[-1].content_sha256,
        policy=build_matched_cuda_placement_policy(bundle.matched.config),
    )
    record = MeasurementSubjectCompactRecord(
        run_id=binding.intent.run_id,
        subject=subject,
        control_plane_sha256=(binding.intent.reviewed_inputs.control_plane.control_plane_sha256),
        reviewed_config_file_sha256=bindings.reviewed_config_file_sha256,
        resolved_config_sha256=bindings.resolved_config_sha256,
        launch_intent_sha256=bindings.launch_intent_sha256,
        post_spawn_acceptance_sha256=(bindings.post_spawn_acceptance_sha256),
        call_id=bindings.call_id,
        attempt_claim_sha256=bindings.attempt_claim_sha256,
        runtime_manifest_sha256=runtime.manifest_sha256,
        hardware_identity_sha256=raw_trials.hardware_identity_sha256,
        model_id=bundle.config.model_id,
        model_revision=bundle.config.revision,
        artifact_inventory=_executable_artifact_inventory(raw_trials),
        protocol_sha256=bindings.protocol_sha256,
        workload_sha256=bindings.workload_sha256,
        raw_blobs=raw_references,
        placement_summaries=placements,
        quality_projection_sha256=(measurement_subject_quality_projection_sha256(quality)),
        performance_projection_sha256=(
            measurement_subject_performance_projection_sha256(performance)
        ),
        prompt_text_recorded=False,
        output_text_recorded=False,
    )
    payload = record.canonical_bytes()
    parsed_record = parse_measurement_subject_compact_record(
        payload,
        run_id=binding.intent.run_id,
        subject=subject,
    )
    kind: Literal["bf16_subject", "q3_subject"] = (
        "bf16_subject" if subject == "bf16" else "q3_subject"
    )
    reference = _supporting_reference(
        run_id=binding.intent.run_id,
        kind=kind,
        payload=payload,
    )
    _validate_supporting_reference(reference, payload)
    publications = {
        raw_reference.relative_path: raw_payload
        for raw_reference, (_, raw_payload) in zip(
            raw_references,
            raw_payload_by_kind,
            strict=True,
        )
    }
    publications[reference.relative_path] = payload
    _commit_and_verify(publications)

    readback_payload = _read_regular_bytes(
        _evidence_path(reference.relative_path),
        maximum_bytes=reference.size_bytes,
    )
    _validate_supporting_reference(reference, readback_payload)
    readback_record = parse_measurement_subject_compact_record(
        readback_payload,
        run_id=binding.intent.run_id,
        subject=subject,
    )
    if readback_record != parsed_record:
        raise RuntimeError("published compact subject differs after commit")
    readback_raw: dict[str, bytes] = {}
    for raw_reference in readback_record.raw_blobs:
        raw_payload = _read_regular_bytes(
            _evidence_path(raw_reference.relative_path),
            maximum_bytes=raw_reference.size_bytes,
        )
        validate_measurement_raw_blob_reference(
            raw_payload,
            expected=raw_reference,
        )
        readback_raw[raw_reference.kind] = raw_payload
    readback_token_nll = parse_token_nll_raw_evidence(readback_raw["token_nll"])
    readback_trials = parse_raw_trials_evidence(readback_raw["raw_trials"])
    readback_telemetry = parse_resource_telemetry_evidence(readback_raw["resource_telemetry"])
    readback_backend = parse_backend_audit_evidence(readback_raw["backend_audit"])
    readback_links = validate_measurement_raw_evidence_links(
        readback_token_nll,
        readback_trials,
        readback_telemetry,
        readback_backend,
    )
    if (
        readback_trials.bindings != raw_bindings
        or readback_links.hardware_identity_sha256 != readback_record.hardware_identity_sha256
    ):
        raise RuntimeError("published subject raw attempt bindings differ")
    validate_measurement_diagnostic_evidence(
        bundle.diagnostic_items,
        prompt_template=bundle.config.quality.prompt_template,
        raw_trials=readback_trials,
    )
    readback_quality = recompute_subject_quality_summary(
        readback_token_nll,
        readback_trials,
    )
    readback_performance = recompute_subject_performance_summary(readback_trials)
    readback_pairing = recompute_pairing_projection_hashes(
        readback_token_nll,
        readback_trials,
    )
    readback_placements = build_measurement_placement_summaries(
        readback_backend,
        backend_audit_content_sha256=(readback_record.raw_blobs[-1].content_sha256),
        policy=build_matched_cuda_placement_policy(bundle.matched.config),
    )
    if (
        readback_record.artifact_inventory != _executable_artifact_inventory(readback_trials)
        or readback_record.placement_summaries != readback_placements
        or readback_record.quality_projection_sha256
        != measurement_subject_quality_projection_sha256(readback_quality)
        or readback_record.performance_projection_sha256
        != measurement_subject_performance_projection_sha256(readback_performance)
        or readback_pairing != initial_pairing
    ):
        raise RuntimeError("published compact subject differs from raw evidence")
    return PublishedSubjectEvidence(
        record=readback_record,
        payload=readback_payload,
        reference=reference,
        raw_payloads=tuple(
            (
                raw_reference.relative_path,
                readback_raw[raw_reference.kind],
            )
            for raw_reference in readback_record.raw_blobs
        ),
        raw_references=readback_record.raw_blobs,
        quality=readback_quality,
        performance=readback_performance,
        pairing=readback_pairing,
    )


def _publish_comparison_evidence(
    *,
    bundle: InklingMeasurementBundle,
    bf16: PublishedSubjectEvidence,
    q3: PublishedSubjectEvidence,
) -> tuple[
    MeasurementComparisonCompactRecord,
    MeasurementSupportingRecordReference,
    MeasurementQualityRollup,
    MeasurementPerformanceRollup,
]:
    validate_pairing_projection_hashes(bf16.pairing, q3.pairing)
    pairing = bf16.pairing
    quality_rollup = build_measurement_quality_rollup(
        bf16.quality,
        q3.quality,
        paired_inputs_validated=True,
    )
    performance_rollup = build_measurement_performance_rollup(
        bf16.performance,
        q3.performance,
        llama_bench_workload_identity=(bundle.config.performance.llama_bench.workload_identity),
        server_workload_identity=bundle.config.performance.server.workload_identity,
        equivalent_trials_validated=True,
    )
    baseline = bf16.record
    record = MeasurementComparisonCompactRecord(
        run_id=baseline.run_id,
        control_plane_sha256=baseline.control_plane_sha256,
        reviewed_config_file_sha256=(baseline.reviewed_config_file_sha256),
        resolved_config_sha256=baseline.resolved_config_sha256,
        launch_intent_sha256=baseline.launch_intent_sha256,
        post_spawn_acceptance_sha256=(baseline.post_spawn_acceptance_sha256),
        call_id=baseline.call_id,
        attempt_claim_sha256=baseline.attempt_claim_sha256,
        runtime_manifest_sha256=baseline.runtime_manifest_sha256,
        hardware_identity_sha256=baseline.hardware_identity_sha256,
        model_id=bundle.config.model_id,
        model_revision=bundle.config.revision,
        protocol_sha256=baseline.protocol_sha256,
        workload_sha256=baseline.workload_sha256,
        subject_records=(bf16.reference, q3.reference),
        raw_blobs=(*bf16.raw_references, *q3.raw_references),
        token_nll_pairing_sha256=pairing.token_nll_pairing_sha256,
        diagnostic_pairing_sha256=pairing.diagnostic_pairing_sha256,
        performance_pairing_sha256=pairing.performance_pairing_sha256,
        quality_rollup_sha256=measurement_quality_rollup_sha256(quality_rollup),
        performance_rollup_sha256=measurement_performance_rollup_sha256(performance_rollup),
        prompt_text_recorded=False,
        output_text_recorded=False,
    )
    payload = record.canonical_bytes()
    parsed = parse_measurement_comparison_compact_record(
        payload,
        run_id=baseline.run_id,
    )
    validate_measurement_comparison_links(
        parsed,
        bf16=bf16.record,
        q3=q3.record,
    )
    reference = _supporting_reference(
        run_id=baseline.run_id,
        kind="comparison",
        payload=payload,
    )
    _validate_supporting_reference(reference, payload)
    _commit_and_verify({reference.relative_path: payload})
    readback = _read_regular_bytes(
        _evidence_path(reference.relative_path),
        maximum_bytes=reference.size_bytes,
    )
    _validate_supporting_reference(reference, readback)
    readback_record = parse_measurement_comparison_compact_record(
        readback,
        run_id=baseline.run_id,
    )
    validate_measurement_comparison_links(
        readback_record,
        bf16=bf16.record,
        q3=q3.record,
    )
    if (
        readback_record != parsed
        or readback_record.token_nll_pairing_sha256 != pairing.token_nll_pairing_sha256
        or readback_record.diagnostic_pairing_sha256 != pairing.diagnostic_pairing_sha256
        or readback_record.performance_pairing_sha256 != pairing.performance_pairing_sha256
        or readback_record.quality_rollup_sha256
        != measurement_quality_rollup_sha256(quality_rollup)
        or readback_record.performance_rollup_sha256
        != measurement_performance_rollup_sha256(performance_rollup)
    ):
        raise RuntimeError("published comparison differs from paired raw evidence")
    return readback_record, reference, quality_rollup, performance_rollup


def _terminal_binding_fields(
    binding: InvocationBinding,
    *,
    completed_at_utc: str,
) -> dict[str, Any]:
    reviewed = binding.intent.reviewed_inputs
    return {
        "run_id": binding.intent.run_id,
        "control_plane_sha256": reviewed.control_plane.control_plane_sha256,
        "reviewed_config_file_sha256": reviewed.measurement_config.sha256,
        "resolved_config_sha256": reviewed.resolved_config_sha256,
        "launch_intent_sha256": binding.intent.intent_sha256(),
        "post_spawn_acceptance_sha256": (binding.acceptance.acceptance_sha256()),
        "call_id": binding.call_id,
        "attempt_claim_sha256": binding.claim_sha256,
        "subject_order": ("bf16", "q3"),
        "prompt_text_recorded": False,
        "output_text_recorded": False,
        "mtp_included": False,
        "mtp_supported": False,
        "single_run_causation_claim_allowed": False,
        "completed_at_utc": completed_at_utc,
    }


def _publish_terminal_receipt(
    receipt: MeasurementSuccessTerminalReceipt | MeasurementFailureTerminalReceipt,
    *,
    outcome: Literal["success", "failure"],
) -> MeasurementTerminalReceiptReference:
    payload = canonical_measurement_json_bytes(receipt.model_dump(mode="json"))
    parsed = parse_measurement_terminal_receipt(
        payload,
        run_id=receipt.run_id,
        outcome=outcome,
    )
    if parsed != receipt:
        raise RuntimeError("terminal receipt changed during canonical parsing")
    reference = build_measurement_terminal_receipt_reference(
        payload,
        evidence_root=EVIDENCE_ROOT.as_posix(),
        run_id=receipt.run_id,
        outcome=outcome,
    )
    validate_measurement_terminal_receipt_reference(
        payload,
        evidence_root=EVIDENCE_ROOT.as_posix(),
        expected=reference,
    )
    _commit_and_verify({reference.relative_path: payload})
    readback = _read_regular_bytes(
        _evidence_path(reference.relative_path),
        maximum_bytes=reference.size_bytes,
    )
    validate_measurement_terminal_receipt_reference(
        readback,
        evidence_root=EVIDENCE_ROOT.as_posix(),
        expected=reference,
    )
    if (
        parse_measurement_terminal_receipt(
            readback,
            run_id=receipt.run_id,
            outcome=outcome,
        )
        != receipt
    ):
        raise RuntimeError("committed terminal receipt differs from exact result")
    return reference


def _complete_stage(
    completed: list[MeasurementStage],
    stage: MeasurementStage,
) -> None:
    if len(completed) >= len(MEASUREMENT_PLANNED_STAGES):
        raise RuntimeError("measurement stage list is already complete")
    if stage != MEASUREMENT_PLANNED_STAGES[len(completed)]:
        raise RuntimeError("measurement stage completion is out of checked order")
    completed.append(stage)


def _is_measurement_stage(value: str) -> TypeGuard[MeasurementStage]:
    """Narrow one checked control-plane stage without duplicating its literals."""

    return value in MEASUREMENT_PLANNED_STAGES


def _failed_subject(
    stage: MeasurementStage,
) -> Literal["bf16", "q3"] | None:
    if stage in {
        "stage_and_rehash_bf16",
        "measure_bf16_quality",
        "measure_bf16_performance",
        "release_bf16",
    }:
        return "bf16"
    if stage in {
        "stage_and_rehash_q3",
        "measure_q3_quality",
        "measure_q3_performance",
        "release_q3",
    }:
        return "q3"
    return None


def _failure_code(error: BaseException) -> str:
    name = type(error).__name__
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    safe = re.sub(r"[^a-z0-9_]", "_", snake).strip("_")
    if not safe or not safe[0].isalpha():
        safe = f"measurement_{safe or 'error'}"
    return safe[:96].rstrip("_")


def _failure_summary_sha256(error: BaseException) -> str:
    summary = {
        "error_type": type(error).__name__[:256],
        "error_message": str(error)[:4096],
    }
    return _sha256_bytes(canonical_measurement_json_bytes(summary))


def _validate_paid_attempt_scope(
    *,
    bundle: InklingMeasurementBundle,
    provenance: MeasurementControlPlaneProvenance,
    intent: MeasurementLaunchIntent,
) -> None:
    deployment = intent.deployment
    config = bundle.config
    expected_resources = MeasurementExecutionResources(
        provider=config.resources.provider,
        gpu_type=config.resources.gpu_type,
        gpu_count=config.resources.gpu_count,
        compute_capability="10.3",
        cpu_cores=config.resources.cpu_cores,
        memory_mib=65_536,
        ephemeral_disk_mib=config.resources.ephemeral_disk_mib,
        startup_timeout_seconds=config.resources.startup_timeout_seconds,
        function_timeout_seconds=config.resources.function_timeout_seconds,
        max_containers=1,
        max_attempts=config.resources.max_attempts,
        network_access=config.execution.network_access,
        cpu_fallback_allowed=config.placement.cpu_fallback,
    )
    if (
        provenance.control_plane_sha256 != _CONTROL_SHA256
        or intent.reviewed_inputs.control_plane != provenance
        or deployment.control_plane_sha256 != _CONTROL_SHA256
        or deployment.app_name != measurement_app_name(_CONTROL_SHA256)
        or deployment.environment_name != "inkling-quant"
        or deployment.function_name != MEASUREMENT_FUNCTION_NAME
        or deployment.attempt_registry_name != MEASUREMENT_ATTEMPT_REGISTRY_NAME
        or deployment.attempt_registry_name != config.storage.attempt_registry
        or deployment.evidence_volume_name != config.storage.evidence_volume
        or config.storage.evidence_mount_path != EVIDENCE_ROOT.as_posix()
        or intent.resources != expected_resources
        or intent.reviewed_inputs.resources != expected_resources
        or intent.reviewed_inputs.resolved_config_sha256 != config.config_hash()
        or intent.subject_order != ("bf16", "q3")
        or not intent.one_atomic_attempt
        or not intent.sequential_same_allocation
        or not intent.fresh_process_per_measurement
        or not intent.rehash_all_subject_files
        or intent.partial_success_allowed
        or not intent.measurement_execution_allowed
        or PLANNED_STAGES != MEASUREMENT_PLANNED_STAGES
    ):
        raise RuntimeError("paid measurement scope differs from the sealed reviewed deployment")

    deployed_function = modal.Function.from_name(
        deployment.app_name,
        MEASUREMENT_FUNCTION_NAME,
        environment_name=deployment.environment_name,
    )
    deployed_function.hydrate()
    if deployed_function.object_id != deployment.function_id:
        raise RuntimeError("running Modal Function differs from the deployment seal")

    evidence_volume.hydrate()
    if evidence_volume.object_id != deployment.evidence_volume_id:
        raise RuntimeError("mounted evidence Volume differs from the deployment seal")


def _validate_remote_diagnostic_provenance(
    expected_sha256: str,
) -> DiagnosticControlPlaneProvenance:
    payload = _read_regular_bytes(
        REMOTE_DIAGNOSTIC_PROVENANCE_PATH,
        maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
    )
    strict_diagnostic_json_object(
        payload,
        maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
    )
    provenance = DiagnosticControlPlaneProvenance.model_validate_json(
        payload,
        strict=True,
    )
    files = {
        item.path: _read_regular_bytes(
            REMOTE_PROJECT_ROOT / item.path,
            maximum_bytes=item.size_bytes,
        )
        for item in provenance.files
    }
    observed = validate_diagnostic_control_plane_provenance(
        payload,
        reviewed_commit_sha=provenance.reviewed_commit_sha,
        reviewed_tree_sha=provenance.reviewed_tree_sha,
        files=files,
        required_paths=tuple(item.path for item in provenance.files),
    )
    if observed.control_plane_sha256 != expected_sha256:
        raise RuntimeError("deployed diagnostic control plane differs from authorization")
    return observed


def _load_diagnostic_intent(
    run_id: str,
    intent_sha256: str,
) -> DiagnosticLaunchIntent:
    relative = diagnostic_launch_intent_path(run_id, intent_sha256)
    payload = _read_regular_bytes(
        _evidence_path(relative),
        maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
    )
    strict_diagnostic_json_object(
        payload,
        maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
    )
    intent = DiagnosticLaunchIntent.model_validate_json(payload, strict=True)
    validate_diagnostic_launch_intent(
        payload,
        expected=intent,
        intent_sha256=intent_sha256,
        evidence_path=relative,
    )
    return intent


def _wait_for_diagnostic_acceptance(
    intent: DiagnosticLaunchIntent,
    *,
    call_id: str,
) -> DiagnosticPostSpawnAcceptance:
    relative = diagnostic_post_spawn_acceptance_path(
        intent.run_id,
        intent.intent_sha256(),
    )
    deadline = time.monotonic() + ACCEPTANCE_TIMEOUT_SECONDS
    while True:
        evidence_volume.reload()
        try:
            payload = _read_regular_bytes(
                _evidence_path(relative),
                maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
            )
            break
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "diagnostic post-spawn acceptance was not published in time"
                ) from None
            time.sleep(0.25)
    strict_diagnostic_json_object(
        payload,
        maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
    )
    raw = DiagnosticPostSpawnAcceptance.model_validate_json(payload, strict=True)
    expected = build_diagnostic_post_spawn_acceptance(
        intent,
        accepted_at_utc=raw.accepted_at_utc,
        call_id=call_id,
    )
    validate_diagnostic_post_spawn_acceptance(
        payload,
        expected=expected,
        acceptance_sha256=raw.acceptance_sha256(),
        evidence_path=relative,
    )
    return raw


def _claim_diagnostic_attempt(
    intent: DiagnosticLaunchIntent,
    acceptance: DiagnosticPostSpawnAcceptance,
    invocation: tuple[str, str, str],
) -> DiagnosticInvocationBinding:
    call_id, input_id, task_id = invocation
    registry = modal.Dict.from_id(intent.deployment.attempt_registry_id)
    registry.hydrate()
    info = registry.info()
    created = cast(object, info.created_at)
    if isinstance(created, datetime):
        if created.tzinfo is None:
            raise RuntimeError("diagnostic attempt registry time has no time zone")
        created_at = created.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    elif isinstance(created, (int, float)) and not isinstance(created, bool):
        created_at = datetime.fromtimestamp(float(created), UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    else:
        raise RuntimeError("diagnostic attempt registry creation time is unavailable")
    if (
        registry.object_id != intent.deployment.attempt_registry_id
        or info.name != DIAGNOSTIC_ATTEMPT_REGISTRY_NAME
        or created_at != intent.deployment.attempt_registry_created_at_utc
    ):
        raise RuntimeError("sealed diagnostic attempt registry identity changed")
    candidate_claim = build_diagnostic_attempt_claim(
        intent,
        acceptance,
        claimed_at_utc=_utc_now(),
        input_id=input_id,
        task_id=task_id,
    )
    claim, resumed = claim_diagnostic_attempt(registry, candidate_claim)
    claim_sha256 = claim.claim_sha256()
    relative = diagnostic_attempt_claim_path(intent.run_id, claim_sha256)
    _commit_and_verify({relative: claim.canonical_bytes()})
    validate_diagnostic_attempt_claim(
        _read_regular_bytes(
            _evidence_path(relative),
            maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
        ),
        expected=claim,
        claim_sha256=claim_sha256,
        evidence_path=relative,
    )
    _require_unused_diagnostic_execution(intent.run_id)
    if resumed:
        print(
            "Adopted the original attempt claim after a provider restart of the same Modal input."
        )
    return DiagnosticInvocationBinding(
        intent=intent,
        acceptance=acceptance,
        claim=claim,
        claim_sha256=claim_sha256,
        call_id=call_id,
        input_id=input_id,
        execution_task_id=task_id,
    )


def _evidence_directory_has_entries(relative: str) -> bool:
    """Inspect one mounted Volume directory without following symbolic links."""

    _, path = _evidence_path_binding(relative)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise RuntimeError("diagnostic evidence boundary directory is unsafe") from error
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise RuntimeError("diagnostic evidence boundary is not a directory")
        return bool(os.listdir(descriptor))
    finally:
        os.close(descriptor)


def _require_unused_diagnostic_execution(run_id: str) -> None:
    """Allow restart only before any server-load or result evidence exists."""

    evidence_volume.reload()
    root = PurePosixPath("runs", run_id, DIAGNOSTIC_STAGE)
    protected_roots = (
        root / "control" / "server-load-claims",
        root / "private" / "raw",
        root / "terminal" / "success",
        root / "terminal" / "failure",
    )
    occupied = tuple(
        path.as_posix()
        for path in protected_roots
        if _evidence_directory_has_entries(path.as_posix())
    )
    if occupied:
        raise RuntimeError(
            "diagnostic execution already crossed its one-server-load or terminal boundary: "
            + ", ".join(occupied)
        )


def _claim_diagnostic_server_load(
    binding: DiagnosticInvocationBinding,
) -> DiagnosticServerLoadClaim:
    """Commit the one durable boundary before llama-server may load Inkling."""

    claim = build_diagnostic_server_load_claim(
        binding.claim,
        load_claimed_at_utc=_utc_now(),
        input_id=binding.input_id,
        execution_task_id=binding.execution_task_id,
    )
    relative = diagnostic_server_load_claim_path(
        binding.intent.run_id,
        binding.claim_sha256,
    )
    payload = claim.canonical_bytes()
    _commit_and_verify({relative: payload})
    return validate_diagnostic_server_load_claim(
        _read_regular_bytes(
            _evidence_path(relative),
            maximum_bytes=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
        ),
        expected=claim,
        evidence_path=relative,
    )


def _validate_diagnostic_paid_attempt_scope(
    *,
    bundle: InklingBF16InterfaceDiagnosticBundle,
    provenance: DiagnosticControlPlaneProvenance,
    intent: DiagnosticLaunchIntent,
) -> None:
    deployment = intent.deployment
    config = bundle.config
    if (
        provenance.control_plane_sha256 != _CONTROL_SHA256
        or intent.reviewed_inputs.control_plane != provenance
        or deployment.control_plane_sha256 != _CONTROL_SHA256
        or deployment.app_name != diagnostic_app_name(_CONTROL_SHA256)
        or deployment.environment_name != "inkling-quant"
        or deployment.function_name != DIAGNOSTIC_FUNCTION_NAME
        or deployment.attempt_registry_name != DIAGNOSTIC_ATTEMPT_REGISTRY_NAME
        or deployment.attempt_registry_name != config.storage.attempt_registry
        or deployment.evidence_volume_name != config.storage.evidence_volume
        or config.storage.evidence_mount_path != EVIDENCE_ROOT.as_posix()
        or intent.resources != config.resources
        or intent.reviewed_inputs.resources != config.resources
        or intent.reviewed_inputs.resolved_config_sha256 != config.config_hash()
        or intent.subject != "bf16"
        or not intent.one_atomic_attempt
        or not intent.one_server_load
        or intent.sequential_request_count != 16
        or not intent.rehash_all_subject_files
        or intent.partial_success_allowed
        or not intent.diagnostic_execution_allowed
        or config.execution.planned_stages != DIAGNOSTIC_PLANNED_STAGES
    ):
        raise RuntimeError("paid diagnostic scope differs from the sealed reviewed deployment")

    deployed_function = modal.Function.from_name(
        deployment.app_name,
        DIAGNOSTIC_FUNCTION_NAME,
        environment_name=deployment.environment_name,
    )
    deployed_function.hydrate()
    if deployed_function.object_id != deployment.function_id:
        raise RuntimeError("running diagnostic Function differs from the deployment seal")

    evidence_volume.hydrate()
    if evidence_volume.object_id != deployment.evidence_volume_id:
        raise RuntimeError("mounted diagnostic evidence Volume differs from deployment seal")


def _diagnostic_binding_fields(
    binding: DiagnosticInvocationBinding,
    *,
    completed_at_utc: str,
    server_load_claim: DiagnosticServerLoadClaim | None,
) -> dict[str, Any]:
    reviewed = binding.intent.reviewed_inputs
    return {
        "run_id": binding.intent.run_id,
        "control_plane_sha256": reviewed.control_plane.control_plane_sha256,
        "reviewed_config_file_sha256": reviewed.diagnostic_config.sha256,
        "resolved_config_sha256": reviewed.resolved_config_sha256,
        "launch_intent_sha256": binding.intent.intent_sha256(),
        "post_spawn_acceptance_sha256": binding.acceptance.acceptance_sha256(),
        "call_id": binding.call_id,
        "input_id": binding.input_id,
        "execution_task_id": (binding.execution_task_id if server_load_claim is not None else None),
        "attempt_claim_sha256": binding.claim_sha256,
        "server_load_claim_sha256": (
            server_load_claim.claim_sha256() if server_load_claim is not None else None
        ),
        "completed_at_utc": completed_at_utc,
    }


def _complete_diagnostic_stage(
    completed: list[DiagnosticStageName],
    stage: DiagnosticStageName,
) -> None:
    if len(completed) >= len(DIAGNOSTIC_PLANNED_STAGES):
        raise RuntimeError("diagnostic stage list is already complete")
    if stage != DIAGNOSTIC_PLANNED_STAGES[len(completed)]:
        raise RuntimeError("diagnostic stage completion is out of checked order")
    completed.append(stage)


def _is_diagnostic_stage(value: str) -> TypeGuard[DiagnosticStageName]:
    return value in DIAGNOSTIC_PLANNED_STAGES


def _publish_diagnostic_private_raw(
    raw: DiagnosticPrivateRawEvidence,
    *,
    bundle: InklingBF16InterfaceDiagnosticBundle,
) -> tuple[DiagnosticPrivateRawEvidence, DiagnosticPrivateRawReference]:
    payload = raw.canonical_bytes()
    parsed = parse_diagnostic_private_raw_evidence(payload, run_id=raw.run_id)
    validate_diagnostic_private_trials(parsed, bundle=bundle)
    if parsed != raw:
        raise RuntimeError("diagnostic private evidence changed during canonical parsing")
    reference = build_diagnostic_private_raw_reference(
        payload,
        evidence_root=EVIDENCE_ROOT.as_posix(),
        run_id=raw.run_id,
    )
    validate_diagnostic_private_raw_reference(payload, expected=reference)
    _commit_and_verify({reference.relative_path: payload})
    readback = _read_regular_bytes(
        _evidence_path(reference.relative_path),
        maximum_bytes=reference.size_bytes,
    )
    validate_diagnostic_private_raw_reference(readback, expected=reference)
    committed = parse_diagnostic_private_raw_evidence(readback, run_id=raw.run_id)
    validate_diagnostic_private_trials(committed, bundle=bundle)
    if committed != raw:
        raise RuntimeError("committed diagnostic private evidence differs from exact result")
    return committed, reference


def _publish_diagnostic_terminal_receipt(
    receipt: DiagnosticSuccessTerminalReceipt | DiagnosticFailureTerminalReceipt,
    *,
    outcome: Literal["success", "failure"],
) -> DiagnosticTerminalReceiptReference:
    payload = canonical_diagnostic_json_bytes(receipt.model_dump(mode="json"))
    parsed = parse_diagnostic_terminal_receipt(
        payload,
        run_id=receipt.run_id,
        outcome=outcome,
    )
    if parsed != receipt:
        raise RuntimeError("diagnostic terminal receipt changed during canonical parsing")
    reference = build_diagnostic_terminal_receipt_reference(
        payload,
        evidence_root=EVIDENCE_ROOT.as_posix(),
        run_id=receipt.run_id,
        outcome=outcome,
    )
    validate_diagnostic_terminal_receipt_reference(payload, expected=reference)
    _commit_and_verify({reference.relative_path: payload})
    readback = _read_regular_bytes(
        _evidence_path(reference.relative_path),
        maximum_bytes=reference.size_bytes,
    )
    validate_diagnostic_terminal_receipt_reference(readback, expected=reference)
    if (
        parse_diagnostic_terminal_receipt(
            readback,
            run_id=receipt.run_id,
            outcome=outcome,
        )
        != receipt
    ):
        raise RuntimeError("committed diagnostic terminal receipt differs from exact result")
    return reference


def _diagnostic_subject_spec(
    bundle: InklingBF16InterfaceDiagnosticBundle,
) -> DiagnosticSubjectSpec:
    artifacts = tuple(
        (
            (Path(bundle.config.storage.bf16_mount_path) / artifact.path).as_posix(),
            artifact.sha256,
            artifact.size_bytes,
        )
        for artifact in bundle.bf16.bf16_shards
    )
    if len(artifacts) != 49 or sum(item[2] for item in artifacts) != (bundle.bf16.bf16_total_bytes):
        raise RuntimeError("diagnostic BF16 artifact inventory is incomplete")
    paths = tuple(item[0] for item in artifacts)
    return DiagnosticSubjectSpec(
        model_path=paths[0],
        shard_paths=paths,
        artifacts=artifacts,
    )


def _prepare_diagnostic_staging_root(
    bundle: InklingBF16InterfaceDiagnosticBundle,
) -> Path:
    execution = bundle.config.execution
    if (
        execution.subject_staging_root != DIAGNOSTIC_STAGING_ROOT.as_posix()
        or execution.subject_staging_headroom_mib * 1024 * 1024 != SUBJECT_STAGING_HEADROOM_BYTES
        or not execution.release_staged_subject
    ):
        raise RuntimeError("diagnostic staging configuration differs from the fixed protocol")
    parent = DIAGNOSTIC_STAGING_ROOT.parent
    if not os.path.lexists(parent):
        os.mkdir(parent, 0o700)
    _require_canonical_directory_components(
        parent,
        label="diagnostic staging parent",
    )
    if not os.path.lexists(DIAGNOSTIC_STAGING_ROOT):
        os.mkdir(DIAGNOSTIC_STAGING_ROOT, 0o700)
    return _require_canonical_directory_components(
        DIAGNOSTIC_STAGING_ROOT,
        label="diagnostic staging root",
    )


def _stage_diagnostic_subject(
    subject: DiagnosticSubjectSpec,
    *,
    bundle: InklingBF16InterfaceDiagnosticBundle,
    work_deadline: float,
) -> DiagnosticSubjectSpec:
    """Copy and rehash only the forty-nine BF16 shards to ephemeral storage."""

    root = _prepare_diagnostic_staging_root(bundle)
    subject_root = root / "bf16"
    if os.path.lexists(subject_root):
        raise RuntimeError("diagnostic BF16 staging directory already exists")
    source_paths = tuple(path for path, _, _ in subject.artifacts)
    if len(source_paths) != 49 or len(set(source_paths)) != 49:
        raise RuntimeError("diagnostic BF16 inventory contains duplicate or missing paths")
    mount = Path(bundle.config.storage.bf16_mount_path)
    resolved_mount = _resolved_modal_mount_root(
        mount,
        label="diagnostic BF16 read-only mount",
    )
    resolved_sources: dict[str, Path] = {}
    for source_path in source_paths:
        source = Path(source_path)
        if (
            not source.is_absolute()
            or "\\" in source_path
            or "\x00" in source_path
            or PurePosixPath(source_path).as_posix() != source_path
            or any(part in {"", ".", ".."} for part in PurePosixPath(source_path).parts[1:])
            or not source.is_relative_to(mount)
        ):
            raise RuntimeError("diagnostic shard path is outside the BF16 read-only mount")
        suffix = source.relative_to(mount)
        resolved_source = resolved_mount.joinpath(*suffix.parts)
        if not resolved_source.is_relative_to(resolved_mount):
            raise RuntimeError("diagnostic shard path escaped its resolved read-only mount")
        _require_canonical_directory_components(
            resolved_source.parent,
            label="diagnostic BF16 shard parent",
        )
        resolved_sources[source_path] = resolved_source
    required_bytes = sum(size for _, _, size in subject.artifacts)
    if required_bytes != bundle.bf16.bf16_total_bytes:
        raise RuntimeError("diagnostic BF16 byte total differs from the reviewed reference")
    filesystem = os.statvfs(root)
    free_bytes = filesystem.f_bavail * filesystem.f_frsize
    if free_bytes < required_bytes + SUBJECT_STAGING_HEADROOM_BYTES:
        raise RuntimeError("ephemeral disk lacks BF16 bytes plus the required 128 GiB headroom")
    staged_by_source: dict[str, str] = {}
    subject_root.mkdir(mode=0o700)
    try:
        for source_path, expected_hash, expected_size in subject.artifacts:
            _remaining_work_timeout(
                work_deadline,
                1.0,
                label="diagnostic BF16 staging",
            )
            staged_path = subject_root / Path(source_path).relative_to(mount)
            _stage_file_once(
                source_path=source_path,
                resolved_source_path=resolved_sources[source_path],
                staged_path=staged_path,
                expected_sha256=expected_hash,
                expected_size_bytes=expected_size,
                work_deadline=work_deadline,
            )
            staged_by_source[source_path] = staged_path.as_posix()
    except BaseException:
        with suppress(OSError):
            shutil.rmtree(subject_root)
        raise
    model_path = staged_by_source.get(subject.model_path)
    shard_paths = tuple(staged_by_source.get(path, "") for path in subject.shard_paths)
    if model_path is None or any(not path for path in shard_paths):
        shutil.rmtree(subject_root)
        raise RuntimeError("staged diagnostic subject does not bind all forty-nine shards")
    return DiagnosticSubjectSpec(
        model_path=model_path,
        shard_paths=shard_paths,
        artifacts=tuple(
            (staged_by_source[path], sha256, size_bytes)
            for path, sha256, size_bytes in subject.artifacts
        ),
    )


def _release_diagnostic_subject(subject: DiagnosticSubjectSpec) -> None:
    root = _require_canonical_directory_components(
        DIAGNOSTIC_STAGING_ROOT,
        label="diagnostic staging root",
    )
    subject_root = root / "bf16"
    staged_paths = {path for path, _, _ in subject.artifacts}
    if (
        len(staged_paths) != 49
        or not all(Path(path).is_relative_to(subject_root) for path in staged_paths)
        or not subject_root.exists()
        or subject_root.is_symlink()
        or subject_root.resolve(strict=True) != subject_root
        or subject_root.parent != root
    ):
        raise RuntimeError("diagnostic BF16 directory is not safe to release")
    shutil.rmtree(subject_root)
    if os.path.lexists(subject_root):
        raise RuntimeError("diagnostic BF16 directory remained after release")


def _expected_server_chat_template(official_chat_template: str) -> str:
    """Return the template text llama-server reports for the official source asset.

    The pinned llama.cpp jinja lexer normalizes carriage returns and then drops one
    trailing line feed before it stores the template source that ``/props`` reports,
    so the official asset text is never byte-identical to the runtime value when the
    asset ends in a newline.  Carriage returns are rejected instead of normalized:
    the asset is SHA-256 pinned and has none, so normalizing them here would add an
    unexercised second rule.
    """
    if "\r" in official_chat_template:
        raise RuntimeError("official source chat template contains a carriage return")
    if official_chat_template.endswith("\n"):
        return official_chat_template[:-1]
    return official_chat_template


def _verify_diagnostic_source_assets(
    bundle: InklingBF16InterfaceDiagnosticBundle,
) -> DiagnosticSourceEvidence:
    source_assets = bundle.config.source_assets
    mount = Path(bundle.config.storage.source_mount_path)
    resolved_mount = _resolved_modal_mount_root(
        mount,
        label="diagnostic source read-only mount",
    )
    identities = (
        source_assets.config,
        source_assets.chat_template,
        source_assets.tokenizer_json,
        source_assets.tokenizer_config,
    )
    payloads: dict[str, bytes] = {}
    for identity in identities:
        source = Path(identity.path)
        if (
            not source.is_absolute()
            or "\\" in identity.path
            or "\x00" in identity.path
            or PurePosixPath(identity.path).as_posix() != identity.path
            or not source.is_relative_to(mount)
        ):
            raise RuntimeError("diagnostic source asset is outside its read-only mount")
        resolved = resolved_mount.joinpath(*source.relative_to(mount).parts)
        if not resolved.is_relative_to(resolved_mount):
            raise RuntimeError("diagnostic source asset escaped its resolved mount")
        _require_canonical_directory_components(
            resolved.parent,
            label="diagnostic source asset parent",
        )
        payload = _read_regular_bytes(resolved, maximum_bytes=identity.size_bytes)
        if len(payload) != identity.size_bytes or _sha256_bytes(payload) != identity.sha256:
            raise RuntimeError("diagnostic source asset differs from its reviewed identity")
        payloads[identity.path] = payload

    config_payload = payloads[source_assets.config.path]
    source_config = strict_diagnostic_json_object(
        config_payload,
        maximum_bytes=source_assets.config.size_bytes,
    )
    if source_config.get("eos_token_id") != DIAGNOSTIC_EOS_TOKEN_ID:
        raise RuntimeError("source config EOS token differs from the diagnostic protocol")

    tokenizer_payload = payloads[source_assets.tokenizer_json.path]
    tokenizer = strict_diagnostic_json_object(
        tokenizer_payload,
        maximum_bytes=source_assets.tokenizer_json.size_bytes,
    )
    added_tokens = tokenizer.get("added_tokens")
    if not isinstance(added_tokens, list):
        raise RuntimeError("source tokenizer added-token inventory is unavailable")
    expected_special_tokens = {
        DIAGNOSTIC_EOS_TOKEN_ID: source_assets.eos_special_token,
        DIAGNOSTIC_COMPARISON_TOKEN_ID: source_assets.comparison_special_token,
    }
    expected_special_token_text = frozenset(expected_special_tokens.values())
    observed_special_tokens: dict[int, str] = {}
    for entry in added_tokens:
        if not isinstance(entry, dict):
            raise RuntimeError("source tokenizer added-token inventory is malformed")
        token_id = entry.get("id")
        content = entry.get("content")
        expected_id = type(token_id) is int and token_id in expected_special_tokens
        expected_content = isinstance(content, str) and content in expected_special_token_text
        if not expected_id and not expected_content:
            continue
        if type(token_id) is not int or not isinstance(content, str):
            raise RuntimeError("source tokenizer special-token mapping is malformed")
        if expected_special_tokens.get(token_id) != content or entry.get("special") is not True:
            raise RuntimeError("source tokenizer special-token mapping differs from the protocol")
        if token_id in observed_special_tokens:
            raise RuntimeError("source tokenizer repeats a protocol special-token ID")
        observed_special_tokens[token_id] = content
    if observed_special_tokens != expected_special_tokens:
        raise RuntimeError("source tokenizer lacks an exact protocol special-token mapping")

    tokenizer_config_payload = payloads[source_assets.tokenizer_config.path]
    tokenizer_config = strict_diagnostic_json_object(
        tokenizer_config_payload,
        maximum_bytes=source_assets.tokenizer_config.size_bytes,
    )
    added_tokens_decoder = tokenizer_config.get("added_tokens_decoder")
    if not isinstance(added_tokens_decoder, dict):
        raise RuntimeError("source tokenizer config added-token decoder is unavailable")
    for token_id, content in expected_special_tokens.items():
        decoder_entry = added_tokens_decoder.get(str(token_id))
        if (
            not isinstance(decoder_entry, dict)
            or decoder_entry.get("content") != content
            or decoder_entry.get("special") is not True
        ):
            raise RuntimeError(
                "source tokenizer config special-token mapping differs from the protocol"
            )
    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, dict) or (
        extra_special_tokens.get("content_model_end_sampling") != source_assets.eos_special_token
        or extra_special_tokens.get("endoftext") != source_assets.comparison_special_token
    ):
        raise RuntimeError("source tokenizer config special-token aliases differ from the protocol")
    try:
        official_chat_template = payloads[source_assets.chat_template.path].decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError("official source chat template is not strict UTF-8") from error
    expected_server_chat_template = _expected_server_chat_template(official_chat_template)
    asset_manifest = {
        "schema_version": "inkling-bf16-interface-source-assets-v1",
        "assets": [identity.model_dump(mode="json") for identity in identities],
    }
    return DiagnosticSourceEvidence(
        source_config_sha256=_sha256_bytes(config_payload),
        asset_manifest_sha256=_sha256_bytes(canonical_diagnostic_json_bytes(asset_manifest)),
        expected_server_chat_template=expected_server_chat_template,
    )


def _diagnostic_server_command(
    subject: DiagnosticSubjectSpec,
    *,
    port: int,
) -> tuple[str, ...]:
    """Build the one exact model-only BF16 diagnostic server command."""

    if port != DIAGNOSTIC_SERVER_PORT:
        raise ValueError("diagnostic server port differs from the fixed protocol")
    model_path = Path(subject.model_path)
    if (
        not model_path.is_absolute()
        or "\\" in subject.model_path
        or "\x00" in subject.model_path
        or PurePosixPath(subject.model_path).as_posix() != subject.model_path
        or subject.model_path != subject.shard_paths[0]
        or len(subject.shard_paths) != 49
    ):
        raise ValueError("diagnostic model path is not the staged first BF16 shard")
    command = build_diagnostic_server_command(subject.model_path)
    if command[0] != COMMAND_BINARIES["llama-server"]:
        raise RuntimeError("diagnostic server command uses the wrong executable")
    return command


def _start_diagnostic_server(
    *,
    subject: DiagnosticSubjectSpec,
    expected_uuids: Sequence[str],
    work_deadline: float,
) -> DiagnosticServerProcess:
    command = _diagnostic_server_command(subject, port=DIAGNOSTIC_SERVER_PORT)
    log_path = Path(f"/tmp/iql-bf16-interface-diagnostic-{DIAGNOSTIC_SERVER_PORT}.log")
    with suppress(FileNotFoundError):
        log_path.unlink()
    environment = build_matched_server_environment(
        os.environ,
        audit_environment=DIAGNOSTIC_SERVER_AUDIT_ENVIRONMENT,
    )
    started_at_utc = _utc_now()
    started_monotonic = time.monotonic()
    descriptor = os.open(
        log_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    process: subprocess.Popen[bytes] | None = None
    monitor: ResourceMonitor | None = None
    monitor_started = False
    try:
        try:
            process = subprocess.Popen(
                command,
                stdout=descriptor,
                stderr=subprocess.STDOUT,
                env=environment,
                shell=False,
            )
        finally:
            os.close(descriptor)
        monitor = ResourceMonitor(process.pid, expected_uuids)
        monitor.start()
        monitor_started = True
        _wait_server_ready(
            DIAGNOSTIC_SERVER_PORT,
            process,
            work_deadline=work_deadline,
        )
        if process is None or monitor is None:
            raise RuntimeError("diagnostic server lifecycle did not initialize")
    except BaseException as error:
        try:
            if process is not None and process.poll() is None:
                process.kill()
        except BaseException as cleanup_error:
            error.add_note(
                "diagnostic server kill after readiness failure also failed: "
                f"{type(cleanup_error).__name__}"
            )
        try:
            if process is not None:
                process.wait(timeout=30)
        except BaseException as cleanup_error:
            error.add_note(
                "diagnostic server wait after readiness failure also failed: "
                f"{type(cleanup_error).__name__}"
            )
        try:
            if monitor is not None and monitor_started:
                monitor.stop()
        except BaseException as cleanup_error:
            error.add_note(
                "diagnostic resource-monitor cleanup after readiness failure also failed: "
                f"{type(cleanup_error).__name__}"
            )
        try:
            log_path.unlink()
        except FileNotFoundError:
            pass
        except BaseException as cleanup_error:
            error.add_note(
                "diagnostic server-log cleanup after readiness failure also failed: "
                f"{type(cleanup_error).__name__}"
            )
        raise
    return DiagnosticServerProcess(
        process=process,
        command=command,
        log_path=log_path,
        started_at_utc=started_at_utc,
        started_monotonic=started_monotonic,
        monitor=monitor,
    )


def _stop_diagnostic_server(
    server: DiagnosticServerProcess,
) -> tuple[str, dict[str, Any], float]:
    primary_error: BaseException | None = None
    try:
        if server.process.poll() is None:
            server.process.terminate()
            try:
                server.process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                server.process.kill()
                server.process.wait(timeout=30)
    except BaseException as error:
        primary_error = error
    telemetry: dict[str, Any] | None = None
    try:
        telemetry = server.monitor.stop()
    except BaseException as error:
        if primary_error is None:
            primary_error = error
        else:
            primary_error.add_note(
                f"diagnostic resource-monitor cleanup also failed: {type(error).__name__}"
            )
    finished_monotonic = time.monotonic()
    log_payload: bytes | None = None
    try:
        log_payload = _read_regular_bytes(server.log_path, maximum_bytes=MAX_LOG_BYTES)
    except BaseException as error:
        if primary_error is None:
            primary_error = error
        else:
            primary_error.add_note(
                f"diagnostic server-log read also failed: {type(error).__name__}"
            )
    try:
        server.log_path.unlink()
    except FileNotFoundError as error:
        if primary_error is None:
            primary_error = error
        else:
            primary_error.add_note(
                f"diagnostic server-log cleanup also failed: {type(error).__name__}"
            )
    except BaseException as error:
        if primary_error is None:
            primary_error = error
        else:
            primary_error.add_note(
                f"diagnostic server-log cleanup also failed: {type(error).__name__}"
            )
    if primary_error is not None:
        raise primary_error
    if telemetry is None or log_payload is None:
        raise RuntimeError("diagnostic server cleanup lacks its exact evidence")
    if server.process.returncode not in (0, -15):
        raise RuntimeError("diagnostic llama-server cleanup observed an unexpected exit")
    try:
        log_text = log_payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise RuntimeError("diagnostic llama-server log is not strict UTF-8") from error
    if not log_text:
        raise RuntimeError("diagnostic llama-server log is empty")
    return log_text, telemetry, finished_monotonic


def _diagnostic_props(
    *,
    source: DiagnosticSourceEvidence,
    work_deadline: float,
) -> int:
    props, _ = _http_json(
        DIAGNOSTIC_SERVER_PORT,
        "GET",
        "/props",
        None,
        timeout=30,
        work_deadline=work_deadline,
    )
    chat_template = props.get("chat_template")
    if not isinstance(chat_template, str) or chat_template != source.expected_server_chat_template:
        raise RuntimeError(
            "llama-server chat template differs from the exact pinned runtime representation"
        )
    build_info = props.get("build_info")
    if not isinstance(build_info, str) or PINNED_LLAMA_CPP_COMMIT[:7] not in build_info:
        raise RuntimeError("diagnostic llama-server build identity is unavailable")
    models, _ = _http_json(
        DIAGNOSTIC_SERVER_PORT,
        "GET",
        "/v1/models",
        None,
        timeout=30,
        work_deadline=work_deadline,
    )
    data = models.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], Mapping):
        raise RuntimeError("diagnostic llama-server model metadata has the wrong shape")
    meta = data[0].get("meta")
    if not isinstance(meta, Mapping) or type(meta.get("n_vocab")) is not int:
        raise RuntimeError("diagnostic llama-server lacks exact vocabulary metadata")
    vocab_size = int(meta["n_vocab"])
    if vocab_size <= max(DIAGNOSTIC_EOS_TOKEN_ID, DIAGNOSTIC_COMPARISON_TOKEN_ID):
        raise RuntimeError("diagnostic llama-server vocabulary is incompatible")
    return vocab_size


def _expected_diagnostic_chat_prompt(instruction: str, item_prompt: str) -> str:
    return (
        f"<|message_system|><|content_text|>{instruction}<|end_message|>"
        "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
        f"<|message_user|><|content_text|>{item_prompt}<|end_message|>"
        "<|message_model|>"
    )


def _render_diagnostic_chat_prompt(
    instruction: str,
    item_prompt: str,
    *,
    work_deadline: float,
) -> str:
    request = {
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": item_prompt},
        ],
        "add_generation_prompt": True,
        "chat_template_kwargs": {"reasoning_effort": "none"},
    }
    response, _ = _http_json(
        DIAGNOSTIC_SERVER_PORT,
        "POST",
        "/apply-template",
        request,
        timeout=60,
        work_deadline=work_deadline,
    )
    rendered = response.get("prompt")
    expected = _expected_diagnostic_chat_prompt(instruction, item_prompt)
    if not isinstance(rendered, str) or rendered != expected:
        raise RuntimeError("diagnostic chat rendering differs from the exact protocol")
    return rendered


def _tokenize_diagnostic_prompt(
    rendered: str,
    *,
    vocab_size: int,
    work_deadline: float,
) -> tuple[int, ...]:
    response, _ = _http_json(
        DIAGNOSTIC_SERVER_PORT,
        "POST",
        "/tokenize",
        {
            "content": rendered,
            "add_special": False,
            "parse_special": True,
            "with_pieces": False,
        },
        timeout=60,
        work_deadline=work_deadline,
    )
    raw_tokens = response.get("tokens")
    if (
        not isinstance(raw_tokens, list)
        or not raw_tokens
        or any(
            type(token_id) is not int or not 0 <= token_id < vocab_size for token_id in raw_tokens
        )
    ):
        raise RuntimeError("diagnostic tokenization returned invalid token IDs")
    return tuple(int(token_id) for token_id in raw_tokens)


def _diagnostic_request(
    input_token_ids: Sequence[int],
    *,
    n_predict: int,
) -> dict[str, Any]:
    return {
        "prompt": list(input_token_ids),
        "n_predict": n_predict,
        "temperature": -1.0,
        "seed": 42,
        "stream": False,
        "cache_prompt": False,
        "return_tokens": True,
        "timings_per_token": True,
        "ignore_eos": False,
        "stop": [],
    }


def _validate_diagnostic_completion(
    payload: Mapping[str, Any],
    *,
    response_sha256: str,
    request_duration_seconds: float,
    input_token_ids: Sequence[int],
    vocab_size: int,
) -> DiagnosticCompletionResult:
    if payload.get("error") is not None:
        raise RuntimeError("diagnostic completion returned an error")
    content = payload.get("content")
    tokens_raw = payload.get("tokens")
    tokens_evaluated = payload.get("tokens_evaluated")
    tokens_predicted = payload.get("tokens_predicted")
    stop = payload.get("stop")
    stop_type = payload.get("stop_type")
    truncated = payload.get("truncated")
    timings_raw = payload.get("timings")
    if not isinstance(content, str):
        raise RuntimeError("diagnostic completion content has the wrong type")
    if (
        not isinstance(tokens_raw, list)
        or not tokens_raw
        or any(
            type(token_id) is not int or not 0 <= token_id < vocab_size for token_id in tokens_raw
        )
    ):
        raise RuntimeError("diagnostic completion returned invalid token IDs")
    if (
        type(tokens_evaluated) is not int
        or type(tokens_predicted) is not int
        or tokens_evaluated != len(input_token_ids)
        or tokens_predicted != len(tokens_raw)
    ):
        raise RuntimeError("diagnostic completion top-level token counts are inconsistent")
    if stop is not True or stop_type not in {"eos", "limit"}:
        raise RuntimeError("diagnostic completion did not reach an accepted terminal state")
    if truncated is not False:
        raise RuntimeError("diagnostic completion is truncated or lacks exact truncation evidence")
    if not isinstance(timings_raw, Mapping):
        raise RuntimeError("diagnostic completion lacks server timing evidence")
    timing_fields: dict[str, Any] = {}
    for field in ("prompt_n", "predicted_n"):
        value = timings_raw.get(field)
        if type(value) is not int or value <= 0:
            raise RuntimeError("diagnostic completion timing count is invalid")
        timing_fields[field] = value
    for field in (
        "prompt_ms",
        "predicted_ms",
        "prompt_per_second",
        "predicted_per_second",
    ):
        value = timings_raw.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise RuntimeError("diagnostic completion timing value is invalid")
        timing_fields[field] = float(value)
    timings = MeasurementDiagnosticTimings.model_validate(timing_fields, strict=True)
    if (
        timings.prompt_n != tokens_evaluated
        or timings.predicted_n != tokens_predicted
        or timings.prompt_n != len(input_token_ids)
        or timings.predicted_n != len(tokens_raw)
    ):
        raise RuntimeError("diagnostic completion timing counts are inconsistent")
    if (
        isinstance(request_duration_seconds, bool)
        or not math.isfinite(request_duration_seconds)
        or request_duration_seconds <= 0.0
    ):
        raise RuntimeError("diagnostic completion duration is invalid")
    return DiagnosticCompletionResult(
        content=content,
        tokens=tuple(int(token_id) for token_id in tokens_raw),
        tokens_evaluated=tokens_evaluated,
        tokens_predicted=tokens_predicted,
        stop=True,
        stop_type=cast('Literal["eos", "limit"]', stop_type),
        truncated=False,
        response_sha256=response_sha256,
        duration_seconds=request_duration_seconds,
        timings=timings,
    )


def _run_diagnostic_completion(
    request: Mapping[str, Any],
    *,
    input_token_ids: Sequence[int],
    vocab_size: int,
    work_deadline: float,
) -> DiagnosticCompletionResult:
    started = time.monotonic()
    payload, response_sha256 = _http_json(
        DIAGNOSTIC_SERVER_PORT,
        "POST",
        "/completion",
        request,
        timeout=REQUEST_TIMEOUT_SECONDS,
        work_deadline=work_deadline,
    )
    duration_seconds = time.monotonic() - started
    return _validate_diagnostic_completion(
        payload,
        response_sha256=response_sha256,
        request_duration_seconds=duration_seconds,
        input_token_ids=input_token_ids,
        vocab_size=vocab_size,
    )


def _extract_diagnostic_content_segment(content: str) -> str:
    before_sampling_end = content.split(DIAGNOSTIC_END_SAMPLING_MARKER, 1)[0]
    if DIAGNOSTIC_CONTENT_TEXT_MARKER in before_sampling_end:
        before_sampling_end = before_sampling_end.rsplit(
            DIAGNOSTIC_CONTENT_TEXT_MARKER,
            1,
        )[1]
    return before_sampling_end.split(DIAGNOSTIC_END_MESSAGE_MARKER, 1)[0]


def _diagnostic_score_detail_sha256(
    *,
    whole_score: DiagnosticScoreEvidence,
    extracted_score: DiagnosticScoreEvidence,
) -> str:
    return _sha256_bytes(
        canonical_diagnostic_json_bytes(
            {
                "schema_version": "inkling-bf16-interface-score-detail-v1",
                "extraction_protocol": (
                    "last_content_text_segment_before_content_model_end_sampling_or_end"
                ),
                "whole_output": asdict(whole_score),
                "extracted_content_text": asdict(extracted_score),
            }
        )
    )


def _parse_diagnostic_runtime_eog_ids(log_text: str) -> tuple[int, ...]:
    eos_pattern = re.compile(
        r"^print_info: EOS token\s+=\s+([0-9]+) '[^'\r\n]*'$",
        flags=re.MULTILINE,
    )
    eog_pattern = re.compile(
        r"^print_info: EOG token\s+=\s+([0-9]+) '[^'\r\n]*'$",
        flags=re.MULTILINE,
    )
    eos_ids = tuple(int(item) for item in eos_pattern.findall(log_text))
    eog_ids = tuple(int(item) for item in eog_pattern.findall(log_text))
    if eos_ids != (DIAGNOSTIC_EOS_TOKEN_ID,):
        raise RuntimeError("authoritative llama-server EOS metadata is not exact")
    if not eog_ids or len(eog_ids) != len(set(eog_ids)):
        raise RuntimeError("authoritative llama-server EOG metadata is incomplete")
    return tuple(sorted(eog_ids))


def _forced_diagnostic_probe(
    token_id: int,
    *,
    input_token_ids: Sequence[int],
    vocab_size: int,
    work_deadline: float,
) -> tuple[DiagnosticCompletionResult, str]:
    request = _diagnostic_request(input_token_ids, n_predict=1)
    request["logit_bias"] = [[token_id, DIAGNOSTIC_FORCED_LOGIT_BIAS]]
    request_sha256 = _sha256_bytes(canonical_diagnostic_json_bytes(request))
    result = _run_diagnostic_completion(
        request,
        input_token_ids=input_token_ids,
        vocab_size=vocab_size,
        work_deadline=work_deadline,
    )
    if result.tokens != (token_id,):
        raise RuntimeError("forced diagnostic probe did not return its exact token")
    return result, request_sha256


def _run_bf16_diagnostic_server(
    *,
    subject: DiagnosticSubjectSpec,
    source: DiagnosticSourceEvidence,
    bundle: InklingBF16InterfaceDiagnosticBundle,
    expected_uuids: Sequence[str],
    work_deadline: float,
) -> DiagnosticServerResult:
    """Load BF16 once, prepare all prompts, then issue sixteen sequential requests."""

    server = _start_diagnostic_server(
        subject=subject,
        expected_uuids=expected_uuids,
        work_deadline=work_deadline,
    )
    try:
        vocab_size = _diagnostic_props(source=source, work_deadline=work_deadline)
        instruction = bundle.config.protocol.raw_instruction
        rendered_prompts: dict[tuple[str, str], str] = {}
        tokenized_prompts: dict[tuple[str, str], tuple[int, ...]] = {}
        for item in bundle.items:
            raw_prompt = f"{instruction}\n{item.prompt}"
            chat_prompt = _render_diagnostic_chat_prompt(
                instruction,
                item.prompt,
                work_deadline=work_deadline,
            )
            rendered_prompts[("raw", item.item_id)] = raw_prompt
            rendered_prompts[("chat_template", item.item_id)] = chat_prompt
        for prompt_mode in ("raw", "chat_template"):
            for item in bundle.items:
                rendered = rendered_prompts[(prompt_mode, item.item_id)]
                tokenized_prompts[(prompt_mode, item.item_id)] = _tokenize_diagnostic_prompt(
                    rendered,
                    vocab_size=vocab_size,
                    work_deadline=work_deadline,
                )

        trials: list[DiagnosticPrivateTrial] = []
        for cell in bundle.config.protocol.cells:
            for item in bundle.items:
                ordinal = len(trials) + 1
                rendered = rendered_prompts[(cell.prompt_mode, item.item_id)]
                input_token_ids = tokenized_prompts[(cell.prompt_mode, item.item_id)]
                requested_cap = (
                    item.max_new_tokens
                    if cell.max_new_tokens_override is None
                    else cell.max_new_tokens_override
                )
                if item.item_id not in bundle.config.protocol.item_ids:
                    raise RuntimeError("diagnostic item differs from the reviewed protocol")
                if item.max_new_tokens not in (4, 8, 16):
                    raise RuntimeError("diagnostic item has an unreviewed original token cap")
                if requested_cap not in (4, 8, 16, 64):
                    raise RuntimeError("diagnostic request has an unreviewed token cap")
                request = _diagnostic_request(input_token_ids, n_predict=requested_cap)
                request_sha256 = _sha256_bytes(canonical_diagnostic_json_bytes(request))
                result = _run_diagnostic_completion(
                    request,
                    input_token_ids=input_token_ids,
                    vocab_size=vocab_size,
                    work_deadline=work_deadline,
                )
                if result.tokens_predicted > requested_cap:
                    raise RuntimeError("natural diagnostic request exceeded its token cap")
                if result.stop_type == "limit" and result.tokens_predicted != requested_cap:
                    raise RuntimeError("limit-stopped diagnostic request missed its token cap")
                whole_score = evaluate_diagnostic_response(
                    result.content,
                    scorer_kind=item.scorer.kind,
                    expected=item.scorer.expected,
                )
                extracted_score = evaluate_diagnostic_response(
                    _extract_diagnostic_content_segment(result.content),
                    scorer_kind=item.scorer.kind,
                    expected=item.scorer.expected,
                )
                trials.append(
                    DiagnosticPrivateTrial(
                        ordinal=ordinal,
                        item_id=item.item_id,
                        cell=cell.name,
                        prompt_mode=cell.prompt_mode,
                        cap_mode=cell.cap_mode,
                        original_max_new_tokens=cast(
                            "Literal[4, 8, 16]",
                            item.max_new_tokens,
                        ),
                        requested_max_new_tokens=cast(
                            "Literal[4, 8, 16, 64]",
                            requested_cap,
                        ),
                        reasoning_effort="none",
                        item_prompt_sha256=_sha256_bytes(item.prompt.encode("utf-8")),
                        rendered_prompt_sha256=_sha256_bytes(rendered.encode("utf-8")),
                        request_sha256=request_sha256,
                        response_sha256=result.response_sha256,
                        input_token_ids=input_token_ids,
                        output_token_ids=result.tokens,
                        tokens_evaluated=result.tokens_evaluated,
                        tokens_predicted=result.tokens_predicted,
                        stop=True,
                        stop_type=result.stop_type,
                        eog_observed=result.stop_type == "eos",
                        cap_hit=result.stop_type == "limit",
                        truncated=False,
                        whole_output_passed=whole_score.score,
                        extracted_content_passed=extracted_score.score,
                        score_detail_sha256=_diagnostic_score_detail_sha256(
                            whole_score=whole_score,
                            extracted_score=extracted_score,
                        ),
                        request_duration_seconds=result.duration_seconds,
                        timings=result.timings,
                    )
                )
        if len(trials) != 16:
            raise RuntimeError("BF16 interface diagnostic did not complete sixteen requests")

        probe_prompt = tokenized_prompts[("raw", bundle.items[0].item_id)]
        eos_probe, eos_probe_request_sha256 = _forced_diagnostic_probe(
            DIAGNOSTIC_EOS_TOKEN_ID,
            input_token_ids=probe_prompt,
            vocab_size=vocab_size,
            work_deadline=work_deadline,
        )
        comparison_probe, comparison_probe_request_sha256 = _forced_diagnostic_probe(
            DIAGNOSTIC_COMPARISON_TOKEN_ID,
            input_token_ids=probe_prompt,
            vocab_size=vocab_size,
            work_deadline=work_deadline,
        )
    except BaseException as error:
        try:
            _stop_diagnostic_server(server)
        except BaseException as cleanup_error:
            error.add_note(
                f"diagnostic llama-server cleanup also failed: {type(cleanup_error).__name__}"
            )
        raise

    log_text, telemetry_payload, finished_monotonic = _stop_diagnostic_server(server)
    text_artifact_load = parse_text_artifact_load_evidence(
        log_text,
        expected_first_shard_path=subject.model_path,
    )
    runtime_eog_ids = _parse_diagnostic_runtime_eog_ids(log_text)
    eog = DiagnosticEogEvidence(
        source_config_sha256=source.source_config_sha256,
        source_config_eos_token_id=DIAGNOSTIC_EOS_TOKEN_ID,
        runtime_eog_token_ids=runtime_eog_ids,
        source_eos_probe=DiagnosticEogTokenProbe(
            token_id=DIAGNOSTIC_EOS_TOKEN_ID,
            runtime_is_eog=DIAGNOSTIC_EOS_TOKEN_ID in runtime_eog_ids,
            forced_token_observed=True,
            request_sha256=eos_probe_request_sha256,
            input_token_ids=probe_prompt,
            generated_token_ids=eos_probe.tokens,
            stop=True,
            stop_type=eos_probe.stop_type,
            truncated=False,
            response_sha256=eos_probe.response_sha256,
            request_duration_seconds=eos_probe.duration_seconds,
            timings=eos_probe.timings,
        ),
        comparison_token_probe=DiagnosticEogTokenProbe(
            token_id=DIAGNOSTIC_COMPARISON_TOKEN_ID,
            runtime_is_eog=DIAGNOSTIC_COMPARISON_TOKEN_ID in runtime_eog_ids,
            forced_token_observed=True,
            request_sha256=comparison_probe_request_sha256,
            input_token_ids=probe_prompt,
            generated_token_ids=comparison_probe.tokens,
            stop=True,
            stop_type=comparison_probe.stop_type,
            truncated=False,
            response_sha256=comparison_probe.response_sha256,
            request_duration_seconds=comparison_probe.duration_seconds,
            timings=comparison_probe.timings,
        ),
    )
    if server.monitor.process_id != server.process.pid:
        raise RuntimeError("diagnostic resource monitor observed a different process")
    resource_sample_summary = MeasurementResourceSampleSummary.model_validate(
        _telemetry_window(
            telemetry_payload,
            started_monotonic=server.started_monotonic,
            finished_monotonic=finished_monotonic,
        ),
        strict=True,
    )
    log_payload = log_text.encode("utf-8")
    command_sha256 = _sha256_bytes(canonical_diagnostic_json_bytes(list(server.command)))
    return DiagnosticServerResult(
        trials=tuple(trials),
        eog=eog,
        text_artifact_load=text_artifact_load,
        command=server.command,
        command_sha256=command_sha256,
        server_log_sha256=_sha256_bytes(log_payload),
        server_log_size_bytes=len(log_payload),
        server_log_text=log_text,
        server_process_id=server.process.pid,
        resource_sample_summary=resource_sample_summary,
        started_at_utc=server.started_at_utc,
        completed_at_utc=_utc_now(),
    )


def run_bf16_interface_diagnostic(
    run_id: str,
    launch_intent_sha256: str,
) -> dict[str, Any]:
    """Run one authorized BF16 interface diagnostic on one exact B300:8 cell."""

    function_started = time.monotonic()
    work_deadline = function_started + FUNCTION_TIMEOUT_SECONDS - PUBLICATION_RESERVE_SECONDS
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("diagnostic run ID is invalid")
    if SHA256_RE.fullmatch(launch_intent_sha256) is None:
        raise ValueError("diagnostic launch-intent SHA-256 is invalid")

    bundle = _LOCAL_DIAGNOSTIC_BUNDLE
    if bundle is None:
        raise RuntimeError("diagnostic runner was not deployed in diagnostic mode")
    provenance = _validate_remote_diagnostic_provenance(_CONTROL_SHA256)
    evidence_volume.reload()
    intent = _load_diagnostic_intent(run_id, launch_intent_sha256)
    if intent.run_id != run_id or intent.intent_sha256() != launch_intent_sha256:
        raise RuntimeError("diagnostic launch intent differs from the requested invocation")
    _validate_diagnostic_paid_attempt_scope(
        bundle=bundle,
        provenance=provenance,
        intent=intent,
    )
    invocation = _invocation_ids()
    acceptance = _wait_for_diagnostic_acceptance(intent, call_id=invocation[0])
    binding = _claim_diagnostic_attempt(intent, acceptance, invocation)

    completed: list[DiagnosticStageName] = []
    runtime: MeasurementRuntimeIdentity | None = None
    hardware: MeasurementHardwareIdentity | None = None
    staged_subject: DiagnosticSubjectSpec | None = None
    server_load_publication_started = False
    server_load_claim: DiagnosticServerLoadClaim | None = None
    private_raw_reference: DiagnosticPrivateRawReference | None = None
    terminal_publication_started = False
    try:
        runtime = _runtime_identity(bundle)
        source = _verify_diagnostic_source_assets(bundle)
        source_subject = _diagnostic_subject_spec(bundle)
        _complete_diagnostic_stage(completed, "verify_references")

        hardware_payload = _observe_hardware(runtime)
        hardware = MeasurementHardwareIdentity.model_validate_json(
            canonical_measurement_raw_json_bytes(hardware_payload),
            strict=True,
        )
        expected_uuids = tuple(item.uuid for item in hardware.gpus)
        _complete_diagnostic_stage(completed, "verify_cuda_preflight")

        staged_subject = _stage_diagnostic_subject(
            source_subject,
            bundle=bundle,
            work_deadline=work_deadline,
        )
        _complete_diagnostic_stage(completed, "stage_and_rehash_bf16")

        server_load_publication_started = True
        server_load_claim = _claim_diagnostic_server_load(binding)
        server = _run_bf16_diagnostic_server(
            subject=staged_subject,
            source=source,
            bundle=bundle,
            expected_uuids=expected_uuids,
            work_deadline=work_deadline,
        )
        _complete_diagnostic_stage(completed, "bf16_interface_diagnostic")

        placement_policy = ExactCudaPlacementPolicy(
            schema_version="iql-exact-cuda-placement-policy-v1",
            gpu_count=bundle.config.resources.gpu_count,
            tensor_split=bundle.config.placement.tensor_split,
            split_mode=bundle.config.placement.split_mode,
            text_graph_policy="at_least_one_all_expected_cuda",
            vision_graph_policy="cuda0_only",
            audio_graph_policy="cuda0_only",
        )
        placement = parse_exact_text_cuda_backend_audit(
            server.server_log_text,
            policy=placement_policy,
        )
        _complete_diagnostic_stage(completed, "verify_gpu_placement")

        _release_diagnostic_subject(staged_subject)
        staged_subject = None
        _complete_diagnostic_stage(completed, "release_bf16")

        runtime_identity_sha256 = diagnostic_runtime_identity_sha256(runtime)
        raw = DiagnosticPrivateRawEvidence(
            **_diagnostic_binding_fields(
                binding,
                completed_at_utc=server.completed_at_utc,
                server_load_claim=server_load_claim,
            ),
            model_id=bundle.config.model_id,
            model_revision=bundle.config.revision,
            architecture=bundle.config.architecture,
            protocol_sha256=diagnostic_protocol_sha256(bundle.config),
            workload_sha256=diagnostic_workload_sha256(bundle.config),
            bf16_inventory_sha256=bundle.bf16.bf16_inventory_sha256,
            bf16_shard_count=49,
            bf16_total_bytes=bundle.bf16.bf16_total_bytes,
            source_asset_manifest_sha256=source.asset_manifest_sha256,
            runtime_identity=runtime,
            runtime_identity_sha256=runtime_identity_sha256,
            runtime_manifest_sha256=runtime.manifest_sha256,
            hardware_identity=hardware,
            hardware_identity_sha256=hardware.identity_sha256,
            command=server.command,
            command_sha256=server.command_sha256,
            server_process_id=server.server_process_id,
            server_log_sha256=server.server_log_sha256,
            server_log_size_bytes=server.server_log_size_bytes,
            text_artifact_load=server.text_artifact_load,
            backend="CUDA",
            logical_devices=bundle.config.placement.logical_devices,
            gpu_device_count=bundle.config.resources.gpu_count,
            gpu_model_graph_operation_count=placement.gpu_operations,
            cpu_model_graph_operation_count=0,
            cpu_fallback_observed=False,
            resource_sample_summary=server.resource_sample_summary,
            eog=server.eog,
            trials=server.trials,
            prompt_text_recorded=False,
            output_text_recorded=False,
            private_token_ids_recorded=True,
            one_server_load=True,
            sequential_request_count=16,
            started_at_utc=server.started_at_utc,
            diagnostic_only=True,
            quality_retention_claim_allowed=False,
            quality_claim_allowed=False,
            speedup_claim_allowed=False,
            performance_claim_allowed=False,
            mtp_included=False,
            mtp_supported=False,
            routing_drift_supported=False,
            single_run_causation_claim_allowed=False,
        )
        raw, private_raw_reference = _publish_diagnostic_private_raw(
            raw,
            bundle=bundle,
        )
        rollup = build_diagnostic_rollup(
            raw,
            private_raw_content_sha256=private_raw_reference.content_sha256,
        )
        success = DiagnosticSuccessTerminalReceipt(
            **_diagnostic_binding_fields(
                binding,
                completed_at_utc=_utc_now(),
                server_load_claim=server_load_claim,
            ),
            completed_stages=DIAGNOSTIC_PLANNED_STAGES,
            model_id=bundle.config.model_id,
            model_revision=bundle.config.revision,
            architecture=bundle.config.architecture,
            bf16_inventory_sha256=bundle.bf16.bf16_inventory_sha256,
            bf16_shard_count=49,
            bf16_total_bytes=bundle.bf16.bf16_total_bytes,
            protocol_sha256=diagnostic_protocol_sha256(bundle.config),
            workload_sha256=diagnostic_workload_sha256(bundle.config),
            runtime_identity=runtime,
            runtime_identity_sha256=runtime_identity_sha256,
            runtime_manifest_sha256=runtime.manifest_sha256,
            hardware_identity_sha256=hardware.identity_sha256,
            command_sha256=server.command_sha256,
            server_log_sha256=server.server_log_sha256,
            server_log_size_bytes=server.server_log_size_bytes,
            private_raw_reference=private_raw_reference,
            rollup=rollup,
            rollup_sha256=diagnostic_rollup_sha256(rollup),
            gpu_placement_verified=True,
            cpu_fallback_observed=False,
        )
        terminal_publication_started = True
        terminal = _publish_diagnostic_terminal_receipt(success, outcome="success")
        _complete_diagnostic_stage(completed, "publish")
        return {
            "status": "completed",
            "run_id": run_id,
            "call_id": binding.call_id,
            "terminal_receipt": terminal.model_dump(mode="json"),
            "diagnostic_completed": True,
            "gpu_placement_verified": True,
            "cpu_fallback_observed": False,
            "function_return_is_success_evidence": False,
        }
    except Exception as error:
        cpu_fallback_observed = isinstance(error, BackendCpuPlacementError)
        if staged_subject is not None:
            try:
                _release_diagnostic_subject(staged_subject)
            except BaseException as cleanup_error:
                error.add_note(
                    f"diagnostic BF16 staging cleanup also failed: {type(cleanup_error).__name__}"
                )
        server_load_publication_unknown = (
            server_load_publication_started and server_load_claim is None
        )
        if not terminal_publication_started and not server_load_publication_unknown:
            try:
                completed_count = len(completed)
                failed_stage = DIAGNOSTIC_PLANNED_STAGES[completed_count]
                if not _is_diagnostic_stage(failed_stage):
                    raise RuntimeError("control plane returned an unknown diagnostic stage")
                placement_stage_index = DIAGNOSTIC_PLANNED_STAGES.index("verify_gpu_placement")
                failure = DiagnosticFailureTerminalReceipt(
                    **_diagnostic_binding_fields(
                        binding,
                        completed_at_utc=_utc_now(),
                        server_load_claim=server_load_claim,
                    ),
                    completed_stages=tuple(completed),
                    failed_stage=failed_stage,
                    error_code=_failure_code(error),
                    error_summary_sha256=_failure_summary_sha256(error),
                    runtime_identity_sha256=(
                        diagnostic_runtime_identity_sha256(runtime) if runtime is not None else None
                    ),
                    runtime_manifest_sha256=(
                        runtime.manifest_sha256 if runtime is not None else None
                    ),
                    hardware_identity_sha256=(
                        hardware.identity_sha256 if hardware is not None else None
                    ),
                    private_raw_reference=private_raw_reference,
                    gpu_placement_verified=completed_count > placement_stage_index,
                    cpu_fallback_observed=cpu_fallback_observed,
                )
                terminal_publication_started = True
                _publish_diagnostic_terminal_receipt(failure, outcome="failure")
            except BaseException as publication_error:
                error.add_note(
                    "immutable diagnostic failure receipt publication also failed: "
                    f"{type(publication_error).__name__}"
                )
        raise


def run_measurement(
    run_id: str,
    launch_intent_sha256: str,
) -> dict[str, Any]:
    """Run one authorized BF16-then-Q3 measurement on one exact B300:8 cell."""

    function_started = time.monotonic()
    work_deadline = function_started + FUNCTION_TIMEOUT_SECONDS - PUBLICATION_RESERVE_SECONDS
    if RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError("measurement run ID is invalid")
    if SHA256_RE.fullmatch(launch_intent_sha256) is None:
        raise ValueError("measurement launch-intent SHA-256 is invalid")

    bundle = _LOCAL_BUNDLE
    if bundle is None:
        raise RuntimeError("measurement runner was not deployed in measurement mode")
    provenance = _validate_remote_provenance(_CONTROL_SHA256)
    evidence_volume.reload()
    intent = _load_intent(run_id, launch_intent_sha256)
    if intent.run_id != run_id or intent.intent_sha256() != launch_intent_sha256:
        raise RuntimeError("launch intent differs from the requested invocation")
    _validate_paid_attempt_scope(
        bundle=bundle,
        provenance=provenance,
        intent=intent,
    )
    invocation = _invocation_ids()
    acceptance = _wait_for_acceptance(intent, call_id=invocation[0])
    binding = _claim_attempt(intent, acceptance, invocation)

    completed: list[MeasurementStage] = []
    published_subjects: list[PublishedSubjectEvidence] = []
    runtime: MeasurementRuntimeIdentity | None = None
    terminal_publication_started = False
    try:
        runtime = _runtime_identity(bundle)
        corpus_identity = _materialized_corpus_identity(bundle)
        _complete_stage(completed, "verify_references")

        hardware_payload = _observe_hardware(runtime)
        hardware = MeasurementHardwareIdentity.model_validate_json(
            canonical_measurement_raw_json_bytes(hardware_payload),
            strict=True,
        )
        expected_uuids = tuple(item.uuid for item in hardware.gpus)
        _complete_stage(completed, "verify_cuda_preflight")

        bf16_subject, q3_subject = _subject_specs(bundle)
        subject_plans: tuple[
            tuple[
                SubjectSpec,
                MeasurementStage,
                MeasurementStage,
                MeasurementStage,
                MeasurementStage,
                int,
            ],
            tuple[
                SubjectSpec,
                MeasurementStage,
                MeasurementStage,
                MeasurementStage,
                MeasurementStage,
                int,
            ],
        ] = (
            (
                bf16_subject,
                "stage_and_rehash_bf16",
                "measure_bf16_quality",
                "measure_bf16_performance",
                "release_bf16",
                19_181,
            ),
            (
                q3_subject,
                "stage_and_rehash_q3",
                "measure_q3_quality",
                "measure_q3_performance",
                "release_q3",
                19_182,
            ),
        )
        for (
            source_subject,
            staging_stage,
            quality_stage,
            performance_stage,
            release_stage,
            port,
        ) in subject_plans:
            staged_subject, staging = _stage_subject(
                source_subject,
                bundle=bundle,
                work_deadline=work_deadline,
            )
            _complete_stage(completed, staging_stage)

            perplexity = _perplexity_measurement(
                subject=staged_subject,
                bundle=bundle,
                corpus_identity=corpus_identity,
                expected_uuids=expected_uuids,
                work_deadline=work_deadline,
            )
            server = _server_measurement(
                subject=staged_subject,
                bundle=bundle,
                expected_uuids=expected_uuids,
                hardware_identity_sha256=hardware.identity_sha256,
                port=port,
                work_deadline=work_deadline,
            )
            vocab_size = server.evidence.get("vocab_size")
            if type(vocab_size) is not int:
                raise RuntimeError("server evidence lacks its validated vocabulary size")
            quality = _quality_measurement(
                subject=staged_subject,
                perplexity=perplexity,
                diagnostics=server.diagnostics,
                vocab_size=vocab_size,
            )
            _complete_stage(completed, quality_stage)

            performance = _performance_measurement(
                subject=staged_subject,
                bundle=bundle,
                expected_uuids=expected_uuids,
                server_evidence=server.evidence,
                work_deadline=work_deadline,
            )
            measurement = SubjectMeasurementResult(
                source_subject=source_subject,
                staged_subject=staged_subject,
                staging=staging,
                quality=quality,
                performance=performance,
                server=server,
            )
            published = _publish_subject_evidence(
                measurement,
                bundle=bundle,
                binding=binding,
                runtime=runtime,
                hardware=hardware.model_dump(mode="json"),
            )
            published_subjects.append(published)
            _complete_stage(completed, performance_stage)

            _release_staged_subject(staged_subject)
            _complete_stage(completed, release_stage)

        if len(published_subjects) != 2:
            raise RuntimeError("measurement did not publish both subject records")
        bf16, q3 = published_subjects
        (
            _comparison,
            comparison_reference,
            quality_rollup,
            performance_rollup,
        ) = _publish_comparison_evidence(
            bundle=bundle,
            bf16=bf16,
            q3=q3,
        )
        success = MeasurementSuccessTerminalReceipt(
            **_terminal_binding_fields(
                binding,
                completed_at_utc=_utc_now(),
            ),
            completed_stages=MEASUREMENT_PLANNED_STAGES,
            runtime_identity=runtime,
            runtime_manifest_sha256=runtime.manifest_sha256,
            hardware_identity_sha256=hardware.identity_sha256,
            model_id=bundle.config.model_id,
            model_revision=bundle.config.revision,
            protocol_sha256=measurement_protocol_sha256(bundle.config),
            workload_sha256=measurement_workload_sha256(bundle.config),
            supporting_records=(
                bf16.reference,
                q3.reference,
                comparison_reference,
            ),
            quality_rollup=quality_rollup,
            performance_rollup=performance_rollup,
            quality_rollup_sha256=measurement_quality_rollup_sha256(quality_rollup),
            performance_rollup_sha256=(measurement_performance_rollup_sha256(performance_rollup)),
            quality_retention_passed=(quality_rollup.non_inferiority_passed),
            performance_comparison_complete=True,
            speedup_claim_allowed=True,
        )
        terminal_publication_started = True
        terminal = _publish_terminal_receipt(success, outcome="success")
        _complete_stage(completed, "compare_and_publish")
        return {
            "status": "completed",
            "run_id": run_id,
            "call_id": binding.call_id,
            "terminal_receipt": terminal.model_dump(mode="json"),
            "measurement_completed": True,
            "quality_retention_passed": success.quality_retention_passed,
            "performance_comparison_complete": True,
            "speedup_claim_allowed": True,
            "function_return_is_success_evidence": False,
        }
    except BaseException as error:
        if not terminal_publication_started:
            try:
                completed_count = len(completed)
                failed_stage = MEASUREMENT_PLANNED_STAGES[completed_count]
                if not _is_measurement_stage(failed_stage):
                    raise RuntimeError("control plane returned an unknown measurement stage")
                bf16_complete_index = MEASUREMENT_PLANNED_STAGES.index("measure_bf16_performance")
                q3_complete_index = MEASUREMENT_PLANNED_STAGES.index("measure_q3_performance")
                supporting_count = 0
                if completed_count > bf16_complete_index:
                    supporting_count = 1
                if completed_count > q3_complete_index:
                    supporting_count = 2
                supporting_records = tuple(
                    item.reference for item in published_subjects[:supporting_count]
                )
                failure = MeasurementFailureTerminalReceipt(
                    **_terminal_binding_fields(
                        binding,
                        completed_at_utc=_utc_now(),
                    ),
                    completed_stages=tuple(completed),
                    failed_stage=failed_stage,
                    failed_subject=_failed_subject(failed_stage),
                    error_code=_failure_code(error),
                    error_summary_sha256=_failure_summary_sha256(error),
                    supporting_records=supporting_records,
                    runtime_identity=(runtime if completed_count > 0 else None),
                )
                terminal_publication_started = True
                _publish_terminal_receipt(failure, outcome="failure")
            except BaseException as publication_error:
                error.add_note(
                    "immutable failure receipt publication also failed: "
                    f"{type(publication_error).__name__}"
                )
        raise


if _DIAGNOSTIC_MODE:
    run_bf16_interface_diagnostic = cast(
        Any,
        app.function(
            image=measurement_image,
            gpu="B300:8",
            cpu=16,
            memory=65_536,
            ephemeral_disk=2_097_152,
            retries=0,
            timeout=FUNCTION_TIMEOUT_SECONDS,
            startup_timeout=1_800,
            max_containers=1,
            single_use_containers=True,
            block_network=True,
            volumes=_FUNCTION_VOLUMES,
        )(run_bf16_interface_diagnostic),
    )
else:
    run_measurement = cast(
        Any,
        app.function(
            image=measurement_image,
            gpu="B300:8",
            cpu=16,
            memory=65_536,
            ephemeral_disk=2_097_152,
            retries=0,
            timeout=FUNCTION_TIMEOUT_SECONDS,
            startup_timeout=1_800,
            max_containers=1,
            single_use_containers=True,
            block_network=True,
            volumes=_FUNCTION_VOLUMES,
        )(run_measurement),
    )
