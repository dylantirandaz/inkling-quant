"""Offline contracts for an exact Inkling Q3 GGUF inference smoke test.

This module does not launch Modal or execute llama.cpp.  It binds a future smoke
run to the verified export, runtime, hardware, deterministic probes, and redacted
evidence contract.  The parsers turn runtime output into small records that do not
retain prompt or generated text.
"""

from __future__ import annotations

import csv
import ctypes
import hashlib
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal
from uuid import UUID

import yaml
from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.security import sensitive_literal_path

PINNED_MODEL_ID: Final = "thinkingmachines/Inkling"
PINNED_MODEL_REVISION: Final = "86b4d430ab871652a707666b89203a866888c5e5"
PINNED_ARCHITECTURE: Final = "InklingForConditionalGeneration"
PINNED_LLAMA_CPP_REPOSITORY: Final = "https://github.com/danielhanchen/llama.cpp.git"
PINNED_LLAMA_CPP_COMMIT: Final = "a015409e6c27b84f60d688823d4c0126a11571fd"
PINNED_VOCAB_SIZE: Final = 201_024
PINNED_UNPADDED_VOCAB_SIZE: Final = 200_058
PINNED_PADDED_VOCAB_SIZE: Final = PINNED_VOCAB_SIZE - PINNED_UNPADDED_VOCAB_SIZE
INSTRUMENTATION_SCHEMA_VERSION: Final = "inkling-llama-smoke-instrumentation-v2"
INSTRUMENTATION_PATCH_RELATIVE_PATH: Final = "patches/inkling-smoke-a015409.patch"
HISTORICAL_INSTRUMENTATION_PATCH_SHA256: Final = (
    "b276d12a4af96c803b71fee6f7be91c230b0fb30b6be04637f61f33d07b10ecf"
)
LEGACY_CURRENT_INSTRUMENTATION_PATCH_SHA256: Final = (
    "301023aea3a19533710e122fbbd55378bf19c2562bd885fa85b58f9d4ea110cb"
)
INSTRUMENTATION_PATCH_SHA256: Final = (
    "0f824d7a77b0e98816e6d62f982b010caada15f8a93a20343d5cc0d129bcca20"
)

SUBJECT_RUN_ID: Final = "inkling-q3km-86b4d430-a015409e-ffd466dd93-8083cf41e1"
SUBJECT_CONFIG_HASH: Final = "ffd466dd934005fa64d36e79e591f6351ccad709c5808828bbf0b65b90ae17fd"
SUBJECT_CONTROL_PLANE_SHA256: Final = (
    "8083cf41e104b3f7164c02a1ad50ab027f630167970c4eb7e0589a6d079c1037"
)
SUBJECT_VERIFY_CALL_ID: Final = "fc-01KY1S3K3XGTG8DT383ZGRDGGR"
SUBJECT_QUANTIZE_CALL_ID: Final = "fc-01KY1FYG09CNTCB6AFR3T6PHFA"
SUBJECT_MMPROJ_CALL_ID: Final = "fc-01KY1AFWNMXSAATQMDM4GP1QCY"

EXPECTED_Q3_SHARD_COUNT: Final = 49
EXPECTED_Q3_TOTAL_BYTES: Final = 451_035_400_288
EXPECTED_Q3_INVENTORY_SHA256: Final = (
    "3643d7e34ed8d3d8d216f7a42bf23daa6511fd53376cd34b2369b2fe1c17d55e"
)
EXPECTED_PROJECTOR_PATH: Final = "mmproj/mmproj-BF16.gguf"
EXPECTED_PROJECTOR_SHA256: Final = (
    "8f954d089a753671321316bd4fbcffae6465748814ca6c1ec3e70f62427514f7"
)
EXPECTED_PROJECTOR_BYTES: Final = 183_264_288
EXPECTED_FIRST_Q3_MOUNT_PATH: Final = "/subject/q3_k_m/inkling-Q3_K_M-00001-of-00049.gguf"
EXPECTED_PROJECTOR_MOUNT_PATH: Final = "/subject/mmproj/mmproj-BF16.gguf"

PINNED_CUDA_IMAGE: Final = "nvidia/cuda:13.1.2-devel-ubuntu24.04"
PINNED_CUDA_IMAGE_DIGEST: Final = (
    "sha256:952e42d23230610a2714c8484f38e9c934ed68e6f9c9c7fac62dcd5f98858a6e"
)
PINNED_CUDA_PLATFORM: Final = "linux/amd64"
B300_COMPUTE_CAPABILITY: Final = "10.3"
B300_CMAKE_ARCHITECTURE: Final = "103"
LLAMA_SERVER_AUDIT_LOG_VERBOSITY: Final = 4
CUDA_DRIVER_STUB_RPATH_LINK_DEFINITION: Final = (
    "CMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/iql-cuda-driver-link"
)

VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH: Final = (
    "configs/experiments/inkling_q3_k_m_verified_export.json"
)
SMOKE_CONFIG_RELATIVE_PATH: Final = "configs/experiments/inkling_q3_k_m_smoke_modal.yaml"
VERIFIED_EXPORT_HASH_DOMAIN: Final = b"inkling-verified-export-reference-v1\0"

# Filled with the checked reference's self-hash.  This is deliberately a constant,
# rather than accepting any valid self-hashed record, so a changed subject requires
# a new code/config review.
EXPECTED_VERIFIED_EXPORT_REFERENCE_SHA256: Final = (
    "9f0fae0a48058e73aab38c2b4f6c86916b69fd32343e0f7b821c7faac5b33198"
)

_EXPECTED_BUILD_TARGETS: Final = (
    "llama-cli",
    "llama-server",
    "llama-bench",
    "llama-perplexity",
)
_EXPECTED_CMAKE_DEFINITIONS: Final = (
    "GGML_CUDA=ON",
    "GGML_NATIVE=OFF",
    "LLAMA_CURL=OFF",
    "LLAMA_BUILD_UI=OFF",
    "LLAMA_USE_PREBUILT_UI=OFF",
    f"CMAKE_CUDA_ARCHITECTURES={B300_CMAKE_ARCHITECTURE}",
    CUDA_DRIVER_STUB_RPATH_LINK_DEFINITION,
)
_NO_GPU_WARNING: Final = "warning: no usable GPU found, --gpu-layers option will be ignored"
_CUDA_DEVICE_COUNT_RE: Final = re.compile(r"ggml_cuda_init: found ([0-9]+) CUDA devices\b")
_OFFLOAD_RE: Final = re.compile(r"load_tensors: offloaded ([0-9]+)/([0-9]+) layers to GPU")
_OUTPUT_OFFLOAD_TEXT: Final = "load_tensors: offloading output layer to GPU"
_HISTORICAL_RAW_LOGIT_AUDIT_MARKER: Final = "IQL_SMOKE_RAW_LOGITS_V1"
_HISTORICAL_RAW_LOGIT_AUDIT_RE: Final = re.compile(
    rf"{_HISTORICAL_RAW_LOGIT_AUDIT_MARKER} task_id=(-?[0-9]+) slot_id=(-?[0-9]+) "
    r"completion_index=(-?[0-9]+) batch_index=(-?[0-9]+) count=([0-9]+) "
    r"finite=([0-9]+) nan=([0-9]+) pos_inf=([0-9]+) neg_inf=([0-9]+)"
)
_RAW_LOGIT_AUDIT_MARKER: Final = "IQL_SMOKE_RAW_LOGITS_V2"
_RAW_LOGIT_AUDIT_RE: Final = re.compile(
    rf"{_RAW_LOGIT_AUDIT_MARKER} task_id=(-?[0-9]+) slot_id=(-?[0-9]+) "
    r"completion_index=(-?[0-9]+) batch_index=(-?[0-9]+) count=([0-9]+) "
    r"unpadded_count=([0-9]+) padded_count=([0-9]+) "
    r"unpadded_finite=([0-9]+) unpadded_nan=([0-9]+) "
    r"unpadded_pos_inf=([0-9]+) unpadded_neg_inf=([0-9]+) "
    r"padded_finite=([0-9]+) padded_nan=([0-9]+) "
    r"padded_pos_inf=([0-9]+) padded_neg_inf=([0-9]+)"
)
_BACKEND_GRAPH_MARKER: Final = "IQL_SMOKE_BACKEND_GRAPH_V1"
_BACKEND_GRAPH_RE: Final = re.compile(
    rf"{_BACKEND_GRAPH_MARKER} graph_uid=([0-9]+) "
    r"phase=(post_assignment_pre_split) scope=(non_view_compute) "
    r"compute=([0-9]+) gpu=([0-9]+) cpu=([0-9]+) accel=([0-9]+) "
    r"other=([0-9]+) unassigned=([0-9]+)"
)
_BACKEND_IDENTITY_MARKER: Final = "IQL_SMOKE_BACKEND_IDENTITY_V1"
_BACKEND_IDENTITY_RE: Final = re.compile(
    rf"{_BACKEND_IDENTITY_MARKER} graph_uid=([0-9]+) backend_index=(-?[0-9]+) "
    r"backend_name=([^\s]+) device_name=([^\s]+) "
    r"device_type=(cpu|gpu|igpu|accel|meta|unassigned) compute=([0-9]+)"
)
_CPU_NODE_MARKER: Final = "IQL_SMOKE_CPU_NODE_V1"
_CPU_NODE_RE: Final = re.compile(
    rf"{_CPU_NODE_MARKER} graph_uid=([0-9]+) ordinal=(-?[0-9]+) "
    r"op=([^\s]+) name=([^\s]*)"
)
_FIRST_SHARD_LOAD_RE: Final = re.compile(
    r"llama_model_loader: loaded meta data with [^\n]* from ([^\s]+)"
)
_ADDITIONAL_SHARD_LOAD_RE: Final = re.compile(
    r"llama_model_loader: additional ([0-9]+) GGUFs metadata loaded\."
)
_PROJECTOR_LOAD_RE: Final = re.compile(r"loaded multimodal model, '([^']+)'")
_TEXT_SHARDS_MARKER: Final = "IQL_SMOKE_TEXT_SHARDS_V1"
_TEXT_SHARDS_RE: Final = re.compile(
    rf"{_TEXT_SHARDS_MARKER} expected=([0-9]+) opened=([0-9]+) "
    r"contexts=([0-9]+) tensors=([0-9]+)"
)
_TEXT_LOAD_MARKER: Final = "IQL_SMOKE_TEXT_LOAD_V1"
_TEXT_LOAD_RE: Final = re.compile(
    rf"{_TEXT_LOAD_MARKER} opened=([0-9]+) accounted=([0-9]+) "
    r"tensors=([0-9]+) bytes=([0-9]+) size_done=([0-9]+) "
    r"size_data=([0-9]+) mmap=([01])"
)
_PROJECTOR_TENSORS_MARKER: Final = "IQL_SMOKE_PROJECTOR_TENSORS_V1"
_PROJECTOR_TENSORS_RE: Final = re.compile(
    rf"{_PROJECTOR_TENSORS_MARKER} modality=(vision|audio) "
    r"projector=(inkling|other) tensors=([0-9]+) bytes=([0-9]+)"
)
_PROJECTOR_READY_MARKER: Final = "IQL_SMOKE_PROJECTOR_READY_V1"
_PROJECTOR_READY_RE: Final = re.compile(
    rf"{_PROJECTOR_READY_MARKER} opened=([01]) vision=([01]) audio=([01]) "
    r"vision_type=(inkling|other) audio_type=(inkling|other) n_embd=(-?[0-9]+)"
)
_CUDA_DRIVER_LINKAGE_RE: Final = re.compile(
    r"^[ \t]*libcuda\.so\.1 => (/[^\s]+) \(0x[0-9A-Fa-f]+\)[ \t]*$",
    re.MULTILINE,
)
_CUDA_DRIVER_STUB_ROOT: Final = PurePosixPath("/usr/local/cuda/lib64/stubs")
_CUDA_DRIVER_LINK_ROOT: Final = PurePosixPath("/opt/iql-cuda-driver-link")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _canonical_relative_path(value: str, *, label: str) -> str:
    if not value or "\x00" in value or "\\" in value or "//" in value:
        raise ValueError(f"{label} must be canonical relative POSIX text")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"{label} must be relative and contain no traversal")
    if parsed.as_posix() != value:
        raise ValueError(f"{label} must be canonical")
    return value


class VerifiedExportArtifact(StrictFrozenModel):
    """One immutable file under the verified final-run root."""

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(gt=0)

    @field_validator("path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return _canonical_relative_path(value, label="verified export artifact path")


class InklingVerifiedExportReference(StrictFrozenModel):
    """Self-hashed record for the only export accepted by the smoke contract."""

    schema_version: Literal["inkling-verified-export-reference-v1"]
    verified: Literal[True]
    subject_run_id: Literal["inkling-q3km-86b4d430-a015409e-ffd466dd93-8083cf41e1"]
    subject_config_hash: Literal["ffd466dd934005fa64d36e79e591f6351ccad709c5808828bbf0b65b90ae17fd"]
    subject_control_plane_sha256: Literal[
        "8083cf41e104b3f7164c02a1ad50ab027f630167970c4eb7e0589a6d079c1037"
    ]
    verify_call_id: Literal["fc-01KY1S3K3XGTG8DT383ZGRDGGR"]
    quantize_call_id: Literal["fc-01KY1FYG09CNTCB6AFR3T6PHFA"]
    mmproj_call_id: Literal["fc-01KY1AFWNMXSAATQMDM4GP1QCY"]
    model_id: Literal["thinkingmachines/Inkling"]
    revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"]
    architecture: Literal["InklingForConditionalGeneration"]
    license: Literal["apache-2.0"]
    llama_cpp_repository: Literal["https://github.com/danielhanchen/llama.cpp.git"]
    llama_cpp_commit: Literal["a015409e6c27b84f60d688823d4c0126a11571fd"]
    final_volume: Literal["inkling-final-v1"]
    final_run_subpath: Literal["runs/inkling-q3km-86b4d430-a015409e-ffd466dd93-8083cf41e1"]
    quant_type: Literal["Q3_K_M"]
    mtp: Literal["omitted_unsupported"]
    quality_measured: Literal[False]
    deployment_benchmark_measured: Literal[False]
    q3_shard_count: Literal[49]
    q3_total_bytes: Literal[451035400288]
    q3_inventory_sha256: Literal["3643d7e34ed8d3d8d216f7a42bf23daa6511fd53376cd34b2369b2fe1c17d55e"]
    q3_shards: tuple[VerifiedExportArtifact, ...]
    projector: VerifiedExportArtifact
    export_manifest: VerifiedExportArtifact
    verify_receipt: VerifiedExportArtifact
    quantize_receipt: VerifiedExportArtifact
    mmproj_receipt: VerifiedExportArtifact
    reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def canonical_payload_dict(self) -> dict[str, Any]:
        """Return the content covered by the self-hash."""

        return self.model_dump(mode="json", exclude={"reference_sha256"})

    def computed_reference_sha256(self) -> str:
        """Compute the domain-separated self-hash."""

        return verified_export_reference_sha256(self.canonical_payload_dict())

    def canonical_json(self) -> str:
        """Serialize the complete reference as canonical JSON."""

        return _canonical_json(self.model_dump(mode="json"))

    @model_validator(mode="after")
    def matches_exact_verified_export(self) -> InklingVerifiedExportReference:
        expected_paths = tuple(
            f"q3_k_m/inkling-Q3_K_M-{index:05d}-of-00049.gguf"
            for index in range(1, EXPECTED_Q3_SHARD_COUNT + 1)
        )
        observed_paths = tuple(artifact.path for artifact in self.q3_shards)
        if observed_paths != expected_paths:
            raise ValueError("Q3 shard paths must be the exact ordered 49-file set")
        if len(set(observed_paths)) != EXPECTED_Q3_SHARD_COUNT:
            raise ValueError("Q3 shard paths must be unique")
        if sum(artifact.size_bytes for artifact in self.q3_shards) != self.q3_total_bytes:
            raise ValueError("Q3 shard sizes do not equal the verified total")
        inventory = [artifact.model_dump(mode="json") for artifact in self.q3_shards]
        inventory_sha256 = hashlib.sha256(_canonical_json(inventory).encode()).hexdigest()
        if inventory_sha256 != self.q3_inventory_sha256:
            raise ValueError("Q3 shard inventory differs from the verified manifest")

        exact_artifacts = {
            "projector": {
                "path": EXPECTED_PROJECTOR_PATH,
                "sha256": EXPECTED_PROJECTOR_SHA256,
                "size_bytes": EXPECTED_PROJECTOR_BYTES,
            },
            "export_manifest": {
                "path": "verification/export_manifest.json",
                "sha256": "23db1314d521210bab5d53df20ed432f784774c59d98e8db3de9004702e1ac7a",
                "size_bytes": 11967,
            },
            "verify_receipt": {
                "path": "verify_export.success.json",
                "sha256": "08b4928333720962e1192ef0af12672c8155c70ddc03813376cbd431c2409291",
                "size_bytes": 1187,
            },
            "quantize_receipt": {
                "path": "quantize_text.success.json",
                "sha256": "c823baccc7f124ac4c8c05d01e19f0ad4c1bc0b499619840eb365edf2efaa6a5",
                "size_bytes": 11454,
            },
            "mmproj_receipt": {
                "path": "convert_multimodal_projector.success.json",
                "sha256": "677af03e216f7299449bf0caa6e48fd0b277da0e18ee955997da86761083c25b",
                "size_bytes": 2067,
            },
        }
        mismatches = [
            name
            for name, expected in exact_artifacts.items()
            if getattr(self, name).model_dump(mode="json") != expected
        ]
        if mismatches:
            raise ValueError("verified export artifacts differ: " + ", ".join(sorted(mismatches)))
        if self.reference_sha256 != self.computed_reference_sha256():
            raise ValueError("verified export reference self-hash does not match its payload")
        return self


def verified_export_reference_sha256(value: Mapping[str, Any]) -> str:
    """Hash a verified export reference without trusting its own hash field."""

    payload = dict(value)
    payload.pop("reference_sha256", None)
    return hashlib.sha256(
        VERIFIED_EXPORT_HASH_DOMAIN + _canonical_json(payload).encode("utf-8")
    ).hexdigest()


def load_verified_export_reference(path: str | Path) -> InklingVerifiedExportReference:
    """Load a byte-canonical reference for the one verified Inkling export."""

    reference_path = Path(path)
    try:
        raw_bytes = reference_path.read_bytes()
        raw = json.loads(raw_bytes)
        if not isinstance(raw, Mapping):
            raise ValueError("reference root must be a JSON object")
        reference = InklingVerifiedExportReference.model_validate(raw)
    except (OSError, ValueError, ValidationError) as error:
        raise ConfigurationError(
            f"Unable to load verified Inkling export reference {reference_path}: {error}",
            component="inkling_smoke_export",
        ) from error
    if raw_bytes != (reference.canonical_json() + "\n").encode("utf-8"):
        raise ConfigurationError(
            "Verified Inkling export reference must use canonical JSON plus one newline",
            component="inkling_smoke_export",
        )
    return reference


class SmokeCudaImageConfig(StrictFrozenModel):
    """Digest-pinned CUDA build image."""

    image: Literal["nvidia/cuda:13.1.2-devel-ubuntu24.04"]
    digest: Literal["sha256:952e42d23230610a2714c8484f38e9c934ed68e6f9c9c7fac62dcd5f98858a6e"]
    platform: Literal["linux/amd64"]


class SmokeRuntimeConfig(StrictFrozenModel):
    """Exact experimental llama.cpp build and serving contract."""

    repository: Literal["https://github.com/danielhanchen/llama.cpp.git"]
    commit: Literal["a015409e6c27b84f60d688823d4c0126a11571fd"]
    instrumentation_schema_version: Literal[
        "inkling-llama-smoke-instrumentation-v1",
        "inkling-llama-smoke-instrumentation-v2",
    ]
    instrumentation_patch_path: Literal["patches/inkling-smoke-a015409.patch"]
    instrumentation_patch_sha256: Literal[
        "b276d12a4af96c803b71fee6f7be91c230b0fb30b6be04637f61f33d07b10ecf",
        "301023aea3a19533710e122fbbd55378bf19c2562bd885fa85b58f9d4ea110cb",
        "0f824d7a77b0e98816e6d62f982b010caada15f8a93a20343d5cc0d129bcca20",
    ]
    image: SmokeCudaImageConfig
    build_targets: tuple[str, ...]
    cmake_definitions: tuple[str, ...]
    server_endpoint: Literal["/completion"]
    log_verbosity: Literal[4]
    context_size: Literal[8192]
    gpu_layers: Literal["all"]
    split_mode: Literal["layer"]
    tensor_split: tuple[Literal[1], Literal[1]]
    no_cpu_fallback: Literal[True]
    network_access: Literal[False]
    trust_remote_code: Literal[False]

    @model_validator(mode="after")
    def exact_build(self) -> SmokeRuntimeConfig:
        if self.build_targets != _EXPECTED_BUILD_TARGETS:
            raise ValueError("smoke build targets must equal the checked executable set")
        if self.cmake_definitions != _EXPECTED_CMAKE_DEFINITIONS:
            raise ValueError("smoke CMake definitions must target B300 compute capability 10.3")
        legacy_patch_hashes = {
            HISTORICAL_INSTRUMENTATION_PATCH_SHA256,
            LEGACY_CURRENT_INSTRUMENTATION_PATCH_SHA256,
        }
        if self.instrumentation_patch_sha256 in legacy_patch_hashes:
            if self.instrumentation_schema_version != "inkling-llama-smoke-instrumentation-v1":
                raise ValueError("legacy instrumentation patches require schema version 1")
        elif (
            self.instrumentation_patch_sha256 != INSTRUMENTATION_PATCH_SHA256
            or self.instrumentation_schema_version != INSTRUMENTATION_SCHEMA_VERSION
        ):
            raise ValueError("current instrumentation patch requires schema version 2")
        return self


class SmokeResourcesConfig(StrictFrozenModel):
    """Exact Modal resource cell to validate."""

    provider: Literal["modal"]
    gpu_type: Literal["B300"]
    gpu_count: Literal[2]
    compute_capability: Literal["10.3"]
    cpu_cores: Literal[16]
    memory_gib: Literal[64]
    ephemeral_disk_mib: Literal[524288]
    startup_timeout_seconds: Literal[900]
    max_hours: Literal[2]
    max_attempts: Literal[1]
    max_recovery_attempts: Literal[0]


class SmokeStorageConfig(StrictFrozenModel):
    """Mount permissions for subject input and smoke evidence."""

    final_volume: Literal["inkling-final-v1"]
    final_run_subpath: Literal["runs/inkling-q3km-86b4d430-a015409e-ffd466dd93-8083cf41e1"]
    final_mount_path: Literal["/subject"]
    final_read_only: Literal[True]
    evidence_volume: Literal["inkling-smoke-evidence-v1"]
    evidence_mount_path: Literal["/evidence"]
    evidence_append_only_after_success: Literal[True]
    attempt_registry: Literal["inkling-smoke-attempt-registry-v1"]
    attempt_registry_append_only: Literal[True]


class SmokeProbeConfig(StrictFrozenModel):
    """One deterministic probe; prompt text is input, never output evidence."""

    probe_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    modality: Literal["text", "image", "audio"]
    prompt: str = Field(min_length=1, max_length=512)
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture: Literal[
        "none",
        "synthetic_rgb8_png_16x16_checkerboard_v1",
        "synthetic_pcm_s16le_wav_16000hz_mono_silence_250ms_v1",
    ]
    seed: Literal[42]
    temperature: float
    n_predict: Literal[8]
    n_probs: Literal[5]
    post_sampling_probs: Literal[False]
    stream: Literal[False]
    cache_prompt: Literal[False]
    return_tokens: Literal[True]
    timings_per_token: Literal[True]
    trials: Literal[2]

    @model_validator(mode="after")
    def fixture_and_prompt_match(self) -> SmokeProbeConfig:
        if self.temperature != 0.0:
            raise ValueError("smoke probe temperature must use llama.cpp greedy mode (0.0)")
        expected_fixture = {
            "text": "none",
            "image": "synthetic_rgb8_png_16x16_checkerboard_v1",
            "audio": "synthetic_pcm_s16le_wav_16000hz_mono_silence_250ms_v1",
        }[self.modality]
        if self.fixture != expected_fixture:
            raise ValueError("probe fixture does not match its modality")
        digest = hashlib.sha256(self.prompt.encode("utf-8")).hexdigest()
        if self.prompt_sha256 != digest:
            raise ValueError("probe prompt SHA-256 does not match prompt bytes")
        return self


class SmokeEvidencePolicy(StrictFrozenModel):
    """Privacy and required payload fields for an immutable smoke receipt."""

    record_prompt_text: Literal[False]
    record_output_text: Literal[False]
    record_token_ids: Literal[True]
    record_logprob_summary: Literal[True]
    record_command_arguments: Literal[True]
    record_artifact_hashes: Literal[True]
    record_runtime_commit: Literal[True]
    record_hardware: Literal[True]
    record_timings: Literal[True]
    record_peak_memory: Literal[True]


class SmokeClaimLimits(StrictFrozenModel):
    """Claims that this small stage is explicitly forbidden to make."""

    purpose: Literal["load_and_inference_smoke_only"]
    mtp_included: Literal[False]
    mtp_supported: Literal[False]
    quality_measured: Literal[False]
    benchmark_measured: Literal[False]
    performance_claim_allowed: Literal[False]
    quality_retention_claim_allowed: Literal[False]
    compatibility_scope: Literal["single_exact_matrix_cell"]


class SmokeOutputVocabularyConfig(StrictFrozenModel):
    """Exact Inkling output vocabulary and padded-logit policy."""

    schema_version: Literal["inkling-output-vocabulary-v1"]
    vocab_size: Literal[201024]
    unpadded_vocab_size: Literal[200058]
    gguf_metadata_key: Literal["inkling.unpadded_vocab_size"]
    padded_suffix_policy: Literal["negative_infinity"]

    @property
    def padded_vocab_size(self) -> int:
        """Return the number of masked output rows."""

        return self.vocab_size - self.unpadded_vocab_size


class InklingSmokeConfig(StrictFrozenModel):
    """Checked pure-data plan for the first real Inkling inference smoke test."""

    schema_version: Literal["inkling-smoke-config-v1", "inkling-smoke-config-v2"]
    verified_export_reference_path: Literal[
        "configs/experiments/inkling_q3_k_m_verified_export.json"
    ]
    verified_export_reference_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime: SmokeRuntimeConfig
    output_vocabulary: SmokeOutputVocabularyConfig | None = None
    resources: SmokeResourcesConfig
    storage: SmokeStorageConfig
    probes: tuple[SmokeProbeConfig, ...]
    evidence: SmokeEvidencePolicy
    claims: SmokeClaimLimits

    @model_validator(mode="after")
    def exact_contract(self) -> InklingSmokeConfig:
        if self.verified_export_reference_sha256 != EXPECTED_VERIFIED_EXPORT_REFERENCE_SHA256:
            raise ValueError("smoke config does not bind the checked verified-export reference")
        if self.schema_version == "inkling-smoke-config-v1":
            if self.output_vocabulary is not None:
                raise ValueError("smoke config version 1 must not define output vocabulary")
            if self.runtime.instrumentation_schema_version != (
                "inkling-llama-smoke-instrumentation-v1"
            ):
                raise ValueError("smoke config version 1 requires instrumentation version 1")
        else:
            if self.output_vocabulary is None:
                raise ValueError("smoke config version 2 requires output vocabulary")
            if self.runtime.instrumentation_schema_version != INSTRUMENTATION_SCHEMA_VERSION:
                raise ValueError("smoke config version 2 requires instrumentation version 2")
        expected_probe_identity = (
            ("text_greedy_v1", "text"),
            ("image_greedy_v1", "image"),
            ("audio_greedy_v1", "audio"),
        )
        observed = tuple((probe.probe_id, probe.modality) for probe in self.probes)
        if observed != expected_probe_identity:
            raise ValueError("smoke probes must be the exact ordered text/image/audio set")
        literal_secret = sensitive_literal_path(self.model_dump(mode="json"))
        if literal_secret is not None:
            raise ValueError(
                "smoke configuration contains literal credential material at "
                + ".".join(literal_secret)
            )
        return self

    def canonical_dict(self) -> dict[str, Any]:
        value = self.model_dump(mode="json")
        if self.schema_version == "inkling-smoke-config-v1":
            value.pop("output_vocabulary", None)
        return value

    def canonical_json(self) -> str:
        return _canonical_json(self.canonical_dict())

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def redacted_smoke_config_record(config: InklingSmokeConfig) -> dict[str, Any]:
    """Return a resolved-config record that never retains probe prompt text."""

    resolved = config.canonical_dict()
    probes = resolved.get("probes")
    if not isinstance(probes, list):
        raise RuntimeError("internal smoke config probe representation is invalid")
    for probe in probes:
        if not isinstance(probe, dict):
            raise RuntimeError("internal smoke probe representation is invalid")
        prompt = probe.pop("prompt", None)
        prompt_sha256 = probe.get("prompt_sha256")
        if (
            not isinstance(prompt, str)
            or not isinstance(prompt_sha256, str)
            or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != prompt_sha256
        ):
            raise RuntimeError("internal smoke prompt identity is invalid")
        probe["prompt_text_recorded"] = False
        probe["prompt_utf8_bytes"] = len(prompt.encode("utf-8"))
    return {
        "schema_version": "inkling-smoke-resolved-config-redacted-v1",
        "smoke_config_hash": config.config_hash(),
        "prompt_text_recorded": False,
        "resolved_config": resolved,
    }


def load_inkling_smoke_config(path: str | Path) -> InklingSmokeConfig:
    """Load the checked smoke YAML without launching any external work."""

    config_path = Path(path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("smoke config root must be a mapping")
        return InklingSmokeConfig.model_validate(raw)
    except (OSError, ValueError, yaml.YAMLError, ValidationError) as error:
        raise ConfigurationError(
            f"Unable to load Inkling smoke config {config_path}: {error}",
            component="inkling_smoke_config",
        ) from error


class ServerCompletionEvidence(StrictFrozenModel):
    """Redacted proof that a completion returned finite token log probabilities."""

    token_ids: tuple[int, ...]
    tokens_predicted: int = Field(gt=0)
    minimum_logprob: float
    maximum_logprob: float
    all_returned_logprobs_finite: Literal[True] = True
    prompt_text_recorded: Literal[False] = False
    output_text_recorded: Literal[False] = False


def _exact_int(value: object, *, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _finite_logprob(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed > 0.0:
        raise ValueError(f"{label} must be finite and no greater than zero")
    return parsed


def _completion_probability_item(
    value: object,
    *,
    label: str,
    vocab_size: int,
) -> tuple[int, float, tuple[int, ...]]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    token_id = _exact_int(value.get("id"), label=f"{label}.id")
    if not 0 <= token_id < vocab_size:
        raise ValueError(f"{label}.id is outside the vocabulary")
    token = value.get("token")
    if not isinstance(token, str):
        raise ValueError(f"{label}.token must be text")
    token_bytes = value.get("bytes")
    if not isinstance(token_bytes, Sequence) or isinstance(token_bytes, str | bytes):
        raise ValueError(f"{label}.bytes must be a byte list")
    if any(type(item) is not int or not 0 <= item <= 255 for item in token_bytes):
        raise ValueError(f"{label}.bytes contains a non-byte value")
    logprob = _finite_logprob(value.get("logprob"), label=f"{label}.logprob")
    top = value.get("top_logprobs")
    if not isinstance(top, Sequence) or isinstance(top, str | bytes):
        raise ValueError(f"{label}.top_logprobs must be a list")
    top_ids: list[int] = []
    for index, entry in enumerate(top):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{label}.top_logprobs[{index}] must be an object")
        top_id = _exact_int(entry.get("id"), label=f"{label}.top_logprobs[{index}].id")
        if not 0 <= top_id < vocab_size:
            raise ValueError(f"{label}.top_logprobs[{index}].id is outside the vocabulary")
        if not isinstance(entry.get("token"), str):
            raise ValueError(f"{label}.top_logprobs[{index}].token must be text")
        entry_bytes = entry.get("bytes")
        if not isinstance(entry_bytes, Sequence) or isinstance(entry_bytes, str | bytes):
            raise ValueError(f"{label}.top_logprobs[{index}].bytes must be a byte list")
        if any(type(item) is not int or not 0 <= item <= 255 for item in entry_bytes):
            raise ValueError(f"{label}.top_logprobs[{index}].bytes has a non-byte value")
        _finite_logprob(entry.get("logprob"), label=f"{label}.top_logprobs[{index}].logprob")
        top_ids.append(top_id)
    if len(set(top_ids)) != len(top_ids):
        raise ValueError(f"{label}.top_logprobs contains duplicate token IDs")
    return token_id, logprob, tuple(top_ids)


def parse_server_completion(
    payload: Mapping[str, Any],
    *,
    vocab_size: int,
    expected_n_probs: int,
    unpadded_vocab_size: int | None = None,
) -> ServerCompletionEvidence:
    """Validate a non-streaming llama-server completion and discard all text."""

    active_vocab_size = vocab_size if unpadded_vocab_size is None else unpadded_vocab_size
    if (
        vocab_size <= 0
        or active_vocab_size <= 0
        or active_vocab_size > vocab_size
        or expected_n_probs <= 0
        or expected_n_probs > active_vocab_size
    ):
        raise ValueError("vocabulary size and n_probs must be positive and consistent")
    if not isinstance(payload.get("content"), str):
        raise ValueError("completion content must be text")
    tokens = payload.get("tokens")
    if not isinstance(tokens, Sequence) or isinstance(tokens, str | bytes) or not tokens:
        raise ValueError("completion tokens must be a non-empty list")
    token_ids = tuple(_exact_int(token, label="completion token") for token in tokens)
    if any(not 0 <= token < active_vocab_size for token in token_ids):
        raise ValueError("completion token is outside the unpadded vocabulary")
    tokens_predicted = _exact_int(
        payload.get("tokens_predicted"), label="completion tokens_predicted"
    )
    if tokens_predicted != len(token_ids):
        raise ValueError("tokens_predicted does not match returned token IDs")
    probabilities = payload.get("completion_probabilities")
    if not isinstance(probabilities, Sequence) or isinstance(probabilities, str | bytes):
        raise ValueError("completion_probabilities must be a list")
    if len(probabilities) != len(token_ids):
        raise ValueError("completion probabilities do not match returned token IDs")
    logprobs: list[float] = []
    for index, value in enumerate(probabilities):
        token_id, logprob, top_ids = _completion_probability_item(
            value,
            label=f"completion_probabilities[{index}]",
            vocab_size=active_vocab_size,
        )
        if token_id != token_ids[index]:
            raise ValueError("completion probability token ID does not match returned token")
        if len(top_ids) != expected_n_probs:
            raise ValueError("top_logprobs length differs from requested n_probs")
        if not top_ids or token_id != top_ids[0]:
            raise ValueError("greedy token is not the top-ranked returned probability")
        logprobs.append(logprob)
    return ServerCompletionEvidence(
        token_ids=token_ids,
        tokens_predicted=tokens_predicted,
        minimum_logprob=min(logprobs),
        maximum_logprob=max(logprobs),
    )


class HistoricalRawLogitAuditRow(StrictFrozenModel):
    """One full-finiteness vector emitted by instrumentation version 1."""

    task_id: int = Field(ge=0)
    slot_id: int = Field(ge=0)
    completion_index: int = Field(gt=0)
    batch_index: int = Field(ge=0)
    count: int = Field(gt=0)
    finite: int = Field(gt=0)
    nan: Literal[0]
    pos_inf: Literal[0]
    neg_inf: Literal[0]

    @model_validator(mode="after")
    def complete_and_finite(self) -> HistoricalRawLogitAuditRow:
        if self.finite != self.count:
            raise ValueError("raw logit vector contains a non-finite value")
        return self


class HistoricalRawLogitAuditEvidence(StrictFrozenModel):
    """Historical proof that every value in each output vector was finite."""

    schema_version: Literal["inkling-raw-logit-audit-v1"] = "inkling-raw-logit-audit-v1"
    expected_generated_token_vectors: int = Field(gt=0)
    observed_generated_token_vectors: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    rows: tuple[HistoricalRawLogitAuditRow, ...]
    all_rows_complete: Literal[True] = True
    all_values_finite: Literal[True] = True

    @model_validator(mode="after")
    def exact_vector_set(self) -> HistoricalRawLogitAuditEvidence:
        if not (
            self.expected_generated_token_vectors
            == self.observed_generated_token_vectors
            == len(self.rows)
        ):
            raise ValueError("raw logit vector count differs from generated-token evidence")
        if any(row.count != self.vocab_size for row in self.rows):
            raise ValueError("raw logit vector does not cover the full vocabulary")
        identities = tuple(
            (row.task_id, row.slot_id, row.completion_index, row.batch_index) for row in self.rows
        )
        if len(identities) != len(set(identities)):
            raise ValueError("raw logit audit contains duplicate vector identities")
        return self


class RawLogitAuditRow(StrictFrozenModel):
    """One output vector split at Inkling's unpadded vocabulary boundary."""

    task_id: int = Field(ge=0)
    slot_id: int = Field(ge=0)
    completion_index: int = Field(gt=0)
    batch_index: int = Field(ge=0)
    count: int = Field(gt=0)
    unpadded_count: int = Field(gt=0)
    padded_count: int = Field(gt=0)
    unpadded_finite: int = Field(ge=0)
    unpadded_nan: int = Field(ge=0)
    unpadded_pos_inf: int = Field(ge=0)
    unpadded_neg_inf: int = Field(ge=0)
    padded_finite: int = Field(ge=0)
    padded_nan: int = Field(ge=0)
    padded_pos_inf: int = Field(ge=0)
    padded_neg_inf: int = Field(ge=0)

    @model_validator(mode="after")
    def exact_regions(self) -> RawLogitAuditRow:
        if self.count != self.unpadded_count + self.padded_count:
            raise ValueError("raw logit region cardinality differs from the full vocabulary")
        if (
            self.unpadded_finite + self.unpadded_nan + self.unpadded_pos_inf + self.unpadded_neg_inf
            != self.unpadded_count
        ):
            raise ValueError("unpadded raw-logit counters have invalid cardinality")
        if (
            self.padded_finite + self.padded_nan + self.padded_pos_inf + self.padded_neg_inf
            != self.padded_count
        ):
            raise ValueError("padded raw-logit counters have invalid cardinality")
        if (
            self.unpadded_finite != self.unpadded_count
            or self.unpadded_nan != 0
            or self.unpadded_pos_inf != 0
            or self.unpadded_neg_inf != 0
        ):
            raise ValueError("unpadded raw-logit vector contains a non-finite value")
        if (
            self.padded_finite != 0
            or self.padded_nan != 0
            or self.padded_pos_inf != 0
            or self.padded_neg_inf != self.padded_count
        ):
            raise ValueError("padded raw-logit vector is not exact negative infinity")
        return self


class RawLogitAuditEvidence(StrictFrozenModel):
    """Proof of finite active logits and the exact padded suffix mask."""

    schema_version: Literal["inkling-raw-logit-audit-v2"] = "inkling-raw-logit-audit-v2"
    expected_generated_token_vectors: int = Field(gt=0)
    observed_generated_token_vectors: int = Field(gt=0)
    vocab_size: int = Field(gt=0)
    unpadded_vocab_size: int = Field(gt=0)
    padded_vocab_size: int = Field(gt=0)
    rows: tuple[RawLogitAuditRow, ...]
    all_rows_complete: Literal[True] = True
    all_unpadded_values_finite: Literal[True] = True
    all_padded_values_negative_infinity: Literal[True] = True

    @model_validator(mode="after")
    def exact_vector_set(self) -> RawLogitAuditEvidence:
        if not (
            self.expected_generated_token_vectors
            == self.observed_generated_token_vectors
            == len(self.rows)
        ):
            raise ValueError("raw logit vector count differs from generated-token evidence")
        if self.unpadded_vocab_size + self.padded_vocab_size != self.vocab_size:
            raise ValueError("raw logit vocabulary partition has invalid cardinality")
        for row in self.rows:
            if row.count != self.vocab_size:
                raise ValueError("raw logit vector does not cover the full vocabulary")
            if row.unpadded_count != self.unpadded_vocab_size:
                raise ValueError("raw logit unpadded vocabulary boundary differs from the contract")
            if row.padded_count != self.padded_vocab_size:
                raise ValueError("raw logit padded vocabulary size differs from the contract")
        identities = tuple(
            (row.task_id, row.slot_id, row.completion_index, row.batch_index) for row in self.rows
        )
        if len(identities) != len(set(identities)):
            raise ValueError("raw logit audit contains duplicate vector identities")
        return self


class BackendGraphAuditRow(StrictFrozenModel):
    """One scheduler graph after backend assignment and before graph splitting."""

    graph_uid: int = Field(gt=0)
    phase: Literal["post_assignment_pre_split"]
    scope: Literal["non_view_compute"]
    compute: int = Field(gt=0)
    gpu: int = Field(gt=0)
    cpu: Literal[0]
    accel: int = Field(ge=0)
    other: Literal[0]
    unassigned: Literal[0]

    @model_validator(mode="after")
    def exact_assignment(self) -> BackendGraphAuditRow:
        if self.compute != self.gpu + self.cpu + self.accel + self.other + self.unassigned:
            raise ValueError("backend graph category counts do not equal its compute count")
        if self.compute != self.gpu + self.accel:
            raise ValueError("backend graph contains a non-accelerated compute operation")
        return self


class BackendIdentityAuditRow(StrictFrozenModel):
    """One scheduler backend identity used by one audited graph."""

    graph_uid: int = Field(gt=0)
    backend_index: int = Field(ge=0)
    backend_name: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    device_type: Literal["cpu", "gpu", "igpu", "accel", "meta", "unassigned"]
    compute: int = Field(gt=0)

    @field_validator("backend_name", "device_name")
    @classmethod
    def marker_identifier_is_canonical(cls, value: str) -> str:
        if "\x00" in value or any(character.isspace() for character in value):
            raise ValueError("backend marker identifiers must be non-whitespace text")
        return value


class BackendAuditEvidence(StrictFrozenModel):
    """Aggregate operation placement from every instrumented inference graph."""

    schema_version: Literal["inkling-backend-audit-v1"] = "inkling-backend-audit-v1"
    observed_graphs: int = Field(gt=0)
    compute_operations: int = Field(gt=0)
    gpu_operations: int = Field(gt=0)
    accelerator_operations: int = Field(ge=0)
    cpu_operations: Literal[0]
    other_operations: Literal[0]
    unassigned_operations: Literal[0]
    graphs: tuple[BackendGraphAuditRow, ...]
    identities: tuple[BackendIdentityAuditRow, ...]
    all_compute_operations_accelerated: Literal[True] = True
    no_cpu_model_graph_fallback: Literal[True] = True

    @model_validator(mode="after")
    def exact_accelerator_placement(self) -> BackendAuditEvidence:
        if self.observed_graphs != len(self.graphs):
            raise ValueError("backend graph count differs from its graph records")
        graph_uids = tuple(row.graph_uid for row in self.graphs)
        if len(graph_uids) != len(set(graph_uids)):
            raise ValueError("backend audit contains duplicate graph identities")
        identity_keys = tuple((row.graph_uid, row.backend_index) for row in self.identities)
        if len(identity_keys) != len(set(identity_keys)):
            raise ValueError("backend audit contains duplicate backend identities")
        if set(row.graph_uid for row in self.identities) != set(graph_uids):
            raise ValueError("backend identities do not cover the exact graph set")
        if any(row.device_type != "gpu" for row in self.identities):
            raise ValueError("backend audit used a non-CUDA accelerator")
        cuda0_identity = (0, "CUDA0", "CUDA0")
        expected_cuda_identities = {
            cuda0_identity,
            (1, "CUDA1", "CUDA1"),
        }
        observed_dual_cuda_graph = False

        category_totals = {
            "gpu": 0,
            "cpu": 0,
            "accel": 0,
            "other": 0,
            "unassigned": 0,
        }
        for graph in self.graphs:
            graph_identities = tuple(
                row for row in self.identities if row.graph_uid == graph.graph_uid
            )
            observed_cuda_identities = {
                (row.backend_index, row.backend_name, row.device_name) for row in graph_identities
            }
            if (
                len(graph_identities) != len(observed_cuda_identities)
                or not observed_cuda_identities.issubset(expected_cuda_identities)
                or cuda0_identity not in observed_cuda_identities
            ):
                raise ValueError(
                    "backend graph does not prove the exact CUDA index and device identities"
                )
            observed_dual_cuda_graph |= observed_cuda_identities == expected_cuda_identities
            if sum(row.compute for row in graph_identities) != graph.compute:
                raise ValueError("backend identity counts do not equal graph compute count")
            observed = {name: 0 for name in category_totals}
            for identity in graph_identities:
                category = {
                    "gpu": "gpu",
                    "igpu": "gpu",
                    "accel": "accel",
                    "cpu": "cpu",
                    "meta": "other",
                    "unassigned": "unassigned",
                }[identity.device_type]
                observed[category] += identity.compute
                category_totals[category] += identity.compute
            expected = {
                "gpu": graph.gpu,
                "cpu": graph.cpu,
                "accel": graph.accel,
                "other": graph.other,
                "unassigned": graph.unassigned,
            }
            if observed != expected:
                raise ValueError("backend identities disagree with graph category counts")

        aggregate = {
            "gpu": self.gpu_operations,
            "cpu": self.cpu_operations,
            "accel": self.accelerator_operations,
            "other": self.other_operations,
            "unassigned": self.unassigned_operations,
        }
        if category_totals != aggregate:
            raise ValueError("backend aggregate counts differ from graph evidence")
        if sum(graph.compute for graph in self.graphs) != self.compute_operations:
            raise ValueError("backend aggregate compute count differs from graph evidence")
        if self.gpu_operations + self.accelerator_operations != self.compute_operations:
            raise ValueError("backend audit observed a non-accelerated compute operation")
        if not observed_dual_cuda_graph:
            raise ValueError("backend audit does not prove one exact dual-CUDA graph")
        return self


class TextShardLoadEvidence(StrictFrozenModel):
    """Exact split metadata and tensor inventory seen by the text loader."""

    expected: Literal[49]
    opened: Literal[49]
    contexts: Literal[49]
    tensors: int = Field(gt=0)


class TextTensorLoadEvidence(StrictFrozenModel):
    """Complete text tensor byte accounting after mmap setup."""

    opened: Literal[49]
    accounted: Literal[49]
    tensors: int = Field(gt=0)
    bytes: int = Field(gt=0)
    size_done: int = Field(gt=0)
    size_data: int = Field(gt=0)
    mmap: Literal[True]

    @model_validator(mode="after")
    def all_bytes_accounted(self) -> TextTensorLoadEvidence:
        if not self.bytes == self.size_done == self.size_data:
            raise ValueError("text tensor byte accounting is incomplete")
        return self


class ProjectorTensorLoadEvidence(StrictFrozenModel):
    """One modality-specific Inkling projector tensor load."""

    modality: Literal["vision", "audio"]
    projector: Literal["inkling"]
    tensors: int = Field(gt=0)
    bytes: int = Field(gt=0)


class ProjectorReadyEvidence(StrictFrozenModel):
    """Proof that both Inkling projector paths are ready."""

    opened: Literal[True]
    vision: Literal[True]
    audio: Literal[True]
    vision_type: Literal["inkling"]
    audio_type: Literal["inkling"]
    n_embd: int = Field(gt=0)


class ArtifactLoadEvidence(StrictFrozenModel):
    """Pinned loader evidence for all 49 model shards and the BF16 projector."""

    schema_version: Literal["inkling-artifact-load-v1"] = "inkling-artifact-load-v1"
    first_shard_path: Literal["/subject/q3_k_m/inkling-Q3_K_M-00001-of-00049.gguf"]
    additional_shards_loaded: Literal[48]
    total_shards_loaded: Literal[49]
    projector_path: Literal["/subject/mmproj/mmproj-BF16.gguf"]
    text_shards: TextShardLoadEvidence
    text_load: TextTensorLoadEvidence
    projector_tensors: tuple[ProjectorTensorLoadEvidence, ...]
    projector_ready: ProjectorReadyEvidence
    all_expected_artifacts_loaded: Literal[True] = True

    @model_validator(mode="after")
    def exact_loaded_inventory(self) -> ArtifactLoadEvidence:
        if self.text_shards.tensors != self.text_load.tensors:
            raise ValueError("text loader tensor inventories disagree")
        if tuple(row.modality for row in self.projector_tensors) != ("vision", "audio"):
            raise ValueError("projector tensor evidence must cover vision and audio once")
        return self


def parse_raw_logit_audit_evidence(
    log_text: str,
    *,
    expected_generated_token_vectors: int,
    vocab_size: int,
    unpadded_vocab_size: int | None = None,
) -> HistoricalRawLogitAuditEvidence | RawLogitAuditEvidence:
    """Parse one versioned pre-softmax audit and require exact coverage."""

    if expected_generated_token_vectors <= 0 or vocab_size <= 0:
        raise ValueError("raw logit expectations must be positive")
    if unpadded_vocab_size is None:
        if _RAW_LOGIT_AUDIT_MARKER in log_text:
            raise ValueError("raw logit version 2 requires an unpadded vocabulary boundary")
        matches = tuple(_HISTORICAL_RAW_LOGIT_AUDIT_RE.findall(log_text))
        if log_text.count(_HISTORICAL_RAW_LOGIT_AUDIT_MARKER) != len(matches):
            raise ValueError("raw logit audit contains a malformed marker")
        if any(any(int(value) != 0 for value in match[6:9]) for match in matches):
            raise ValueError("raw logit vector contains a non-finite value")
        historical_rows = tuple(
            HistoricalRawLogitAuditRow(
                task_id=int(task_id),
                slot_id=int(slot_id),
                completion_index=int(completion_index),
                batch_index=int(batch_index),
                count=int(count),
                finite=int(finite),
                nan=int(nan),
                pos_inf=int(pos_inf),
                neg_inf=int(neg_inf),
            )
            for (
                task_id,
                slot_id,
                completion_index,
                batch_index,
                count,
                finite,
                nan,
                pos_inf,
                neg_inf,
            ) in matches
        )
        return HistoricalRawLogitAuditEvidence(
            expected_generated_token_vectors=expected_generated_token_vectors,
            observed_generated_token_vectors=len(historical_rows),
            vocab_size=vocab_size,
            rows=historical_rows,
        )

    if not 0 < unpadded_vocab_size < vocab_size:
        raise ValueError("unpadded vocabulary boundary must be inside the full vocabulary")
    if _HISTORICAL_RAW_LOGIT_AUDIT_MARKER in log_text:
        raise ValueError("raw logit version 2 rejects historical aggregate markers")
    matches = tuple(_RAW_LOGIT_AUDIT_RE.findall(log_text))
    if log_text.count(_RAW_LOGIT_AUDIT_MARKER) != len(matches):
        raise ValueError("raw logit audit contains a malformed marker")
    if any(int(match[5]) != unpadded_vocab_size for match in matches):
        raise ValueError("raw logit unpadded vocabulary boundary differs from the contract")
    rows = tuple(
        RawLogitAuditRow(
            task_id=int(task_id),
            slot_id=int(slot_id),
            completion_index=int(completion_index),
            batch_index=int(batch_index),
            count=int(count),
            unpadded_count=int(unpadded_count),
            padded_count=int(padded_count),
            unpadded_finite=int(unpadded_finite),
            unpadded_nan=int(unpadded_nan),
            unpadded_pos_inf=int(unpadded_pos_inf),
            unpadded_neg_inf=int(unpadded_neg_inf),
            padded_finite=int(padded_finite),
            padded_nan=int(padded_nan),
            padded_pos_inf=int(padded_pos_inf),
            padded_neg_inf=int(padded_neg_inf),
        )
        for (
            task_id,
            slot_id,
            completion_index,
            batch_index,
            count,
            unpadded_count,
            padded_count,
            unpadded_finite,
            unpadded_nan,
            unpadded_pos_inf,
            unpadded_neg_inf,
            padded_finite,
            padded_nan,
            padded_pos_inf,
            padded_neg_inf,
        ) in matches
    )
    padded_vocab_size = vocab_size - unpadded_vocab_size
    return RawLogitAuditEvidence(
        expected_generated_token_vectors=expected_generated_token_vectors,
        observed_generated_token_vectors=len(rows),
        vocab_size=vocab_size,
        unpadded_vocab_size=unpadded_vocab_size,
        padded_vocab_size=padded_vocab_size,
        rows=rows,
    )


def parse_backend_audit_evidence(log_text: str) -> BackendAuditEvidence:
    """Parse graph placement markers and reject any CPU or unassigned operation."""

    graph_matches = tuple(_BACKEND_GRAPH_RE.findall(log_text))
    if log_text.count(_BACKEND_GRAPH_MARKER) != len(graph_matches):
        raise ValueError("backend graph audit contains a malformed marker")
    identity_matches = tuple(_BACKEND_IDENTITY_RE.findall(log_text))
    if log_text.count(_BACKEND_IDENTITY_MARKER) != len(identity_matches):
        raise ValueError("backend identity audit contains a malformed marker")
    cpu_matches = tuple(_CPU_NODE_RE.findall(log_text))
    if log_text.count(_CPU_NODE_MARKER) != len(cpu_matches):
        raise ValueError("backend CPU-node audit contains a malformed marker")
    if cpu_matches:
        raise ValueError("backend audit observed a CPU model graph operation")
    if not graph_matches or not identity_matches:
        raise ValueError("backend audit contains no graph markers")
    graphs = tuple(
        BackendGraphAuditRow(
            graph_uid=int(graph_uid),
            phase=phase,
            scope=scope,
            compute=int(compute),
            gpu=int(gpu),
            cpu=int(cpu),
            accel=int(accel),
            other=int(other),
            unassigned=int(unassigned),
        )
        for graph_uid, phase, scope, compute, gpu, cpu, accel, other, unassigned in graph_matches
    )
    identities = tuple(
        BackendIdentityAuditRow(
            graph_uid=int(graph_uid),
            backend_index=int(backend_index),
            backend_name=backend_name,
            device_name=device_name,
            device_type=device_type,
            compute=int(compute),
        )
        for graph_uid, backend_index, backend_name, device_name, device_type, compute in (
            identity_matches
        )
    )
    return BackendAuditEvidence(
        observed_graphs=len(graphs),
        compute_operations=sum(row.compute for row in graphs),
        gpu_operations=sum(row.gpu for row in graphs),
        accelerator_operations=sum(row.accel for row in graphs),
        cpu_operations=sum(row.cpu for row in graphs),
        other_operations=sum(row.other for row in graphs),
        unassigned_operations=sum(row.unassigned for row in graphs),
        graphs=graphs,
        identities=identities,
    )


def parse_artifact_load_evidence(log_text: str) -> ArtifactLoadEvidence:
    """Parse exact first-shard, split-count, and multimodal-projector loader lines."""

    first_shards = tuple(_FIRST_SHARD_LOAD_RE.findall(log_text))
    if len(first_shards) != 1 or first_shards[0] != EXPECTED_FIRST_Q3_MOUNT_PATH:
        raise ValueError("loader log does not bind the exact first shard")
    additional = tuple(_ADDITIONAL_SHARD_LOAD_RE.findall(log_text))
    if len(additional) != 1 or additional[0] != "48":
        raise ValueError("loader log does not prove the exact additional shard count")
    projectors = tuple(_PROJECTOR_LOAD_RE.findall(log_text))
    if len(projectors) != 1 or projectors[0] != EXPECTED_PROJECTOR_MOUNT_PATH:
        raise ValueError("loader log does not bind the exact projector")

    shard_matches = tuple(_TEXT_SHARDS_RE.findall(log_text))
    if log_text.count(_TEXT_SHARDS_MARKER) != len(shard_matches) or len(shard_matches) != 1:
        raise ValueError("loader log lacks one valid text-shard audit marker")
    expected, opened, contexts, shard_tensors = (int(value) for value in shard_matches[0])
    text_shards = TextShardLoadEvidence(
        expected=expected,
        opened=opened,
        contexts=contexts,
        tensors=shard_tensors,
    )

    load_matches = tuple(_TEXT_LOAD_RE.findall(log_text))
    if log_text.count(_TEXT_LOAD_MARKER) != len(load_matches) or len(load_matches) != 1:
        raise ValueError("loader log lacks one valid text-load audit marker")
    load_values = tuple(int(value) for value in load_matches[0])
    text_load = TextTensorLoadEvidence(
        opened=load_values[0],
        accounted=load_values[1],
        tensors=load_values[2],
        bytes=load_values[3],
        size_done=load_values[4],
        size_data=load_values[5],
        mmap=bool(load_values[6]),
    )

    projector_matches = tuple(_PROJECTOR_TENSORS_RE.findall(log_text))
    if (
        log_text.count(_PROJECTOR_TENSORS_MARKER) != len(projector_matches)
        or len(projector_matches) != 2
    ):
        raise ValueError("loader log lacks the exact two projector tensor markers")
    projector_order = {"vision": 0, "audio": 1}
    projector_tensors = tuple(
        sorted(
            (
                ProjectorTensorLoadEvidence(
                    modality=modality,
                    projector=projector,
                    tensors=int(tensors),
                    bytes=int(byte_count),
                )
                for modality, projector, tensors, byte_count in projector_matches
            ),
            key=lambda row: projector_order[row.modality],
        )
    )

    ready_matches = tuple(_PROJECTOR_READY_RE.findall(log_text))
    if log_text.count(_PROJECTOR_READY_MARKER) != len(ready_matches) or len(ready_matches) != 1:
        raise ValueError("loader log lacks one valid projector-ready marker")
    ready = ready_matches[0]
    projector_ready = ProjectorReadyEvidence(
        opened=bool(int(ready[0])),
        vision=bool(int(ready[1])),
        audio=bool(int(ready[2])),
        vision_type=ready[3],
        audio_type=ready[4],
        n_embd=int(ready[5]),
    )
    return ArtifactLoadEvidence(
        first_shard_path=EXPECTED_FIRST_Q3_MOUNT_PATH,
        additional_shards_loaded=48,
        total_shards_loaded=EXPECTED_Q3_SHARD_COUNT,
        projector_path=EXPECTED_PROJECTOR_MOUNT_PATH,
        text_shards=text_shards,
        text_load=text_load,
        projector_tensors=projector_tensors,
        projector_ready=projector_ready,
    )


class LoaderOffloadEvidence(StrictFrozenModel):
    """Parsed proof that every llama.cpp layer eligible for offload reached CUDA."""

    cuda_device_count: int = Field(gt=0)
    offloaded_layers: int = Field(gt=0)
    offloadable_layers: int = Field(gt=0)
    output_layer_offloaded: Literal[True] = True
    all_offloadable_layers_on_gpu: Literal[True] = True
    no_gpu_warning_observed: Literal[True] = True


def _one_consistent_match(values: Sequence[tuple[str, ...]], *, label: str) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"loader log lacks {label}")
    if len(set(values)) != 1:
        raise ValueError(f"loader log has conflicting {label}")
    return values[0]


def parse_loader_offload_evidence(
    log_text: str,
    *,
    expected_gpu_count: int = 2,
) -> LoaderOffloadEvidence:
    """Parse pinned llama.cpp loader lines and reject partial or CPU-only loading."""

    if expected_gpu_count <= 0:
        raise ValueError("expected_gpu_count must be positive")
    if not log_text or "\x00" in log_text:
        raise ValueError("loader log must be non-empty text without NUL")
    if _NO_GPU_WARNING in log_text:
        raise ValueError("llama.cpp reported that no usable GPU was found")
    count_matches = [(value,) for value in _CUDA_DEVICE_COUNT_RE.findall(log_text)]
    count_text = _one_consistent_match(count_matches, label="CUDA device-count evidence")[0]
    device_count = int(count_text)
    if device_count != expected_gpu_count:
        raise ValueError("CUDA device count differs from the configured hardware cell")
    offload_text = _one_consistent_match(
        _OFFLOAD_RE.findall(log_text), label="layer-offload evidence"
    )
    offloaded, offloadable = (int(value) for value in offload_text)
    if offloadable <= 0 or offloaded != offloadable:
        raise ValueError("llama.cpp did not offload every offloadable layer")
    if _OUTPUT_OFFLOAD_TEXT not in log_text:
        raise ValueError("loader log does not prove that the output layer was offloaded")
    return LoaderOffloadEvidence(
        cuda_device_count=device_count,
        offloaded_layers=offloaded,
        offloadable_layers=offloadable,
    )


_GPU_UUID_PATTERN = r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$"
_PCI_BUS_ID_PATTERN = r"^[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]$"
_CUDA_COMPUTE_CAPABILITY_MAJOR_ATTRIBUTE = 75
_CUDA_COMPUTE_CAPABILITY_MINOR_ATTRIBUTE = 76
_CUDA_P2P_PERFORMANCE_RANK_ATTRIBUTE = 0x01
_CUDA_P2P_ACCESS_SUPPORTED_ATTRIBUTE = 0x02
_CUDA_P2P_NATIVE_ATOMIC_SUPPORTED_ATTRIBUTE = 0x03
_CUDA_P2P_CUDA_ARRAY_ACCESS_SUPPORTED_ATTRIBUTE = 0x04
_CUDA_P2P_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED_ATTRIBUTE = 0x05


def _canonical_gpu_uuid(value: str, *, label: str) -> str:
    """Return ``GPU-`` plus a full, hyphenated, lowercase UUID.

    Only hexadecimal letter case is normalized.  The uppercase ``GPU-`` prefix
    and complete UUID layout are required so truncated or otherwise rewritten
    identifiers cannot join successfully.

    Receipt models intentionally continue to accept either hexadecimal case.
    Live evidence is normalized at its input boundaries instead, which preserves
    the bytes and hashes of already stored receipts.
    """

    if re.fullmatch(_GPU_UUID_PATTERN, value) is None:
        raise ValueError(f"{label} must be a full GPU UUID")
    return f"GPU-{value.removeprefix('GPU-').lower()}"


class CudaDriverGpuEvidence(StrictFrozenModel):
    """One process-visible GPU enumerated through the exact real CUDA driver."""

    cuda_ordinal: int = Field(ge=0)
    uuid: str = Field(pattern=_GPU_UUID_PATTERN)
    cuda_driver_name: str = Field(min_length=1)
    cuda_compute_capability: Literal["10.3"]
    cuda_total_memory_bytes: int = Field(gt=0)
    pci_bus_id: str | None = Field(default=None, pattern=_PCI_BUS_ID_PATTERN)
    pci_bus_id_status: Literal["available", "unavailable"]
    pci_bus_id_source: Literal["cuda_driver_api"]
    pci_bus_id_error_code: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def pci_evidence_is_consistent(self) -> CudaDriverGpuEvidence:
        if self.pci_bus_id_status == "available":
            if self.pci_bus_id is None or self.pci_bus_id_error_code is not None:
                raise ValueError("available CUDA PCI evidence is incomplete")
        elif self.pci_bus_id is not None or self.pci_bus_id_error_code is None:
            raise ValueError("unavailable CUDA PCI evidence lacks its error code")
        if "B300" not in self.cuda_driver_name.upper():
            raise ValueError("CUDA driver GPU name is not B300")
        return self


class CudaDriverPeerLinkEvidence(StrictFrozenModel):
    """One ordered CUDA Driver peer-capability query between two visible GPUs."""

    source_cuda_ordinal: StrictInt = Field(ge=0)
    destination_cuda_ordinal: StrictInt = Field(ge=0)
    source_uuid: str = Field(pattern=_GPU_UUID_PATTERN)
    destination_uuid: str = Field(pattern=_GPU_UUID_PATTERN)
    can_access_peer: StrictBool
    performance_rank: StrictInt = Field(ge=0)
    access_supported: StrictBool
    native_atomic_supported: StrictBool
    cuda_array_access_supported: StrictBool
    only_partial_native_atomic_supported: StrictBool

    @model_validator(mode="after")
    def ordered_peer_query_is_consistent(self) -> CudaDriverPeerLinkEvidence:
        if self.source_cuda_ordinal == self.destination_cuda_ordinal:
            raise ValueError("CUDA peer link must join distinct ordinals")
        if self.source_uuid.lower() == self.destination_uuid.lower():
            raise ValueError("CUDA peer link must join distinct GPU UUIDs")
        if self.can_access_peer != self.access_supported:
            raise ValueError("CUDA peer access queries disagree")
        if self.native_atomic_supported and self.only_partial_native_atomic_supported:
            raise ValueError("CUDA peer atomic support cannot be both full and partial")
        return self


class CudaDriverPeerTopologyEvidence(StrictFrozenModel):
    """Exact process-visible CUDA peer capability, not physical fabric topology."""

    topology_protocol: Literal["cuda-driver-p2p-attributes-v1"]
    cuda_driver_api_version: StrictInt = Field(gt=0)
    links: tuple[CudaDriverPeerLinkEvidence, CudaDriverPeerLinkEvidence]

    @model_validator(mode="after")
    def exact_bidirectional_pair_is_present(self) -> CudaDriverPeerTopologyEvidence:
        pairs = tuple(
            (link.source_cuda_ordinal, link.destination_cuda_ordinal) for link in self.links
        )
        if pairs != ((0, 1), (1, 0)):
            raise ValueError("CUDA peer topology must contain ordered pairs 0->1 and 1->0")
        forward, reverse = self.links
        if (
            forward.source_uuid.lower() != reverse.destination_uuid.lower()
            or forward.destination_uuid.lower() != reverse.source_uuid.lower()
        ):
            raise ValueError("CUDA peer topology UUID directions do not reverse exactly")
        return self


class NvidiaSmiGpuEvidence(StrictFrozenModel):
    """One UUID-keyed row from the exact nvidia-smi identity query."""

    uuid: str = Field(pattern=_GPU_UUID_PATTERN)
    nvidia_smi_name: str = Field(min_length=1)
    nvidia_smi_memory_total_mib: int = Field(gt=0)
    driver_version: str = Field(pattern=r"^[0-9]+(?:\.[0-9]+)+$")
    nvidia_smi_compute_capability: Literal["10.3"]

    @field_validator("nvidia_smi_name")
    @classmethod
    def gpu_is_b300(cls, value: str) -> str:
        if "B300" not in value.upper():
            raise ValueError("nvidia-smi GPU name is not B300")
        return value


class NvidiaGpuEvidence(StrictFrozenModel):
    """Joined CUDA-driver and nvidia-smi identity for one CUDA ordinal."""

    identity_protocol: Literal["cuda-driver-uuid+nvidia-smi-uuid-v1"]
    cuda_driver_api_version: int = Field(gt=0)
    cuda_ordinal: int = Field(ge=0)
    uuid: str = Field(pattern=_GPU_UUID_PATTERN)
    cuda_driver_name: str = Field(min_length=1)
    nvidia_smi_name: str = Field(min_length=1)
    cuda_compute_capability: Literal["10.3"]
    nvidia_smi_compute_capability: Literal["10.3"]
    cuda_total_memory_bytes: int = Field(gt=0)
    nvidia_smi_memory_total_mib: int = Field(gt=0)
    driver_version: str = Field(pattern=r"^[0-9]+(?:\.[0-9]+)+$")
    pci_bus_id: str | None = Field(default=None, pattern=_PCI_BUS_ID_PATTERN)
    pci_bus_id_status: Literal["available", "unavailable"]
    pci_bus_id_source: Literal["cuda_driver_api"]
    pci_bus_id_error_code: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def sources_are_consistent(self) -> NvidiaGpuEvidence:
        if "B300" not in self.cuda_driver_name.upper():
            raise ValueError("CUDA driver GPU name is not B300")
        if "B300" not in self.nvidia_smi_name.upper():
            raise ValueError("nvidia-smi GPU name is not B300")
        if self.pci_bus_id_status == "available":
            if self.pci_bus_id is None or self.pci_bus_id_error_code is not None:
                raise ValueError("available CUDA PCI evidence is incomplete")
        elif self.pci_bus_id is not None or self.pci_bus_id_error_code is None:
            raise ValueError("unavailable CUDA PCI evidence lacks its error code")
        return self


class NvidiaGpuResourceSample(StrictFrozenModel):
    """One UUID-keyed nvidia-smi resource sample."""

    uuid: str = Field(pattern=_GPU_UUID_PATTERN)
    memory_used_mib: int = Field(ge=0)
    utilization_percent: int = Field(ge=0, le=100)


class _CudaUuid(ctypes.Structure):
    _fields_ = [("value", ctypes.c_ubyte * 16)]


def _cuda_function(library: Any, name: str, argtypes: list[object]) -> Any:
    try:
        function = getattr(library, name)
    except AttributeError as error:
        raise RuntimeError(f"CUDA driver lacks required function {name}") from error
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function


def _require_cuda_success(function: Any, *arguments: object, label: str) -> None:
    result = int(function(*arguments))
    if result != 0:
        raise RuntimeError(f"CUDA driver {label} failed with error code {result}")


def _canonical_cuda_pci_bus_id(value: str) -> str:
    match = re.fullmatch(
        r"([0-9A-Fa-f]{4}|[0-9A-Fa-f]{8}):([0-9A-Fa-f]{2}):([0-9A-Fa-f]{2})\.([0-7])",
        value,
    )
    if match is None:
        raise RuntimeError("CUDA driver returned a malformed PCI bus ID")
    domain, bus, device, function = match.groups()
    return f"{int(domain, 16):08x}:{bus.lower()}:{device.lower()}.{function}"


def _load_cuda_driver(cuda_driver_library_path: str) -> Any:
    path = PurePosixPath(cuda_driver_library_path)
    if (
        not path.is_absolute()
        or path.as_posix() != cuda_driver_library_path
        or any(part in {"", ".", ".."} for part in path.parts)
        or "stubs" in path.parts
        or path in (_CUDA_DRIVER_STUB_ROOT, _CUDA_DRIVER_LINK_ROOT)
        or _CUDA_DRIVER_STUB_ROOT in path.parents
        or _CUDA_DRIVER_LINK_ROOT in path.parents
    ):
        raise ValueError("CUDA driver library path must be canonical and absolute")
    try:
        return ctypes.CDLL(cuda_driver_library_path)
    except OSError as error:
        raise RuntimeError("CUDA driver library could not be loaded") from error


def enumerate_cuda_driver_gpus(
    cuda_driver_library_path: str,
) -> tuple[int, tuple[CudaDriverGpuEvidence, CudaDriverGpuEvidence]]:
    """Enumerate the exact two process-visible GPUs through a real CUDA driver."""

    library = _load_cuda_driver(cuda_driver_library_path)
    cu_init = _cuda_function(library, "cuInit", [ctypes.c_uint])
    cu_driver_get_version = _cuda_function(
        library, "cuDriverGetVersion", [ctypes.POINTER(ctypes.c_int)]
    )
    cu_device_get_count = _cuda_function(
        library, "cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)]
    )
    cu_device_get = _cuda_function(
        library, "cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    )
    cu_device_get_uuid = _cuda_function(
        library, "cuDeviceGetUuid_v2", [ctypes.POINTER(_CudaUuid), ctypes.c_int]
    )
    cu_device_get_name = _cuda_function(
        library,
        "cuDeviceGetName",
        [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int],
    )
    cu_device_get_attribute = _cuda_function(
        library,
        "cuDeviceGetAttribute",
        [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
    )
    cu_device_total_mem = _cuda_function(
        library, "cuDeviceTotalMem_v2", [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    )
    cu_device_get_pci_bus_id = _cuda_function(
        library,
        "cuDeviceGetPCIBusId",
        [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int],
    )
    _require_cuda_success(cu_init, 0, label="initialization")
    driver_api_version = ctypes.c_int()
    _require_cuda_success(
        cu_driver_get_version,
        ctypes.byref(driver_api_version),
        label="API-version query",
    )
    if driver_api_version.value <= 0:
        raise RuntimeError("CUDA driver API version is invalid")
    count = ctypes.c_int()
    _require_cuda_success(cu_device_get_count, ctypes.byref(count), label="device-count query")
    if count.value != 2:
        raise RuntimeError("CUDA driver must expose exactly two GPUs")
    gpus: list[CudaDriverGpuEvidence] = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        _require_cuda_success(
            cu_device_get, ctypes.byref(device), ordinal, label=f"device {ordinal} lookup"
        )
        raw_uuid = _CudaUuid()
        _require_cuda_success(
            cu_device_get_uuid,
            ctypes.byref(raw_uuid),
            device,
            label=f"device {ordinal} UUID query",
        )
        name_buffer = ctypes.create_string_buffer(256)
        _require_cuda_success(
            cu_device_get_name,
            name_buffer,
            len(name_buffer),
            device,
            label=f"device {ordinal} name query",
        )
        try:
            name = name_buffer.value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise RuntimeError("CUDA driver returned a non-UTF-8 GPU name") from error
        major = ctypes.c_int()
        minor = ctypes.c_int()
        _require_cuda_success(
            cu_device_get_attribute,
            ctypes.byref(major),
            _CUDA_COMPUTE_CAPABILITY_MAJOR_ATTRIBUTE,
            device,
            label=f"device {ordinal} compute-major query",
        )
        _require_cuda_success(
            cu_device_get_attribute,
            ctypes.byref(minor),
            _CUDA_COMPUTE_CAPABILITY_MINOR_ATTRIBUTE,
            device,
            label=f"device {ordinal} compute-minor query",
        )
        total_memory = ctypes.c_size_t()
        _require_cuda_success(
            cu_device_total_mem,
            ctypes.byref(total_memory),
            device,
            label=f"device {ordinal} memory query",
        )
        pci_buffer = ctypes.create_string_buffer(32)
        pci_result = int(cu_device_get_pci_bus_id(pci_buffer, len(pci_buffer), device))
        pci_bus_id: str | None = None
        pci_status = "unavailable"
        pci_error_code: int | None = pci_result
        if pci_result == 0:
            try:
                pci_bus_id = _canonical_cuda_pci_bus_id(
                    pci_buffer.value.decode("ascii", errors="strict")
                )
            except UnicodeDecodeError as error:
                raise RuntimeError("CUDA driver returned a non-ASCII PCI bus ID") from error
            pci_status = "available"
            pci_error_code = None
        gpus.append(
            CudaDriverGpuEvidence(
                cuda_ordinal=ordinal,
                uuid=_canonical_gpu_uuid(
                    f"GPU-{UUID(bytes=bytes(raw_uuid.value))}",
                    label=f"CUDA device {ordinal} UUID",
                ),
                cuda_driver_name=name,
                cuda_compute_capability=f"{major.value}.{minor.value}",
                cuda_total_memory_bytes=total_memory.value,
                pci_bus_id=pci_bus_id,
                pci_bus_id_status=pci_status,
                pci_bus_id_source="cuda_driver_api",
                pci_bus_id_error_code=pci_error_code,
            )
        )
    if len({gpu.uuid for gpu in gpus}) != 2:
        raise RuntimeError("CUDA driver GPU UUIDs must be unique")
    return driver_api_version.value, (gpus[0], gpus[1])


def _cuda_binary_peer_value(value: int, *, label: str) -> bool:
    if value not in (0, 1):
        raise RuntimeError(f"CUDA driver {label} must be zero or one")
    return bool(value)


def enumerate_cuda_driver_peer_topology(
    cuda_driver_library_path: str,
) -> CudaDriverPeerTopologyEvidence:
    """Enumerate exact ordered P2P capabilities through the real CUDA Driver API.

    The result proves the peer-capability view exposed to this process.  It does
    not claim a physical NVLink, NVSwitch, or PCIe hop topology.
    """

    library = _load_cuda_driver(cuda_driver_library_path)
    cu_init = _cuda_function(library, "cuInit", [ctypes.c_uint])
    cu_driver_get_version = _cuda_function(
        library, "cuDriverGetVersion", [ctypes.POINTER(ctypes.c_int)]
    )
    cu_device_get_count = _cuda_function(
        library, "cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)]
    )
    cu_device_get = _cuda_function(
        library, "cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    )
    cu_device_get_uuid = _cuda_function(
        library, "cuDeviceGetUuid_v2", [ctypes.POINTER(_CudaUuid), ctypes.c_int]
    )
    cu_device_can_access_peer = _cuda_function(
        library,
        "cuDeviceCanAccessPeer",
        [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
    )
    cu_device_get_p2p_attribute = _cuda_function(
        library,
        "cuDeviceGetP2PAttribute",
        [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int],
    )
    _require_cuda_success(cu_init, 0, label="initialization")
    driver_api_version = ctypes.c_int()
    _require_cuda_success(
        cu_driver_get_version,
        ctypes.byref(driver_api_version),
        label="API-version query",
    )
    if driver_api_version.value <= 0:
        raise RuntimeError("CUDA driver API version is invalid")
    count = ctypes.c_int()
    _require_cuda_success(cu_device_get_count, ctypes.byref(count), label="device-count query")
    if count.value != 2:
        raise RuntimeError("CUDA driver must expose exactly two GPUs")

    devices: list[ctypes.c_int] = []
    uuids: list[str] = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        _require_cuda_success(
            cu_device_get, ctypes.byref(device), ordinal, label=f"device {ordinal} lookup"
        )
        raw_uuid = _CudaUuid()
        _require_cuda_success(
            cu_device_get_uuid,
            ctypes.byref(raw_uuid),
            device,
            label=f"device {ordinal} UUID query",
        )
        devices.append(device)
        uuids.append(
            _canonical_gpu_uuid(
                f"GPU-{UUID(bytes=bytes(raw_uuid.value))}",
                label=f"CUDA device {ordinal} UUID",
            )
        )
    if len(set(uuids)) != 2:
        raise RuntimeError("CUDA driver GPU UUIDs must be unique")

    attribute_queries = (
        ("performance-rank", _CUDA_P2P_PERFORMANCE_RANK_ATTRIBUTE),
        ("access-supported", _CUDA_P2P_ACCESS_SUPPORTED_ATTRIBUTE),
        ("native-atomic-supported", _CUDA_P2P_NATIVE_ATOMIC_SUPPORTED_ATTRIBUTE),
        (
            "CUDA-array-access-supported",
            _CUDA_P2P_CUDA_ARRAY_ACCESS_SUPPORTED_ATTRIBUTE,
        ),
        (
            "only-partial-native-atomic-supported",
            _CUDA_P2P_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED_ATTRIBUTE,
        ),
    )
    links: list[CudaDriverPeerLinkEvidence] = []
    for source_ordinal, destination_ordinal in ((0, 1), (1, 0)):
        can_access_peer_raw = ctypes.c_int()
        _require_cuda_success(
            cu_device_can_access_peer,
            ctypes.byref(can_access_peer_raw),
            devices[source_ordinal],
            devices[destination_ordinal],
            label=f"peer {source_ordinal}->{destination_ordinal} access query",
        )
        can_access_peer = _cuda_binary_peer_value(
            can_access_peer_raw.value,
            label=f"peer {source_ordinal}->{destination_ordinal} access result",
        )
        attributes: dict[int, int] = {}
        for attribute_label, attribute in attribute_queries:
            raw_value = ctypes.c_int()
            _require_cuda_success(
                cu_device_get_p2p_attribute,
                ctypes.byref(raw_value),
                attribute,
                devices[source_ordinal],
                devices[destination_ordinal],
                label=(f"peer {source_ordinal}->{destination_ordinal} {attribute_label} query"),
            )
            attributes[attribute] = raw_value.value
        performance_rank = attributes[_CUDA_P2P_PERFORMANCE_RANK_ATTRIBUTE]
        if performance_rank < 0:
            raise RuntimeError("CUDA driver peer performance rank must be non-negative")
        access_supported = _cuda_binary_peer_value(
            attributes[_CUDA_P2P_ACCESS_SUPPORTED_ATTRIBUTE],
            label=f"peer {source_ordinal}->{destination_ordinal} access-supported result",
        )
        if can_access_peer != access_supported:
            raise RuntimeError(
                f"CUDA driver peer {source_ordinal}->{destination_ordinal} access queries disagree"
            )
        native_atomic_supported = _cuda_binary_peer_value(
            attributes[_CUDA_P2P_NATIVE_ATOMIC_SUPPORTED_ATTRIBUTE],
            label=(f"peer {source_ordinal}->{destination_ordinal} native-atomic-supported result"),
        )
        cuda_array_access_supported = _cuda_binary_peer_value(
            attributes[_CUDA_P2P_CUDA_ARRAY_ACCESS_SUPPORTED_ATTRIBUTE],
            label=(
                f"peer {source_ordinal}->{destination_ordinal} CUDA-array-access-supported result"
            ),
        )
        only_partial_native_atomic_supported = _cuda_binary_peer_value(
            attributes[_CUDA_P2P_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED_ATTRIBUTE],
            label=(
                f"peer {source_ordinal}->{destination_ordinal} "
                "only-partial-native-atomic-supported result"
            ),
        )
        if native_atomic_supported and only_partial_native_atomic_supported:
            raise RuntimeError(
                f"CUDA driver peer {source_ordinal}->{destination_ordinal} "
                "atomic support cannot be both full and partial"
            )
        links.append(
            CudaDriverPeerLinkEvidence(
                source_cuda_ordinal=source_ordinal,
                destination_cuda_ordinal=destination_ordinal,
                source_uuid=uuids[source_ordinal],
                destination_uuid=uuids[destination_ordinal],
                can_access_peer=can_access_peer,
                performance_rank=performance_rank,
                access_supported=access_supported,
                native_atomic_supported=native_atomic_supported,
                cuda_array_access_supported=cuda_array_access_supported,
                only_partial_native_atomic_supported=(only_partial_native_atomic_supported),
            )
        )
    return CudaDriverPeerTopologyEvidence(
        topology_protocol="cuda-driver-p2p-attributes-v1",
        cuda_driver_api_version=driver_api_version.value,
        links=(links[0], links[1]),
    )


def parse_nvidia_smi_csv(payload: str) -> tuple[NvidiaSmiGpuEvidence, ...]:
    """Parse two UUID-keyed B300 rows without relying on nvidia-smi ordinals or PCI."""

    if not payload.strip() or "\x00" in payload:
        raise ValueError("nvidia-smi CSV must be non-empty text without NUL")
    rows = tuple(csv.reader(io.StringIO(payload), skipinitialspace=True))
    if len(rows) != 2:
        raise ValueError("nvidia-smi CSV must contain exactly two GPU rows")
    evidence: list[NvidiaSmiGpuEvidence] = []
    for row_index, row in enumerate(rows):
        if len(row) != 5 or any(not value.strip() for value in row):
            raise ValueError(f"nvidia-smi row {row_index} must contain five populated fields")
        memory_text = row[2].strip()
        memory_match = re.fullmatch(r"([0-9]+)(?: MiB)?", memory_text)
        if memory_match is None:
            raise ValueError(f"nvidia-smi row {row_index} memory must be MiB")
        evidence.append(
            NvidiaSmiGpuEvidence(
                uuid=_canonical_gpu_uuid(
                    row[0].strip(),
                    label=f"nvidia-smi row {row_index} UUID",
                ),
                nvidia_smi_name=row[1].strip(),
                nvidia_smi_memory_total_mib=int(memory_match.group(1)),
                driver_version=row[3].strip(),
                nvidia_smi_compute_capability=row[4].strip(),
            )
        )
    if len({gpu.uuid for gpu in evidence}) != 2:
        raise ValueError("nvidia-smi GPU UUIDs must be unique")
    if len({gpu.driver_version for gpu in evidence}) != 1:
        raise ValueError("nvidia-smi rows must report one driver version")
    return tuple(evidence)


def combine_gpu_identity(
    cuda_driver_api_version: int,
    cuda_gpus: Sequence[CudaDriverGpuEvidence],
    nvidia_smi_gpus: Sequence[NvidiaSmiGpuEvidence],
) -> tuple[NvidiaGpuEvidence, NvidiaGpuEvidence]:
    """Join independent CUDA and nvidia-smi evidence by full GPU UUID."""

    if tuple(gpu.cuda_ordinal for gpu in cuda_gpus) != (0, 1):
        raise ValueError("CUDA GPU ordinals must be exactly zero and one")
    canonical_cuda_uuids = tuple(
        _canonical_gpu_uuid(gpu.uuid, label=f"CUDA GPU {gpu.cuda_ordinal} UUID")
        for gpu in cuda_gpus
    )
    if len(set(canonical_cuda_uuids)) != 2:
        raise ValueError("CUDA GPU UUIDs must be unique")
    smi_by_uuid = {
        _canonical_gpu_uuid(gpu.uuid, label="nvidia-smi GPU UUID"): gpu for gpu in nvidia_smi_gpus
    }
    if len(smi_by_uuid) != 2 or set(smi_by_uuid) != set(canonical_cuda_uuids):
        raise ValueError("CUDA and nvidia-smi GPU UUID inventories differ")
    joined: list[NvidiaGpuEvidence] = []
    for cuda_gpu, canonical_uuid in zip(cuda_gpus, canonical_cuda_uuids, strict=True):
        smi_gpu = smi_by_uuid[canonical_uuid]
        joined.append(
            NvidiaGpuEvidence(
                identity_protocol="cuda-driver-uuid+nvidia-smi-uuid-v1",
                cuda_driver_api_version=cuda_driver_api_version,
                cuda_ordinal=cuda_gpu.cuda_ordinal,
                uuid=canonical_uuid,
                cuda_driver_name=cuda_gpu.cuda_driver_name,
                nvidia_smi_name=smi_gpu.nvidia_smi_name,
                cuda_compute_capability=cuda_gpu.cuda_compute_capability,
                nvidia_smi_compute_capability=smi_gpu.nvidia_smi_compute_capability,
                cuda_total_memory_bytes=cuda_gpu.cuda_total_memory_bytes,
                nvidia_smi_memory_total_mib=smi_gpu.nvidia_smi_memory_total_mib,
                driver_version=smi_gpu.driver_version,
                pci_bus_id=cuda_gpu.pci_bus_id,
                pci_bus_id_status=cuda_gpu.pci_bus_id_status,
                pci_bus_id_source=cuda_gpu.pci_bus_id_source,
                pci_bus_id_error_code=cuda_gpu.pci_bus_id_error_code,
            )
        )
    if len({gpu.driver_version for gpu in joined}) != 1:
        raise ValueError("joined GPU identities must report one driver version")
    return joined[0], joined[1]


def parse_nvidia_smi_monitor_csv(
    payload: str,
    *,
    expected_uuids: Sequence[str],
) -> tuple[NvidiaGpuResourceSample, NvidiaGpuResourceSample]:
    """Parse and reorder resource samples by the established CUDA UUID order."""

    canonical_expected_uuids = tuple(
        _canonical_gpu_uuid(value, label=f"expected monitor GPU {index} UUID")
        for index, value in enumerate(expected_uuids)
    )
    if len(canonical_expected_uuids) != 2 or len(set(canonical_expected_uuids)) != 2:
        raise ValueError("expected monitor UUID inventory must contain two unique GPUs")
    if not payload.strip() or "\x00" in payload:
        raise ValueError("resource monitor CSV must be non-empty text without NUL")
    rows = tuple(csv.reader(io.StringIO(payload), skipinitialspace=True))
    if len(rows) != 2:
        raise ValueError("resource monitor must observe exactly two GPU rows")
    samples: list[NvidiaGpuResourceSample] = []
    for row_index, row in enumerate(rows):
        if len(row) != 3 or any(not value.strip() for value in row):
            raise ValueError(f"resource monitor row {row_index} must contain three fields")
        samples.append(
            NvidiaGpuResourceSample(
                uuid=_canonical_gpu_uuid(
                    row[0].strip(),
                    label=f"resource monitor row {row_index} UUID",
                ),
                memory_used_mib=_exact_decimal(
                    row[1].strip(), label=f"resource monitor row {row_index} memory"
                ),
                utilization_percent=_exact_decimal(
                    row[2].strip(), label=f"resource monitor row {row_index} utilization"
                ),
            )
        )
    by_uuid = {sample.uuid: sample for sample in samples}
    if len(by_uuid) != 2 or set(by_uuid) != set(canonical_expected_uuids):
        raise ValueError("resource monitor GPU UUID inventory drifted")
    return by_uuid[canonical_expected_uuids[0]], by_uuid[canonical_expected_uuids[1]]


def parse_cuda_driver_linkage(payload: str) -> str:
    """Return the one real ``libcuda.so.1`` path resolved by ``ldd``."""

    matches = _CUDA_DRIVER_LINKAGE_RE.findall(payload)
    if len(matches) != 1:
        raise ConfigurationError(
            "Runtime linkage must resolve exactly one absolute libcuda.so.1 path"
        )
    value = str(matches[0])
    path = PurePosixPath(value)
    if (
        "\x00" in value
        or "\\" in value
        or "//" in value
        or not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ConfigurationError("Runtime CUDA driver path is not canonical")
    if (
        "stubs" in path.parts
        or path in (_CUDA_DRIVER_STUB_ROOT, _CUDA_DRIVER_LINK_ROOT)
        or _CUDA_DRIVER_STUB_ROOT in path.parents
        or _CUDA_DRIVER_LINK_ROOT in path.parents
    ):
        raise ConfigurationError("Runtime CUDA driver resolved to the build driver stub")
    return value


def _exact_decimal(value: str, *, label: str) -> int:
    if re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValueError(f"{label} must be canonical decimal text")
    return int(value)


__all__ = [
    "B300_CMAKE_ARCHITECTURE",
    "B300_COMPUTE_CAPABILITY",
    "CUDA_DRIVER_STUB_RPATH_LINK_DEFINITION",
    "EXPECTED_FIRST_Q3_MOUNT_PATH",
    "EXPECTED_PROJECTOR_BYTES",
    "EXPECTED_PROJECTOR_MOUNT_PATH",
    "EXPECTED_PROJECTOR_PATH",
    "EXPECTED_PROJECTOR_SHA256",
    "EXPECTED_Q3_INVENTORY_SHA256",
    "EXPECTED_Q3_SHARD_COUNT",
    "EXPECTED_Q3_TOTAL_BYTES",
    "EXPECTED_VERIFIED_EXPORT_REFERENCE_SHA256",
    "INSTRUMENTATION_PATCH_RELATIVE_PATH",
    "INSTRUMENTATION_PATCH_SHA256",
    "INSTRUMENTATION_SCHEMA_VERSION",
    "LEGACY_CURRENT_INSTRUMENTATION_PATCH_SHA256",
    "LLAMA_SERVER_AUDIT_LOG_VERBOSITY",
    "PINNED_CUDA_IMAGE",
    "PINNED_CUDA_IMAGE_DIGEST",
    "PINNED_CUDA_PLATFORM",
    "PINNED_PADDED_VOCAB_SIZE",
    "PINNED_UNPADDED_VOCAB_SIZE",
    "PINNED_VOCAB_SIZE",
    "SMOKE_CONFIG_RELATIVE_PATH",
    "SUBJECT_CONFIG_HASH",
    "SUBJECT_CONTROL_PLANE_SHA256",
    "SUBJECT_MMPROJ_CALL_ID",
    "SUBJECT_QUANTIZE_CALL_ID",
    "SUBJECT_RUN_ID",
    "SUBJECT_VERIFY_CALL_ID",
    "VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH",
    "ArtifactLoadEvidence",
    "BackendAuditEvidence",
    "BackendGraphAuditRow",
    "BackendIdentityAuditRow",
    "CudaDriverGpuEvidence",
    "CudaDriverPeerLinkEvidence",
    "CudaDriverPeerTopologyEvidence",
    "InklingSmokeConfig",
    "InklingVerifiedExportReference",
    "LoaderOffloadEvidence",
    "NvidiaGpuEvidence",
    "NvidiaGpuResourceSample",
    "NvidiaSmiGpuEvidence",
    "ProjectorReadyEvidence",
    "ProjectorTensorLoadEvidence",
    "RawLogitAuditEvidence",
    "RawLogitAuditRow",
    "ServerCompletionEvidence",
    "SmokeOutputVocabularyConfig",
    "TextShardLoadEvidence",
    "TextTensorLoadEvidence",
    "VerifiedExportArtifact",
    "combine_gpu_identity",
    "enumerate_cuda_driver_gpus",
    "enumerate_cuda_driver_peer_topology",
    "load_inkling_smoke_config",
    "load_verified_export_reference",
    "parse_artifact_load_evidence",
    "parse_backend_audit_evidence",
    "parse_cuda_driver_linkage",
    "parse_loader_offload_evidence",
    "parse_nvidia_smi_csv",
    "parse_nvidia_smi_monitor_csv",
    "parse_raw_logit_audit_evidence",
    "parse_server_completion",
    "redacted_smoke_config_record",
    "verified_export_reference_sha256",
]
