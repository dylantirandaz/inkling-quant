"""Contracts for the isolated BF16 prompt-interface diagnostic.

This module is control-plane code only.  It validates checked-in plans and
records produced by the approved eight-B300 Modal workload.  It does not load
Inkling, execute llama.cpp, start Modal, or provide a CPU substitute.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Hashable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal, Protocol, TypeAlias, TypeVar, cast

import yaml
from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.gguf.inkling import (
    InklingSourceAdoptionReference,
    load_inkling_source_adoption_reference,
)
from inkling_quant_lab.gguf.inkling_matched import (
    InklingBF16SubjectReference,
    MatchedRuntimeConfig,
    load_bf16_subject_reference,
)
from inkling_quant_lab.gguf.inkling_measurement import DiagnosticItem, load_diagnostic_items
from inkling_quant_lab.gguf.inkling_measurement_control import MeasurementRuntimeIdentity
from inkling_quant_lab.gguf.inkling_measurement_raw_evidence import (
    MeasurementDiagnosticTimings,
    MeasurementHardwareIdentity,
    MeasurementResourceSampleSummary,
)
from inkling_quant_lab.gguf.inkling_smoke import TextArtifactLoadEvidence
from inkling_quant_lab.security import sensitive_literal_path

DIAGNOSTIC_CONFIG_RELATIVE_PATH: Final = (
    "configs/experiments/inkling_bf16_interface_diagnostic_modal.yaml"
)
DIAGNOSTIC_STAGE: Final = "bf16_interface_diagnostic"
DIAGNOSTIC_FUNCTION_NAME: Final = "run_bf16_interface_diagnostic"
DIAGNOSTIC_ENVIRONMENT_NAME: Final = "inkling-quant"
DIAGNOSTIC_ATTEMPT_REGISTRY_NAME: Final = "inkling-measurement-attempt-registry-v1"
DIAGNOSTIC_EVIDENCE_VOLUME_NAME: Final = "inkling-measurement-evidence-v1"
DIAGNOSTIC_DEPLOY_CONFIRMATION_PREFIX: Final = "CONFIRM BF16 DIAGNOSTIC DEPLOY"
DIAGNOSTIC_LAUNCH_CONFIRMATION_PREFIX: Final = "CONFIRM BF16 DIAGNOSTIC LAUNCH"
DIAGNOSTIC_ITEM_ORDER: Final = ("text_01", "text_02", "text_03", "text_04")
DIAGNOSTIC_CELL_ORDER: Final = (
    "raw_original",
    "raw_64",
    "chat_original",
    "chat_64",
)
DIAGNOSTIC_REQUEST_COUNT: Final = 16
DIAGNOSTIC_EOS_TOKEN_ID: Final = 200_006
DIAGNOSTIC_COMPARISON_TOKEN_ID: Final = 199_999
DIAGNOSTIC_PLANNED_STAGES: Final = (
    "verify_references",
    "verify_cuda_preflight",
    "stage_and_rehash_bf16",
    "bf16_interface_diagnostic",
    "verify_gpu_placement",
    "release_bf16",
    "publish",
)
DIAGNOSTIC_SCOPE_WARNING: Final = (
    "This is a BF16 prompt-interface diagnostic, not a quality-retention or "
    "performance comparison. Read the machine-readable record before use and "
    "do not apply it to a different model, runtime, hardware, or protocol."
)

DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES: Final = 512 * 1024
DIAGNOSTIC_RAW_RECORD_MAX_BYTES: Final = 16 * 1024 * 1024
DIAGNOSTIC_DEPLOY_CHALLENGE_MAX_AGE_SECONDS: Final = 2 * 60 * 60
DIAGNOSTIC_LAUNCH_CHALLENGE_MAX_AGE_SECONDS: Final = 2 * 60 * 60

DIAGNOSTIC_PROTOCOL_HASH_DOMAIN: Final = b"inkling-bf16-interface-protocol-v1\0"
DIAGNOSTIC_WORKLOAD_HASH_DOMAIN: Final = b"inkling-bf16-interface-workload-v1\0"
DIAGNOSTIC_CONTROL_PLANE_HASH_DOMAIN: Final = b"inkling-bf16-interface-control-v1\0"
DIAGNOSTIC_DEPLOY_CHALLENGE_HASH_DOMAIN: Final = b"inkling-bf16-interface-deploy-v1\0"
DIAGNOSTIC_LAUNCH_CHALLENGE_HASH_DOMAIN: Final = b"inkling-bf16-interface-launch-v1\0"
DIAGNOSTIC_LAUNCH_INTENT_HASH_DOMAIN: Final = b"inkling-bf16-interface-intent-v1\0"
DIAGNOSTIC_ACCEPTANCE_HASH_DOMAIN: Final = b"inkling-bf16-interface-acceptance-v1\0"
DIAGNOSTIC_ATTEMPT_CLAIM_HASH_DOMAIN: Final = b"inkling-bf16-interface-claim-v1\0"
DIAGNOSTIC_RAW_HASH_DOMAIN: Final = b"inkling-bf16-interface-private-raw-v1\0"
DIAGNOSTIC_ROLLUP_HASH_DOMAIN: Final = b"inkling-bf16-interface-rollup-v1\0"
DIAGNOSTIC_SUCCESS_HASH_DOMAIN: Final = b"inkling-bf16-interface-success-v1\0"
DIAGNOSTIC_FAILURE_HASH_DOMAIN: Final = b"inkling-bf16-interface-failure-v1\0"

_RUN_ID_PATTERN: Final = r"^[a-z0-9][a-z0-9._-]{0,95}$"
_SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
_GIT_OBJECT_PATTERN: Final = r"^[0-9a-f]{40}$"
_CANONICAL_UTC_PATTERN: Final = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$"
_MODAL_FUNCTION_ID_PATTERN: Final = r"^fu-[A-Za-z0-9]+$"
_MODAL_CALL_ID_PATTERN: Final = r"^fc-[A-Za-z0-9]+$"
_MODAL_INPUT_ID_PATTERN: Final = r"^in-[A-Za-z0-9]+(?::[0-9]+-[0-9]+)?$"
_MODAL_TASK_ID_PATTERN: Final = r"^ta-[A-Za-z0-9]+$"
_MODAL_DICT_ID_PATTERN: Final = r"^di-[A-Za-z0-9]+$"
_MODAL_VOLUME_ID_PATTERN: Final = r"^vo-[A-Za-z0-9]+$"
_SAFE_ERROR_CODE_PATTERN: Final = r"^[a-z][a-z0-9_]{0,95}$"

DiagnosticCellName: TypeAlias = Literal["raw_original", "raw_64", "chat_original", "chat_64"]
DiagnosticItemId: TypeAlias = Literal["text_01", "text_02", "text_03", "text_04"]
DiagnosticStageName: TypeAlias = Literal[
    "verify_references",
    "verify_cuda_preflight",
    "stage_and_rehash_bf16",
    "bf16_interface_diagnostic",
    "verify_gpu_placement",
    "release_bf16",
    "publish",
]
DiagnosticOutcome: TypeAlias = Literal["success", "failure"]


class _StrictDiagnosticModel(StrictFrozenModel):
    """Immutable, extra-forbid, finite base for diagnostic records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


_DiagnosticModelT = TypeVar("_DiagnosticModelT", bound=_StrictDiagnosticModel)


def canonical_diagnostic_json_bytes(value: object) -> bytes:
    """Encode canonical JSON with exactly one trailing line feed."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _domain_hash(domain: bytes, value: object) -> str:
    return hashlib.sha256(domain + canonical_diagnostic_json_bytes(value)).hexdigest()


def diagnostic_runtime_identity_sha256(identity: MeasurementRuntimeIdentity) -> str:
    """Hash the complete validated runtime identity as canonical diagnostic JSON."""

    if not isinstance(identity, MeasurementRuntimeIdentity):
        raise TypeError("diagnostic runtime hash requires a validated runtime identity")
    return hashlib.sha256(
        canonical_diagnostic_json_bytes(identity.model_dump(mode="json"))
    ).hexdigest()


def build_diagnostic_server_command(model_path: str) -> tuple[str, ...]:
    """Build the one reviewed model-only CUDA diagnostic server command."""

    validated_model_path = validate_absolute_path(model_path)
    return (
        "/opt/llama.cpp/build/bin/llama-server",
        "--log-verbosity",
        "4",
        "-m",
        validated_model_path,
        "-c",
        "8192",
        "-b",
        "2048",
        "-ub",
        "512",
        "-ngl",
        "all",
        "-ncmoe",
        "0",
        "-sm",
        "layer",
        "-dev",
        "CUDA0,CUDA1,CUDA2,CUDA3,CUDA4,CUDA5,CUDA6,CUDA7",
        "-ts",
        "1,1,1,1,1,1,1,1",
        "-fa",
        "on",
        "-kvo",
        "--mmap",
        "--op-offload",
        "--no-warmup",
        "-fit",
        "off",
        "--no-host",
        "-np",
        "1",
        "-cb",
        "--threads",
        "16",
        "--threads-batch",
        "16",
        "--host",
        "127.0.0.1",
        "--port",
        "19183",
        "--metrics",
        "--slots",
        "--no-ui",
        "--jinja",
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate diagnostic JSON key: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite diagnostic JSON constant: {value}")


def strict_diagnostic_json_object(payload: bytes | str, *, maximum_bytes: int) -> dict[str, Any]:
    """Parse one bounded, unambiguous UTF-8 JSON object."""

    if isinstance(payload, bytes):
        encoded = payload
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("diagnostic JSON must be strict UTF-8") from error
    elif isinstance(payload, str):
        text = payload
        encoded = payload.encode("utf-8")
    else:
        raise TypeError("diagnostic JSON must be bytes or text")
    if not encoded or len(encoded) > maximum_bytes:
        raise ValueError("diagnostic JSON is empty or exceeds its size limit")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_finite_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("diagnostic JSON is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("diagnostic JSON root must be an object")
    return value


def _canonical_json_model(
    payload: bytes,
    model: type[_DiagnosticModelT],
    *,
    maximum_bytes: int = DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES,
) -> _DiagnosticModelT:
    value = strict_diagnostic_json_object(payload, maximum_bytes=maximum_bytes)
    if payload != canonical_diagnostic_json_bytes(value):
        raise ValueError("diagnostic JSON bytes are not canonical")
    try:
        return model.model_validate_json(payload, strict=True)
    except ValidationError as error:
        raise ValueError("diagnostic JSON schema is invalid") from error


def validate_repository_relative_path(value: str) -> str:
    """Require a canonical repository-relative POSIX path."""

    if (
        type(value) is not str
        or not value
        or "\\" in value
        or "\x00" in value
        or value.startswith("/")
    ):
        raise ValueError("path must be canonical and repository-relative")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("path must be canonical and repository-relative")
    return value


def validate_absolute_path(value: str) -> str:
    """Require a non-root canonical absolute POSIX path."""

    if (
        type(value) is not str
        or not value
        or "\\" in value
        or "\x00" in value
        or not value.startswith("/")
        or value.startswith("//")
    ):
        raise ValueError("path must be a canonical absolute POSIX path")
    path = PurePosixPath(value)
    if (
        value == "/"
        or not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("path must be a canonical absolute POSIX path")
    return value


def _validate_run_id(value: str) -> str:
    if type(value) is not str or re.fullmatch(_RUN_ID_PATTERN, value) is None:
        raise ValueError("diagnostic run ID is invalid")
    return value


def _validate_sha256(value: str, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(_SHA256_PATTERN, value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _canonical_utc(value: str, *, label: str) -> str:
    if type(value) is not str or re.fullmatch(_CANONICAL_UTC_PATTERN, value) is None:
        raise ValueError(f"{label} must use canonical UTC microsecond text")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as error:
        raise ValueError(f"{label} is not a real UTC time") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%S.%fZ") != value:
        raise ValueError(f"{label} must use canonical UTC microsecond text")
    return value


def _utc_datetime(value: str, *, label: str) -> datetime:
    _canonical_utc(value, label=label)
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


class _DuplicateKeyRejectingSafeLoader(yaml.SafeLoader):
    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
        if not isinstance(node, MappingNode):
            raise ConstructorError(None, None, "expected a mapping node", node.start_mark)
        self.flatten_mapping(node)
        result: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found unhashable key",
                    key_node.start_mark,
                )
            if key in result:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            result[key] = self.construct_object(value_node, deep=deep)
        return result


class DiagnosticFileIdentity(_StrictDiagnosticModel):
    """One checked-in file bound by path, byte hash, and byte count."""

    path: StrictStr
    sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    size_bytes: StrictInt = Field(gt=0)

    @field_validator("path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        return validate_repository_relative_path(value)


class DiagnosticRecordIdentity(DiagnosticFileIdentity):
    reference_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)


class DiagnosticRemoteFileIdentity(_StrictDiagnosticModel):
    """One immutable file expected on a read-only Modal volume mount."""

    path: StrictStr
    sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    size_bytes: StrictInt = Field(gt=0)

    @field_validator("path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        return validate_absolute_path(value)


class DiagnosticSourceAssets(_StrictDiagnosticModel):
    """Exact source configuration and tokenizer/template assets used by the protocol."""

    config: DiagnosticRemoteFileIdentity
    chat_template: DiagnosticRemoteFileIdentity
    tokenizer_json: DiagnosticRemoteFileIdentity
    tokenizer_config: DiagnosticRemoteFileIdentity
    eos_token_id: Literal[200006]
    comparison_token_id: Literal[199999]
    eos_special_token: Literal["<|content_model_end_sampling|>"]
    comparison_special_token: Literal["<|endoftext|>"]

    @model_validator(mode="after")
    def exact_assets(self) -> DiagnosticSourceAssets:
        expected = {
            "config": (
                "/source/snapshot/config.json",
                "58720f145bcecef9a7ab2b419ab346e7c634af8d2f3e7362e900d00f789ea46c",
                2_415,
            ),
            "chat_template": (
                "/source/snapshot/chat_template.jinja",
                "0aa1aa0c729d90176dcaa00c440c8faffca2957ffb2cc4b79456ee6d02bcf43b",
                6_294,
            ),
            "tokenizer_json": (
                "/source/snapshot/tokenizer.json",
                "9fb6333a7db8fe5da90728e741e4a3ee4ac2ae12c5dd4958cc6f31688787d3c2",
                27_875_797,
            ),
            "tokenizer_config": (
                "/source/snapshot/tokenizer_config.json",
                "2e36c9748a2081abb935b2e745ee22e82efa32589c2500df7e5bc0f93145cd77",
                12_111,
            ),
        }
        for field_name, identity in expected.items():
            observed = getattr(self, field_name)
            if (observed.path, observed.sha256, observed.size_bytes) != identity:
                raise ValueError(f"source asset {field_name} differs from the pinned identity")
        return self


class DiagnosticCellConfig(_StrictDiagnosticModel):
    name: DiagnosticCellName
    prompt_mode: Literal["raw", "chat_template"]
    cap_mode: Literal["original", "fixed_64"]
    max_new_tokens_override: Literal[64] | None
    reasoning_effort: Literal["none"]

    @model_validator(mode="after")
    def exact_cell(self) -> DiagnosticCellConfig:
        expected = {
            "raw_original": ("raw", "original", None),
            "raw_64": ("raw", "fixed_64", 64),
            "chat_original": ("chat_template", "original", None),
            "chat_64": ("chat_template", "fixed_64", 64),
        }[self.name]
        if (self.prompt_mode, self.cap_mode, self.max_new_tokens_override) != expected:
            raise ValueError("diagnostic cell differs from its checked protocol")
        return self


class DiagnosticProtocolConfig(_StrictDiagnosticModel):
    item_ids: tuple[DiagnosticItemId, ...]
    cells: tuple[DiagnosticCellConfig, ...]
    request_order: Literal["cell_major_then_item_order"]
    request_count: Literal[16]
    requests_sequential: Literal[True]
    one_server_load: Literal[True]
    server_endpoint: Literal["/completion"]
    apply_template_endpoint: Literal["/apply-template"]
    tokenize_endpoint: Literal["/tokenize"]
    properties_endpoint: Literal["/props"]
    raw_instruction: Literal[
        "Answer the task directly and emit only the response form requested by the task."
    ]
    raw_prompt_protocol: Literal["instruction_then_lf_then_item_prompt"]
    chat_template_protocol: Literal["system_reasoning_effort_none_then_user_then_generation_prompt"]
    tokenize_add_special: Literal[False]
    tokenize_parse_special: Literal[True]
    temperature: StrictFloat
    seed: Literal[42]
    cache_prompt: Literal[False]
    stream: Literal[False]
    return_tokens: Literal[True]
    ignore_eos: Literal[False]
    stop: tuple[()]
    timings_per_token: Literal[True]
    score_whole_output: Literal[True]
    score_extracted_content_text: Literal[True]
    final_extraction_protocol: Literal[
        "last_content_text_segment_before_content_model_end_sampling_or_end"
    ]
    source_config_eos_must_equal: Literal[200006]
    runtime_eog_classification_required: Literal[True]
    forced_eog_probe_required: Literal[True]
    forced_comparison_token_probe_required: Literal[True]

    @field_validator("item_ids", "cells", "stop", mode="before")
    @classmethod
    def yaml_sequences_are_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("temperature")
    @classmethod
    def greedy_temperature(cls, value: float) -> float:
        if value != -1.0:
            raise ValueError("diagnostic temperature must be exactly -1.0 for greedy decoding")
        return value

    @model_validator(mode="after")
    def exact_matrix(self) -> DiagnosticProtocolConfig:
        if self.item_ids != DIAGNOSTIC_ITEM_ORDER:
            raise ValueError("diagnostic item order differs from the approved four items")
        if tuple(cell.name for cell in self.cells) != DIAGNOSTIC_CELL_ORDER:
            raise ValueError("diagnostic cell order differs from the approved matrix")
        return self


class DiagnosticPlacementConfig(_StrictDiagnosticModel):
    backend: Literal["CUDA"]
    logical_devices: Literal["cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"]
    gpu_layers: Literal["all"]
    cpu_moe_layers: Literal[0]
    split_mode: Literal["layer"]
    tensor_split: tuple[
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
        Literal[1],
    ]
    flash_attention: Literal["on"]
    mmap: Literal[True]
    fit: Literal["off"]
    cpu_fallback: Literal[False]

    @field_validator("tensor_split", mode="before")
    @classmethod
    def yaml_tensor_split_is_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class DiagnosticResources(_StrictDiagnosticModel):
    provider: Literal["modal"]
    gpu_type: Literal["B300"]
    gpu_count: Literal[8]
    compute_capability: Literal["10.3"]
    cpu_cores: Literal[16]
    memory_gib: Literal[64]
    ephemeral_disk_mib: Literal[2097152]
    startup_timeout_seconds: Literal[1800]
    function_timeout_seconds: Literal[86400]
    max_containers: Literal[1]
    max_attempts: Literal[1]
    network_access: Literal[False]
    cpu_fallback_allowed: Literal[False]


class DiagnosticStorage(_StrictDiagnosticModel):
    bf16_volume: Literal["inkling-work-v1"]
    bf16_volume_version: Literal[1]
    bf16_run_subpath: Literal["runs/inkling-q3km-86b4d430-a015409e-ffd466dd93-8083cf41e1"]
    bf16_mount_path: Literal["/baseline"]
    bf16_read_only: Literal[True]
    bf16_create_if_missing: Literal[False]
    source_volume: Literal["inkling-source-v1"]
    source_volume_version: Literal[1]
    source_run_subpath: Literal["runs/inkling-q3km-86b4d430-a015409e-551ab8f240-bcc168525e"]
    source_snapshot_subpath: Literal["snapshot"]
    source_mount_path: Literal["/source"]
    source_read_only: Literal[True]
    source_create_if_missing: Literal[False]
    evidence_volume: Literal["inkling-measurement-evidence-v1"]
    evidence_volume_version: Literal[1]
    evidence_mount_path: Literal["/evidence"]
    evidence_read_only: Literal[False]
    evidence_create_if_missing: Literal[True]
    evidence_append_only_after_terminal: Literal[True]
    attempt_registry: Literal["inkling-measurement-attempt-registry-v1"]
    attempt_registry_append_only: Literal[True]


class DiagnosticExecution(_StrictDiagnosticModel):
    remote_execution_policy: Literal["fresh_content_addressed_confirmation_required"]
    remote_execution_default_enabled: Literal[False]
    confirmation_reuse_allowed: Literal[False]
    network_access: Literal[False]
    subject: Literal["bf16"]
    subject_staging: Literal["verified_ephemeral_copy"]
    subject_staging_root: Literal["/cache/inkling-bf16-interface-diagnostic"]
    subject_staging_headroom_mib: Literal[131072]
    rehash_during_verified_staging_copy: Literal[True]
    release_staged_subject: Literal[True]
    partial_success_allowed: Literal[False]
    planned_stages: tuple[DiagnosticStageName, ...]

    @field_validator("planned_stages", mode="before")
    @classmethod
    def yaml_stages_are_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def exact_stages(self) -> DiagnosticExecution:
        if self.planned_stages != DIAGNOSTIC_PLANNED_STAGES:
            raise ValueError("diagnostic stages differ from the checked order")
        return self


class DiagnosticEvidencePolicy(_StrictDiagnosticModel):
    record_prompt_text: Literal[False]
    record_output_text: Literal[False]
    private_raw_record_token_ids: Literal[True]
    compact_receipt_token_ids: Literal[False]
    record_item_scores: Literal[True]
    record_stop_and_cap_fields: Literal[True]
    record_artifact_hashes: Literal[True]
    record_runtime_identity_hash: Literal[True]
    record_hardware_identity_hash: Literal[True]
    record_command_hash: Literal[True]
    record_server_timings: Literal[True]
    record_exact_command: Literal[True]
    record_resource_sample_summary: Literal[True]
    record_server_log_hash_and_size_only: Literal[True]
    immutable_after_terminal: Literal[True]


class DiagnosticClaims(_StrictDiagnosticModel):
    diagnostic_only: Literal[True]
    compatibility_scope: Literal["single_exact_matrix_cell"]
    quality_retention_claim_allowed: Literal[False]
    quality_claim_allowed: Literal[False]
    speedup_claim_allowed: Literal[False]
    performance_claim_allowed: Literal[False]
    mtp_included: Literal[False]
    mtp_supported: Literal[False]
    routing_drift_supported: Literal[False]
    single_run_causation_claim_allowed: Literal[False]
    scope_warning: Literal[
        "This is a BF16 prompt-interface diagnostic, not a quality-retention or "
        "performance comparison. Read the machine-readable record before use and "
        "do not apply it to a different model, runtime, hardware, or protocol."
    ]


class InklingBF16InterfaceDiagnosticConfig(_StrictDiagnosticModel):
    """Complete checked plan for the four-by-four BF16 interface diagnostic."""

    schema_version: Literal["inkling-bf16-interface-diagnostic-config-v1"]
    model_id: Literal["thinkingmachines/Inkling"]
    revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"]
    architecture: Literal["InklingForConditionalGeneration"]
    bf16_subject_reference: DiagnosticRecordIdentity
    source_adoption_reference: DiagnosticRecordIdentity
    diagnostic_dataset: DiagnosticFileIdentity
    source_assets: DiagnosticSourceAssets
    runtime: MatchedRuntimeConfig
    runtime_measurement_patch: DiagnosticFileIdentity
    runtime_measurement_patch_apply_after: Literal["runtime.instrumentation_patch"]
    runtime_binary_identity_policy: Literal["rebuild_after_patches_then_hash_and_record"]
    protocol: DiagnosticProtocolConfig
    placement: DiagnosticPlacementConfig
    resources: DiagnosticResources
    storage: DiagnosticStorage
    execution: DiagnosticExecution
    evidence: DiagnosticEvidencePolicy
    claims: DiagnosticClaims

    @model_validator(mode="after")
    def safe_exact_plan(self) -> InklingBF16InterfaceDiagnosticConfig:
        secret = sensitive_literal_path(self.model_dump(mode="json"))
        if secret is not None:
            raise ValueError(
                "diagnostic config contains literal credential material at " + ".".join(secret)
            )
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_diagnostic_json_bytes(self.model_dump(mode="json"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def diagnostic_protocol_sha256(config: InklingBF16InterfaceDiagnosticConfig) -> str:
    """Hash the exact runtime, interface method, placement, and reporting rules."""

    if not isinstance(config, InklingBF16InterfaceDiagnosticConfig):
        raise TypeError("diagnostic protocol hash requires a validated config")
    payload = {
        "schema_version": "inkling-bf16-interface-protocol-v1",
        "runtime": config.runtime.model_dump(mode="json"),
        "runtime_measurement_patch": config.runtime_measurement_patch.model_dump(mode="json"),
        "runtime_measurement_patch_apply_after": config.runtime_measurement_patch_apply_after,
        "runtime_binary_identity_policy": config.runtime_binary_identity_policy,
        "source_assets": config.source_assets.model_dump(mode="json"),
        "protocol": config.protocol.model_dump(mode="json"),
        "placement": config.placement.model_dump(mode="json"),
        "resources": config.resources.model_dump(mode="json"),
        "execution": config.execution.model_dump(mode="json"),
        "evidence": config.evidence.model_dump(mode="json"),
        "claims": config.claims.model_dump(mode="json"),
    }
    return _domain_hash(DIAGNOSTIC_PROTOCOL_HASH_DOMAIN, payload)


def diagnostic_workload_sha256(config: InklingBF16InterfaceDiagnosticConfig) -> str:
    """Hash the exact BF16 subject and four-item workload identity."""

    if not isinstance(config, InklingBF16InterfaceDiagnosticConfig):
        raise TypeError("diagnostic workload hash requires a validated config")
    payload = {
        "schema_version": "inkling-bf16-interface-workload-v1",
        "model_id": config.model_id,
        "model_revision": config.revision,
        "architecture": config.architecture,
        "bf16_subject_reference": config.bf16_subject_reference.model_dump(mode="json"),
        "source_adoption_reference": config.source_adoption_reference.model_dump(mode="json"),
        "diagnostic_dataset": config.diagnostic_dataset.model_dump(mode="json"),
        "item_ids": config.protocol.item_ids,
        "cells": [cell.model_dump(mode="json") for cell in config.protocol.cells],
    }
    return _domain_hash(DIAGNOSTIC_WORKLOAD_HASH_DOMAIN, payload)


class InklingBF16InterfaceDiagnosticBundle(_StrictDiagnosticModel):
    """Validated plan plus exact checked-in references and selected items."""

    config: InklingBF16InterfaceDiagnosticConfig
    bf16: InklingBF16SubjectReference
    source: InklingSourceAdoptionReference
    items: tuple[DiagnosticItem, DiagnosticItem, DiagnosticItem, DiagnosticItem]


def parse_bf16_interface_diagnostic_config_bytes(
    payload: bytes,
    *,
    source: str | Path = "<bytes>",
) -> InklingBF16InterfaceDiagnosticConfig:
    """Parse one duplicate-free YAML plan without external work."""

    try:
        raw = yaml.load(payload.decode("utf-8"), Loader=_DuplicateKeyRejectingSafeLoader)
        if not isinstance(raw, Mapping):
            raise ValueError("diagnostic config root must be a mapping")
        return InklingBF16InterfaceDiagnosticConfig.model_validate(raw)
    except (UnicodeDecodeError, ValueError, yaml.YAMLError, ValidationError) as error:
        raise ConfigurationError(
            f"Unable to parse BF16 interface diagnostic config {source}: {error}",
            component="inkling_bf16_interface_diagnostic_config",
        ) from error


def load_bf16_interface_diagnostic_config(
    path: str | Path,
) -> InklingBF16InterfaceDiagnosticConfig:
    config_path = Path(path)
    try:
        payload = config_path.read_bytes()
    except OSError as error:
        raise ConfigurationError(
            f"Unable to load BF16 interface diagnostic config {config_path}: {error}",
            component="inkling_bf16_interface_diagnostic_config",
        ) from error
    return parse_bf16_interface_diagnostic_config_bytes(payload, source=config_path)


def _project_file(root: Path, relative_path: str) -> Path:
    relative = validate_repository_relative_path(relative_path)
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ConfigurationError(
            "Diagnostic project reference resolves outside the project root",
            component="inkling_bf16_interface_diagnostic_bundle",
        )
    return candidate


def _verify_file(path: Path, identity: DiagnosticFileIdentity) -> None:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ConfigurationError(
            f"Unable to read diagnostic input {path}: {error}",
            component="inkling_bf16_interface_diagnostic_bundle",
        ) from error
    if (
        len(payload) != identity.size_bytes
        or hashlib.sha256(payload).hexdigest() != identity.sha256
    ):
        raise ConfigurationError(
            "Diagnostic input byte identity differs from the checked config",
            component="inkling_bf16_interface_diagnostic_bundle",
            details={"path": identity.path},
        )


def _translated_source_mount_path(
    *,
    origin_path: str,
    origin_run_root: str,
    mounted_run_root: str,
) -> str:
    """Translate a source-reference path through the configured Volume subpath mount."""

    origin = PurePosixPath(validate_absolute_path(origin_path))
    run_root = PurePosixPath(validate_absolute_path(origin_run_root))
    mount_root = PurePosixPath(validate_absolute_path(mounted_run_root))
    try:
        suffix = origin.relative_to(run_root)
    except ValueError as error:
        raise ValueError("source artifact is outside its adopted run root") from error
    return validate_absolute_path((mount_root / suffix).as_posix())


def load_bf16_interface_diagnostic_bundle(
    project_root: str | Path,
    *,
    config_relative_path: str = DIAGNOSTIC_CONFIG_RELATIVE_PATH,
) -> InklingBF16InterfaceDiagnosticBundle:
    """Load and cross-check the BF16-only diagnostic plan without model work."""

    root = Path(project_root).expanduser().resolve()
    if not root.is_dir():
        raise ConfigurationError(
            f"Diagnostic project root is not a directory: {root}",
            component="inkling_bf16_interface_diagnostic_bundle",
        )
    config_path = _project_file(root, config_relative_path)
    config = load_bf16_interface_diagnostic_config(config_path)
    for identity in (
        config.bf16_subject_reference,
        config.source_adoption_reference,
        config.diagnostic_dataset,
        config.runtime_measurement_patch,
    ):
        _verify_file(_project_file(root, identity.path), identity)

    bf16 = load_bf16_subject_reference(_project_file(root, config.bf16_subject_reference.path))
    source = load_inkling_source_adoption_reference(
        _project_file(root, config.source_adoption_reference.path)
    )
    all_items = load_diagnostic_items(_project_file(root, config.diagnostic_dataset.path))
    selected = tuple(item for item in all_items if item.item_id in DIAGNOSTIC_ITEM_ORDER)
    if tuple(item.item_id for item in selected) != DIAGNOSTIC_ITEM_ORDER:
        raise ConfigurationError(
            "Diagnostic dataset does not contain the exact four approved items in order",
            component="inkling_bf16_interface_diagnostic_bundle",
        )

    mismatches: list[str] = []
    if config.bf16_subject_reference.reference_sha256 != bf16.reference_sha256:
        mismatches.append("bf16_reference_sha256")
    if config.source_adoption_reference.reference_sha256 != source.reference_sha256:
        mismatches.append("source_reference_sha256")
    mounted_source_config = _translated_source_mount_path(
        origin_path=source.snapshot_config.path,
        origin_run_root=source.source_run_root,
        mounted_run_root=config.storage.source_mount_path,
    )
    if config.source_assets.config.path != mounted_source_config:
        mismatches.append("source_config_path")
    if config.source_assets.config.sha256 != source.snapshot_config.sha256:
        mismatches.append("source_config_sha256")
    if config.source_assets.config.size_bytes != source.snapshot_config.size_bytes:
        mismatches.append("source_config_size")
    if bf16.source_adoption_reference_sha256 != source.reference_sha256:
        mismatches.append("bf16_source_reference")
    if bf16.model_id != config.model_id or source.model_id != config.model_id:
        mismatches.append("model_id")
    if bf16.revision != config.revision or source.revision != config.revision:
        mismatches.append("model_revision")
    if mismatches:
        raise ConfigurationError(
            "BF16 interface diagnostic inputs are incompatible",
            component="inkling_bf16_interface_diagnostic_bundle",
            details={"mismatches": mismatches},
        )
    return InklingBF16InterfaceDiagnosticBundle(
        config=config,
        bf16=bf16,
        source=source,
        items=cast(
            tuple[DiagnosticItem, DiagnosticItem, DiagnosticItem, DiagnosticItem],
            selected,
        ),
    )


class DiagnosticControlPlaneFile(_StrictDiagnosticModel):
    """One exact file in the reviewed diagnostic implementation closure."""

    path: StrictStr
    size_bytes: StrictInt = Field(ge=0)
    sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        return validate_repository_relative_path(value)


def diagnostic_control_plane_sha256(
    *,
    reviewed_commit_sha: str,
    reviewed_tree_sha: str,
    files: Sequence[DiagnosticControlPlaneFile],
) -> str:
    """Hash one sorted, exact reviewed Git/file closure."""

    if re.fullmatch(_GIT_OBJECT_PATTERN, reviewed_commit_sha) is None:
        raise ValueError("reviewed diagnostic commit SHA is invalid")
    if re.fullmatch(_GIT_OBJECT_PATTERN, reviewed_tree_sha) is None:
        raise ValueError("reviewed diagnostic tree SHA is invalid")
    paths = tuple(item.path for item in files)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise ValueError("diagnostic control-plane paths must be sorted and unique")
    return _domain_hash(
        DIAGNOSTIC_CONTROL_PLANE_HASH_DOMAIN,
        {
            "schema_version": "inkling-bf16-interface-control-plane-v1",
            "reviewed_commit_sha": reviewed_commit_sha,
            "reviewed_tree_sha": reviewed_tree_sha,
            "files": [item.model_dump(mode="json") for item in files],
        },
    )


class DiagnosticControlPlaneProvenance(_StrictDiagnosticModel):
    schema_version: Literal["inkling-bf16-interface-control-plane-v1"] = (
        "inkling-bf16-interface-control-plane-v1"
    )
    reviewed_commit_sha: StrictStr = Field(pattern=_GIT_OBJECT_PATTERN)
    reviewed_tree_sha: StrictStr = Field(pattern=_GIT_OBJECT_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    file_count: StrictInt = Field(gt=0)
    files: tuple[DiagnosticControlPlaneFile, ...]

    @model_validator(mode="after")
    def exact_manifest(self) -> DiagnosticControlPlaneProvenance:
        if self.file_count != len(self.files):
            raise ValueError("diagnostic control-plane count differs from its manifest")
        expected = diagnostic_control_plane_sha256(
            reviewed_commit_sha=self.reviewed_commit_sha,
            reviewed_tree_sha=self.reviewed_tree_sha,
            files=self.files,
        )
        if self.control_plane_sha256 != expected:
            raise ValueError("diagnostic control-plane hash differs from its manifest")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_diagnostic_json_bytes(self.model_dump(mode="json"))


def build_diagnostic_control_plane_provenance(
    *,
    reviewed_commit_sha: str,
    reviewed_tree_sha: str,
    files: Mapping[str, bytes],
    required_paths: Sequence[str],
) -> DiagnosticControlPlaneProvenance:
    """Build provenance from an explicitly closed file set."""

    required = tuple(required_paths)
    if len(required) != len(set(required)):
        raise ValueError("diagnostic required paths must be unique")
    for path in (*required, *files):
        validate_repository_relative_path(path)
    if set(files) != set(required) or len(files) != len(required):
        raise ValueError("diagnostic control-plane files must equal the required path set")
    manifest = tuple(
        DiagnosticControlPlaneFile(
            path=path,
            size_bytes=len(files[path]),
            sha256=hashlib.sha256(files[path]).hexdigest(),
        )
        for path in sorted(required)
    )
    digest = diagnostic_control_plane_sha256(
        reviewed_commit_sha=reviewed_commit_sha,
        reviewed_tree_sha=reviewed_tree_sha,
        files=manifest,
    )
    return DiagnosticControlPlaneProvenance(
        reviewed_commit_sha=reviewed_commit_sha,
        reviewed_tree_sha=reviewed_tree_sha,
        control_plane_sha256=digest,
        file_count=len(manifest),
        files=manifest,
    )


def validate_diagnostic_control_plane_provenance(
    provenance: DiagnosticControlPlaneProvenance | Mapping[str, Any] | bytes,
    *,
    reviewed_commit_sha: str,
    reviewed_tree_sha: str,
    files: Mapping[str, bytes],
    required_paths: Sequence[str],
) -> DiagnosticControlPlaneProvenance:
    """Rebuild and compare exact local or mounted provenance."""

    if isinstance(provenance, bytes):
        observed = _canonical_json_model(provenance, DiagnosticControlPlaneProvenance)
    elif isinstance(provenance, DiagnosticControlPlaneProvenance):
        observed = provenance
    elif isinstance(provenance, Mapping):
        try:
            observed = DiagnosticControlPlaneProvenance.model_validate(provenance)
        except ValidationError as error:
            raise ValueError("diagnostic control-plane schema is invalid") from error
    else:
        raise TypeError("diagnostic control-plane provenance has an unsupported type")
    expected = build_diagnostic_control_plane_provenance(
        reviewed_commit_sha=reviewed_commit_sha,
        reviewed_tree_sha=reviewed_tree_sha,
        files=files,
        required_paths=required_paths,
    )
    if observed != expected:
        raise ValueError("diagnostic control-plane provenance differs from deployed bytes")
    return observed


class DiagnosticReviewedInputs(_StrictDiagnosticModel):
    """Exact reviewed code, BF16 subject, source assets, and workload."""

    schema_version: Literal["inkling-bf16-interface-reviewed-inputs-v1"] = (
        "inkling-bf16-interface-reviewed-inputs-v1"
    )
    control_plane: DiagnosticControlPlaneProvenance
    diagnostic_config: DiagnosticControlPlaneFile
    resolved_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    protocol_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    workload_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    bf16_subject_reference: DiagnosticControlPlaneFile
    source_adoption_reference: DiagnosticControlPlaneFile
    diagnostic_dataset: DiagnosticControlPlaneFile
    runtime_measurement_patch: DiagnosticControlPlaneFile
    subject: Literal["bf16"] = "bf16"
    resources: DiagnosticResources = Field(
        default_factory=lambda: DiagnosticResources(
            provider="modal",
            gpu_type="B300",
            gpu_count=8,
            compute_capability="10.3",
            cpu_cores=16,
            memory_gib=64,
            ephemeral_disk_mib=2_097_152,
            startup_timeout_seconds=1_800,
            function_timeout_seconds=86_400,
            max_containers=1,
            max_attempts=1,
            network_access=False,
            cpu_fallback_allowed=False,
        )
    )

    @model_validator(mode="after")
    def roles_are_in_closure(self) -> DiagnosticReviewedInputs:
        expected_paths = {
            "diagnostic_config": DIAGNOSTIC_CONFIG_RELATIVE_PATH,
            "bf16_subject_reference": ("configs/experiments/inkling_bf16_subject_reference.json"),
            "source_adoption_reference": (
                "configs/experiments/inkling_q3_k_m_source_adoption.json"
            ),
            "diagnostic_dataset": ("configs/experiments/inkling_quality_diagnostic_v1.jsonl"),
            "runtime_measurement_patch": "patches/inkling-measurement-a015409.patch",
        }
        bound = []
        for role, expected_path in expected_paths.items():
            item = getattr(self, role)
            if item.path != expected_path:
                raise ValueError(f"diagnostic reviewed role {role} has the wrong path")
            bound.append(item)
        manifest = {item.path: item for item in self.control_plane.files}
        if any(manifest.get(item.path) != item for item in bound):
            raise ValueError("diagnostic reviewed input differs from its control closure")
        return self


def diagnostic_app_name(control_plane_sha256: str) -> str:
    _validate_sha256(control_plane_sha256, label="diagnostic control-plane hash")
    return f"inkling-bf16-diag-{control_plane_sha256[:12]}"


def diagnostic_deployment_tag(control_plane_sha256: str) -> str:
    _validate_sha256(control_plane_sha256, label="diagnostic control-plane hash")
    return f"iql-bf16-diag-{control_plane_sha256[:34]}"


class DiagnosticDeployConfirmationChallenge(_StrictDiagnosticModel):
    schema_version: Literal["inkling-bf16-interface-deploy-confirmation-v1"] = (
        "inkling-bf16-interface-deploy-confirmation-v1"
    )
    status: Literal["prepared_before_deploy"] = "prepared_before_deploy"
    created_at_utc: StrictStr
    expires_at_utc: StrictStr
    confirmation_nonce: StrictStr = Field(pattern=_SHA256_PATTERN)
    reviewed_inputs: DiagnosticReviewedInputs
    app_name: StrictStr = Field(pattern=r"^inkling-bf16-diag-[0-9a-f]{12}$")
    environment_name: Literal["inkling-quant"] = DIAGNOSTIC_ENVIRONMENT_NAME
    function_name: Literal["run_bf16_interface_diagnostic"] = DIAGNOSTIC_FUNCTION_NAME
    attempt_registry_name: Literal["inkling-measurement-attempt-registry-v1"] = (
        DIAGNOSTIC_ATTEMPT_REGISTRY_NAME
    )
    evidence_volume_name: Literal["inkling-measurement-evidence-v1"] = (
        DIAGNOSTIC_EVIDENCE_VOLUME_NAME
    )
    starts_gpu_compute: Literal[False] = False

    @field_validator("created_at_utc", "expires_at_utc")
    @classmethod
    def times_are_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="diagnostic deploy challenge time")

    @model_validator(mode="after")
    def exact_app_and_lifetime(self) -> DiagnosticDeployConfirmationChallenge:
        if self.app_name != diagnostic_app_name(
            self.reviewed_inputs.control_plane.control_plane_sha256
        ):
            raise ValueError("diagnostic deploy challenge has the wrong App name")
        created = _utc_datetime(self.created_at_utc, label="deploy challenge creation time")
        expires = _utc_datetime(self.expires_at_utc, label="deploy challenge expiration time")
        if not created < expires or expires - created > timedelta(
            seconds=DIAGNOSTIC_DEPLOY_CHALLENGE_MAX_AGE_SECONDS
        ):
            raise ValueError("diagnostic deploy challenge lifetime is invalid")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_diagnostic_json_bytes(self.model_dump(mode="json"))

    def challenge_sha256(self) -> str:
        return _domain_hash(DIAGNOSTIC_DEPLOY_CHALLENGE_HASH_DOMAIN, self.model_dump(mode="json"))

    def confirmation_text(self) -> str:
        return f"{DIAGNOSTIC_DEPLOY_CONFIRMATION_PREFIX}\n{self.challenge_sha256()}"

    def confirm(self, value: str) -> DiagnosticDeployConfirmationChallenge:
        if type(value) is not str or value != self.confirmation_text():
            raise ValueError("diagnostic deploy confirmation does not match its challenge")
        return self


def validate_diagnostic_deploy_challenge_not_expired(
    challenge: DiagnosticDeployConfirmationChallenge,
    *,
    observed_at_utc: str,
) -> DiagnosticDeployConfirmationChallenge:
    observed = _utc_datetime(observed_at_utc, label="deploy challenge observation time")
    created = _utc_datetime(challenge.created_at_utc, label="deploy challenge creation time")
    expires = _utc_datetime(challenge.expires_at_utc, label="deploy challenge expiration time")
    if not created <= observed < expires:
        raise ValueError("diagnostic deploy challenge is not active")
    return challenge


class DiagnosticDeploymentIdentity(_StrictDiagnosticModel):
    schema_version: Literal["inkling-bf16-interface-deployment-v1"] = (
        "inkling-bf16-interface-deployment-v1"
    )
    deployed_at_utc: StrictStr
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    app_name: StrictStr = Field(pattern=r"^inkling-bf16-diag-[0-9a-f]{12}$")
    environment_name: Literal["inkling-quant"] = DIAGNOSTIC_ENVIRONMENT_NAME
    deployment_version: StrictInt = Field(gt=0)
    deployment_tag: StrictStr = Field(pattern=r"^iql-bf16-diag-[0-9a-f]{34}$")
    function_id: StrictStr = Field(pattern=_MODAL_FUNCTION_ID_PATTERN)
    function_name: Literal["run_bf16_interface_diagnostic"] = DIAGNOSTIC_FUNCTION_NAME
    attempt_registry_name: Literal["inkling-measurement-attempt-registry-v1"] = (
        DIAGNOSTIC_ATTEMPT_REGISTRY_NAME
    )
    attempt_registry_id: StrictStr = Field(pattern=_MODAL_DICT_ID_PATTERN)
    attempt_registry_created_at_utc: StrictStr
    evidence_volume_name: Literal["inkling-measurement-evidence-v1"] = (
        DIAGNOSTIC_EVIDENCE_VOLUME_NAME
    )
    evidence_volume_id: StrictStr = Field(pattern=_MODAL_VOLUME_ID_PATTERN)

    @field_validator("deployed_at_utc", "attempt_registry_created_at_utc")
    @classmethod
    def times_are_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="diagnostic deployment time")

    @model_validator(mode="after")
    def exact_derived_identity(self) -> DiagnosticDeploymentIdentity:
        if self.app_name != diagnostic_app_name(self.control_plane_sha256):
            raise ValueError("diagnostic deployment App differs from control identity")
        if self.deployment_tag != diagnostic_deployment_tag(self.control_plane_sha256):
            raise ValueError("diagnostic deployment tag differs from control identity")
        if _utc_datetime(
            self.attempt_registry_created_at_utc,
            label="attempt registry creation time",
        ) > _utc_datetime(self.deployed_at_utc, label="deployment time"):
            raise ValueError("diagnostic deployment predates its attempt registry")
        return self

    def validate_reviewed_inputs(
        self, reviewed_inputs: DiagnosticReviewedInputs
    ) -> DiagnosticDeploymentIdentity:
        if self.control_plane_sha256 != reviewed_inputs.control_plane.control_plane_sha256:
            raise ValueError("diagnostic deployment differs from reviewed control")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_diagnostic_json_bytes(self.model_dump(mode="json"))


class DiagnosticLaunchConfirmationChallenge(_StrictDiagnosticModel):
    schema_version: Literal["inkling-bf16-interface-launch-confirmation-v1"] = (
        "inkling-bf16-interface-launch-confirmation-v1"
    )
    status: Literal["prepared_before_launch"] = "prepared_before_launch"
    created_at_utc: StrictStr
    expires_at_utc: StrictStr
    authorization_nonce: StrictStr = Field(pattern=_SHA256_PATTERN)
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    reviewed_inputs: DiagnosticReviewedInputs
    deployment: DiagnosticDeploymentIdentity

    @field_validator("created_at_utc", "expires_at_utc")
    @classmethod
    def times_are_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="diagnostic launch challenge time")

    @model_validator(mode="after")
    def exact_bindings_and_lifetime(self) -> DiagnosticLaunchConfirmationChallenge:
        self.deployment.validate_reviewed_inputs(self.reviewed_inputs)
        created = _utc_datetime(self.created_at_utc, label="launch challenge creation time")
        expires = _utc_datetime(self.expires_at_utc, label="launch challenge expiration time")
        deployed = _utc_datetime(self.deployment.deployed_at_utc, label="deployment time")
        if (
            created < deployed
            or not created < expires
            or expires - created > timedelta(seconds=DIAGNOSTIC_LAUNCH_CHALLENGE_MAX_AGE_SECONDS)
        ):
            raise ValueError("diagnostic launch challenge lifetime is invalid")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_diagnostic_json_bytes(self.model_dump(mode="json"))

    def challenge_sha256(self) -> str:
        return _domain_hash(DIAGNOSTIC_LAUNCH_CHALLENGE_HASH_DOMAIN, self.model_dump(mode="json"))

    def confirmation_text(self) -> str:
        return f"{DIAGNOSTIC_LAUNCH_CONFIRMATION_PREFIX}\n{self.challenge_sha256()}"

    def confirm(self, value: str) -> DiagnosticLaunchConfirmationChallenge:
        if type(value) is not str or value != self.confirmation_text():
            raise ValueError("diagnostic launch confirmation does not match its challenge")
        return self


class DiagnosticLaunchIntent(_StrictDiagnosticModel):
    schema_version: Literal["inkling-bf16-interface-launch-intent-v1"] = (
        "inkling-bf16-interface-launch-intent-v1"
    )
    status: Literal["authorized_before_spawn"] = "authorized_before_spawn"
    authorization_scope: Literal["one_bf16_interface_diagnostic_attempt"] = (
        "one_bf16_interface_diagnostic_attempt"
    )
    authorized_at_utc: StrictStr
    expires_at_utc: StrictStr
    launch_challenge_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    authorization_nonce: StrictStr = Field(pattern=_SHA256_PATTERN)
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    reviewed_inputs: DiagnosticReviewedInputs
    deployment: DiagnosticDeploymentIdentity
    subject: Literal["bf16"] = "bf16"
    resources: DiagnosticResources
    one_atomic_attempt: Literal[True] = True
    one_server_load: Literal[True] = True
    sequential_request_count: Literal[16] = 16
    rehash_all_subject_files: Literal[True] = True
    partial_success_allowed: Literal[False] = False
    diagnostic_execution_allowed: Literal[True] = True

    @field_validator("authorized_at_utc", "expires_at_utc")
    @classmethod
    def times_are_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="diagnostic launch intent time")

    @model_validator(mode="after")
    def exact_redundant_bindings(self) -> DiagnosticLaunchIntent:
        if self.resources != self.reviewed_inputs.resources:
            raise ValueError("diagnostic launch resources differ from reviewed inputs")
        self.deployment.validate_reviewed_inputs(self.reviewed_inputs)
        authorized = _utc_datetime(self.authorized_at_utc, label="authorization time")
        expires = _utc_datetime(self.expires_at_utc, label="authorization expiration time")
        if not authorized < expires:
            raise ValueError("diagnostic launch authorization is expired")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_diagnostic_json_bytes(self.model_dump(mode="json"))

    def intent_sha256(self) -> str:
        return _domain_hash(DIAGNOSTIC_LAUNCH_INTENT_HASH_DOMAIN, self.model_dump(mode="json"))


def build_diagnostic_launch_intent(
    challenge: DiagnosticLaunchConfirmationChallenge,
    *,
    confirmation: str,
    authorized_at_utc: str,
) -> DiagnosticLaunchIntent:
    challenge.confirm(confirmation)
    authorized = _utc_datetime(authorized_at_utc, label="authorization time")
    created = _utc_datetime(challenge.created_at_utc, label="launch challenge creation time")
    expires = _utc_datetime(challenge.expires_at_utc, label="launch challenge expiration time")
    if not created <= authorized < expires:
        raise ValueError("diagnostic authorization is outside its challenge lifetime")
    return DiagnosticLaunchIntent(
        authorized_at_utc=authorized_at_utc,
        expires_at_utc=challenge.expires_at_utc,
        launch_challenge_sha256=challenge.challenge_sha256(),
        authorization_nonce=challenge.authorization_nonce,
        run_id=challenge.run_id,
        reviewed_inputs=challenge.reviewed_inputs,
        deployment=challenge.deployment,
        resources=challenge.reviewed_inputs.resources,
    )


def validate_diagnostic_launch_intent_not_expired(
    intent: DiagnosticLaunchIntent,
    *,
    observed_at_utc: str,
) -> DiagnosticLaunchIntent:
    observed = _utc_datetime(observed_at_utc, label="launch intent observation time")
    authorized = _utc_datetime(intent.authorized_at_utc, label="authorization time")
    expires = _utc_datetime(intent.expires_at_utc, label="authorization expiration time")
    if not authorized <= observed < expires:
        raise ValueError("diagnostic launch intent is not active")
    return intent


def diagnostic_launch_intent_path(run_id: str, intent_sha256: str) -> str:
    _validate_run_id(run_id)
    _validate_sha256(intent_sha256, label="diagnostic launch-intent hash")
    return PurePosixPath(
        "runs", run_id, DIAGNOSTIC_STAGE, "control", "launch-intents", f"{intent_sha256}.json"
    ).as_posix()


def _require_relative_or_exact_evidence_path(evidence_path: str, relative: str) -> None:
    if evidence_path == relative:
        return
    absolute = validate_absolute_path(evidence_path)
    expected = PurePosixPath("/evidence", relative).as_posix()
    if absolute != expected:
        raise ValueError("diagnostic evidence path differs from its content binding")


def validate_diagnostic_launch_intent(
    payload: bytes,
    *,
    expected: DiagnosticLaunchIntent,
    intent_sha256: str,
    evidence_path: str,
) -> DiagnosticLaunchIntent:
    observed = _canonical_json_model(payload, DiagnosticLaunchIntent)
    if observed.intent_sha256() != intent_sha256:
        raise ValueError("diagnostic launch-intent hash differs from canonical bytes")
    relative = diagnostic_launch_intent_path(observed.run_id, intent_sha256)
    _require_relative_or_exact_evidence_path(evidence_path, relative)
    if observed != expected:
        raise ValueError("diagnostic launch intent differs from expected")
    return observed


class DiagnosticPostSpawnAcceptance(_StrictDiagnosticModel):
    schema_version: Literal["inkling-bf16-interface-post-spawn-acceptance-v1"] = (
        "inkling-bf16-interface-post-spawn-acceptance-v1"
    )
    status: Literal["accepted_after_spawn"] = "accepted_after_spawn"
    accepted_at_utc: StrictStr
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    call_id: StrictStr = Field(pattern=_MODAL_CALL_ID_PATTERN)
    deployment: DiagnosticDeploymentIdentity
    reviewed_config_file_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    resolved_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("accepted_at_utc")
    @classmethod
    def accepted_time_is_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="diagnostic acceptance time")

    @model_validator(mode="after")
    def exact_control_binding(self) -> DiagnosticPostSpawnAcceptance:
        if self.deployment.control_plane_sha256 != self.control_plane_sha256:
            raise ValueError("diagnostic acceptance deployment differs from control")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_diagnostic_json_bytes(self.model_dump(mode="json"))

    def acceptance_sha256(self) -> str:
        return _domain_hash(DIAGNOSTIC_ACCEPTANCE_HASH_DOMAIN, self.model_dump(mode="json"))


def build_diagnostic_post_spawn_acceptance(
    intent: DiagnosticLaunchIntent,
    *,
    accepted_at_utc: str,
    call_id: str,
) -> DiagnosticPostSpawnAcceptance:
    validate_diagnostic_launch_intent_not_expired(intent, observed_at_utc=accepted_at_utc)
    return DiagnosticPostSpawnAcceptance(
        accepted_at_utc=accepted_at_utc,
        run_id=intent.run_id,
        launch_intent_sha256=intent.intent_sha256(),
        call_id=call_id,
        deployment=intent.deployment,
        reviewed_config_file_sha256=intent.reviewed_inputs.diagnostic_config.sha256,
        resolved_config_sha256=intent.reviewed_inputs.resolved_config_sha256,
        control_plane_sha256=intent.reviewed_inputs.control_plane.control_plane_sha256,
    )


def diagnostic_post_spawn_acceptance_path(run_id: str, launch_intent_sha256: str) -> str:
    _validate_run_id(run_id)
    _validate_sha256(launch_intent_sha256, label="diagnostic launch-intent hash")
    return PurePosixPath(
        "runs",
        run_id,
        DIAGNOSTIC_STAGE,
        "control",
        "post-spawn-acceptances",
        f"{launch_intent_sha256}.json",
    ).as_posix()


def validate_diagnostic_post_spawn_acceptance(
    payload: bytes,
    *,
    expected: DiagnosticPostSpawnAcceptance,
    acceptance_sha256: str,
    evidence_path: str,
) -> DiagnosticPostSpawnAcceptance:
    observed = _canonical_json_model(payload, DiagnosticPostSpawnAcceptance)
    if observed.acceptance_sha256() != acceptance_sha256:
        raise ValueError("diagnostic acceptance hash differs from canonical bytes")
    relative = diagnostic_post_spawn_acceptance_path(observed.run_id, observed.launch_intent_sha256)
    _require_relative_or_exact_evidence_path(evidence_path, relative)
    if observed != expected:
        raise ValueError("diagnostic acceptance differs from expected")
    return observed


def diagnostic_attempt_registry_key(run_id: str, stage: str = DIAGNOSTIC_STAGE) -> str:
    _validate_run_id(run_id)
    if stage != DIAGNOSTIC_STAGE:
        raise ValueError("diagnostic attempt stage is invalid")
    return f"{run_id}:{stage}"


class DiagnosticAttemptClaim(_StrictDiagnosticModel):
    schema_version: Literal["inkling-bf16-interface-attempt-claim-v1"] = (
        "inkling-bf16-interface-attempt-claim-v1"
    )
    registry_name: Literal["inkling-measurement-attempt-registry-v1"] = (
        DIAGNOSTIC_ATTEMPT_REGISTRY_NAME
    )
    registry_id: StrictStr = Field(pattern=_MODAL_DICT_ID_PATTERN)
    registry_created_at_utc: StrictStr
    claimed_at_utc: StrictStr
    registry_key: StrictStr
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    stage: Literal["bf16_interface_diagnostic"] = DIAGNOSTIC_STAGE
    call_id: StrictStr = Field(pattern=_MODAL_CALL_ID_PATTERN)
    input_id: StrictStr = Field(pattern=_MODAL_INPUT_ID_PATTERN)
    task_id: StrictStr = Field(pattern=_MODAL_TASK_ID_PATTERN)
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    post_spawn_acceptance_path: StrictStr
    post_spawn_acceptance_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    reviewed_config_file_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    resolved_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    subject: Literal["bf16"] = "bf16"

    @field_validator("registry_created_at_utc", "claimed_at_utc")
    @classmethod
    def times_are_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="diagnostic attempt time")

    @field_validator("post_spawn_acceptance_path")
    @classmethod
    def acceptance_path_is_canonical(cls, value: str) -> str:
        return validate_repository_relative_path(value)

    @model_validator(mode="after")
    def exact_derived_bindings(self) -> DiagnosticAttemptClaim:
        if self.registry_key != diagnostic_attempt_registry_key(self.run_id, self.stage):
            raise ValueError("diagnostic attempt registry key differs from run and stage")
        expected = diagnostic_post_spawn_acceptance_path(self.run_id, self.launch_intent_sha256)
        if self.post_spawn_acceptance_path != expected:
            raise ValueError("diagnostic attempt acceptance path is invalid")
        if _utc_datetime(self.claimed_at_utc, label="claim time") < _utc_datetime(
            self.registry_created_at_utc,
            label="registry creation time",
        ):
            raise ValueError("diagnostic attempt claim predates its registry")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_diagnostic_json_bytes(self.model_dump(mode="json"))

    def claim_sha256(self) -> str:
        return _domain_hash(DIAGNOSTIC_ATTEMPT_CLAIM_HASH_DOMAIN, self.model_dump(mode="json"))


class DiagnosticAttemptRegistryProtocol(Protocol):
    def put(self, key: Any, value: Any, *, skip_if_exists: bool = False) -> bool: ...


def build_diagnostic_attempt_claim(
    intent: DiagnosticLaunchIntent,
    acceptance: DiagnosticPostSpawnAcceptance,
    *,
    claimed_at_utc: str,
    input_id: str,
    task_id: str,
) -> DiagnosticAttemptClaim:
    expected_acceptance = build_diagnostic_post_spawn_acceptance(
        intent,
        accepted_at_utc=acceptance.accepted_at_utc,
        call_id=acceptance.call_id,
    )
    if acceptance != expected_acceptance:
        raise ValueError("diagnostic acceptance differs from its launch intent")
    if _utc_datetime(claimed_at_utc, label="claim time") < _utc_datetime(
        acceptance.accepted_at_utc,
        label="acceptance time",
    ):
        raise ValueError("diagnostic attempt claim predates provider acceptance")
    deployment = intent.deployment
    return DiagnosticAttemptClaim(
        registry_id=deployment.attempt_registry_id,
        registry_created_at_utc=deployment.attempt_registry_created_at_utc,
        claimed_at_utc=claimed_at_utc,
        registry_key=diagnostic_attempt_registry_key(intent.run_id),
        run_id=intent.run_id,
        call_id=acceptance.call_id,
        input_id=input_id,
        task_id=task_id,
        launch_intent_sha256=intent.intent_sha256(),
        post_spawn_acceptance_path=diagnostic_post_spawn_acceptance_path(
            intent.run_id, intent.intent_sha256()
        ),
        post_spawn_acceptance_sha256=acceptance.acceptance_sha256(),
        reviewed_config_file_sha256=intent.reviewed_inputs.diagnostic_config.sha256,
        resolved_config_sha256=intent.reviewed_inputs.resolved_config_sha256,
        control_plane_sha256=intent.reviewed_inputs.control_plane.control_plane_sha256,
    )


def claim_diagnostic_attempt(
    registry: DiagnosticAttemptRegistryProtocol,
    claim: DiagnosticAttemptClaim,
) -> str:
    created = registry.put(claim.registry_key, claim.canonical_bytes(), skip_if_exists=True)
    if created is not True:
        raise RuntimeError("The one authorized BF16 diagnostic attempt was already consumed")
    return claim.claim_sha256()


def diagnostic_attempt_claim_path(run_id: str, claim_sha256: str) -> str:
    _validate_run_id(run_id)
    _validate_sha256(claim_sha256, label="diagnostic attempt-claim hash")
    return PurePosixPath(
        "runs",
        run_id,
        DIAGNOSTIC_STAGE,
        "control",
        "attempt-claims",
        f"{claim_sha256}.json",
    ).as_posix()


def validate_diagnostic_attempt_claim(
    payload: bytes,
    *,
    expected: DiagnosticAttemptClaim,
    claim_sha256: str,
    evidence_path: str,
) -> DiagnosticAttemptClaim:
    observed = _canonical_json_model(payload, DiagnosticAttemptClaim)
    if observed.claim_sha256() != claim_sha256:
        raise ValueError("diagnostic attempt-claim hash differs from canonical bytes")
    relative = diagnostic_attempt_claim_path(observed.run_id, claim_sha256)
    _require_relative_or_exact_evidence_path(evidence_path, relative)
    if observed != expected:
        raise ValueError("diagnostic attempt claim differs from expected")
    return observed


def diagnostic_absolute_evidence_path(evidence_root: str, relative_path: str) -> str:
    """Join one canonical evidence mount and repository-relative record path."""

    root = validate_absolute_path(evidence_root)
    relative = validate_repository_relative_path(relative_path)
    return validate_absolute_path((PurePosixPath(root) / PurePosixPath(relative)).as_posix())


class DiagnosticEogTokenProbe(_StrictDiagnosticModel):
    """Private result for one forced token and runtime EOG-classification probe."""

    token_id: Literal[199999, 200006]
    runtime_is_eog: StrictBool
    forced_token_observed: Literal[True]
    request_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    input_token_ids: tuple[StrictInt, ...]
    generated_token_ids: tuple[StrictInt, ...]
    stop: Literal[True]
    stop_type: Literal["eos", "limit"]
    truncated: Literal[False]
    response_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    request_duration_seconds: StrictFloat = Field(gt=0.0)
    timings: MeasurementDiagnosticTimings

    @model_validator(mode="after")
    def forced_token_result_is_exact(self) -> DiagnosticEogTokenProbe:
        if not self.input_token_ids:
            raise ValueError("diagnostic probe must retain private input token IDs")
        if any(token_id < 0 for token_id in self.input_token_ids):
            raise ValueError("diagnostic probe input token IDs must be nonnegative")
        if self.generated_token_ids != (self.token_id,):
            raise ValueError("diagnostic probe must generate exactly its forced token")
        if self.runtime_is_eog != (self.stop_type == "eos"):
            raise ValueError("diagnostic probe EOG classification differs from its stop type")
        if self.timings.prompt_n != len(self.input_token_ids) or self.timings.predicted_n != len(
            self.generated_token_ids
        ):
            raise ValueError("diagnostic probe timings differ from retained token counts")
        request = {
            "prompt": list(self.input_token_ids),
            "n_predict": 1,
            "temperature": -1.0,
            "seed": 42,
            "stream": False,
            "cache_prompt": False,
            "return_tokens": True,
            "timings_per_token": True,
            "ignore_eos": False,
            "stop": [],
            "logit_bias": [[self.token_id, 1_000_000_000.0]],
        }
        expected_request_sha256 = hashlib.sha256(
            canonical_diagnostic_json_bytes(request)
        ).hexdigest()
        if self.request_sha256 != expected_request_sha256:
            raise ValueError("diagnostic probe request hash differs from its exact request")
        return self


class DiagnosticEogEvidence(_StrictDiagnosticModel):
    """Private source-EOS declaration and runtime classification observations."""

    source_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    source_config_eos_token_id: Literal[200006]
    runtime_eog_token_ids: tuple[StrictInt, ...]
    source_eos_probe: DiagnosticEogTokenProbe
    comparison_token_probe: DiagnosticEogTokenProbe

    @model_validator(mode="after")
    def probes_match_runtime_set(self) -> DiagnosticEogEvidence:
        token_ids = self.runtime_eog_token_ids
        if (
            not token_ids
            or any(token_id < 0 for token_id in token_ids)
            or token_ids != tuple(sorted(set(token_ids)))
        ):
            raise ValueError("runtime EOG token IDs must be nonempty, sorted, and unique")
        if self.source_eos_probe.token_id != DIAGNOSTIC_EOS_TOKEN_ID:
            raise ValueError("source-EOS probe used the wrong token")
        if self.comparison_token_probe.token_id != DIAGNOSTIC_COMPARISON_TOKEN_ID:
            raise ValueError("comparison-token probe used the wrong token")
        runtime_set = set(token_ids)
        for probe in (self.source_eos_probe, self.comparison_token_probe):
            if probe.runtime_is_eog != (probe.token_id in runtime_set):
                raise ValueError("runtime EOG classification differs from the recorded EOG set")
        return self


class DiagnosticPrivateTrial(_StrictDiagnosticModel):
    """Private reversible evidence for one of the sixteen sequential requests."""

    ordinal: StrictInt = Field(ge=1, le=DIAGNOSTIC_REQUEST_COUNT)
    item_id: DiagnosticItemId
    cell: DiagnosticCellName
    prompt_mode: Literal["raw", "chat_template"]
    cap_mode: Literal["original", "fixed_64"]
    original_max_new_tokens: Literal[4, 8, 16]
    requested_max_new_tokens: Literal[4, 8, 16, 64]
    reasoning_effort: Literal["none"]
    item_prompt_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    rendered_prompt_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    request_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    response_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    input_token_ids: tuple[StrictInt, ...]
    output_token_ids: tuple[StrictInt, ...]
    tokens_evaluated: StrictInt = Field(ge=0)
    tokens_predicted: StrictInt = Field(ge=0)
    stop: Literal[True]
    stop_type: Literal["eos", "limit"]
    eog_observed: StrictBool
    cap_hit: StrictBool
    truncated: Literal[False]
    whole_output_passed: StrictBool
    extracted_content_passed: StrictBool
    score_detail_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    request_duration_seconds: StrictFloat = Field(gt=0.0)
    timings: MeasurementDiagnosticTimings

    @model_validator(mode="after")
    def exact_request_shape(self) -> DiagnosticPrivateTrial:
        expected = {
            "raw_original": ("raw", "original"),
            "raw_64": ("raw", "fixed_64"),
            "chat_original": ("chat_template", "original"),
            "chat_64": ("chat_template", "fixed_64"),
        }[self.cell]
        if (self.prompt_mode, self.cap_mode) != expected:
            raise ValueError("diagnostic trial interface differs from its cell")
        requested = self.original_max_new_tokens if self.cap_mode == "original" else 64
        if self.requested_max_new_tokens != requested:
            raise ValueError("diagnostic trial token cap differs from its cell")
        if not self.input_token_ids:
            raise ValueError("diagnostic request must retain private input token IDs")
        if any(token_id < 0 for token_id in (*self.input_token_ids, *self.output_token_ids)):
            raise ValueError("diagnostic trial token IDs must be nonnegative")
        if self.tokens_predicted != len(self.output_token_ids):
            raise ValueError("predicted-token count differs from retained private output tokens")
        if self.tokens_evaluated != len(self.input_token_ids):
            raise ValueError("evaluated-token count differs from submitted private input tokens")
        if (
            self.timings.prompt_n != self.tokens_evaluated
            or self.timings.predicted_n != self.tokens_predicted
        ):
            raise ValueError("diagnostic server timings differ from retained token counts")
        if self.tokens_predicted > self.requested_max_new_tokens:
            raise ValueError("diagnostic request exceeded its checked generation cap")
        if self.cap_hit != (self.stop_type == "limit"):
            raise ValueError("diagnostic cap flag differs from the server stop type")
        if self.stop_type == "limit" and (self.tokens_predicted != self.requested_max_new_tokens):
            raise ValueError("limit-stopped diagnostic request did not reach its token cap")
        if self.eog_observed != (self.stop_type == "eos"):
            raise ValueError("diagnostic EOG flag differs from the server stop type")
        return self

    def private_sha256(self) -> str:
        return _domain_hash(
            DIAGNOSTIC_RAW_HASH_DOMAIN,
            {"kind": "trial", "trial": self.model_dump(mode="json")},
        )


class DiagnosticPrivateRawEvidence(_StrictDiagnosticModel):
    """Complete private record; token arrays must never be copied into public receipts."""

    schema_version: Literal["inkling-bf16-interface-private-raw-v1"] = (
        "inkling-bf16-interface-private-raw-v1"
    )
    record_scope: Literal["private_reversible_token_evidence"] = "private_reversible_token_evidence"
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    stage: Literal["bf16_interface_diagnostic"] = DIAGNOSTIC_STAGE
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    reviewed_config_file_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    resolved_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    post_spawn_acceptance_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    call_id: StrictStr = Field(pattern=_MODAL_CALL_ID_PATTERN)
    attempt_claim_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    model_id: Literal["thinkingmachines/Inkling"]
    model_revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"]
    architecture: Literal["InklingForConditionalGeneration"]
    protocol_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    workload_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    bf16_inventory_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    bf16_shard_count: Literal[49]
    bf16_total_bytes: Literal[1894278547552]
    source_asset_manifest_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    runtime_identity: MeasurementRuntimeIdentity
    runtime_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    runtime_manifest_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    hardware_identity: MeasurementHardwareIdentity
    hardware_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    command: tuple[StrictStr, ...] = Field(min_length=1, max_length=64)
    command_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    server_process_id: StrictInt = Field(gt=0)
    server_log_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    server_log_size_bytes: StrictInt = Field(gt=0)
    text_artifact_load: TextArtifactLoadEvidence
    backend: Literal["CUDA"]
    logical_devices: Literal["cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7"]
    gpu_device_count: Literal[8]
    gpu_model_graph_operation_count: StrictInt = Field(gt=0)
    cpu_model_graph_operation_count: Literal[0]
    cpu_fallback_observed: Literal[False]
    resource_sample_summary: MeasurementResourceSampleSummary
    eog: DiagnosticEogEvidence
    trials: tuple[DiagnosticPrivateTrial, ...]
    prompt_text_recorded: Literal[False]
    output_text_recorded: Literal[False]
    private_token_ids_recorded: Literal[True]
    one_server_load: Literal[True]
    sequential_request_count: Literal[16]
    started_at_utc: StrictStr
    completed_at_utc: StrictStr
    diagnostic_only: Literal[True]
    quality_retention_claim_allowed: Literal[False]
    quality_claim_allowed: Literal[False]
    speedup_claim_allowed: Literal[False]
    performance_claim_allowed: Literal[False]
    mtp_included: Literal[False]
    mtp_supported: Literal[False]
    routing_drift_supported: Literal[False]
    single_run_causation_claim_allowed: Literal[False]

    @field_validator("started_at_utc", "completed_at_utc")
    @classmethod
    def times_are_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="diagnostic raw evidence time")

    @field_validator("command")
    @classmethod
    def command_strings_are_safe(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item or "\x00" in item for item in value):
            raise ValueError("diagnostic command contains an empty or NUL argument")
        return value

    @model_validator(mode="after")
    def complete_matrix_is_exact(self) -> DiagnosticPrivateRawEvidence:
        if _utc_datetime(self.completed_at_utc, label="completion time") < _utc_datetime(
            self.started_at_utc,
            label="start time",
        ):
            raise ValueError("diagnostic raw evidence completes before it starts")
        if len(self.trials) != DIAGNOSTIC_REQUEST_COUNT:
            raise ValueError("diagnostic raw evidence must contain exactly sixteen requests")
        model_paths = tuple(
            self.command[index + 1]
            for index, argument in enumerate(self.command[:-1])
            if argument == "-m"
        )
        if model_paths != (self.text_artifact_load.first_shard_path,):
            raise ValueError("diagnostic loader evidence differs from the exact model command")
        if self.text_artifact_load.total_shards_loaded != self.bf16_shard_count:
            raise ValueError("diagnostic loader evidence differs from the BF16 shard inventory")
        expected_pairs = tuple(
            (cell, item_id) for cell in DIAGNOSTIC_CELL_ORDER for item_id in DIAGNOSTIC_ITEM_ORDER
        )
        observed_pairs = tuple((trial.cell, trial.item_id) for trial in self.trials)
        if observed_pairs != expected_pairs:
            raise ValueError("diagnostic requests differ from the checked cell-major order")
        if tuple(trial.ordinal for trial in self.trials) != tuple(
            range(1, DIAGNOSTIC_REQUEST_COUNT + 1)
        ):
            raise ValueError("diagnostic request ordinals are incomplete or out of order")
        original_caps = {"text_01": 4, "text_02": 8, "text_03": 4, "text_04": 16}
        for trial in self.trials:
            if trial.original_max_new_tokens != original_caps[trial.item_id]:
                raise ValueError("diagnostic original token cap differs from the checked item")
            if not trial.output_token_ids:
                raise ValueError("diagnostic request must retain at least one output token")
            if any(
                token_id in self.eog.runtime_eog_token_ids
                for token_id in trial.output_token_ids[:-1]
            ):
                raise ValueError("diagnostic request continued after a runtime EOG token")
            ended_in_runtime_eog = trial.output_token_ids[-1] in self.eog.runtime_eog_token_ids
            if trial.eog_observed != ended_in_runtime_eog:
                raise ValueError(
                    "diagnostic request stop type differs from its final runtime EOG token"
                )
        raw_text_01_input_ids = tuple(
            trial.input_token_ids
            for trial in self.trials
            if trial.item_id == "text_01" and trial.cell in {"raw_original", "raw_64"}
        )
        if len(raw_text_01_input_ids) != 2 or raw_text_01_input_ids[0] != raw_text_01_input_ids[1]:
            raise ValueError("raw text_01 diagnostic cells used different input token IDs")
        for probe in (self.eog.source_eos_probe, self.eog.comparison_token_probe):
            if probe.input_token_ids != raw_text_01_input_ids[0]:
                raise ValueError("diagnostic EOG probe input differs from raw text_01")
        if self.eog.source_config_sha256 != (
            "58720f145bcecef9a7ab2b419ab346e7c634af8d2f3e7362e900d00f789ea46c"
        ):
            raise ValueError("diagnostic EOG evidence used a different source config")
        if self.runtime_identity_sha256 != diagnostic_runtime_identity_sha256(
            self.runtime_identity
        ):
            raise ValueError("diagnostic runtime identity hash differs from its contents")
        if self.runtime_manifest_sha256 != self.runtime_identity.manifest_sha256:
            raise ValueError("diagnostic runtime manifest hash differs from its identity")
        if self.hardware_identity_sha256 != self.hardware_identity.identity_sha256:
            raise ValueError("diagnostic hardware identity hash differs from its contents")
        if (
            self.backend != self.hardware_identity.backend
            or self.logical_devices != ",".join(self.hardware_identity.logical_devices)
            or self.gpu_device_count != len(self.hardware_identity.gpus)
            or self.hardware_identity.gpu_layers != "all"
            or self.hardware_identity.cpu_moe_layers != 0
            or self.hardware_identity.cpu_fallback
        ):
            raise ValueError("diagnostic placement fields differ from hardware identity")
        expected_command_sha256 = hashlib.sha256(
            canonical_diagnostic_json_bytes(list(self.command))
        ).hexdigest()
        if self.command_sha256 != expected_command_sha256:
            raise ValueError("diagnostic command hash differs from its exact argument vector")
        if len(self.command) <= 4 or self.command != build_diagnostic_server_command(
            self.command[4]
        ):
            raise ValueError("diagnostic command differs from the exact server argument vector")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_diagnostic_json_bytes(self.model_dump(mode="json"))


def parse_diagnostic_private_raw_evidence(
    payload: bytes,
    *,
    run_id: str,
) -> DiagnosticPrivateRawEvidence:
    """Parse exact canonical private evidence under its separate larger size limit."""

    if not isinstance(payload, bytes):
        raise TypeError("diagnostic private raw evidence must be bytes")
    _validate_run_id(run_id)
    observed = _canonical_json_model(
        payload,
        DiagnosticPrivateRawEvidence,
        maximum_bytes=DIAGNOSTIC_RAW_RECORD_MAX_BYTES,
    )
    if observed.run_id != run_id:
        raise ValueError("diagnostic private raw evidence has the wrong run ID")
    return observed


def validate_diagnostic_private_trials(
    raw: DiagnosticPrivateRawEvidence,
    *,
    bundle: InklingBF16InterfaceDiagnosticBundle,
) -> DiagnosticPrivateRawEvidence:
    """Bind all private trial hashes to the reviewed prompts and exact request bodies."""

    if not isinstance(raw, DiagnosticPrivateRawEvidence):
        raise TypeError("diagnostic trial validation requires validated private evidence")
    if not isinstance(bundle, InklingBF16InterfaceDiagnosticBundle):
        raise TypeError("diagnostic trial validation requires a validated bundle")

    items = {item.item_id: item for item in bundle.items}
    cells = {cell.name: cell for cell in bundle.config.protocol.cells}
    instruction = bundle.config.protocol.raw_instruction
    for trial in raw.trials:
        item = items[trial.item_id]
        cell = cells[trial.cell]
        item_prompt_sha256 = hashlib.sha256(item.prompt.encode("utf-8")).hexdigest()
        if trial.item_prompt_sha256 != item_prompt_sha256:
            raise ValueError(f"diagnostic item prompt hash differs for request {trial.ordinal}")

        if cell.prompt_mode == "raw":
            rendered_prompt = instruction + "\n" + item.prompt
        else:
            rendered_prompt = (
                f"<|message_system|><|content_text|>{instruction}<|end_message|>"
                "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
                f"<|message_user|><|content_text|>{item.prompt}<|end_message|>"
                "<|message_model|>"
            )
        rendered_prompt_sha256 = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
        if trial.rendered_prompt_sha256 != rendered_prompt_sha256:
            raise ValueError(f"diagnostic rendered prompt hash differs for request {trial.ordinal}")

        request = {
            "prompt": list(trial.input_token_ids),
            "n_predict": trial.requested_max_new_tokens,
            "temperature": -1.0,
            "seed": 42,
            "stream": False,
            "cache_prompt": False,
            "return_tokens": True,
            "timings_per_token": True,
            "ignore_eos": False,
            "stop": [],
        }
        request_sha256 = hashlib.sha256(canonical_diagnostic_json_bytes(request)).hexdigest()
        if trial.request_sha256 != request_sha256:
            raise ValueError(f"diagnostic request hash differs for request {trial.ordinal}")
    return raw


def diagnostic_private_raw_content_sha256(payload: bytes, *, run_id: str) -> str:
    parse_diagnostic_private_raw_evidence(payload, run_id=run_id)
    return hashlib.sha256(DIAGNOSTIC_RAW_HASH_DOMAIN + payload).hexdigest()


def diagnostic_private_raw_path(run_id: str, content_sha256: str) -> str:
    _validate_run_id(run_id)
    _validate_sha256(content_sha256, label="diagnostic private-raw content hash")
    return PurePosixPath(
        "runs",
        run_id,
        DIAGNOSTIC_STAGE,
        "private",
        "raw",
        f"{content_sha256}.json",
    ).as_posix()


class DiagnosticPrivateRawReference(_StrictDiagnosticModel):
    """Content-addressed identity for one immutable private raw evidence record."""

    schema_version: Literal["inkling-bf16-interface-private-raw-reference-v1"] = (
        "inkling-bf16-interface-private-raw-reference-v1"
    )
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    evidence_root: StrictStr
    relative_path: StrictStr
    absolute_path: StrictStr
    content_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    size_bytes: StrictInt = Field(gt=0, le=DIAGNOSTIC_RAW_RECORD_MAX_BYTES)

    @field_validator("evidence_root", "absolute_path")
    @classmethod
    def absolute_paths_are_canonical(cls, value: str) -> str:
        return validate_absolute_path(value)

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_canonical(cls, value: str) -> str:
        return validate_repository_relative_path(value)

    @model_validator(mode="after")
    def paths_are_content_addressed(self) -> DiagnosticPrivateRawReference:
        expected = diagnostic_private_raw_path(self.run_id, self.content_sha256)
        if self.relative_path != expected:
            raise ValueError("diagnostic private-raw path is not content addressed")
        if self.absolute_path != diagnostic_absolute_evidence_path(self.evidence_root, expected):
            raise ValueError("diagnostic private-raw absolute path has the wrong root")
        return self


def build_diagnostic_private_raw_reference(
    payload: bytes,
    *,
    evidence_root: str,
    run_id: str,
) -> DiagnosticPrivateRawReference:
    content_sha256 = diagnostic_private_raw_content_sha256(payload, run_id=run_id)
    relative_path = diagnostic_private_raw_path(run_id, content_sha256)
    return DiagnosticPrivateRawReference(
        run_id=run_id,
        evidence_root=evidence_root,
        relative_path=relative_path,
        absolute_path=diagnostic_absolute_evidence_path(evidence_root, relative_path),
        content_sha256=content_sha256,
        size_bytes=len(payload),
    )


def validate_diagnostic_private_raw_reference(
    payload: bytes,
    *,
    expected: DiagnosticPrivateRawReference,
) -> DiagnosticPrivateRawReference:
    observed = build_diagnostic_private_raw_reference(
        payload,
        evidence_root=expected.evidence_root,
        run_id=expected.run_id,
    )
    if observed != expected:
        raise ValueError("diagnostic private-raw reference differs from exact bytes")
    return observed


def diagnostic_runtime_eog_set_sha256(token_ids: Sequence[int]) -> str:
    """Hash a sorted runtime EOG set without publishing its reversible token IDs."""

    exact = tuple(token_ids)
    if not exact or exact != tuple(sorted(set(exact))) or any(token_id < 0 for token_id in exact):
        raise ValueError("runtime EOG token IDs must be nonempty, sorted, and unique")
    return _domain_hash(
        DIAGNOSTIC_ROLLUP_HASH_DOMAIN,
        {"kind": "runtime_eog_set", "token_ids": exact},
    )


class DiagnosticCellRollup(_StrictDiagnosticModel):
    """Compact hash/count-only result for one four-request interface cell."""

    cell: DiagnosticCellName
    request_count: Literal[4]
    trial_sha256s: tuple[StrictStr, StrictStr, StrictStr, StrictStr]
    whole_output_pass_count: StrictInt = Field(ge=0, le=4)
    extracted_content_pass_count: StrictInt = Field(ge=0, le=4)
    stop_count: StrictInt = Field(ge=0, le=4)
    eog_observed_count: StrictInt = Field(ge=0, le=4)
    cap_hit_count: StrictInt = Field(ge=0, le=4)
    truncated_count: StrictInt = Field(ge=0, le=4)

    @field_validator("trial_sha256s")
    @classmethod
    def trial_hashes_are_sha256(cls, value: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
        for digest in value:
            _validate_sha256(digest, label="diagnostic private-trial hash")
        return value


class DiagnosticRollup(_StrictDiagnosticModel):
    """Public diagnostic summary containing only hashes, booleans, and counts."""

    schema_version: Literal["inkling-bf16-interface-rollup-v1"] = "inkling-bf16-interface-rollup-v1"
    private_raw_content_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    request_count: Literal[16]
    cell_rollups: tuple[
        DiagnosticCellRollup,
        DiagnosticCellRollup,
        DiagnosticCellRollup,
        DiagnosticCellRollup,
    ]
    whole_output_pass_count: StrictInt = Field(ge=0, le=16)
    extracted_content_pass_count: StrictInt = Field(ge=0, le=16)
    runtime_eog_set_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    runtime_eog_token_count: StrictInt = Field(gt=0)
    source_eos_runtime_is_eog: StrictBool
    comparison_token_runtime_is_eog: StrictBool
    source_eos_forced_token_observed: StrictBool
    comparison_forced_token_observed: StrictBool
    gpu_placement_verified: Literal[True]
    cpu_fallback_observed: Literal[False]
    diagnostic_only: Literal[True]
    quality_retention_claim_allowed: Literal[False]
    quality_claim_allowed: Literal[False]
    speedup_claim_allowed: Literal[False]
    performance_claim_allowed: Literal[False]
    mtp_included: Literal[False]
    mtp_supported: Literal[False]
    routing_drift_supported: Literal[False]
    single_run_causation_claim_allowed: Literal[False]

    @model_validator(mode="after")
    def totals_and_cells_are_exact(self) -> DiagnosticRollup:
        if tuple(item.cell for item in self.cell_rollups) != DIAGNOSTIC_CELL_ORDER:
            raise ValueError("diagnostic rollup cells differ from the checked order")
        if self.whole_output_pass_count != sum(
            item.whole_output_pass_count for item in self.cell_rollups
        ):
            raise ValueError("whole-output pass total differs from cell rollups")
        if self.extracted_content_pass_count != sum(
            item.extracted_content_pass_count for item in self.cell_rollups
        ):
            raise ValueError("extracted-content pass total differs from cell rollups")
        return self


def build_diagnostic_rollup(
    raw: DiagnosticPrivateRawEvidence,
    *,
    private_raw_content_sha256: str,
) -> DiagnosticRollup:
    """Derive the only public compact result from validated private evidence."""

    if not isinstance(raw, DiagnosticPrivateRawEvidence):
        raise TypeError("diagnostic rollup requires validated private raw evidence")
    _validate_sha256(private_raw_content_sha256, label="diagnostic private-raw content hash")
    cells: list[DiagnosticCellRollup] = []
    for cell_name in DIAGNOSTIC_CELL_ORDER:
        trials = tuple(trial for trial in raw.trials if trial.cell == cell_name)
        if len(trials) != 4:
            raise ValueError("diagnostic raw evidence has an incomplete interface cell")
        cells.append(
            DiagnosticCellRollup(
                cell=cell_name,
                request_count=4,
                trial_sha256s=cast(
                    tuple[StrictStr, StrictStr, StrictStr, StrictStr],
                    tuple(trial.private_sha256() for trial in trials),
                ),
                whole_output_pass_count=sum(trial.whole_output_passed for trial in trials),
                extracted_content_pass_count=sum(
                    trial.extracted_content_passed for trial in trials
                ),
                stop_count=sum(trial.stop for trial in trials),
                eog_observed_count=sum(trial.eog_observed for trial in trials),
                cap_hit_count=sum(trial.cap_hit for trial in trials),
                truncated_count=sum(trial.truncated for trial in trials),
            )
        )
    eog = raw.eog
    return DiagnosticRollup(
        private_raw_content_sha256=private_raw_content_sha256,
        request_count=DIAGNOSTIC_REQUEST_COUNT,
        cell_rollups=cast(
            tuple[
                DiagnosticCellRollup,
                DiagnosticCellRollup,
                DiagnosticCellRollup,
                DiagnosticCellRollup,
            ],
            tuple(cells),
        ),
        whole_output_pass_count=sum(item.whole_output_pass_count for item in cells),
        extracted_content_pass_count=sum(item.extracted_content_pass_count for item in cells),
        runtime_eog_set_sha256=diagnostic_runtime_eog_set_sha256(eog.runtime_eog_token_ids),
        runtime_eog_token_count=len(eog.runtime_eog_token_ids),
        source_eos_runtime_is_eog=eog.source_eos_probe.runtime_is_eog,
        comparison_token_runtime_is_eog=eog.comparison_token_probe.runtime_is_eog,
        source_eos_forced_token_observed=eog.source_eos_probe.forced_token_observed,
        comparison_forced_token_observed=eog.comparison_token_probe.forced_token_observed,
        gpu_placement_verified=True,
        cpu_fallback_observed=False,
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


def diagnostic_rollup_sha256(rollup: DiagnosticRollup) -> str:
    if not isinstance(rollup, DiagnosticRollup):
        raise TypeError("diagnostic rollup hash requires a validated rollup")
    return _domain_hash(DIAGNOSTIC_ROLLUP_HASH_DOMAIN, rollup.model_dump(mode="json"))


class DiagnosticTerminalBindings(_StrictDiagnosticModel):
    """Bindings and fail-closed claim flags common to all terminal receipts."""

    stage: Literal["bf16_interface_diagnostic"] = DIAGNOSTIC_STAGE
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    reviewed_config_file_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    resolved_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    post_spawn_acceptance_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    call_id: StrictStr = Field(pattern=_MODAL_CALL_ID_PATTERN)
    attempt_claim_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    prompt_text_recorded: Literal[False] = False
    output_text_recorded: Literal[False] = False
    compact_receipt_token_ids: Literal[False] = False
    diagnostic_only: Literal[True] = True
    quality_retention_claim_allowed: Literal[False] = False
    quality_claim_allowed: Literal[False] = False
    speedup_claim_allowed: Literal[False] = False
    performance_claim_allowed: Literal[False] = False
    mtp_included: Literal[False] = False
    mtp_supported: Literal[False] = False
    routing_drift_supported: Literal[False] = False
    single_run_causation_claim_allowed: Literal[False] = False
    scope_warning: Literal[
        "This is a BF16 prompt-interface diagnostic, not a quality-retention or "
        "performance comparison. Read the machine-readable record before use and "
        "do not apply it to a different model, runtime, hardware, or protocol."
    ] = DIAGNOSTIC_SCOPE_WARNING
    completed_at_utc: StrictStr

    @field_validator("completed_at_utc")
    @classmethod
    def completion_time_is_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="diagnostic terminal completion time")


class DiagnosticSuccessTerminalReceipt(DiagnosticTerminalBindings):
    """Compact terminal success receipt for the BF16 interface diagnostic."""

    schema_version: Literal["inkling-bf16-interface-success-v1"] = (
        "inkling-bf16-interface-success-v1"
    )
    status: Literal["completed"] = "completed"
    diagnostic_completed: Literal[True] = True
    completed_stages: tuple[DiagnosticStageName, ...]
    model_id: Literal["thinkingmachines/Inkling"]
    model_revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"]
    architecture: Literal["InklingForConditionalGeneration"]
    bf16_inventory_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    bf16_shard_count: Literal[49]
    bf16_total_bytes: Literal[1894278547552]
    protocol_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    workload_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    runtime_identity: MeasurementRuntimeIdentity
    runtime_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    runtime_manifest_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    hardware_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    command_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    server_log_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    server_log_size_bytes: StrictInt = Field(gt=0)
    private_raw_reference: DiagnosticPrivateRawReference
    rollup: DiagnosticRollup
    rollup_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    gpu_placement_verified: Literal[True]
    cpu_fallback_observed: Literal[False]

    @model_validator(mode="after")
    def successful_result_is_complete(self) -> DiagnosticSuccessTerminalReceipt:
        if self.completed_stages != DIAGNOSTIC_PLANNED_STAGES:
            raise ValueError("successful diagnostic must complete every checked stage")
        if self.private_raw_reference.run_id != self.run_id:
            raise ValueError("diagnostic private raw evidence belongs to another run")
        if self.private_raw_reference.content_sha256 != self.rollup.private_raw_content_sha256:
            raise ValueError("diagnostic rollup differs from private raw evidence")
        if self.rollup_sha256 != diagnostic_rollup_sha256(self.rollup):
            raise ValueError("diagnostic terminal rollup hash differs from its result")
        if self.gpu_placement_verified != self.rollup.gpu_placement_verified:
            raise ValueError("diagnostic placement result differs from its rollup")
        if self.runtime_identity_sha256 != diagnostic_runtime_identity_sha256(
            self.runtime_identity
        ):
            raise ValueError("diagnostic terminal runtime hash differs from its identity")
        if self.runtime_manifest_sha256 != self.runtime_identity.manifest_sha256:
            raise ValueError("diagnostic terminal runtime manifest differs from its identity")
        return self


class DiagnosticFailureTerminalReceipt(DiagnosticTerminalBindings):
    """Fail-closed compact terminal receipt for an incomplete diagnostic attempt."""

    schema_version: Literal["inkling-bf16-interface-failure-v1"] = (
        "inkling-bf16-interface-failure-v1"
    )
    status: Literal["failed"] = "failed"
    diagnostic_completed: Literal[False] = False
    completed_stages: tuple[DiagnosticStageName, ...]
    failed_stage: DiagnosticStageName
    error_code: StrictStr = Field(pattern=_SAFE_ERROR_CODE_PATTERN)
    error_summary_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    runtime_identity_sha256: StrictStr | None = Field(default=None, pattern=_SHA256_PATTERN)
    runtime_manifest_sha256: StrictStr | None = Field(default=None, pattern=_SHA256_PATTERN)
    hardware_identity_sha256: StrictStr | None = Field(default=None, pattern=_SHA256_PATTERN)
    private_raw_reference: DiagnosticPrivateRawReference | None = None
    gpu_placement_verified: StrictBool
    cpu_fallback_observed: StrictBool

    @model_validator(mode="after")
    def failed_result_is_checked_prefix(self) -> DiagnosticFailureTerminalReceipt:
        completed_count = len(self.completed_stages)
        if completed_count >= len(DIAGNOSTIC_PLANNED_STAGES):
            raise ValueError("failure receipt cannot claim every diagnostic stage completed")
        if self.completed_stages != DIAGNOSTIC_PLANNED_STAGES[:completed_count]:
            raise ValueError("failure stages must be an exact checked prefix")
        if self.failed_stage != DIAGNOSTIC_PLANNED_STAGES[completed_count]:
            raise ValueError("failed diagnostic stage must follow the completed prefix")
        placement_stage_index = DIAGNOSTIC_PLANNED_STAGES.index("verify_gpu_placement")
        expected_placement = completed_count > placement_stage_index
        if self.gpu_placement_verified != expected_placement:
            raise ValueError("failure placement flag differs from the completed diagnostic stages")
        if self.private_raw_reference is not None and (
            self.private_raw_reference.run_id != self.run_id
        ):
            raise ValueError("failure private raw evidence belongs to another run")
        return self


DiagnosticTerminalReceipt: TypeAlias = (
    DiagnosticSuccessTerminalReceipt | DiagnosticFailureTerminalReceipt
)


def parse_diagnostic_terminal_receipt(
    payload: bytes,
    *,
    run_id: str,
    outcome: DiagnosticOutcome,
) -> DiagnosticTerminalReceipt:
    """Parse one canonical terminal receipt through its explicit outcome schema."""

    if not isinstance(payload, bytes):
        raise TypeError("diagnostic terminal receipt must be bytes")
    _validate_run_id(run_id)
    if outcome not in {"success", "failure"}:
        raise ValueError("diagnostic terminal outcome is invalid")
    model_type: type[DiagnosticSuccessTerminalReceipt | DiagnosticFailureTerminalReceipt] = (
        DiagnosticSuccessTerminalReceipt
        if outcome == "success"
        else DiagnosticFailureTerminalReceipt
    )
    observed = _canonical_json_model(payload, model_type)
    if observed.run_id != run_id:
        raise ValueError("diagnostic terminal receipt has the wrong run ID")
    return observed


def diagnostic_terminal_receipt_content_sha256(
    payload: bytes,
    *,
    run_id: str,
    outcome: DiagnosticOutcome,
) -> str:
    parse_diagnostic_terminal_receipt(payload, run_id=run_id, outcome=outcome)
    domain = (
        DIAGNOSTIC_SUCCESS_HASH_DOMAIN if outcome == "success" else DIAGNOSTIC_FAILURE_HASH_DOMAIN
    )
    return hashlib.sha256(domain + payload).hexdigest()


def diagnostic_terminal_receipt_path(
    run_id: str,
    *,
    outcome: DiagnosticOutcome,
    content_sha256: str,
) -> str:
    _validate_run_id(run_id)
    if outcome not in {"success", "failure"}:
        raise ValueError("diagnostic terminal outcome is invalid")
    _validate_sha256(content_sha256, label="diagnostic terminal content hash")
    return PurePosixPath(
        "runs",
        run_id,
        DIAGNOSTIC_STAGE,
        "terminal",
        outcome,
        f"{content_sha256}.json",
    ).as_posix()


class DiagnosticTerminalReceiptReference(_StrictDiagnosticModel):
    """Portable and mounted paths for one immutable compact terminal receipt."""

    schema_version: Literal["inkling-bf16-interface-terminal-reference-v1"] = (
        "inkling-bf16-interface-terminal-reference-v1"
    )
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    outcome: DiagnosticOutcome
    evidence_root: StrictStr
    relative_path: StrictStr
    absolute_path: StrictStr
    content_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    size_bytes: StrictInt = Field(gt=0, le=DIAGNOSTIC_CONTROL_RECORD_MAX_BYTES)

    @field_validator("evidence_root", "absolute_path")
    @classmethod
    def absolute_paths_are_canonical(cls, value: str) -> str:
        return validate_absolute_path(value)

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_canonical(cls, value: str) -> str:
        return validate_repository_relative_path(value)

    @model_validator(mode="after")
    def paths_are_content_addressed(self) -> DiagnosticTerminalReceiptReference:
        expected = diagnostic_terminal_receipt_path(
            self.run_id,
            outcome=self.outcome,
            content_sha256=self.content_sha256,
        )
        if self.relative_path != expected:
            raise ValueError("diagnostic terminal relative path is not content addressed")
        if self.absolute_path != diagnostic_absolute_evidence_path(self.evidence_root, expected):
            raise ValueError("diagnostic terminal absolute path has the wrong root")
        return self


def build_diagnostic_terminal_receipt_reference(
    payload: bytes,
    *,
    evidence_root: str,
    run_id: str,
    outcome: DiagnosticOutcome,
) -> DiagnosticTerminalReceiptReference:
    content_sha256 = diagnostic_terminal_receipt_content_sha256(
        payload,
        run_id=run_id,
        outcome=outcome,
    )
    relative_path = diagnostic_terminal_receipt_path(
        run_id,
        outcome=outcome,
        content_sha256=content_sha256,
    )
    return DiagnosticTerminalReceiptReference(
        run_id=run_id,
        outcome=outcome,
        evidence_root=evidence_root,
        relative_path=relative_path,
        absolute_path=diagnostic_absolute_evidence_path(evidence_root, relative_path),
        content_sha256=content_sha256,
        size_bytes=len(payload),
    )


def validate_diagnostic_terminal_receipt_reference(
    payload: bytes,
    *,
    expected: DiagnosticTerminalReceiptReference,
) -> DiagnosticTerminalReceiptReference:
    observed = build_diagnostic_terminal_receipt_reference(
        payload,
        evidence_root=expected.evidence_root,
        run_id=expected.run_id,
        outcome=expected.outcome,
    )
    if observed != expected:
        raise ValueError("diagnostic terminal reference differs from exact receipt bytes")
    return observed
