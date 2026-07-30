"""Pure control-plane contracts for matched Inkling measurement on Modal.

This module authorizes one sequential BF16-then-Q3 measurement attempt. It
does not import Modal, start compute, load a model, or execute llama.cpp.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Any, Final, Literal, Protocol, TypeAlias

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

from inkling_quant_lab.config import StrictFrozenModel

MEASUREMENT_STAGE: Final = "matched_measurement"
MEASUREMENT_ENVIRONMENT_NAME: Final = "inkling-quant"
MEASUREMENT_FUNCTION_NAME: Final = "run_measurement"
MEASUREMENT_ATTEMPT_REGISTRY_NAME: Final = "inkling-measurement-attempt-registry-v1"
MEASUREMENT_EVIDENCE_VOLUME_NAME: Final = "inkling-measurement-evidence-v1"

MEASUREMENT_DEPLOY_CONFIRMATION_PREFIX: Final = "CONFIRM MEASUREMENT DEPLOY"
MEASUREMENT_LAUNCH_CONFIRMATION_PREFIX: Final = "CONFIRM MEASUREMENT LAUNCH"
MEASUREMENT_CONTROL_RECORD_MAX_BYTES: Final = 512 * 1024
MEASUREMENT_DEPLOY_CHALLENGE_MAX_AGE_SECONDS: Final = 15 * 60
MEASUREMENT_LAUNCH_CHALLENGE_MAX_AGE_SECONDS: Final = 15 * 60

MEASUREMENT_CONTROL_PLANE_HASH_DOMAIN: Final = b"inkling-measurement-control-plane-v1\0"
MEASUREMENT_DEPLOY_CHALLENGE_HASH_DOMAIN: Final = b"inkling-measurement-deploy-challenge-v1\0"
MEASUREMENT_LAUNCH_CHALLENGE_HASH_DOMAIN: Final = b"inkling-measurement-launch-challenge-v1\0"
MEASUREMENT_LAUNCH_INTENT_HASH_DOMAIN: Final = b"inkling-measurement-launch-intent-v1\0"
MEASUREMENT_POST_SPAWN_ACCEPTANCE_HASH_DOMAIN: Final = (
    b"inkling-measurement-post-spawn-acceptance-v1\0"
)
MEASUREMENT_ATTEMPT_CLAIM_HASH_DOMAIN: Final = b"inkling-measurement-attempt-claim-v1\0"
MEASUREMENT_SUCCESS_RECEIPT_HASH_DOMAIN: Final = (
    b"inkling-measurement-terminal-success-receipt-v1\0"
)
MEASUREMENT_FAILURE_RECEIPT_HASH_DOMAIN: Final = (
    b"inkling-measurement-terminal-failure-receipt-v1\0"
)
MEASUREMENT_RUNTIME_MANIFEST_HASH_DOMAIN: Final = b"inkling-measurement-runtime-manifest-v1\0"
MEASUREMENT_QUALITY_ROLLUP_HASH_DOMAIN: Final = b"inkling-measurement-quality-rollup-v1\0"
MEASUREMENT_PERFORMANCE_ROLLUP_HASH_DOMAIN: Final = b"inkling-measurement-performance-rollup-v1\0"

MEASUREMENT_SUBJECT_ORDER: Final = ("bf16", "q3")
MEASUREMENT_PLANNED_STAGES: Final = (
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
MEASUREMENT_QUALITY_SUITES: Final = (
    "text",
    "math",
    "code",
    "multilingual",
    "instruction",
    "vision",
    "audio",
    "post_training",
)
MEASUREMENT_BENCH_CASES: Final = ("pp512", "pp2048", "tg128")
MEASUREMENT_SERVER_CONCURRENCIES: Final = (1, 2, 4)
MEASUREMENT_PRINTED_PERPLEXITY_ABSOLUTE_TOLERANCE: Final = 0.0000501
MEASUREMENT_RUNTIME_COMMANDS: Final = (
    "llama-cli",
    "llama-server",
    "llama-bench",
    "llama-perplexity",
)
MEASUREMENT_SCOPE_WARNING: Final = (
    "Read the machine-readable record before use. Do not apply a result to a "
    "different model, dataset, runtime, software, hardware, or protocol."
)
MEASUREMENT_SERVER_PROMPT_SEGMENT: Final = "matched Inkling measurement input "
MEASUREMENT_SERVER_PROMPT_REPEAT_COUNT: Final = 2_048
MEASUREMENT_SERVER_PROMPT_TEMPLATE_PROTOCOL: Final = (
    "repeat_utf8_literal_2048_then_tokenize_without_special_tokens_then_take_first_512_ids"
)
MEASUREMENT_LLAMA_BENCH_PROMPT_TEMPLATE_PROTOCOL: Final = (
    "c_stdlib_rand_default_seed_1_without_srand_with_optional_bos_first_token"
)

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
_REVIEWED_ROLE_PATHS: Final = {
    "measurement_config": "configs/experiments/inkling_q3_k_m_measurement_modal.yaml",
    "diagnostic_dataset": "configs/experiments/inkling_quality_diagnostic_v1.jsonl",
    "corpus_reference": "configs/experiments/inkling_wikitext2_raw_test_reference.json",
    "corpus_materializer": "scripts/materialize_inkling_measurement_corpus.py",
    "bf16_subject_reference": "configs/experiments/inkling_bf16_subject_reference.json",
    "q3_verified_export_reference": ("configs/experiments/inkling_q3_k_m_verified_export.json"),
    "source_adoption_reference": "configs/experiments/inkling_q3_k_m_source_adoption.json",
}

MeasurementSubject: TypeAlias = Literal["bf16", "q3"]
MeasurementOutcome: TypeAlias = Literal["success", "failure"]
MeasurementStage: TypeAlias = Literal[
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
]
MeasurementQualitySuite: TypeAlias = Literal[
    "text",
    "math",
    "code",
    "multilingual",
    "instruction",
    "vision",
    "audio",
    "post_training",
]
MeasurementBenchCase: TypeAlias = Literal["pp512", "pp2048", "tg128"]
MeasurementRuntimeCommand: TypeAlias = Literal[
    "llama-cli",
    "llama-server",
    "llama-bench",
    "llama-perplexity",
]
MeasurementSupportingRecordKind: TypeAlias = Literal[
    "bf16_subject",
    "q3_subject",
    "comparison",
]


class _StrictControlModel(StrictFrozenModel):
    """Fail-closed base for immutable measurement control records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
    )


def canonical_measurement_json_bytes(value: object) -> bytes:
    """Return canonical UTF-8 JSON with exactly one trailing line feed."""

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


def measurement_server_prompt_source_text() -> str:
    """Return the exact synthetic text tokenized for every server request."""

    return MEASUREMENT_SERVER_PROMPT_SEGMENT * MEASUREMENT_SERVER_PROMPT_REPEAT_COUNT


def measurement_llama_bench_dataset_bytes() -> bytes:
    """Return the inspectable identity bytes for llama-bench's synthetic tokens."""

    return canonical_measurement_json_bytes(
        {
            "generator": (
                "c_stdlib_rand_default_seed_1_without_srand_with_optional_bos_first_token"
            ),
            "cases": [
                {
                    "sample_id": "pp512",
                    "prompt_tokens": 512,
                    "generation_tokens": 0,
                },
                {
                    "sample_id": "pp2048",
                    "prompt_tokens": 2_048,
                    "generation_tokens": 0,
                },
                {
                    "sample_id": "tg128",
                    "prompt_tokens": 0,
                    "generation_tokens": 128,
                },
            ],
        }
    )


class MeasurementLlamaBenchCaseIdentity(_StrictControlModel):
    """One ordered synthetic llama-bench workload case."""

    sample_id: Literal["pp512", "pp2048", "tg128"]
    prompt_tokens: Literal[0, 512, 2048]
    generation_tokens: Literal[0, 128]


class MeasurementLlamaBenchWorkloadIdentity(_StrictControlModel):
    """Inspectable identity for the pinned llama-bench synthetic workload."""

    schema_version: Literal["inkling-llama-bench-workload-v1"]
    dataset_id: Literal["llama.cpp/llama-bench-synthetic-token-workload"]
    dataset_revision: Literal["a015409e6c27b84f60d688823d4c0126a11571fd"]
    split: Literal["benchmark"]
    content_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    content_size_bytes: StrictInt = Field(gt=0)
    ordered_sample_ids: tuple[
        Literal["pp512"],
        Literal["pp2048"],
        Literal["tg128"],
    ]
    seed: Literal[1]
    seed_protocol: Literal["c_stdlib_rand_default_seed_1_without_srand"]
    prompt_template_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    prompt_template_protocol: Literal[
        "c_stdlib_rand_default_seed_1_without_srand_with_optional_bos_first_token"
    ]
    execution_mode: Literal["single_process_single_model_load_ordered_cases"]
    cases: tuple[
        MeasurementLlamaBenchCaseIdentity,
        MeasurementLlamaBenchCaseIdentity,
        MeasurementLlamaBenchCaseIdentity,
    ]

    @model_validator(mode="after")
    def exact_identity(self) -> MeasurementLlamaBenchWorkloadIdentity:
        content = measurement_llama_bench_dataset_bytes()
        if self.content_sha256 != hashlib.sha256(
            content
        ).hexdigest() or self.content_size_bytes != len(content):
            raise ValueError("llama-bench workload content identity differs from its generator")
        expected_cases = (
            ("pp512", 512, 0),
            ("pp2048", 2_048, 0),
            ("tg128", 0, 128),
        )
        observed_cases = tuple(
            (item.sample_id, item.prompt_tokens, item.generation_tokens) for item in self.cases
        )
        if (
            self.ordered_sample_ids != tuple(item[0] for item in expected_cases)
            or observed_cases != expected_cases
        ):
            raise ValueError("llama-bench workload cases differ from the exact order")
        prompt_template = (
            b"inkling-llama-bench-prompt-template-v1\0"
            b"c_stdlib_rand_default_seed_1_without_srand\0"
            b"first_token_optional_model_bos_else_rand_mod_vocab\0"
            b"remaining_tokens_rand_mod_vocab"
        )
        if self.prompt_template_sha256 != hashlib.sha256(prompt_template).hexdigest():
            raise ValueError("llama-bench prompt-template hash differs from its protocol")
        return self


class MeasurementServerWorkloadIdentity(_StrictControlModel):
    """Inspectable identity and decoding contract for server measurements."""

    schema_version: Literal["inkling-server-benchmark-workload-v1"]
    dataset_id: Literal["inkling-quant-lab/synthetic-server-benchmark"]
    dataset_revision: Literal["inkling-server-benchmark-v1"]
    split: Literal["benchmark"]
    content_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    content_size_bytes: StrictInt = Field(gt=0)
    ordered_sample_ids: tuple[Literal["server_prompt_0001"]]
    seed: Literal[42]
    prompt_template_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    prompt_template_protocol: Literal[
        "repeat_utf8_literal_2048_then_tokenize_without_special_tokens_then_take_first_512_ids"
    ]
    prompt_tokens: Literal[512]
    output_tokens: Literal[128]
    temperature: StrictFloat
    streaming: Literal[True]
    cache_prompt: Literal[False]
    return_tokens: Literal[True]
    ignore_eos: Literal[True]
    execution_mode: Literal["llama_server_streaming_completion_concurrency_1_2_4"]

    @model_validator(mode="after")
    def exact_identity(self) -> MeasurementServerWorkloadIdentity:
        content = measurement_server_prompt_source_text().encode("utf-8")
        if self.content_sha256 != hashlib.sha256(
            content
        ).hexdigest() or self.content_size_bytes != len(content):
            raise ValueError("server workload content identity differs from its source")
        if self.temperature != 0.0:
            raise ValueError("server workload temperature must be exactly 0.0")
        prompt_template = (
            b"inkling-server-prompt-template-v1\0"
            b"repeat_utf8_literal=matched Inkling measurement input \\x20\0"
            b"repeat_count=2048\0"
            b"tokenize_add_special=false\0"
            b"tokenize_parse_special=false\0"
            b"take_first_token_ids=512"
        )
        if self.prompt_template_sha256 != hashlib.sha256(prompt_template).hexdigest():
            raise ValueError("server prompt-template hash differs from its protocol")
        return self


def _domain_hash(domain: bytes, payload: bytes) -> str:
    return hashlib.sha256(domain + payload).hexdigest()


def _reject_duplicate_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key in measurement control JSON: {key}")
        result[key] = value
    return result


def _reject_non_finite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def strict_measurement_json_object(payload: bytes | str) -> dict[str, Any]:
    """Parse one bounded UTF-8 JSON object and reject ambiguous JSON."""

    if isinstance(payload, bytes):
        encoded = payload
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("measurement control JSON must be UTF-8") from error
    elif isinstance(payload, str):
        text = payload
        encoded = payload.encode("utf-8")
    else:
        raise TypeError("measurement control JSON must be bytes or text")
    if len(encoded) > MEASUREMENT_CONTROL_RECORD_MAX_BYTES:
        raise ValueError("measurement control JSON exceeds its size limit")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_object_pairs,
            parse_constant=_reject_non_finite_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("measurement control JSON is invalid") from error
    if not isinstance(value, dict):
        raise ValueError("measurement control JSON root must be an object")
    return value


def validate_repository_relative_path(value: str) -> str:
    """Require one canonical repository-relative POSIX path."""

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


def validate_absolute_evidence_path(value: str) -> str:
    """Require one non-root canonical absolute POSIX evidence path."""

    if (
        type(value) is not str
        or not value
        or "\\" in value
        or "\x00" in value
        or not value.startswith("/")
        or value.startswith("//")
    ):
        raise ValueError("evidence path must be a canonical absolute POSIX path")
    path = PurePosixPath(value)
    if (
        value == "/"
        or not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("evidence path must be a canonical absolute POSIX path")
    return value


def measurement_absolute_evidence_path(evidence_root: str, relative_path: str) -> str:
    """Join a canonical evidence root and repository-style relative path."""

    root = validate_absolute_evidence_path(evidence_root)
    relative = validate_repository_relative_path(relative_path)
    result = PurePosixPath(root, relative).as_posix()
    validate_absolute_evidence_path(result)
    if not PurePosixPath(result).is_relative_to(PurePosixPath(root)):
        raise ValueError("evidence path must remain below its evidence root")
    return result


def _canonical_utc(value: str, *, label: str) -> str:
    if re.fullmatch(_CANONICAL_UTC_PATTERN, value) is None:
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


def _validate_run_id(value: str) -> str:
    if re.fullmatch(_RUN_ID_PATTERN, value) is None:
        raise ValueError("measurement run ID is invalid")
    return value


def _validate_sha256(value: str, *, label: str) -> str:
    if re.fullmatch(_SHA256_PATTERN, value) is None:
        raise ValueError(f"{label} is invalid")
    return value


class MeasurementControlPlaneFile(_StrictControlModel):
    """One exact file in the reviewed and deployed implementation closure."""

    path: StrictStr
    size_bytes: StrictInt = Field(ge=0)
    sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        return validate_repository_relative_path(value)


def measurement_control_plane_sha256(
    *,
    reviewed_commit_sha: str,
    reviewed_tree_sha: str,
    files: Sequence[MeasurementControlPlaneFile],
) -> str:
    """Hash the reviewed Git identity and exact ordered file manifest."""

    if re.fullmatch(_GIT_OBJECT_PATTERN, reviewed_commit_sha) is None:
        raise ValueError("reviewed Git commit SHA is invalid")
    if re.fullmatch(_GIT_OBJECT_PATTERN, reviewed_tree_sha) is None:
        raise ValueError("reviewed Git tree SHA is invalid")
    paths = tuple(item.path for item in files)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise ValueError("measurement control-plane paths must be sorted and unique")
    payload = {
        "schema_version": "inkling-measurement-control-plane-v1",
        "reviewed_commit_sha": reviewed_commit_sha,
        "reviewed_tree_sha": reviewed_tree_sha,
        "files": [item.model_dump(mode="json") for item in files],
    }
    return _domain_hash(
        MEASUREMENT_CONTROL_PLANE_HASH_DOMAIN,
        canonical_measurement_json_bytes(payload),
    )


class MeasurementControlPlaneProvenance(_StrictControlModel):
    """Content-addressed reviewed Git and deployed file identity."""

    schema_version: Literal["inkling-measurement-control-plane-v1"] = (
        "inkling-measurement-control-plane-v1"
    )
    reviewed_commit_sha: StrictStr = Field(pattern=_GIT_OBJECT_PATTERN)
    reviewed_tree_sha: StrictStr = Field(pattern=_GIT_OBJECT_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    file_count: StrictInt = Field(gt=0)
    files: tuple[MeasurementControlPlaneFile, ...]

    @model_validator(mode="after")
    def manifest_is_exact(self) -> MeasurementControlPlaneProvenance:
        if self.file_count != len(self.files):
            raise ValueError("measurement control-plane file count differs from its manifest")
        expected = measurement_control_plane_sha256(
            reviewed_commit_sha=self.reviewed_commit_sha,
            reviewed_tree_sha=self.reviewed_tree_sha,
            files=self.files,
        )
        if self.control_plane_sha256 != expected:
            raise ValueError("measurement control-plane hash differs from its manifest")
        return self

    def canonical_bytes(self) -> bytes:
        """Return exact bytes for local-to-remote provenance comparison."""

        return canonical_measurement_json_bytes(self.model_dump(mode="json"))


def build_measurement_control_plane_provenance(
    *,
    reviewed_commit_sha: str,
    reviewed_tree_sha: str,
    files: Mapping[str, bytes],
    required_paths: Sequence[str],
) -> MeasurementControlPlaneProvenance:
    """Build provenance from an explicitly closed required file set."""

    required = tuple(required_paths)
    if len(required) != len(set(required)):
        raise ValueError("measurement control-plane required paths must be unique")
    for path in required:
        validate_repository_relative_path(path)
    observed_paths = tuple(files)
    for path in observed_paths:
        validate_repository_relative_path(path)
    if set(observed_paths) != set(required) or len(files) != len(required):
        raise ValueError("measurement control-plane files must equal the required path set")
    manifest: list[MeasurementControlPlaneFile] = []
    for path in sorted(required):
        payload = files[path]
        if not isinstance(payload, bytes):
            raise TypeError("measurement control-plane file payloads must be bytes")
        manifest.append(
            MeasurementControlPlaneFile(
                path=path,
                size_bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    control_plane_sha256 = measurement_control_plane_sha256(
        reviewed_commit_sha=reviewed_commit_sha,
        reviewed_tree_sha=reviewed_tree_sha,
        files=manifest,
    )
    return MeasurementControlPlaneProvenance(
        reviewed_commit_sha=reviewed_commit_sha,
        reviewed_tree_sha=reviewed_tree_sha,
        control_plane_sha256=control_plane_sha256,
        file_count=len(manifest),
        files=tuple(manifest),
    )


def validate_measurement_control_plane_provenance(
    provenance: MeasurementControlPlaneProvenance | Mapping[str, Any] | bytes,
    *,
    reviewed_commit_sha: str,
    reviewed_tree_sha: str,
    files: Mapping[str, bytes],
    required_paths: Sequence[str],
) -> MeasurementControlPlaneProvenance:
    """Rebuild and compare provenance against exact local or mounted bytes."""

    if isinstance(provenance, bytes):
        strict_measurement_json_object(provenance)
        try:
            observed = MeasurementControlPlaneProvenance.model_validate_json(
                provenance,
                strict=True,
            )
        except ValidationError as error:
            raise ValueError("measurement control-plane provenance schema is invalid") from error
        if provenance != observed.canonical_bytes():
            raise ValueError("measurement control-plane provenance bytes are not canonical")
    elif isinstance(provenance, MeasurementControlPlaneProvenance):
        observed = provenance
    elif isinstance(provenance, Mapping):
        try:
            observed = MeasurementControlPlaneProvenance.model_validate(provenance)
        except ValidationError as error:
            raise ValueError("measurement control-plane provenance schema is invalid") from error
    else:
        raise TypeError("measurement control-plane provenance has an unsupported type")
    expected = build_measurement_control_plane_provenance(
        reviewed_commit_sha=reviewed_commit_sha,
        reviewed_tree_sha=reviewed_tree_sha,
        files=files,
        required_paths=required_paths,
    )
    if observed != expected:
        raise ValueError("measurement control-plane provenance differs from deployed bytes")
    return observed


class MeasurementExecutionResources(_StrictControlModel):
    """Exact Modal B300:8 resource cell authorized for measurement."""

    provider: Literal["modal"] = "modal"
    gpu_type: Literal["B300"] = "B300"
    gpu_count: Literal[8] = 8
    compute_capability: Literal["10.3"] = "10.3"
    cpu_cores: Literal[16] = 16
    memory_mib: Literal[65536] = 65_536
    ephemeral_disk_mib: Literal[2097152] = 2_097_152
    startup_timeout_seconds: Literal[1800] = 1_800
    function_timeout_seconds: Literal[86400] = 86_400
    max_containers: Literal[1] = 1
    max_attempts: Literal[1] = 1
    network_access: Literal[False] = False
    cpu_fallback_allowed: Literal[False] = False


class MeasurementReviewedInputs(_StrictControlModel):
    """Exact reviewed code, protocol, subjects, and data authorized to run."""

    schema_version: Literal["inkling-measurement-reviewed-inputs-v2"] = (
        "inkling-measurement-reviewed-inputs-v2"
    )
    control_plane: MeasurementControlPlaneProvenance
    measurement_config: MeasurementControlPlaneFile
    resolved_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    diagnostic_dataset: MeasurementControlPlaneFile
    corpus_reference: MeasurementControlPlaneFile
    corpus_materializer: MeasurementControlPlaneFile
    bf16_subject_reference: MeasurementControlPlaneFile
    q3_verified_export_reference: MeasurementControlPlaneFile
    source_adoption_reference: MeasurementControlPlaneFile
    subject_order: tuple[MeasurementSubject, ...] = ("bf16", "q3")
    resources: MeasurementExecutionResources = Field(default_factory=MeasurementExecutionResources)

    @model_validator(mode="after")
    def bindings_are_in_the_reviewed_closure(self) -> MeasurementReviewedInputs:
        if self.subject_order != ("bf16", "q3"):
            raise ValueError("measurement subject order must be exactly BF16 then Q3")
        bound_by_role = {role: getattr(self, role) for role in _REVIEWED_ROLE_PATHS}
        for role, expected_path in _REVIEWED_ROLE_PATHS.items():
            if bound_by_role[role].path != expected_path:
                raise ValueError(f"measurement reviewed role {role} has the wrong file path")
        bound = tuple(bound_by_role.values())
        paths = tuple(item.path for item in bound)
        if len(paths) != len(set(paths)):
            raise ValueError("measurement reviewed input roles must bind distinct files")
        manifest = {item.path: item for item in self.control_plane.files}
        if any(manifest.get(item.path) != item for item in bound):
            raise ValueError("measurement reviewed input differs from the reviewed file closure")
        return self


def measurement_app_name(control_plane_sha256: str) -> str:
    """Derive the Modal App name from the complete control identity."""

    _validate_sha256(control_plane_sha256, label="measurement control-plane SHA-256")
    return f"inkling-measurement-{control_plane_sha256[:12]}"


def measurement_deployment_tag(control_plane_sha256: str) -> str:
    """Derive the deployment tag while retaining the full hash in records."""

    _validate_sha256(control_plane_sha256, label="measurement control-plane SHA-256")
    return f"iql-measurement-{control_plane_sha256[:34]}"


class MeasurementDeployConfirmationChallenge(_StrictControlModel):
    """Operator challenge that authorizes deployment but not GPU execution."""

    schema_version: Literal["inkling-measurement-deploy-confirmation-v1"] = (
        "inkling-measurement-deploy-confirmation-v1"
    )
    status: Literal["prepared_before_deploy"] = "prepared_before_deploy"
    created_at_utc: StrictStr
    expires_at_utc: StrictStr
    confirmation_nonce: StrictStr = Field(pattern=_SHA256_PATTERN)
    reviewed_inputs: MeasurementReviewedInputs
    app_name: StrictStr = Field(pattern=r"^inkling-measurement-[0-9a-f]{12}$")
    environment_name: Literal["inkling-quant"] = "inkling-quant"
    function_name: Literal["run_measurement"] = "run_measurement"
    attempt_registry_name: Literal["inkling-measurement-attempt-registry-v1"] = (
        "inkling-measurement-attempt-registry-v1"
    )
    evidence_volume_name: Literal["inkling-measurement-evidence-v1"] = (
        "inkling-measurement-evidence-v1"
    )
    starts_gpu_compute: Literal[False] = False

    @field_validator("created_at_utc", "expires_at_utc")
    @classmethod
    def times_are_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="deploy challenge time")

    @model_validator(mode="after")
    def app_name_and_expiration_are_exact(self) -> MeasurementDeployConfirmationChallenge:
        expected = measurement_app_name(self.reviewed_inputs.control_plane.control_plane_sha256)
        if self.app_name != expected:
            raise ValueError("measurement deploy challenge has the wrong App name")
        created_at = _utc_datetime(
            self.created_at_utc,
            label="deploy challenge creation time",
        )
        expires_at = _utc_datetime(
            self.expires_at_utc,
            label="deploy challenge expiration time",
        )
        if not created_at < expires_at:
            raise ValueError("measurement deploy challenge must expire after creation")
        if expires_at - created_at > timedelta(
            seconds=MEASUREMENT_DEPLOY_CHALLENGE_MAX_AGE_SECONDS
        ):
            raise ValueError("measurement deploy challenge lifetime exceeds its maximum")
        return self

    def canonical_bytes(self) -> bytes:
        """Return exact challenge bytes."""

        return canonical_measurement_json_bytes(self.model_dump(mode="json"))

    def challenge_sha256(self) -> str:
        """Return the domain-separated deploy challenge digest."""

        return _domain_hash(
            MEASUREMENT_DEPLOY_CHALLENGE_HASH_DOMAIN,
            self.canonical_bytes(),
        )

    def confirmation_text(self) -> str:
        """Return the only accepted deployment confirmation."""

        return f"{MEASUREMENT_DEPLOY_CONFIRMATION_PREFIX}\n{self.challenge_sha256()}"

    def confirm(self, value: str) -> MeasurementDeployConfirmationChallenge:
        """Validate the exact deployment confirmation."""

        if type(value) is not str or value != self.confirmation_text():
            raise ValueError("measurement deploy confirmation does not match its challenge")
        return self


def validate_measurement_deploy_challenge_not_expired(
    challenge: MeasurementDeployConfirmationChallenge,
    *,
    observed_at_utc: str,
) -> MeasurementDeployConfirmationChallenge:
    """Fail when deploy authorization is observed outside its bounded lifetime."""

    observed_at = _utc_datetime(observed_at_utc, label="deploy challenge observation time")
    created_at = _utc_datetime(
        challenge.created_at_utc,
        label="deploy challenge creation time",
    )
    expires_at = _utc_datetime(
        challenge.expires_at_utc,
        label="deploy challenge expiration time",
    )
    if not created_at <= observed_at < expires_at:
        raise ValueError("measurement deploy challenge is not active at the observed time")
    return challenge


class MeasurementDeploymentIdentity(_StrictControlModel):
    """Exact deployed objects permitted to receive measurement work."""

    schema_version: Literal["inkling-measurement-deployment-identity-v1"] = (
        "inkling-measurement-deployment-identity-v1"
    )
    deployed_at_utc: StrictStr
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    app_name: StrictStr = Field(pattern=r"^inkling-measurement-[0-9a-f]{12}$")
    environment_name: Literal["inkling-quant"] = "inkling-quant"
    deployment_version: StrictInt = Field(gt=0)
    deployment_tag: StrictStr = Field(pattern=r"^iql-measurement-[0-9a-f]{34}$")
    function_id: StrictStr = Field(pattern=_MODAL_FUNCTION_ID_PATTERN)
    function_name: Literal["run_measurement"] = "run_measurement"
    attempt_registry_name: Literal["inkling-measurement-attempt-registry-v1"] = (
        "inkling-measurement-attempt-registry-v1"
    )
    attempt_registry_id: StrictStr = Field(pattern=_MODAL_DICT_ID_PATTERN)
    attempt_registry_created_at_utc: StrictStr
    evidence_volume_name: Literal["inkling-measurement-evidence-v1"] = (
        "inkling-measurement-evidence-v1"
    )
    evidence_volume_id: StrictStr = Field(pattern=_MODAL_VOLUME_ID_PATTERN)

    @field_validator("deployed_at_utc", "attempt_registry_created_at_utc")
    @classmethod
    def times_are_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="measurement deployment time")

    @model_validator(mode="after")
    def identity_is_derived_and_ordered(self) -> MeasurementDeploymentIdentity:
        if self.app_name != measurement_app_name(self.control_plane_sha256):
            raise ValueError("measurement deployment App name differs from its control identity")
        if self.deployment_tag != measurement_deployment_tag(self.control_plane_sha256):
            raise ValueError("measurement deployment tag differs from its control identity")
        deployed_at = _utc_datetime(self.deployed_at_utc, label="deployment time")
        registry_at = _utc_datetime(
            self.attempt_registry_created_at_utc,
            label="attempt registry creation time",
        )
        if registry_at > deployed_at:
            raise ValueError("measurement deployment predates its attempt registry")
        return self

    def validate_reviewed_inputs(
        self,
        reviewed_inputs: MeasurementReviewedInputs,
    ) -> MeasurementDeploymentIdentity:
        """Require the deployment and reviewed file closure to agree."""

        if self.control_plane_sha256 != reviewed_inputs.control_plane.control_plane_sha256:
            raise ValueError("measurement deployment differs from reviewed control")
        return self

    def canonical_bytes(self) -> bytes:
        """Return exact stored deployment-identity bytes."""

        return canonical_measurement_json_bytes(self.model_dump(mode="json"))


class MeasurementLaunchConfirmationChallenge(_StrictControlModel):
    """Short-lived operator challenge shown immediately before paid launch."""

    schema_version: Literal["inkling-measurement-launch-confirmation-v1"] = (
        "inkling-measurement-launch-confirmation-v1"
    )
    status: Literal["prepared_before_launch"] = "prepared_before_launch"
    created_at_utc: StrictStr
    expires_at_utc: StrictStr
    authorization_nonce: StrictStr = Field(pattern=_SHA256_PATTERN)
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    reviewed_inputs: MeasurementReviewedInputs
    deployment: MeasurementDeploymentIdentity

    @field_validator("created_at_utc", "expires_at_utc")
    @classmethod
    def times_are_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="launch challenge time")

    @model_validator(mode="after")
    def identities_and_expiration_are_exact(
        self,
    ) -> MeasurementLaunchConfirmationChallenge:
        self.deployment.validate_reviewed_inputs(self.reviewed_inputs)
        created_at = _utc_datetime(
            self.created_at_utc,
            label="launch challenge creation time",
        )
        expires_at = _utc_datetime(
            self.expires_at_utc,
            label="launch challenge expiration time",
        )
        if not created_at < expires_at:
            raise ValueError("measurement launch challenge must expire after creation")
        if expires_at - created_at > timedelta(
            seconds=MEASUREMENT_LAUNCH_CHALLENGE_MAX_AGE_SECONDS
        ):
            raise ValueError("measurement launch challenge lifetime exceeds its maximum")
        deployed_at = _utc_datetime(self.deployment.deployed_at_utc, label="deployment time")
        if created_at < deployed_at:
            raise ValueError("measurement launch challenge predates its deployment")
        return self

    def canonical_bytes(self) -> bytes:
        """Return exact launch challenge bytes."""

        return canonical_measurement_json_bytes(self.model_dump(mode="json"))

    def challenge_sha256(self) -> str:
        """Return the domain-separated launch challenge digest."""

        return _domain_hash(
            MEASUREMENT_LAUNCH_CHALLENGE_HASH_DOMAIN,
            self.canonical_bytes(),
        )

    def confirmation_text(self) -> str:
        """Return the only accepted paid-launch confirmation."""

        return f"{MEASUREMENT_LAUNCH_CONFIRMATION_PREFIX}\n{self.challenge_sha256()}"

    def confirm(self, value: str) -> MeasurementLaunchConfirmationChallenge:
        """Validate the exact launch confirmation without starting compute."""

        if type(value) is not str or value != self.confirmation_text():
            raise ValueError("measurement launch confirmation does not match its challenge")
        return self


class MeasurementLaunchIntent(_StrictControlModel):
    """One immutable authorization for one BF16-then-Q3 remote attempt."""

    schema_version: Literal["inkling-measurement-launch-intent-v1"] = (
        "inkling-measurement-launch-intent-v1"
    )
    status: Literal["authorized_before_spawn"] = "authorized_before_spawn"
    authorization_scope: Literal["one_bf16_then_q3_measurement_attempt"] = (
        "one_bf16_then_q3_measurement_attempt"
    )
    authorized_at_utc: StrictStr
    expires_at_utc: StrictStr
    launch_challenge_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    authorization_nonce: StrictStr = Field(pattern=_SHA256_PATTERN)
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    reviewed_inputs: MeasurementReviewedInputs
    deployment: MeasurementDeploymentIdentity
    subject_order: tuple[MeasurementSubject, ...] = ("bf16", "q3")
    resources: MeasurementExecutionResources = Field(default_factory=MeasurementExecutionResources)
    one_atomic_attempt: Literal[True] = True
    sequential_same_allocation: Literal[True] = True
    fresh_process_per_measurement: Literal[True] = True
    rehash_all_subject_files: Literal[True] = True
    partial_success_allowed: Literal[False] = False
    measurement_execution_allowed: Literal[True] = True

    @field_validator("authorized_at_utc", "expires_at_utc")
    @classmethod
    def times_are_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="launch intent time")

    @model_validator(mode="after")
    def redundant_bindings_are_exact(self) -> MeasurementLaunchIntent:
        if self.subject_order != ("bf16", "q3"):
            raise ValueError("measurement launch must run BF16 then Q3")
        if self.subject_order != self.reviewed_inputs.subject_order:
            raise ValueError("measurement launch order differs from reviewed inputs")
        if self.resources != self.reviewed_inputs.resources:
            raise ValueError("measurement launch resources differ from reviewed inputs")
        self.deployment.validate_reviewed_inputs(self.reviewed_inputs)
        authorized_at = _utc_datetime(
            self.authorized_at_utc,
            label="launch authorization time",
        )
        expires_at = _utc_datetime(
            self.expires_at_utc,
            label="launch authorization expiration time",
        )
        if not authorized_at < expires_at:
            raise ValueError("measurement launch authorization is already expired")
        return self

    def canonical_bytes(self) -> bytes:
        """Return exact durable launch-intent bytes."""

        return canonical_measurement_json_bytes(self.model_dump(mode="json"))

    def intent_sha256(self) -> str:
        """Return the domain-separated launch-intent digest."""

        return _domain_hash(
            MEASUREMENT_LAUNCH_INTENT_HASH_DOMAIN,
            self.canonical_bytes(),
        )


def build_measurement_launch_intent(
    challenge: MeasurementLaunchConfirmationChallenge,
    *,
    confirmation: str,
    authorized_at_utc: str,
) -> MeasurementLaunchIntent:
    """Turn one exact, unexpired confirmation into remote authorization."""

    challenge.confirm(confirmation)
    authorized_at = _utc_datetime(
        authorized_at_utc,
        label="launch authorization time",
    )
    created_at = _utc_datetime(
        challenge.created_at_utc,
        label="launch challenge creation time",
    )
    expires_at = _utc_datetime(
        challenge.expires_at_utc,
        label="launch challenge expiration time",
    )
    if not created_at <= authorized_at < expires_at:
        raise ValueError(
            "measurement authorization must follow challenge creation and precede expiration"
        )
    return MeasurementLaunchIntent(
        authorized_at_utc=authorized_at_utc,
        expires_at_utc=challenge.expires_at_utc,
        launch_challenge_sha256=challenge.challenge_sha256(),
        authorization_nonce=challenge.authorization_nonce,
        run_id=challenge.run_id,
        reviewed_inputs=challenge.reviewed_inputs,
        deployment=challenge.deployment,
        subject_order=challenge.reviewed_inputs.subject_order,
        resources=challenge.reviewed_inputs.resources,
    )


def validate_measurement_launch_intent_not_expired(
    intent: MeasurementLaunchIntent,
    *,
    observed_at_utc: str,
) -> MeasurementLaunchIntent:
    """Fail when a runner observes an intent at or after its expiration."""

    observed_at = _utc_datetime(observed_at_utc, label="launch intent observation time")
    authorized_at = _utc_datetime(intent.authorized_at_utc, label="launch authorization time")
    expires_at = _utc_datetime(
        intent.expires_at_utc,
        label="launch authorization expiration time",
    )
    if not authorized_at <= observed_at < expires_at:
        raise ValueError("measurement launch intent is not active at the observed time")
    return intent


def measurement_launch_intent_path(run_id: str, intent_sha256: str) -> str:
    """Return the content-addressed relative launch-intent path."""

    _validate_run_id(run_id)
    _validate_sha256(intent_sha256, label="measurement launch-intent SHA-256")
    return PurePosixPath(
        "runs",
        run_id,
        "control",
        "launch-intents",
        f"{intent_sha256}.json",
    ).as_posix()


def validate_measurement_launch_intent(
    payload: bytes,
    *,
    expected: MeasurementLaunchIntent,
    intent_sha256: str,
    evidence_path: str,
) -> MeasurementLaunchIntent:
    """Validate canonical intent bytes, hash, path, and full expected value."""

    if not isinstance(payload, bytes):
        raise TypeError("measurement launch intent must be bytes")
    strict_measurement_json_object(payload)
    try:
        observed = MeasurementLaunchIntent.model_validate_json(payload, strict=True)
    except ValidationError as error:
        raise ValueError("measurement launch intent schema is invalid") from error
    if payload != observed.canonical_bytes():
        raise ValueError("measurement launch intent bytes are not canonical")
    if observed.intent_sha256() != intent_sha256:
        raise ValueError("measurement launch-intent hash differs from canonical bytes")
    relative = measurement_launch_intent_path(observed.run_id, intent_sha256)
    _validate_relative_evidence_path_matches(evidence_path, relative)
    if observed != expected:
        raise ValueError("measurement launch intent differs from the expected launch")
    return observed


class MeasurementPostSpawnAcceptance(_StrictControlModel):
    """Durable evidence that Modal accepted one remote function call."""

    schema_version: Literal["inkling-measurement-post-spawn-acceptance-v1"] = (
        "inkling-measurement-post-spawn-acceptance-v1"
    )
    status: Literal["accepted_after_spawn"] = "accepted_after_spawn"
    accepted_at_utc: StrictStr
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    call_id: StrictStr = Field(pattern=_MODAL_CALL_ID_PATTERN)
    deployment: MeasurementDeploymentIdentity
    reviewed_config_file_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    resolved_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("accepted_at_utc")
    @classmethod
    def accepted_at_is_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="post-spawn acceptance time")

    @model_validator(mode="after")
    def deployment_is_exact(self) -> MeasurementPostSpawnAcceptance:
        if self.deployment.control_plane_sha256 != self.control_plane_sha256:
            raise ValueError("measurement acceptance deployment differs from its control")
        return self

    def canonical_bytes(self) -> bytes:
        """Return exact durable acceptance bytes."""

        return canonical_measurement_json_bytes(self.model_dump(mode="json"))

    def acceptance_sha256(self) -> str:
        """Return the domain-separated acceptance digest."""

        return _domain_hash(
            MEASUREMENT_POST_SPAWN_ACCEPTANCE_HASH_DOMAIN,
            self.canonical_bytes(),
        )


def build_measurement_post_spawn_acceptance(
    intent: MeasurementLaunchIntent,
    *,
    accepted_at_utc: str,
    call_id: str,
) -> MeasurementPostSpawnAcceptance:
    """Bind provider acceptance to one still-active launch authorization."""

    validate_measurement_launch_intent_not_expired(
        intent,
        observed_at_utc=accepted_at_utc,
    )
    return MeasurementPostSpawnAcceptance(
        accepted_at_utc=accepted_at_utc,
        run_id=intent.run_id,
        launch_intent_sha256=intent.intent_sha256(),
        call_id=call_id,
        deployment=intent.deployment,
        reviewed_config_file_sha256=intent.reviewed_inputs.measurement_config.sha256,
        resolved_config_sha256=intent.reviewed_inputs.resolved_config_sha256,
        control_plane_sha256=intent.reviewed_inputs.control_plane.control_plane_sha256,
    )


def measurement_post_spawn_acceptance_path(
    run_id: str,
    launch_intent_sha256: str,
) -> str:
    """Return the single relative acceptance path for one launch intent."""

    _validate_run_id(run_id)
    _validate_sha256(
        launch_intent_sha256,
        label="measurement launch-intent SHA-256",
    )
    return PurePosixPath(
        "runs",
        run_id,
        "control",
        "post-spawn-acceptances",
        f"{launch_intent_sha256}.json",
    ).as_posix()


def validate_measurement_post_spawn_acceptance(
    payload: bytes,
    *,
    expected: MeasurementPostSpawnAcceptance,
    acceptance_sha256: str,
    evidence_path: str,
) -> MeasurementPostSpawnAcceptance:
    """Validate exact canonical post-spawn acceptance evidence."""

    if not isinstance(payload, bytes):
        raise TypeError("measurement post-spawn acceptance must be bytes")
    strict_measurement_json_object(payload)
    try:
        observed = MeasurementPostSpawnAcceptance.model_validate_json(payload, strict=True)
    except ValidationError as error:
        raise ValueError("measurement post-spawn acceptance schema is invalid") from error
    if payload != observed.canonical_bytes():
        raise ValueError("measurement post-spawn acceptance bytes are not canonical")
    if observed.acceptance_sha256() != acceptance_sha256:
        raise ValueError("measurement acceptance hash differs from canonical bytes")
    relative = measurement_post_spawn_acceptance_path(
        observed.run_id,
        observed.launch_intent_sha256,
    )
    _validate_relative_evidence_path_matches(evidence_path, relative)
    if observed != expected:
        raise ValueError("measurement acceptance differs from the expected acceptance")
    return observed


def measurement_attempt_registry_key(
    run_id: str,
    stage: str = MEASUREMENT_STAGE,
) -> str:
    """Return the only atomic registry key for a measurement run."""

    _validate_run_id(run_id)
    if stage != MEASUREMENT_STAGE:
        raise ValueError("measurement attempt stage is invalid")
    return f"{run_id}:{stage}"


class MeasurementAttemptClaim(_StrictControlModel):
    """One immutable claim of the launch intent's only remote attempt."""

    schema_version: Literal["inkling-measurement-attempt-claim-v1"] = (
        "inkling-measurement-attempt-claim-v1"
    )
    registry_name: Literal["inkling-measurement-attempt-registry-v1"] = (
        "inkling-measurement-attempt-registry-v1"
    )
    registry_id: StrictStr = Field(pattern=_MODAL_DICT_ID_PATTERN)
    registry_created_at_utc: StrictStr
    claimed_at_utc: StrictStr
    registry_key: StrictStr
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    stage: Literal["matched_measurement"] = "matched_measurement"
    call_id: StrictStr = Field(pattern=_MODAL_CALL_ID_PATTERN)
    input_id: StrictStr = Field(pattern=_MODAL_INPUT_ID_PATTERN)
    task_id: StrictStr = Field(pattern=_MODAL_TASK_ID_PATTERN)
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    post_spawn_acceptance_path: StrictStr
    post_spawn_acceptance_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    reviewed_config_file_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    resolved_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    subject_order: tuple[MeasurementSubject, ...] = ("bf16", "q3")

    @field_validator("registry_created_at_utc", "claimed_at_utc")
    @classmethod
    def registry_time_is_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="attempt registry creation time")

    @field_validator("post_spawn_acceptance_path")
    @classmethod
    def acceptance_path_is_canonical(cls, value: str) -> str:
        return validate_repository_relative_path(value)

    @model_validator(mode="after")
    def derived_bindings_are_exact(self) -> MeasurementAttemptClaim:
        if self.registry_key != measurement_attempt_registry_key(self.run_id, self.stage):
            raise ValueError("measurement attempt registry key differs from run and stage")
        if self.subject_order != ("bf16", "q3"):
            raise ValueError("measurement attempt must claim BF16 then Q3")
        expected = measurement_post_spawn_acceptance_path(
            self.run_id,
            self.launch_intent_sha256,
        )
        _validate_relative_evidence_path_matches(self.post_spawn_acceptance_path, expected)
        registry_created_at = _utc_datetime(
            self.registry_created_at_utc,
            label="attempt registry creation time",
        )
        claimed_at = _utc_datetime(self.claimed_at_utc, label="attempt claim time")
        if claimed_at < registry_created_at:
            raise ValueError("measurement attempt claim predates its registry")
        return self

    def canonical_bytes(self) -> bytes:
        """Return the exact bytes stored by the atomic claim."""

        return canonical_measurement_json_bytes(self.model_dump(mode="json"))

    def claim_sha256(self) -> str:
        """Return the domain-separated attempt-claim digest."""

        return _domain_hash(
            MEASUREMENT_ATTEMPT_CLAIM_HASH_DOMAIN,
            self.canonical_bytes(),
        )


class MeasurementAttemptRegistryProtocol(Protocol):
    """Minimal atomic operation required from the Modal Dict adapter."""

    def put(
        self,
        key: Any,
        value: Any,
        *,
        skip_if_exists: bool = False,
    ) -> bool: ...


def build_measurement_attempt_claim(
    intent: MeasurementLaunchIntent,
    acceptance: MeasurementPostSpawnAcceptance,
    *,
    claimed_at_utc: str,
    input_id: str,
    task_id: str,
) -> MeasurementAttemptClaim:
    """Bind one active launch and its accepted call to the atomic claim."""

    expected_acceptance = build_measurement_post_spawn_acceptance(
        intent,
        accepted_at_utc=acceptance.accepted_at_utc,
        call_id=acceptance.call_id,
    )
    if acceptance != expected_acceptance:
        raise ValueError("measurement acceptance differs from its launch intent")
    claimed_at = _utc_datetime(claimed_at_utc, label="attempt claim time")
    accepted_at = _utc_datetime(
        acceptance.accepted_at_utc,
        label="post-spawn acceptance time",
    )
    if claimed_at < accepted_at:
        raise ValueError("measurement attempt claim predates provider acceptance")
    deployment = intent.deployment
    return MeasurementAttemptClaim(
        registry_id=deployment.attempt_registry_id,
        registry_created_at_utc=deployment.attempt_registry_created_at_utc,
        claimed_at_utc=claimed_at_utc,
        registry_key=measurement_attempt_registry_key(intent.run_id),
        run_id=intent.run_id,
        call_id=acceptance.call_id,
        input_id=input_id,
        task_id=task_id,
        launch_intent_sha256=intent.intent_sha256(),
        post_spawn_acceptance_path=measurement_post_spawn_acceptance_path(
            intent.run_id,
            intent.intent_sha256(),
        ),
        post_spawn_acceptance_sha256=acceptance.acceptance_sha256(),
        reviewed_config_file_sha256=intent.reviewed_inputs.measurement_config.sha256,
        resolved_config_sha256=intent.reviewed_inputs.resolved_config_sha256,
        control_plane_sha256=intent.reviewed_inputs.control_plane.control_plane_sha256,
        subject_order=intent.subject_order,
    )


def claim_measurement_attempt(
    registry: MeasurementAttemptRegistryProtocol,
    claim: MeasurementAttemptClaim,
) -> str:
    """Atomically consume the one authorized remote measurement attempt."""

    created = registry.put(
        claim.registry_key,
        claim.canonical_bytes(),
        skip_if_exists=True,
    )
    if created is not True:
        raise RuntimeError("The one authorized measurement attempt has already been consumed")
    return claim.claim_sha256()


def measurement_attempt_claim_path(run_id: str, claim_sha256: str) -> str:
    """Return the content-addressed relative attempt-claim path."""

    _validate_run_id(run_id)
    _validate_sha256(claim_sha256, label="measurement attempt-claim SHA-256")
    return PurePosixPath(
        "runs",
        run_id,
        "control",
        "attempt-claims",
        f"{claim_sha256}.json",
    ).as_posix()


def validate_measurement_attempt_claim(
    payload: bytes,
    *,
    expected: MeasurementAttemptClaim,
    claim_sha256: str,
    evidence_path: str,
) -> MeasurementAttemptClaim:
    """Validate the exact claim stored in the Dict and evidence volume."""

    if not isinstance(payload, bytes):
        raise TypeError("measurement attempt claim must be bytes")
    strict_measurement_json_object(payload)
    try:
        observed = MeasurementAttemptClaim.model_validate_json(payload, strict=True)
    except ValidationError as error:
        raise ValueError("measurement attempt claim schema is invalid") from error
    if payload != observed.canonical_bytes():
        raise ValueError("measurement attempt claim bytes are not canonical")
    if observed.claim_sha256() != claim_sha256:
        raise ValueError("measurement attempt-claim hash differs from canonical bytes")
    relative = measurement_attempt_claim_path(observed.run_id, claim_sha256)
    _validate_relative_evidence_path_matches(evidence_path, relative)
    if observed != expected:
        raise ValueError("measurement attempt claim differs from the expected claim")
    return observed


def _validate_relative_evidence_path_matches(
    observed: str,
    expected_relative: str,
) -> None:
    """Require an exact canonical relative evidence path."""

    relative = validate_repository_relative_path(expected_relative)
    if validate_repository_relative_path(observed) != relative:
        raise ValueError("relative evidence path differs from its required path")


def _float_equal(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)


def _is_exact_unit_fraction(value: float, denominator: int) -> bool:
    scaled = value * denominator
    return _float_equal(scaled, float(round(scaled)))


class MeasurementAppliedPatch(_StrictControlModel):
    """One exact patch applied to the pinned llama.cpp tree."""

    path: StrictStr
    sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    size_bytes: StrictInt = Field(gt=0)

    @field_validator("path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        return validate_repository_relative_path(value)


class MeasurementPrePatchExecutable(_StrictControlModel):
    """Reviewed executable identity before the measurement patch is applied."""

    name: MeasurementRuntimeCommand
    path: StrictStr
    sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    size_bytes: StrictInt = Field(gt=0)

    @field_validator("path")
    @classmethod
    def path_is_absolute(cls, value: str) -> str:
        return validate_absolute_evidence_path(value)


class MeasurementRuntimeRegularFile(_StrictControlModel):
    """One regular file below the post-patch build/bin root."""

    relative_path: StrictStr
    sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    size_bytes: StrictInt = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        return validate_repository_relative_path(value)


class MeasurementRuntimeSymlink(_StrictControlModel):
    """One symlink and its fully resolved regular-file target in build/bin."""

    relative_path: StrictStr
    raw_target: StrictStr = Field(min_length=1, max_length=4096)
    resolved_relative_path: StrictStr

    @field_validator("relative_path", "resolved_relative_path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        return validate_repository_relative_path(value)

    @field_validator("raw_target")
    @classmethod
    def target_is_posix_text(cls, value: str) -> str:
        if "\\" in value or "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("runtime symlink target must be bounded POSIX text")
        return value


class MeasurementRuntimeDependency(_StrictControlModel):
    """One exact dynamic-loader dependency for a measured command."""

    classification: Literal["project_owned", "system", "cuda", "virtual"]
    soname: StrictStr = Field(min_length=1, max_length=512)
    resolved_path: StrictStr | None = None
    sha256: StrictStr | None = Field(default=None, pattern=_SHA256_PATTERN)
    size_bytes: StrictInt | None = Field(default=None, gt=0)

    @field_validator("soname")
    @classmethod
    def soname_is_safe_text(cls, value: str) -> str:
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError("runtime dependency name must be bounded single-line text")
        return value

    @field_validator("resolved_path")
    @classmethod
    def resolved_path_is_absolute(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_absolute_evidence_path(value)

    @model_validator(mode="after")
    def identity_matches_classification(self) -> MeasurementRuntimeDependency:
        identities = (self.resolved_path, self.sha256, self.size_bytes)
        if self.classification == "virtual":
            if identities != (None, None, None):
                raise ValueError("virtual runtime dependencies must not claim file identity")
            return self
        if any(value is None for value in identities):
            raise ValueError("file-backed runtime dependencies require complete identity")
        assert self.resolved_path is not None
        if self.classification == "cuda" and (
            "/stubs/" in self.resolved_path or self.resolved_path.endswith("/lib64/stubs")
        ):
            raise ValueError("CUDA runtime identity must not resolve through a stub library")
        return self


class MeasurementRuntimeCommandClosure(_StrictControlModel):
    """The measured command plus its exact post-patch DSO closure."""

    command: MeasurementRuntimeCommand
    binary_path: StrictStr
    binary_manifest_path: StrictStr
    binary_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    binary_size_bytes: StrictInt = Field(gt=0)
    ldd_output_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    dependencies: tuple[MeasurementRuntimeDependency, ...]

    @field_validator("binary_path")
    @classmethod
    def binary_path_is_absolute(cls, value: str) -> str:
        return validate_absolute_evidence_path(value)

    @field_validator("binary_manifest_path")
    @classmethod
    def manifest_path_is_canonical(cls, value: str) -> str:
        return validate_repository_relative_path(value)

    @model_validator(mode="after")
    def dependencies_are_sorted_and_complete(self) -> MeasurementRuntimeCommandClosure:
        expected_path = f"/opt/llama.cpp/build/bin/{self.command}"
        if self.binary_path != expected_path or self.binary_manifest_path != self.command:
            raise ValueError("runtime command path differs from its pinned build/bin path")
        if not self.dependencies:
            raise ValueError("runtime command dependency closure must not be empty")
        identities = tuple(
            (
                dependency.soname,
                "" if dependency.resolved_path is None else dependency.resolved_path,
            )
            for dependency in self.dependencies
        )
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(identities):
            raise ValueError("runtime command dependencies must be sorted and unique")
        return self


def measurement_runtime_manifest_sha256(
    *,
    build_bin_root: str,
    regular_files: Sequence[MeasurementRuntimeRegularFile],
    symlinks: Sequence[MeasurementRuntimeSymlink],
) -> str:
    """Hash the exact sorted post-patch build/bin regular-file and symlink inventory."""

    root = validate_absolute_evidence_path(build_bin_root)
    payload = canonical_measurement_json_bytes(
        {
            "build_bin_root": root,
            "regular_files": [item.model_dump(mode="json") for item in regular_files],
            "symlinks": [item.model_dump(mode="json") for item in symlinks],
        }
    )
    return _domain_hash(MEASUREMENT_RUNTIME_MANIFEST_HASH_DOMAIN, payload)


class MeasurementRuntimeIdentity(_StrictControlModel):
    """Complete post-both-patches command and shared-library identity."""

    schema_version: Literal["inkling-measurement-runtime-identity-v2"] = (
        "inkling-measurement-runtime-identity-v2"
    )
    repository: Literal["https://github.com/danielhanchen/llama.cpp.git"]
    repository_commit: Literal["a015409e6c27b84f60d688823d4c0126a11571fd"]
    cuda_image: Literal["nvidia/cuda:13.1.2-devel-ubuntu24.04"]
    cuda_image_digest: Literal[
        "sha256:952e42d23230610a2714c8484f38e9c934ed68e6f9c9c7fac62dcd5f98858a6e"
    ]
    platform: Literal["linux/amd64"]
    patches_applied_in_order: tuple[
        MeasurementAppliedPatch,
        MeasurementAppliedPatch,
    ]
    base_pre_measurement_patch_executables: tuple[
        MeasurementPrePatchExecutable,
        MeasurementPrePatchExecutable,
        MeasurementPrePatchExecutable,
        MeasurementPrePatchExecutable,
    ]
    cmake_generator: Literal["Ninja"]
    effective_cmake_definitions: tuple[StrictStr, ...]
    build_targets: tuple[
        Literal["llama-cli"],
        Literal["llama-server"],
        Literal["llama-bench"],
        Literal["llama-perplexity"],
    ]
    build_shared_libs: Literal[True]
    cmake_version: StrictStr = Field(min_length=1, max_length=512)
    cxx_compiler_version: StrictStr = Field(min_length=1, max_length=2048)
    cuda_compiler_version: StrictStr = Field(min_length=1, max_length=2048)
    build_bin_root: Literal["/opt/llama.cpp/build/bin"]
    regular_files: tuple[MeasurementRuntimeRegularFile, ...]
    symlinks: tuple[MeasurementRuntimeSymlink, ...]
    commands: tuple[
        MeasurementRuntimeCommandClosure,
        MeasurementRuntimeCommandClosure,
        MeasurementRuntimeCommandClosure,
        MeasurementRuntimeCommandClosure,
    ]
    manifest_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def exact_post_patch_runtime(self) -> MeasurementRuntimeIdentity:
        expected_patches = (
            (
                "patches/inkling-smoke-a015409.patch",
                "005f1f342511fc3fc843bdcc7be814ed8a60e67033b733eb7e7e4af53925be04",
                48409,
            ),
            (
                "patches/inkling-measurement-a015409.patch",
                "5896c17b3069e65bde15be276d02a41d6c83b443caf37e350ca24710f4a3a73f",
                5275,
            ),
        )
        observed_patches = tuple(
            (patch.path, patch.sha256, patch.size_bytes) for patch in self.patches_applied_in_order
        )
        if observed_patches != expected_patches:
            raise ValueError("runtime patches differ from the pinned application order")

        expected_pre_patch = (
            (
                "llama-cli",
                "/opt/llama.cpp/build/bin/llama-cli",
                "098d8b9c6e57f25b846c5b5b43ded5bb1194cbb3d1ce985f17bbd09c87a82dbc",
                1246680,
            ),
            (
                "llama-server",
                "/opt/llama.cpp/build/bin/llama-server",
                "e960cfe4dcb2f7e541fc0b15bf97a4c1f6feb5fc304267796ef2bdd004cd1b93",
                17920,
            ),
            (
                "llama-bench",
                "/opt/llama.cpp/build/bin/llama-bench",
                "e0844ac337c419ebd8b6cee4902ba13e210a067d6fe47cb652429c71ae97382b",
                17920,
            ),
            (
                "llama-perplexity",
                "/opt/llama.cpp/build/bin/llama-perplexity",
                "d04051888a157ee50a7d6286cffcc78da3a9ca5295c79aa99ea2d92672ebf733",
                15968,
            ),
        )
        observed_pre_patch = tuple(
            (item.name, item.path, item.sha256, item.size_bytes)
            for item in self.base_pre_measurement_patch_executables
        )
        if observed_pre_patch != expected_pre_patch:
            raise ValueError("pre-measurement executable identities differ from the reviewed base")

        expected_cmake = (
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
        if self.effective_cmake_definitions != expected_cmake:
            raise ValueError("effective CMake definitions differ from the measured build")
        if self.build_targets != MEASUREMENT_RUNTIME_COMMANDS:
            raise ValueError("runtime build targets differ from the measured commands")

        regular_paths = tuple(item.relative_path for item in self.regular_files)
        symlink_paths = tuple(item.relative_path for item in self.symlinks)
        if not regular_paths:
            raise ValueError("runtime regular-file inventory must not be empty")
        if regular_paths != tuple(sorted(regular_paths)) or len(set(regular_paths)) != len(
            regular_paths
        ):
            raise ValueError("runtime regular-file inventory must be sorted and unique")
        if symlink_paths != tuple(sorted(symlink_paths)) or len(set(symlink_paths)) != len(
            symlink_paths
        ):
            raise ValueError("runtime symlink inventory must be sorted and unique")
        if set(regular_paths) & set(symlink_paths):
            raise ValueError("runtime paths cannot be both regular files and symlinks")
        if any(link.resolved_relative_path not in set(regular_paths) for link in self.symlinks):
            raise ValueError("runtime symlink must resolve to an inventoried regular file")

        if tuple(command.command for command in self.commands) != MEASUREMENT_RUNTIME_COMMANDS:
            raise ValueError("runtime command closures must use the pinned order")
        regular_by_path = {item.relative_path: item for item in self.regular_files}
        symlink_by_path = {item.relative_path: item for item in self.symlinks}
        for command in self.commands:
            resolved_path = command.binary_manifest_path
            if resolved_path in symlink_by_path:
                resolved_path = symlink_by_path[resolved_path].resolved_relative_path
            identity = regular_by_path.get(resolved_path)
            if identity is None:
                raise ValueError("runtime command binary is absent from the manifest")
            if (
                identity.sha256 != command.binary_sha256
                or identity.size_bytes != command.binary_size_bytes
            ):
                raise ValueError("runtime command binary differs from its manifest identity")
            for dependency in command.dependencies:
                if dependency.classification != "project_owned":
                    continue
                assert dependency.resolved_path is not None
                prefix = self.build_bin_root + "/"
                if not dependency.resolved_path.startswith(prefix):
                    raise ValueError("project-owned dependency is outside build/bin")
                relative = dependency.resolved_path.removeprefix(prefix)
                dependency_identity = regular_by_path.get(relative)
                if dependency_identity is None:
                    raise ValueError("project-owned dependency is absent from the manifest")
                if (
                    dependency_identity.sha256 != dependency.sha256
                    or dependency_identity.size_bytes != dependency.size_bytes
                ):
                    raise ValueError("project-owned dependency differs from its manifest identity")
        perplexity = self.commands[-1]
        if not any(
            dependency.classification == "project_owned"
            and "libllama-perplexity-impl" in dependency.soname
            for dependency in perplexity.dependencies
        ):
            raise ValueError("llama-perplexity closure omits its patched implementation DSO")

        expected_manifest = measurement_runtime_manifest_sha256(
            build_bin_root=self.build_bin_root,
            regular_files=self.regular_files,
            symlinks=self.symlinks,
        )
        if self.manifest_sha256 != expected_manifest:
            raise ValueError("runtime manifest hash differs from its exact inventory")
        return self


def measurement_supporting_record_path(
    run_id: str,
    *,
    kind: MeasurementSupportingRecordKind,
    content_sha256: str,
) -> str:
    """Return the only content-addressed path for a compact supporting record."""

    _validate_run_id(run_id)
    _validate_sha256(content_sha256, label="measurement supporting-record SHA-256")
    return PurePosixPath(
        "runs",
        run_id,
        "records",
        kind,
        f"{content_sha256}.json",
    ).as_posix()


class MeasurementSupportingRecordReference(_StrictControlModel):
    """Content-addressed reference to one compact subject or comparison record."""

    schema_version: Literal["inkling-measurement-supporting-reference-v1"] = (
        "inkling-measurement-supporting-reference-v1"
    )
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    kind: MeasurementSupportingRecordKind
    relative_path: StrictStr
    content_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    size_bytes: StrictInt = Field(gt=0, le=MEASUREMENT_CONTROL_RECORD_MAX_BYTES)

    @field_validator("relative_path")
    @classmethod
    def path_is_canonical(cls, value: str) -> str:
        return validate_repository_relative_path(value)

    @model_validator(mode="after")
    def path_is_content_addressed(self) -> MeasurementSupportingRecordReference:
        expected = measurement_supporting_record_path(
            self.run_id,
            kind=self.kind,
            content_sha256=self.content_sha256,
        )
        if self.relative_path != expected:
            raise ValueError("supporting record path differs from run, kind, and content hash")
        return self


def build_measurement_supporting_record_reference(
    payload: bytes,
    *,
    run_id: str,
    kind: MeasurementSupportingRecordKind,
) -> MeasurementSupportingRecordReference:
    """Build a run-bound compact-record reference from exact canonical bytes."""

    if not isinstance(payload, bytes):
        raise TypeError("measurement supporting record must be bytes")
    if len(payload) > MEASUREMENT_CONTROL_RECORD_MAX_BYTES:
        raise ValueError("measurement supporting record exceeds its size limit")
    raw = strict_measurement_json_object(payload)
    if payload != canonical_measurement_json_bytes(raw):
        raise ValueError("measurement supporting record bytes are not canonical")
    digest = hashlib.sha256(payload).hexdigest()
    return MeasurementSupportingRecordReference(
        run_id=run_id,
        kind=kind,
        relative_path=measurement_supporting_record_path(
            run_id,
            kind=kind,
            content_sha256=digest,
        ),
        content_sha256=digest,
        size_bytes=len(payload),
    )


def validate_measurement_supporting_record_reference(
    payload: bytes,
    *,
    expected: MeasurementSupportingRecordReference,
) -> MeasurementSupportingRecordReference:
    """Rebuild and compare a compact-record reference against exact bytes."""

    observed = build_measurement_supporting_record_reference(
        payload,
        run_id=expected.run_id,
        kind=expected.kind,
    )
    if observed != expected:
        raise ValueError("supporting record reference differs from exact bytes")
    return observed


class MeasurementSuiteQuality(_StrictControlModel):
    """Paired accuracy for one exact eight-item diagnostic suite."""

    suite: MeasurementQualitySuite
    item_count: Literal[8]
    bf16_accuracy: StrictFloat = Field(ge=0.0, le=1.0)
    q3_accuracy: StrictFloat = Field(ge=0.0, le=1.0)
    accuracy_loss: StrictFloat = Field(ge=-1.0, le=1.0)
    bf16_floor_passed: StrictBool
    q3_non_inferiority_passed: StrictBool

    @model_validator(mode="after")
    def derived_values_are_exact(self) -> MeasurementSuiteQuality:
        if not _is_exact_unit_fraction(self.bf16_accuracy, self.item_count) or not (
            _is_exact_unit_fraction(self.q3_accuracy, self.item_count)
        ):
            raise ValueError("suite accuracy must be an exact multiple of one eighth")
        loss = self.bf16_accuracy - self.q3_accuracy
        if not _float_equal(self.accuracy_loss, loss):
            raise ValueError("suite accuracy loss differs from paired accuracies")
        if self.bf16_floor_passed != (self.bf16_accuracy >= 0.5):
            raise ValueError("suite BF16 adequacy gate differs from measured accuracy")
        if self.q3_non_inferiority_passed != (loss <= 0.125):
            raise ValueError("suite Q3 gate differs from measured accuracy loss")
        return self


class MeasurementQualityRollup(_StrictControlModel):
    """Complete paired NLL, perplexity, and diagnostic quality result."""

    paired_token_positions: Literal[16320]
    bf16_diagnostic_items_scored: Literal[64]
    q3_diagnostic_items_scored: Literal[64]
    bf16_mean_nll: StrictFloat = Field(ge=0.0)
    q3_mean_nll: StrictFloat = Field(ge=0.0)
    mean_nll_delta: StrictFloat
    bf16_perplexity: StrictFloat = Field(gt=0.0)
    q3_perplexity: StrictFloat = Field(gt=0.0)
    printed_perplexity_absolute_tolerance: StrictFloat = (
        MEASUREMENT_PRINTED_PERPLEXITY_ABSOLUTE_TOLERANCE
    )
    bf16_overall_accuracy: StrictFloat = Field(ge=0.0, le=1.0)
    q3_overall_accuracy: StrictFloat = Field(ge=0.0, le=1.0)
    overall_accuracy_loss: StrictFloat = Field(ge=-1.0, le=1.0)
    suites: tuple[
        MeasurementSuiteQuality,
        MeasurementSuiteQuality,
        MeasurementSuiteQuality,
        MeasurementSuiteQuality,
        MeasurementSuiteQuality,
        MeasurementSuiteQuality,
        MeasurementSuiteQuality,
        MeasurementSuiteQuality,
    ]
    all_items_scored: Literal[True]
    all_suites_interpretable: Literal[True]
    mean_nll_gate_passed: StrictBool
    overall_accuracy_gate_passed: StrictBool
    bf16_overall_floor_passed: StrictBool
    bf16_suite_floors_passed: StrictBool
    suite_accuracy_gates_passed: StrictBool
    non_inferiority_passed: StrictBool

    @field_validator("printed_perplexity_absolute_tolerance")
    @classmethod
    def exact_printed_perplexity_tolerance(cls, value: float) -> float:
        if value != MEASUREMENT_PRINTED_PERPLEXITY_ABSOLUTE_TOLERANCE:
            raise ValueError("printed perplexity tolerance differs from the exact protocol")
        return value

    @model_validator(mode="after")
    def gates_are_derived_from_results(self) -> MeasurementQualityRollup:
        if tuple(item.suite for item in self.suites) != MEASUREMENT_QUALITY_SUITES:
            raise ValueError("quality suites must use the checked order")
        for subject in ("bf16", "q3"):
            mean_nll = getattr(self, f"{subject}_mean_nll")
            try:
                expected_perplexity = math.exp(mean_nll)
            except OverflowError as error:
                raise ValueError("mean NLL cannot produce finite perplexity") from error
            observed_perplexity = getattr(self, f"{subject}_perplexity")
            if (
                abs(observed_perplexity - expected_perplexity)
                > self.printed_perplexity_absolute_tolerance
            ):
                raise ValueError(f"{subject.upper()} printed perplexity differs from exp(mean NLL)")
            overall = getattr(self, f"{subject}_overall_accuracy")
            if not _is_exact_unit_fraction(overall, 64):
                raise ValueError("overall accuracy must be an exact multiple of one sixty-fourth")
            suite_mean = sum(getattr(suite, f"{subject}_accuracy") for suite in self.suites) / len(
                self.suites
            )
            if not _float_equal(overall, suite_mean):
                raise ValueError("overall accuracy differs from the exact suite mean")
        nll_delta = self.q3_mean_nll - self.bf16_mean_nll
        if not _float_equal(self.mean_nll_delta, nll_delta):
            raise ValueError("mean NLL delta differs from paired means")
        overall_loss = self.bf16_overall_accuracy - self.q3_overall_accuracy
        if not _float_equal(self.overall_accuracy_loss, overall_loss):
            raise ValueError("overall accuracy loss differs from paired accuracies")
        derived = {
            "mean_nll_gate_passed": nll_delta < 0.1,
            "overall_accuracy_gate_passed": overall_loss <= 0.05,
            "bf16_overall_floor_passed": self.bf16_overall_accuracy >= 0.75,
            "bf16_suite_floors_passed": all(suite.bf16_floor_passed for suite in self.suites),
            "suite_accuracy_gates_passed": all(
                suite.q3_non_inferiority_passed for suite in self.suites
            ),
        }
        for field_name, expected in derived.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"{field_name} differs from measured quality")
        expected_non_inferiority = all(derived.values())
        if self.non_inferiority_passed != expected_non_inferiority:
            raise ValueError("quality non-inferiority result differs from its exact gates")
        return self


class MeasurementPairedPositiveValue(_StrictControlModel):
    """One positive BF16/Q3 metric pair with a derived Q3/BF16 ratio."""

    bf16: StrictFloat = Field(gt=0.0)
    q3: StrictFloat = Field(gt=0.0)
    q3_to_bf16_ratio: StrictFloat = Field(gt=0.0)

    @model_validator(mode="after")
    def ratio_is_exact(self) -> MeasurementPairedPositiveValue:
        if not _float_equal(self.q3_to_bf16_ratio, self.q3 / self.bf16):
            raise ValueError("paired metric ratio differs from BF16 and Q3 values")
        return self


class MeasurementPairedNonnegativeValue(_StrictControlModel):
    """One BF16/Q3 metric pair that may legitimately contain zero."""

    bf16: StrictFloat = Field(ge=0.0)
    q3: StrictFloat = Field(ge=0.0)


class MeasurementPairedBytes(_StrictControlModel):
    """One positive BF16/Q3 byte-count pair."""

    bf16: StrictInt = Field(gt=0)
    q3: StrictInt = Field(gt=0)


class MeasurementPairedGpuBytes(_StrictControlModel):
    """Per-GPU sampled memory maxima in exact CUDA logical-device order."""

    bf16: tuple[StrictInt, ...]
    q3: tuple[StrictInt, ...]

    @model_validator(mode="after")
    def exact_gpu_count(self) -> MeasurementPairedGpuBytes:
        if len(self.bf16) != 8 or len(self.q3) != 8:
            raise ValueError("per-GPU memory must contain exactly eight devices")
        if any(value <= 0 for value in (*self.bf16, *self.q3)):
            raise ValueError("per-GPU sampled memory maxima must be positive")
        return self


class MeasurementPairedGpuUtilization(_StrictControlModel):
    """Per-GPU sampled utilization maxima in exact device order."""

    bf16: tuple[StrictFloat, ...]
    q3: tuple[StrictFloat, ...]

    @model_validator(mode="after")
    def exact_gpu_count_and_range(self) -> MeasurementPairedGpuUtilization:
        if len(self.bf16) != 8 or len(self.q3) != 8:
            raise ValueError("per-GPU utilization must contain exactly eight devices")
        if any(not 0.0 <= value <= 100.0 for value in (*self.bf16, *self.q3)):
            raise ValueError("per-GPU utilization must be from 0 through 100 percent")
        return self


class MeasurementBenchCaseRollup(_StrictControlModel):
    """Repeated llama-bench throughput for one exact workload."""

    case: MeasurementBenchCase
    repetitions_per_subject: Literal[5]
    average_tokens_per_second: MeasurementPairedPositiveValue
    median_tokens_per_second: MeasurementPairedPositiveValue
    standard_deviation_tokens_per_second: MeasurementPairedNonnegativeValue


class MeasurementPairedFiveBatchMetricSummary(_StrictControlModel):
    """Paired five-batch distribution with fully derived statistics."""

    trial_count_per_subject: Literal[5]
    bf16_samples: tuple[
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
    ]
    q3_samples: tuple[
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
    ]
    mean: MeasurementPairedPositiveValue
    median: MeasurementPairedPositiveValue
    sample_standard_deviation: MeasurementPairedNonnegativeValue

    @model_validator(mode="after")
    def statistics_are_derived(self) -> MeasurementPairedFiveBatchMetricSummary:
        if any(value <= 0.0 for value in (*self.bf16_samples, *self.q3_samples)):
            raise ValueError("paired server batch samples must be positive")
        for subject, samples in (
            ("bf16", self.bf16_samples),
            ("q3", self.q3_samples),
        ):
            expected = (
                statistics.fmean(samples),
                statistics.median(samples),
                statistics.stdev(samples),
            )
            observed = (
                getattr(self.mean, subject),
                getattr(self.median, subject),
                getattr(self.sample_standard_deviation, subject),
            )
            if any(
                not _float_equal(item, target)
                for item, target in zip(observed, expected, strict=True)
            ):
                raise ValueError("paired server batch statistics differ from retained samples")
        return self


class MeasurementPairedFiveBatchNonnegativeMetricSummary(_StrictControlModel):
    """Paired five-batch distribution whose samples may legitimately be zero."""

    trial_count_per_subject: Literal[5]
    bf16_samples: tuple[
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
    ]
    q3_samples: tuple[
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
        StrictFloat,
    ]
    mean: MeasurementPairedNonnegativeValue
    median: MeasurementPairedNonnegativeValue
    sample_standard_deviation: MeasurementPairedNonnegativeValue

    @model_validator(mode="after")
    def statistics_are_derived(
        self,
    ) -> MeasurementPairedFiveBatchNonnegativeMetricSummary:
        if any(value < 0.0 for value in (*self.bf16_samples, *self.q3_samples)):
            raise ValueError("paired server batch samples must be nonnegative")
        for subject, samples in (
            ("bf16", self.bf16_samples),
            ("q3", self.q3_samples),
        ):
            expected = (
                statistics.fmean(samples),
                statistics.median(samples),
                statistics.stdev(samples),
            )
            observed = (
                getattr(self.mean, subject),
                getattr(self.median, subject),
                getattr(self.sample_standard_deviation, subject),
            )
            if any(
                not _float_equal(item, target)
                for item, target in zip(observed, expected, strict=True)
            ):
                raise ValueError("paired server batch statistics differ from retained samples")
        return self


class MeasurementPairedRepeatedLoadDurations(_StrictControlModel):
    """Three real cold or warm process-load trials for both subjects."""

    trial_count_per_subject: Literal[3]
    bf16_durations_seconds: tuple[StrictFloat, StrictFloat, StrictFloat]
    q3_durations_seconds: tuple[StrictFloat, StrictFloat, StrictFloat]
    median_seconds: MeasurementPairedPositiveValue
    sample_standard_deviation_seconds: MeasurementPairedNonnegativeValue

    @model_validator(mode="after")
    def statistics_are_derived(self) -> MeasurementPairedRepeatedLoadDurations:
        if any(
            value <= 0.0 for value in (*self.bf16_durations_seconds, *self.q3_durations_seconds)
        ):
            raise ValueError("paired server load durations must be positive")
        for subject, samples in (
            ("bf16", self.bf16_durations_seconds),
            ("q3", self.q3_durations_seconds),
        ):
            expected = (
                statistics.median(samples),
                statistics.stdev(samples),
            )
            observed = (
                getattr(self.median_seconds, subject),
                getattr(self.sample_standard_deviation_seconds, subject),
            )
            if any(
                not _float_equal(item, target)
                for item, target in zip(observed, expected, strict=True)
            ):
                raise ValueError("paired server load statistics differ from retained trials")
        return self


class MeasurementServerCellRollup(_StrictControlModel):
    """Complete paired server metrics for one concurrency level."""

    concurrency: Literal[1, 2, 4]
    measured_batches_per_subject: Literal[5]
    measured_requests_per_subject: StrictInt = Field(gt=0)
    request_end_to_end_latency_seconds: MeasurementPairedFiveBatchMetricSummary
    ttft_seconds: MeasurementPairedFiveBatchMetricSummary
    prompt_tokens_per_second: MeasurementPairedFiveBatchMetricSummary
    decode_tokens_per_second: MeasurementPairedFiveBatchMetricSummary
    aggregate_decode_tokens_per_second: MeasurementPairedFiveBatchMetricSummary
    inter_token_latency_p50_seconds: MeasurementPairedFiveBatchNonnegativeMetricSummary
    inter_token_latency_p95_seconds: MeasurementPairedFiveBatchNonnegativeMetricSummary
    inter_token_latency_p99_seconds: MeasurementPairedFiveBatchNonnegativeMetricSummary
    bf16_resource_sample_count: StrictInt = Field(gt=0)
    q3_resource_sample_count: StrictInt = Field(gt=0)
    max_sampled_host_rss_bytes: MeasurementPairedBytes
    max_sampled_per_gpu_memory_bytes: MeasurementPairedGpuBytes
    max_sampled_per_gpu_utilization_percent: MeasurementPairedGpuUtilization

    @model_validator(mode="after")
    def request_count_and_percentiles_are_exact(self) -> MeasurementServerCellRollup:
        if self.measured_requests_per_subject != 5 * self.concurrency:
            raise ValueError("server request count differs from batches and concurrency")
        for subject in ("bf16", "q3"):
            p50 = getattr(self.inter_token_latency_p50_seconds, f"{subject}_samples")
            p95 = getattr(self.inter_token_latency_p95_seconds, f"{subject}_samples")
            p99 = getattr(self.inter_token_latency_p99_seconds, f"{subject}_samples")
            if any(
                not lower <= middle <= upper
                for lower, middle, upper in zip(p50, p95, p99, strict=True)
            ):
                raise ValueError("server inter-token latency percentiles are not ordered")
        return self


class MeasurementPerformanceRollup(_StrictControlModel):
    """Complete matched deployment-performance result for both subjects.

    The speedup flag permits a metric-specific claim because an exact BF16
    control exists. It does not state that Q3 was faster.
    """

    llama_bench_workload_identity: MeasurementLlamaBenchWorkloadIdentity
    server_workload_identity: MeasurementServerWorkloadIdentity
    text_checkpoint_size_bytes: MeasurementPairedBytes
    multimodal_projector_size_bytes: MeasurementPairedBytes
    executable_gguf_bundle_size_bytes: MeasurementPairedBytes
    load_pair_repetitions_per_subject: Literal[3]
    workload_load_pair_trial_index: Literal[3]
    cold_server_load_trials: MeasurementPairedRepeatedLoadDurations
    warm_server_load_trials: MeasurementPairedRepeatedLoadDurations
    cold_cache_conditioning: Literal[
        "file_level_posix_fadvise_posix_fadv_dontneed_on_all_staged_gguf_files"
    ] = "file_level_posix_fadvise_posix_fadv_dontneed_on_all_staged_gguf_files"
    warm_load_protocol: Literal[
        "second_same_artifact_process_after_cold_termination_without_requested_cache_conditioning_or_eviction"
    ]
    global_cache_flush_claimed: Literal[False] = False
    cold_load_readiness_only: Literal[True] = True
    cold_load_generation_requests_executed: Literal[0] = 0
    explicit_cache_conditioning_or_eviction_requested_between_server_loads: Literal[False] = False
    warm_load_is_next_model_load_after_cold: Literal[True] = True
    requested_telemetry_sampling_interval_seconds: StrictFloat
    bench_cases: tuple[
        MeasurementBenchCaseRollup,
        MeasurementBenchCaseRollup,
        MeasurementBenchCaseRollup,
    ]
    server_cells: tuple[
        MeasurementServerCellRollup,
        MeasurementServerCellRollup,
        MeasurementServerCellRollup,
    ]
    single_request_warmups_per_subject: Literal[2]
    concurrent_batch_warmups_per_cell_per_subject: Literal[1]
    bench_cases_share_one_model_load: Literal[True]
    server_quality_and_performance_share_one_model_load: Literal[True]
    warmups_excluded_from_measurement: Literal[True]
    raw_trials_recorded: Literal[True]
    matched_runtime_hardware_workload: Literal[True]
    all_metrics_complete: Literal[True]
    equivalent_trials_valid: Literal[True]
    comparison_complete: Literal[True]
    speedup_claim_allowed: Literal[True]

    @field_validator("requested_telemetry_sampling_interval_seconds")
    @classmethod
    def telemetry_interval_is_exact(cls, value: float) -> float:
        if value != 1.0:
            raise ValueError("requested telemetry sampling interval must be exactly 1.0 seconds")
        return value

    @model_validator(mode="after")
    def exact_cells(self) -> MeasurementPerformanceRollup:
        if tuple(item.case for item in self.bench_cases) != MEASUREMENT_BENCH_CASES:
            raise ValueError("performance benchmark cases must use the checked order")
        if (
            tuple(item.concurrency for item in self.server_cells)
            != MEASUREMENT_SERVER_CONCURRENCIES
        ):
            raise ValueError("server concurrency cells must use the checked order")
        for subject in ("bf16", "q3"):
            text_size = getattr(self.text_checkpoint_size_bytes, subject)
            projector_size = getattr(self.multimodal_projector_size_bytes, subject)
            bundle_size = getattr(self.executable_gguf_bundle_size_bytes, subject)
            if bundle_size != text_size + projector_size:
                raise ValueError("executable GGUF bundle size differs from text plus projector")
        if self.multimodal_projector_size_bytes.bf16 != self.multimodal_projector_size_bytes.q3:
            raise ValueError("matched subjects must use the same multimodal projector bytes")
        return self


def measurement_quality_rollup_sha256(rollup: MeasurementQualityRollup) -> str:
    """Hash one validated paired quality rollup."""

    if not isinstance(rollup, MeasurementQualityRollup):
        raise TypeError("quality rollup hash requires a validated quality rollup")
    return _domain_hash(
        MEASUREMENT_QUALITY_ROLLUP_HASH_DOMAIN,
        canonical_measurement_json_bytes(rollup.model_dump(mode="json")),
    )


def measurement_performance_rollup_sha256(
    rollup: MeasurementPerformanceRollup,
) -> str:
    """Hash one validated paired performance rollup."""

    if not isinstance(rollup, MeasurementPerformanceRollup):
        raise TypeError("performance rollup hash requires a validated performance rollup")
    return _domain_hash(
        MEASUREMENT_PERFORMANCE_ROLLUP_HASH_DOMAIN,
        canonical_measurement_json_bytes(rollup.model_dump(mode="json")),
    )


class MeasurementTerminalBindings(_StrictControlModel):
    """Fields that bind every terminal record to one accepted remote attempt."""

    stage: Literal["matched_measurement"] = "matched_measurement"
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    control_plane_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    reviewed_config_file_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    resolved_config_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    launch_intent_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    post_spawn_acceptance_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    call_id: StrictStr = Field(pattern=_MODAL_CALL_ID_PATTERN)
    attempt_claim_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    subject_order: tuple[Literal["bf16"], Literal["q3"]] = ("bf16", "q3")
    prompt_text_recorded: Literal[False] = False
    output_text_recorded: Literal[False] = False
    mtp_included: Literal[False] = False
    mtp_supported: Literal[False] = False
    single_run_causation_claim_allowed: Literal[False] = False
    scope_warning: Literal[
        "Read the machine-readable record before use. Do not apply a result to a "
        "different model, dataset, runtime, software, hardware, or protocol."
    ] = MEASUREMENT_SCOPE_WARNING
    completed_at_utc: StrictStr

    @field_validator("completed_at_utc")
    @classmethod
    def completion_time_is_canonical(cls, value: str) -> str:
        return _canonical_utc(value, label="measurement terminal completion time")


class MeasurementSuccessTerminalReceipt(MeasurementTerminalBindings):
    """Protocol-complete matched result; scientific non-inferiority may be false."""

    schema_version: Literal["inkling-measurement-rollup-v1"] = "inkling-measurement-rollup-v1"
    status: Literal["completed"] = "completed"
    measurement_completed: Literal[True] = True
    completed_stages: tuple[MeasurementStage, ...]
    runtime_identity: MeasurementRuntimeIdentity
    runtime_manifest_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    hardware_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    model_id: Literal["thinkingmachines/Inkling"]
    model_revision: Literal["86b4d430ab871652a707666b89203a866888c5e5"]
    protocol_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    workload_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    supporting_records: tuple[
        MeasurementSupportingRecordReference,
        MeasurementSupportingRecordReference,
        MeasurementSupportingRecordReference,
    ]
    quality_rollup: MeasurementQualityRollup
    performance_rollup: MeasurementPerformanceRollup
    quality_rollup_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    performance_rollup_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    quality_retention_passed: StrictBool
    performance_comparison_complete: Literal[True]
    speedup_claim_allowed: Literal[True]

    @model_validator(mode="after")
    def complete_result_is_consistent(self) -> MeasurementSuccessTerminalReceipt:
        if self.completed_stages != MEASUREMENT_PLANNED_STAGES:
            raise ValueError("successful measurement must complete every checked stage")
        if tuple(item.kind for item in self.supporting_records) != (
            "bf16_subject",
            "q3_subject",
            "comparison",
        ):
            raise ValueError("successful measurement supporting records are incomplete")
        if any(item.run_id != self.run_id for item in self.supporting_records):
            raise ValueError("successful supporting record belongs to another run")
        if self.runtime_manifest_sha256 != self.runtime_identity.manifest_sha256:
            raise ValueError("terminal runtime manifest hash differs from its identity")
        if self.quality_rollup_sha256 != measurement_quality_rollup_sha256(self.quality_rollup):
            raise ValueError("terminal quality rollup hash differs from its result")
        if self.performance_rollup_sha256 != measurement_performance_rollup_sha256(
            self.performance_rollup
        ):
            raise ValueError("terminal performance rollup hash differs from its result")
        if self.quality_retention_passed != self.quality_rollup.non_inferiority_passed:
            raise ValueError("terminal quality result differs from the quality rollup")
        if (
            self.performance_comparison_complete != self.performance_rollup.comparison_complete
            or self.speedup_claim_allowed != self.performance_rollup.speedup_claim_allowed
        ):
            raise ValueError("terminal performance flags differ from the performance rollup")
        return self


class MeasurementFailureTerminalReceipt(MeasurementTerminalBindings):
    """Fail-closed terminal evidence for an incomplete matched attempt."""

    schema_version: Literal["inkling-measurement-failure-v1"] = "inkling-measurement-failure-v1"
    status: Literal["failed"] = "failed"
    measurement_completed: Literal[False] = False
    completed_stages: tuple[MeasurementStage, ...]
    failed_stage: MeasurementStage
    failed_subject: MeasurementSubject | None
    error_code: StrictStr = Field(pattern=_SAFE_ERROR_CODE_PATTERN)
    error_summary_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    supporting_records: tuple[MeasurementSupportingRecordReference, ...]
    runtime_identity: MeasurementRuntimeIdentity | None = None
    quality_retention_passed: Literal[False] = False
    performance_comparison_complete: Literal[False] = False
    speedup_claim_allowed: Literal[False] = False

    @model_validator(mode="after")
    def failed_result_is_a_checked_prefix(self) -> MeasurementFailureTerminalReceipt:
        completed_count = len(self.completed_stages)
        if completed_count >= len(MEASUREMENT_PLANNED_STAGES):
            raise ValueError("failure receipt cannot claim every measurement stage completed")
        if self.completed_stages != MEASUREMENT_PLANNED_STAGES[:completed_count]:
            raise ValueError("failure completed stages must be an exact checked prefix")
        if self.failed_stage != MEASUREMENT_PLANNED_STAGES[completed_count]:
            raise ValueError("failed stage must immediately follow the completed prefix")
        expected_subject: MeasurementSubject | None
        if self.failed_stage in {
            "stage_and_rehash_bf16",
            "measure_bf16_quality",
            "measure_bf16_performance",
            "release_bf16",
        }:
            expected_subject = "bf16"
        elif self.failed_stage in {
            "stage_and_rehash_q3",
            "measure_q3_quality",
            "measure_q3_performance",
            "release_q3",
        }:
            expected_subject = "q3"
        else:
            expected_subject = None
        if self.failed_subject != expected_subject:
            raise ValueError("failed subject differs from the failed stage")
        bf16_complete_index = MEASUREMENT_PLANNED_STAGES.index("measure_bf16_performance")
        q3_complete_index = MEASUREMENT_PLANNED_STAGES.index("measure_q3_performance")
        expected_kinds: tuple[MeasurementSupportingRecordKind, ...] = ()
        if completed_count > bf16_complete_index:
            expected_kinds = ("bf16_subject",)
        if completed_count > q3_complete_index:
            expected_kinds = ("bf16_subject", "q3_subject")
        kinds = tuple(item.kind for item in self.supporting_records)
        if kinds != expected_kinds:
            raise ValueError("failure supporting records differ from completed subject stages")
        if any(item.run_id != self.run_id for item in self.supporting_records):
            raise ValueError("failure supporting record belongs to another run")
        if completed_count > 0 and self.runtime_identity is None:
            raise ValueError("runtime identity is required after reference verification")
        return self


MeasurementTerminalReceipt: TypeAlias = (
    MeasurementSuccessTerminalReceipt | MeasurementFailureTerminalReceipt
)


def parse_measurement_terminal_receipt(
    payload: bytes,
    *,
    run_id: str,
    outcome: MeasurementOutcome,
) -> MeasurementTerminalReceipt:
    """Parse one canonical terminal receipt through its strict outcome schema."""

    if not isinstance(payload, bytes):
        raise TypeError("measurement terminal receipt must be bytes")
    _validate_run_id(run_id)
    strict_measurement_json_object(payload)
    model_type: type[MeasurementSuccessTerminalReceipt | MeasurementFailureTerminalReceipt] = (
        MeasurementSuccessTerminalReceipt
        if outcome == "success"
        else MeasurementFailureTerminalReceipt
    )
    try:
        receipt = model_type.model_validate_json(payload, strict=True)
    except ValidationError as error:
        raise ValueError("measurement terminal receipt schema is invalid") from error
    if receipt.run_id != run_id:
        raise ValueError("measurement terminal receipt has the wrong run ID")
    if payload != canonical_measurement_json_bytes(receipt.model_dump(mode="json")):
        raise ValueError("measurement terminal receipt bytes are not canonical")
    return receipt


def measurement_terminal_receipt_content_sha256(
    payload: bytes,
    *,
    run_id: str,
    outcome: MeasurementOutcome,
) -> str:
    """Validate and domain-hash one exact canonical terminal receipt."""

    parse_measurement_terminal_receipt(
        payload,
        run_id=run_id,
        outcome=outcome,
    )
    domain = (
        MEASUREMENT_SUCCESS_RECEIPT_HASH_DOMAIN
        if outcome == "success"
        else MEASUREMENT_FAILURE_RECEIPT_HASH_DOMAIN
    )
    return _domain_hash(domain, payload)


def measurement_terminal_receipt_path(
    run_id: str,
    *,
    outcome: MeasurementOutcome,
    content_sha256: str,
) -> str:
    """Return the content-addressed relative terminal receipt path."""

    _validate_run_id(run_id)
    _validate_sha256(content_sha256, label="measurement terminal content SHA-256")
    return PurePosixPath(
        "runs",
        run_id,
        "terminal",
        outcome,
        f"{content_sha256}.json",
    ).as_posix()


def measurement_terminal_receipt_absolute_path(
    evidence_root: str,
    run_id: str,
    *,
    outcome: MeasurementOutcome,
    content_sha256: str,
) -> str:
    """Return the canonical absolute path below one evidence mount root."""

    return measurement_absolute_evidence_path(
        evidence_root,
        measurement_terminal_receipt_path(
            run_id,
            outcome=outcome,
            content_sha256=content_sha256,
        ),
    )


class MeasurementTerminalReceiptReference(_StrictControlModel):
    """Portable and mounted identities of one immutable terminal receipt."""

    schema_version: Literal["inkling-measurement-terminal-reference-v1"] = (
        "inkling-measurement-terminal-reference-v1"
    )
    run_id: StrictStr = Field(pattern=_RUN_ID_PATTERN)
    outcome: MeasurementOutcome
    evidence_root: StrictStr
    relative_path: StrictStr
    absolute_path: StrictStr
    content_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    size_bytes: StrictInt = Field(gt=0)

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_canonical(cls, value: str) -> str:
        return validate_repository_relative_path(value)

    @field_validator("evidence_root", "absolute_path")
    @classmethod
    def absolute_path_is_canonical(cls, value: str) -> str:
        return validate_absolute_evidence_path(value)

    @model_validator(mode="after")
    def paths_are_content_addressed(self) -> MeasurementTerminalReceiptReference:
        expected = measurement_terminal_receipt_path(
            self.run_id,
            outcome=self.outcome,
            content_sha256=self.content_sha256,
        )
        if self.relative_path != expected:
            raise ValueError("measurement terminal relative path is not content addressed")
        if self.absolute_path != measurement_absolute_evidence_path(
            self.evidence_root,
            expected,
        ):
            raise ValueError("measurement terminal absolute path has the wrong evidence root")
        return self


def build_measurement_terminal_receipt_reference(
    payload: bytes,
    *,
    evidence_root: str,
    run_id: str,
    outcome: MeasurementOutcome,
) -> MeasurementTerminalReceiptReference:
    """Build a terminal reference from validated canonical receipt bytes."""

    content_sha256 = measurement_terminal_receipt_content_sha256(
        payload,
        run_id=run_id,
        outcome=outcome,
    )
    relative_path = measurement_terminal_receipt_path(
        run_id,
        outcome=outcome,
        content_sha256=content_sha256,
    )
    return MeasurementTerminalReceiptReference(
        run_id=run_id,
        outcome=outcome,
        evidence_root=evidence_root,
        relative_path=relative_path,
        absolute_path=measurement_absolute_evidence_path(evidence_root, relative_path),
        content_sha256=content_sha256,
        size_bytes=len(payload),
    )


def validate_measurement_terminal_receipt_reference(
    payload: bytes,
    *,
    evidence_root: str,
    expected: MeasurementTerminalReceiptReference,
) -> MeasurementTerminalReceiptReference:
    """Rebuild and compare one terminal reference against exact bytes."""

    observed = build_measurement_terminal_receipt_reference(
        payload,
        evidence_root=evidence_root,
        run_id=expected.run_id,
        outcome=expected.outcome,
    )
    if observed != expected:
        raise ValueError("measurement terminal reference differs from exact receipt bytes")
    return observed


__all__ = [
    "MEASUREMENT_ATTEMPT_CLAIM_HASH_DOMAIN",
    "MEASUREMENT_ATTEMPT_REGISTRY_NAME",
    "MEASUREMENT_BENCH_CASES",
    "MEASUREMENT_CONTROL_PLANE_HASH_DOMAIN",
    "MEASUREMENT_CONTROL_RECORD_MAX_BYTES",
    "MEASUREMENT_DEPLOY_CHALLENGE_HASH_DOMAIN",
    "MEASUREMENT_DEPLOY_CHALLENGE_MAX_AGE_SECONDS",
    "MEASUREMENT_DEPLOY_CONFIRMATION_PREFIX",
    "MEASUREMENT_ENVIRONMENT_NAME",
    "MEASUREMENT_EVIDENCE_VOLUME_NAME",
    "MEASUREMENT_FAILURE_RECEIPT_HASH_DOMAIN",
    "MEASUREMENT_FUNCTION_NAME",
    "MEASUREMENT_LAUNCH_CHALLENGE_HASH_DOMAIN",
    "MEASUREMENT_LAUNCH_CHALLENGE_MAX_AGE_SECONDS",
    "MEASUREMENT_LAUNCH_CONFIRMATION_PREFIX",
    "MEASUREMENT_LAUNCH_INTENT_HASH_DOMAIN",
    "MEASUREMENT_LLAMA_BENCH_PROMPT_TEMPLATE_PROTOCOL",
    "MEASUREMENT_PERFORMANCE_ROLLUP_HASH_DOMAIN",
    "MEASUREMENT_PLANNED_STAGES",
    "MEASUREMENT_POST_SPAWN_ACCEPTANCE_HASH_DOMAIN",
    "MEASUREMENT_PRINTED_PERPLEXITY_ABSOLUTE_TOLERANCE",
    "MEASUREMENT_QUALITY_ROLLUP_HASH_DOMAIN",
    "MEASUREMENT_QUALITY_SUITES",
    "MEASUREMENT_RUNTIME_COMMANDS",
    "MEASUREMENT_RUNTIME_MANIFEST_HASH_DOMAIN",
    "MEASUREMENT_SCOPE_WARNING",
    "MEASUREMENT_SERVER_CONCURRENCIES",
    "MEASUREMENT_SERVER_PROMPT_REPEAT_COUNT",
    "MEASUREMENT_SERVER_PROMPT_SEGMENT",
    "MEASUREMENT_SERVER_PROMPT_TEMPLATE_PROTOCOL",
    "MEASUREMENT_STAGE",
    "MEASUREMENT_SUBJECT_ORDER",
    "MEASUREMENT_SUCCESS_RECEIPT_HASH_DOMAIN",
    "MeasurementAppliedPatch",
    "MeasurementAttemptClaim",
    "MeasurementAttemptRegistryProtocol",
    "MeasurementBenchCase",
    "MeasurementBenchCaseRollup",
    "MeasurementControlPlaneFile",
    "MeasurementControlPlaneProvenance",
    "MeasurementDeployConfirmationChallenge",
    "MeasurementDeploymentIdentity",
    "MeasurementExecutionResources",
    "MeasurementFailureTerminalReceipt",
    "MeasurementLaunchConfirmationChallenge",
    "MeasurementLaunchIntent",
    "MeasurementLlamaBenchCaseIdentity",
    "MeasurementLlamaBenchWorkloadIdentity",
    "MeasurementOutcome",
    "MeasurementPairedBytes",
    "MeasurementPairedFiveBatchMetricSummary",
    "MeasurementPairedFiveBatchNonnegativeMetricSummary",
    "MeasurementPairedGpuBytes",
    "MeasurementPairedGpuUtilization",
    "MeasurementPairedNonnegativeValue",
    "MeasurementPairedPositiveValue",
    "MeasurementPairedRepeatedLoadDurations",
    "MeasurementPerformanceRollup",
    "MeasurementPostSpawnAcceptance",
    "MeasurementPrePatchExecutable",
    "MeasurementQualityRollup",
    "MeasurementQualitySuite",
    "MeasurementReviewedInputs",
    "MeasurementRuntimeCommand",
    "MeasurementRuntimeCommandClosure",
    "MeasurementRuntimeDependency",
    "MeasurementRuntimeIdentity",
    "MeasurementRuntimeRegularFile",
    "MeasurementRuntimeSymlink",
    "MeasurementServerCellRollup",
    "MeasurementServerWorkloadIdentity",
    "MeasurementStage",
    "MeasurementSubject",
    "MeasurementSuccessTerminalReceipt",
    "MeasurementSuiteQuality",
    "MeasurementSupportingRecordKind",
    "MeasurementSupportingRecordReference",
    "MeasurementTerminalReceipt",
    "MeasurementTerminalReceiptReference",
    "build_measurement_attempt_claim",
    "build_measurement_control_plane_provenance",
    "build_measurement_launch_intent",
    "build_measurement_post_spawn_acceptance",
    "build_measurement_supporting_record_reference",
    "build_measurement_terminal_receipt_reference",
    "canonical_measurement_json_bytes",
    "claim_measurement_attempt",
    "measurement_absolute_evidence_path",
    "measurement_app_name",
    "measurement_attempt_claim_path",
    "measurement_attempt_registry_key",
    "measurement_control_plane_sha256",
    "measurement_deployment_tag",
    "measurement_launch_intent_path",
    "measurement_llama_bench_dataset_bytes",
    "measurement_performance_rollup_sha256",
    "measurement_post_spawn_acceptance_path",
    "measurement_quality_rollup_sha256",
    "measurement_runtime_manifest_sha256",
    "measurement_server_prompt_source_text",
    "measurement_supporting_record_path",
    "measurement_terminal_receipt_absolute_path",
    "measurement_terminal_receipt_content_sha256",
    "measurement_terminal_receipt_path",
    "parse_measurement_terminal_receipt",
    "strict_measurement_json_object",
    "validate_absolute_evidence_path",
    "validate_measurement_attempt_claim",
    "validate_measurement_control_plane_provenance",
    "validate_measurement_deploy_challenge_not_expired",
    "validate_measurement_launch_intent",
    "validate_measurement_launch_intent_not_expired",
    "validate_measurement_post_spawn_acceptance",
    "validate_measurement_supporting_record_reference",
    "validate_measurement_terminal_receipt_reference",
    "validate_repository_relative_path",
]
