"""CPU-safe execution contracts for the exact matched Inkling CUDA cell.

This module does not start Modal or execute llama.cpp. It validates the
owner-tagged backend markers that a later matched runner must retain.
Historical two-GPU smoke receipts keep their original validators.
"""

from __future__ import annotations

import csv
import ctypes
import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum, StrEnum
from pathlib import PurePosixPath
from typing import Any, Final, Literal, TypeAlias
from uuid import UUID

from pydantic import (
    BaseModel,
    Field,
    FiniteFloat,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.gguf.inkling_matched import InklingMatchedCellConfig
from inkling_quant_lab.gguf.inkling_smoke import (
    MAX_BACKEND_FAILURE_LINE_BYTES,
    PINNED_UNPADDED_VOCAB_SIZE,
    PINNED_VOCAB_SIZE,
    ArtifactLoadEvidence,
    LoaderOffloadEvidence,
    RawLogitAuditEvidence,
    parse_backend_audit_rows_v2,
)

CudaBackendIdentity: TypeAlias = tuple[StrictInt, StrictStr, StrictStr]

_MAX_CUDA_GPU_COUNT: Final = 64
_MAX_AUDIT_LOG_CHARACTERS: Final = 16 * 1024 * 1024
_MAX_AUDIT_GRAPH_ROWS: Final = 1_024
_MAX_AUDIT_IDENTITY_ROWS: Final = 8_192
_MAX_AUDIT_CPU_ROWS: Final = 1_024
_MAX_BACKEND_IDENTIFIER_CHARACTERS: Final = 64
_MARKER_PREFIX: Final = "IQL_SMOKE_"
_GRAPH_MARKER_V2: Final = "IQL_SMOKE_BACKEND_GRAPH_V2"
_IDENTITY_MARKER_V2: Final = "IQL_SMOKE_BACKEND_IDENTITY_V2"
_CPU_NODE_MARKER_V2: Final = "IQL_SMOKE_CPU_NODE_V2"
_TEXT_ONLY_GRAPH_OWNERS: Final = frozenset({"text"})
_MULTIMODAL_GRAPH_OWNERS: Final = frozenset({"text", "vision", "audio"})


class ExactCudaPlacementPolicy(StrictFrozenModel):
    """One explicit CUDA graph-placement policy."""

    schema_version: Literal["iql-exact-cuda-placement-policy-v1"]
    gpu_count: StrictInt = Field(ge=1, le=_MAX_CUDA_GPU_COUNT)
    tensor_split: tuple[StrictInt, ...]
    split_mode: Literal["layer"]
    text_graph_policy: Literal["at_least_one_all_expected_cuda"]
    vision_graph_policy: Literal["cuda0_only"]
    audio_graph_policy: Literal["cuda0_only"]

    @model_validator(mode="after")
    def exact_tensor_split(self) -> ExactCudaPlacementPolicy:
        if len(self.tensor_split) != self.gpu_count:
            raise ValueError("CUDA tensor split length differs from the GPU count")
        if any(value != 1 for value in self.tensor_split):
            raise ValueError("CUDA tensor split must assign one equal part to every GPU")
        return self


def expected_cuda_identities(gpu_count: int) -> tuple[tuple[int, str, str], ...]:
    """Return the exact ordered llama.cpp CUDA identities for one cell."""

    if type(gpu_count) is not int or not 1 <= gpu_count <= _MAX_CUDA_GPU_COUNT:
        raise ValueError(f"gpu_count must be an integer from 1 to {_MAX_CUDA_GPU_COUNT}")
    return tuple((index, f"CUDA{index}", f"CUDA{index}") for index in range(gpu_count))


def build_matched_cuda_placement_policy(
    config: InklingMatchedCellConfig,
) -> ExactCudaPlacementPolicy:
    """Build the CUDA placement policy from one validated matched-cell config."""

    return ExactCudaPlacementPolicy(
        schema_version="iql-exact-cuda-placement-policy-v1",
        gpu_count=config.resources.gpu_count,
        tensor_split=config.runtime.tensor_split,
        split_mode=config.runtime.split_mode,
        text_graph_policy="at_least_one_all_expected_cuda",
        vision_graph_policy="cuda0_only",
        audio_graph_policy="cuda0_only",
    )


class ExactCudaGraphAuditRow(StrictFrozenModel):
    """One owner-tagged graph row with strict numeric fields."""

    graph_uid: StrictInt = Field(gt=0)
    graph_owner: Literal["text", "vision", "audio", "unknown"]
    phase: Literal["post_assignment_pre_split"]
    scope: Literal["non_view_compute"]
    compute: StrictInt = Field(gt=0)
    gpu: StrictInt = Field(gt=0)
    cpu: StrictInt = Field(ge=0, le=0)
    accel: StrictInt = Field(ge=0)
    other: StrictInt = Field(ge=0, le=0)
    unassigned: StrictInt = Field(ge=0, le=0)

    @model_validator(mode="after")
    def exact_assignment(self) -> ExactCudaGraphAuditRow:
        if self.compute != self.gpu + self.cpu + self.accel + self.other + self.unassigned:
            raise ValueError("backend graph category counts do not equal its compute count")
        return self


class ExactCudaIdentityAuditRow(StrictFrozenModel):
    """One owner-tagged backend identity row with strict scalar fields."""

    graph_uid: StrictInt = Field(gt=0)
    graph_owner: Literal["text", "vision", "audio", "unknown"]
    backend_index: StrictInt = Field(ge=0)
    backend_name: StrictStr = Field(
        min_length=1,
        max_length=_MAX_BACKEND_IDENTIFIER_CHARACTERS,
    )
    device_name: StrictStr = Field(
        min_length=1,
        max_length=_MAX_BACKEND_IDENTIFIER_CHARACTERS,
    )
    device_type: Literal["cpu", "gpu", "igpu", "accel", "meta", "unassigned"]
    compute: StrictInt = Field(gt=0)

    @field_validator("backend_name", "device_name")
    @classmethod
    def marker_identifier_is_canonical(cls, value: str) -> str:
        if "\x00" in value or any(character.isspace() for character in value):
            raise ValueError("backend marker identifiers must be non-whitespace text")
        return value


class ExactCudaBackendAuditEvidence(StrictFrozenModel):
    """Complete owner-tagged graph evidence for one explicit CUDA policy."""

    schema_version: Literal["inkling-exact-cuda-backend-audit-v1"]
    policy: ExactCudaPlacementPolicy
    expected_identities: tuple[CudaBackendIdentity, ...]
    observed_graphs: StrictInt = Field(gt=0, le=_MAX_AUDIT_GRAPH_ROWS)
    compute_operations: StrictInt = Field(gt=0)
    gpu_operations: StrictInt = Field(gt=0)
    accelerator_operations: StrictInt = Field(ge=0, le=0)
    cpu_operations: StrictInt = Field(ge=0, le=0)
    other_operations: StrictInt = Field(ge=0, le=0)
    unassigned_operations: StrictInt = Field(ge=0, le=0)
    graphs: tuple[ExactCudaGraphAuditRow, ...] = Field(
        min_length=1,
        max_length=_MAX_AUDIT_GRAPH_ROWS,
    )
    identities: tuple[ExactCudaIdentityAuditRow, ...] = Field(
        min_length=1,
        max_length=_MAX_AUDIT_IDENTITY_ROWS,
    )
    exact_cuda_identity_inventory: StrictBool
    text_full_cell_observed: StrictBool
    projector_graphs_cuda0_only: StrictBool
    all_compute_operations_accelerated: StrictBool
    no_cpu_model_graph_fallback: StrictBool

    @model_validator(mode="after")
    def exact_cuda_placement(self) -> ExactCudaBackendAuditEvidence:
        _validate_exact_cuda_placement_evidence(
            self,
            required_graph_owners=_MULTIMODAL_GRAPH_OWNERS,
        )
        return self


class ExactCudaTextBackendAuditEvidence(StrictFrozenModel):
    """Exact CUDA graph evidence for a text-only llama.cpp workload."""

    schema_version: Literal["inkling-exact-text-cuda-backend-audit-v1"]
    policy: ExactCudaPlacementPolicy
    expected_identities: tuple[CudaBackendIdentity, ...]
    observed_graphs: StrictInt = Field(gt=0, le=_MAX_AUDIT_GRAPH_ROWS)
    compute_operations: StrictInt = Field(gt=0)
    gpu_operations: StrictInt = Field(gt=0)
    accelerator_operations: StrictInt = Field(ge=0, le=0)
    cpu_operations: StrictInt = Field(ge=0, le=0)
    other_operations: StrictInt = Field(ge=0, le=0)
    unassigned_operations: StrictInt = Field(ge=0, le=0)
    graphs: tuple[ExactCudaGraphAuditRow, ...] = Field(
        min_length=1,
        max_length=_MAX_AUDIT_GRAPH_ROWS,
    )
    identities: tuple[ExactCudaIdentityAuditRow, ...] = Field(
        min_length=1,
        max_length=_MAX_AUDIT_IDENTITY_ROWS,
    )
    exact_cuda_identity_inventory: StrictBool
    text_full_cell_observed: StrictBool
    projector_graphs_cuda0_only: StrictBool
    all_compute_operations_accelerated: StrictBool
    no_cpu_model_graph_fallback: StrictBool

    @model_validator(mode="after")
    def exact_cuda_placement(self) -> ExactCudaTextBackendAuditEvidence:
        _validate_exact_cuda_placement_evidence(
            self,
            required_graph_owners=_TEXT_ONLY_GRAPH_OWNERS,
        )
        return self


def _validate_exact_cuda_placement_evidence(
    evidence: ExactCudaBackendAuditEvidence | ExactCudaTextBackendAuditEvidence,
    *,
    required_graph_owners: frozenset[str],
) -> None:
    """Validate fields shared by exact multimodal and text-only CUDA records."""

    if not all(
        (
            evidence.exact_cuda_identity_inventory,
            evidence.text_full_cell_observed,
            evidence.projector_graphs_cuda0_only,
            evidence.all_compute_operations_accelerated,
            evidence.no_cpu_model_graph_fallback,
        )
    ):
        raise ValueError("backend audit proof fields must all be true")

    expected_order = expected_cuda_identities(evidence.policy.gpu_count)
    if evidence.expected_identities != expected_order:
        raise ValueError("backend audit expected CUDA identities differ from its policy")
    expected = set(expected_order)
    cuda0_identity = expected_order[0]

    if evidence.observed_graphs != len(evidence.graphs):
        raise ValueError("backend graph count differs from its graph records")
    graph_uids = tuple(row.graph_uid for row in evidence.graphs)
    if len(graph_uids) != len(set(graph_uids)):
        raise ValueError("backend audit contains duplicate graph identities")
    if frozenset(row.graph_owner for row in evidence.graphs) != required_graph_owners:
        raise ValueError("backend audit graph owners differ from the closed workload scope")

    identity_keys = tuple((row.graph_uid, row.backend_index) for row in evidence.identities)
    if len(identity_keys) != len(set(identity_keys)):
        raise ValueError("backend audit contains duplicate backend identities")
    if {row.graph_uid for row in evidence.identities} != set(graph_uids):
        raise ValueError("backend identities do not cover the exact graph set")
    if any(row.graph_owner == "unknown" for row in evidence.identities):
        raise ValueError("backend identity contains an unknown graph owner")
    if any(row.device_type != "gpu" for row in evidence.identities):
        raise ValueError("backend audit used a non-CUDA accelerator")

    identities_by_graph: dict[int, list[ExactCudaIdentityAuditRow]] = {}
    for identity in evidence.identities:
        identities_by_graph.setdefault(identity.graph_uid, []).append(identity)

    text_full_cell_observed = False
    category_totals = {
        "gpu": 0,
        "cpu": 0,
        "accel": 0,
        "other": 0,
        "unassigned": 0,
    }
    for graph in evidence.graphs:
        if (
            graph.cpu != 0
            or graph.accel != 0
            or graph.other != 0
            or graph.unassigned != 0
            or graph.gpu != graph.compute
        ):
            raise ValueError("backend graph contains a forbidden device category")

        graph_identities = tuple(identities_by_graph[graph.graph_uid])
        if any(row.graph_owner != graph.graph_owner for row in graph_identities):
            raise ValueError("backend graph and identity owner fields differ")
        if tuple(row.backend_index for row in graph_identities) != tuple(
            sorted(row.backend_index for row in graph_identities)
        ):
            raise ValueError("backend identities are not in CUDA ordinal order")
        observed = {
            (row.backend_index, row.backend_name, row.device_name) for row in graph_identities
        }
        if (
            len(graph_identities) != len(observed)
            or not observed.issubset(expected)
            or cuda0_identity not in observed
        ):
            raise ValueError(
                "backend graph does not prove the exact CUDA index and device identities"
            )
        if graph.graph_owner == "text":
            text_full_cell_observed |= observed == expected
        elif observed != {cuda0_identity}:
            raise ValueError("vision and audio graphs must use CUDA0 only")

        if sum(row.compute for row in graph_identities) != graph.compute:
            raise ValueError("backend identity counts do not equal graph compute count")
        category_totals["gpu"] += sum(row.compute for row in graph_identities)

    if not text_full_cell_observed:
        raise ValueError("backend audit does not prove one full-cell text graph")

    aggregate = {
        "gpu": evidence.gpu_operations,
        "cpu": evidence.cpu_operations,
        "accel": evidence.accelerator_operations,
        "other": evidence.other_operations,
        "unassigned": evidence.unassigned_operations,
    }
    if category_totals != aggregate:
        raise ValueError("backend aggregate counts differ from graph evidence")
    if sum(graph.compute for graph in evidence.graphs) != evidence.compute_operations:
        raise ValueError("backend aggregate compute count differs from graph evidence")


def _validate_audit_input_bounds(log_text: str) -> None:
    """Reject backend logs that exceed the checked parser resource bounds."""

    if len(log_text) > _MAX_AUDIT_LOG_CHARACTERS:
        raise ValueError("backend audit log exceeds the character limit")

    graph_markers = 0
    identity_markers = 0
    cpu_markers = 0
    for line in log_text.split("\n"):
        if _MARKER_PREFIX in line:
            if len(line) > MAX_BACKEND_FAILURE_LINE_BYTES:
                raise ValueError("backend audit marker line exceeds the length limit")
            graph_markers += line.count(_GRAPH_MARKER_V2)
            identity_markers += line.count(_IDENTITY_MARKER_V2)
            cpu_markers += line.count(_CPU_NODE_MARKER_V2)
            if graph_markers > _MAX_AUDIT_GRAPH_ROWS:
                raise ValueError("backend audit exceeds the graph marker limit")
            if identity_markers > _MAX_AUDIT_IDENTITY_ROWS:
                raise ValueError("backend audit exceeds the identity marker limit")
            if cpu_markers > _MAX_AUDIT_CPU_ROWS:
                raise ValueError("backend audit exceeds the CPU-node marker limit")


def _parse_exact_cuda_backend_audit_fields(
    log_text: str,
    *,
    policy: ExactCudaPlacementPolicy,
) -> dict[str, object]:
    """Parse common V2 marker fields for one explicit CUDA policy."""

    _validate_audit_input_bounds(log_text)
    rows = parse_backend_audit_rows_v2(log_text)
    graphs = tuple(
        ExactCudaGraphAuditRow.model_validate(row.model_dump(mode="python")) for row in rows.graphs
    )
    identities = tuple(
        ExactCudaIdentityAuditRow.model_validate(row.model_dump(mode="python"))
        for row in rows.identities
    )
    return {
        "policy": policy,
        "expected_identities": expected_cuda_identities(policy.gpu_count),
        "observed_graphs": len(graphs),
        "compute_operations": sum(row.compute for row in graphs),
        "gpu_operations": sum(row.gpu for row in graphs),
        "accelerator_operations": sum(row.accel for row in graphs),
        "cpu_operations": sum(row.cpu for row in graphs),
        "other_operations": sum(row.other for row in graphs),
        "unassigned_operations": sum(row.unassigned for row in graphs),
        "graphs": graphs,
        "identities": identities,
        "exact_cuda_identity_inventory": True,
        "text_full_cell_observed": True,
        "projector_graphs_cuda0_only": True,
        "all_compute_operations_accelerated": True,
        "no_cpu_model_graph_fallback": True,
    }


def parse_exact_cuda_backend_audit(
    log_text: str,
    *,
    policy: ExactCudaPlacementPolicy,
) -> ExactCudaBackendAuditEvidence:
    """Parse a multimodal workload's V2 markers against one CUDA policy."""

    return ExactCudaBackendAuditEvidence.model_validate(
        {
            "schema_version": "inkling-exact-cuda-backend-audit-v1",
            **_parse_exact_cuda_backend_audit_fields(log_text, policy=policy),
        }
    )


def parse_exact_text_cuda_backend_audit(
    log_text: str,
    *,
    policy: ExactCudaPlacementPolicy,
) -> ExactCudaTextBackendAuditEvidence:
    """Parse a text-only workload's V2 markers against one CUDA policy."""

    return ExactCudaTextBackendAuditEvidence.model_validate(
        {
            "schema_version": "inkling-exact-text-cuda-backend-audit-v1",
            **_parse_exact_cuda_backend_audit_fields(log_text, policy=policy),
        }
    )


def parse_matched_cuda_backend_audit(
    log_text: str,
    *,
    config: InklingMatchedCellConfig,
) -> ExactCudaBackendAuditEvidence:
    """Parse V2 markers against the policy in one validated matched-cell config."""

    return parse_exact_cuda_backend_audit(
        log_text,
        policy=build_matched_cuda_placement_policy(config),
    )


_SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
_GPU_UUID_PATTERN: Final = r"^GPU-[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12}$"
_MATCHED_GPU_COUNT: Final = 8
_MATCHED_TENSOR_SPLIT: Final = "1,1,1,1,1,1,1,1"
_MATCHED_SERVER_BINARY: Final = "/opt/llama.cpp/build/bin/llama-server"
_MATCHED_PROJECTOR_RELATIVE_PATH: Final = "mmproj/mmproj-BF16.gguf"
_MATCHED_SUBJECT_RECEIPT_DOMAIN: Final = "inkling-matched-subject-smoke-v1\n"
_MATCHED_ROLLUP_RECEIPT_DOMAIN: Final = "inkling-matched-rollup-v1\n"
_MATCHED_FAILURE_RECEIPT_DOMAIN: Final = "inkling-matched-failure-v1\n"
_MATCHED_CUDA_P2P_PERFORMANCE_RANK_ATTRIBUTE: Final = 0x01
_MATCHED_CUDA_P2P_ACCESS_SUPPORTED_ATTRIBUTE: Final = 0x02
_MATCHED_CUDA_P2P_NATIVE_ATOMIC_SUPPORTED_ATTRIBUTE: Final = 0x03
_MATCHED_CUDA_P2P_CUDA_ARRAY_ACCESS_SUPPORTED_ATTRIBUTE: Final = 0x04
_MATCHED_CUDA_P2P_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED_ATTRIBUTE: Final = 0x05
_MATCHED_CUDA_DRIVER_STUB_ROOT: Final = PurePosixPath("/usr/local/cuda/lib64/stubs")
_MATCHED_CUDA_DRIVER_LINK_ROOT: Final = PurePosixPath("/opt/iql-cuda-driver-link")


class MatchedSubject(StrEnum):
    """One subject in the fixed matched-comparison execution order."""

    BF16 = "bf16"
    Q3 = "q3"


MATCHED_SUBJECT_ORDER: Final = (MatchedSubject.BF16, MatchedSubject.Q3)


def _subject_ordinal(subject: MatchedSubject) -> int:
    return MATCHED_SUBJECT_ORDER.index(subject)


def _subject_server_port(subject: MatchedSubject) -> int:
    return 18_080 + _subject_ordinal(subject)


def _subject_server_log_path(subject: MatchedSubject) -> str:
    return f"/tmp/inkling-matched-{subject.value}-llama-server.log"


def _subject_first_shard_name(subject: MatchedSubject) -> str:
    if subject is MatchedSubject.BF16:
        return "inkling-BF16-00001-of-00049.gguf"
    return "inkling-Q3_K_M-00001-of-00049.gguf"


def _subject_shard_relative_path(subject: MatchedSubject, ordinal: int) -> str:
    if subject is MatchedSubject.BF16:
        return f"bf16/inkling-BF16-{ordinal:05d}-of-00049.gguf"
    return f"q3_k_m/inkling-Q3_K_M-{ordinal:05d}-of-00049.gguf"


def _canonical_absolute_path(value: str, *, label: str) -> str:
    if "\x00" in value or "\\" in value or "//" in value:
        raise ValueError(f"{label} must be a canonical absolute POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a canonical absolute POSIX path")
    canonical = path.as_posix()
    if canonical != value:
        raise ValueError(f"{label} must be a canonical absolute POSIX path")
    return value


def _canonical_relative_path(value: str, *, label: str) -> str:
    if "\x00" in value or "\\" in value or "//" in value:
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    canonical = path.as_posix()
    if not canonical or canonical != value:
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return value


def _canonical_gpu_uuid(value: str, *, label: str) -> str:
    if re.fullmatch(_GPU_UUID_PATTERN, value) is None:
        raise ValueError(f"{label} is not a full GPU UUID")
    return value


def _completed_at_utc(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("completed_at_utc must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("completed_at_utc must use an explicit UTC offset")
    return value


def _json_compatible(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_compatible(item) for item in value]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _json_compatible(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _receipt_sha256(domain: str, payload: Mapping[str, object]) -> str:
    canonical = dict(payload)
    canonical.pop("receipt_sha256", None)
    return hashlib.sha256(domain.encode("utf-8") + _canonical_json_bytes(canonical)).hexdigest()


class MatchedServerCommandSpec(StrictFrozenModel):
    """Exact server command inputs for one isolated comparison subject."""

    schema_version: Literal["inkling-matched-server-command-v1"]
    subject: MatchedSubject
    server_binary: Literal["/opt/llama.cpp/build/bin/llama-server"]
    first_shard_path: StrictStr = Field(min_length=1)
    projector_path: StrictStr = Field(min_length=1)
    host: Literal["127.0.0.1"]
    port: StrictInt = Field(ge=1, le=65_535)
    server_log_path: StrictStr = Field(min_length=1)
    endpoint: Literal["/completion"]
    log_verbosity: Literal[4]
    context_size: Literal[8192]

    @field_validator("first_shard_path", "projector_path", "server_log_path")
    @classmethod
    def paths_are_canonical(cls, value: str, info: Any) -> str:
        return _canonical_absolute_path(value, label=info.field_name)

    @model_validator(mode="after")
    def subject_paths_and_socket_are_exact(self) -> MatchedServerCommandSpec:
        if PurePosixPath(self.first_shard_path).name != _subject_first_shard_name(self.subject):
            raise ValueError("server model path is not the subject's metadata-only first shard")
        if not self.first_shard_path.endswith(f"/{_subject_shard_relative_path(self.subject, 1)}"):
            raise ValueError("server first shard path differs from the matched artifact contract")
        if not self.projector_path.endswith(f"/{_MATCHED_PROJECTOR_RELATIVE_PATH}"):
            raise ValueError("server projector path differs from the shared projector contract")
        if self.port != _subject_server_port(self.subject):
            raise ValueError("server port differs from the subject-isolated port")
        if self.server_log_path != _subject_server_log_path(self.subject):
            raise ValueError("server log path differs from the subject-isolated log path")
        return self


def build_matched_server_command(spec: MatchedServerCommandSpec) -> tuple[str, ...]:
    """Build the fixed full-offload eight-GPU llama-server command."""

    return (
        spec.server_binary,
        "--log-verbosity",
        str(spec.log_verbosity),
        "--model",
        spec.first_shard_path,
        "--mmproj",
        spec.projector_path,
        "--host",
        spec.host,
        "--port",
        str(spec.port),
        "--ctx-size",
        str(spec.context_size),
        "--n-gpu-layers",
        "all",
        "--n-cpu-moe",
        "0",
        "--split-mode",
        "layer",
        "--tensor-split",
        _MATCHED_TENSOR_SPLIT,
        "--flash-attn",
        "on",
        "--mmap",
        "--mmproj-offload",
        "--parallel",
        "1",
        "--threads",
        "16",
        "--threads-batch",
        "16",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--no-webui",
    )


def build_matched_server_environment(
    inherited_environment: Mapping[str, str],
    *,
    audit_environment: Mapping[str, str],
) -> dict[str, str]:
    """Remove every inherited llama-server argument before adding audit controls."""

    clean: dict[str, str] = {}
    for key, value in inherited_environment.items():
        if "\x00" in key or "\x00" in value or "=" in key:
            raise ValueError("server environment contains an invalid key or value")
        if not key.startswith("LLAMA_ARG_"):
            clean[key] = value
    for key, value in audit_environment.items():
        if "\x00" in key or "\x00" in value or "=" in key:
            raise ValueError("server audit environment contains an invalid key or value")
        if key.startswith("LLAMA_ARG_"):
            raise ValueError("audit environment cannot inject llama-server arguments")
        if key in clean and clean[key] != value:
            raise ValueError("audit environment cannot replace an inherited environment value")
        clean[key] = value
    return clean


class MatchedNvidiaSmiGpuEvidence(StrictFrozenModel):
    """One ordered B300 identity row from the exact eight-GPU allocation."""

    cuda_ordinal: StrictInt = Field(ge=0, le=7)
    uuid: StrictStr = Field(pattern=_GPU_UUID_PATTERN)
    name: Literal["NVIDIA B300 SXM6 AC"]
    memory_total_mib: Literal[275040]
    driver_version: StrictStr = Field(pattern=r"^[0-9]+(?:\.[0-9]+)+$")
    compute_capability: Literal["10.3"]


class MatchedNvidiaSmiResourceSample(StrictFrozenModel):
    """One UUID-keyed resource-monitor row."""

    uuid: StrictStr = Field(pattern=_GPU_UUID_PATTERN)
    memory_used_mib: StrictInt = Field(ge=0)
    utilization_percent: StrictInt = Field(ge=0, le=100)


def _parse_exact_decimal(value: str, *, label: str) -> int:
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", value) is None:
        raise ValueError(f"{label} must be an unsigned canonical decimal")
    return int(value)


def parse_matched_nvidia_smi_identity_csv(
    payload: str,
) -> tuple[MatchedNvidiaSmiGpuEvidence, ...]:
    """Parse the exact eight B300 identity rows in CUDA ordinal order."""

    if not payload.strip() or "\x00" in payload:
        raise ValueError("nvidia-smi identity CSV must be non-empty text without NUL")
    rows = tuple(csv.reader(io.StringIO(payload), skipinitialspace=True))
    if len(rows) != _MATCHED_GPU_COUNT:
        raise ValueError("nvidia-smi identity CSV must contain exactly eight GPU rows")
    evidence: list[MatchedNvidiaSmiGpuEvidence] = []
    for index, row in enumerate(rows):
        if len(row) != 5 or any(not value.strip() for value in row):
            raise ValueError(f"nvidia-smi identity row {index} must have five populated fields")
        memory_match = re.fullmatch(r"([0-9]+)(?: MiB)?", row[2].strip())
        if memory_match is None:
            raise ValueError(f"nvidia-smi identity row {index} memory must be MiB")
        evidence.append(
            MatchedNvidiaSmiGpuEvidence(
                cuda_ordinal=index,
                uuid=_canonical_gpu_uuid(row[0].strip(), label=f"GPU {index} UUID"),
                name=row[1].strip(),
                memory_total_mib=int(memory_match.group(1)),
                driver_version=row[3].strip(),
                compute_capability=row[4].strip(),
            )
        )
    if len({gpu.uuid.lower() for gpu in evidence}) != _MATCHED_GPU_COUNT:
        raise ValueError("nvidia-smi identity GPU UUIDs must be unique")
    if len({gpu.driver_version for gpu in evidence}) != 1:
        raise ValueError("nvidia-smi identity rows must report one driver version")
    return tuple(evidence)


def parse_matched_nvidia_smi_monitor_csv(
    payload: str,
    *,
    expected_uuids: Sequence[str],
) -> tuple[MatchedNvidiaSmiResourceSample, ...]:
    """Parse and order one complete eight-GPU resource sample by UUID."""

    canonical_expected = tuple(
        _canonical_gpu_uuid(value, label=f"expected GPU {index} UUID")
        for index, value in enumerate(expected_uuids)
    )
    if (
        len(canonical_expected) != _MATCHED_GPU_COUNT
        or len({value.lower() for value in canonical_expected}) != _MATCHED_GPU_COUNT
    ):
        raise ValueError("expected monitor UUID inventory must contain eight unique GPUs")
    if not payload.strip() or "\x00" in payload:
        raise ValueError("nvidia-smi monitor CSV must be non-empty text without NUL")
    rows = tuple(csv.reader(io.StringIO(payload), skipinitialspace=True))
    if len(rows) != _MATCHED_GPU_COUNT:
        raise ValueError("nvidia-smi monitor CSV must contain exactly eight GPU rows")
    samples: list[MatchedNvidiaSmiResourceSample] = []
    for index, row in enumerate(rows):
        if len(row) != 3 or any(not value.strip() for value in row):
            raise ValueError(f"nvidia-smi monitor row {index} must have three populated fields")
        samples.append(
            MatchedNvidiaSmiResourceSample(
                uuid=_canonical_gpu_uuid(row[0].strip(), label=f"monitor row {index} UUID"),
                memory_used_mib=_parse_exact_decimal(
                    row[1].strip(), label=f"monitor row {index} memory"
                ),
                utilization_percent=_parse_exact_decimal(
                    row[2].strip(), label=f"monitor row {index} utilization"
                ),
            )
        )
    by_uuid = {sample.uuid.lower(): sample for sample in samples}
    if len(by_uuid) != _MATCHED_GPU_COUNT or set(by_uuid) != {
        value.lower() for value in canonical_expected
    }:
        raise ValueError("nvidia-smi monitor UUID inventory drifted")
    return tuple(by_uuid[value.lower()] for value in canonical_expected)


class MatchedCapacityInputs(StrictFrozenModel):
    """Observed GPU-memory inputs used by the matched capacity calculation."""

    schema_version: Literal["inkling-matched-capacity-inputs-v1"]
    gpu_count: Literal[8]
    observed_gpu_memory_bytes: tuple[StrictInt, ...] = Field(min_length=8, max_length=8)
    observed_total_gpu_memory_bytes: StrictInt = Field(gt=0)

    @model_validator(mode="after")
    def capacity_is_complete(self) -> MatchedCapacityInputs:
        if any(value < 274_113 * 1024 * 1024 for value in self.observed_gpu_memory_bytes):
            raise ValueError("matched capacity input reports insufficient B300 memory")
        if self.observed_total_gpu_memory_bytes != sum(self.observed_gpu_memory_bytes):
            raise ValueError("matched total GPU memory differs from per-GPU observations")
        return self


def build_matched_capacity_inputs(
    gpus: Sequence[MatchedNvidiaSmiGpuEvidence],
) -> MatchedCapacityInputs:
    """Build exact capacity inputs from the established ordered GPU inventory."""

    gpu_tuple = tuple(gpus)
    if tuple(gpu.cuda_ordinal for gpu in gpu_tuple) != tuple(range(_MATCHED_GPU_COUNT)):
        raise ValueError(
            "matched GPU capacity inventory must use exact ordinals zero through seven"
        )
    memory = tuple(gpu.memory_total_mib * 1024 * 1024 for gpu in gpu_tuple)
    return MatchedCapacityInputs(
        schema_version="inkling-matched-capacity-inputs-v1",
        gpu_count=_MATCHED_GPU_COUNT,
        observed_gpu_memory_bytes=memory,
        observed_total_gpu_memory_bytes=sum(memory),
    )


class MatchedCudaPeerEdgeEvidence(StrictFrozenModel):
    """One directed CUDA peer-capability edge in the eight-GPU cell."""

    source_cuda_ordinal: StrictInt = Field(ge=0, le=7)
    source_uuid: StrictStr = Field(pattern=_GPU_UUID_PATTERN)
    destination_cuda_ordinal: StrictInt = Field(ge=0, le=7)
    destination_uuid: StrictStr = Field(pattern=_GPU_UUID_PATTERN)
    can_access_peer: StrictBool
    performance_rank: StrictInt = Field(ge=0)
    access_supported: StrictBool
    native_atomic_supported: StrictBool
    cuda_array_access_supported: StrictBool
    only_partial_native_atomic_supported: StrictBool

    @model_validator(mode="after")
    def directed_edge_is_consistent(self) -> MatchedCudaPeerEdgeEvidence:
        if (
            self.source_cuda_ordinal == self.destination_cuda_ordinal
            or self.source_uuid.lower() == self.destination_uuid.lower()
        ):
            raise ValueError("CUDA peer edge must join two different GPUs")
        if self.can_access_peer != self.access_supported:
            raise ValueError("CUDA peer access queries disagree")
        if self.native_atomic_supported and self.only_partial_native_atomic_supported:
            raise ValueError("CUDA peer atomic support cannot be both full and partial")
        return self


class MatchedCudaPeerTopologyEvidence(StrictFrozenModel):
    """All 56 process-visible directed peer edges for the exact eight GPUs."""

    schema_version: Literal["inkling-matched-cuda-peer-topology-v1"]
    protocol: Literal["cuda-driver-p2p-attributes-v1"]
    cuda_driver_api_version: StrictInt = Field(gt=0)
    gpu_uuids: tuple[StrictStr, ...] = Field(min_length=8, max_length=8)
    edges: tuple[MatchedCudaPeerEdgeEvidence, ...] = Field(min_length=56, max_length=56)

    @field_validator("gpu_uuids")
    @classmethod
    def uuids_are_canonical(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        canonical = tuple(
            _canonical_gpu_uuid(value, label=f"topology GPU {index} UUID")
            for index, value in enumerate(values)
        )
        if len({value.lower() for value in canonical}) != _MATCHED_GPU_COUNT:
            raise ValueError("CUDA peer topology GPU UUIDs must be unique")
        return canonical

    @model_validator(mode="after")
    def all_directed_edges_are_exact(self) -> MatchedCudaPeerTopologyEvidence:
        expected_pairs = tuple(
            (source, destination)
            for source in range(_MATCHED_GPU_COUNT)
            for destination in range(_MATCHED_GPU_COUNT)
            if source != destination
        )
        observed_pairs = tuple(
            (edge.source_cuda_ordinal, edge.destination_cuda_ordinal) for edge in self.edges
        )
        if len(self.edges) != 56 or observed_pairs != expected_pairs:
            raise ValueError("CUDA peer topology must contain all 56 ordered directed edges")
        for edge in self.edges:
            if (
                edge.source_uuid.lower() != self.gpu_uuids[edge.source_cuda_ordinal].lower()
                or edge.destination_uuid.lower()
                != self.gpu_uuids[edge.destination_cuda_ordinal].lower()
            ):
                raise ValueError("CUDA peer edge UUIDs differ from the ordered GPU inventory")
        return self


class _MatchedCudaUuid(ctypes.Structure):
    _fields_ = [("value", ctypes.c_ubyte * 16)]


def _matched_cuda_function(library: Any, name: str, argtypes: list[object]) -> Any:
    try:
        function = getattr(library, name)
    except AttributeError as error:
        raise RuntimeError(f"CUDA driver lacks required function {name}") from error
    function.argtypes = argtypes
    function.restype = ctypes.c_int
    return function


def _require_matched_cuda_success(
    function: Any,
    *arguments: object,
    label: str,
) -> None:
    result = int(function(*arguments))
    if result != 0:
        raise RuntimeError(f"CUDA driver {label} failed with error code {result}")


def _load_matched_cuda_driver(cuda_driver_library_path: str) -> Any:
    canonical_path = _canonical_absolute_path(
        cuda_driver_library_path,
        label="CUDA driver library path",
    )
    path = PurePosixPath(canonical_path)
    if (
        path in (_MATCHED_CUDA_DRIVER_STUB_ROOT, _MATCHED_CUDA_DRIVER_LINK_ROOT)
        or _MATCHED_CUDA_DRIVER_STUB_ROOT in path.parents
        or _MATCHED_CUDA_DRIVER_LINK_ROOT in path.parents
    ):
        raise ValueError("CUDA driver library path must not resolve through a build stub")
    try:
        return ctypes.CDLL(canonical_path)
    except OSError as error:
        raise RuntimeError("CUDA driver library could not be loaded") from error


def _matched_cuda_binary_value(value: int, *, label: str) -> bool:
    if value not in (0, 1):
        raise RuntimeError(f"CUDA driver {label} must be zero or one")
    return bool(value)


def order_matched_nvidia_smi_identity_by_cuda_uuid(
    gpus: Sequence[MatchedNvidiaSmiGpuEvidence],
    *,
    cuda_gpu_uuids: Sequence[str],
) -> tuple[MatchedNvidiaSmiGpuEvidence, ...]:
    """Join nvidia-smi rows by UUID and return them in CUDA ordinal order."""

    gpu_tuple = tuple(gpus)
    cuda_uuids = tuple(
        _canonical_gpu_uuid(value, label=f"CUDA device {ordinal} UUID")
        for ordinal, value in enumerate(cuda_gpu_uuids)
    )
    if len(gpu_tuple) != _MATCHED_GPU_COUNT:
        raise ValueError("nvidia-smi identity must contain exactly eight GPUs")
    if (
        len(cuda_uuids) != _MATCHED_GPU_COUNT
        or len({value.lower() for value in cuda_uuids}) != _MATCHED_GPU_COUNT
    ):
        raise ValueError("CUDA Driver UUID inventory must contain eight unique GPUs")
    if any(not isinstance(gpu, MatchedNvidiaSmiGpuEvidence) for gpu in gpu_tuple):
        raise TypeError("nvidia-smi identity contains an invalid GPU record")

    by_uuid = {gpu.uuid.lower(): gpu for gpu in gpu_tuple}
    if len(by_uuid) != _MATCHED_GPU_COUNT or set(by_uuid) != {
        value.lower() for value in cuda_uuids
    }:
        raise RuntimeError("CUDA Driver UUID inventory differs from nvidia-smi identity")

    return tuple(
        MatchedNvidiaSmiGpuEvidence(
            cuda_ordinal=ordinal,
            uuid=cuda_uuid,
            name=by_uuid[cuda_uuid.lower()].name,
            memory_total_mib=by_uuid[cuda_uuid.lower()].memory_total_mib,
            driver_version=by_uuid[cuda_uuid.lower()].driver_version,
            compute_capability=by_uuid[cuda_uuid.lower()].compute_capability,
        )
        for ordinal, cuda_uuid in enumerate(cuda_uuids)
    )


def enumerate_matched_cuda_peer_topology(
    cuda_driver_library_path: str,
    *,
    nvidia_smi_gpus: Sequence[MatchedNvidiaSmiGpuEvidence],
) -> MatchedCudaPeerTopologyEvidence:
    """Enumerate all directed peer capabilities for one exact eight-GPU cell.

    CUDA Driver enumeration establishes process-visible ordinals. The parsed
    nvidia-smi inventory is joined only by full UUID, so nvidia-smi row order
    cannot redefine CUDA ordinals. This evidence does not prove the physical
    NVLink, NVSwitch, PCIe, or other interconnect fabric.
    """

    library = _load_matched_cuda_driver(cuda_driver_library_path)
    cu_init = _matched_cuda_function(library, "cuInit", [ctypes.c_uint])
    cu_driver_get_version = _matched_cuda_function(
        library,
        "cuDriverGetVersion",
        [ctypes.POINTER(ctypes.c_int)],
    )
    cu_device_get_count = _matched_cuda_function(
        library,
        "cuDeviceGetCount",
        [ctypes.POINTER(ctypes.c_int)],
    )
    cu_device_get = _matched_cuda_function(
        library,
        "cuDeviceGet",
        [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
    )
    cu_device_get_uuid = _matched_cuda_function(
        library,
        "cuDeviceGetUuid_v2",
        [ctypes.POINTER(_MatchedCudaUuid), ctypes.c_int],
    )
    cu_device_can_access_peer = _matched_cuda_function(
        library,
        "cuDeviceCanAccessPeer",
        [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
    )
    cu_device_get_p2p_attribute = _matched_cuda_function(
        library,
        "cuDeviceGetP2PAttribute",
        [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int],
    )

    _require_matched_cuda_success(cu_init, 0, label="initialization")
    driver_api_version = ctypes.c_int()
    _require_matched_cuda_success(
        cu_driver_get_version,
        ctypes.byref(driver_api_version),
        label="API-version query",
    )
    if driver_api_version.value <= 0:
        raise RuntimeError("CUDA driver API version is invalid")

    device_count = ctypes.c_int()
    _require_matched_cuda_success(
        cu_device_get_count,
        ctypes.byref(device_count),
        label="device-count query",
    )
    if device_count.value != _MATCHED_GPU_COUNT:
        raise RuntimeError("CUDA driver must expose exactly eight GPUs")

    devices: list[ctypes.c_int] = []
    cuda_uuids: list[str] = []
    for ordinal in range(_MATCHED_GPU_COUNT):
        device = ctypes.c_int()
        _require_matched_cuda_success(
            cu_device_get,
            ctypes.byref(device),
            ordinal,
            label=f"device {ordinal} lookup",
        )
        raw_uuid = _MatchedCudaUuid()
        _require_matched_cuda_success(
            cu_device_get_uuid,
            ctypes.byref(raw_uuid),
            device,
            label=f"device {ordinal} UUID query",
        )
        devices.append(device)
        cuda_uuids.append(
            _canonical_gpu_uuid(
                f"GPU-{UUID(bytes=bytes(raw_uuid.value))}",
                label=f"CUDA device {ordinal} UUID",
            )
        )

    ordered_nvidia_gpus = order_matched_nvidia_smi_identity_by_cuda_uuid(
        nvidia_smi_gpus,
        cuda_gpu_uuids=cuda_uuids,
    )
    ordered_uuids = tuple(gpu.uuid for gpu in ordered_nvidia_gpus)
    attribute_queries = (
        ("performance-rank", _MATCHED_CUDA_P2P_PERFORMANCE_RANK_ATTRIBUTE),
        ("access-supported", _MATCHED_CUDA_P2P_ACCESS_SUPPORTED_ATTRIBUTE),
        (
            "native-atomic-supported",
            _MATCHED_CUDA_P2P_NATIVE_ATOMIC_SUPPORTED_ATTRIBUTE,
        ),
        (
            "CUDA-array-access-supported",
            _MATCHED_CUDA_P2P_CUDA_ARRAY_ACCESS_SUPPORTED_ATTRIBUTE,
        ),
        (
            "only-partial-native-atomic-supported",
            _MATCHED_CUDA_P2P_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED_ATTRIBUTE,
        ),
    )

    edges: list[MatchedCudaPeerEdgeEvidence] = []
    for source_ordinal in range(_MATCHED_GPU_COUNT):
        for destination_ordinal in range(_MATCHED_GPU_COUNT):
            if source_ordinal == destination_ordinal:
                continue
            can_access_raw = ctypes.c_int()
            _require_matched_cuda_success(
                cu_device_can_access_peer,
                ctypes.byref(can_access_raw),
                devices[source_ordinal],
                devices[destination_ordinal],
                label=f"peer {source_ordinal}->{destination_ordinal} access query",
            )
            can_access = _matched_cuda_binary_value(
                can_access_raw.value,
                label=f"peer {source_ordinal}->{destination_ordinal} access result",
            )

            attributes: dict[int, int] = {}
            for attribute_label, attribute in attribute_queries:
                raw_attribute = ctypes.c_int()
                _require_matched_cuda_success(
                    cu_device_get_p2p_attribute,
                    ctypes.byref(raw_attribute),
                    attribute,
                    devices[source_ordinal],
                    devices[destination_ordinal],
                    label=(f"peer {source_ordinal}->{destination_ordinal} {attribute_label} query"),
                )
                attributes[attribute] = raw_attribute.value

            performance_rank = attributes[_MATCHED_CUDA_P2P_PERFORMANCE_RANK_ATTRIBUTE]
            if performance_rank < 0:
                raise RuntimeError("CUDA driver peer performance rank must be non-negative")
            access_supported = _matched_cuda_binary_value(
                attributes[_MATCHED_CUDA_P2P_ACCESS_SUPPORTED_ATTRIBUTE],
                label=(f"peer {source_ordinal}->{destination_ordinal} access-supported result"),
            )
            if can_access != access_supported:
                raise RuntimeError(
                    f"CUDA driver peer {source_ordinal}->{destination_ordinal} "
                    "access queries disagree"
                )
            native_atomic_supported = _matched_cuda_binary_value(
                attributes[_MATCHED_CUDA_P2P_NATIVE_ATOMIC_SUPPORTED_ATTRIBUTE],
                label=(
                    f"peer {source_ordinal}->{destination_ordinal} native-atomic-supported result"
                ),
            )
            cuda_array_access_supported = _matched_cuda_binary_value(
                attributes[_MATCHED_CUDA_P2P_CUDA_ARRAY_ACCESS_SUPPORTED_ATTRIBUTE],
                label=(
                    f"peer {source_ordinal}->{destination_ordinal} "
                    "CUDA-array-access-supported result"
                ),
            )
            only_partial_native_atomic_supported = _matched_cuda_binary_value(
                attributes[_MATCHED_CUDA_P2P_ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED_ATTRIBUTE],
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
            edges.append(
                MatchedCudaPeerEdgeEvidence(
                    source_cuda_ordinal=source_ordinal,
                    source_uuid=ordered_uuids[source_ordinal],
                    destination_cuda_ordinal=destination_ordinal,
                    destination_uuid=ordered_uuids[destination_ordinal],
                    can_access_peer=can_access,
                    performance_rank=performance_rank,
                    access_supported=access_supported,
                    native_atomic_supported=native_atomic_supported,
                    cuda_array_access_supported=cuda_array_access_supported,
                    only_partial_native_atomic_supported=(only_partial_native_atomic_supported),
                )
            )

    return MatchedCudaPeerTopologyEvidence(
        schema_version="inkling-matched-cuda-peer-topology-v1",
        protocol="cuda-driver-p2p-attributes-v1",
        cuda_driver_api_version=driver_api_version.value,
        gpu_uuids=ordered_uuids,
        edges=tuple(edges),
    )


MatchedArtifactKind: TypeAlias = Literal[
    "text_shard",
    "projector",
    "receipt",
    "manifest",
    "tokenizer",
]


class MatchedArtifactHashObservation(StrictFrozenModel):
    """One expected-to-observed artifact hash and size comparison."""

    subject: MatchedSubject
    kind: MatchedArtifactKind
    relative_path: StrictStr = Field(min_length=1)
    absolute_path: StrictStr = Field(min_length=1)
    shard_ordinal: StrictInt | None = Field(default=None, ge=1, le=49)
    expected_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    observed_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    expected_size_bytes: StrictInt = Field(gt=0)
    observed_size_bytes: StrictInt = Field(gt=0)
    hash_matches: Literal[True]
    size_matches: Literal[True]

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_canonical(cls, value: str) -> str:
        return _canonical_relative_path(value, label="artifact relative path")

    @field_validator("absolute_path")
    @classmethod
    def absolute_path_is_canonical(cls, value: str) -> str:
        return _canonical_absolute_path(value, label="artifact absolute path")

    @model_validator(mode="after")
    def observation_matches(self) -> MatchedArtifactHashObservation:
        if not self.absolute_path.endswith(f"/{self.relative_path}"):
            raise ValueError("artifact absolute and relative paths differ")
        if self.expected_sha256 != self.observed_sha256:
            raise ValueError("artifact hash does not match its expected SHA-256")
        if self.expected_size_bytes != self.observed_size_bytes:
            raise ValueError("artifact size does not match its expected byte count")
        if self.kind == "text_shard":
            if self.shard_ordinal is None:
                raise ValueError("text shard is missing its ordinal")
            expected_path = _subject_shard_relative_path(self.subject, self.shard_ordinal)
            if self.relative_path != expected_path:
                raise ValueError("text shard path differs from its subject and ordinal")
        elif self.shard_ordinal is not None:
            raise ValueError("non-shard artifact cannot have a shard ordinal")
        if self.kind == "projector" and (
            self.subject is not MatchedSubject.Q3
            or self.relative_path != _MATCHED_PROJECTOR_RELATIVE_PATH
        ):
            raise ValueError("shared projector must be the verified Q3 export projector")
        return self


def matched_shard_inventory_sha256(
    subject: MatchedSubject,
    shards: Sequence[MatchedArtifactHashObservation],
) -> str:
    """Hash the exact ordered 49-shard inventory with its source convention."""

    shard_tuple = tuple(shards)
    if len(shard_tuple) != 49:
        raise ValueError("matched shard inventory must contain exactly 49 shards")
    for ordinal, shard in enumerate(shard_tuple, start=1):
        if (
            shard.subject is not subject
            or shard.kind != "text_shard"
            or shard.shard_ordinal != ordinal
            or shard.relative_path != _subject_shard_relative_path(subject, ordinal)
        ):
            raise ValueError("matched shard inventory order or identity drifted")
    rows = [
        {
            "path": shard.relative_path,
            "sha256": shard.observed_sha256,
            "size_bytes": shard.observed_size_bytes,
        }
        for shard in shard_tuple
    ]
    payload = _canonical_json_bytes(rows)
    # Preserve the source receipts' byte domains: BF16 hashed the canonical
    # inventory with one trailing newline; Q3 hashed the same form without one.
    if subject is MatchedSubject.BF16:
        payload += b"\n"
    return hashlib.sha256(payload).hexdigest()


class MatchedSubjectArtifactRehashEvidence(StrictFrozenModel):
    """Complete pre-load artifact rehash for one matched subject."""

    schema_version: Literal["inkling-matched-artifact-rehash-v1"]
    subject: MatchedSubject
    subject_reference_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    assignments: tuple[MatchedArtifactHashObservation, ...] = Field(min_length=1)
    assignment_count: StrictInt = Field(gt=0)
    text_shard_count: Literal[49]
    text_shard_total_bytes: StrictInt = Field(gt=0)
    expected_text_shard_inventory_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    observed_text_shard_inventory_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    first_shard_path: StrictStr = Field(min_length=1)
    metadata_only_first_shard: Literal[True]
    shared_projector: MatchedArtifactHashObservation
    rehash_completed: Literal[True]
    all_hashes_match: Literal[True]

    @field_validator("first_shard_path")
    @classmethod
    def first_shard_path_is_canonical(cls, value: str) -> str:
        return _canonical_absolute_path(value, label="first shard path")

    @model_validator(mode="after")
    def complete_assignment_contract(self) -> MatchedSubjectArtifactRehashEvidence:
        expected_count = 50 if self.subject is MatchedSubject.BF16 else 60
        if (
            self.assignment_count != len(self.assignments)
            or self.assignment_count != expected_count
        ):
            raise ValueError(
                f"{self.subject.value} artifact assignment must contain {expected_count}"
            )
        if any(item.subject is not self.subject for item in self.assignments):
            raise ValueError("artifact assignment contains a different comparison subject")
        relative_paths = tuple(item.relative_path for item in self.assignments)
        absolute_paths = tuple(item.absolute_path for item in self.assignments)
        if len(set(relative_paths)) != expected_count or len(set(absolute_paths)) != expected_count:
            raise ValueError("artifact assignment paths must be unique")

        shards = tuple(item for item in self.assignments if item.kind == "text_shard")
        if len(shards) != self.text_shard_count:
            raise ValueError("artifact assignment must contain exactly 49 text shards")
        if self.text_shard_total_bytes != sum(item.observed_size_bytes for item in shards):
            raise ValueError("text shard total bytes differ from the rehash assignments")
        inventory_sha256 = matched_shard_inventory_sha256(self.subject, shards)
        if (
            self.expected_text_shard_inventory_sha256 != inventory_sha256
            or self.observed_text_shard_inventory_sha256 != inventory_sha256
        ):
            raise ValueError("text shard inventory hash differs from the exact assignments")
        if self.first_shard_path != shards[0].absolute_path:
            raise ValueError("first shard path differs from the metadata-only first assignment")

        kind_counts = {
            kind: sum(item.kind == kind for item in self.assignments)
            for kind in ("projector", "receipt", "manifest", "tokenizer")
        }
        expected_kind_counts = (
            {"projector": 0, "receipt": 1, "manifest": 0, "tokenizer": 0}
            if self.subject is MatchedSubject.BF16
            else {"projector": 1, "receipt": 3, "manifest": 1, "tokenizer": 6}
        )
        if kind_counts != expected_kind_counts:
            raise ValueError("artifact assignment kind cardinality differs from its subject")
        if self.shared_projector.kind != "projector":
            raise ValueError("shared projector evidence is not a projector artifact")
        if self.subject is MatchedSubject.Q3:
            projector = tuple(item for item in self.assignments if item.kind == "projector")
            if projector != (self.shared_projector,):
                raise ValueError("Q3 assignment does not contain the exact shared projector")
        return self


class MatchedProbeTrialEvidence(StrictFrozenModel):
    """Safe actual evidence from one deterministic probe trial."""

    trial_index: StrictInt = Field(ge=1, le=2)
    token_ids: tuple[StrictInt, ...] = Field(min_length=1, max_length=8)
    generated_token_count: StrictInt = Field(ge=1, le=8)
    minimum_logprob: FiniteFloat
    maximum_logprob: FiniteFloat
    mean_logprob: FiniteFloat
    prompt_processing_ms: FiniteFloat = Field(gt=0)
    decode_ms: FiniteFloat = Field(gt=0)
    response_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    finite_logits: Literal[True]
    valid_token_ids: Literal[True]

    @model_validator(mode="after")
    def summary_matches_tokens(self) -> MatchedProbeTrialEvidence:
        if self.generated_token_count != len(self.token_ids):
            raise ValueError("generated token count differs from retained token IDs")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("generated token IDs must be non-negative")
        if not self.minimum_logprob <= self.mean_logprob <= self.maximum_logprob:
            raise ValueError("mean logprob is outside the retained finite range")
        return self


class MatchedProbeEvidence(StrictFrozenModel):
    """Two repeatable greedy trials without raw prompt or output text."""

    probe_id: StrictStr = Field(min_length=1, pattern=r"^[a-z0-9_]+$")
    modality: Literal["text", "image", "audio"]
    prompt_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    fixture_sha256: StrictStr | None = Field(default=None, pattern=_SHA256_PATTERN)
    fixture_size_bytes: StrictInt | None = Field(default=None, gt=0)
    seed: Literal[42]
    temperature: FiniteFloat = Field(ge=0.0, le=0.0)
    n_predict: Literal[8]
    n_probs: Literal[5]
    usable_vocab_size: Literal[200058]
    trials: tuple[MatchedProbeTrialEvidence, MatchedProbeTrialEvidence]
    repeatable_greedy_token_ids: Literal[True]
    prompt_text_recorded: Literal[False]
    output_text_recorded: Literal[False]

    @model_validator(mode="after")
    def deterministic_probe_is_complete(self) -> MatchedProbeEvidence:
        if self.probe_id != f"{self.modality}_greedy_v1":
            raise ValueError("probe identifier differs from its modality")
        if self.modality == "text":
            if self.fixture_sha256 is not None or self.fixture_size_bytes is not None:
                raise ValueError("text probe cannot retain fixture evidence")
        elif self.fixture_sha256 is None or self.fixture_size_bytes is None:
            raise ValueError("multimodal probe requires hashed fixture evidence")
        if tuple(trial.trial_index for trial in self.trials) != (1, 2):
            raise ValueError("probe must retain exact trial indices one and two")
        if self.trials[0].token_ids != self.trials[1].token_ids:
            raise ValueError("repeatable greedy token IDs differ between trials")
        if any(
            token_id >= self.usable_vocab_size
            for trial in self.trials
            for token_id in trial.token_ids
        ):
            raise ValueError("probe contains a token ID outside the usable vocabulary")
        return self


class MatchedGpuResourceEvidence(StrictFrozenModel):
    """Peak host and per-GPU resources observed during one subject."""

    schema_version: Literal["inkling-matched-resource-evidence-v1"]
    sampling_interval_seconds: FiniteFloat = Field(ge=1.0, le=1.0)
    sample_count: StrictInt = Field(ge=2)
    server_peak_host_rss_mib: StrictInt = Field(gt=0)
    gpu_peak_memory_used_mib: tuple[StrictInt, ...] = Field(min_length=8, max_length=8)
    gpu_peak_utilization_percent: tuple[StrictInt, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def all_gpu_peaks_are_valid(self) -> MatchedGpuResourceEvidence:
        if any(value <= 0 for value in self.gpu_peak_memory_used_mib):
            raise ValueError("resource evidence must observe memory use on all eight GPUs")
        if any(not 0 <= value <= 100 for value in self.gpu_peak_utilization_percent):
            raise ValueError("GPU utilization percentages must be from zero through 100")
        return self


class MatchedServerCleanupEvidence(StrictFrozenModel):
    """Ordered cleanup proof for one isolated llama-server process."""

    monitor_stopped_before_server: Literal[True]
    monitor_joined: Literal[True]
    server_stopped: Literal[True]
    process_exit_observed: Literal[True]
    log_scan_completed: Literal[True]


def matched_subject_smoke_receipt_sha256(payload: Mapping[str, object]) -> str:
    """Return the self-hash for one subject success receipt."""

    return _receipt_sha256(_MATCHED_SUBJECT_RECEIPT_DOMAIN, payload)


class MatchedSubjectSmokeReceipt(StrictFrozenModel):
    """Immutable terminal success receipt for one comparison subject."""

    schema_version: Literal["inkling-matched-subject-smoke-v1"]
    status: Literal["passed"]
    stage: Literal["matched_smoke"]
    run_id: StrictStr = Field(min_length=1, max_length=200)
    subject: MatchedSubject
    subject_ordinal: StrictInt = Field(ge=0, le=1)
    allocation_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    runtime_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    probe_control_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    server_process_id: StrictInt = Field(gt=0)
    server_command: tuple[StrictStr, ...] = Field(min_length=1, max_length=64)
    server_log_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    load_time_seconds: FiniteFloat = Field(gt=0)
    artifact_rehash: MatchedSubjectArtifactRehashEvidence
    loader_offload: LoaderOffloadEvidence
    artifact_load: ArtifactLoadEvidence
    raw_logit_audit: RawLogitAuditEvidence
    backend_audit: ExactCudaBackendAuditEvidence
    probes: tuple[MatchedProbeEvidence, MatchedProbeEvidence, MatchedProbeEvidence]
    resources: MatchedGpuResourceEvidence
    cleanup: MatchedServerCleanupEvidence
    prompt_text_recorded: Literal[False]
    output_text_recorded: Literal[False]
    raw_server_log_recorded: Literal[False]
    quality_measured: Literal[False]
    benchmark_measured: Literal[False]
    completed_at_utc: StrictStr = Field(min_length=1)
    receipt_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("completed_at_utc")
    @classmethod
    def completion_is_utc(cls, value: str) -> str:
        return _completed_at_utc(value)

    @model_validator(mode="after")
    def subject_success_is_exact(self) -> MatchedSubjectSmokeReceipt:
        if self.subject_ordinal != _subject_ordinal(self.subject):
            raise ValueError("subject ordinal differs from the fixed execution order")
        if self.artifact_rehash.subject is not self.subject:
            raise ValueError("subject receipt and artifact rehash subjects differ")
        if self.loader_offload.cuda_device_count != _MATCHED_GPU_COUNT:
            raise ValueError("subject loader evidence does not prove the exact eight-GPU cell")
        if self.artifact_load.first_shard_path != self.artifact_rehash.first_shard_path:
            raise ValueError("subject artifact-load evidence binds a different first shard")
        if self.artifact_load.projector_path != self.artifact_rehash.shared_projector.absolute_path:
            raise ValueError("subject artifact-load evidence binds a different projector")
        expected_generated_vectors = sum(
            trial.generated_token_count for probe in self.probes for trial in probe.trials
        )
        if self.raw_logit_audit.expected_generated_token_vectors != expected_generated_vectors:
            raise ValueError(
                "subject raw-logit evidence does not cover every generated token vector"
            )
        if (
            self.raw_logit_audit.vocab_size != PINNED_VOCAB_SIZE
            or self.raw_logit_audit.unpadded_vocab_size != PINNED_UNPADDED_VOCAB_SIZE
            or any(
                probe.usable_vocab_size != self.raw_logit_audit.unpadded_vocab_size
                for probe in self.probes
            )
        ):
            raise ValueError(
                "subject raw-logit and probe evidence differ from the exact vocabulary"
            )
        expected_spec = MatchedServerCommandSpec(
            schema_version="inkling-matched-server-command-v1",
            subject=self.subject,
            server_binary=_MATCHED_SERVER_BINARY,
            first_shard_path=self.artifact_rehash.first_shard_path,
            projector_path=self.artifact_rehash.shared_projector.absolute_path,
            host="127.0.0.1",
            port=_subject_server_port(self.subject),
            server_log_path=_subject_server_log_path(self.subject),
            endpoint="/completion",
            log_verbosity=4,
            context_size=8192,
        )
        if self.server_command != build_matched_server_command(expected_spec):
            raise ValueError("subject server command differs from the exact eight-GPU command")
        if (
            self.backend_audit.policy.gpu_count != _MATCHED_GPU_COUNT
            or self.backend_audit.policy.tensor_split != (1,) * _MATCHED_GPU_COUNT
        ):
            raise ValueError("subject backend audit does not prove exact eight-GPU placement")
        if tuple(probe.modality for probe in self.probes) != ("text", "image", "audio"):
            raise ValueError("subject probes must be retained in text, image, audio order")
        expected_hash = matched_subject_smoke_receipt_sha256(
            self.model_dump(mode="python", exclude={"receipt_sha256"})
        )
        if self.receipt_sha256 != expected_hash:
            raise ValueError("subject success receipt hash does not match its content")
        return self


class MatchedSubjectReceiptReference(StrictFrozenModel):
    """Safe terminal reference to one independently persisted subject receipt."""

    subject: MatchedSubject
    subject_ordinal: StrictInt = Field(ge=0, le=1)
    receipt_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    server_process_id: StrictInt = Field(gt=0)
    allocation_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    runtime_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    probe_control_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    projector_path: StrictStr = Field(min_length=1)
    projector_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def order_is_exact(self) -> MatchedSubjectReceiptReference:
        if self.subject_ordinal != _subject_ordinal(self.subject):
            raise ValueError("subject receipt reference ordinal is wrong")
        return self


def _subject_receipt_reference(
    receipt: MatchedSubjectSmokeReceipt,
) -> MatchedSubjectReceiptReference:
    projector = receipt.artifact_rehash.shared_projector
    return MatchedSubjectReceiptReference(
        subject=receipt.subject,
        subject_ordinal=receipt.subject_ordinal,
        receipt_sha256=receipt.receipt_sha256,
        server_process_id=receipt.server_process_id,
        allocation_identity_sha256=receipt.allocation_identity_sha256,
        runtime_identity_sha256=receipt.runtime_identity_sha256,
        probe_control_sha256=receipt.probe_control_sha256,
        projector_path=projector.absolute_path,
        projector_sha256=projector.observed_sha256,
    )


def matched_rollup_receipt_sha256(payload: Mapping[str, object]) -> str:
    """Return the self-hash for the terminal two-subject rollup."""

    return _receipt_sha256(_MATCHED_ROLLUP_RECEIPT_DOMAIN, payload)


class MatchedRollupReceipt(StrictFrozenModel):
    """Terminal proof that both subjects passed the same exact allocation."""

    schema_version: Literal["inkling-matched-rollup-v1"]
    status: Literal["passed"]
    stage: Literal["matched_smoke"]
    run_id: StrictStr = Field(min_length=1, max_length=200)
    subjects: tuple[MatchedSubjectReceiptReference, MatchedSubjectReceiptReference]
    allocation_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    runtime_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    probe_control_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    both_subjects_passed: Literal[True]
    same_allocation: Literal[True]
    same_runtime: Literal[True]
    same_probe_control: Literal[True]
    fresh_server_processes: Literal[True]
    prompt_text_recorded: Literal[False]
    output_text_recorded: Literal[False]
    quality_measured: Literal[False]
    benchmark_measured: Literal[False]
    completed_at_utc: StrictStr = Field(min_length=1)
    receipt_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("completed_at_utc")
    @classmethod
    def completion_is_utc(cls, value: str) -> str:
        return _completed_at_utc(value)

    @model_validator(mode="after")
    def rollup_is_exact(self) -> MatchedRollupReceipt:
        if tuple(reference.subject for reference in self.subjects) != MATCHED_SUBJECT_ORDER:
            raise ValueError("rollup subject order must be BF16 followed by Q3")
        if tuple(reference.subject_ordinal for reference in self.subjects) != (0, 1):
            raise ValueError("rollup subject ordinals differ from the fixed order")
        if len({reference.server_process_id for reference in self.subjects}) != 2:
            raise ValueError("rollup subjects must use fresh server process IDs")
        for field_name in (
            "allocation_identity_sha256",
            "runtime_identity_sha256",
            "probe_control_sha256",
        ):
            expected = getattr(self, field_name)
            if any(getattr(reference, field_name) != expected for reference in self.subjects):
                raise ValueError(f"rollup subject {field_name} values differ")
        first, second = self.subjects
        if (
            first.projector_path != second.projector_path
            or first.projector_sha256 != second.projector_sha256
        ):
            raise ValueError("rollup subjects did not use the same shared projector")
        expected_hash = matched_rollup_receipt_sha256(
            self.model_dump(mode="python", exclude={"receipt_sha256"})
        )
        if self.receipt_sha256 != expected_hash:
            raise ValueError("matched rollup receipt hash does not match its content")
        return self


def build_matched_rollup_receipt(
    *,
    run_id: str,
    subject_receipts: Sequence[MatchedSubjectSmokeReceipt],
    completed_at_utc: str,
) -> MatchedRollupReceipt:
    """Build the terminal matched rollup after validating both success receipts."""

    receipts = tuple(subject_receipts)
    if (
        len(receipts) != 2
        or tuple(receipt.subject for receipt in receipts) != MATCHED_SUBJECT_ORDER
    ):
        raise ValueError("matched rollup requires BF16 then Q3 subject receipts")
    if any(receipt.run_id != run_id for receipt in receipts):
        raise ValueError("matched rollup subject run IDs differ")
    allocation_values = {receipt.allocation_identity_sha256 for receipt in receipts}
    if len(allocation_values) != 1:
        raise ValueError("matched rollup subject allocation identities differ")
    runtime_values = {receipt.runtime_identity_sha256 for receipt in receipts}
    if len(runtime_values) != 1:
        raise ValueError("matched rollup subject runtime identities differ")
    probe_values = {receipt.probe_control_sha256 for receipt in receipts}
    if len(probe_values) != 1:
        raise ValueError("matched rollup subject probe controls differ")
    if len({receipt.server_process_id for receipt in receipts}) != 2:
        raise ValueError("matched rollup requires fresh server processes")
    references = tuple(_subject_receipt_reference(receipt) for receipt in receipts)
    payload: dict[str, object] = {
        "schema_version": "inkling-matched-rollup-v1",
        "status": "passed",
        "stage": "matched_smoke",
        "run_id": run_id,
        "subjects": references,
        "allocation_identity_sha256": receipts[0].allocation_identity_sha256,
        "runtime_identity_sha256": receipts[0].runtime_identity_sha256,
        "probe_control_sha256": receipts[0].probe_control_sha256,
        "both_subjects_passed": True,
        "same_allocation": True,
        "same_runtime": True,
        "same_probe_control": True,
        "fresh_server_processes": True,
        "prompt_text_recorded": False,
        "output_text_recorded": False,
        "quality_measured": False,
        "benchmark_measured": False,
        "completed_at_utc": completed_at_utc,
    }
    payload["receipt_sha256"] = matched_rollup_receipt_sha256(payload)
    return MatchedRollupReceipt.model_validate(payload)


class MatchedPublicationState(StrictFrozenModel):
    """Monotonic publication state that suppresses ambiguous failure receipts."""

    state: Literal[
        "not_started",
        "success_files_installed",
        "commit_requested",
        "reloaded",
        "verified",
        "unknown",
    ]
    success_files_installed: StrictBool
    commit_requested: StrictBool
    volume_reloaded: StrictBool
    mounted_readback_verified: StrictBool
    terminal_success_proven: StrictBool
    failure_receipt_allowed: StrictBool

    @model_validator(mode="after")
    def lifecycle_is_monotonic(self) -> MatchedPublicationState:
        if self.commit_requested and not self.success_files_installed:
            raise ValueError("publication commit cannot precede installed success files")
        if self.volume_reloaded and not self.commit_requested:
            raise ValueError("publication reload cannot precede its commit")
        if self.mounted_readback_verified and not self.volume_reloaded:
            raise ValueError("publication readback cannot precede its reload")
        if self.terminal_success_proven and not self.mounted_readback_verified:
            raise ValueError("terminal success cannot precede mounted readback")

        expected: dict[str, tuple[bool, bool, bool, bool, bool, bool]] = {
            "not_started": (False, False, False, False, False, True),
            "success_files_installed": (True, False, False, False, False, False),
            "commit_requested": (True, True, False, False, False, False),
            "reloaded": (True, True, True, False, False, False),
            "verified": (True, True, True, True, True, False),
        }
        observed = (
            self.success_files_installed,
            self.commit_requested,
            self.volume_reloaded,
            self.mounted_readback_verified,
            self.terminal_success_proven,
            self.failure_receipt_allowed,
        )
        if self.state == "unknown":
            if self.failure_receipt_allowed or self.terminal_success_proven:
                raise ValueError("unknown publication state must suppress terminal receipts")
        elif observed != expected[self.state]:
            raise ValueError("publication state flags differ from the named lifecycle state")
        return self


MatchedFailureCategory: TypeAlias = Literal[
    "artifact_rehash",
    "hardware_identity",
    "hardware_capacity",
    "peer_topology",
    "server_start",
    "server_health",
    "probe",
    "backend_placement",
    "resource_monitor",
    "cleanup",
    "publication",
]


class MatchedFailureCauseCode(StrEnum):
    """Allowlisted reason retained in one sanitized matched-run failure."""

    ARTIFACT_CONTRACT_FAILED = "artifact_contract_failed"
    ARTIFACT_HASH_MISMATCH = "artifact_hash_mismatch"
    ARTIFACT_READ_FAILED = "artifact_read_failed"
    ARTIFACT_SIZE_MISMATCH = "artifact_size_mismatch"
    BACKEND_PLACEMENT_FAILED = "backend_placement_failed"
    CLEANUP_FAILED = "cleanup_failed"
    COMPLETION_CONTRACT_FAILED = "completion_contract_failed"
    DEADLINE_EXHAUSTED = "deadline_exhausted"
    GREEDY_REPEATABILITY_FAILED = "greedy_repeatability_failed"
    HARDWARE_CAPACITY_FAILED = "hardware_capacity_failed"
    HARDWARE_IDENTITY_FAILED = "hardware_identity_failed"
    PEER_TOPOLOGY_FAILED = "peer_topology_failed"
    PUBLICATION_FAILED = "publication_failed"
    RESOURCE_MONITOR_FAILED = "resource_monitor_failed"
    SERVER_HEALTH_FAILED = "server_health_failed"
    SERVER_START_FAILED = "server_start_failed"


class MatchedSanitizedFailureDiagnostic(StrictFrozenModel):
    """Bounded failure identity without traceback, raw logs, prompts, or outputs."""

    schema_version: Literal["inkling-matched-sanitized-failure-v1"]
    category: MatchedFailureCategory
    failure_type: StrictStr = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z_][A-Za-z0-9_.]*$",
    )
    subject: MatchedSubject
    cause_code: MatchedFailureCauseCode
    artifact_path: StrictStr | None = Field(default=None, min_length=1)
    message_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    raw_message_recorded: Literal[False]
    traceback_recorded: Literal[False]
    raw_server_log_recorded: Literal[False]

    @field_validator("artifact_path")
    @classmethod
    def artifact_path_is_safe(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _canonical_relative_path(value, label="failure artifact path")

    @model_validator(mode="after")
    def bounded_cause_matches_category(self) -> MatchedSanitizedFailureDiagnostic:
        expected_default = {
            "artifact_rehash": MatchedFailureCauseCode.ARTIFACT_CONTRACT_FAILED,
            "probe": MatchedFailureCauseCode.COMPLETION_CONTRACT_FAILED,
        }.get(self.category)
        if expected_default is None:
            expected_default = MatchedFailureCauseCode(f"{self.category}_failed")
        allowed_causes = {expected_default}
        if self.category != "cleanup":
            allowed_causes.add(MatchedFailureCauseCode.DEADLINE_EXHAUSTED)
        if self.category == "artifact_rehash":
            allowed_causes.update(
                {
                    MatchedFailureCauseCode.ARTIFACT_CONTRACT_FAILED,
                    MatchedFailureCauseCode.ARTIFACT_HASH_MISMATCH,
                    MatchedFailureCauseCode.ARTIFACT_READ_FAILED,
                    MatchedFailureCauseCode.ARTIFACT_SIZE_MISMATCH,
                }
            )
        elif self.category == "probe":
            allowed_causes.update(
                {
                    MatchedFailureCauseCode.COMPLETION_CONTRACT_FAILED,
                    MatchedFailureCauseCode.GREEDY_REPEATABILITY_FAILED,
                }
            )
        if self.cause_code not in allowed_causes:
            raise ValueError("failure cause is incompatible with its category")
        path_required = {
            MatchedFailureCauseCode.ARTIFACT_HASH_MISMATCH,
            MatchedFailureCauseCode.ARTIFACT_READ_FAILED,
            MatchedFailureCauseCode.ARTIFACT_SIZE_MISMATCH,
        }
        if self.cause_code in path_required and self.artifact_path is None:
            raise ValueError("artifact failure cause requires a safe relative artifact path")
        if self.artifact_path is not None and self.category != "artifact_rehash":
            raise ValueError("only artifact-rehash failures may retain an artifact path")
        return self


def matched_failure_receipt_sha256(payload: Mapping[str, object]) -> str:
    """Return the self-hash for one sanitized pre-publication failure receipt."""

    return _receipt_sha256(_MATCHED_FAILURE_RECEIPT_DOMAIN, payload)


class MatchedFailureReceipt(StrictFrozenModel):
    """Immutable sanitized failure receipt, allowed only before publication."""

    schema_version: Literal["inkling-matched-failure-v1"]
    status: Literal["failed"]
    stage: Literal["matched_smoke"]
    run_id: StrictStr = Field(min_length=1, max_length=200)
    allocation_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    runtime_identity_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    probe_control_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)
    subject_at_failure: MatchedSubject
    completed_subject_receipts: tuple[MatchedSubjectReceiptReference, ...] = Field(max_length=2)
    diagnostic: MatchedSanitizedFailureDiagnostic
    publication: MatchedPublicationState
    prompt_text_recorded: Literal[False]
    output_text_recorded: Literal[False]
    completed_at_utc: StrictStr = Field(min_length=1)
    receipt_sha256: StrictStr = Field(pattern=_SHA256_PATTERN)

    @field_validator("completed_at_utc")
    @classmethod
    def completion_is_utc(cls, value: str) -> str:
        return _completed_at_utc(value)

    @model_validator(mode="after")
    def failure_is_sanitized_and_unambiguous(self) -> MatchedFailureReceipt:
        expected_hash = matched_failure_receipt_sha256(
            self.model_dump(mode="python", exclude={"receipt_sha256"})
        )
        if self.receipt_sha256 != expected_hash:
            raise ValueError("matched failure receipt hash does not match its content")
        if self.diagnostic.subject is not self.subject_at_failure:
            raise ValueError("failure diagnostic subject differs from the failure receipt")
        if not self.publication.failure_receipt_allowed:
            raise ValueError("failure receipt is suppressed after publication starts")
        completed_subjects = tuple(
            reference.subject for reference in self.completed_subject_receipts
        )
        allowed_completed = (
            ((),)
            if self.subject_at_failure is MatchedSubject.BF16
            else (
                (MatchedSubject.BF16,),
                (MatchedSubject.BF16, MatchedSubject.Q3),
            )
        )
        if completed_subjects not in allowed_completed:
            raise ValueError("completed subject receipts differ from the fixed execution order")
        if (
            completed_subjects == (MatchedSubject.BF16, MatchedSubject.Q3)
            and self.diagnostic.category != "publication"
        ):
            raise ValueError(
                "two completed subject receipts are valid only for publication failure"
            )
        for reference in self.completed_subject_receipts:
            if (
                reference.allocation_identity_sha256 != self.allocation_identity_sha256
                or reference.runtime_identity_sha256 != self.runtime_identity_sha256
                or reference.probe_control_sha256 != self.probe_control_sha256
            ):
                raise ValueError("completed subject receipt belongs to a different matched cell")
        return self


__all__ = [
    "MATCHED_SUBJECT_ORDER",
    "CudaBackendIdentity",
    "ExactCudaBackendAuditEvidence",
    "ExactCudaGraphAuditRow",
    "ExactCudaIdentityAuditRow",
    "ExactCudaPlacementPolicy",
    "ExactCudaTextBackendAuditEvidence",
    "MatchedArtifactHashObservation",
    "MatchedCapacityInputs",
    "MatchedCudaPeerEdgeEvidence",
    "MatchedCudaPeerTopologyEvidence",
    "MatchedFailureCauseCode",
    "MatchedFailureReceipt",
    "MatchedGpuResourceEvidence",
    "MatchedNvidiaSmiGpuEvidence",
    "MatchedNvidiaSmiResourceSample",
    "MatchedProbeEvidence",
    "MatchedProbeTrialEvidence",
    "MatchedPublicationState",
    "MatchedRollupReceipt",
    "MatchedSanitizedFailureDiagnostic",
    "MatchedServerCleanupEvidence",
    "MatchedServerCommandSpec",
    "MatchedSubject",
    "MatchedSubjectArtifactRehashEvidence",
    "MatchedSubjectReceiptReference",
    "MatchedSubjectSmokeReceipt",
    "build_matched_capacity_inputs",
    "build_matched_cuda_placement_policy",
    "build_matched_rollup_receipt",
    "build_matched_server_command",
    "build_matched_server_environment",
    "enumerate_matched_cuda_peer_topology",
    "expected_cuda_identities",
    "matched_failure_receipt_sha256",
    "matched_rollup_receipt_sha256",
    "matched_shard_inventory_sha256",
    "matched_subject_smoke_receipt_sha256",
    "order_matched_nvidia_smi_identity_by_cuda_uuid",
    "parse_exact_cuda_backend_audit",
    "parse_exact_text_cuda_backend_audit",
    "parse_matched_cuda_backend_audit",
    "parse_matched_nvidia_smi_identity_csv",
    "parse_matched_nvidia_smi_monitor_csv",
]
