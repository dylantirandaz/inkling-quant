"""CPU-safe execution contracts for the exact matched Inkling CUDA cell.

This module does not start Modal or execute llama.cpp. It validates the
owner-tagged backend markers that a later matched runner must retain.
Historical two-GPU smoke receipts keep their original validators.
"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from inkling_quant_lab.config import StrictFrozenModel
from inkling_quant_lab.gguf.inkling_matched import InklingMatchedCellConfig
from inkling_quant_lab.gguf.inkling_smoke import (
    MAX_BACKEND_FAILURE_LINE_BYTES,
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
_REQUIRED_GRAPH_OWNERS: Final = frozenset({"text", "vision", "audio"})


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
        if not all(
            (
                self.exact_cuda_identity_inventory,
                self.text_full_cell_observed,
                self.projector_graphs_cuda0_only,
                self.all_compute_operations_accelerated,
                self.no_cpu_model_graph_fallback,
            )
        ):
            raise ValueError("backend audit proof fields must all be true")

        expected_order = expected_cuda_identities(self.policy.gpu_count)
        if self.expected_identities != expected_order:
            raise ValueError("backend audit expected CUDA identities differ from its policy")
        expected = set(expected_order)
        cuda0_identity = expected_order[0]

        if self.observed_graphs != len(self.graphs):
            raise ValueError("backend graph count differs from its graph records")
        graph_uids = tuple(row.graph_uid for row in self.graphs)
        if len(graph_uids) != len(set(graph_uids)):
            raise ValueError("backend audit contains duplicate graph identities")
        if {row.graph_owner for row in self.graphs} != _REQUIRED_GRAPH_OWNERS:
            raise ValueError("backend audit does not cover text, vision, and audio graph owners")

        identity_keys = tuple((row.graph_uid, row.backend_index) for row in self.identities)
        if len(identity_keys) != len(set(identity_keys)):
            raise ValueError("backend audit contains duplicate backend identities")
        if {row.graph_uid for row in self.identities} != set(graph_uids):
            raise ValueError("backend identities do not cover the exact graph set")
        if any(row.graph_owner == "unknown" for row in self.identities):
            raise ValueError("backend identity contains an unknown graph owner")
        if any(row.device_type != "gpu" for row in self.identities):
            raise ValueError("backend audit used a non-CUDA accelerator")

        identities_by_graph: dict[int, list[ExactCudaIdentityAuditRow]] = {}
        for identity in self.identities:
            identities_by_graph.setdefault(identity.graph_uid, []).append(identity)

        text_full_cell_observed = False
        category_totals = {
            "gpu": 0,
            "cpu": 0,
            "accel": 0,
            "other": 0,
            "unassigned": 0,
        }
        for graph in self.graphs:
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
        return self


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


def parse_exact_cuda_backend_audit(
    log_text: str,
    *,
    policy: ExactCudaPlacementPolicy,
) -> ExactCudaBackendAuditEvidence:
    """Parse V2 markers and validate them against one explicit CUDA policy."""

    _validate_audit_input_bounds(log_text)
    rows = parse_backend_audit_rows_v2(log_text)
    graphs = tuple(
        ExactCudaGraphAuditRow.model_validate(row.model_dump(mode="python")) for row in rows.graphs
    )
    identities = tuple(
        ExactCudaIdentityAuditRow.model_validate(row.model_dump(mode="python"))
        for row in rows.identities
    )
    return ExactCudaBackendAuditEvidence(
        schema_version="inkling-exact-cuda-backend-audit-v1",
        policy=policy,
        expected_identities=expected_cuda_identities(policy.gpu_count),
        observed_graphs=len(graphs),
        compute_operations=sum(row.compute for row in graphs),
        gpu_operations=sum(row.gpu for row in graphs),
        accelerator_operations=sum(row.accel for row in graphs),
        cpu_operations=sum(row.cpu for row in graphs),
        other_operations=sum(row.other for row in graphs),
        unassigned_operations=sum(row.unassigned for row in graphs),
        graphs=graphs,
        identities=identities,
        exact_cuda_identity_inventory=True,
        text_full_cell_observed=True,
        projector_graphs_cuda0_only=True,
        all_compute_operations_accelerated=True,
        no_cpu_model_graph_fallback=True,
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


__all__ = [
    "CudaBackendIdentity",
    "ExactCudaBackendAuditEvidence",
    "ExactCudaGraphAuditRow",
    "ExactCudaIdentityAuditRow",
    "ExactCudaPlacementPolicy",
    "build_matched_cuda_placement_policy",
    "expected_cuda_identities",
    "parse_exact_cuda_backend_audit",
    "parse_matched_cuda_backend_audit",
]
