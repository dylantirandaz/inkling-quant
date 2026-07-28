"""CPU-only contracts for exact CUDA placement in matched Inkling smoke runs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from inkling_quant_lab.gguf import inkling_matched_execution as matched_execution
from inkling_quant_lab.gguf.inkling_matched import (
    MATCHED_CELL_CONFIG_RELATIVE_PATH,
    load_matched_cell_config,
)
from inkling_quant_lab.gguf.inkling_matched_execution import (
    ExactCudaBackendAuditEvidence,
    ExactCudaPlacementPolicy,
    build_matched_cuda_placement_policy,
    expected_cuda_identities,
    parse_exact_cuda_backend_audit,
    parse_matched_cuda_backend_audit,
)
from inkling_quant_lab.gguf.inkling_smoke import parse_backend_audit_evidence_v2

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPU_COUNT = 8
TENSOR_SPLIT = (1,) * GPU_COUNT


def _policy(
    *,
    gpu_count: object = GPU_COUNT,
    tensor_split: object = TENSOR_SPLIT,
) -> ExactCudaPlacementPolicy:
    return ExactCudaPlacementPolicy.model_validate(
        {
            "schema_version": "iql-exact-cuda-placement-policy-v1",
            "gpu_count": gpu_count,
            "tensor_split": tensor_split,
            "split_mode": "layer",
            "text_graph_policy": "at_least_one_all_expected_cuda",
            "vision_graph_policy": "cuda0_only",
            "audio_graph_policy": "cuda0_only",
        }
    )


def _identity_line(
    *,
    graph_uid: int,
    graph_owner: str,
    backend_index: int,
    compute: int,
    backend_name: str | None = None,
    device_name: str | None = None,
    device_type: str = "gpu",
) -> str:
    resolved_backend_name = f"CUDA{backend_index}" if backend_name is None else backend_name
    resolved_device_name = f"CUDA{backend_index}" if device_name is None else device_name
    return (
        f"IQL_SMOKE_BACKEND_IDENTITY_V2 graph_uid={graph_uid} "
        f"graph_owner={graph_owner} backend_index={backend_index} "
        f"backend_name={resolved_backend_name} device_name={resolved_device_name} "
        f"device_type={device_type} compute={compute}"
    )


def _graph_line(
    *,
    graph_uid: int,
    graph_owner: str,
    compute: int,
    gpu: int | None = None,
    cpu: int = 0,
    accel: int = 0,
    other: int = 0,
    unassigned: int = 0,
) -> str:
    resolved_gpu = compute if gpu is None else gpu
    return (
        f"IQL_SMOKE_BACKEND_GRAPH_V2 graph_uid={graph_uid} "
        f"graph_owner={graph_owner} "
        "phase=post_assignment_pre_split scope=non_view_compute "
        f"compute={compute} gpu={resolved_gpu} cpu={cpu} accel={accel} "
        f"other={other} unassigned={unassigned}"
    )


def _graph_block(
    *,
    graph_uid: int,
    graph_owner: str,
    backend_indices: Sequence[int],
    first_compute: int,
) -> tuple[str, ...]:
    identity_lines = tuple(
        _identity_line(
            graph_uid=graph_uid,
            graph_owner=graph_owner,
            backend_index=backend_index,
            compute=first_compute + offset,
        )
        for offset, backend_index in enumerate(backend_indices)
    )
    compute = sum(first_compute + offset for offset, _index in enumerate(backend_indices))
    return (
        *identity_lines,
        _graph_line(
            graph_uid=graph_uid,
            graph_owner=graph_owner,
            compute=compute,
        ),
    )


def _audit_log(
    *,
    text_indices: Sequence[int] = tuple(range(GPU_COUNT)),
    vision_indices: Sequence[int] = (0,),
    audio_indices: Sequence[int] = (0,),
) -> str:
    return "\n".join(
        (
            *_graph_block(
                graph_uid=1,
                graph_owner="text",
                backend_indices=text_indices,
                first_compute=10,
            ),
            *_graph_block(
                graph_uid=2,
                graph_owner="vision",
                backend_indices=vision_indices,
                first_compute=20,
            ),
            *_graph_block(
                graph_uid=3,
                graph_owner="audio",
                backend_indices=audio_indices,
                first_compute=30,
            ),
        )
    )


def _replace_once(log_text: str, old: str, new: str) -> str:
    assert log_text.count(old) == 1
    return log_text.replace(old, new, 1)


def _without_lines(log_text: str, fragments: Iterable[str]) -> str:
    required_fragments = tuple(fragments)
    return "\n".join(
        line
        for line in log_text.splitlines()
        if not any(fragment in line for fragment in required_fragments)
    )


def _parse(
    log_text: str,
    *,
    policy: ExactCudaPlacementPolicy | None = None,
) -> ExactCudaBackendAuditEvidence:
    return parse_exact_cuda_backend_audit(
        log_text,
        policy=_policy() if policy is None else policy,
    )


def test_exact_cuda_policy_and_expected_identity_inventory_are_canonical() -> None:
    policy = _policy()

    assert policy.schema_version == "iql-exact-cuda-placement-policy-v1"
    assert policy.gpu_count == GPU_COUNT
    assert policy.tensor_split == TENSOR_SPLIT
    assert policy.split_mode == "layer"
    assert policy.text_graph_policy == "at_least_one_all_expected_cuda"
    assert policy.vision_graph_policy == "cuda0_only"
    assert policy.audio_graph_policy == "cuda0_only"
    assert expected_cuda_identities(policy.gpu_count) == tuple(
        (index, f"CUDA{index}", f"CUDA{index}") for index in range(GPU_COUNT)
    )


def test_checked_matched_config_builds_and_uses_the_exact_policy() -> None:
    config = load_matched_cell_config(PROJECT_ROOT / MATCHED_CELL_CONFIG_RELATIVE_PATH)

    policy = build_matched_cuda_placement_policy(config)
    evidence = parse_matched_cuda_backend_audit(_audit_log(), config=config)

    assert policy.gpu_count == GPU_COUNT
    assert policy.tensor_split == TENSOR_SPLIT
    assert policy.split_mode == "layer"
    assert evidence.policy == policy


@pytest.mark.parametrize(
    "field_name",
    (
        "schema_version",
        "split_mode",
        "text_graph_policy",
        "vision_graph_policy",
        "audio_graph_policy",
    ),
)
def test_exact_cuda_policy_requires_persisted_contract_fields(
    field_name: str,
) -> None:
    raw = _policy().model_dump(mode="python")
    del raw[field_name]

    with pytest.raises(ValidationError):
        ExactCudaPlacementPolicy.model_validate(raw)


@pytest.mark.parametrize("split_count", (7, 9))
def test_exact_cuda_policy_rejects_tensor_split_cardinality_drift(
    split_count: int,
) -> None:
    with pytest.raises(ValidationError):
        _policy(tensor_split=(1,) * split_count)


@pytest.mark.parametrize(
    "tensor_split",
    (
        (1, 1, 1, 1, 1, 1, 1, 2),
        (1, 1, 1, 1, 1, 1, 1, 0),
        (1, 1, 1, 1, 1, 1, 1, -1),
    ),
)
def test_exact_cuda_policy_rejects_nonuniform_tensor_split(
    tensor_split: tuple[int, ...],
) -> None:
    with pytest.raises(ValidationError):
        _policy(tensor_split=tensor_split)


def test_exact_cuda_policy_rejects_boolean_tensor_split_values() -> None:
    with pytest.raises(ValidationError):
        _policy(tensor_split=(1, 1, 1, 1, 1, 1, 1, True))


@pytest.mark.parametrize("gpu_count", (True, False, "8", 8.0))
def test_exact_cuda_policy_rejects_coerced_or_boolean_gpu_count(
    gpu_count: object,
) -> None:
    with pytest.raises(ValidationError):
        _policy(gpu_count=gpu_count)


@pytest.mark.parametrize("gpu_count", (0, 65))
def test_exact_cuda_policy_rejects_out_of_range_gpu_count(
    gpu_count: int,
) -> None:
    with pytest.raises(ValidationError):
        _policy(gpu_count=gpu_count)


@pytest.mark.parametrize("gpu_count", (True, False, "8", 8.0, 0, 65))
def test_expected_cuda_identities_rejects_invalid_gpu_count(
    gpu_count: object,
) -> None:
    with pytest.raises(ValueError):
        expected_cuda_identities(gpu_count)  # type: ignore[arg-type]


def test_exact_cuda_backend_audit_accepts_the_eight_gpu_matched_cell() -> None:
    policy = _policy()
    evidence = _parse(_audit_log(), policy=policy)

    assert isinstance(evidence, ExactCudaBackendAuditEvidence)
    assert evidence.schema_version == "inkling-exact-cuda-backend-audit-v1"
    assert evidence.policy == policy
    assert evidence.expected_identities == expected_cuda_identities(GPU_COUNT)
    assert evidence.observed_graphs == 3
    assert evidence.compute_operations == evidence.gpu_operations == 158
    assert evidence.accelerator_operations == 0
    assert evidence.cpu_operations == 0
    assert evidence.other_operations == 0
    assert evidence.unassigned_operations == 0
    assert tuple(row.graph_owner for row in evidence.graphs) == (
        "text",
        "vision",
        "audio",
    )
    assert tuple(
        (row.backend_index, row.backend_name, row.device_name)
        for row in evidence.identities
        if row.graph_uid == 1
    ) == expected_cuda_identities(GPU_COUNT)
    assert evidence.exact_cuda_identity_inventory is True
    assert evidence.text_full_cell_observed is True
    assert evidence.projector_graphs_cuda0_only is True
    assert evidence.all_compute_operations_accelerated is True
    assert evidence.no_cpu_model_graph_fallback is True


@pytest.mark.parametrize(
    "log_text",
    (
        _audit_log(text_indices=tuple(range(7))),
        _audit_log(text_indices=tuple(range(9))),
    ),
    ids=("missing-cuda7", "unexpected-cuda8"),
)
def test_exact_cuda_backend_audit_rejects_missing_or_extra_identity(
    log_text: str,
) -> None:
    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_a_text_graph_without_cuda0() -> None:
    with pytest.raises((ValidationError, ValueError)):
        _parse(_audit_log(text_indices=tuple(range(1, GPU_COUNT))))


def test_exact_cuda_backend_audit_rejects_swapped_cuda_identity() -> None:
    log_text = _replace_once(
        _audit_log(),
        "backend_index=1 backend_name=CUDA1 device_name=CUDA1",
        "backend_index=1 backend_name=CUDA2 device_name=CUDA2",
    )

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_duplicate_backend_identity() -> None:
    duplicate = _identity_line(
        graph_uid=1,
        graph_owner="text",
        backend_index=7,
        compute=17,
    )
    log_text = _audit_log() + "\n" + duplicate

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_out_of_order_backend_identities() -> None:
    lines = _audit_log().splitlines()
    lines[0], lines[1] = lines[1], lines[0]

    with pytest.raises((ValidationError, ValueError)):
        _parse("\n".join(lines))


def test_exact_cuda_backend_audit_requires_the_text_graph_to_use_the_full_cell() -> None:
    log_text = _audit_log(
        text_indices=(0,),
        vision_indices=tuple(range(GPU_COUNT)),
    )

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


@pytest.mark.parametrize("graph_owner", ("vision", "audio"))
@pytest.mark.parametrize(
    "backend_indices",
    (
        (1,),
        (0, 1),
    ),
    ids=("cuda1-instead-of-cuda0", "more-than-one-gpu"),
)
def test_exact_cuda_backend_audit_rejects_projector_graph_cuda_drift(
    graph_owner: str,
    backend_indices: tuple[int, ...],
) -> None:
    keyword = f"{graph_owner}_indices"
    log_text = _audit_log(**{keyword: backend_indices})

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_a_missing_graph_owner() -> None:
    log_text = _without_lines(_audit_log(), ("graph_uid=3 ",))

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_an_unknown_graph_owner() -> None:
    log_text = _audit_log().replace("graph_owner=audio", "graph_owner=unknown")

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_an_unknown_identity_owner() -> None:
    log_text = _replace_once(
        _audit_log(),
        "graph_uid=3 graph_owner=audio backend_index=0",
        "graph_uid=3 graph_owner=unknown backend_index=0",
    )

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_graph_identity_owner_mismatch() -> None:
    log_text = _replace_once(
        _audit_log(),
        "graph_uid=3 graph_owner=audio backend_index=0",
        "graph_uid=3 graph_owner=vision backend_index=0",
    )

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_reused_graph_uid() -> None:
    log_text = _audit_log().replace("graph_uid=2 ", "graph_uid=1 ")

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_an_orphan_identity() -> None:
    log_text = (
        _audit_log()
        + "\n"
        + _identity_line(
            graph_uid=99,
            graph_owner="text",
            backend_index=0,
            compute=1,
        )
    )

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_a_cpu_node_marker() -> None:
    log_text = (
        _audit_log() + "\n" + "IQL_SMOKE_CPU_NODE_V2 graph_uid=1 graph_owner=text "
        "backend_index=8 device_type=cpu ordinal=0 op=MUL_MAT "
        "name_len=4 name_hex=6e6f6465"
    )

    with pytest.raises(ValueError):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_a_non_gpu_identity() -> None:
    log_text = _replace_once(
        _audit_log(),
        "device_type=gpu compute=10",
        "device_type=accel compute=10",
    )

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


@pytest.mark.parametrize(
    ("category", "replacement"),
    (
        (
            "accel",
            "compute=21 gpu=20 cpu=0 accel=1 other=0 unassigned=0",
        ),
        (
            "cpu",
            "compute=21 gpu=20 cpu=1 accel=0 other=0 unassigned=0",
        ),
        (
            "other",
            "compute=21 gpu=20 cpu=0 accel=0 other=1 unassigned=0",
        ),
        (
            "unassigned",
            "compute=21 gpu=20 cpu=0 accel=0 other=0 unassigned=1",
        ),
    ),
)
def test_exact_cuda_backend_audit_rejects_forbidden_graph_counts(
    category: str,
    replacement: str,
) -> None:
    del category
    log_text = _replace_once(
        _audit_log(),
        "compute=20 gpu=20 cpu=0 accel=0 other=0 unassigned=0",
        replacement,
    )

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_graph_category_count_drift() -> None:
    log_text = _replace_once(
        _audit_log(),
        "compute=108 gpu=108 cpu=0 accel=0 other=0 unassigned=0",
        "compute=109 gpu=108 cpu=0 accel=0 other=0 unassigned=0",
    )

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_identity_count_drift() -> None:
    log_text = _replace_once(
        _audit_log(),
        "backend_index=7 backend_name=CUDA7 device_name=CUDA7 device_type=gpu compute=17",
        "backend_index=7 backend_name=CUDA7 device_name=CUDA7 device_type=gpu compute=18",
    )

    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_exact_cuda_backend_evidence_rejects_aggregate_count_drift() -> None:
    evidence = _parse(_audit_log())
    raw = evidence.model_dump(mode="python")
    raw["compute_operations"] = evidence.compute_operations + 1

    with pytest.raises(ValidationError):
        ExactCudaBackendAuditEvidence.model_validate(raw)


def test_exact_cuda_backend_evidence_rejects_expected_identity_drift() -> None:
    raw = _parse(_audit_log()).model_dump(mode="python")
    raw["expected_identities"] = raw["expected_identities"][:-1]

    with pytest.raises(ValidationError):
        ExactCudaBackendAuditEvidence.model_validate(raw)


def test_exact_cuda_backend_evidence_rejects_observed_graph_count_drift() -> None:
    raw = _parse(_audit_log()).model_dump(mode="python")
    raw["observed_graphs"] += 1

    with pytest.raises(ValidationError):
        ExactCudaBackendAuditEvidence.model_validate(raw)


def test_exact_cuda_backend_evidence_rejects_gpu_aggregate_count_drift() -> None:
    raw = _parse(_audit_log()).model_dump(mode="python")
    raw["gpu_operations"] -= 1

    with pytest.raises(ValidationError):
        ExactCudaBackendAuditEvidence.model_validate(raw)


@pytest.mark.parametrize(
    ("section", "row_index", "field_name", "invalid_value"),
    (
        ("graphs", 0, "graph_uid", True),
        ("graphs", 0, "compute", "108"),
        ("identities", 0, "graph_uid", True),
        ("identities", 1, "backend_index", True),
        ("identities", 0, "compute", "10"),
    ),
)
def test_exact_cuda_backend_evidence_rejects_coerced_nested_integers(
    section: str,
    row_index: int,
    field_name: str,
    invalid_value: object,
) -> None:
    raw = _parse(_audit_log()).model_dump(mode="python")
    raw[section][row_index][field_name] = invalid_value

    with pytest.raises(ValidationError):
        ExactCudaBackendAuditEvidence.model_validate(raw)


def test_exact_cuda_backend_evidence_rejects_nested_graph_count_drift() -> None:
    raw = _parse(_audit_log()).model_dump(mode="python")
    raw["graphs"][0]["compute"] += 1

    with pytest.raises(ValidationError):
        ExactCudaBackendAuditEvidence.model_validate(raw)


def test_exact_cuda_backend_evidence_rejects_nested_accelerator_work() -> None:
    raw = _parse(_audit_log()).model_dump(mode="python")
    raw["graphs"][0]["compute"] += 1
    raw["graphs"][0]["accel"] = 1

    with pytest.raises(ValidationError):
        ExactCudaBackendAuditEvidence.model_validate(raw)


def test_exact_cuda_backend_evidence_rejects_noncanonical_nested_identifier() -> None:
    raw = _parse(_audit_log()).model_dump(mode="python")
    raw["identities"][0]["backend_name"] = "CUDA 0"

    with pytest.raises(ValidationError):
        ExactCudaBackendAuditEvidence.model_validate(raw)


@pytest.mark.parametrize(
    "field_name",
    (
        "accelerator_operations",
        "cpu_operations",
        "other_operations",
        "unassigned_operations",
    ),
)
def test_exact_cuda_backend_evidence_rejects_boolean_zero_counts(
    field_name: str,
) -> None:
    raw = _parse(_audit_log()).model_dump(mode="python")
    raw[field_name] = False

    with pytest.raises(ValidationError):
        ExactCudaBackendAuditEvidence.model_validate(raw)


@pytest.mark.parametrize(
    "field_name",
    (
        "exact_cuda_identity_inventory",
        "text_full_cell_observed",
        "projector_graphs_cuda0_only",
        "all_compute_operations_accelerated",
        "no_cpu_model_graph_fallback",
    ),
)
@pytest.mark.parametrize("invalid_value", (False, 1))
def test_exact_cuda_backend_evidence_rejects_false_or_coerced_proof_flags(
    field_name: str,
    invalid_value: object,
) -> None:
    raw = _parse(_audit_log()).model_dump(mode="python")
    raw[field_name] = invalid_value

    with pytest.raises(ValidationError):
        ExactCudaBackendAuditEvidence.model_validate(raw)


@pytest.mark.parametrize(
    "field_name",
    (
        "schema_version",
        "exact_cuda_identity_inventory",
        "text_full_cell_observed",
        "projector_graphs_cuda0_only",
        "all_compute_operations_accelerated",
        "no_cpu_model_graph_fallback",
    ),
)
def test_exact_cuda_backend_evidence_requires_persisted_contract_fields(
    field_name: str,
) -> None:
    raw = _parse(_audit_log()).model_dump(mode="python")
    del raw[field_name]

    with pytest.raises(ValidationError):
        ExactCudaBackendAuditEvidence.model_validate(raw)


def test_exact_cuda_backend_audit_rejects_an_oversized_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_text = _audit_log()
    monkeypatch.setattr(matched_execution, "_MAX_AUDIT_LOG_CHARACTERS", len(log_text) - 1)

    with pytest.raises(ValueError, match="character limit"):
        _parse(log_text)


def test_exact_cuda_backend_audit_rejects_an_oversized_marker_line(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_text = _audit_log()
    monkeypatch.setattr(matched_execution, "MAX_BACKEND_FAILURE_LINE_BYTES", 32)

    with pytest.raises(ValueError, match="marker line"):
        _parse(log_text)


def test_exact_cuda_backend_audit_uses_the_parser_newline_framing_for_line_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = _audit_log().split("\n")
    line_limit = max(len(line) for line in lines)
    lines[0] = ("x" * line_limit) + "\r" + lines[0]
    monkeypatch.setattr(
        matched_execution,
        "MAX_BACKEND_FAILURE_LINE_BYTES",
        line_limit,
    )

    with pytest.raises(ValueError, match="marker line"):
        _parse("\n".join(lines))


def test_exact_cuda_backend_audit_caps_cpu_node_markers_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_marker = (
        "IQL_SMOKE_CPU_NODE_V2 graph_uid=1 graph_owner=text "
        "backend_index=8 device_type=cpu ordinal=0 op=MUL_MAT "
        "name_len=4 name_hex=6e6f6465"
    )
    monkeypatch.setattr(matched_execution, "_MAX_AUDIT_CPU_ROWS", 0)

    with pytest.raises(ValueError, match="CPU-node marker limit"):
        _parse(_audit_log() + "\n" + cpu_marker)


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "message"),
    (
        ("_MAX_AUDIT_GRAPH_ROWS", 2, "graph marker limit"),
        ("_MAX_AUDIT_IDENTITY_ROWS", 9, "identity marker limit"),
    ),
)
def test_exact_cuda_backend_audit_rejects_excess_marker_counts(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
    message: str,
) -> None:
    monkeypatch.setattr(matched_execution, limit_name, limit_value)

    with pytest.raises(ValueError, match=message):
        _parse(_audit_log())


@pytest.mark.parametrize(
    "log_text",
    (
        _replace_once(
            _audit_log(),
            "graph_uid=1 graph_owner=text "
            "phase=post_assignment_pre_split scope=non_view_compute "
            "compute=108 gpu=108 cpu=0 accel=0 other=0 unassigned=0",
            "graph_uid=1 graph_owner=text "
            "phase=post_assignment_pre_split "
            "compute=108 gpu=108 cpu=0 accel=0 other=0 unassigned=0",
        ),
        _replace_once(
            _audit_log(),
            "device_type=gpu compute=10",
            "device_type=gpu compute=10 trailing=1",
        ),
        _audit_log() + "\n" + "IQL_SMOKE_BACKEND_GRAPH_V1 graph_uid=90 "
        "phase=post_assignment_pre_split scope=non_view_compute "
        "compute=1 gpu=1 cpu=0 accel=0 other=0 unassigned=0",
        _audit_log().replace("\n", " ", 1),
    ),
    ids=(
        "malformed-graph-marker",
        "trailing-marker-data",
        "mixed-version-one-marker",
        "two-version-two-markers-on-one-line",
    ),
)
def test_exact_cuda_backend_audit_rejects_noncanonical_marker_input(
    log_text: str,
) -> None:
    with pytest.raises((ValidationError, ValueError)):
        _parse(log_text)


def test_historical_two_gpu_parser_still_accepts_its_exact_fixture() -> None:
    evidence = parse_backend_audit_evidence_v2(
        _audit_log(
            text_indices=(0, 1),
            vision_indices=(0,),
            audio_indices=(0,),
        )
    )

    assert evidence.schema_version == "inkling-backend-audit-v2"
    assert evidence.observed_graphs == 3
    assert {
        (row.backend_index, row.backend_name, row.device_name)
        for row in evidence.identities
        if row.graph_uid == 1
    } == {
        (0, "CUDA0", "CUDA0"),
        (1, "CUDA1", "CUDA1"),
    }


def test_historical_two_gpu_parser_rejects_the_eight_gpu_fixture() -> None:
    with pytest.raises((ValidationError, ValueError)):
        parse_backend_audit_evidence_v2(_audit_log())
