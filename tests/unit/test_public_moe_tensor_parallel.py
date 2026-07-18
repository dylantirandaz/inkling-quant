"""Contracts for the executed public-MoE expert tensor-parallel slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.distributed as torch_dist
from pydantic import ValidationError
from safetensors.torch import save_file

from inkling_quant_lab import public_moe_tensor_parallel as public_tp
from inkling_quant_lab.exceptions import CapabilityError
from inkling_quant_lab.public_moe_tensor_parallel import (
    PublicMoeTensorParallelResult,
    audit_stories15m_snapshot,
    run_public_moe_expert_tensor_parallel,
)
from scripts import run_public_moe_tensor_parallel_smoke as runner

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_PATH = PROJECT_ROOT / "docs/experiments/stories15m-moe-expert-tensor-parallel-m3.json"
EVIDENCE_SHA256 = "580102da5b1c03fd838de3d229f83b62cf5c0de27902637809fdca398da36e2b"
RUNNER_SHA256 = "8d957647d7b9a6bd0ed730d288d9a6dc5b294e6d69ea17f20a234d582df04616"
MODULE_SHA256 = "808393f7a5200fe0b628eb296e57691cd7dd5a7a4f6ba71adb80ee8e3312ca7b"


def _record() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(EVIDENCE_PATH.read_text(encoding="utf-8")))


def _result_payload() -> dict[str, Any]:
    return deepcopy(_record()["result"])


def test_checked_public_moe_tensor_parallel_record_is_exact_and_strict() -> None:
    evidence_bytes = EVIDENCE_PATH.read_bytes()
    record = json.loads(evidence_bytes)
    result = PublicMoeTensorParallelResult.model_validate(record["result"])

    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SHA256
    assert result.source.weight_sha256 == public_tp.MODEL_WEIGHT_SHA256
    assert result.public_checkpoint_executed is True
    assert result.all_public_moe_expert_blocks_tensor_parallel_validated is True
    assert result.parameter_sharded_forward_validated is True
    assert result.end_to_end_transformer_forward_validated is False
    assert tuple(len(rank.shard_evidence) for rank in result.ranks) == (72, 72)
    assert tuple(rank.local_fraction for rank in result.ranks) == (0.5, 0.5)
    assert max(max(layer.rank_max_abs_errors) for layer in result.layers) < 5e-9
    assert all(layer.all_experts_executed for layer in result.layers)


def test_checked_record_binds_the_executed_implementation_bytes() -> None:
    record = _record()
    runner_bytes = (PROJECT_ROOT / record["implementation"]["runner"]["path"]).read_bytes()
    module_bytes = (PROJECT_ROOT / record["implementation"]["module"]["path"]).read_bytes()

    assert hashlib.sha256(runner_bytes).hexdigest() == RUNNER_SHA256
    assert hashlib.sha256(module_bytes).hexdigest() == MODULE_SHA256
    assert record["implementation"]["runner"]["sha256_at_start"] == RUNNER_SHA256
    assert record["implementation"]["runner"]["sha256_at_end"] == RUNNER_SHA256
    assert record["implementation"]["module"]["sha256_at_start"] == MODULE_SHA256
    assert record["implementation"]["module"]["sha256_at_end"] == MODULE_SHA256


def test_every_expert_projection_is_an_exact_two_rank_partition() -> None:
    result = PublicMoeTensorParallelResult.model_validate(_result_payload())
    by_rank = [{item.tensor_name: item for item in rank.shard_evidence} for rank in result.ranks]

    assert set(by_rank[0]) == set(by_rank[1])
    for name in sorted(by_rank[0]):
        first = by_rank[0][name]
        second = by_rank[1][name]
        assert first.shard_range == (0, 384)
        assert second.shard_range == (384, 768)
        assert first.source_tensor_sha256 == second.source_tensor_sha256
        assert first.source_float32_sha256 == second.source_float32_sha256
        assert first.reconstructed_float32_sha256 == first.source_float32_sha256
        assert second.reconstructed_float32_sha256 == second.source_float32_sha256
        assert first.local_float32_sha256 != second.local_float32_sha256


def test_checked_record_contains_only_finite_numbers_and_redacted_paths() -> None:
    record = _record()

    def visit(value: Any) -> None:
        if isinstance(value, float):
            assert math.isfinite(value)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(record)
    serialized = json.dumps(record, sort_keys=True)
    assert "/Users/" not in serialized
    assert "/private/" not in serialized
    assert "Authorization" not in serialized
    assert "Bearer " not in serialized


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload["ranks"][0]["shard_evidence"][0].update(
                {"shard_range": [1, 385]}
            ),
            "exact half",
        ),
        (
            lambda payload: payload["layers"][0].update(
                {"rank_max_tolerance_ratios": [1.01, 1.01]}
            ),
            "at most one",
        ),
        (
            lambda payload: payload.update({"end_to_end_transformer_forward_validated": True}),
            "False",
        ),
        (
            lambda payload: payload.update(
                {"operator_declared_hardware_label": "Authorization: Bearer abc123"}
            ),
            "credential-like",
        ),
    ),
)
def test_result_rejects_scope_or_integrity_mutation(
    mutation: Any,
    message: str,
) -> None:
    payload = _result_payload()
    mutation(payload)

    with pytest.raises(ValidationError, match=message):
        PublicMoeTensorParallelResult.model_validate(payload)


def test_routing_normalizes_top2_and_hashes_tensor_identity() -> None:
    hidden = torch.tensor(((1.0, 0.0), (0.0, 1.0)), dtype=torch.float32)
    router = torch.tensor(
        ((2.0, 0.0), (0.0, 2.0), (-2.0, 0.0), (0.0, -2.0)),
        dtype=torch.float32,
    )

    weights, indices = public_tp._routing(hidden, router)

    assert tuple(weights.shape) == (2, 2)
    assert tuple(indices.shape) == (2, 2)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))
    assert public_tp._tensor_sha256(hidden) != public_tp._tensor_sha256(hidden.double())


def test_full_expert_forward_executes_all_selected_experts() -> None:
    hidden = torch.tensor(
        ((1.0, 0.5), (0.5, 1.0), (1.0, -0.5), (-0.5, 1.0)),
        dtype=torch.float32,
    )
    top_k_indices = torch.tensor(((0, 1), (1, 2), (2, 3), (3, 0)), dtype=torch.long)
    top_k_weights = torch.full((4, 2), 0.5, dtype=torch.float32)
    tensors: dict[str, torch.Tensor] = {}
    for expert in range(4):
        scale = float(expert + 1)
        tensors[public_tp._expert_tensor_name(0, expert, "w1")] = torch.full((4, 2), scale / 4.0)
        tensors[public_tp._expert_tensor_name(0, expert, "w2")] = torch.full((2, 4), scale / 8.0)
        tensors[public_tp._expert_tensor_name(0, expert, "w3")] = torch.full((4, 2), scale / 6.0)

    class Checkpoint:
        def get_tensor(self, name: str) -> torch.Tensor:
            return tensors[name]

    output = public_tp._full_expert_forward(
        Checkpoint(),
        layer=0,
        hidden_states=hidden,
        top_k_weights=top_k_weights,
        top_k_indices=top_k_indices,
    )

    assert tuple(output.shape) == tuple(hidden.shape)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == output.numel()


def test_rank_slice_selects_complementary_intermediate_halves() -> None:
    w1 = torch.arange(768 * 288, dtype=torch.float32).reshape(768, 288)
    w2 = torch.arange(288 * 768, dtype=torch.float32).reshape(288, 768)

    class Slice:
        def __init__(self, value: torch.Tensor) -> None:
            self.value = value

        def __getitem__(self, index: Any) -> torch.Tensor:
            return self.value[index]

    class Checkpoint:
        def get_slice(self, name: str) -> Slice:
            return Slice(w1 if name == "w1" else w2)

    checkpoint = Checkpoint()
    w1_first = public_tp._rank_slice(checkpoint, "w1", "w1", 0)
    w1_second = public_tp._rank_slice(checkpoint, "w1", "w1", 1)
    w2_first = public_tp._rank_slice(checkpoint, "w2", "w2", 0)
    w2_second = public_tp._rank_slice(checkpoint, "w2", "w2", 1)

    torch.testing.assert_close(torch.cat((w1_first, w1_second), dim=0), w1)
    torch.testing.assert_close(torch.cat((w2_first, w2_second), dim=1), w2)
    assert tuple(w1_first.shape) == (384, 288)
    assert tuple(w2_first.shape) == (288, 384)


def test_combined_tolerance_ratio_matches_torch_close_contract() -> None:
    expected = torch.tensor((0.0, 10.0), dtype=torch.float32)
    actual = torch.tensor((5e-8, 10.00005), dtype=torch.float32)

    relative = public_tp._relative_error(actual, expected, 1e-7)
    ratio = public_tp._combined_tolerance_ratio(actual, expected, 1e-7, 1e-5)

    assert relative > 1e-5
    assert ratio < 1.0


def test_runner_publishes_once_and_rejects_unsafe_arguments(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    runner._publish_json(output, {"finite": 1.0})
    assert json.loads(output.read_text(encoding="utf-8")) == {"finite": 1.0}
    with pytest.raises(FileExistsError, match="immutable"):
        runner._publish_json(output, {"replacement": 2.0})
    with pytest.raises(ValueError, match="Out of range"):
        runner._publish_json(tmp_path / "nan.json", {"bad": float("nan")})
    with pytest.raises(argparse.ArgumentTypeError, match="boundary whitespace"):
        runner._non_empty_label(" bad")
    with pytest.raises(argparse.ArgumentTypeError, match="parent traversal"):
        runner._output_path("artifacts/../escape.json")


def test_runner_rejects_invalid_execution_contract_before_snapshot_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        public_tp,
        "probe_gloo_capability",
        lambda: SimpleNamespace(status="available", reason=None),
    )
    with pytest.raises(ValueError, match="boundary whitespace"):
        run_public_moe_expert_tensor_parallel(tmp_path, hardware_label=" bad")
    with pytest.raises(ValueError, match="finite and positive"):
        run_public_moe_expert_tensor_parallel(
            tmp_path, hardware_label="unit-test CPU", timeout_seconds=float("inf")
        )


def test_source_audit_rejects_unpinned_directory_name(tmp_path: Path) -> None:
    snapshot = tmp_path / "wrong-revision"
    snapshot.mkdir()

    with pytest.raises(ValueError, match="exact pinned revision"):
        audit_stories15m_snapshot(snapshot)


def test_evidence_models_reject_every_cross_record_integrity_failure() -> None:
    mutations: tuple[Callable[[dict[str, Any]], object], ...] = (
        lambda payload: payload["source"]["file_inventory"].pop(),
        lambda payload: payload["source"]["file_inventory"].append(
            deepcopy(payload["source"]["file_inventory"][0])
        ),
        lambda payload: payload["ranks"][0]["shard_evidence"][0].update(
            {"tensor_name": "model.layers.0.wrong"}
        ),
        lambda payload: payload["ranks"][0]["shard_evidence"][0].update({"global_shape": [1, 1]}),
        lambda payload: payload["ranks"][0]["shard_evidence"][0].update(
            {"reconstructed_float32_sha256": "0" * 64}
        ),
        lambda payload: payload["ranks"][0]["shard_evidence"].pop(),
        lambda payload: payload["ranks"][0].update({"local_fraction": 0.25}),
        lambda payload: payload["ranks"][0]["shard_evidence"][0].update(
            {"shard_range": [384, 768]}
        ),
        lambda payload: payload["ranks"][0].update(
            {"layer_output_sha256": payload["ranks"][0]["layer_output_sha256"][:-1]}
        ),
        lambda payload: payload["layers"][0].update({"selected_expert_counts": [1, 1, 1, 1]}),
        lambda payload: payload["layers"][0].update({"selected_expert_counts": [0, 46, 51, 31]}),
        lambda payload: payload["layers"][0].update({"rank_max_abs_errors": [0.0, 1e-9]}),
        lambda payload: payload["layers"][0].update({"rank_max_relative_errors": [0.0, 1e-9]}),
        lambda payload: payload["layers"][0].update({"rank_max_tolerance_ratios": [0.0, 0.5]}),
        lambda payload: payload["layers"][0].update({"rank_max_abs_errors": [-1.0, -1.0]}),
        lambda payload: payload["layers"][0].update(
            {"rank_max_relative_errors": [float("inf"), float("inf")]}
        ),
        lambda payload: payload["layers"][0].update({"rank_max_tolerance_ratios": [1.01, 1.01]}),
        lambda payload: payload.update({"layers": list(reversed(payload["layers"]))}),
        lambda payload: payload.update({"ranks": list(reversed(payload["ranks"]))}),
        lambda payload: payload["ranks"][0].update({"input_sha256": "0" * 64}),
        lambda payload: payload["ranks"][0]["layer_output_sha256"].__setitem__(0, "0" * 64),
        lambda payload: payload["layers"][0].update({"rank_max_abs_errors": [1e-6, 1e-6]}),
        lambda payload: payload.update({"scope": "generic distributed public-model support"}),
        lambda payload: payload.update({"interface": "en0"}),
    )

    for mutation in mutations:
        payload = _result_payload()
        mutation(payload)
        with pytest.raises(ValidationError):
            PublicMoeTensorParallelResult.model_validate(payload)


def test_physical_memory_probe_handles_success_failure_and_nonpositive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {"SC_PAGE_SIZE": 4096, "SC_PHYS_PAGES": 10}
    monkeypatch.setattr(os, "sysconf", lambda name: values[name])
    assert public_tp._physical_memory_bytes() == 40_960

    values["SC_PHYS_PAGES"] = 0
    assert public_tp._physical_memory_bytes() is None

    def unavailable(_name: str) -> int:
        raise OSError("unavailable")

    monkeypatch.setattr(os, "sysconf", unavailable)
    assert public_tp._physical_memory_bytes() is None


def _worker_payload(rank: int, *, mismatched_routes: bool = False) -> dict[str, Any]:
    result = _result_payload()
    rank_record = result["ranks"][rank]
    layers = []
    for layer_index, layer in enumerate(result["layers"]):
        counts = list(layer["selected_expert_counts"])
        if mismatched_routes and rank == 1 and layer_index == 0:
            counts[0] += 1
            counts[1] -= 1
        layers.append(
            {
                "layer_id": layer_index,
                "selected_expert_counts": counts,
                "output_sha256": rank_record["layer_output_sha256"][layer_index],
                "max_abs_error": layer["rank_max_abs_errors"][rank],
                "max_relative_error": layer["rank_max_relative_errors"][rank],
                "max_tolerance_ratio": layer["rank_max_tolerance_ratios"][rank],
            }
        )
    return {
        "rank": rank,
        "input_sha256": rank_record["input_sha256"],
        "shard_evidence": rank_record["shard_evidence"],
        "layers": layers,
    }


class _FakeProcess:
    def __init__(self, args: tuple[Any, ...], *, mode: str, mismatched_routes: bool) -> None:
        self._args = args
        self._mode = mode
        self._mismatched_routes = mismatched_routes
        self._alive = mode == "timeout"
        self.exitcode = 1 if mode == "failed" and args[0] == 0 else 0

    def start(self) -> None:
        rank = self._args[0]
        result_directory = Path(self._args[2])
        if self._mode == "failed" and rank == 0:
            (result_directory / "rank-0.error.json").write_text(
                json.dumps({"rank": 0, "error_type": "RuntimeError", "error": "test failure"}),
                encoding="utf-8",
            )
            return
        if self._mode != "timeout":
            (result_directory / f"rank-{rank}.json").write_text(
                json.dumps(
                    _worker_payload(rank, mismatched_routes=self._mismatched_routes),
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

    def join(self, _timeout: float | None = None) -> None:
        return None

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self._alive = False


class _FakeSpawnContext:
    def __init__(self, *, mode: str = "success", mismatched_routes: bool = False) -> None:
        self.mode = mode
        self.mismatched_routes = mismatched_routes

    def Process(self, *, target: Any, args: tuple[Any, ...]) -> _FakeProcess:
        assert target is public_tp._worker
        return _FakeProcess(args, mode=self.mode, mismatched_routes=self.mismatched_routes)


def _prepare_fake_orchestration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    mode: str = "success",
    mismatched_routes: bool = False,
) -> Path:
    result = _result_payload()
    source = public_tp.PublicMoeSourceEvidence.model_validate(result["source"])
    reference = {
        "input_sha256": result["input_sha256"],
        "layers": [
            {"reference_output_sha256": layer["reference_output_sha256"]}
            for layer in result["layers"]
        ],
    }
    monkeypatch.setattr(
        public_tp,
        "probe_gloo_capability",
        lambda: SimpleNamespace(status="available", reason=None),
    )
    monkeypatch.setattr(public_tp, "audit_stories15m_snapshot", lambda _snapshot: source)
    monkeypatch.setattr(
        public_tp,
        "_build_reference_files",
        lambda _snapshot, directory: (
            directory / "reference.safetensors",
            directory / "hashes.json",
            reference,
        ),
    )
    monkeypatch.setattr(
        multiprocessing,
        "get_context",
        lambda method: (
            _FakeSpawnContext(mode=mode, mismatched_routes=mismatched_routes)
            if method == "spawn"
            else pytest.fail("unexpected start method")
        ),
    )
    monkeypatch.setattr(public_tp, "_physical_memory_bytes", lambda: 16_000_000_000)
    snapshot = tmp_path / public_tp.MODEL_REVISION
    snapshot.mkdir(parents=True)
    return snapshot


def test_parent_orchestration_builds_strict_result_from_two_worker_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = _prepare_fake_orchestration(monkeypatch, tmp_path)

    result = run_public_moe_expert_tensor_parallel(snapshot, hardware_label="unit-test CPU")

    assert tuple(rank.rank for rank in result.ranks) == (0, 1)
    assert tuple(layer.layer_id for layer in result.layers) == tuple(range(6))
    assert result.hardware.physical_memory_bytes == 16_000_000_000
    assert result.interface in {"lo", "lo0"}


@pytest.mark.parametrize(
    ("mode", "mismatched_routes", "message"),
    (
        ("timeout", False, "timed out"),
        ("failed", False, "worker failed"),
        ("success", True, "different router selections"),
    ),
)
def test_parent_orchestration_reports_worker_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    mismatched_routes: bool,
    message: str,
) -> None:
    snapshot = _prepare_fake_orchestration(
        monkeypatch,
        tmp_path,
        mode=mode,
        mismatched_routes=mismatched_routes,
    )

    with pytest.raises(CapabilityError, match=message):
        run_public_moe_expert_tensor_parallel(
            snapshot, hardware_label="unit-test CPU", timeout_seconds=1
        )


def test_parent_orchestration_rejects_unavailable_capability_and_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        public_tp,
        "probe_gloo_capability",
        lambda: SimpleNamespace(status="unavailable", reason="no gloo"),
    )
    with pytest.raises(CapabilityError, match="no gloo"):
        run_public_moe_expert_tensor_parallel(tmp_path, hardware_label="unit-test CPU")

    snapshot = _prepare_fake_orchestration(monkeypatch, tmp_path / "second")
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    with pytest.raises(CapabilityError, match="logical CPU count"):
        run_public_moe_expert_tensor_parallel(snapshot, hardware_label="unit-test CPU")


def test_parent_orchestration_rejects_secret_label_and_invalid_tolerances(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        public_tp,
        "probe_gloo_capability",
        lambda: SimpleNamespace(status="available", reason=None),
    )
    with pytest.raises(ValueError, match="credential-like"):
        run_public_moe_expert_tensor_parallel(
            tmp_path,
            hardware_label="Authorization: Bearer abc123",
        )
    with pytest.raises(ValueError, match="tolerances"):
        run_public_moe_expert_tensor_parallel(
            tmp_path,
            hardware_label="unit-test CPU",
            absolute_tolerance=0.0,
        )


class _TensorSlice:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value

    def __getitem__(self, index: Any) -> torch.Tensor:
        return self.value[index]


class _TinyCheckpoint:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def __enter__(self) -> _TinyCheckpoint:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.tensors[name]

    def get_slice(self, name: str) -> _TensorSlice:
        return _TensorSlice(self.tensors[name])


def _tiny_worker_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_TinyCheckpoint, _TinyCheckpoint, dict[str, dict[str, str]]]:
    monkeypatch.setattr(public_tp, "_LAYER_COUNT", 1)
    monkeypatch.setattr(public_tp, "_EXPERT_COUNT", 2)
    monkeypatch.setattr(public_tp, "_TOP_K", 2)
    monkeypatch.setattr(public_tp, "_HIDDEN_SIZE", 2)
    monkeypatch.setattr(public_tp, "_INTERMEDIATE_SIZE", 4)
    monkeypatch.setattr(public_tp, "_LOCAL_INTERMEDIATE_SIZE", 2)
    monkeypatch.setattr(public_tp, "_INPUT_RANGES", ((0, 2),))

    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.tensor(((1.0, 0.5), (0.5, 1.0))),
        public_tp._router_tensor_name(0): torch.tensor(((1.0, 0.0), (0.0, 1.0))),
    }
    for expert in range(2):
        scale = float(expert + 1)
        tensors[public_tp._expert_tensor_name(0, expert, "w1")] = torch.full((4, 2), scale / 4)
        tensors[public_tp._expert_tensor_name(0, expert, "w2")] = torch.full((2, 4), scale / 8)
        tensors[public_tp._expert_tensor_name(0, expert, "w3")] = torch.full((4, 2), scale / 6)
    source = _TinyCheckpoint(tensors)
    hidden = public_tp._workload_input(source)
    weights, indices = public_tp._routing(hidden, tensors[public_tp._router_tensor_name(0)])
    reference = public_tp._full_expert_forward(
        source,
        layer=0,
        hidden_states=hidden,
        top_k_weights=weights,
        top_k_indices=indices,
    )
    expected = _TinyCheckpoint({"input": hidden, "layer_0": reference})
    hashes = {
        name: {
            "source_tensor_sha256": public_tp._tensor_sha256(tensor),
            "source_float32_sha256": public_tp._tensor_sha256(tensor.float()),
        }
        for name, tensor in tensors.items()
        if ".experts." in name
    }
    return source, expected, hashes


def test_worker_executes_tiny_sharded_forward_and_writes_strict_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, expected, hashes = _tiny_worker_checkpoints(monkeypatch)
    hashes_path = tmp_path / "hashes.json"
    hashes_path.write_text(json.dumps(hashes), encoding="utf-8")
    expected_path = tmp_path / "expected.safetensors"
    expected_path.touch()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").touch()
    result_directory = tmp_path / "results"
    result_directory.mkdir()

    monkeypatch.setattr(
        public_tp,
        "safe_open",
        lambda path, **_kwargs: expected if Path(path) == expected_path else source,
    )
    monkeypatch.setattr(torch_dist, "init_process_group", lambda **_kwargs: None)
    monkeypatch.setattr(torch_dist, "is_initialized", lambda: True)
    destroyed: list[bool] = []
    monkeypatch.setattr(torch_dist, "destroy_process_group", lambda: destroyed.append(True))
    monkeypatch.setattr(
        torch_dist,
        "all_gather",
        lambda gathered, local: [target.copy_(local) for target in gathered],
    )
    reference = expected.get_tensor("layer_0")
    monkeypatch.setattr(torch_dist, "all_reduce", lambda output, **_kwargs: output.copy_(reference))
    monkeypatch.setattr(torch, "set_num_threads", lambda _count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda _count: None)

    public_tp._worker(
        0,
        str(tmp_path / "init"),
        str(result_directory),
        str(snapshot),
        str(expected_path),
        str(hashes_path),
        "lo",
        1.0,
        1e-7,
        1e-5,
    )

    payload = json.loads((result_directory / "rank-0.json").read_text(encoding="utf-8"))
    assert payload["rank"] == 0
    assert len(payload["shard_evidence"]) == 6
    assert payload["layers"][0]["selected_expert_counts"] == [2, 2]
    assert destroyed == [True]


def test_worker_records_typed_failure_and_preserves_original_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_directory = tmp_path / "results"
    result_directory.mkdir()
    monkeypatch.setattr(
        torch_dist,
        "init_process_group",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("gloo failed")),
    )
    monkeypatch.setattr(torch_dist, "is_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="gloo failed"):
        public_tp._worker(
            1,
            str(tmp_path / "init"),
            str(result_directory),
            str(tmp_path),
            str(tmp_path / "expected"),
            str(tmp_path / "hashes"),
            "lo",
            1.0,
            1e-7,
            1e-5,
        )

    error = json.loads((result_directory / "rank-1.error.json").read_text(encoding="utf-8"))
    assert error == {"rank": 1, "error_type": "RuntimeError", "error": "gloo failed"}


def test_reference_builder_writes_expected_tensors_and_source_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, _expected, hashes = _tiny_worker_checkpoints(monkeypatch)
    monkeypatch.setattr(public_tp, "safe_open", lambda *_args, **_kwargs: source)

    expected_path, hashes_path, metadata = public_tp._build_reference_files(tmp_path, tmp_path)

    assert expected_path.exists()
    assert json.loads(hashes_path.read_text(encoding="utf-8")) == hashes
    assert metadata["input_sha256"] == public_tp._tensor_sha256(
        source.tensors["model.embed_tokens.weight"]
    )
    assert metadata["layers"][0]["selected_expert_counts"] == [2, 2]


def _tiny_audit_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(public_tp, "_LAYER_COUNT", 1)
    monkeypatch.setattr(public_tp, "_EXPERT_COUNT", 2)
    monkeypatch.setattr(public_tp, "_TOP_K", 1)
    monkeypatch.setattr(public_tp, "_HIDDEN_SIZE", 2)
    monkeypatch.setattr(public_tp, "_INTERMEDIATE_SIZE", 4)
    root = tmp_path / "cache" / "snapshots" / public_tp.MODEL_REVISION
    root.mkdir(parents=True)
    config = {
        "model_type": "mixtral",
        "architectures": ["MixtralForCausalLM"],
        "hidden_size": 2,
        "intermediate_size": 4,
        "num_hidden_layers": 1,
        "num_local_experts": 2,
        "num_experts_per_tok": 1,
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    tensors: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.zeros((32_000, 2), dtype=torch.float16),
        public_tp._router_tensor_name(0): torch.zeros((2, 2), dtype=torch.float32),
    }
    for expert in range(2):
        tensors[public_tp._expert_tensor_name(0, expert, "w1")] = torch.zeros(
            (4, 2), dtype=torch.float16
        )
        tensors[public_tp._expert_tensor_name(0, expert, "w2")] = torch.zeros(
            (2, 4), dtype=torch.float16
        )
        tensors[public_tp._expert_tensor_name(0, expert, "w3")] = torch.zeros(
            (4, 2), dtype=torch.float16
        )
    for index in range(117 - len(tensors)):
        tensors[f"unused.{index}"] = torch.zeros(1)
    save_path = root / "model.safetensors"
    save_file(tensors, save_path, metadata={"format": "pt"})
    monkeypatch.setattr(
        public_tp,
        "_EXPECTED_SOURCE_FILES",
        {
            path.name: (path.stat().st_size, public_tp._file_sha256(path))
            for path in (root / "config.json", save_path)
        },
    )
    return root


def test_source_audit_accepts_only_exact_safe_architecture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _tiny_audit_snapshot(monkeypatch, tmp_path)

    source = audit_stories15m_snapshot(root)

    assert tuple(item.path for item in source.file_inventory) == (
        "config.json",
        "model.safetensors",
    )
    assert all(item.symlink is False for item in source.file_inventory)


def test_source_audit_rejects_executable_or_custom_model_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _tiny_audit_snapshot(monkeypatch, tmp_path)
    executable = root / "model.py"
    executable.write_text("raise RuntimeError", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe executable"):
        audit_stories15m_snapshot(root)

    executable.unlink()
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["auto_map"] = {"AutoModel": "model.Custom"}
    config_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(
        public_tp,
        "_EXPECTED_SOURCE_FILES",
        {
            path.name: (path.stat().st_size, public_tp._file_sha256(path))
            for path in (config_path, root / "model.safetensors")
        },
    )
    with pytest.raises(ValueError, match="custom model code"):
        audit_stories15m_snapshot(root)
