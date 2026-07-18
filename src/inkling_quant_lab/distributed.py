"""Truthful local Gloo collective and tiny-model tensor-parallel evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from queue import Empty
from typing import Any, Literal, Self, cast

import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from inkling_quant_lab.exceptions import CapabilityError
from inkling_quant_lab.security import sensitive_literal_path

_SCOPE = (
    "local CPU two-process gloo all-reduce only; does not validate model sharding, "
    "distributed inference, routing, checkpointing, or performance"
)
_TENSOR_PARALLEL_SCOPE = (
    "local two-process CPU DTensor tensor-parallel forward for one deterministic tiny MLP; "
    "validates that its two linear parameters are sharded, reconstructed, and numerically "
    "equivalent to a local float32 reference; does not validate public-model or MoE sharding, "
    "distributed training, routing, checkpointing, performance, fault tolerance, or multi-host "
    "execution"
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class _TinyTensorParallelMLP(nn.Module):
    """Two-linears fixture whose hidden dimension is evenly sharded over two ranks."""

    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(8, 12, bias=False)
        self.down = nn.Linear(12, 5, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.down(torch.relu(self.up(inputs))))


class GlooCapability(BaseModel):
    """Compiled collective availability without an inferred runtime support claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["available", "unavailable"]
    backend: Literal["gloo"] = "gloo"
    availability_kind: Literal["compiled_backend"] = "compiled_backend"
    smoke_tested: Literal[False] = False
    torch_version: str = Field(min_length=1)
    supports_collectives: bool
    supports_model_sharding: Literal[False] = False
    scope: str = _SCOPE
    reason: str | None = None

    @model_validator(mode="after")
    def status_is_consistent(self) -> Self:
        if self.status == "available" and (
            not self.supports_collectives or self.reason is not None
        ):
            raise ValueError("available gloo capability requires collectives without a reason")
        if self.status == "unavailable" and (self.supports_collectives or not self.reason):
            raise ValueError("unavailable gloo capability requires an explicit reason")
        return self


class GlooSmokeResult(BaseModel):
    """Evidence from a completed local two-process sum all-reduce."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["gloo"] = "gloo"
    world_size: Literal[2] = 2
    rank_results: tuple[float, float]
    expected_sum: float
    interface: str = Field(min_length=1)
    scope: str = _SCOPE
    supports_model_sharding: Literal[False] = False

    @model_validator(mode="after")
    def all_ranks_observed_the_expected_sum(self) -> Self:
        if any(value != self.expected_sum for value in self.rank_results):
            raise ValueError("gloo smoke ranks did not observe the expected reduction sum")
        return self


class TensorParallelShardEvidence(BaseModel):
    """One rank's real local shard and collectively reconstructed parameter evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameter_name: Literal["up.weight", "down.weight"]
    global_shape: tuple[int, int]
    local_shape: tuple[int, int]
    placement: Literal["Shard(dim=0)", "Shard(dim=1)"]
    shard_dim: Literal[0, 1]
    shard_range: tuple[int, int]
    global_numel: int = Field(gt=0)
    local_numel: int = Field(gt=0)
    local_sha256: str = Field(pattern=_SHA256_PATTERN)
    reconstructed_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_full_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def shard_contract_is_consistent(self) -> Self:
        if self.placement != f"Shard(dim={self.shard_dim})":
            raise ValueError("placement must identify the recorded shard dimension")
        start, end = self.shard_range
        if start < 0 or end <= start or end > self.global_shape[self.shard_dim]:
            raise ValueError("shard range must be a non-empty interval within the global tensor")
        if end - start != self.local_shape[self.shard_dim]:
            raise ValueError("shard range length must equal the local sharded dimension")
        if self.global_numel != self.global_shape[0] * self.global_shape[1]:
            raise ValueError("global numel does not match the global shape")
        if self.local_numel != self.local_shape[0] * self.local_shape[1]:
            raise ValueError("local numel does not match the local shape")
        if self.reconstructed_sha256 != self.source_full_sha256:
            raise ValueError("reconstructed parameter checksum differs from the source parameter")
        return self


class TensorParallelRankEvidence(BaseModel):
    """One CPU rank's parameter placement and forward-equivalence evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: Literal[0, 1]
    mesh_coordinate: tuple[Literal[0, 1]]
    device: Literal["cpu"] = "cpu"
    intraop_threads: Literal[1] = 1
    interop_threads: Literal[1] = 1
    parameter_shards: tuple[TensorParallelShardEvidence, TensorParallelShardEvidence]
    reference_output_sha256: str = Field(pattern=_SHA256_PATTERN)
    tensor_parallel_output_sha256: str = Field(pattern=_SHA256_PATTERN)
    max_abs_error: float = Field(ge=0.0, allow_inf_nan=False)
    max_relative_error: float = Field(ge=0.0, allow_inf_nan=False)
    outputs_match_reference: Literal[True] = True

    @model_validator(mode="after")
    def rank_contract_is_consistent(self) -> Self:
        if self.mesh_coordinate != (self.rank,):
            raise ValueError("mesh coordinate must equal the process rank")
        if tuple(shard.parameter_name for shard in self.parameter_shards) != (
            "up.weight",
            "down.weight",
        ):
            raise ValueError("rank evidence must retain both tensor-parallel parameters in order")
        return self


class TensorParallelModelSpec(BaseModel):
    """Exact deterministic model and tensor-parallel plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Literal["tiny_tensor_parallel_mlp"] = "tiny_tensor_parallel_mlp"
    architecture: Literal["Linear(8,12,bias=False)->ReLU->Linear(12,5,bias=False)"] = (
        "Linear(8,12,bias=False)->ReLU->Linear(12,5,bias=False)"
    )
    parameter_count: Literal[156] = 156
    up_parallel_style: Literal["ColwiseParallel"] = "ColwiseParallel"
    up_weight_placement: Literal["Shard(dim=0)"] = "Shard(dim=0)"
    down_parallel_style: Literal["RowwiseParallel"] = "RowwiseParallel"
    down_weight_placement: Literal["Shard(dim=1)"] = "Shard(dim=1)"
    output_layout: Literal["replicated_local_tensor"] = "replicated_local_tensor"


class TensorParallelHardwareEvidence(BaseModel):
    """Host facts attached to an executed local tensor-parallel result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    system: str = Field(min_length=1)
    release: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    processor: str
    logical_cpu_count: int = Field(gt=0)
    physical_memory_bytes: int | None = Field(default=None, gt=0)


class TensorParallelSoftwareEvidence(BaseModel):
    """Exact interpreter and PyTorch identities for the execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    python: str = Field(min_length=1)
    torch: str = Field(min_length=1)
    torch_git_version: str | None = None


class LocalCPUTensorParallelResult(BaseModel):
    """Executed two-rank parameter-sharded tiny-model forward evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["local-cpu-tensor-parallel-smoke-v1"] = (
        "local-cpu-tensor-parallel-smoke-v1"
    )
    backend: Literal["gloo"] = "gloo"
    api: Literal["torch.distributed.tensor.parallel.parallelize_module"] = (
        "torch.distributed.tensor.parallel.parallelize_module"
    )
    world_size: Literal[2] = 2
    device_mesh: Literal["1d_cpu_mesh_named_tp"] = "1d_cpu_mesh_named_tp"
    source_data_rank: Literal[0] = 0
    rendezvous: Literal["ephemeral_file"] = "ephemeral_file"
    process_start_method: Literal["spawn"] = "spawn"
    interface: str = Field(min_length=1)
    seed: int = Field(ge=0, strict=True)
    dtype: Literal["float32"] = "float32"
    model: TensorParallelModelSpec
    input_shape: tuple[Literal[3], Literal[8]] = (3, 8)
    input_sha256: str = Field(pattern=_SHA256_PATTERN)
    checksum_contract: Literal[
        "sha256(torch dtype ASCII + NUL + compact JSON shape + NUL + contiguous CPU tensor bytes)"
    ] = "sha256(torch dtype ASCII + NUL + compact JSON shape + NUL + contiguous CPU tensor bytes)"
    absolute_tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    relative_tolerance: float = Field(gt=0.0, allow_inf_nan=False)
    rank_results: tuple[TensorParallelRankEvidence, TensorParallelRankEvidence]
    parameter_sharding_executed: Literal[True] = True
    tiny_model_forward_validated: Literal[True] = True
    public_model_sharding_validated: Literal[False] = False
    distributed_training_validated: Literal[False] = False
    routing_validated: Literal[False] = False
    checkpointing_validated: Literal[False] = False
    performance_validated: Literal[False] = False
    multi_host_validated: Literal[False] = False
    scope: str = _TENSOR_PARALLEL_SCOPE
    operator_declared_hardware_label: str = Field(min_length=1)
    hardware: TensorParallelHardwareEvidence
    software: TensorParallelSoftwareEvidence

    @model_validator(mode="after")
    def execution_contract_is_consistent(self) -> Self:
        if self.operator_declared_hardware_label != self.operator_declared_hardware_label.strip():
            raise ValueError("hardware label must be non-empty without boundary whitespace")
        if sensitive_literal_path(self.operator_declared_hardware_label) is not None:
            raise ValueError("hardware label must not contain credential-like material")
        if tuple(rank.rank for rank in self.rank_results) != (0, 1):
            raise ValueError("tensor-parallel evidence must retain ranks zero and one in order")
        if len({rank.tensor_parallel_output_sha256 for rank in self.rank_results}) != 1:
            raise ValueError("tensor-parallel ranks produced different replicated outputs")
        if len({rank.reference_output_sha256 for rank in self.rank_results}) != 1:
            raise ValueError("tensor-parallel ranks used different reference outputs")
        for rank in self.rank_results:
            if rank.max_abs_error > self.absolute_tolerance:
                raise ValueError("rank maximum absolute error exceeds the recorded tolerance")
            if rank.max_relative_error > self.relative_tolerance:
                raise ValueError("rank maximum relative error exceeds the recorded tolerance")

        expected_parameters = (
            ("up.weight", (12, 8), (6, 8), 0, "Shard(dim=0)"),
            ("down.weight", (5, 12), (5, 6), 1, "Shard(dim=1)"),
        )
        for parameter_index in range(2):
            shards = tuple(rank.parameter_shards[parameter_index] for rank in self.rank_results)
            expected_name, expected_global, expected_local, expected_dim, expected_placement = (
                expected_parameters[parameter_index]
            )
            for shard in shards:
                if (
                    shard.parameter_name != expected_name
                    or shard.global_shape != expected_global
                    or shard.local_shape != expected_local
                    or shard.shard_dim != expected_dim
                    or shard.placement != expected_placement
                ):
                    raise ValueError("parameter shard metadata differs from the exact model plan")
                for dimension, local_extent in enumerate(shard.local_shape):
                    if (
                        dimension != shard.shard_dim
                        and local_extent != shard.global_shape[dimension]
                    ):
                        raise ValueError("unsharded local dimensions must equal global dimensions")
            global_metadata = {
                (
                    shard.parameter_name,
                    shard.global_shape,
                    shard.global_numel,
                    shard.shard_dim,
                    shard.placement,
                    shard.source_full_sha256,
                    shard.reconstructed_sha256,
                )
                for shard in shards
            }
            if len(global_metadata) != 1:
                raise ValueError("ranks retained inconsistent global parameter metadata")
            if len({shard.reconstructed_sha256 for shard in shards}) != 1:
                raise ValueError("ranks reconstructed different full parameters")
            if len({shard.local_sha256 for shard in shards}) != 2:
                raise ValueError("the deterministic fixture must retain distinct local shards")
            if sum(shard.local_numel for shard in shards) != shards[0].global_numel:
                raise ValueError("local shard sizes must partition the full parameter")
            if shards[0].shard_range[0] != 0:
                raise ValueError("rank zero shard range must start at the global origin")
            if shards[0].shard_range[1] != shards[1].shard_range[0]:
                raise ValueError("rank shard ranges must form one contiguous partition")
            if shards[1].shard_range[1] != shards[0].global_shape[shards[0].shard_dim]:
                raise ValueError("rank one shard range must end at the global dimension")
        return self


def probe_gloo_capability() -> GlooCapability:
    """Report compiled gloo presence without claiming that a process group was started."""

    available = dist.is_available() and dist.is_gloo_available()
    return GlooCapability(
        status="available" if available else "unavailable",
        torch_version=torch.__version__,
        supports_collectives=available,
        reason=None if available else "this PyTorch build does not provide torch.distributed gloo",
    )


def _gloo_smoke_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_queue: Any,
    timeout_seconds: float,
    interface: str,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = interface
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )
    try:
        value = torch.tensor([float(rank + 1)], dtype=torch.float64)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        result_queue.put((rank, float(value.item())))
    finally:
        dist.destroy_process_group()


def _tensor_sha256(tensor: Tensor) -> str:
    """Hash tensor identity, shape, and exact contiguous CPU bytes."""

    normalized = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(normalized.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(normalized.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(normalized.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _physical_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    total = page_size * page_count
    return total if total > 0 else None


def _collect_tensor_parallel_payloads(
    result_queue: Any,
    *,
    exit_codes: list[int | None],
    world_size: int,
    timeout_seconds: float,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Collect successful payloads without waiting after workers already failed."""

    payloads: list[dict[str, Any]] = []
    if exit_codes != [0, 0]:
        for _ in range(world_size):
            try:
                payloads.append(result_queue.get_nowait())
            except Empty:
                break
        raise CapabilityError(
            "local CPU tensor-parallel smoke worker failed",
            component="distributed_cpu_tensor_parallel",
            details={
                "exit_codes": exit_codes,
                "worker_failures": [payload for payload in payloads if "error" in payload],
            },
        )

    deadline = time.monotonic() + timeout_seconds
    for _ in range(world_size):
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise CapabilityError(
                "local CPU tensor-parallel smoke did not return every rank result",
                component="distributed_cpu_tensor_parallel",
            )
        try:
            payloads.append(result_queue.get(timeout=remaining))
        except Empty as error:
            raise CapabilityError(
                "local CPU tensor-parallel smoke did not return every rank result",
                component="distributed_cpu_tensor_parallel",
            ) from error

    worker_failures = [payload for payload in payloads if "error" in payload]
    if worker_failures:
        raise CapabilityError(
            "local CPU tensor-parallel smoke worker failed",
            component="distributed_cpu_tensor_parallel",
            details={
                "exit_codes": exit_codes,
                "worker_failures": worker_failures,
            },
        )
    rank_payloads = {int(payload["rank"]): payload for payload in payloads}
    return rank_payloads, worker_failures


def _tensor_parallel_worker(
    rank: int,
    world_size: int,
    init_file: str,
    result_queue: Any,
    timeout_seconds: float,
    interface: str,
    seed: int,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    os.environ["GLOO_SOCKET_IFNAME"] = interface
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(seed)
        reference = _TinyTensorParallelMLP().eval()
        inputs = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 16.0
        with torch.no_grad():
            expected = reference(inputs)
        source_parameter_sha256 = {
            name: _tensor_sha256(parameter) for name, parameter in reference.named_parameters()
        }

        model: nn.Module = _TinyTensorParallelMLP().eval()
        model.load_state_dict(reference.state_dict())
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))
        model = parallelize_module(
            model,
            mesh,
            {
                "up": ColwiseParallel(),
                "down": RowwiseParallel(),
            },
            src_data_rank=0,
        )
        with torch.no_grad():
            output = model(inputs)
        torch.testing.assert_close(
            output,
            expected,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        )

        parameter_shards: list[dict[str, Any]] = []
        for name, parameter in model.named_parameters():
            if not isinstance(parameter, DTensor):
                raise TypeError(f"tensor-parallel parameter {name} is not a DTensor")
            if len(parameter.placements) != 1 or not isinstance(parameter.placements[0], Shard):
                raise ValueError(
                    f"tensor-parallel parameter {name} is not one-dimensionally sharded"
                )
            placement = parameter.placements[0]
            shard_dim = int(placement.dim)
            local = parameter.to_local().detach()
            with torch.no_grad():
                reconstructed = parameter.full_tensor()
            global_shape = tuple(int(value) for value in parameter.shape)
            local_shape = tuple(int(value) for value in local.shape)
            shard_start = rank * local_shape[shard_dim]
            shard_end = shard_start + local_shape[shard_dim]
            parameter_shards.append(
                {
                    "parameter_name": name,
                    "global_shape": global_shape,
                    "local_shape": local_shape,
                    "placement": f"Shard(dim={shard_dim})",
                    "shard_dim": shard_dim,
                    "shard_range": (shard_start, shard_end),
                    "global_numel": parameter.numel(),
                    "local_numel": local.numel(),
                    "local_sha256": _tensor_sha256(local),
                    "reconstructed_sha256": _tensor_sha256(reconstructed),
                    "source_full_sha256": source_parameter_sha256[name],
                }
            )

        difference = (output - expected).abs()
        relative_difference = difference / expected.abs().clamp_min(torch.finfo(torch.float32).eps)
        result_queue.put(
            {
                "rank": rank,
                "mesh_coordinate": (rank,),
                "device": "cpu",
                "intraop_threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
                "parameter_shards": parameter_shards,
                "reference_output_sha256": _tensor_sha256(expected),
                "tensor_parallel_output_sha256": _tensor_sha256(output),
                "max_abs_error": float(difference.max().item()),
                "max_relative_error": float(relative_difference.max().item()),
                "outputs_match_reference": True,
            }
        )
    except (AssertionError, ImportError, RuntimeError, TypeError, ValueError) as error:
        result_queue.put(
            {
                "rank": rank,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        raise
    finally:
        dist.destroy_process_group()


def run_local_gloo_smoke(*, timeout_seconds: float = 20.0) -> GlooSmokeResult:
    """Execute a real two-process local CPU all-reduce, or fail with scoped context."""

    capability = probe_gloo_capability()
    if capability.status != "available":
        raise CapabilityError(
            capability.reason or "gloo unavailable",
            component="distributed_gloo",
        )
    if timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive")
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    world_size = 2
    interface = "lo0" if platform.system() == "Darwin" else "lo"
    try:
        with tempfile.TemporaryDirectory(prefix="iql-gloo-") as directory:
            init_file = str(Path(directory) / "process-group-init")
            processes = [
                context.Process(
                    target=_gloo_smoke_worker,
                    args=(
                        rank,
                        world_size,
                        init_file,
                        result_queue,
                        timeout_seconds,
                        interface,
                    ),
                )
                for rank in range(world_size)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout_seconds)
            alive = [process for process in processes if process.is_alive()]
            for process in alive:
                process.terminate()
                process.join()
            if alive:
                raise CapabilityError(
                    "local gloo smoke timed out",
                    component="distributed_gloo",
                    details={"timeout_seconds": timeout_seconds},
                )
            exit_codes = [process.exitcode for process in processes]
            if exit_codes != [0, 0]:
                raise CapabilityError(
                    "local gloo smoke worker failed",
                    component="distributed_gloo",
                    details={"exit_codes": exit_codes},
                )
            results: dict[int, float] = {}
            try:
                for _ in range(world_size):
                    rank, value = result_queue.get(timeout=timeout_seconds)
                    results[int(rank)] = float(value)
            except Empty as error:
                raise CapabilityError(
                    "local gloo smoke did not return every rank result",
                    component="distributed_gloo",
                ) from error
    finally:
        result_queue.close()
        result_queue.join_thread()
    expected = float(world_size * (world_size + 1) // 2)
    return GlooSmokeResult(
        rank_results=(results[0], results[1]),
        expected_sum=expected,
        interface=interface,
    )


def run_local_cpu_tensor_parallel_smoke(
    *,
    hardware_label: str,
    seed: int = 20_260_716,
    timeout_seconds: float = 30.0,
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-6,
) -> LocalCPUTensorParallelResult:
    """Execute a real two-rank CPU tensor-parallel tiny-model forward."""

    capability = probe_gloo_capability()
    if capability.status != "available":
        raise CapabilityError(
            capability.reason or "gloo unavailable",
            component="distributed_cpu_tensor_parallel",
        )
    try:
        tensor_parallel_spec = importlib.util.find_spec("torch.distributed.tensor.parallel")
    except (ImportError, ModuleNotFoundError, ValueError) as error:
        raise CapabilityError(
            f"PyTorch tensor-parallel API discovery failed: {error}",
            component="distributed_cpu_tensor_parallel",
        ) from error
    if tensor_parallel_spec is None:
        raise CapabilityError(
            "this PyTorch build does not provide torch.distributed.tensor.parallel",
            component="distributed_cpu_tensor_parallel",
        )
    if not hardware_label or hardware_label != hardware_label.strip():
        raise ValueError("hardware_label must be non-empty without boundary whitespace")
    if sensitive_literal_path(hardware_label) is not None:
        raise ValueError("hardware_label must not contain credential-like material")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be finite and positive")
    if (
        not math.isfinite(absolute_tolerance)
        or not math.isfinite(relative_tolerance)
        or absolute_tolerance <= 0.0
        or relative_tolerance <= 0.0
    ):
        raise ValueError("tensor-parallel numerical tolerances must be finite and positive")

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    world_size = 2
    interface = "lo0" if platform.system() == "Darwin" else "lo"
    try:
        with tempfile.TemporaryDirectory(prefix="iql-cpu-tensor-parallel-") as directory:
            init_file = str(Path(directory) / "process-group-init")
            processes = [
                context.Process(
                    target=_tensor_parallel_worker,
                    args=(
                        rank,
                        world_size,
                        init_file,
                        result_queue,
                        timeout_seconds,
                        interface,
                        seed,
                        absolute_tolerance,
                        relative_tolerance,
                    ),
                )
                for rank in range(world_size)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout_seconds)
            alive = [process for process in processes if process.is_alive()]
            for process in alive:
                process.terminate()
                process.join()
            if alive:
                raise CapabilityError(
                    "local CPU tensor-parallel smoke timed out",
                    component="distributed_cpu_tensor_parallel",
                    details={"timeout_seconds": timeout_seconds},
                )
            exit_codes = [process.exitcode for process in processes]
            rank_payloads, _ = _collect_tensor_parallel_payloads(
                result_queue,
                exit_codes=exit_codes,
                world_size=world_size,
                timeout_seconds=timeout_seconds,
            )
            if tuple(sorted(rank_payloads)) != (0, 1):
                raise CapabilityError(
                    "local CPU tensor-parallel smoke did not retain ranks zero and one",
                    component="distributed_cpu_tensor_parallel",
                    details={"returned_ranks": sorted(rank_payloads)},
                )
    finally:
        result_queue.close()
        result_queue.join_thread()

    git_version = getattr(torch.version, "git_version", None)
    if not isinstance(git_version, str):
        git_version = None
    logical_cpu_count = os.cpu_count()
    if logical_cpu_count is None or logical_cpu_count <= 0:
        raise CapabilityError(
            "logical CPU count is unavailable for tensor-parallel provenance",
            component="distributed_cpu_tensor_parallel",
        )
    return LocalCPUTensorParallelResult(
        interface=interface,
        seed=seed,
        model=TensorParallelModelSpec(),
        input_sha256=_tensor_sha256(torch.arange(24, dtype=torch.float32).reshape(3, 8) / 16.0),
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        rank_results=(
            TensorParallelRankEvidence.model_validate(rank_payloads[0]),
            TensorParallelRankEvidence.model_validate(rank_payloads[1]),
        ),
        operator_declared_hardware_label=hardware_label,
        hardware=TensorParallelHardwareEvidence(
            system=platform.system(),
            release=platform.release(),
            machine=platform.machine(),
            processor=platform.processor(),
            logical_cpu_count=logical_cpu_count,
            physical_memory_bytes=_physical_memory_bytes(),
        ),
        software=TensorParallelSoftwareEvidence(
            python=platform.python_version(),
            torch=torch.__version__,
            torch_git_version=git_version,
        ),
    )


__all__ = [
    "GlooCapability",
    "GlooSmokeResult",
    "LocalCPUTensorParallelResult",
    "TensorParallelHardwareEvidence",
    "TensorParallelModelSpec",
    "TensorParallelRankEvidence",
    "TensorParallelShardEvidence",
    "TensorParallelSoftwareEvidence",
    "probe_gloo_capability",
    "run_local_cpu_tensor_parallel_smoke",
    "run_local_gloo_smoke",
]
