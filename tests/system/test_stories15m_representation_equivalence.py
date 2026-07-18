"""Opt-in offline proof of both exact Stories15M Mixtral state layouts."""

from __future__ import annotations

import importlib.metadata
import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file
from torch import Tensor

from inkling_quant_lab.config import ExperimentConfig, load_config
from inkling_quant_lab.models.hf_causal_lm import (
    HFCausalLMAdapter,
    HFLinearMixtralCausalLMAdapter,
)
from inkling_quant_lab.models.mixtral_compat import (
    DefuserMixtralBindings,
    capture_defuser_mixtral_bindings,
    restore_defuser_mixtral_bindings,
)
from inkling_quant_lab.models.state import model_state_sha256
from inkling_quant_lab.public_moe_tensor_parallel import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_WEIGHT_SHA256,
    audit_stories15m_snapshot,
)
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = [
    pytest.mark.backend_gptq,
    pytest.mark.model_public,
    pytest.mark.slow,
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs/experiments/hf_stories15m_gptq_cpu_pilot.yaml"
_RUN_EQUIVALENCE = os.environ.get("IQL_RUN_STORIES15M_REPRESENTATION_EQUIVALENCE") == "1"
_OPT_IN = pytest.mark.skipif(
    not _RUN_EQUIVALENCE,
    reason=(
        "set IQL_RUN_STORIES15M_REPRESENTATION_EQUIVALENCE=1 after caching the exact "
        "pinned Stories15M snapshot and installing the exact Transformers/Defuser matrix"
    ),
)

_LAYER_COUNT = 6
_EXPERT_COUNT = 4
_INTERMEDIATE_SIZE = 768
_SOURCE_TENSOR_COUNT = 117
_SOURCE_EXPERT_TENSOR_COUNT = 72
_NATIVE_TENSOR_COUNT = 57
_NATIVE_STATE_SHA256 = "93e36334ff1be21096ca5f59c6b4d8bdfb212c8854b815583110755df75d6ed9"
_LINEAR_STATE_SHA256 = "368fc0265ea7f6b86ef2103ab33e5d929c592521b029f024100d7910b576ed51"
_EXACT_DEPENDENCIES = {
    "defuser": "0.0.23",
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "transformers": "5.12.1",
}


def _require_exact_dependencies() -> None:
    observed: dict[str, str] = {}
    missing: list[str] = []
    for distribution in _EXACT_DEPENDENCIES:
        try:
            observed[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing or observed != _EXACT_DEPENDENCIES:
        pytest.skip(
            "exact representation-conversion dependency matrix is unavailable: "
            f"missing={missing}, observed={observed}, required={_EXACT_DEPENDENCIES}"
        )


def _require_exact_snapshot(config: ExperimentConfig) -> Path:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        pytest.skip("the exact representation test requires huggingface-hub")
    cached = try_to_load_from_cache(
        config.model.model_id,
        "config.json",
        revision=config.model.revision,
    )
    if not isinstance(cached, str):
        pytest.skip("the exact pinned Stories15M snapshot is not cached")
    snapshot = Path(cached).parent
    try:
        source = audit_stories15m_snapshot(snapshot)
    except (OSError, RuntimeError, ValueError) as error:
        pytest.skip(f"cached Stories15M snapshot does not satisfy the exact audit: {error}")
    assert source.weight_sha256 == MODEL_WEIGHT_SHA256
    assert source.safetensors_tensor_count == _SOURCE_TENSOR_COUNT
    return snapshot


def _binding_signature(bindings: DefuserMixtralBindings) -> tuple[tuple[int, str, bool, int], ...]:
    return tuple(
        (id(binding.owner), binding.name, binding.existed, id(binding.value))
        for binding in bindings.attributes
    )


def _source_expert_name(layer: int, expert: int, projection: str) -> str:
    return f"model.layers.{layer}.block_sparse_moe.experts.{expert}.{projection}.weight"


def _linear_name(source_name: str) -> str:
    return (
        source_name.replace(".block_sparse_moe.", ".mlp.")
        .replace(".w1.weight", ".gate_proj.weight")
        .replace(".w2.weight", ".down_proj.weight")
        .replace(".w3.weight", ".up_proj.weight")
    )


def _native_nonexpert_name(source_name: str) -> str:
    return source_name.replace(".block_sparse_moe.", ".mlp.")


def _assert_same_float32(actual: Tensor, source: Tensor) -> None:
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.float32
    assert torch.equal(actual, source.float())


def _assert_complete_value_mapping(
    source_state: dict[str, Tensor],
    native_state: dict[str, Tensor],
    linear_state: dict[str, Tensor],
) -> None:
    expected_expert_names = {
        _source_expert_name(layer, expert, projection)
        for layer in range(_LAYER_COUNT)
        for expert in range(_EXPERT_COUNT)
        for projection in ("w1", "w2", "w3")
    }
    assert len(expected_expert_names) == _SOURCE_EXPERT_TENSOR_COUNT
    assert expected_expert_names < source_state.keys()
    nonexpert_names = source_state.keys() - expected_expert_names
    assert len(nonexpert_names) == _SOURCE_TENSOR_COUNT - _SOURCE_EXPERT_TENSOR_COUNT

    linear_names = {_linear_name(name) for name in source_state}
    assert len(linear_names) == _SOURCE_TENSOR_COUNT
    assert linear_names == linear_state.keys()

    # Each source tensor has one unique, disjoint logical region in the native
    # representation. Expert regions share a fused container but never overlap.
    source_to_native_region: dict[str, tuple[str, int | None, str]] = {}
    for source_name in nonexpert_names:
        native_name = _native_nonexpert_name(source_name)
        source_to_native_region[source_name] = (native_name, None, "whole")
        _assert_same_float32(native_state[native_name], source_state[source_name])
        _assert_same_float32(linear_state[_linear_name(source_name)], source_state[source_name])
        assert torch.equal(native_state[native_name], linear_state[_linear_name(source_name)])

    fused_names: set[str] = set()
    for layer in range(_LAYER_COUNT):
        source_prefix = f"model.layers.{layer}.block_sparse_moe.experts"
        native_prefix = f"model.layers.{layer}.mlp.experts"
        gate_up_name = f"{native_prefix}.gate_up_proj"
        down_name = f"{native_prefix}.down_proj"
        fused_names.update((gate_up_name, down_name))

        expected_gate_up = torch.stack(
            [
                torch.cat(
                    (
                        source_state[f"{source_prefix}.{expert}.w1.weight"],
                        source_state[f"{source_prefix}.{expert}.w3.weight"],
                    ),
                    dim=0,
                )
                for expert in range(_EXPERT_COUNT)
            ],
            dim=0,
        ).float()
        expected_down = torch.stack(
            [
                source_state[f"{source_prefix}.{expert}.w2.weight"]
                for expert in range(_EXPERT_COUNT)
            ],
            dim=0,
        ).float()
        assert torch.equal(native_state[gate_up_name], expected_gate_up)
        assert torch.equal(native_state[down_name], expected_down)

        for expert in range(_EXPERT_COUNT):
            source_names = {
                projection: f"{source_prefix}.{expert}.{projection}.weight"
                for projection in ("w1", "w2", "w3")
            }
            native_regions = {
                "w1": native_state[gate_up_name][expert, :_INTERMEDIATE_SIZE, :],
                "w2": native_state[down_name][expert],
                "w3": native_state[gate_up_name][expert, _INTERMEDIATE_SIZE:, :],
            }
            region_ids = {
                "w1": (gate_up_name, expert, "gate"),
                "w2": (down_name, expert, "down"),
                "w3": (gate_up_name, expert, "up"),
            }
            for projection in ("w1", "w2", "w3"):
                source_name = source_names[projection]
                linear_tensor = linear_state[_linear_name(source_name)]
                native_tensor = native_regions[projection]
                source_to_native_region[source_name] = region_ids[projection]
                _assert_same_float32(native_tensor, source_state[source_name])
                _assert_same_float32(linear_tensor, source_state[source_name])
                assert torch.equal(native_tensor, linear_tensor)

    native_names = {_native_nonexpert_name(name) for name in nonexpert_names} | fused_names
    assert len(native_names) == _NATIVE_TENSOR_COUNT
    assert native_names == native_state.keys()
    assert source_to_native_region.keys() == source_state.keys()
    assert len(set(source_to_native_region.values())) == _SOURCE_TENSOR_COUNT
    assert {region[0] for region in source_to_native_region.values()} == native_state.keys()
    assert sum(tensor.numel() for tensor in source_state.values()) == sum(
        tensor.numel() for tensor in native_state.values()
    )
    assert sum(tensor.numel() for tensor in native_state.values()) == sum(
        tensor.numel() for tensor in linear_state.values()
    )


@_OPT_IN
def test_stories15m_native_fused_and_defuser_linear_states_are_exactly_equivalent() -> None:
    """Reconstruct and invert all 117-to-57 conversion regions without network access."""

    _require_exact_dependencies()
    environment_before = dict(os.environ)
    with (
        patch.dict(
            os.environ,
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
            },
            clear=False,
        ),
        ExitStack() as cleanup,
    ):
        original_bindings = capture_defuser_mixtral_bindings()
        bindings_before = _binding_signature(original_bindings)
        cleanup.callback(restore_defuser_mixtral_bindings, original_bindings)
        native_runtime = TorchEagerCPURuntime()
        linear_runtime = TorchEagerCPURuntime()
        cleanup.callback(linear_runtime.cleanup)
        cleanup.callback(native_runtime.cleanup)
        linear_config = load_config(CONFIG)
        assert linear_config.model.model_id == MODEL_ID
        assert linear_config.model.revision == MODEL_REVISION
        assert linear_config.model.adapter == "hf_causal_lm_linear_mixtral"
        assert linear_config.model.local_files_only is True
        assert linear_config.model.dtype == linear_config.runtime.dtype == "float32"
        snapshot = _require_exact_snapshot(linear_config)
        native_config = linear_config.model_copy(
            update={"model": linear_config.model.model_copy(update={"adapter": "hf_causal_lm"})}
        )

        source_state = load_file(snapshot / "model.safetensors", device="cpu")
        assert len(source_state) == _SOURCE_TENSOR_COUNT
        assert all(tensor.device.type == "cpu" for tensor in source_state.values())

        native_loaded = HFCausalLMAdapter().load(native_config, native_runtime)
        assert _binding_signature(capture_defuser_mixtral_bindings()) == bindings_before
        linear_loaded = HFLinearMixtralCausalLMAdapter().load(
            linear_config,
            linear_runtime,
        )
        assert _binding_signature(capture_defuser_mixtral_bindings()) == bindings_before

        native_block = native_loaded.model.model.layers[0].mlp
        linear_block = linear_loaded.model.model.layers[0].mlp
        assert (
            f"{native_block.__class__.__module__}.{native_block.__class__.__qualname__}"
            == "transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock"
        )
        assert (
            f"{linear_block.__class__.__module__}.{linear_block.__class__.__qualname__}"
            == "defuser.modeling.unfused_moe.mixtral.LinearMixtralSparseMoeBlock"
        )

        native_state = dict(native_loaded.model.state_dict())
        linear_state = dict(linear_loaded.model.state_dict())
        assert len(native_state) == _NATIVE_TENSOR_COUNT
        assert len(linear_state) == _SOURCE_TENSOR_COUNT
        assert native_loaded.descriptor.checksum == _NATIVE_STATE_SHA256
        assert linear_loaded.descriptor.checksum == _LINEAR_STATE_SHA256
        assert model_state_sha256(native_loaded.model) == _NATIVE_STATE_SHA256
        assert model_state_sha256(linear_loaded.model) == _LINEAR_STATE_SHA256
        assert set(tensor.dtype for tensor in native_state.values()) == {torch.float32}
        assert set(tensor.dtype for tensor in linear_state.values()) == {torch.float32}

        _assert_complete_value_mapping(source_state, native_state, linear_state)
        assert _binding_signature(capture_defuser_mixtral_bindings()) == bindings_before

    assert dict(os.environ) == environment_before
