"""Shared model-adapter contract coverage for the validated local fixtures."""

from __future__ import annotations

from typing import cast

import pytest
import torch

from inkling_quant_lab.config import DecodeConfig, ExperimentConfig
from inkling_quant_lab.exceptions import CapabilityError
from inkling_quant_lab.models.base import LoadedModel, ModelBatch
from inkling_quant_lab.models.fixtures import FixedTokenizer
from inkling_quant_lab.models.local import LocalFixtureAdapter
from inkling_quant_lab.routing import BatchMeta, InMemoryRoutingSink
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.contract


def _config(model_id: str, *, routing_mode: str = "off") -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "local-adapter-contract",
            "model": {
                "model_id": model_id,
                "revision": "fixture-contract-v1",
                "adapter": "local_fixture",
                "dtype": "float32",
                "trust_remote_code": False,
                "local_files_only": True,
                "checkpoint_format": "fixture",
            },
            "routing": {"mode": routing_mode, "required": False},
            "benchmark": {"enabled": False},
            "reporting": {"markdown": False, "html": False, "plots": False},
        }
    )


def _load(
    model_id: str, *, routing_mode: str = "off"
) -> tuple[
    ExperimentConfig,
    LocalFixtureAdapter,
    LoadedModel,
    ModelBatch,
]:
    config = _config(model_id, routing_mode=routing_mode)
    adapter = LocalFixtureAdapter()
    loaded = adapter.load(config, TorchEagerCPURuntime())
    tokenizer = cast(FixedTokenizer, loaded.tokenizer)
    input_ids, attention_mask = tokenizer.batch_encode(("alpha route", "beta expert"))
    batch = ModelBatch(
        sample_ids=("contract-alpha", "contract-beta"),
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    return config, adapter, loaded, batch


@pytest.mark.parametrize(
    ("model_id", "is_moe", "supports_images"),
    (
        ("local://fixtures/tiny-dense", False, False),
        ("local://fixtures/tiny-moe", True, False),
        ("local://fixtures/tiny-multimodal", False, True),
    ),
)
def test_local_adapter_capabilities_and_inventory_are_factual_and_stable(
    model_id: str,
    is_moe: bool,
    supports_images: bool,
) -> None:
    """Capability flags and normalized module roles match each loaded fixture."""

    config, adapter, loaded, _batch = _load(model_id)
    capabilities = adapter.capabilities(config)
    first = adapter.enumerate_modules(loaded)
    second = adapter.enumerate_modules(loaded)

    assert capabilities == loaded.descriptor.capabilities
    assert capabilities.supports_text is True
    assert capabilities.is_moe is is_moe
    assert capabilities.supports_images is supports_images
    assert capabilities.supports_audio is supports_images
    assert capabilities.supports_token_level_routes is is_moe
    assert capabilities.requires_remote_code is False
    assert first == second
    assert first
    assert len({module.name for module in first}) == len(first)
    assert all(module.parameter_count > 0 and module.size_bytes > 0 for module in first)
    assert any(module.is_embedding for module in first)
    if model_id != "local://fixtures/tiny-multimodal":
        assert any(module.is_output_head for module in first)
    assert any(module.is_router for module in first) is is_moe
    assert any(module.is_expert for module in first) is is_moe
    assert any(module.is_multimodal_projector for module in first) is supports_images


@pytest.mark.parametrize(
    "model_id",
    ("local://fixtures/tiny-dense", "local://fixtures/tiny-moe"),
)
def test_local_adapter_generation_and_loss_preserve_stable_sample_identity(
    model_id: str,
) -> None:
    """Declared text execution is deterministic and keeps caller-owned sample IDs."""

    _config_value, adapter, loaded, batch = _load(model_id)
    decode = DecodeConfig(max_new_tokens=2)

    first_generation = adapter.generate(loaded, batch, decode)
    second_generation = adapter.generate(loaded, batch, decode)
    first_loss = adapter.forward_loss(loaded, batch)
    second_loss = adapter.forward_loss(loaded, batch)

    assert first_generation == second_generation
    assert first_generation.sample_ids == batch.sample_ids
    assert all(len(tokens) == 2 for tokens in first_generation.token_ids)
    assert first_loss == second_loss
    assert first_loss.sample_ids == batch.sample_ids
    assert first_loss.mean_nll > 0.0


def test_local_moe_routing_contract_and_hook_cleanup() -> None:
    """MoE discovery, event capture, output preservation, and cleanup are proven together."""

    config, adapter, loaded, batch = _load("local://fixtures/tiny-moe", routing_mode="aggregate")
    descriptor = adapter.discover_moe(loaded)
    assert descriptor is not None
    assert len(descriptor.layers) == 2
    assert all(layer.expert_count == 4 and layer.top_k == 2 for layer in descriptor.layers)

    expected_loss = adapter.forward_loss(loaded, batch)
    sink = InMemoryRoutingSink(
        "aggregate",
        expert_counts={layer.layer_id: layer.expert_count for layer in descriptor.layers},
    )
    handle = adapter.attach_routing_hooks(loaded, sink, config)
    sink.start_batch(BatchMeta(sample_ids=batch.sample_ids, batch_id="contract"))
    actual_loss = adapter.forward_loss(loaded, batch)
    sink.end_batch()
    handle.remove()
    handle.remove()

    # A leaked hook would add events or raise because no sink batch is active.
    adapter.forward_loss(loaded, batch)
    artifact = sink.close()

    assert actual_loss == expected_loss
    assert artifact.batch_count == 1
    assert artifact.observed_event_count == int(batch.input_ids.numel()) * 2
    assert artifact.recorded_event_count == 0
    assert set(artifact.aggregates) == {"moe_0", "moe_1"}
    assert adapter.discover_moe(_load("local://fixtures/tiny-dense")[2]) is None


def test_local_adapter_rejects_unsupported_multimodal_inputs() -> None:
    """Dense text fixtures never silently discard a supplied modality tensor."""

    _config_value, adapter, loaded, batch = _load("local://fixtures/tiny-dense")
    multimodal_batch = ModelBatch(
        sample_ids=batch.sample_ids,
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        multimodal_inputs=torch.ones((len(batch.sample_ids), 4), dtype=torch.float32),
    )

    with pytest.raises(CapabilityError, match="does not support multimodal"):
        adapter.forward_loss(loaded, multimodal_batch)
    with pytest.raises(CapabilityError, match="does not support multimodal"):
        adapter.generate(loaded, multimodal_batch, DecodeConfig(max_new_tokens=1))

    routing = _config("local://fixtures/tiny-dense", routing_mode="aggregate")
    with pytest.raises(CapabilityError, match="Routing analysis is unsupported"):
        adapter.attach_routing_hooks(
            loaded,
            InMemoryRoutingSink("aggregate"),
            routing,
        )

    _multimodal_config, multimodal_adapter, multimodal, multimodal_text_batch = _load(
        "local://fixtures/tiny-multimodal"
    )
    supported_batch = ModelBatch(
        sample_ids=multimodal_text_batch.sample_ids,
        input_ids=multimodal_text_batch.input_ids,
        attention_mask=multimodal_text_batch.attention_mask,
        multimodal_inputs=torch.ones(
            (len(multimodal_text_batch.sample_ids), 4), dtype=torch.float32
        ),
    )
    assert multimodal_adapter.forward_loss(multimodal, supported_batch).mean_nll > 0.0
