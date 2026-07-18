"""Opt-in executable contracts for pinned public native-Mixtral checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from torch import nn

from inkling_quant_lab.config import DecodeConfig, load_inspection_config
from inkling_quant_lab.models.base import ModelBatch
from inkling_quant_lab.models.hf_causal_lm import HFCausalLMAdapter
from inkling_quant_lab.routing import BatchMeta, InMemoryRoutingSink
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = [
    pytest.mark.network,
    pytest.mark.model_public,
    pytest.mark.slow,
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("config_name", "revision", "layer_count", "expert_count", "top_k"),
    (
        (
            "hf_mixtral_tiny_random.yaml",
            "c2faa23d97931c5481999c382d79163b02c4793c",
            2,
            8,
            2,
        ),
        (
            "hf_stories15m_moe.yaml",
            "b6dd737497465570b5f5e962dbc9d9454ed1e0eb",
            6,
            4,
            2,
        ),
    ),
)
def test_public_native_mixtral_adapter_contract(
    config_name: str,
    revision: str,
    layer_count: int,
    expert_count: int,
    top_k: int,
) -> None:
    """Prove safe load, inventory, loss, generation, routing, and hook cleanup."""

    config = load_inspection_config(PROJECT_ROOT / "configs" / "models" / config_name)
    runtime = TorchEagerCPURuntime()
    adapter = HFCausalLMAdapter()
    loaded = adapter.load(config, runtime)
    try:
        capabilities = adapter.capabilities(config)
        inventory = adapter.enumerate_modules(loaded)
        moe = adapter.discover_moe(loaded)
        assert capabilities == loaded.descriptor.capabilities
        assert capabilities.is_moe
        assert capabilities.requires_remote_code is False
        assert loaded.descriptor.revision == revision
        assert len(loaded.descriptor.checksum) == 64
        assert inventory
        assert any(module.is_router for module in inventory)
        assert sum(module.is_expert for module in inventory) >= layer_count * expert_count
        named_modules = dict(loaded.model.named_modules())
        assert all(
            module.supported_precisions == ("float32",)
            for module in inventory
            if module.class_name.endswith(".ExpertSlice")
        )
        assert all(
            isinstance(named_modules.get(module.name), nn.Linear)
            for module in inventory
            if "int8" in module.supported_precisions or "int4" in module.supported_precisions
        )
        assert all(
            module.supported_precisions == ("float32",)
            for module in inventory
            if not isinstance(named_modules.get(module.name), nn.Linear)
        )
        assert moe is not None
        assert len(moe.layers) == layer_count
        assert all(layer.expert_count == expert_count for layer in moe.layers)
        assert all(layer.top_k == top_k for layer in moe.layers)

        input_ids, attention_mask = loaded.tokenizer.batch_encode(("Once upon a time,",))
        batch = ModelBatch(
            sample_ids=("public-smoke-1",),
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        loss = adapter.forward_loss(loaded, batch)
        generated = adapter.generate(
            loaded,
            batch,
            DecodeConfig(max_new_tokens=2, do_sample=False),
        )
        assert loss.sample_ids == batch.sample_ids
        assert loss.mean_nll > 0.0
        assert generated.sample_ids == batch.sample_ids
        assert len(generated.token_ids[0]) == 2

        sink = InMemoryRoutingSink(
            "full_trace",
            expert_counts={layer.layer_id: layer.expert_count for layer in moe.layers},
        )
        handle = adapter.attach_routing_hooks(loaded, sink, config)
        sink.start_batch(BatchMeta(sample_ids=batch.sample_ids, batch_id="public-smoke"))
        adapter.forward_loss(loaded, batch)
        sink.end_batch()
        handle.remove()
        handle.remove()

        # A leaked hook would raise because no sink batch is active.
        adapter.forward_loss(loaded, batch)
        artifact = sink.close()
        assert artifact.batch_count == 1
        assert len(artifact.aggregates) == layer_count
        assert artifact.observed_event_count > 0
        assert artifact.recorded_event_count == artifact.observed_event_count
        assert all(
            aggregate.assignment_count == aggregate.token_count * top_k
            for aggregate in artifact.aggregates.values()
        )
        assert all(event.router_logits is not None for event in artifact.raw_traces)
    finally:
        runtime.cleanup()
