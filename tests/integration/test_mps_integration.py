"""Opt-in evidence that the local MoE executes and routes on Apple MPS."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.models.base import ModelBatch
from inkling_quant_lab.models.local import LocalFixtureAdapter
from inkling_quant_lab.pipeline.routing_capture import capture_routing_dataset
from inkling_quant_lab.runtimes.torch_mps import TorchEagerMPSRuntime

pytestmark = [pytest.mark.integration, pytest.mark.gpu]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS is unavailable")
def test_tiny_moe_forward_and_routing_execute_on_mps(
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    raw = config_factory().canonical_dict()
    raw["runtime"] = {
        "backend": "torch_eager_mps",
        "device": "mps",
        "dtype": "float32",
        "device_map": "single",
        "sharding": None,
    }
    config = ExperimentConfig.model_validate(raw)
    adapter = LocalFixtureAdapter()
    runtime = TorchEagerMPSRuntime()
    assert runtime.probe(config.runtime).available

    loaded = adapter.load(config, runtime)
    try:
        assert next(loaded.model.parameters()).device.type == "mps"
        input_ids, attention_mask = loaded.tokenizer.batch_encode(("alpha route expert",))
        batch = ModelBatch(
            sample_ids=("mps-sample",),
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        loss = adapter.forward_loss(loaded, batch)
        runtime.synchronize()
        routing = capture_routing_dataset(adapter, loaded, (batch,), config)
        snapshot = runtime.memory_snapshot()

        assert loss.token_counts == (4,)
        assert routing.observed_event_count == 10
        assert tuple(routing.aggregates) == ("moe_0", "moe_1")
        assert snapshot.device_available
        assert snapshot.device_bytes is not None and snapshot.device_bytes > 0
    finally:
        runtime.cleanup()
