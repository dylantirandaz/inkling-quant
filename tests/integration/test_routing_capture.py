"""Local pipeline routing-capture integration tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
import torch

from inkling_quant_lab.config import (
    BenchmarkConfig,
    ExperimentConfig,
    ModelConfig,
    ReportingConfig,
    RoutingConfig,
)
from inkling_quant_lab.exceptions import RoutingInstrumentationError
from inkling_quant_lab.models.base import ModelBatch, MoEDescriptor
from inkling_quant_lab.models.fixtures import FixedTokenizer
from inkling_quant_lab.models.local import LocalFixtureAdapter
from inkling_quant_lab.pipeline.routing_capture import (
    RoutingUnsupportedError,
    capture_routing_dataset,
    expert_usage_by_module,
)
from inkling_quant_lab.routing import RoutingMode
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.integration


def _config(
    model_id: str,
    *,
    mode: RoutingMode,
    capture_router_logits: bool = False,
    sampled_token_positions: tuple[int, ...] = (),
) -> ExperimentConfig:
    return ExperimentConfig(
        name="routing-capture-test",
        model=ModelConfig(model_id=model_id, checkpoint_format="fixture"),
        routing=RoutingConfig(
            mode=mode,
            capture_router_logits=capture_router_logits,
            sampled_token_positions=sampled_token_positions,
        ),
        benchmark=BenchmarkConfig(enabled=False),
        reporting=ReportingConfig(markdown=False, html=False, plots=False),
    )


def _loaded_batch(
    config: ExperimentConfig,
) -> tuple[LocalFixtureAdapter, Any, ModelBatch]:
    adapter = LocalFixtureAdapter()
    loaded = adapter.load(config, TorchEagerCPURuntime())
    tokenizer = cast(FixedTokenizer, loaded.tokenizer)
    input_ids, attention_mask = tokenizer.batch_encode(("alpha route", "beta expert"))
    batch = ModelBatch(
        sample_ids=("route-alpha", "route-beta"),
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
    )
    return adapter, loaded, batch


def test_pipeline_capture_honors_mode_alignment_override_and_preserves_outputs() -> None:
    config = _config(
        "local://fixtures/tiny-moe",
        mode="aggregate",
        capture_router_logits=True,
    )
    adapter, loaded, batch = _loaded_batch(config)
    module = cast(Any, loaded.model)
    with torch.inference_mode():
        logits_before = module(batch.input_ids).detach().clone()

    aggregate = capture_routing_dataset(adapter, loaded, (batch,), config)

    assert aggregate.mode == "aggregate"
    assert aggregate.raw_traces == ()
    expected_event_count = int(batch.input_ids.numel()) * 2
    assert aggregate.observed_event_count == expected_event_count
    assert config.routing.mode == "aggregate"

    aligned = capture_routing_dataset(
        adapter,
        loaded,
        (batch,),
        config,
        require_trace_alignment=True,
    )

    assert aligned.mode == "full_trace"
    assert aligned.recorded_event_count == expected_event_count
    assert {event.sample_id for event in aligned.raw_traces} == {
        "route-alpha",
        "route-beta",
    }
    assert all(event.router_logits is not None for event in aligned.raw_traces)
    assert all(event.router_probabilities is not None for event in aligned.raw_traces)
    with torch.inference_mode():
        logits_after = module(batch.input_ids).detach()
    assert torch.equal(logits_after, logits_before)

    descriptor = adapter.discover_moe(loaded)
    assert descriptor is not None
    usage = expert_usage_by_module(aligned, descriptor)
    expected_names = {
        module_name for layer in descriptor.layers for module_name in layer.expert_module_names
    }
    assert set(usage) == expected_names
    assert sum(usage.values()) == sum(
        layer.assignment_count for layer in aligned.aggregates.values()
    )


def test_pipeline_capture_removes_hooks_after_sample_identity_failure() -> None:
    config = _config("local://fixtures/tiny-moe", mode="full_trace")
    adapter, loaded, batch = _loaded_batch(config)

    with pytest.raises(RoutingInstrumentationError, match="repeats stable sample IDs"):
        capture_routing_dataset(adapter, loaded, (batch, batch), config)

    # A leaked hook would try to write into the failed capture's sink here.
    module = cast(Any, loaded.model)
    with torch.inference_mode():
        logits = module(batch.input_ids)
    assert logits.shape[:2] == batch.input_ids.shape


def test_pipeline_capture_rejects_empty_dataset_instead_of_empty_success() -> None:
    config = _config("local://fixtures/tiny-moe", mode="aggregate")
    adapter, loaded, _batch = _loaded_batch(config)

    with pytest.raises(RoutingInstrumentationError, match="produced no batches"):
        capture_routing_dataset(adapter, loaded, (), config)


def test_dense_capture_is_typed_unsupported_not_empty_success() -> None:
    config = _config("local://fixtures/tiny-dense", mode="aggregate")
    adapter, loaded, batch = _loaded_batch(config)

    with pytest.raises(RoutingUnsupportedError) as captured:
        capture_routing_dataset(adapter, loaded, (batch,), config)

    assert captured.value.code == "ROUTING_UNSUPPORTED"
    assert captured.value.details == {
        "model_id": "local://fixtures/tiny-dense",
        "mode": "aggregate",
    }


def test_disabled_capture_is_explicitly_off_even_for_dense_model() -> None:
    config = _config("local://fixtures/tiny-dense", mode="off")
    adapter, loaded, batch = _loaded_batch(config)

    artifact = capture_routing_dataset(adapter, loaded, (batch,), config)

    assert artifact.mode == "off"
    assert artifact.observed_event_count == 0
    assert artifact.aggregates == {}


def test_expert_usage_rejects_descriptor_and_aggregate_mismatches() -> None:
    config = _config("local://fixtures/tiny-moe", mode="aggregate")
    adapter, loaded, batch = _loaded_batch(config)
    artifact = capture_routing_dataset(adapter, loaded, (batch,), config)
    descriptor = adapter.discover_moe(loaded)
    assert descriptor is not None
    first, second = descriptor.layers

    duplicate_layers = replace(descriptor, layers=(first, first))
    with pytest.raises(RoutingInstrumentationError, match="duplicate layer IDs"):
        expert_usage_by_module(artifact, duplicate_layers)

    replacement_layer = replace(first, layer_id="replacement")
    mismatched_layers = replace(descriptor, layers=(replacement_layer, second))
    with pytest.raises(RoutingInstrumentationError, match=r"missing layers.*unknown layers"):
        expert_usage_by_module(artifact, mismatched_layers)

    unnamed_expert = replace(first, expert_module_names=first.expert_module_names[:-1])
    with pytest.raises(RoutingInstrumentationError, match="does not name every expert"):
        expert_usage_by_module(
            artifact,
            replace(descriptor, layers=(unnamed_expert, second)),
        )

    wrong_expert_count = replace(
        first,
        expert_count=first.expert_count - 1,
        expert_module_names=first.expert_module_names[:-1],
    )
    with pytest.raises(RoutingInstrumentationError, match="expert count does not match"):
        expert_usage_by_module(
            artifact,
            replace(descriptor, layers=(wrong_expert_count, second)),
        )

    duplicate_module = replace(
        second,
        expert_module_names=(first.expert_module_names[0], *second.expert_module_names[1:]),
    )
    descriptor_with_duplicate_module = MoEDescriptor(
        layers=(first, duplicate_module),
        supports_router_logits=descriptor.supports_router_logits,
        supports_token_level_routes=descriptor.supports_token_level_routes,
    )
    with pytest.raises(RoutingInstrumentationError, match="duplicate expert module name"):
        expert_usage_by_module(artifact, descriptor_with_duplicate_module)
