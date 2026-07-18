from __future__ import annotations

import pytest
import torch

from inkling_quant_lab.exceptions import CapabilityError, ModelLoadError
from inkling_quant_lab.models.base import ModelBatch
from inkling_quant_lab.models.fixtures import FixedTokenizer
from inkling_quant_lab.models.hf_causal_lm import (
    HFCausalLMAdapter,
    HFLinearMixtralCausalLMAdapter,
    _validate_checkpoint_loading_info,
    create_adapter,
    create_linear_mixtral_adapter,
)
from inkling_quant_lab.models.local import LocalFixtureAdapter
from inkling_quant_lab.pipeline.operations import Components, probe_capabilities
from inkling_quant_lab.quantization.reference import NoopQuantizer
from inkling_quant_lab.runtimes.fake import FakeRuntime
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.unit


def test_linear_mixtral_loading_is_an_explicit_adapter_choice() -> None:
    native = create_adapter()
    linear = create_linear_mixtral_adapter()

    assert type(native) is HFCausalLMAdapter
    assert native.name == "hf_causal_lm"
    assert native.mixtral_checkpoint_layout == "transformers_native_fused"
    assert type(linear) is HFLinearMixtralCausalLMAdapter
    assert linear.name == "hf_causal_lm_linear_mixtral"
    assert linear.mixtral_checkpoint_layout == "defuser_linear"


def test_hf_loading_info_must_prove_exact_checkpoint_realization() -> None:
    _validate_checkpoint_loading_info(
        {
            "missing_keys": set(),
            "unexpected_keys": set(),
            "mismatched_keys": set(),
            "error_msgs": [],
        }
    )

    with pytest.raises(ModelLoadError, match="missing_keys=1, unexpected_keys=1"):
        _validate_checkpoint_loading_info(
            {
                "missing_keys": {"model.layers.0.mlp.experts.down_proj"},
                "unexpected_keys": {"model.layers.0.mlp.experts.0.down_proj.weight"},
                "mismatched_keys": set(),
                "error_msgs": [],
            }
        )


def test_fixed_tokenizer_is_deterministic():
    tokenizer = FixedTokenizer()
    assert tokenizer.encode("Alpha beta") == tokenizer.encode("alpha beta")
    assert tokenizer.decode(tokenizer.encode("alpha beta")) == "alpha beta"


def test_tiny_dense_adapter_contract(config_factory):
    config = config_factory(model_id="local://fixtures/tiny-dense", routing_mode="off")
    adapter = LocalFixtureAdapter()
    loaded = adapter.load(config, TorchEagerCPURuntime())
    tokenizer = loaded.tokenizer
    input_ids, attention_mask = tokenizer.batch_encode(("alpha beta gamma",))
    batch = ModelBatch(sample_ids=("sample-1",), input_ids=input_ids, attention_mask=attention_mask)

    first = adapter.generate(loaded, batch, config.evaluation.suites[0].decode)
    second = adapter.generate(loaded, batch, config.evaluation.suites[0].decode)
    loss = adapter.forward_loss(loaded, batch)

    assert first == second
    assert loss.sample_ids == ("sample-1",)
    assert loss.mean_nll > 0
    assert adapter.enumerate_modules(loaded)
    assert adapter.discover_moe(loaded) is None
    assert loaded.descriptor.capabilities.is_moe is False


def test_tiny_moe_discovery_and_token_preferences(config_factory):
    config = config_factory()
    adapter = LocalFixtureAdapter()
    loaded = adapter.load(config, TorchEagerCPURuntime())
    descriptor = adapter.discover_moe(loaded)
    tokenizer = loaded.tokenizer
    ids, _ = tokenizer.batch_encode(("alpha red one cat", "delta yellow four fish"))
    with torch.inference_mode():
        loaded.model(ids)
    routes = [layer.routing_snapshot()["selected_expert_ids"] for layer in loaded.model.moe_layers]

    assert descriptor is not None
    assert len(descriptor.layers) == 2
    assert all(layer.expert_count == 4 and layer.top_k == 2 for layer in descriptor.layers)
    assert all(route.shape[-1] == 2 for route in routes)
    assert any(torch.unique(route).numel() > 1 for route in routes)


def test_multimodal_stub_accepts_tensor(config_factory):
    config = config_factory(model_id="local://fixtures/tiny-multimodal", routing_mode="off")
    adapter = LocalFixtureAdapter()
    loaded = adapter.load(config, TorchEagerCPURuntime())
    tokenizer = loaded.tokenizer
    ids, mask = tokenizer.batch_encode(("image answer",))
    result = adapter.forward_loss(
        loaded,
        ModelBatch(
            sample_ids=("mm-1",),
            input_ids=ids,
            attention_mask=mask,
            multimodal_inputs=torch.tensor(((1.0, 0.0, 0.5, -0.5),)),
        ),
    )
    assert result.mean_nll > 0
    assert loaded.descriptor.capabilities.supports_images


def test_capability_probe_rejects_runtime_dtype_and_sharding_before_load(config_factory) -> None:
    config = config_factory(model_id="local://fixtures/tiny-dense", routing_mode="off")
    unsupported_dtype = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"dtype": "bfloat16"})}
    )
    fake_components = Components(
        adapter=LocalFixtureAdapter(),
        runtime=FakeRuntime(),
        quantizer=NoopQuantizer(),
    )

    with pytest.raises(CapabilityError, match="does not support dtype bfloat16"):
        probe_capabilities(unsupported_dtype, fake_components)

    unsupported_sharding = config.model_copy(
        update={"runtime": config.runtime.model_copy(update={"sharding": {"layers": 2}})}
    )
    cpu_components = Components(
        adapter=LocalFixtureAdapter(),
        runtime=TorchEagerCPURuntime(),
        quantizer=NoopQuantizer(),
    )

    with pytest.raises(CapabilityError, match="does not support sharding"):
        probe_capabilities(unsupported_sharding, cpu_components)


def test_hf_adapter_requires_immutable_revision_before_hub_access(config_factory) -> None:
    config = config_factory(model_id="organization/model", routing_mode="off")
    model = config.model.model_copy(
        update={
            "adapter": "hf_causal_lm",
            "revision": "main",
            "checkpoint_format": "safetensors",
            "local_files_only": False,
        }
    )
    external = config.model_copy(update={"model": model})

    with pytest.raises(CapabilityError, match="immutable 40-character commit revision"):
        HFCausalLMAdapter().capabilities(external)


def test_linear_mixtral_adapter_rejects_other_pinned_models_before_hub_access(
    config_factory,
) -> None:
    config = config_factory(model_id="organization/model", routing_mode="off")
    model = config.model.model_copy(
        update={
            "adapter": "hf_causal_lm_linear_mixtral",
            "revision": "0" * 40,
            "checkpoint_format": "safetensors",
        }
    )
    external = config.model_copy(update={"model": model})

    with pytest.raises(CapabilityError, match="exact pinned Stories15M revision"):
        HFLinearMixtralCausalLMAdapter().capabilities(external)


def test_hf_adapter_rejects_remote_code_even_when_process_authorized(config_factory) -> None:
    config = config_factory(model_id="organization/model", routing_mode="off")
    model = config.model.model_copy(
        update={
            "adapter": "hf_causal_lm",
            "revision": "0" * 40,
            "checkpoint_format": "safetensors",
            "trust_remote_code": True,
        }
    )
    security = config.security.model_copy(update={"allow_remote_code": True})
    external = config.model_copy(update={"model": model, "security": security})

    with pytest.raises(CapabilityError, match="does not execute remote model code"):
        HFCausalLMAdapter().capabilities(external)


def test_hf_adapter_fails_closed_for_unvalidated_accelerator_placement(config_factory) -> None:
    config = config_factory(model_id="organization/model", routing_mode="off")
    model = config.model.model_copy(
        update={
            "adapter": "hf_causal_lm",
            "revision": "0" * 40,
            "checkpoint_format": "safetensors",
        }
    )
    runtime = config.runtime.model_copy(update={"backend": "torch_eager_mps", "device": "mps"})
    external = config.model_copy(update={"model": model, "runtime": runtime})

    with pytest.raises(CapabilityError, match="unsharded single-CPU eager placement"):
        HFCausalLMAdapter().capabilities(external)
