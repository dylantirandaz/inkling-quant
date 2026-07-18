"""Adapter for deterministic local dense, MoE, and multimodal fixtures."""

from __future__ import annotations

import re
import time
from typing import Any, cast

import torch
from torch import Tensor, nn

from inkling_quant_lab.config import DecodeConfig, ExperimentConfig
from inkling_quant_lab.exceptions import CapabilityError, ModelLoadError
from inkling_quant_lab.models.base import (
    HookHandle,
    LoadedModel,
    LossOutput,
    ModelBatch,
    ModelCapabilities,
    ModelDescriptor,
    ModelOutput,
    ModuleInfo,
    MoEDescriptor,
    MoELayerDescriptor,
    RoutingSinkLike,
    RuntimeForModel,
)
from inkling_quant_lab.models.fixtures import (
    FixedTokenizer,
    TinyDenseCausalLM,
    TinyMoECausalLM,
    TinyMoELayer,
    TinyMultimodalCausalLM,
    causal_loss_per_sample,
)
from inkling_quant_lab.models.state import model_state_sha256

_EXPERT_NAME = re.compile(r"^moe_layers\.(?P<layer>\d+)\.experts\.(?P<expert>\d+)(?:\.|$)")


def _model_kind(model_id: str) -> str:
    aliases = {
        "local://fixtures/tiny-dense": "dense",
        "local://fixtures/tiny-moe": "moe",
        "local://fixtures/tiny-multimodal": "multimodal",
    }
    try:
        return aliases[model_id]
    except KeyError as error:
        raise CapabilityError(
            f"The local fixture adapter does not recognize model_id '{model_id}'",
            component="local_fixture",
            remediation="Use one of: " + ", ".join(sorted(aliases)),
        ) from error


def _model_device(model: nn.Module) -> torch.device:
    """Return the device owning the first model tensor, defaulting to CPU."""

    value: Tensor | None = next(model.parameters(), None)
    if value is None:
        value = next(model.buffers(), None)
    return torch.device("cpu") if value is None else value.device


def _on_model_device(value: Tensor | None, model: nn.Module) -> Tensor | None:
    """Move an optional batch tensor to the model without mutating its source batch."""

    if value is None:
        return None
    return value.to(device=_model_device(model))


class LocalFixtureAdapter:
    """Architecture-normalized adapter for offline CPU fixtures."""

    name = "local_fixture"

    def capabilities(self, config: ExperimentConfig) -> ModelCapabilities:
        """Report capabilities from the explicit local fixture identity."""

        kind = _model_kind(config.model.model_id)
        return ModelCapabilities(
            supports_text=True,
            supports_images=kind == "multimodal",
            supports_audio=kind == "multimodal",
            is_moe=kind == "moe",
            supports_router_logits=kind == "moe",
            supports_token_level_routes=kind == "moe",
            supported_dtypes=("float32", "bfloat16"),
            supported_device_maps=("single",),
            max_context_length=64,
            requires_remote_code=False,
        )

    def load(self, config: ExperimentConfig, runtime: RuntimeForModel) -> LoadedModel:
        """Construct a new deterministic fixture without network or remote code."""

        capabilities = self.capabilities(config)
        if config.model.dtype not in capabilities.supported_dtypes:
            raise ModelLoadError(
                f"Local fixture does not support dtype {config.model.dtype}",
                component=self.name,
            )
        if config.model.trust_remote_code:
            raise ModelLoadError(
                "Local fixtures never require remote code; refusing unnecessary opt-in",
                component=self.name,
            )
        tokenizer = FixedTokenizer()
        kind = _model_kind(config.model.model_id)
        started = time.perf_counter()
        with runtime.execution_context():
            if kind == "dense":
                module: nn.Module = TinyDenseCausalLM(tokenizer.vocab_size)
            elif kind == "moe":
                module = TinyMoECausalLM(tokenizer.vocab_size)
            else:
                module = TinyMultimodalCausalLM()
            if config.model.dtype == "bfloat16":
                module = module.to(dtype=torch.bfloat16)
            place_module = getattr(runtime, "place_module", None)
            if callable(place_module):
                module = cast(nn.Module, place_module(module))
            module.eval()
        elapsed = time.perf_counter() - started
        architecture = str(getattr(module, "architecture_name", module.__class__.__name__))
        descriptor = ModelDescriptor(
            model_id=config.model.model_id,
            revision=config.model.revision or f"{architecture}:fixture-v1",
            resolved_class=f"{module.__class__.__module__}.{module.__class__.__qualname__}",
            architecture=architecture,
            checksum=model_state_sha256(module),
            capabilities=capabilities,
        )
        return LoadedModel(
            model=module,
            tokenizer=tokenizer,
            descriptor=descriptor,
            load_time_seconds=elapsed,
        )

    def enumerate_modules(self, model: LoadedModel) -> tuple[ModuleInfo, ...]:
        """Return stable leaf-module inventory with normalized MoE roles."""

        inventory: list[ModuleInfo] = []
        root = cast(nn.Module, model.model)
        for name, module in root.named_modules():
            if not name:
                continue
            direct_parameters = tuple(module.parameters(recurse=False))
            direct_buffers = tuple(module.buffers(recurse=False))
            if not direct_parameters and not direct_buffers:
                continue
            parameter_count = sum(parameter.numel() for parameter in direct_parameters)
            size_bytes = sum(
                value.numel() * value.element_size()
                for value in (*direct_parameters, *direct_buffers)
            )
            expert_match = _EXPERT_NAME.match(name)
            is_embedding = isinstance(module, nn.Embedding)
            supported = (
                ("float32", "float16", "bfloat16")
                if is_embedding or isinstance(module, nn.LayerNorm)
                else ("float32", "float16", "bfloat16", "int8", "int4")
            )
            inventory.append(
                ModuleInfo(
                    name=name,
                    class_name=f"{module.__class__.__module__}.{module.__class__.__qualname__}",
                    parameter_count=parameter_count,
                    size_bytes=size_bytes,
                    is_router=name.endswith(".router"),
                    is_output_head=name == "lm_head",
                    is_embedding=is_embedding,
                    is_multimodal_projector=name == "multimodal_projector",
                    is_expert=expert_match is not None,
                    layer_id=(f"moe_{expert_match.group('layer')}" if expert_match else None),
                    expert_id=(int(expert_match.group("expert")) if expert_match else None),
                    supported_precisions=cast(Any, supported),
                )
            )
        return tuple(sorted(inventory, key=lambda item: item.name))

    def discover_moe(self, model: LoadedModel) -> MoEDescriptor | None:
        """Use the fixture-specific layer type instead of a universal router name."""

        root = cast(nn.Module, model.model)
        discovered: list[MoELayerDescriptor] = []
        for name, module in root.named_modules():
            if not isinstance(module, TinyMoELayer):
                continue
            discovered.append(
                MoELayerDescriptor(
                    layer_id=module.layer_id,
                    module_name=name,
                    router_module_name=f"{name}.router",
                    expert_module_names=tuple(
                        f"{name}.experts.{index}" for index in range(len(module.experts))
                    ),
                    expert_count=len(module.experts),
                    top_k=module.top_k,
                )
            )
        if not discovered:
            return None
        return MoEDescriptor(
            layers=tuple(discovered),
            supports_router_logits=True,
            supports_token_level_routes=True,
        )

    def attach_routing_hooks(
        self, model: LoadedModel, sink: RoutingSinkLike, config: ExperimentConfig
    ) -> HookHandle:
        """Attach adapter-owned hooks to discovered routed layer types."""

        if self.discover_moe(model) is None:
            raise CapabilityError(
                "Routing analysis is unsupported for this dense model",
                component=self.name,
            )
        from inkling_quant_lab.routing.hooks import attach_routing_hooks

        return attach_routing_hooks(
            cast(nn.Module, model.model),
            sink,
            capture_router_logits=config.routing.capture_router_logits,
        )

    def generate(self, model: LoadedModel, batch: ModelBatch, config: DecodeConfig) -> ModelOutput:
        """Run deterministic greedy generation and return only generated suffixes."""

        self._reject_unsupported_multimodal(model, batch)
        module = cast(Any, model.model)
        input_ids = cast(Tensor, _on_model_device(cast(Tensor, batch.input_ids), module))
        with torch.inference_mode():
            generated = module.generate(input_ids, max_new_tokens=config.max_new_tokens)
        suffix = generated[:, input_ids.shape[1] :].detach().cpu().tolist()
        token_ids = tuple(tuple(int(token) for token in row) for row in suffix)
        tokenizer = cast(FixedTokenizer, model.tokenizer)
        texts = tuple(tokenizer.decode(row) for row in token_ids)
        return ModelOutput(sample_ids=batch.sample_ids, token_ids=token_ids, texts=texts)

    def forward_loss(self, model: LoadedModel, batch: ModelBatch) -> LossOutput:
        """Compute token-level causal loss on a deterministic batch."""

        self._reject_unsupported_multimodal(model, batch)
        module = cast(Any, model.model)
        input_ids = cast(Tensor, _on_model_device(cast(Tensor, batch.input_ids), module))
        attention_mask = _on_model_device(cast(Tensor | None, batch.attention_mask), module)
        with torch.inference_mode():
            if isinstance(module, TinyMultimodalCausalLM):
                logits = module(
                    input_ids,
                    multimodal_inputs=_on_model_device(
                        cast(Tensor | None, batch.multimodal_inputs), module
                    ),
                )
            else:
                logits = module(input_ids)
        losses, counts = causal_loss_per_sample(logits, input_ids, attention_mask)
        return LossOutput(
            sample_ids=batch.sample_ids,
            negative_log_likelihoods=losses,
            token_counts=counts,
        )

    def _reject_unsupported_multimodal(self, model: LoadedModel, batch: ModelBatch) -> None:
        if (
            batch.multimodal_inputs is not None
            and not model.descriptor.capabilities.supports_images
        ):
            raise CapabilityError(
                f"Model {model.descriptor.model_id} does not support multimodal inputs",
                component=self.name,
            )


def create_adapter() -> LocalFixtureAdapter:
    """Registry factory for deterministic local fixtures."""

    return LocalFixtureAdapter()
