"""Registered MLX-LM adapter for the exact pinned Stories15M Mixtral model."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from inkling_quant_lab.config import DecodeConfig, ExperimentConfig, Precision
from inkling_quant_lab.exceptions import (
    CapabilityError,
    ModelLoadError,
    RoutingInstrumentationError,
)
from inkling_quant_lab.mlx_contract import (
    MLX_MODEL_CLASS,
    MODEL_ARCHITECTURE,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_WEIGHT_SHA256,
    audit_converted_bundle,
    audit_source_snapshot,
    expected_float32_only_names,
    expected_quantized_leaf_names,
    mlx_environment_status,
    runtime_quantization_proof,
    validate_model_config_security,
)
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
from inkling_quant_lab.routing.hooks import events_from_routing_snapshot
from inkling_quant_lab.routing.traces import BatchMeta

_MAX_CONTEXT = 256


@dataclass(slots=True)
class MLXModelHandle:
    """MLX module plus the audited local source and resolved model config."""

    module: Any
    config: dict[str, Any]
    source_snapshot: Path
    source_audit: dict[str, Any]
    seed: int
    bits: int | None = None


class MLXTokenizerAdapter:
    """Expose the tokenizer surface used by the architecture-neutral evaluators."""

    def __init__(self, tokenizer: Any) -> None:
        self.raw = tokenizer

    def encode(self, text: str, *, special_tokens: bool = True) -> list[int]:
        """Encode one prompt through the audited snapshot tokenizer."""

        return [int(token) for token in self.raw.encode(text, add_special_tokens=special_tokens)]

    def decode(self, token_ids: tuple[int, ...] | list[int]) -> str:
        """Decode ephemeral output without changing persistence behavior."""

        return str(self.raw.decode(list(token_ids), skip_special_tokens=True))

    def batch_encode(
        self, texts: tuple[str, ...]
    ) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
        """Return padded immutable integer rows without constructing a torch tensor."""

        if not texts:
            raise ValueError("texts must not be empty")
        rows = tuple(tuple(self.encode(text)) for text in texts)
        if any(not row for row in rows):
            raise ValueError("tokenized text must contain at least one token")
        if any(len(row) > _MAX_CONTEXT for row in rows):
            raise ValueError(
                "MLX input exceeds the exact 256-token contract; explicit dataset-side "
                "selection/truncation is required"
            )
        width = max(len(row) for row in rows)
        eos_ids = tuple(int(value) for value in getattr(self.raw, "eos_token_ids", ()))
        pad = eos_ids[0] if eos_ids else 0
        padded = tuple(row + (pad,) * (width - len(row)) for row in rows)
        masks = tuple((1,) * len(row) + (0,) * (width - len(row)) for row in rows)
        return padded, masks


def _snapshot_path(config: ExperimentConfig) -> Path:
    value = config.model.local_snapshot_path
    if value is None:  # guarded by ExperimentConfig, retained for direct adapter calls
        raise CapabilityError(
            "mlx_lm_mixtral requires model.local_snapshot_path",
            component="mlx_lm_mixtral",
        )
    return Path(value).expanduser().resolve(strict=True)


def _require_environment() -> dict[str, str]:
    status = mlx_environment_status()
    if not status.available:
        raise CapabilityError(
            "The exact MLX environment is unavailable: " + "; ".join(status.reasons),
            component="mlx_lm_mixtral",
            remediation="Install the pinned matrix with `uv sync --extra mlx` on Apple Silicon.",
        )
    return status.versions


def _load_raw_mlx_model(
    path: Path,
    *,
    seed: int,
    cast_float32: bool,
) -> tuple[Any, MLXTokenizerAdapter, dict[str, Any], float]:
    """Load only after the caller has completed the appropriate static audit."""

    _require_environment()
    try:
        import mlx.core as mx
        from mlx_lm.utils import load

        mx.random.seed(seed)
        started = time.perf_counter()
        module, tokenizer, raw_config = cast(
            tuple[Any, Any, Any],
            load(
                str(path),
                tokenizer_config={"trust_remote_code": False},
                return_config=True,
                lazy=False,
            ),
        )
        if cast_float32:
            module.set_dtype(mx.float32)
            mx.eval(module.parameters())
        mx.synchronize()
        elapsed = time.perf_counter() - started
    except CapabilityError:
        raise
    except Exception as error:
        raise ModelLoadError(
            f"MLX-LM failed to load the audited local Mixtral bundle: {error}",
            component="mlx_lm_mixtral",
        ) from error
    if elapsed <= 0.0 or not math.isfinite(elapsed):
        raise ModelLoadError("MLX model load duration is invalid", component="mlx_lm_mixtral")
    resolved = f"{type(module).__module__}.{type(module).__qualname__}"
    if resolved != MLX_MODEL_CLASS:
        raise ModelLoadError(
            f"MLX-LM resolved {resolved!r}, expected {MLX_MODEL_CLASS!r}",
            component="mlx_lm_mixtral",
        )
    if not isinstance(raw_config, dict):
        raise ModelLoadError("MLX-LM returned no model configuration", component="mlx_lm_mixtral")
    config = {str(key): value for key, value in raw_config.items()}
    validate_model_config_security(config, source="loaded MLX model")
    return module, MLXTokenizerAdapter(tokenizer), config, elapsed


def load_exact_source(
    snapshot: Path, *, seed: int
) -> tuple[MLXModelHandle, MLXTokenizerAdapter, float]:
    """Audit and materialize the exact source as an all-float32 MLX control."""

    audit = audit_source_snapshot(snapshot)
    module, tokenizer, config, elapsed = _load_raw_mlx_model(
        snapshot,
        seed=seed,
        cast_float32=True,
    )
    handle = MLXModelHandle(
        module=module,
        config=config,
        source_snapshot=snapshot,
        source_audit=audit,
        seed=seed,
    )
    _validated_inventory(handle)
    return handle, tokenizer, elapsed


def load_exact_converted_bundle(
    path: Path, *, bits: int, seed: int, source_snapshot: Path
) -> tuple[MLXModelHandle, MLXTokenizerAdapter, float, dict[str, Any]]:
    """Audit, reload, and inspect a deterministic q4/q8 bundle."""

    audit = audit_converted_bundle(path, bits=bits)
    module, tokenizer, config, elapsed = _load_raw_mlx_model(
        path,
        seed=seed,
        cast_float32=False,
    )
    proof = runtime_quantization_proof(module, bits=bits)
    return (
        MLXModelHandle(
            module=module,
            config=config,
            source_snapshot=source_snapshot,
            source_audit=audit,
            seed=seed,
            bits=bits,
        ),
        tokenizer,
        elapsed,
        proof,
    )


def _weight_facts(module: Any) -> tuple[int, int]:
    values = [module.weight]
    bias = getattr(module, "bias", None)
    if bias is not None:
        values.append(bias)
    return (
        sum(math.prod(int(value) for value in tensor.shape) for tensor in values),
        sum(int(tensor.nbytes) for tensor in values),
    )


def _layer_id(name: str) -> str | None:
    parts = name.split(".")
    if len(parts) >= 3 and parts[0:2] == ["model", "layers"] and parts[2].isdigit():
        return f"model.layers.{parts[2]}.block_sparse_moe"
    return None


def _validated_inventory(handle: MLXModelHandle) -> tuple[ModuleInfo, ...]:
    named = {str(name): module for name, module in handle.module.named_modules()}
    expected_quantized = expected_quantized_leaf_names()
    expected_float = expected_float32_only_names()
    expected = set(expected_quantized | expected_float)
    observed = {name for name, module in named.items() if name and hasattr(module, "weight")}
    if observed != expected:
        raise CapabilityError(
            "MLX Stories15M parameter-leaf inventory mismatch; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}",
            component="mlx_lm_mixtral",
        )
    inventory: list[ModuleInfo] = []
    for name in sorted(observed):
        module = named[name]
        parameter_count, size_bytes = _weight_facts(module)
        class_name = f"{type(module).__module__}.{type(module).__qualname__}"
        eligible = name in expected_quantized
        if eligible and not hasattr(module, "to_quantized"):
            raise CapabilityError(
                f"expected MLX quantizable leaf has no to_quantized method: {name}",
                component="mlx_lm_mixtral",
            )
        inventory.append(
            ModuleInfo(
                name=name,
                class_name=class_name,
                parameter_count=parameter_count,
                size_bytes=size_bytes,
                is_router=name.endswith(".block_sparse_moe.gate"),
                is_output_head=name == "lm_head",
                is_embedding=name == "model.embed_tokens",
                is_expert=".switch_mlp." in name,
                layer_id=_layer_id(name),
                supported_precisions=(
                    cast(tuple[Precision, ...], ("float32", "int8", "int4"))
                    if eligible
                    else ("float32",)
                ),
            )
        )
    return tuple(inventory)


def _rows(batch: ModelBatch) -> tuple[tuple[int, ...], ...]:
    value = batch.input_ids
    if not isinstance(value, (tuple, list)):
        raise ValueError("MLX adapter input_ids must be immutable integer rows")
    rows: list[tuple[int, ...]] = []
    for row in value:
        if not isinstance(row, (tuple, list)) or not row:
            raise ValueError("MLX adapter input rows must be non-empty")
        normalized = tuple(int(token) for token in row)
        if any(token < 0 or isinstance(token, bool) for token in row):
            raise ValueError("MLX adapter token IDs must be non-negative integers")
        rows.append(normalized)
    if len(rows) != len(batch.sample_ids):
        raise ValueError("MLX token rows do not match stable sample IDs")
    masks = batch.attention_mask
    if masks is not None:
        if not isinstance(masks, (tuple, list)) or len(masks) != len(rows):
            raise ValueError("MLX attention masks do not match token rows")
        for row, mask in zip(rows, masks, strict=True):
            if tuple(mask) != (1,) * len(row):
                raise ValueError(
                    "the exact MLX adapter accepts unpadded single-sample batches only"
                )
    return tuple(rows)


class _MLXHookHandle:
    def __init__(self, replacements: tuple[tuple[Any, Any], ...]) -> None:
        self._replacements = list(replacements)
        self._removed = False

    def remove(self) -> None:
        """Restore every exact gate once, idempotently."""

        if self._removed:
            return
        for block, gate in self._replacements:
            block.gate = gate
        self._replacements.clear()
        self._removed = True


class MLXMixtralAdapter:
    """Exact single-device MLX-LM Mixtral model adapter."""

    name = "mlx_lm_mixtral"

    def _validate_config(self, config: ExperimentConfig) -> Path:
        if (
            config.model.model_id != MODEL_ID
            or config.model.revision != MODEL_REVISION
            or config.model.adapter != self.name
            or config.runtime.backend != "mlx_metal"
            or config.runtime.device != "mps"
            or config.runtime.device_map != "single"
            or config.runtime.sharding is not None
            or config.model.dtype != "float32"
            or config.runtime.dtype != "float32"
            or not config.model.local_files_only
            or config.model.trust_remote_code
            or config.security.allow_remote_code
            or config.model.checkpoint_format != "safetensors"
        ):
            raise CapabilityError(
                "mlx_lm_mixtral supports only the exact offline float32 Stories15M/Metal "
                "configuration",
                component=self.name,
            )
        path = _snapshot_path(config)
        audit_source_snapshot(path)
        _require_environment()
        return path

    def capabilities(self, config: ExperimentConfig) -> ModelCapabilities:
        """Audit source/dependencies without loading model weights."""

        self._validate_config(config)
        return ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_audio=False,
            is_moe=True,
            supports_router_logits=True,
            supports_token_level_routes=True,
            supported_dtypes=("float32",),
            supported_device_maps=("single",),
            max_context_length=_MAX_CONTEXT,
            requires_remote_code=False,
        )

    def load(self, config: ExperimentConfig, runtime: RuntimeForModel) -> LoadedModel:
        """Load a fresh all-float32 control after source and runtime capability probes."""

        del runtime
        snapshot = self._validate_config(config)
        handle, tokenizer, elapsed = load_exact_source(snapshot, seed=config.seed)
        return LoadedModel(
            model=handle,
            tokenizer=tokenizer,
            descriptor=ModelDescriptor(
                model_id=MODEL_ID,
                revision=MODEL_REVISION,
                resolved_class=MLX_MODEL_CLASS,
                architecture=MODEL_ARCHITECTURE,
                checksum=SOURCE_WEIGHT_SHA256,
                capabilities=self.capabilities(config),
            ),
            load_time_seconds=elapsed,
            load_time_kind="cold_model_load",
        )

    def enumerate_modules(self, model: LoadedModel) -> tuple[ModuleInfo, ...]:
        """Return all 63 exact parameter leaves and their real MLX classes."""

        handle = model.model
        if not isinstance(handle, MLXModelHandle):
            raise CapabilityError("loaded model is not owned by MLX", component=self.name)
        return _validated_inventory(handle)

    def discover_moe(self, model: LoadedModel) -> MoEDescriptor | None:
        """Validate the six built-in Mixtral blocks and expose top-2 routing."""

        handle = model.model
        if not isinstance(handle, MLXModelHandle):
            return None
        layers = tuple(handle.module.model.layers)
        if len(layers) != 6:
            raise CapabilityError("MLX Mixtral layer count is not six", component=self.name)
        descriptors: list[MoELayerDescriptor] = []
        for index, layer in enumerate(layers):
            block = layer.block_sparse_moe
            if int(block.num_experts) != 4 or int(block.num_experts_per_tok) != 2:
                raise CapabilityError(
                    f"MLX Mixtral routing dimensions changed at layer {index}",
                    component=self.name,
                )
            prefix = f"model.layers.{index}.block_sparse_moe"
            descriptors.append(
                MoELayerDescriptor(
                    layer_id=prefix,
                    module_name=prefix,
                    router_module_name=f"{prefix}.gate",
                    expert_module_names=tuple(f"{prefix}.experts.{expert}" for expert in range(4)),
                    expert_count=4,
                    top_k=2,
                )
            )
        return MoEDescriptor(
            layers=tuple(descriptors),
            supports_router_logits=True,
            supports_token_level_routes=True,
        )

    def attach_routing_hooks(
        self, model: LoadedModel, sink: RoutingSinkLike, config: ExperimentConfig
    ) -> HookHandle:
        """Wrap exact MLX gates and emit shared-schema routing events."""

        handle = model.model
        if not isinstance(handle, MLXModelHandle):
            raise CapabilityError("loaded model is not owned by MLX", component=self.name)
        try:
            import mlx.core as mx
            import mlx.nn as nn
        except ImportError as error:
            raise CapabilityError(
                "MLX routing requires the mlx optional extra", component=self.name
            ) from error

        replacements: list[tuple[Any, Any]] = []
        capture_logits = config.routing.capture_router_logits

        def wrapper_class(layer_id: str, original: Any) -> Any:
            class CaptureGate(nn.Module):  # type: ignore[misc, name-defined]
                def __init__(self) -> None:
                    super().__init__()
                    self.gate = original

                def __call__(self, inputs: Any) -> Any:
                    logits = self.gate(inputs)
                    active = getattr(sink, "active_batch", None)
                    if not isinstance(active, BatchMeta):
                        raise RoutingInstrumentationError(
                            "MLX routing gate fired without an active stable-ID batch",
                            component="mlx_lm_mixtral",
                        )
                    indices = mx.argpartition(-logits, kth=1, axis=-1)[..., :2]
                    selected = mx.softmax(
                        mx.take_along_axis(logits, indices, axis=-1),
                        axis=-1,
                        precise=True,
                    )
                    probabilities = mx.softmax(logits, axis=-1, precise=True)
                    mx.eval(indices, selected, probabilities, logits)
                    snapshot: dict[str, Any] = {
                        "layer_id": layer_id,
                        "selected_expert_ids": indices.tolist(),
                        "selected_weights": selected.tolist(),
                        "router_probabilities": probabilities.tolist(),
                    }
                    if capture_logits:
                        snapshot["router_logits"] = logits.tolist()
                    for event in events_from_routing_snapshot(
                        snapshot,
                        sample_ids=active.sample_ids,
                        capture_router_logits=capture_logits,
                        fallback_layer_id=layer_id,
                    ):
                        sink.record(event)
                    return logits

            return CaptureGate()

        try:
            for index, layer in enumerate(handle.module.model.layers):
                block = layer.block_sparse_moe
                original = block.gate
                replacements.append((block, original))
                block.gate = wrapper_class(f"model.layers.{index}.block_sparse_moe", original)
        except Exception as error:
            hook = _MLXHookHandle(tuple(replacements))
            hook.remove()
            if isinstance(error, RoutingInstrumentationError):
                raise
            raise RoutingInstrumentationError(
                f"failed to attach exact MLX routing gates: {error}",
                component=self.name,
            ) from error
        return _MLXHookHandle(tuple(replacements))

    def generate(self, model: LoadedModel, batch: ModelBatch, config: DecodeConfig) -> ModelOutput:
        """Run bounded greedy generation through MLX-LM's direct generator."""

        if config.do_sample:
            raise CapabilityError(
                "the exact registered MLX adapter currently supports greedy decoding only",
                component=self.name,
            )
        if batch.multimodal_inputs is not None:
            raise CapabilityError("the MLX adapter is text-only", component=self.name)
        rows = _rows(batch)
        if len(rows) != 1:
            raise CapabilityError(
                "the exact MLX generation contract accepts one sample per call",
                component=self.name,
            )
        if len(rows[0]) + config.max_new_tokens > _MAX_CONTEXT:
            raise CapabilityError(
                "prompt plus generated tokens exceeds the 256-token model context",
                component=self.name,
            )
        handle = model.model
        if not isinstance(handle, MLXModelHandle):
            raise CapabilityError("loaded model is not owned by MLX", component=self.name)
        import mlx.core as mx
        from mlx_lm.generate import generate_step

        generated: list[int] = []
        for token, _logprobs in generate_step(
            mx.array(rows[0]), handle.module, max_tokens=config.max_new_tokens
        ):
            normalized = int(token)
            generated.append(normalized)
            if normalized in tuple(int(value) for value in model.tokenizer.raw.eos_token_ids):
                break
        mx.synchronize()
        tokens = tuple(generated)
        return ModelOutput(
            sample_ids=batch.sample_ids,
            token_ids=(tokens,),
            texts=(model.tokenizer.decode(tokens),),
        )

    def forward_loss(self, model: LoadedModel, batch: ModelBatch) -> LossOutput:
        """Compute one sample's causal cross-entropy on MLX/Metal."""

        if batch.multimodal_inputs is not None:
            raise CapabilityError("the MLX adapter is text-only", component=self.name)
        rows = _rows(batch)
        if len(rows) != 1:
            raise CapabilityError(
                "the exact MLX loss contract accepts one unpadded sample per call",
                component=self.name,
            )
        if len(rows[0]) < 2:
            raise ValueError("causal loss requires at least two tokens")
        handle = model.model
        if not isinstance(handle, MLXModelHandle):
            raise CapabilityError("loaded model is not owned by MLX", component=self.name)
        import mlx.core as mx
        import mlx.nn as nn

        tokens = mx.array([rows[0]])
        logits = handle.module(tokens)
        losses = nn.losses.cross_entropy(logits[:, :-1, :], tokens[:, 1:], reduction="none")
        mx.eval(losses)
        mean_nll = float(mx.mean(losses).item())
        if not math.isfinite(mean_nll) or mean_nll < 0.0:
            raise ModelLoadError("MLX causal loss is non-finite", component=self.name)
        return LossOutput(
            sample_ids=batch.sample_ids,
            negative_log_likelihoods=(mean_nll,),
            token_counts=(len(rows[0]) - 1,),
        )


def create_adapter() -> MLXMixtralAdapter:
    """Lazy registry factory."""

    return MLXMixtralAdapter()


__all__ = [
    "MLXMixtralAdapter",
    "MLXModelHandle",
    "MLXTokenizerAdapter",
    "create_adapter",
    "load_exact_converted_bundle",
    "load_exact_source",
]
