"""Safe Hugging Face causal-LM adapter with validated Mixtral introspection."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from inkling_quant_lab.config import DecodeConfig, ExperimentConfig, Precision
from inkling_quant_lab.exceptions import (
    CapabilityError,
    ModelLoadError,
    RoutingInstrumentationError,
)
from inkling_quant_lab.models.base import (
    EmptyExportModelShell,
    ExportedModelIdentity,
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
from inkling_quant_lab.models.fixtures import causal_loss_per_sample
from inkling_quant_lab.models.mixtral_compat import scoped_defuser_linear_mixtral
from inkling_quant_lab.models.state import model_state_sha256
from inkling_quant_lab.routing.hooks import events_from_routing_snapshot
from inkling_quant_lab.routing.traces import BatchMeta

_COMMIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SUPPORTED_MODEL_TYPE = "mixtral"
_SUPPORTED_MODEL_CLASS = "MixtralForCausalLM"
_SUPPORTED_RESOLVED_CLASS = "transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM"
_SUPPORTED_MOE_BLOCK_CLASSES = frozenset(
    {
        "transformers.models.mixtral.modeling_mixtral.MixtralSparseMoeBlock",
        "defuser.modeling.unfused_moe.mixtral.LinearMixtralSparseMoeBlock",
    }
)
_STORIES15M_MODEL_ID = "ggml-org/stories15M_MOE"
_STORIES15M_REVISION = "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
_SOURCE_METADATA_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class _OfflineSourceMetadataFiles:
    snapshot_path: Path
    file_sha256: tuple[tuple[str, str], ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class HFTokenizerAdapter:
    """Expose the small tokenizer surface used by the evaluator pipeline."""

    def __init__(self, tokenizer: Any) -> None:
        self.raw = tokenizer
        if getattr(tokenizer, "pad_token_id", None) is None:
            eos_token = getattr(tokenizer, "eos_token", None)
            if eos_token is None:
                raise ModelLoadError(
                    "The pinned Hugging Face tokenizer defines neither a pad nor EOS token",
                    component="hf_causal_lm",
                )
            tokenizer.pad_token = eos_token

    @property
    def vocab_size(self) -> int:
        """Return the declared tokenizer vocabulary size."""

        return int(self.raw.vocab_size)

    def encode(self, text: str, *, special_tokens: bool = True) -> list[int]:
        """Encode one string without importing tokenizer-specific types."""

        return [int(token) for token in self.raw.encode(text, add_special_tokens=special_tokens)]

    def decode(self, token_ids: list[int] | tuple[int, ...], *, skip_special: bool = True) -> str:
        """Decode token IDs with the same privacy behavior as the local tokenizer."""

        return str(
            self.raw.decode(
                list(token_ids),
                skip_special_tokens=skip_special,
                clean_up_tokenization_spaces=False,
            )
        )

    def batch_encode(self, texts: tuple[str, ...]) -> tuple[Tensor, Tensor]:
        """Tokenize and pad a non-empty batch into CPU tensors."""

        if not texts:
            raise ValueError("texts must not be empty")
        encoded = self.raw(
            list(texts),
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded.get("input_ids")
        attention_mask = encoded.get("attention_mask")
        if not isinstance(input_ids, Tensor) or not isinstance(attention_mask, Tensor):
            raise ModelLoadError(
                "The Hugging Face tokenizer did not return input_ids and attention_mask tensors",
                component="hf_causal_lm",
            )
        return input_ids.cpu(), attention_mask.cpu()


@dataclass(frozen=True, slots=True)
class _MixtralLayer:
    name: str
    module: nn.Module
    router: nn.Module
    expert_count: int
    top_k: int
    expert_names: tuple[str, ...]
    fused_experts: bool


class _MixtralStrategy:
    """Architecture-specific Mixtral discovery without universal layer-name guesses."""

    def layers(self, root: nn.Module) -> tuple[_MixtralLayer, ...]:
        discovered: list[_MixtralLayer] = []
        for name, module in root.named_modules():
            class_path = f"{module.__class__.__module__}.{module.__class__.__qualname__}"
            if class_path not in _SUPPORTED_MOE_BLOCK_CLASSES:
                continue
            router = getattr(module, "gate", None)
            experts = getattr(module, "experts", None)
            top_k = getattr(module, "top_k", None)
            expert_count = getattr(experts, "num_experts", None)
            if expert_count is None:
                expert_count = getattr(router, "num_experts", None)
            if expert_count is None:
                expert_count = getattr(module, "num_experts", None)
            if not isinstance(router, nn.Module) or not isinstance(experts, nn.Module):
                raise CapabilityError(
                    f"Mixtral layer {name} does not expose validated gate/expert modules",
                    component="hf_causal_lm",
                )
            if not isinstance(top_k, int) or top_k <= 0:
                raise CapabilityError(
                    f"Mixtral layer {name} does not expose a valid top-k value",
                    component="hf_causal_lm",
                )
            if not isinstance(expert_count, int) or expert_count <= 0 or top_k > expert_count:
                raise CapabilityError(
                    f"Mixtral layer {name} does not expose a valid expert count",
                    component="hf_causal_lm",
                )

            children = tuple(experts.named_children())
            indexed_children = tuple(
                (child_name, child)
                for child_name, child in children
                if child_name.isdigit() and isinstance(child, nn.Module)
            )
            fused = len(indexed_children) != expert_count
            if fused:
                expert_names = tuple(
                    f"{name}.experts.{expert_id}" for expert_id in range(expert_count)
                )
            else:
                expert_names = tuple(
                    f"{name}.experts.{child_name}"
                    for child_name, _child in sorted(
                        indexed_children, key=lambda item: int(item[0])
                    )
                )
            discovered.append(
                _MixtralLayer(
                    name=name,
                    module=module,
                    router=router,
                    expert_count=expert_count,
                    top_k=top_k,
                    expert_names=expert_names,
                    fused_experts=fused,
                )
            )
        return tuple(sorted(discovered, key=lambda layer: layer.name))


class _MixtralRoutingHookHandle:
    """Normalize exact Mixtral gate outputs into the shared routing schema."""

    def __init__(
        self,
        layers: tuple[_MixtralLayer, ...],
        sink: RoutingSinkLike,
        *,
        capture_router_logits: bool,
    ) -> None:
        self._sink = sink
        self._capture_router_logits = capture_router_logits
        self._removed = False
        self._handles: list[Any] = []
        try:
            for layer in layers:
                self._handles.append(layer.router.register_forward_hook(self._callback(layer)))
        except (AttributeError, RuntimeError, TypeError) as error:
            self.remove()
            raise RoutingInstrumentationError(
                f"Failed to attach Mixtral routing hooks: {error}",
                component="hf_causal_lm",
            ) from error

    def _callback(self, layer: _MixtralLayer) -> Any:
        def capture(_module: nn.Module, _inputs: Any, output: Any) -> None:
            if self._removed:
                return
            active_batch = getattr(self._sink, "active_batch", None)
            if not isinstance(active_batch, BatchMeta):
                raise RoutingInstrumentationError(
                    "Mixtral routing hook fired without an active stable-ID batch",
                    component="hf_causal_lm",
                )
            try:
                logits, weights, expert_ids = self._normalize_gate_output(output, layer)
                batch_size = len(active_batch.sample_ids)
                if logits.shape[0] % batch_size != 0:
                    raise ValueError("router rows are not divisible by the active batch size")
                sequence_length = logits.shape[0] // batch_size
                probabilities = torch.softmax(logits.float(), dim=-1)
                snapshot: dict[str, Any] = {
                    "layer_id": layer.name,
                    "selected_expert_ids": expert_ids.reshape(
                        batch_size, sequence_length, layer.top_k
                    ),
                    "selected_weights": weights.reshape(batch_size, sequence_length, layer.top_k),
                    "router_probabilities": probabilities.reshape(
                        batch_size, sequence_length, layer.expert_count
                    ),
                }
                if self._capture_router_logits:
                    snapshot["router_logits"] = logits.reshape(
                        batch_size, sequence_length, layer.expert_count
                    )
                for event in events_from_routing_snapshot(
                    snapshot,
                    sample_ids=active_batch.sample_ids,
                    capture_router_logits=self._capture_router_logits,
                    fallback_layer_id=layer.name,
                ):
                    self._sink.record(event)
            except RoutingInstrumentationError:
                raise
            except (RuntimeError, TypeError, ValueError) as error:
                raise RoutingInstrumentationError(
                    f"Invalid Mixtral routing output for {layer.name}: {error}",
                    component="hf_causal_lm",
                ) from error

        return capture

    @staticmethod
    def _normalize_gate_output(output: Any, layer: _MixtralLayer) -> tuple[Tensor, Tensor, Tensor]:
        if isinstance(output, Tensor):
            logits = output
            probabilities = torch.softmax(logits.float(), dim=-1)
            weights, expert_ids = torch.topk(probabilities, layer.top_k, dim=-1)
            weights = weights / weights.sum(dim=-1, keepdim=True)
        elif (
            isinstance(output, (tuple, list))
            and len(output) >= 3
            and all(isinstance(value, Tensor) for value in output[:3])
        ):
            logits, weights, expert_ids = cast(tuple[Tensor, Tensor, Tensor], tuple(output[:3]))
        else:
            raise ValueError("gate output is neither logits nor a logits/weights/indices tuple")
        if logits.ndim != 2 or logits.shape[-1] != layer.expert_count:
            raise ValueError("router logits have an unexpected shape")
        if weights.shape != expert_ids.shape or weights.ndim != 2:
            raise ValueError("selected routing weights and IDs have inconsistent shapes")
        if weights.shape[-1] != layer.top_k or weights.shape[0] != logits.shape[0]:
            raise ValueError("selected routing output does not match the declared top-k")
        return logits.detach(), weights.detach(), expert_ids.detach()

    def remove(self) -> None:
        """Remove all architecture hooks idempotently."""

        if self._removed:
            return
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._removed = True


class HFCausalLMAdapter:
    """Validated CPU adapter for pinned, Transformers-native Mixtral checkpoints."""

    name = "hf_causal_lm"
    mixtral_checkpoint_layout = "transformers_native_fused"

    def __init__(self) -> None:
        self._strategy = _MixtralStrategy()
        self._config_cache: dict[tuple[str, str, bool], Any] = {}

    def _require_dependencies(self) -> None:
        if importlib.util.find_spec("transformers") is None:
            raise CapabilityError(
                "The Hugging Face adapter is unavailable because transformers is not installed",
                component=self.name,
                remediation="Install with `uv sync --extra hf`.",
            )

    def _validate_safe_identity(self, config: ExperimentConfig) -> str:
        revision = config.model.revision
        if revision is None or _COMMIT_REVISION.fullmatch(revision) is None:
            raise CapabilityError(
                "Hugging Face models require an immutable 40-character commit revision",
                component=self.name,
                remediation="Pin model.revision to the exact Hugging Face repository commit SHA.",
            )
        if config.model.trust_remote_code or config.security.allow_remote_code:
            raise CapabilityError(
                "The validated Hugging Face Mixtral strategy does not execute remote model code",
                component=self.name,
                remediation=(
                    "Set both remote-code opt-ins to false for a native Transformers model."
                ),
            )
        if config.model.checkpoint_format != "safetensors":
            raise CapabilityError(
                "The validated Hugging Face strategy requires safetensors weights",
                component=self.name,
            )
        if config.model.tokenizer_id not in {None, config.model.model_id}:
            raise CapabilityError(
                "A separately hosted tokenizer needs its own immutable revision and is not yet "
                "supported",
                component=self.name,
                remediation="Use the tokenizer stored in the pinned model repository.",
            )
        if (
            config.runtime.backend != "torch_eager_cpu"
            or config.runtime.device != "cpu"
            or config.runtime.device_map != "single"
            or config.runtime.sharding is not None
        ):
            raise CapabilityError(
                "The validated Hugging Face Mixtral strategy requires unsharded single-CPU eager "
                "placement",
                component=self.name,
                remediation=(
                    "Use runtime.backend=torch_eager_cpu, device=cpu, device_map=single, and no "
                    "sharding."
                ),
            )
        if config.model.dtype != "float32" or config.runtime.dtype != config.model.dtype:
            raise CapabilityError(
                "The validated Hugging Face Mixtral strategy requires matching float32 model and "
                "runtime dtypes",
                component=self.name,
            )
        return revision

    def _validate_resolved_config(self, resolved: Any) -> None:
        model_type = getattr(resolved, "model_type", None)
        architectures = tuple(getattr(resolved, "architectures", ()) or ())
        if model_type != _SUPPORTED_MODEL_TYPE or _SUPPORTED_MODEL_CLASS not in architectures:
            raise CapabilityError(
                f"Hugging Face architecture is unsupported: {model_type!r} {architectures!r}",
                component=self.name,
                remediation=(
                    "Use a native MixtralForCausalLM checkpoint or add a validated strategy."
                ),
            )
        expert_count = getattr(resolved, "num_local_experts", None)
        top_k = getattr(resolved, "num_experts_per_tok", None)
        layer_count = getattr(resolved, "num_hidden_layers", None)
        structure = (expert_count, top_k, layer_count)
        if not all(isinstance(value, int) and value > 0 for value in structure):
            raise CapabilityError(
                "Pinned Mixtral configuration lacks valid layer, expert, or top-k metadata",
                component=self.name,
            )
        if cast(int, top_k) > cast(int, expert_count):
            raise CapabilityError(
                "Pinned Mixtral top-k exceeds its expert count", component=self.name
            )

    def _capabilities_from_resolved(self, resolved: Any) -> ModelCapabilities:
        return ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_audio=False,
            is_moe=True,
            supports_router_logits=True,
            supports_token_level_routes=True,
            supported_dtypes=("float32",),
            supported_device_maps=("single",),
            max_context_length=int(resolved.max_position_embeddings),
            requires_remote_code=False,
        )

    def _resolve_source_metadata_files(
        self, config: ExperimentConfig
    ) -> _OfflineSourceMetadataFiles:
        """Resolve and hash only the exact cached config/tokenizer files."""

        revision = self._validate_safe_identity(config)
        self._require_dependencies()
        if not config.model.local_files_only:
            raise CapabilityError(
                "Source-weight-free loading requires model.local_files_only=true",
                component=self.name,
            )
        try:
            from huggingface_hub import hf_hub_download

            paths = tuple(
                Path(
                    hf_hub_download(
                        config.model.model_id,
                        filename,
                        revision=revision,
                        local_files_only=True,
                    )
                )
                for filename in _SOURCE_METADATA_FILES
            )
        except Exception as error:
            raise ModelLoadError(
                f"Unable to resolve exact offline Hugging Face metadata files: {error}",
                component=self.name,
                remediation=(
                    "Populate the exact pinned config and tokenizer files in the local Hub cache."
                ),
            ) from error
        if any(
            not path.is_file() or path.name != name
            for path, name in zip(paths, _SOURCE_METADATA_FILES, strict=True)
        ):
            raise ModelLoadError(
                "Pinned Hugging Face metadata cache returned a missing or mismatched file",
                component=self.name,
            )
        snapshot_paths = {path.parent for path in paths}
        if len(snapshot_paths) != 1:
            raise ModelLoadError(
                "Pinned Hugging Face metadata files did not resolve to one snapshot",
                component=self.name,
            )
        snapshot_path = next(iter(snapshot_paths))
        if snapshot_path.name != revision:
            raise ModelLoadError(
                "Pinned Hugging Face metadata snapshot does not match the exact revision",
                component=self.name,
            )
        return _OfflineSourceMetadataFiles(
            snapshot_path=snapshot_path,
            file_sha256=tuple(
                (name, _sha256_file(path))
                for name, path in zip(_SOURCE_METADATA_FILES, paths, strict=True)
            ),
        )

    def _load_offline_config_and_tokenizer(
        self, config: ExperimentConfig
    ) -> tuple[Any, Any, tuple[tuple[str, str], ...]]:
        """Hash exact cached metadata before parsing config or tokenizer state."""

        files = self._resolve_source_metadata_files(config)
        try:
            from transformers import AutoConfig, AutoTokenizer

            resolved = AutoConfig.from_pretrained(
                files.snapshot_path,
                local_files_only=True,
                trust_remote_code=False,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                files.snapshot_path,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as error:
            raise ModelLoadError(
                f"Unable to construct exact offline Hugging Face metadata: {error}",
                component=self.name,
            ) from error
        return resolved, tokenizer, files.file_sha256

    def _hf_config(self, config: ExperimentConfig) -> Any:
        revision = self._validate_safe_identity(config)
        self._require_dependencies()
        key = (config.model.model_id, revision, config.model.local_files_only)
        if key in self._config_cache:
            return self._config_cache[key]
        try:
            from transformers import AutoConfig

            resolved = AutoConfig.from_pretrained(
                config.model.model_id,
                revision=revision,
                local_files_only=config.model.local_files_only,
                trust_remote_code=False,
            )
        except Exception as error:
            raise ModelLoadError(
                f"Unable to load pinned Hugging Face configuration: {error}",
                component=self.name,
                remediation=(
                    "Set model.local_files_only=false for the first download, or verify the "
                    "model ID and immutable revision."
                ),
            ) from error
        self._validate_resolved_config(resolved)
        self._config_cache[key] = resolved
        return resolved

    def capabilities(self, config: ExperimentConfig) -> ModelCapabilities:
        """Inspect a pinned native config and report only the validated CPU surface."""

        resolved = self._hf_config(config)
        return self._capabilities_from_resolved(resolved)

    def load_empty_export_shell(
        self,
        config: ExperimentConfig,
        runtime: RuntimeForModel,
        expected: ExportedModelIdentity,
    ) -> EmptyExportModelShell:
        """Build a pinned offline Mixtral shell without loading source checkpoint weights."""

        revision = self._validate_safe_identity(config)
        configured_identity = (config.model.model_id, revision)
        exported_identity = (expected.model_id, expected.revision)
        if configured_identity != exported_identity:
            raise ValueError(
                "source-weight-free export model identity does not match the resolved config"
            )
        if (
            expected.resolved_class != _SUPPORTED_RESOLVED_CLASS
            or expected.architecture != _SUPPORTED_MODEL_CLASS
            or _SHA256.fullmatch(expected.source_checksum) is None
        ):
            raise ValueError(
                "source-weight-free export model identity is not the validated Mixtral contract"
            )
        if getattr(runtime, "name", None) != "torch_eager_cpu":
            raise ModelLoadError(
                "The source-weight-free Hugging Face shell requires torch_eager_cpu",
                component=self.name,
            )
        if not config.model.local_files_only:
            raise ModelLoadError(
                "The source-weight-free Hugging Face shell requires offline metadata",
                component=self.name,
            )

        resolved, raw_tokenizer, metadata_hashes = self._load_offline_config_and_tokenizer(config)
        self._validate_resolved_config(resolved)
        try:
            from transformers import AutoModelForCausalLM

            with runtime.execution_context(), torch.device("meta"):
                if self.mixtral_checkpoint_layout == "defuser_linear":
                    with scoped_defuser_linear_mixtral():
                        module = cast(Any, AutoModelForCausalLM).from_config(
                            resolved,
                            trust_remote_code=False,
                        )
                else:
                    module = cast(Any, AutoModelForCausalLM).from_config(
                        resolved,
                        trust_remote_code=False,
                    )
        except Exception as error:
            raise ModelLoadError(
                f"Unable to construct the pinned Hugging Face meta shell: {error}",
                component=self.name,
            ) from error
        resolved_class = f"{module.__class__.__module__}.{module.__class__.__qualname__}"
        if resolved_class != expected.resolved_class:
            raise ModelLoadError(
                "Meta shell resolved class does not match the exported model identity",
                component=self.name,
            )
        persistent_state = cast(nn.Module, module).state_dict()
        if not persistent_state or any(not tensor.is_meta for tensor in persistent_state.values()):
            raise ModelLoadError(
                "Hugging Face architecture-only construction materialized persistent state",
                component=self.name,
            )
        layers = self._strategy.layers(cast(nn.Module, module))
        if len(layers) != int(resolved.num_hidden_layers) or any(
            layer.expert_count != int(resolved.num_local_experts)
            or layer.top_k != int(resolved.num_experts_per_tok)
            for layer in layers
        ):
            raise ModelLoadError(
                "Meta shell routing structure does not match the pinned configuration",
                component=self.name,
            )

        model_body = getattr(module, "model", None)
        rotary = getattr(model_body, "rotary_emb", None)
        if not isinstance(model_body, nn.Module) or not isinstance(rotary, nn.Module):
            raise ModelLoadError(
                "Meta shell does not expose the validated Mixtral rotary embedding",
                component=self.name,
            )
        try:
            model_body.rotary_emb = rotary.__class__(
                config=resolved,
                device=torch.device("cpu"),
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise ModelLoadError(
                f"Unable to reconstruct Mixtral non-persistent rotary buffers: {error}",
                component=self.name,
            ) from error
        remaining_nonpersistent_meta = tuple(
            name
            for name, buffer in cast(nn.Module, module).named_buffers()
            if name not in persistent_state and buffer.is_meta
        )
        if remaining_nonpersistent_meta:
            raise ModelLoadError(
                "Meta shell contains non-persistent buffers that cannot be loaded: "
                + ", ".join(remaining_nonpersistent_meta),
                component=self.name,
            )
        capabilities = self._capabilities_from_resolved(resolved)
        descriptor = ModelDescriptor(
            model_id=expected.model_id,
            revision=expected.revision,
            resolved_class=expected.resolved_class,
            architecture=expected.architecture,
            checksum=expected.source_checksum,
            capabilities=capabilities,
        )
        return EmptyExportModelShell(
            model=module,
            tokenizer=HFTokenizerAdapter(raw_tokenizer),
            descriptor=descriptor,
            source_metadata_file_sha256=metadata_hashes,
        )

    def load(self, config: ExperimentConfig, runtime: RuntimeForModel) -> LoadedModel:
        """Load an immutable native Mixtral checkpoint using safetensors only."""

        if getattr(runtime, "name", None) != "torch_eager_cpu":
            raise ModelLoadError(
                "The validated Hugging Face Mixtral strategy received a non-CPU runtime object",
                component=self.name,
            )
        resolved = self._hf_config(config)
        capabilities = self.capabilities(config)
        if config.model.dtype not in capabilities.supported_dtypes:
            raise ModelLoadError(
                f"Validated Hugging Face CPU smoke path does not support {config.model.dtype}",
                component=self.name,
            )
        revision = cast(str, config.model.revision)
        started = time.perf_counter()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                config.model.model_id,
                revision=revision,
                local_files_only=config.model.local_files_only,
                trust_remote_code=False,
            )
            with runtime.execution_context():
                load_kwargs = {
                    "revision": revision,
                    "config": resolved,
                    "local_files_only": config.model.local_files_only,
                    "trust_remote_code": False,
                    "use_safetensors": True,
                    "weights_only": True,
                    "dtype": torch.float32,
                    "output_loading_info": True,
                }
                exact_legacy_stories = (config.model.model_id, revision) == (
                    _STORIES15M_MODEL_ID,
                    _STORIES15M_REVISION,
                )
                if exact_legacy_stories and self.mixtral_checkpoint_layout == "defuser_linear":
                    with scoped_defuser_linear_mixtral():
                        loaded_result = AutoModelForCausalLM.from_pretrained(
                            config.model.model_id,
                            **load_kwargs,
                        )
                else:
                    loaded_result = AutoModelForCausalLM.from_pretrained(
                        config.model.model_id,
                        **load_kwargs,
                    )
                if (
                    not isinstance(loaded_result, tuple)
                    or len(loaded_result) != 2
                    or not isinstance(loaded_result[1], dict)
                ):
                    raise ModelLoadError(
                        "Transformers did not return auditable checkpoint loading information",
                        component=self.name,
                    )
                module, loading_info = loaded_result
                _validate_checkpoint_loading_info(loading_info)
                cast(nn.Module, module).eval()
        except ModelLoadError:
            raise
        except Exception as error:
            raise ModelLoadError(
                f"Unable to load pinned Hugging Face safetensors checkpoint: {error}",
                component=self.name,
                remediation=(
                    "Verify the immutable revision, install `uv sync --extra hf`, and permit "
                    "network access only for the initial cache population."
                ),
            ) from error
        elapsed = time.perf_counter() - started
        if module.__class__.__name__ != _SUPPORTED_MODEL_CLASS:
            raise ModelLoadError(
                "Resolved model class is not "
                f"{_SUPPORTED_MODEL_CLASS}: {module.__class__.__name__}",
                component=self.name,
            )
        layers = self._strategy.layers(cast(nn.Module, module))
        if len(layers) != int(resolved.num_hidden_layers):
            raise ModelLoadError(
                "Loaded Mixtral module inventory does not match the pinned layer count",
                component=self.name,
            )
        if any(
            layer.expert_count != int(resolved.num_local_experts)
            or layer.top_k != int(resolved.num_experts_per_tok)
            for layer in layers
        ):
            raise ModelLoadError(
                "Loaded Mixtral routing structure does not match the pinned configuration",
                component=self.name,
            )
        descriptor = ModelDescriptor(
            model_id=config.model.model_id,
            revision=revision,
            resolved_class=f"{module.__class__.__module__}.{module.__class__.__qualname__}",
            architecture=_SUPPORTED_MODEL_CLASS,
            checksum=model_state_sha256(cast(nn.Module, module)),
            capabilities=capabilities,
        )
        return LoadedModel(
            model=module,
            tokenizer=HFTokenizerAdapter(tokenizer),
            descriptor=descriptor,
            load_time_seconds=elapsed,
        )

    def enumerate_modules(self, model: LoadedModel) -> tuple[ModuleInfo, ...]:
        """Return an architecture-validated inventory, including fused expert slices."""

        root = cast(nn.Module, model.model)
        layers = self._strategy.layers(root)
        routers = {f"{layer.name}.gate": layer for layer in layers}
        fused = {f"{layer.name}.experts": layer for layer in layers if layer.fused_experts}
        expert_prefixes = {
            expert_name: (layer.name, expert_id)
            for layer in layers
            for expert_id, expert_name in enumerate(layer.expert_names)
        }
        get_input_embeddings = getattr(root, "get_input_embeddings", None)
        get_output_embeddings = getattr(root, "get_output_embeddings", None)
        input_embedding = get_input_embeddings() if callable(get_input_embeddings) else None
        output_embedding = get_output_embeddings() if callable(get_output_embeddings) else None
        inventory: list[ModuleInfo] = []
        for name, module in root.named_modules():
            if not name:
                continue
            if name in fused:
                inventory.extend(_fused_expert_inventory(name, module, fused[name]))
                continue
            if any(name.startswith(f"{container}.") for container in fused):
                continue
            direct_parameters = tuple(module.parameters(recurse=False))
            direct_buffers = tuple(module.buffers(recurse=False))
            if not direct_parameters and not direct_buffers:
                continue
            expert_match = next(
                (
                    (prefix, layer_id, expert_id)
                    for prefix, (layer_id, expert_id) in expert_prefixes.items()
                    if name == prefix or name.startswith(f"{prefix}.")
                ),
                None,
            )
            inventory.append(
                ModuleInfo(
                    name=name,
                    class_name=f"{module.__class__.__module__}.{module.__class__.__qualname__}",
                    parameter_count=sum(value.numel() for value in direct_parameters),
                    size_bytes=sum(
                        value.numel() * value.element_size()
                        for value in (*direct_parameters, *direct_buffers)
                    ),
                    is_router=name in routers,
                    is_output_head=module is output_embedding,
                    is_embedding=module is input_embedding or isinstance(module, nn.Embedding),
                    is_expert=expert_match is not None,
                    layer_id=None if expert_match is None else expert_match[1],
                    expert_id=None if expert_match is None else expert_match[2],
                    supported_precisions=_module_supported_precisions(module),
                )
            )
        return tuple(sorted(inventory, key=lambda item: item.name))

    def discover_moe(self, model: LoadedModel) -> MoEDescriptor | None:
        """Discover exact Mixtral gates and expert collections by validated module type."""

        layers = self._strategy.layers(cast(nn.Module, model.model))
        if not layers:
            return None
        return MoEDescriptor(
            layers=tuple(
                MoELayerDescriptor(
                    layer_id=layer.name,
                    module_name=layer.name,
                    router_module_name=f"{layer.name}.gate",
                    expert_module_names=layer.expert_names,
                    expert_count=layer.expert_count,
                    top_k=layer.top_k,
                )
                for layer in layers
            ),
            supports_router_logits=True,
            supports_token_level_routes=True,
        )

    def attach_routing_hooks(
        self, model: LoadedModel, sink: RoutingSinkLike, config: ExperimentConfig
    ) -> HookHandle:
        """Attach removable hooks to validated Mixtral router modules."""

        layers = self._strategy.layers(cast(nn.Module, model.model))
        if not layers:
            raise CapabilityError(
                "The loaded model exposes no validated Mixtral routing layers",
                component=self.name,
            )
        return _MixtralRoutingHookHandle(
            layers,
            sink,
            capture_router_logits=config.routing.capture_router_logits,
        )

    def generate(self, model: LoadedModel, batch: ModelBatch, config: DecodeConfig) -> ModelOutput:
        """Run deterministic or explicitly configured Transformers generation."""

        if batch.multimodal_inputs is not None:
            raise CapabilityError("The validated Mixtral adapter is text-only", component=self.name)
        module = cast(Any, model.model)
        input_ids = cast(Tensor, batch.input_ids)
        attention_mask = cast(Tensor | None, batch.attention_mask)
        generation: dict[str, Any] = {
            "max_new_tokens": config.max_new_tokens,
            "do_sample": config.do_sample,
            "pad_token_id": model.tokenizer.raw.pad_token_id,
            "eos_token_id": model.tokenizer.raw.eos_token_id,
        }
        if config.do_sample:
            generation["temperature"] = config.temperature
            if config.top_k is not None:
                generation["top_k"] = config.top_k
        with torch.inference_mode():
            generated = module.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation,
            )
        suffix = generated[:, input_ids.shape[1] :].detach().cpu().tolist()
        token_ids = tuple(tuple(int(token) for token in row) for row in suffix)
        return ModelOutput(
            sample_ids=batch.sample_ids,
            token_ids=token_ids,
            texts=tuple(model.tokenizer.decode(row) for row in token_ids),
        )

    def forward_loss(self, model: LoadedModel, batch: ModelBatch) -> LossOutput:
        """Compute token-level causal loss without retaining model output text."""

        if batch.multimodal_inputs is not None:
            raise CapabilityError("The validated Mixtral adapter is text-only", component=self.name)
        input_ids = cast(Tensor, batch.input_ids)
        attention_mask = cast(Tensor | None, batch.attention_mask)
        with torch.inference_mode():
            output = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        logits = getattr(output, "logits", None)
        if not isinstance(logits, Tensor):
            raise ModelLoadError(
                "The loaded Hugging Face model did not return causal-LM logits",
                component=self.name,
            )
        losses, counts = causal_loss_per_sample(logits, input_ids, attention_mask)
        return LossOutput(
            sample_ids=batch.sample_ids,
            negative_log_likelihoods=losses,
            token_counts=counts,
        )


def _validate_checkpoint_loading_info(loading_info: dict[str, Any]) -> None:
    """Reject silently missing, unused, mismatched, or errored checkpoint tensors."""

    failures: list[str] = []
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        value = loading_info.get(key, ())
        if value:
            try:
                count = len(value)
            except TypeError:
                count = 1
            failures.append(f"{key}={count}")
    if failures:
        raise ModelLoadError(
            "Pinned Hugging Face checkpoint did not load exactly: " + ", ".join(failures),
            component="hf_causal_lm",
            remediation=(
                "Use a checkpoint/model-class conversion with an exact empty Transformers "
                "loading-info report; do not evaluate newly initialized or unused weights."
            ),
        )


def _fused_expert_inventory(
    name: str, module: nn.Module, layer: _MixtralLayer
) -> tuple[ModuleInfo, ...]:
    parameters = tuple(module.parameters(recurse=False))
    buffers = tuple(module.buffers(recurse=False))
    values = (*parameters, *buffers)
    invalid_values = any(
        value.ndim == 0 or value.shape[0] != layer.expert_count for value in values
    )
    if not values or invalid_values:
        raise CapabilityError(
            f"Fused Mixtral experts in {name} cannot be partitioned by expert ID",
            component="hf_causal_lm",
        )
    return tuple(
        ModuleInfo(
            name=f"{name}.{expert_id}",
            class_name=f"{module.__class__.__module__}.{module.__class__.__qualname__}.ExpertSlice",
            parameter_count=sum(value[expert_id].numel() for value in parameters),
            size_bytes=sum(value[expert_id].numel() * value.element_size() for value in values),
            is_expert=True,
            layer_id=layer.name,
            expert_id=expert_id,
            supported_precisions=("float32",),
        )
        for expert_id in range(layer.expert_count)
    )


def _module_supported_precisions(module: nn.Module) -> tuple[Precision, ...]:
    """Advertise native CPU weight quantization only for addressable linears.

    The validated Transformers Mixtral model represents its experts as fused
    tensor containers.  Those logical ``ExpertSlice`` inventory entries are
    intentionally created elsewhere with a float32-only precision set.  A
    concrete ``nn.Linear`` leaf, by contrast, is addressable by the native CPU
    replacement quantizers and may participate in an INT8 or INT4 policy.
    """

    if isinstance(module, nn.Linear):
        return ("float32", "int8", "int4")
    return ("float32",)


class HFLinearMixtralCausalLMAdapter(HFCausalLMAdapter):
    """Exact Stories15M adapter exposing Defuser's per-expert linear layout."""

    name = "hf_causal_lm_linear_mixtral"
    mixtral_checkpoint_layout = "defuser_linear"

    def _validate_safe_identity(self, config: ExperimentConfig) -> str:
        revision = super()._validate_safe_identity(config)
        if (config.model.model_id, revision) != (
            _STORIES15M_MODEL_ID,
            _STORIES15M_REVISION,
        ):
            raise CapabilityError(
                "The linear Mixtral adapter is limited to the exact pinned Stories15M revision",
                component=self.name,
                remediation="Use hf_causal_lm for other native Transformers checkpoints.",
            )
        return revision


def create_adapter() -> HFCausalLMAdapter:
    """Registry factory for the lazy, safely pinned Hugging Face adapter."""

    return HFCausalLMAdapter()


def create_linear_mixtral_adapter() -> HFLinearMixtralCausalLMAdapter:
    """Registry factory for the explicit Defuser-linear Stories15M adapter."""

    return HFLinearMixtralCausalLMAdapter()
