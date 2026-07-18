"""Architecture-independent model adapter contracts and transport records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from inkling_quant_lab.config import DecodeConfig, ExperimentConfig, Precision


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Factual model features and constraints."""

    supports_text: bool
    supports_images: bool
    supports_audio: bool
    is_moe: bool
    supports_router_logits: bool
    supports_token_level_routes: bool
    supported_dtypes: tuple[str, ...]
    supported_device_maps: tuple[str, ...]
    max_context_length: int
    requires_remote_code: bool = False


@dataclass(frozen=True, slots=True)
class ModuleInfo:
    """Normalized module inventory entry for policy resolution."""

    name: str
    class_name: str
    parameter_count: int
    size_bytes: int
    is_router: bool = False
    is_output_head: bool = False
    is_embedding: bool = False
    is_multimodal_projector: bool = False
    is_expert: bool = False
    layer_id: str | None = None
    expert_id: int | None = None
    supported_precisions: tuple[Precision, ...] = (
        "float32",
        "float16",
        "bfloat16",
        "int8",
        "int4",
    )


@dataclass(frozen=True, slots=True)
class MoELayerDescriptor:
    """One discovered routed layer without assuming universal attribute names."""

    layer_id: str
    module_name: str
    router_module_name: str
    expert_module_names: tuple[str, ...]
    expert_count: int
    top_k: int


@dataclass(frozen=True, slots=True)
class MoEDescriptor:
    """Architecture-normalized MoE structure."""

    layers: tuple[MoELayerDescriptor, ...]
    supports_router_logits: bool
    supports_token_level_routes: bool


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Stable model identity used for capability and quantizer checks."""

    model_id: str
    revision: str | None
    resolved_class: str
    architecture: str
    checksum: str
    capabilities: ModelCapabilities


@dataclass(frozen=True, slots=True)
class ExportedModelIdentity:
    """Model identity declared by a safe exported-candidate recipe."""

    model_id: str
    revision: str
    resolved_class: str
    architecture: str
    source_checksum: str


@dataclass(frozen=True, slots=True)
class EmptyExportModelShell:
    """Metadata-only model shell whose persistent tensors remain on ``meta``."""

    model: Any
    tokenizer: Any
    descriptor: ModelDescriptor
    source_metadata_file_sha256: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class SourceWeightFreeReloadProvenance:
    """Auditable facts from loading one candidate export without source weights."""

    reload_adapter: str
    backend: str
    model_id: str
    revision: str
    resolved_class: str
    architecture: str
    source_model_checksum: str
    metadata_file: str
    metadata_sha256: str
    tensor_file: str
    tensor_sha256: str
    bundle_sha256: str
    source_metadata_file_sha256: tuple[tuple[str, str], ...]
    candidate_state_checksum: str
    quantized_module_names: tuple[str, ...]
    native_wrapper_count: int
    tensor_count: int
    strict_load: bool
    assign: bool
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    meta_tensor_names: tuple[str, ...]
    source_weights_loaded: bool
    schema_version: Literal["source-weight-free-reload-v1"] = "source-weight-free-reload-v1"

    def as_dict(self) -> dict[str, Any]:
        """Return a stable JSON-safe representation."""

        return {
            "schema_version": self.schema_version,
            "reload_adapter": self.reload_adapter,
            "backend": self.backend,
            "model_id": self.model_id,
            "revision": self.revision,
            "resolved_class": self.resolved_class,
            "architecture": self.architecture,
            "source_model_checksum": self.source_model_checksum,
            "metadata_file": self.metadata_file,
            "metadata_sha256": self.metadata_sha256,
            "tensor_file": self.tensor_file,
            "tensor_sha256": self.tensor_sha256,
            "bundle_sha256": self.bundle_sha256,
            "source_metadata_file_sha256": dict(self.source_metadata_file_sha256),
            "candidate_state_checksum": self.candidate_state_checksum,
            "quantized_module_names": list(self.quantized_module_names),
            "native_wrapper_count": self.native_wrapper_count,
            "tensor_count": self.tensor_count,
            "strict_load": self.strict_load,
            "assign": self.assign,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "meta_tensor_names": list(self.meta_tensor_names),
            "source_weights_loaded": self.source_weights_loaded,
        }


@dataclass(frozen=True, slots=True)
class ModelBatch:
    """Tokenized model inputs with stable sample identity."""

    sample_ids: tuple[str, ...]
    input_ids: Any
    attention_mask: Any | None = None
    labels: Any | None = None
    multimodal_inputs: Any | None = None


@dataclass(frozen=True, slots=True)
class ModelOutput:
    """Generated token IDs and optional ephemeral decoded text."""

    sample_ids: tuple[str, ...]
    token_ids: tuple[tuple[int, ...], ...]
    texts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LossOutput:
    """Per-sample negative log likelihood and token counts."""

    sample_ids: tuple[str, ...]
    negative_log_likelihoods: tuple[float, ...]
    token_counts: tuple[int, ...]

    @property
    def mean_nll(self) -> float:
        """Return token-weighted mean negative log likelihood."""

        denominator = sum(self.token_counts)
        if denominator <= 0:
            raise ValueError("loss output contains no evaluated tokens")
        return (
            sum(
                loss * count
                for loss, count in zip(
                    self.negative_log_likelihoods, self.token_counts, strict=True
                )
            )
            / denominator
        )


@dataclass(slots=True)
class LoadedModel:
    """Runtime-owned model plus adapter/tokenizer metadata."""

    model: Any
    tokenizer: Any
    descriptor: ModelDescriptor
    load_time_seconds: float
    load_time_kind: Literal[
        "cold_model_load",
        "candidate_reconstruction",
        "candidate_pinned_source_quantization",
        "candidate_export_reload",
        "candidate_source_weight_free_export_load",
        "unmeasured_internal_fixture",
    ] = "cold_model_load"


class HookHandle(Protocol):
    """A removable instrumentation handle."""

    def remove(self) -> None: ...


class RuntimeForModel(Protocol):
    """Narrow runtime surface needed by model adapters."""

    def execution_context(self) -> Any: ...


@runtime_checkable
class SourceWeightFreeModelAdapter(Protocol):
    """Adapter surface for constructing a trusted shell without source weights."""

    def load_empty_export_shell(
        self,
        config: ExperimentConfig,
        runtime: RuntimeForModel,
        expected: ExportedModelIdentity,
    ) -> EmptyExportModelShell: ...


class RoutingSinkLike(Protocol):
    """Narrow routing sink surface to avoid model/routing coupling."""

    def start_batch(self, batch_meta: Any) -> None: ...

    def record(self, event: Any) -> None: ...

    def record_aggregate_batch(
        self,
        *,
        layer_id: str,
        sample_ids: tuple[str, ...],
        tokens_per_sample: int,
        expert_selection_counts: Any,
        expert_weight_sums: Any,
    ) -> None: ...

    def end_batch(self) -> None: ...

    def close(self) -> Any: ...


class ModelAdapter(Protocol):
    """Architecture-independent model operation contract."""

    def capabilities(self, config: ExperimentConfig) -> ModelCapabilities: ...

    def load(self, config: ExperimentConfig, runtime: RuntimeForModel) -> LoadedModel: ...

    def enumerate_modules(self, model: LoadedModel) -> tuple[ModuleInfo, ...]: ...

    def discover_moe(self, model: LoadedModel) -> MoEDescriptor | None: ...

    def attach_routing_hooks(
        self, model: LoadedModel, sink: RoutingSinkLike, config: ExperimentConfig
    ) -> HookHandle: ...

    def generate(
        self, model: LoadedModel, batch: ModelBatch, config: DecodeConfig
    ) -> ModelOutput: ...

    def forward_loss(self, model: LoadedModel, batch: ModelBatch) -> LossOutput: ...
