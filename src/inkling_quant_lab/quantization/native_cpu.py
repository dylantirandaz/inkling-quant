"""Capability-gated native PyTorch CPU quantization kernels.

The portable reference backends intentionally reconstruct quantized values in
Python on every forward.  This module is a separate execution path: INT8 uses
PyTorch's prepacked dynamic quantized-linear operator and INT4 uses the ATen
KleidiAI opaque W4A8 packing format.  The serialized state remains ordinary
tensors so exports stay pickle-free.
"""

from __future__ import annotations

import copy
import time
import warnings
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, Self, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from inkling_quant_lab.config import QuantizationConfig
from inkling_quant_lab.models.base import LoadedModel, ModelBatch, ModelDescriptor
from inkling_quant_lab.quantization.base import (
    CalibrationArtifact,
    ExportArtifact,
    QuantizedModel,
    ResolvedPolicyLike,
    SupportReport,
)
from inkling_quant_lab.quantization.int8 import _replace_module
from inkling_quant_lab.quantization.reference import (
    build_manifest,
    export_recipe,
    finalize_serialized_manifest,
    model_storage_bytes,
)
from inkling_quant_lab.runtimes.base import RuntimeCapabilities

_INT8_OPERATOR = "quantized::linear_dynamic"
_INT4_PACK_OPERATOR = "aten::_dyn_quant_pack_4bit_weight"
_INT4_MATMUL_OPERATOR = "aten::_dyn_quant_matmul_4bit"


@dataclass(frozen=True, slots=True)
class NativeKernelCapability:
    """Result of executing a minimal native-kernel capability probe."""

    supported: bool
    implementation: str | None
    reasons: tuple[str, ...] = ()


def _quantized_engine_candidates(requested: str | None) -> tuple[str, ...]:
    supported = tuple(
        engine for engine in torch.backends.quantized.supported_engines if engine != "none"
    )
    if requested is not None and requested != "auto":
        return (requested,)
    active = torch.backends.quantized.engine
    ordered = (active, "x86", "fbgemm", "onednn", "qnnpack")
    return tuple(dict.fromkeys(engine for engine in ordered if engine in supported))


def _activate_quantized_engine(engine: str) -> None:
    if engine not in torch.backends.quantized.supported_engines:
        raise RuntimeError(
            f"quantized engine {engine!r} is not compiled into this PyTorch build; "
            f"available engines: {', '.join(torch.backends.quantized.supported_engines) or 'none'}"
        )
    if torch.backends.quantized.engine != engine:
        torch.backends.quantized.engine = engine


def _prepack_dynamic_int8(
    weight_int8: Tensor,
    weight_scales: Tensor,
    bias: Tensor | None,
    engine: str,
) -> Any:
    _activate_quantized_engine(engine)
    zero_points = torch.zeros(weight_int8.shape[0], dtype=torch.int64, device="cpu")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="torch.quantize_per_tensor, torch.quantize_per_channel.*",
            category=UserWarning,
        )
        quantized_weight = torch._make_per_channel_quantized_tensor(
            weight_int8.contiguous(), weight_scales.float(), zero_points, 0
        )
    return torch.ops.quantized.linear_prepack(quantized_weight, bias)


@cache
def probe_native_dynamic_int8(engine: str | None = None) -> NativeKernelCapability:
    """Execute a tiny prepack/matmul probe for a requested or auto-selected engine."""

    candidates = _quantized_engine_candidates(engine)
    if not candidates:
        return NativeKernelCapability(
            supported=False,
            implementation=None,
            reasons=("this PyTorch build exposes no quantized CPU engine",),
        )
    failures: list[str] = []
    for candidate in candidates:
        try:
            weight = torch.tensor(((1, -2, 3, -4), (-4, 3, -2, 1)), dtype=torch.int8)
            scales = torch.tensor((0.01, 0.02), dtype=torch.float32)
            packed = _prepack_dynamic_int8(weight, scales, None, candidate)
            output = torch.ops.quantized.linear_dynamic(
                torch.ones((1, 4), dtype=torch.float32), packed, False
            )
            if output.shape != (1, 2) or not bool(torch.isfinite(output).all()):
                raise RuntimeError("native dynamic INT8 probe returned invalid output")
            return NativeKernelCapability(supported=True, implementation=candidate)
        except (AttributeError, RuntimeError, TypeError, NotImplementedError) as error:
            failures.append(f"{candidate}: {error}")
    return NativeKernelCapability(
        supported=False,
        implementation=None,
        reasons=("native dynamic INT8 probe failed: " + " | ".join(failures),),
    )


def _pack_int4_nibbles(weight: Tensor) -> tuple[Tensor, Tensor]:
    float_weight = weight.detach().float()
    scales = float_weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7.0
    signed = torch.round(float_weight / scales).clamp(-8, 7).to(torch.int16)
    unsigned = (signed + 8).to(torch.uint8)
    if unsigned.shape[1] % 2:
        unsigned = F.pad(unsigned, (0, 1), value=8)
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
    return packed.contiguous(), scales.contiguous()


def _opaque_int4_pack(source: nn.Linear) -> Tensor:
    packed_nibbles, scales = _pack_int4_nibbles(source.weight)
    bias = None if source.bias is None else source.bias.detach().float().contiguous()
    return cast(
        Tensor,
        torch.ops.aten._dyn_quant_pack_4bit_weight(
            packed_nibbles,
            scales,
            bias,
            source.in_features,
            source.in_features,
            source.out_features,
        ),
    )


@lru_cache(maxsize=1)
def probe_native_int4_kleidiai() -> NativeKernelCapability:
    """Execute a tiny ATen opaque-pack and KleidiAI W4A8 matmul probe."""

    try:
        if not bool(cast(Any, torch.backends.kleidiai).is_available()):
            return NativeKernelCapability(
                supported=False,
                implementation=None,
                reasons=("this PyTorch build reports torch.backends.kleidiai unavailable",),
            )
        source = nn.Linear(16, 16, bias=True).eval()
        packed = _opaque_int4_pack(source)
        output = torch.ops.aten._dyn_quant_matmul_4bit(
            torch.ones((1, 16), dtype=torch.float32), packed, 16, 16, 16
        )
        if output.shape != (1, 16) or not bool(torch.isfinite(output).all()):
            raise RuntimeError("native KleidiAI INT4 probe returned invalid output")
    except (AttributeError, RuntimeError, TypeError, NotImplementedError) as error:
        return NativeKernelCapability(
            supported=False,
            implementation=None,
            reasons=(f"native KleidiAI INT4 probe failed: {error}",),
        )
    return NativeKernelCapability(supported=True, implementation="aten_kleidiai")


class NativeDynamicInt8Linear(nn.Module):
    """Linear using a prepacked native W8A8 dynamic-activation CPU kernel."""

    def __init__(self, source: nn.Linear, *, engine: str) -> None:
        super().__init__()
        weight = source.weight.detach().float()
        scales = weight.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        quantized = torch.round(weight / scales.unsqueeze(1)).clamp(-127, 127).to(torch.int8)
        self.weight_int8: Tensor
        self.weight_scales: Tensor
        self.register_buffer("weight_int8", quantized.contiguous())
        self.register_buffer("weight_scales", scales.contiguous())
        self.bias: nn.Parameter | None
        source_bias = getattr(source, "bias", None)
        if source_bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(source_bias.detach().float().clone(), requires_grad=False)
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.engine = engine
        self._packed_weight: Any = None
        self._repack()
        cast(Any, self).register_load_state_dict_post_hook(self._repack_after_load)

    @classmethod
    def from_empty(
        cls,
        *,
        in_features: int,
        out_features: int,
        bias: bool,
        engine: str,
        device: str | torch.device = "meta",
    ) -> Self:
        """Create a state-compatible shell without reading floating-point source weights."""

        if in_features <= 0 or out_features <= 0:
            raise ValueError("native dynamic INT8 dimensions must be positive")
        if not engine:
            raise ValueError("native dynamic INT8 empty construction requires an engine")
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result.register_buffer(
            "weight_int8",
            torch.empty((out_features, in_features), dtype=torch.int8, device=device),
        )
        result.register_buffer(
            "weight_scales",
            torch.empty((out_features,), dtype=torch.float32, device=device),
        )
        if bias:
            result.bias = nn.Parameter(
                torch.empty((out_features,), dtype=torch.float32, device=device),
                requires_grad=False,
            )
        else:
            result.bias = None
        result.in_features = in_features
        result.out_features = out_features
        result.engine = engine
        result._packed_weight = None
        cast(Any, result).register_load_state_dict_post_hook(result._repack_after_load)
        return result

    def _repack(self) -> None:
        self._packed_weight = _prepack_dynamic_int8(
            self.weight_int8, self.weight_scales, self.bias, self.engine
        )

    def _repack_after_load(self, module: nn.Module, incompatible_keys: Any) -> None:
        if module is not self:
            raise RuntimeError("native INT8 load hook received a different module")
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            return
        state: tuple[Tensor, ...] = (self.weight_int8, self.weight_scales)
        if self.bias is not None:
            state = (*state, self.bias)
        if any(value.is_meta for value in state):
            return
        self._repack()

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the prepacked native kernel without reconstructing weight tensors."""

        if self._packed_weight is None:
            raise RuntimeError("native dynamic INT8 linear has not loaded and prepacked weights")
        if inputs.device.type != "cpu" or inputs.dtype != torch.float32:
            raise RuntimeError("native dynamic INT8 linear requires float32 CPU activations")
        if inputs.shape[-1] != self.in_features:
            raise RuntimeError(
                f"native dynamic INT8 expected {self.in_features} input features, "
                f"received {inputs.shape[-1]}"
            )
        output_shape = (*inputs.shape[:-1], self.out_features)
        flat = inputs.reshape(-1, self.in_features).contiguous()
        result = torch.ops.quantized.linear_dynamic(flat, self._packed_weight, False)
        return cast(Tensor, result.reshape(output_shape))


class NativeInt4KleidiAILinear(nn.Module):
    """Linear using an opaque, prepacked ATen KleidiAI W4A8 kernel."""

    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        capability = probe_native_int4_kleidiai()
        if not capability.supported:
            raise RuntimeError("; ".join(capability.reasons))
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.group_size = source.in_features
        self.packed_weight: Tensor
        self.register_buffer("packed_weight", _opaque_int4_pack(source))

    def forward(self, inputs: Tensor) -> Tensor:
        """Run the opaque native kernel without unpacking weights on the forward path."""

        if inputs.device.type != "cpu" or inputs.dtype != torch.float32:
            raise RuntimeError("native KleidiAI INT4 linear requires float32 CPU activations")
        if inputs.shape[-1] != self.in_features:
            raise RuntimeError(
                f"native KleidiAI INT4 expected {self.in_features} input features, "
                f"received {inputs.shape[-1]}"
            )
        output_shape = (*inputs.shape[:-1], self.out_features)
        flat = inputs.reshape(-1, self.in_features).contiguous()
        result = torch.ops.aten._dyn_quant_matmul_4bit(
            flat,
            self.packed_weight,
            self.group_size,
            self.in_features,
            self.out_features,
        )
        return cast(Tensor, result.reshape(output_shape))


def _replace_native_linears(
    root: nn.Module,
    precision_map: dict[str, str],
    *,
    precision: str,
    factory: Any,
) -> tuple[str, ...]:
    named = dict(root.named_modules())
    replaced: list[str] = []
    for name in sorted(precision_map):
        module = named.get(name)
        if precision_map[name] == precision and isinstance(module, nn.Linear):
            _replace_module(root, name, factory(module))
            replaced.append(name)
    return tuple(replaced)


class TorchNativeDynamicInt8Quantizer:
    """Native PyTorch dynamic INT8 backend with one-time weight prepacking."""

    name = "torch_native_dynamic_int8"

    def check_support(
        self,
        model: ModelDescriptor,
        runtime: RuntimeCapabilities,
        config: QuantizationConfig,
    ) -> SupportReport:
        """Require CPU float32 execution and a successfully executed quantized kernel."""

        del model
        requested = config.parameters.get("quantized_engine")
        if requested is not None and not isinstance(requested, str):
            return SupportReport(
                available=True,
                supported=False,
                component=self.name,
                reasons=("quantized_engine must be a string or omitted",),
                supported_precisions=("float32", "int8"),
            )
        capability = probe_native_dynamic_int8(requested)
        runtime_supported = runtime.available and "cpu" in runtime.devices
        reasons = (
            *(() if runtime_supported else (*runtime.reasons, "native INT8 requires CPU")),
            *capability.reasons,
        )
        return SupportReport(
            available=True,
            supported=runtime_supported and capability.supported,
            component=self.name,
            reasons=reasons,
            warnings=(
                "uses a PyTorch quantized-engine ABI; exact engine and torch version are pinned",
            ),
            remediation=(
                None
                if capability.supported
                else "Use torch_dynamic_int8 as the portable fallback or install a PyTorch build "
                "with a working quantized CPU engine."
            ),
            supported_precisions=("float32", "int8"),
        )

    def calibrate(
        self,
        model: LoadedModel,
        samples: tuple[ModelBatch, ...],
        config: QuantizationConfig,
    ) -> CalibrationArtifact | None:
        """Dynamic per-token activation quantization needs no offline calibration."""

        del model, samples, config
        return None

    def quantize(
        self,
        model: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        """Replace selected linears with prepacked native dynamic INT8 modules."""

        requested = config.parameters.get("quantized_engine")
        capability = probe_native_dynamic_int8(requested if isinstance(requested, str) else None)
        if not capability.supported or capability.implementation is None:
            raise RuntimeError("; ".join(capability.reasons) or "native INT8 kernel unavailable")
        started = time.perf_counter()
        candidate = copy.deepcopy(cast(nn.Module, model.model))
        engine = capability.implementation
        replaced = _replace_native_linears(
            candidate,
            policy.precision_map,
            precision="int8",
            factory=lambda module: NativeDynamicInt8Linear(module, engine=engine),
        )
        if not replaced:
            raise ValueError("native dynamic INT8 policy selected no eligible linear modules")
        loaded = LoadedModel(
            model=candidate,
            tokenizer=copy.deepcopy(model.tokenizer),
            descriptor=model.descriptor,
            load_time_seconds=time.perf_counter() - started,
            load_time_kind="candidate_reconstruction",
        )
        manifest = build_manifest(
            backend=self.name,
            backend_version=torch.__version__,
            method=config.method,
            source=model.descriptor,
            precision_map=policy.precision_map,
            serialized_size_bytes=model_storage_bytes(candidate),
            calibration=calibration,
            parameters={
                **config.parameters,
                "activation_quantization": "dynamic_per_tensor_uint8",
                "kernel": _INT8_OPERATOR,
                "quantized_engine": engine,
                "weight_granularity": "per_output_channel",
                "weight_bits": 8,
            },
            warnings=(
                "Native execution was capability-probed; results do not generalize to other "
                "PyTorch builds or CPUs without a new probe.",
            ),
        )
        return finalize_serialized_manifest(
            QuantizedModel(loaded=loaded, manifest=manifest), config
        )

    def export(
        self, model: QuantizedModel, destination: Path, config: QuantizationConfig
    ) -> ExportArtifact:
        """Export raw quantized tensors; native packed state is rebuilt safely on reload."""

        return export_recipe(model, destination, config)


class TorchNativeInt4KleidiAIQuantizer:
    """Native ATen/KleidiAI dynamic-activation W4A8 CPU backend."""

    name = "torch_native_int4_kleidiai"

    def check_support(
        self,
        model: ModelDescriptor,
        runtime: RuntimeCapabilities,
        config: QuantizationConfig,
    ) -> SupportReport:
        """Require CPU execution and a successfully executed KleidiAI kernel probe."""

        del model, config
        capability = probe_native_int4_kleidiai()
        runtime_supported = runtime.available and "cpu" in runtime.devices
        reasons = (
            *(() if runtime_supported else (*runtime.reasons, "native INT4 requires CPU")),
            *capability.reasons,
        )
        return SupportReport(
            available=True,
            supported=runtime_supported and capability.supported,
            component=self.name,
            reasons=reasons,
            warnings=("opaque packed weights require the recorded PyTorch/KleidiAI kernel ABI",),
            remediation=(
                None
                if capability.supported
                else "Use torch_weight_only_int4 as the portable fallback or install a PyTorch "
                "CPU build with KleidiAI enabled."
            ),
            supported_precisions=("float32", "int4"),
        )

    def calibrate(
        self,
        model: LoadedModel,
        samples: tuple[ModelBatch, ...],
        config: QuantizationConfig,
    ) -> CalibrationArtifact | None:
        """Symmetric per-channel weights and dynamic activations need no sample calibration."""

        del model, samples, config
        return None

    def quantize(
        self,
        model: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        """Replace selected linears with opaque prepacked native W4A8 modules."""

        capability = probe_native_int4_kleidiai()
        if not capability.supported:
            raise RuntimeError("; ".join(capability.reasons) or "native INT4 kernel unavailable")
        started = time.perf_counter()
        candidate = copy.deepcopy(cast(nn.Module, model.model))
        replaced = _replace_native_linears(
            candidate,
            policy.precision_map,
            precision="int4",
            factory=NativeInt4KleidiAILinear,
        )
        if not replaced:
            raise ValueError("native KleidiAI INT4 policy selected no eligible linear modules")
        loaded = LoadedModel(
            model=candidate,
            tokenizer=copy.deepcopy(model.tokenizer),
            descriptor=model.descriptor,
            load_time_seconds=time.perf_counter() - started,
            load_time_kind="candidate_reconstruction",
        )
        manifest = build_manifest(
            backend=self.name,
            backend_version=torch.__version__,
            method=config.method,
            source=model.descriptor,
            precision_map=policy.precision_map,
            serialized_size_bytes=model_storage_bytes(candidate),
            calibration=calibration,
            parameters={
                **config.parameters,
                "activation_quantization": "dynamic_per_token_int8_asymmetric",
                "kernel": _INT4_MATMUL_OPERATOR,
                "packing_kernel": _INT4_PACK_OPERATOR,
                "packing_layout": "opaque_aten_kleidiai",
                "weight_granularity": "per_output_channel",
                "weight_bits": 4,
            },
            warnings=(
                "Opaque packed tensors are reloadable only where the recorded PyTorch version "
                "passes the same KleidiAI capability probe.",
            ),
        )
        return finalize_serialized_manifest(
            QuantizedModel(loaded=loaded, manifest=manifest), config
        )

    def export(
        self, model: QuantizedModel, destination: Path, config: QuantizationConfig
    ) -> ExportArtifact:
        """Export opaque packed bytes and metadata in a pickle-free bundle."""

        return export_recipe(model, destination, config)


def create_native_dynamic_int8_quantizer() -> TorchNativeDynamicInt8Quantizer:
    """Registry factory for native dynamic INT8."""

    return TorchNativeDynamicInt8Quantizer()


def create_native_int4_kleidiai_quantizer() -> TorchNativeInt4KleidiAIQuantizer:
    """Registry factory for native KleidiAI INT4."""

    return TorchNativeInt4KleidiAIQuantizer()


__all__ = [
    "NativeDynamicInt8Linear",
    "NativeInt4KleidiAILinear",
    "NativeKernelCapability",
    "TorchNativeDynamicInt8Quantizer",
    "TorchNativeInt4KleidiAIQuantizer",
    "create_native_dynamic_int8_quantizer",
    "create_native_int4_kleidiai_quantizer",
    "probe_native_dynamic_int8",
    "probe_native_int4_kleidiai",
]
