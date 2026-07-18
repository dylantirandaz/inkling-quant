"""CPU reference dynamic INT8 quantizer implemented with PyTorch tensors."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import cast

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
from inkling_quant_lab.quantization.reference import (
    build_manifest,
    export_recipe,
    finalize_serialized_manifest,
    model_storage_bytes,
)
from inkling_quant_lab.runtimes.base import RuntimeCapabilities


class DynamicInt8Linear(nn.Module):
    """Storage-quantized linear with dynamic per-call activation quantization.

    This deterministic CPU reference dequantizes before the PyTorch matmul. It is
    intended for correctness and artifact tests, not optimized-kernel claims.
    """

    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        weight = source.weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        quantized = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
        self.weight_int8: Tensor
        self.weight_scale: Tensor
        self.register_buffer("weight_int8", quantized)
        self.register_buffer("weight_scale", scale)
        self.bias: nn.Parameter | None
        source_bias = getattr(source, "bias", None)
        if source_bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(source_bias.detach().float().clone(), requires_grad=False)
        self.in_features = source.in_features
        self.out_features = source.out_features

    def forward(self, inputs: Tensor) -> Tensor:
        """Dynamically quantize activations, dequantize, and apply the stored weight."""

        activation_scale = inputs.detach().float().abs().amax().clamp_min(1e-8) / 127.0
        activation_int8 = torch.round(inputs.float() / activation_scale).clamp(-127, 127)
        dequantized_inputs = activation_int8 * activation_scale
        weight = self.weight_int8.float() * self.weight_scale
        return F.linear(dequantized_inputs, weight, self.bias)


def _replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    if child_name.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def _replace_int8_linears(root: nn.Module, precision_map: dict[str, str]) -> tuple[str, ...]:
    replaced: list[str] = []
    named = dict(root.named_modules())
    for name in sorted(precision_map):
        module = named.get(name)
        if precision_map[name] == "int8" and isinstance(module, nn.Linear):
            _replace_module(root, name, DynamicInt8Linear(module))
            replaced.append(name)
    return tuple(replaced)


class TorchDynamicInt8Quantizer:
    """Deterministic CPU reference for eligible ``torch.nn.Linear`` modules."""

    name = "torch_dynamic_int8"

    def check_support(
        self,
        model: ModelDescriptor,
        runtime: RuntimeCapabilities,
        config: QuantizationConfig,
    ) -> SupportReport:
        """Require an available CPU eager runtime and INT8 policy support."""

        del model, config
        cpu = "cpu" in runtime.devices
        supported = runtime.available and cpu
        reasons = () if supported else (*runtime.reasons, "dynamic INT8 requires a CPU runtime")
        return SupportReport(
            available=True,
            supported=supported,
            component=self.name,
            reasons=reasons,
            warnings=("reference implementation dequantizes before matrix multiplication",),
            supported_precisions=("float32", "int8"),
        )

    def calibrate(
        self,
        model: LoadedModel,
        samples: tuple[ModelBatch, ...],
        config: QuantizationConfig,
    ) -> CalibrationArtifact | None:
        """Dynamic activation quantization requires no offline calibration."""

        del model, samples, config
        return None

    def quantize(
        self,
        model: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        """Replace policy-selected linear layers with dynamic INT8 wrappers."""

        started = time.perf_counter()
        candidate = copy.deepcopy(cast(nn.Module, model.model))
        replaced = _replace_int8_linears(candidate, policy.precision_map)
        if not replaced:
            raise ValueError("dynamic INT8 policy selected no eligible linear modules")
        loaded = LoadedModel(
            model=candidate,
            tokenizer=copy.deepcopy(model.tokenizer),
            descriptor=model.descriptor,
            load_time_seconds=time.perf_counter() - started,
            load_time_kind="candidate_reconstruction",
        )
        if loaded.load_time_seconds <= 0.0:
            raise RuntimeError("dynamic INT8 candidate reconstruction duration was not measurable")
        size = model_storage_bytes(candidate)
        manifest = build_manifest(
            backend=self.name,
            backend_version=torch.__version__,
            method=config.method,
            source=model.descriptor,
            precision_map=policy.precision_map,
            serialized_size_bytes=size,
            calibration=calibration,
            parameters={**config.parameters, "dynamic_activation_quantization": True},
            warnings=(
                "CPU reference path dequantizes before matmul; latency is measured, not assumed.",
            ),
        )
        return finalize_serialized_manifest(
            QuantizedModel(loaded=loaded, manifest=manifest), config
        )

    def export(
        self, model: QuantizedModel, destination: Path, config: QuantizationConfig
    ) -> ExportArtifact:
        """Export a non-pickle reconstruction recipe."""

        return export_recipe(model, destination, config)


def create_quantizer() -> TorchDynamicInt8Quantizer:
    """Registry factory for dynamic INT8."""

    return TorchDynamicInt8Quantizer()
