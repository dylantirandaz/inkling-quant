"""Portable packed INT4 weight-only reference quantizer for tiny CPU models."""

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
from inkling_quant_lab.quantization.int8 import _replace_module
from inkling_quant_lab.quantization.reference import (
    build_manifest,
    export_recipe,
    finalize_serialized_manifest,
    model_storage_bytes,
)
from inkling_quant_lab.runtimes.base import RuntimeCapabilities


class PackedInt4Linear(nn.Module):
    """Linear layer storing two signed four-bit weights per byte."""

    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        weight = source.weight.detach().float()
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 7.0
        signed = torch.round(weight / scale).clamp(-8, 7).to(torch.int16)
        unsigned = (signed + 8).to(torch.uint8)
        if unsigned.shape[1] % 2:
            unsigned = F.pad(unsigned, (0, 1), value=8)
        packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)
        self.weight_int4_packed: Tensor
        self.weight_scale: Tensor
        self.register_buffer("weight_int4_packed", packed)
        self.register_buffer("weight_scale", scale)
        self.bias: nn.Parameter | None
        source_bias = getattr(source, "bias", None)
        if source_bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(source_bias.detach().float().clone(), requires_grad=False)
        self.in_features = source.in_features
        self.out_features = source.out_features

    def _dequantized_weight(self) -> Tensor:
        low = (self.weight_int4_packed & 0x0F).to(torch.int16) - 8
        high = ((self.weight_int4_packed >> 4) & 0x0F).to(torch.int16) - 8
        interleaved = torch.stack((low, high), dim=-1).reshape(self.out_features, -1)
        return interleaved[:, : self.in_features].float() * self.weight_scale

    def forward(self, inputs: Tensor) -> Tensor:
        """Unpack/dequantize weights and evaluate through PyTorch."""

        return F.linear(inputs.float(), self._dequantized_weight(), self.bias)


class TorchWeightOnlyInt4Quantizer:
    """Working reference 4-bit backend for local linear layers."""

    name = "torch_weight_only_int4"

    def check_support(
        self,
        model: ModelDescriptor,
        runtime: RuntimeCapabilities,
        config: QuantizationConfig,
    ) -> SupportReport:
        """Require a local available CPU runtime."""

        del model, config
        supported = runtime.available and "cpu" in runtime.devices
        return SupportReport(
            available=True,
            supported=supported,
            component=self.name,
            reasons=() if supported else (*runtime.reasons, "packed INT4 reference requires CPU"),
            warnings=("portable reference kernel unpacks weights for each forward pass",),
            supported_precisions=("float32", "int4"),
        )

    def calibrate(
        self,
        model: LoadedModel,
        samples: tuple[ModelBatch, ...],
        config: QuantizationConfig,
    ) -> CalibrationArtifact | None:
        """Per-row symmetric weight-only quantization requires no samples."""

        del model, samples, config
        return None

    def quantize(
        self,
        model: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        """Replace policy-selected linear modules with packed INT4 layers."""

        started = time.perf_counter()
        candidate = copy.deepcopy(cast(nn.Module, model.model))
        named = dict(candidate.named_modules())
        replaced: list[str] = []
        for name in sorted(policy.precision_map):
            module = named.get(name)
            if policy.precision_map[name] == "int4" and isinstance(module, nn.Linear):
                _replace_module(candidate, name, PackedInt4Linear(module))
                replaced.append(name)
        if not replaced:
            raise ValueError("INT4 policy selected no eligible linear modules")
        loaded = LoadedModel(
            model=candidate,
            tokenizer=copy.deepcopy(model.tokenizer),
            descriptor=model.descriptor,
            load_time_seconds=time.perf_counter() - started,
            load_time_kind="candidate_reconstruction",
        )
        if loaded.load_time_seconds <= 0.0:
            raise RuntimeError("INT4 candidate reconstruction duration was not measurable")
        manifest = build_manifest(
            backend=self.name,
            backend_version=torch.__version__,
            method=config.method,
            source=model.descriptor,
            precision_map=policy.precision_map,
            serialized_size_bytes=model_storage_bytes(candidate),
            calibration=calibration,
            parameters={**config.parameters, "weight_bits": 4, "grouping": "per_output_row"},
            warnings=("Reference path unpacks INT4 weights at inference time.",),
        )
        return finalize_serialized_manifest(
            QuantizedModel(loaded=loaded, manifest=manifest), config
        )

    def export(
        self, model: QuantizedModel, destination: Path, config: QuantizationConfig
    ) -> ExportArtifact:
        """Export a non-pickle reconstruction recipe."""

        return export_recipe(model, destination, config)


def create_quantizer() -> TorchWeightOnlyInt4Quantizer:
    """Registry factory for packed INT4."""

    return TorchWeightOnlyInt4Quantizer()
