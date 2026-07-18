"""CPU reference mixed INT8/INT4 execution for resolved expert-aware policies."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import cast

import torch
from torch import nn

from inkling_quant_lab.config import QuantizationConfig
from inkling_quant_lab.models.base import LoadedModel, ModelBatch, ModelDescriptor
from inkling_quant_lab.quantization.base import (
    CalibrationArtifact,
    ExportArtifact,
    QuantizedModel,
    ResolvedPolicyLike,
    SupportReport,
)
from inkling_quant_lab.quantization.int8 import DynamicInt8Linear, _replace_module
from inkling_quant_lab.quantization.reference import (
    build_manifest,
    export_recipe,
    finalize_serialized_manifest,
    model_storage_bytes,
)
from inkling_quant_lab.quantization.weight_only import PackedInt4Linear
from inkling_quant_lab.runtimes.base import RuntimeCapabilities


class TorchReferenceMixedQuantizer:
    """Apply INT8 and packed INT4 wrappers according to one resolved policy."""

    name = "torch_reference_mixed"

    def check_support(
        self,
        model: ModelDescriptor,
        runtime: RuntimeCapabilities,
        config: QuantizationConfig,
    ) -> SupportReport:
        """Support CPU execution with explicit reference-kernel warnings."""

        del model, config
        supported = runtime.available and "cpu" in runtime.devices
        return SupportReport(
            available=True,
            supported=supported,
            component=self.name,
            reasons=() if supported else (*runtime.reasons, "mixed reference backend requires CPU"),
            warnings=("reference INT8/INT4 modules prioritize reproducibility over latency",),
            supported_precisions=("float32", "int8", "int4"),
        )

    def calibrate(
        self,
        model: LoadedModel,
        samples: tuple[ModelBatch, ...],
        config: QuantizationConfig,
    ) -> CalibrationArtifact | None:
        """Return no backend calibration; policy statistics are supplied separately."""

        del model, samples, config
        return None

    def quantize(
        self,
        model: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        """Replace every selected linear with its requested reference precision."""

        started = time.perf_counter()
        candidate = copy.deepcopy(cast(nn.Module, model.model))
        named = dict(candidate.named_modules())
        replaced: list[str] = []
        for name in sorted(policy.precision_map):
            module = named.get(name)
            if not isinstance(module, nn.Linear):
                continue
            precision = policy.precision_map[name]
            if precision == "int8":
                _replace_module(candidate, name, DynamicInt8Linear(module))
                replaced.append(name)
            elif precision == "int4":
                _replace_module(candidate, name, PackedInt4Linear(module))
                replaced.append(name)
        if not replaced:
            raise ValueError("mixed precision policy selected no eligible linear modules")
        loaded = LoadedModel(
            model=candidate,
            tokenizer=copy.deepcopy(model.tokenizer),
            descriptor=model.descriptor,
            load_time_seconds=time.perf_counter() - started,
            load_time_kind="candidate_reconstruction",
        )
        if loaded.load_time_seconds <= 0.0:
            raise RuntimeError("mixed candidate reconstruction duration was not measurable")
        manifest = build_manifest(
            backend=self.name,
            backend_version=torch.__version__,
            method=config.method,
            source=model.descriptor,
            precision_map=policy.precision_map,
            serialized_size_bytes=model_storage_bytes(candidate),
            calibration=calibration,
            parameters={**config.parameters, "supported_weight_bits": "4,8"},
            warnings=("Reference kernels dequantize values during matrix multiplication.",),
        )
        return finalize_serialized_manifest(
            QuantizedModel(loaded=loaded, manifest=manifest), config
        )

    def export(
        self, model: QuantizedModel, destination: Path, config: QuantizationConfig
    ) -> ExportArtifact:
        """Export a safe reconstruction recipe."""

        return export_recipe(model, destination, config)


def create_quantizer() -> TorchReferenceMixedQuantizer:
    """Registry factory for reference mixed precision."""

    return TorchReferenceMixedQuantizer()


__all__ = [
    "DynamicInt8Linear",
    "PackedInt4Linear",
    "TorchReferenceMixedQuantizer",
    "create_quantizer",
]
