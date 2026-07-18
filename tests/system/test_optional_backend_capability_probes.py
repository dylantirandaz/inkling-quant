"""Opt-in hardware/dependency probes for production optional quantizers.

These checks intentionally stop before model loading or conversion.  The
validated Hugging Face baseline adapter remains CPU-only, so the repository
does not yet expose an end-to-end experiment that combines that baseline with
the CUDA-owned source-reload lifecycle used by these quantizers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from inkling_quant_lab.config import QuantizationConfig, RuntimeConfig
from inkling_quant_lab.models.base import ModelCapabilities, ModelDescriptor
from inkling_quant_lab.quantization.optional import (
    create_awq_quantizer,
    create_fp8_quantizer,
    create_gptq_quantizer,
)
from inkling_quant_lab.runtimes.torch_cuda import TorchEagerCUDARuntime

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIGS = {
    "awq": _PROJECT_ROOT / "configs/quantization/awq_gptqmodel_capability_probe.yaml",
    "gptq": _PROJECT_ROOT / "configs/quantization/gptq_gptqmodel_capability_probe.yaml",
    "fp8": _PROJECT_ROOT / "configs/quantization/finegrained_fp8_capability_probe.yaml",
}
_FACTORIES = {
    "awq": create_awq_quantizer,
    "gptq": create_gptq_quantizer,
    "fp8": create_fp8_quantizer,
}
_RUN_PROBES = os.environ.get("IQL_RUN_OPTIONAL_BACKEND_PROBES") == "1"
_EXTERNAL_ONLY = pytest.mark.skipif(
    not _RUN_PROBES,
    reason="set IQL_RUN_OPTIONAL_BACKEND_PROBES=1 to probe installed CUDA backends",
)


def _quantization_config(backend: str) -> QuantizationConfig:
    payload: Any = yaml.safe_load(_CONFIGS[backend].read_text(encoding="utf-8"))
    assert isinstance(payload, dict) and set(payload) == {"quantization"}
    return QuantizationConfig.model_validate(payload["quantization"])


def _pinned_mixtral_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        model_id="ggml-org/stories15M_MOE",
        revision="b6dd737497465570b5f5e962dbc9d9454ed1e0eb",
        resolved_class="transformers.models.mixtral.MixtralForCausalLM",
        architecture="MixtralForCausalLM",
        checksum="0" * 64,
        capabilities=ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_audio=False,
            is_moe=True,
            supports_router_logits=True,
            supports_token_level_routes=True,
            supported_dtypes=("float32",),
            supported_device_maps=("single",),
            max_context_length=1024,
            requires_remote_code=False,
        ),
    )


def _probe(backend: str) -> None:
    config = _quantization_config(backend)
    runtime = TorchEagerCUDARuntime().probe(
        RuntimeConfig(
            backend="torch_eager_cuda",
            device="cuda",
            dtype="float32",
            device_map="single",
        )
    )
    report = _FACTORIES[backend]().check_support(_pinned_mixtral_descriptor(), runtime, config)
    assert runtime.available, runtime.reasons
    assert report.available, report.reasons
    assert report.supported, report.reasons


@pytest.mark.gpu
@pytest.mark.backend_awq
@_EXTERNAL_ONLY
def test_awq_dependency_and_hardware_capability_probe() -> None:
    _probe("awq")


@pytest.mark.gpu
@pytest.mark.backend_gptq
@_EXTERNAL_ONLY
def test_gptq_dependency_and_hardware_capability_probe() -> None:
    _probe("gptq")


@pytest.mark.gpu
@pytest.mark.backend_fp8
@_EXTERNAL_ONLY
def test_fp8_dependency_and_hardware_capability_probe() -> None:
    _probe("fp8")
