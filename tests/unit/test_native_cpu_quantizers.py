from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

import inkling_quant_lab.quantization.native_cpu as native_cpu
from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.models.base import LoadedModel
from inkling_quant_lab.models.local import LocalFixtureAdapter
from inkling_quant_lab.quantization.native_cpu import (
    NativeDynamicInt8Linear,
    NativeInt4KleidiAILinear,
    NativeKernelCapability,
    TorchNativeDynamicInt8Quantizer,
    TorchNativeInt4KleidiAIQuantizer,
    probe_native_dynamic_int8,
    probe_native_int4_kleidiai,
)
from inkling_quant_lab.quantization.policies import (
    ResolvedPrecisionPolicy,
    resolve_precision_policy,
)
from inkling_quant_lab.quantization.reference import safe_model_serialized_size_bytes
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.unit


def _loaded_policy(
    config_factory: Callable[..., ExperimentConfig], *, backend: str, precision: str
) -> tuple[
    ExperimentConfig,
    LocalFixtureAdapter,
    TorchEagerCPURuntime,
    LoadedModel,
    ResolvedPrecisionPolicy,
]:
    config = config_factory(backend=backend, precision=precision)
    adapter = LocalFixtureAdapter()
    runtime = TorchEagerCPURuntime()
    loaded = adapter.load(config, runtime)
    policy = resolve_precision_policy(adapter.enumerate_modules(loaded), config.quantization.policy)
    return config, adapter, runtime, loaded, policy


def test_native_dynamic_int8_executes_quantized_kernel_without_dequantization() -> None:
    capability = probe_native_dynamic_int8()
    if not capability.supported or capability.implementation is None:
        pytest.skip("this PyTorch build has no native dynamic INT8 CPU kernel")
    torch.manual_seed(7)
    source = nn.Linear(128, 96).eval()
    native = NativeDynamicInt8Linear(source, engine=capability.implementation).eval()
    inputs = torch.randn(4, 128)

    with profile(activities=[ProfilerActivity.CPU]) as recorded:
        actual = native(inputs)

    operators = {event.key for event in recorded.key_averages()}
    assert "quantized::linear_dynamic" in operators
    assert "aten::dequantize" not in operators
    assert torch.allclose(actual, source(inputs), atol=0.08, rtol=0.03)
    assert set(native.state_dict()) == {"bias", "weight_int8", "weight_scales"}


def test_native_int4_executes_opaque_kernel_without_unpacking() -> None:
    capability = probe_native_int4_kleidiai()
    if not capability.supported:
        pytest.skip("this PyTorch build has no native KleidiAI INT4 CPU kernel")
    torch.manual_seed(11)
    source = nn.Linear(127, 65, bias=False).eval()
    native = NativeInt4KleidiAILinear(source).eval()
    inputs = torch.randn(4, 127)

    with profile(activities=[ProfilerActivity.CPU]) as recorded:
        actual = native(inputs)

    operators = {event.key for event in recorded.key_averages()}
    assert "aten::_dyn_quant_matmul_4bit" in operators
    assert "aten::bitwise_and" not in operators
    assert "aten::bitwise_right_shift" not in operators
    assert torch.allclose(actual, source(inputs), atol=0.25, rtol=0.12)
    assert set(native.state_dict()) == {"packed_weight"}
    assert native.packed_weight.dtype == torch.uint8


@pytest.mark.parametrize(
    ("backend", "precision", "factory", "wrapper"),
    [
        (
            "torch_native_dynamic_int8",
            "int8",
            TorchNativeDynamicInt8Quantizer,
            NativeDynamicInt8Linear,
        ),
        (
            "torch_native_int4_kleidiai",
            "int4",
            TorchNativeInt4KleidiAIQuantizer,
            NativeInt4KleidiAILinear,
        ),
    ],
)
def test_native_quantizers_transform_fixture_and_record_kernel(
    config_factory: Callable[..., ExperimentConfig],
    backend: str,
    precision: str,
    factory: type[TorchNativeDynamicInt8Quantizer] | type[TorchNativeInt4KleidiAIQuantizer],
    wrapper: type[nn.Module],
) -> None:
    config, _, runtime, loaded, policy = _loaded_policy(
        config_factory, backend=backend, precision=precision
    )
    quantizer = factory()
    support = quantizer.check_support(
        loaded.descriptor, runtime.probe(config.runtime), config.quantization
    )
    if not support.supported:
        pytest.skip(support.message())
    baseline_size = safe_model_serialized_size_bytes(loaded)

    candidate = quantizer.quantize(loaded, policy, None, config.quantization)

    assert any(isinstance(module, wrapper) for module in candidate.loaded.model.modules())
    assert candidate.manifest.serialized_size_bytes < baseline_size
    assert candidate.manifest.quantization_parameters["kernel"] in {
        "quantized::linear_dynamic",
        "aten::_dyn_quant_matmul_4bit",
    }
    assert not candidate.manifest.warnings[0].lower().startswith("reference")


def test_native_support_reports_probe_failure_and_bad_engine_parameter(
    config_factory: Callable[..., ExperimentConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, runtime, loaded, _ = _loaded_policy(
        config_factory, backend="torch_native_dynamic_int8", precision="int8"
    )
    unavailable = NativeKernelCapability(
        supported=False, implementation=None, reasons=("operator was not compiled",)
    )
    monkeypatch.setattr(native_cpu, "probe_native_dynamic_int8", lambda engine=None: unavailable)
    quantizer = TorchNativeDynamicInt8Quantizer()

    report = quantizer.check_support(
        loaded.descriptor, runtime.probe(config.runtime), config.quantization
    )

    assert report.available
    assert not report.supported
    assert "operator was not compiled" in report.message()
    assert report.remediation is not None and "portable fallback" in report.remediation

    invalid = config.quantization.model_copy(update={"parameters": {"quantized_engine": 7}})
    invalid_report = quantizer.check_support(
        loaded.descriptor, runtime.probe(config.runtime), invalid
    )
    assert not invalid_report.supported
    assert "must be a string" in invalid_report.message()


@pytest.mark.parametrize(
    ("backend", "precision"),
    [
        ("torch_native_dynamic_int8", "int8"),
        ("torch_native_int4_kleidiai", "int4"),
    ],
)
def test_native_config_requires_cpu_float32(
    config_factory: Callable[..., ExperimentConfig], backend: str, precision: str
) -> None:
    config = config_factory(backend=backend, precision=precision)
    payload: dict[str, Any] = config.model_dump(mode="python")
    payload["runtime"]["device"] = "mps"
    with pytest.raises(ValueError, match=r"requires runtime\.device=cpu"):
        ExperimentConfig.model_validate(payload)

    payload = config.model_dump(mode="python")
    payload["model"]["dtype"] = "bfloat16"
    payload["runtime"]["dtype"] = "bfloat16"
    with pytest.raises(ValueError, match="requires explicit float32"):
        ExperimentConfig.model_validate(payload)


def test_native_linear_rejects_wrong_activation_contract() -> None:
    int8_capability = probe_native_dynamic_int8()
    if int8_capability.supported and int8_capability.implementation is not None:
        int8 = NativeDynamicInt8Linear(nn.Linear(16, 8), engine=int8_capability.implementation)
        with pytest.raises(RuntimeError, match="float32 CPU"):
            int8(torch.ones((1, 16), dtype=torch.bfloat16))
        with pytest.raises(RuntimeError, match="expected 16"):
            int8(torch.ones((1, 15)))

    if probe_native_int4_kleidiai().supported:
        int4 = NativeInt4KleidiAILinear(nn.Linear(16, 8))
        with pytest.raises(RuntimeError, match="float32 CPU"):
            int4(torch.ones((1, 16), dtype=torch.bfloat16))
        with pytest.raises(RuntimeError, match="expected 16"):
            int4(torch.ones((1, 15)))


def test_native_quantizer_factories_return_expected_types() -> None:
    assert isinstance(
        native_cpu.create_native_dynamic_int8_quantizer(), TorchNativeDynamicInt8Quantizer
    )
    assert isinstance(
        native_cpu.create_native_int4_kleidiai_quantizer(), TorchNativeInt4KleidiAIQuantizer
    )
