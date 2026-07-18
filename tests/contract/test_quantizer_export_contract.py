from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError
from torch import nn

from inkling_quant_lab.config import ExperimentConfig, ExportConfig
from inkling_quant_lab.models.base import ModelBatch
from inkling_quant_lab.models.local import LocalFixtureAdapter
from inkling_quant_lab.quantization.base import QuantizedModel, Quantizer
from inkling_quant_lab.quantization.int8 import DynamicInt8Linear, TorchDynamicInt8Quantizer
from inkling_quant_lab.quantization.mixed_precision import TorchReferenceMixedQuantizer
from inkling_quant_lab.quantization.native_cpu import (
    NativeDynamicInt8Linear,
    NativeInt4KleidiAILinear,
    TorchNativeDynamicInt8Quantizer,
    TorchNativeInt4KleidiAIQuantizer,
)
from inkling_quant_lab.quantization.policies import resolve_precision_policy
from inkling_quant_lab.quantization.reference import (
    NoopQuantizer,
    load_export_recipe,
    reload_exported_model,
)
from inkling_quant_lab.quantization.weight_only import (
    PackedInt4Linear,
    TorchWeightOnlyInt4Quantizer,
)
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.contract


@pytest.mark.parametrize("export_format", ["recipe_json", "safetensors"])
@pytest.mark.parametrize(
    ("backend", "precision", "factory", "wrappers"),
    [
        ("noop", "float32", NoopQuantizer, ()),
        ("torch_dynamic_int8", "int8", TorchDynamicInt8Quantizer, (DynamicInt8Linear,)),
        (
            "torch_weight_only_int4",
            "int4",
            TorchWeightOnlyInt4Quantizer,
            (PackedInt4Linear,),
        ),
        (
            "torch_reference_mixed",
            "int8",
            TorchReferenceMixedQuantizer,
            (DynamicInt8Linear, PackedInt4Linear),
        ),
        (
            "torch_native_dynamic_int8",
            "int8",
            TorchNativeDynamicInt8Quantizer,
            (NativeDynamicInt8Linear,),
        ),
        (
            "torch_native_int4_kleidiai",
            "int4",
            TorchNativeInt4KleidiAIQuantizer,
            (NativeInt4KleidiAILinear,),
        ),
    ],
)
def test_cpu_quantizer_export_is_safe_complete_and_reloadable(
    config_factory: Callable[..., ExperimentConfig],
    tmp_path: Path,
    export_format: str,
    backend: str,
    precision: str,
    factory: Callable[[], Quantizer],
    wrappers: tuple[type[nn.Module], ...],
) -> None:
    """TC-QUANT-001/002 and the shared export/reload backend contract."""

    config = config_factory(backend=backend, precision=precision)
    quantization = config.quantization.model_copy(
        update={"export": config.quantization.export.model_copy(update={"format": export_format})}
    )
    if backend == "torch_reference_mixed":
        quantization = quantization.model_copy(
            update={
                "policy": quantization.policy.model_copy(
                    update={
                        "explicit_overrides": {
                            "moe_layers.0.experts.0.0": "int4",
                        }
                    }
                )
            }
        )
    adapter = LocalFixtureAdapter()
    runtime = TorchEagerCPURuntime()
    baseline = adapter.load(config, runtime)
    policy = resolve_precision_policy(adapter.enumerate_modules(baseline), quantization.policy)
    quantizer = factory()

    support = quantizer.check_support(
        baseline.descriptor,
        runtime.probe(config.runtime),
        quantization,
    )
    if backend.startswith("torch_native_") and not support.supported:
        pytest.skip(support.message())
    assert support.available and support.supported
    assert support.component == backend

    candidate = quantizer.quantize(baseline, policy, None, quantization)
    artifact = quantizer.export(candidate, tmp_path / f"{backend}-{export_format}", quantization)
    recipe = load_export_recipe(Path(artifact.path))
    reloaded = reload_exported_model(Path(artifact.path), baseline)

    input_ids, attention_mask = baseline.tokenizer.batch_encode(("alpha beta",))
    batch = ModelBatch(
        sample_ids=("sample",),
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    expected = adapter.generate(candidate.loaded, batch, config.evaluation.suites[0].decode)
    actual = adapter.generate(reloaded.loaded, batch, config.evaluation.suites[0].decode)

    assert actual == expected
    assert recipe["reload"]["backend"] == backend
    assert recipe["reload"]["format"] == export_format
    assert recipe["quantization"]["serialized_size_bytes"] == artifact.size_bytes
    assert candidate.manifest.serialized_size_bytes == artifact.size_bytes
    assert candidate.manifest.backend == backend
    assert candidate.manifest.source_model_checksum == baseline.descriptor.checksum
    assert candidate.manifest.module_precision_map == policy.precision_map
    assert set(candidate.manifest.excluded_modules) == {
        name
        for name, assigned_precision in policy.precision_map.items()
        if assigned_precision not in {"int8", "int4"}
    }
    assert artifact.size_bytes == sum(path.stat().st_size for path in Path(artifact.path).iterdir())
    assert all(
        path.suffix != ".pt" and "pickle" not in path.name for path in Path(artifact.path).iterdir()
    )
    assert candidate.loaded.load_time_seconds > 0.0
    assert candidate.loaded.load_time_kind == "candidate_reconstruction"
    assert reloaded.loaded.load_time_seconds > 0.0
    assert reloaded.loaded.load_time_kind == "candidate_export_reload"
    for wrapper in wrappers:
        assert any(isinstance(module, wrapper) for module in reloaded.loaded.model.modules())
    assert isinstance(reloaded, QuantizedModel)


def test_export_contract_rejects_undeclared_formats() -> None:
    with pytest.raises(ValidationError, match="format"):
        ExportConfig.model_validate({"format": "pickle", "destination": "candidate"})


def test_export_reload_rejects_mutated_tensor_payload(
    config_factory: Callable[..., ExperimentConfig], tmp_path: Path
) -> None:
    config = config_factory(backend="noop", precision="float32")
    adapter = LocalFixtureAdapter()
    baseline = adapter.load(config, TorchEagerCPURuntime())
    policy = resolve_precision_policy(
        adapter.enumerate_modules(baseline), config.quantization.policy
    )
    quantizer = NoopQuantizer()
    candidate = quantizer.quantize(baseline, policy, None, config.quantization)
    artifact = quantizer.export(candidate, tmp_path / "mutated", config.quantization)
    tensor_path = Path(artifact.path) / "weights.safetensors"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"corruption")

    with pytest.raises(ValueError, match="checksum"):
        reload_exported_model(Path(artifact.path), baseline)
