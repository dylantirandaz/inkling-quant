from __future__ import annotations

import subprocess
import sys
import textwrap
from dataclasses import replace

import pytest

from inkling_quant_lab.config import CalibrationConfig, ExperimentConfig, QuantizationConfig
from inkling_quant_lab.exceptions import CapabilityError, QuantizationError
from inkling_quant_lab.models.base import LoadedModel, ModelBatch
from inkling_quant_lab.models.local import LocalFixtureAdapter
from inkling_quant_lab.quantization.fake_optional import FakeFailurePoint, FakeOptionalQuantizer
from inkling_quant_lab.quantization.policies import (
    ResolvedPrecisionPolicy,
    resolve_precision_policy,
)
from inkling_quant_lab.quantization.reference import load_export_recipe
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.unit


def _fixture(
    config_factory,
    *,
    parameters: dict[str, str | int | float | bool] | None = None,
) -> tuple[
    ExperimentConfig,
    QuantizationConfig,
    LocalFixtureAdapter,
    TorchEagerCPURuntime,
    LoadedModel,
    ResolvedPrecisionPolicy,
]:
    config = config_factory(backend="fake_optional_cpu", precision="float32")
    quantization = config.quantization.model_copy(
        update={"method": "none", "parameters": parameters or {}}
    )
    adapter = LocalFixtureAdapter()
    runtime = TorchEagerCPURuntime()
    loaded = adapter.load(config, runtime)
    policy = resolve_precision_policy(adapter.enumerate_modules(loaded), quantization.policy)
    return config, quantization, adapter, runtime, loaded, policy


def _batch(loaded: LoadedModel, *sample_ids: str) -> ModelBatch:
    prompts = tuple(f"fixture prompt {index}" for index, _ in enumerate(sample_ids))
    input_ids, attention_mask = loaded.tokenizer.batch_encode(prompts)
    return ModelBatch(
        sample_ids=sample_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )


def test_default_fixture_reports_honest_cpu_float32_support(config_factory) -> None:
    _, quantization, _, runtime, loaded, _ = _fixture(config_factory)

    report = FakeOptionalQuantizer().check_support(
        loaded.descriptor,
        runtime.probe(config_factory().runtime),
        quantization,
    )

    assert report.available
    assert report.supported
    assert report.supported_precisions == ("float32",)
    assert report.install_extra is None
    assert "Test-only" in report.warnings[0]
    assert "no weights are quantized" in report.warnings[0]


@pytest.mark.parametrize(
    ("parameter", "expected_available", "reason_fragment"),
    [
        ("fake_available", False, "configured unavailable"),
        ("fake_supported", True, "configured unsupported"),
    ],
)
def test_configured_unavailable_and_unsupported_states_are_actionable(
    config_factory,
    parameter: str,
    expected_available: bool,
    reason_fragment: str,
) -> None:
    _, quantization, _, runtime, loaded, policy = _fixture(
        config_factory, parameters={parameter: False}
    )
    quantizer = FakeOptionalQuantizer()

    report = quantizer.check_support(
        loaded.descriptor,
        runtime.probe(config_factory().runtime),
        quantization,
    )

    assert report.available is expected_available
    assert not report.supported
    assert reason_fragment in report.message()
    assert f"{parameter}=true" in report.message()
    with pytest.raises(CapabilityError, match=reason_fragment):
        quantizer.quantize(loaded, policy, None, quantization)


def test_runtime_and_precision_constraints_are_reported(config_factory) -> None:
    config, quantization, adapter, runtime, loaded, _ = _fixture(config_factory)
    capabilities = runtime.probe(config.runtime)
    non_cpu = replace(
        capabilities,
        devices=("cuda",),
        reasons=("selected runtime cannot execute on CPU",),
    )
    int8 = quantization.model_copy(
        update={"policy": quantization.policy.model_copy(update={"default_precision": "int8"})}
    )
    quantizer = FakeOptionalQuantizer()

    runtime_report = quantizer.check_support(loaded.descriptor, non_cpu, quantization)
    precision_report = quantizer.check_support(loaded.descriptor, capabilities, int8)

    assert not runtime_report.supported
    assert "requires an available CPU runtime" in runtime_report.message()
    assert not precision_report.supported
    assert "only supports float32 policies" in precision_report.message()
    int8_policy = resolve_precision_policy(adapter.enumerate_modules(loaded), int8.policy)
    with pytest.raises(CapabilityError, match="only supports float32 policies"):
        quantizer.quantize(loaded, int8_policy, None, int8)


def test_success_preserves_outputs_and_exports_explicit_fake_manifest(
    config_factory, tmp_path
) -> None:
    config, quantization, adapter, _, loaded, policy = _fixture(config_factory)
    batch = _batch(loaded, "sample-a")
    baseline = adapter.generate(loaded, batch, config.evaluation.suites[0].decode)
    quantizer = FakeOptionalQuantizer()

    candidate = quantizer.quantize(loaded, policy, None, quantization)
    result = adapter.generate(candidate.loaded, batch, config.evaluation.suites[0].decode)
    destination = tmp_path / "fake.bundle"
    export = quantizer.export(candidate, destination, quantization)
    recipe = load_export_recipe(destination)

    assert result == baseline
    assert candidate.loaded.model is not loaded.model
    assert candidate.manifest.backend == "fake_optional_cpu"
    assert candidate.manifest.backend_version == "fixture-v1"
    assert candidate.manifest.serialized_size_bytes > 0
    assert candidate.manifest.quantization_parameters["performs_real_quantization"] is False
    assert "no weights are quantized" in candidate.manifest.warnings[0]
    assert recipe["reload"]["backend"] == "fake_optional_cpu"
    assert export.size_bytes == sum(path.stat().st_size for path in destination.iterdir())


def test_optional_calibration_is_deterministic_and_recorded(config_factory) -> None:
    _, quantization, _, _, loaded, policy = _fixture(config_factory)
    calibration_config = CalibrationConfig(
        dataset="local://fixtures/calibration",
        revision="fixture-data-v1",
        split="calibration",
        sample_ids=("cal-a", "cal-b"),
        max_samples=2,
    )
    quantization = quantization.model_copy(
        update={
            "calibration": calibration_config,
            "parameters": {"fake_requires_calibration": True},
        }
    )
    samples = (_batch(loaded, "cal-a", "cal-b"),)
    quantizer = FakeOptionalQuantizer()

    first = quantizer.calibrate(loaded, samples, quantization)
    second = quantizer.calibrate(loaded, samples, quantization)

    assert first is not None
    assert second is not None
    assert first == second
    assert first.sample_ids == ("cal-a", "cal-b")
    assert first.statistics == {"fake_sample_count": 2.0}
    candidate = quantizer.quantize(loaded, policy, first, quantization)
    assert candidate.manifest.calibration_sample_ids == first.sample_ids
    assert candidate.manifest.calibration_dataset == {
        "dataset": calibration_config.dataset,
        "revision": calibration_config.revision,
        "split": calibration_config.split,
    }


def test_required_calibration_rejects_missing_config_samples_and_artifact(
    config_factory,
) -> None:
    _, quantization, _, _, loaded, policy = _fixture(
        config_factory, parameters={"fake_requires_calibration": True}
    )
    quantizer = FakeOptionalQuantizer()

    with pytest.raises(QuantizationError, match="explicit calibration configuration"):
        quantizer.calibrate(loaded, (), quantization)

    calibration = CalibrationConfig(
        dataset="local://fixtures/calibration",
        revision="fixture-data-v1",
    )
    with_config = quantization.model_copy(update={"calibration": calibration})
    with pytest.raises(QuantizationError, match="at least one sample"):
        quantizer.calibrate(loaded, (), with_config)
    with pytest.raises(QuantizationError, match="calibration artifact"):
        quantizer.quantize(loaded, policy, None, with_config)


@pytest.mark.parametrize(
    "failure_point",
    ["check_support", "calibrate", "quantize", "export"],
)
def test_each_operation_has_a_controlled_typed_failure(
    config_factory,
    tmp_path,
    failure_point: FakeFailurePoint,
) -> None:
    config, base, _, runtime, loaded, policy = _fixture(config_factory)
    failing = base.model_copy(update={"parameters": {"fake_fail_at": failure_point}})
    quantizer = FakeOptionalQuantizer()

    with pytest.raises(QuantizationError, match=f"failure during {failure_point}") as raised:
        if failure_point == "check_support":
            quantizer.check_support(loaded.descriptor, runtime.probe(config.runtime), failing)
        elif failure_point == "calibrate":
            quantizer.calibrate(loaded, (), failing)
        elif failure_point == "quantize":
            quantizer.quantize(loaded, policy, None, failing)
        else:
            candidate = quantizer.quantize(loaded, policy, None, base)
            quantizer.export(candidate, tmp_path / "never-created.json", failing)

    assert raised.value.component == "fake_optional_cpu"
    assert raised.value.details == {"injected": True, "operation": failure_point}


@pytest.mark.parametrize(
    "parameters",
    [
        {"fake_available": "yes"},
        {"fake_supported": 1},
        {"fake_fail_at": "load"},
        {"fake_requires_calibration": 0},
    ],
)
def test_invalid_fixture_controls_fail_explicitly(config_factory, parameters) -> None:
    config, quantization, _, runtime, loaded, _ = _fixture(config_factory, parameters=parameters)

    with pytest.raises(QuantizationError, match="Invalid fake optional backend parameter"):
        FakeOptionalQuantizer().check_support(
            loaded.descriptor,
            runtime.probe(config.runtime),
            quantization,
        )


@pytest.mark.parametrize(
    ("constructor_parameters", "error"),
    [
        ({"available": 1}, TypeError),
        ({"supported": "yes"}, TypeError),
        ({"fail_at": "load"}, ValueError),
        ({"requires_calibration": 0}, TypeError),
    ],
)
def test_constructor_rejects_invalid_controls(constructor_parameters, error) -> None:
    with pytest.raises(error):
        FakeOptionalQuantizer(**constructor_parameters)


def test_builtin_fake_backend_registration_is_lazy() -> None:
    script = textwrap.dedent(
        """
        import sys

        from inkling_quant_lab.bootstrap import register_builtins
        from inkling_quant_lab.registry import QUANTIZERS

        module = "inkling_quant_lab.quantization.fake_optional"
        register_builtins()
        descriptor = next(
            item for item in QUANTIZERS.descriptors() if item.name == "fake_optional_cpu"
        )
        assert module not in sys.modules
        assert descriptor.available is True
        assert descriptor.optional_extra is None
        assert "Test-only" in descriptor.description
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_builtin_registry_resolves_configurable_fixture(config_factory) -> None:
    from inkling_quant_lab.bootstrap import register_builtins
    from inkling_quant_lab.registry import QUANTIZERS

    config, quantization, _, runtime, loaded, _ = _fixture(config_factory)
    register_builtins()

    quantizer = QUANTIZERS.create("fake_optional_cpu", available=False)
    report = quantizer.check_support(
        loaded.descriptor,
        runtime.probe(config.runtime),
        quantization,
    )

    assert isinstance(quantizer, FakeOptionalQuantizer)
    assert not report.available
    assert not report.supported
