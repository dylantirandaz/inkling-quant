from __future__ import annotations

import json

import pytest

from inkling_quant_lab.models.base import ModelBatch
from inkling_quant_lab.models.local import LocalFixtureAdapter
from inkling_quant_lab.quantization.int8 import DynamicInt8Linear, TorchDynamicInt8Quantizer
from inkling_quant_lab.quantization.optional import UnavailableOptionalQuantizer
from inkling_quant_lab.quantization.policies import resolve_precision_policy
from inkling_quant_lab.quantization.reference import (
    NoopQuantizer,
    load_export_recipe,
    safe_model_serialized_size_bytes,
)
from inkling_quant_lab.quantization.weight_only import (
    PackedInt4Linear,
    TorchWeightOnlyInt4Quantizer,
)
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.unit


def _loaded_policy(config_factory, *, backend, precision):
    config = config_factory(backend=backend, precision=precision)
    adapter = LocalFixtureAdapter()
    runtime = TorchEagerCPURuntime()
    loaded = adapter.load(config, runtime)
    policy = resolve_precision_policy(adapter.enumerate_modules(loaded), config.quantization.policy)
    return config, adapter, runtime, loaded, policy


def test_noop_preserves_outputs_exactly_and_exports_recipe(config_factory, tmp_path):
    config, adapter, _, loaded, policy = _loaded_policy(
        config_factory, backend="noop", precision="float32"
    )
    tokenizer = loaded.tokenizer
    ids, mask = tokenizer.batch_encode(("alpha beta",))
    batch = ModelBatch(sample_ids=("x",), input_ids=ids, attention_mask=mask)
    baseline = adapter.generate(loaded, batch, config.evaluation.suites[0].decode)

    quantizer = NoopQuantizer()
    candidate = quantizer.quantize(loaded, policy, None, config.quantization)
    result = adapter.generate(candidate.loaded, batch, config.evaluation.suites[0].decode)
    destination = tmp_path / "noop.bundle"
    export = quantizer.export(candidate, destination, config.quantization)

    assert result == baseline
    assert candidate.loaded.model is not loaded.model
    assert candidate.manifest.backend == "noop"
    assert load_export_recipe(destination)["reload"]["backend"] == "noop"
    assert export.size_bytes == sum(path.stat().st_size for path in destination.iterdir())


def test_dynamic_int8_changes_eligible_modules_and_reduces_storage(config_factory):
    config, adapter, runtime, loaded, policy = _loaded_policy(
        config_factory, backend="torch_dynamic_int8", precision="int8"
    )
    baseline_size = safe_model_serialized_size_bytes(loaded)
    quantizer = TorchDynamicInt8Quantizer()
    support = quantizer.check_support(
        loaded.descriptor, runtime.probe(config.runtime), config.quantization
    )
    candidate = quantizer.quantize(loaded, policy, None, config.quantization)

    assert support.supported
    assert any(isinstance(module, DynamicInt8Linear) for module in candidate.loaded.model.modules())
    assert candidate.manifest.serialized_size_bytes < baseline_size
    assert not isinstance(candidate.loaded.model.lm_head, DynamicInt8Linear)
    assert all(
        not isinstance(layer.router, DynamicInt8Linear)
        for layer in candidate.loaded.model.moe_layers
    )

    tokenizer = candidate.loaded.tokenizer
    ids, mask = tokenizer.batch_encode(("alpha beta",))
    output = adapter.generate(
        candidate.loaded,
        ModelBatch(sample_ids=("x",), input_ids=ids, attention_mask=mask),
        config.evaluation.suites[0].decode,
    )
    assert output.token_ids


def test_packed_int4_is_working_reference_backend(config_factory):
    config, adapter, runtime, loaded, policy = _loaded_policy(
        config_factory, backend="torch_weight_only_int4", precision="int4"
    )
    baseline_size = safe_model_serialized_size_bytes(loaded)
    quantizer = TorchWeightOnlyInt4Quantizer()
    assert quantizer.check_support(
        loaded.descriptor, runtime.probe(config.runtime), config.quantization
    ).supported
    candidate = quantizer.quantize(loaded, policy, None, config.quantization)
    assert any(isinstance(module, PackedInt4Linear) for module in candidate.loaded.model.modules())
    assert candidate.manifest.serialized_size_bytes < baseline_size
    tokenizer = candidate.loaded.tokenizer
    ids, mask = tokenizer.batch_encode(("alpha beta",))
    assert adapter.generate(
        candidate.loaded,
        ModelBatch(sample_ids=("x",), input_ids=ids, attention_mask=mask),
        config.evaluation.suites[0].decode,
    ).token_ids


def test_missing_backend_capability_is_actionable(config_factory):
    config, _, runtime, loaded, _ = _loaded_policy(
        config_factory, backend="noop", precision="float32"
    )
    backend = UnavailableOptionalQuantizer("awq", "awq", "AWQ-style")
    support = backend.check_support(
        loaded.descriptor, runtime.probe(config.runtime), config.quantization
    )
    assert not support.available
    assert support.install_extra == "awq"
    assert "No validated implementation is bundled" in support.message()
    assert "uv sync --extra" not in support.message()


def test_quantization_manifest_json_contains_no_pickle(config_factory):
    config, _, _, loaded, policy = _loaded_policy(
        config_factory, backend="noop", precision="float32"
    )
    candidate = NoopQuantizer().quantize(loaded, policy, None, config.quantization)
    payload = json.dumps(candidate.manifest.model_dump(mode="json"))
    assert "pickle" not in payload
    assert candidate.manifest.source_model_checksum == loaded.descriptor.checksum
