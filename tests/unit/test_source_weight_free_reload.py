from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

import inkling_quant_lab.quantization.reference as reference
from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.models.base import (
    EmptyExportModelShell,
    ExportedModelIdentity,
    ModelCapabilities,
    ModelDescriptor,
)
from inkling_quant_lab.quantization.base import QuantizationManifest, QuantizedModel
from inkling_quant_lab.quantization.native_cpu import (
    NativeDynamicInt8Linear,
    probe_native_dynamic_int8,
)
from inkling_quant_lab.quantization.reference import (
    HF_CAUSAL_LM_SOURCE_WEIGHT_FREE_RELOAD_ADAPTER,
    TRUSTED_SOURCE_MODEL_RELOAD_ADAPTER,
    export_recipe,
    load_export_recipe,
    reload_exported_model_source_weight_free,
)
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.unit

_MODEL_ID = "example/source-weight-free-mixtral"
_REVISION = "1" * 40
_RESOLVED_CLASS = "transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM"
_ARCHITECTURE = "MixtralForCausalLM"
_SOURCE_CHECKSUM = "2" * 64
_SOURCE_METADATA_HASHES = tuple(
    (name, hashlib.sha256(name.encode("utf-8")).hexdigest())
    for name in (
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
)


class _TinyProjection(nn.Module):
    def __init__(self, *, device: str | torch.device | None = None) -> None:
        super().__init__()
        self.projection = nn.Linear(8, 6, bias=False, device=device)

    def forward(self, value: Tensor) -> Tensor:
        return self.projection(value)


def _capabilities() -> ModelCapabilities:
    return ModelCapabilities(
        supports_text=True,
        supports_images=False,
        supports_audio=False,
        is_moe=True,
        supports_router_logits=True,
        supports_token_level_routes=True,
        supported_dtypes=("float32",),
        supported_device_maps=("single",),
        max_context_length=32,
        requires_remote_code=False,
    )


def _descriptor(*, resolved_class: str = _RESOLVED_CLASS) -> ModelDescriptor:
    return ModelDescriptor(
        model_id=_MODEL_ID,
        revision=_REVISION,
        resolved_class=resolved_class,
        architecture=_ARCHITECTURE,
        checksum=_SOURCE_CHECKSUM,
        capabilities=_capabilities(),
    )


class _FakeEmptyHFAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def load_empty_export_shell(
        self,
        config: ExperimentConfig,
        runtime: Any,
        expected: ExportedModelIdentity,
    ) -> EmptyExportModelShell:
        del runtime
        self.calls += 1
        assert config.model.model_id == expected.model_id
        with torch.device("meta"):
            model = _TinyProjection()
        return EmptyExportModelShell(
            model=model,
            tokenizer=object(),
            descriptor=ModelDescriptor(
                model_id=expected.model_id,
                revision=expected.revision,
                resolved_class=expected.resolved_class,
                architecture=expected.architecture,
                checksum=expected.source_checksum,
                capabilities=_capabilities(),
            ),
            source_metadata_file_sha256=_SOURCE_METADATA_HASHES,
        )


def _source_weight_free_config(
    config_factory: Callable[..., ExperimentConfig],
    *,
    backend: str,
    engine: str | None = None,
) -> ExperimentConfig:
    base = config_factory(backend="noop", precision="float32", routing_mode="off")
    model = base.model.model_copy(
        update={
            "model_id": _MODEL_ID,
            "revision": _REVISION,
            "adapter": "hf_causal_lm",
            "checkpoint_format": "safetensors",
            "local_files_only": True,
        }
    )
    parameters: dict[str, str] = {}
    method = "none"
    if backend == "torch_native_dynamic_int8":
        assert engine is not None
        parameters = {"quantized_engine": engine}
        method = "native_dynamic_w8a8"
    quantization = base.quantization.model_copy(
        update={
            "backend": backend,
            "method": method,
            "parameters": parameters,
            "export": base.quantization.export.model_copy(update={"format": "safetensors"}),
        }
    )
    return base.model_copy(update={"model": model, "quantization": quantization})


def _export_candidate(
    destination: Path,
    config: ExperimentConfig,
    *,
    backend: str,
    engine: str | None = None,
    resolved_class: str = _RESOLVED_CLASS,
) -> tuple[Path, nn.Module]:
    torch.manual_seed(101)
    candidate = _TinyProjection().eval()
    precision_map = {"projection": "float32"}
    parameters: dict[str, str | int | float | bool] = {}
    if backend == "torch_native_dynamic_int8":
        assert engine is not None
        candidate.projection = NativeDynamicInt8Linear(candidate.projection, engine=engine)
        precision_map["projection"] = "int8"
        parameters = {
            "activation_quantization": "dynamic_per_tensor_uint8",
            "kernel": "quantized::linear_dynamic",
            "quantized_engine": engine,
            "weight_bits": 8,
            "weight_granularity": "per_output_channel",
        }
    manifest = QuantizationManifest(
        backend=backend,
        backend_version=torch.__version__,
        method=config.quantization.method,
        source_model_checksum=_SOURCE_CHECKSUM,
        module_precision_map=precision_map,
        excluded_modules=tuple(
            name for name, precision in precision_map.items() if precision == "float32"
        ),
        quantization_parameters=parameters,
        serialized_size_bytes=0,
    )
    loaded = reference.LoadedModel(
        model=candidate,
        tokenizer=object(),
        descriptor=_descriptor(resolved_class=resolved_class),
        load_time_seconds=0.01,
    )
    quantized = QuantizedModel(loaded=loaded, manifest=manifest)
    artifact = export_recipe(quantized, destination, config.quantization)
    return Path(artifact.path), candidate


def _rewrite_recipe(path: Path, update: Callable[[dict[str, Any]], None]) -> None:
    metadata = path / "metadata.json"
    raw: dict[str, Any] = json.loads(metadata.read_text(encoding="utf-8"))
    update(raw)
    for _ in range(16):
        payload = (json.dumps(raw, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        measured = len(payload) + (path / "model.safetensors").stat().st_size
        if raw["quantization"]["serialized_size_bytes"] == measured:
            metadata.write_bytes(payload)
            return
        raw["quantization"]["serialized_size_bytes"] = measured
    raise AssertionError("mutated recipe size did not converge")


def test_export_records_versioned_truthful_reload_adapter(
    config_factory: Callable[..., ExperimentConfig], tmp_path: Path
) -> None:
    config = _source_weight_free_config(config_factory, backend="noop")
    hf_path, _ = _export_candidate(tmp_path / "hf", config, backend="noop")
    trusted_path, _ = _export_candidate(
        tmp_path / "trusted",
        config,
        backend="noop",
        resolved_class="tests.fixtures.TrustedModel",
    )

    assert (
        load_export_recipe(hf_path)["reload"]["adapter"]
        == HF_CAUSAL_LM_SOURCE_WEIGHT_FREE_RELOAD_ADAPTER
    )
    assert (
        load_export_recipe(trusted_path)["reload"]["adapter"] == TRUSTED_SOURCE_MODEL_RELOAD_ADAPTER
    )


def test_recipe_checksum_validation_streams_tensor_payload(
    config_factory: Callable[..., ExperimentConfig],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _source_weight_free_config(config_factory, backend="noop")
    export, _ = _export_candidate(tmp_path / "streaming", config, backend="noop")

    def fail_read_bytes(_path: Path) -> bytes:
        raise AssertionError("tensor validation must not materialize the whole file")

    monkeypatch.setattr(Path, "read_bytes", fail_read_bytes)
    assert load_export_recipe(export)["reload"]["backend"] == "noop"


def test_source_weight_free_noop_reload_is_strict_and_provenanced(
    config_factory: Callable[..., ExperimentConfig],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _source_weight_free_config(config_factory, backend="noop")
    export, expected = _export_candidate(tmp_path / "noop", config, backend="noop")
    adapter = _FakeEmptyHFAdapter()

    def fail_deepcopy(_value: Any) -> Any:
        raise AssertionError("source-weight-free reload must not deepcopy a source model")

    monkeypatch.setattr(reference.copy, "deepcopy", fail_deepcopy)
    loaded, provenance = reload_exported_model_source_weight_free(
        export,
        adapter,
        config,
        TorchEagerCPURuntime(),
    )

    inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 10.0
    assert torch.equal(loaded.loaded.model(inputs), expected(inputs))
    assert adapter.calls == 1
    assert loaded.loaded.load_time_kind == "candidate_source_weight_free_export_load"
    assert not any(
        tensor.is_meta
        for _, tensor in (
            *loaded.loaded.model.named_parameters(),
            *loaded.loaded.model.named_buffers(),
        )
    )
    assert provenance.reload_adapter == HF_CAUSAL_LM_SOURCE_WEIGHT_FREE_RELOAD_ADAPTER
    assert provenance.backend == "noop"
    assert provenance.native_wrapper_count == 0
    assert provenance.strict_load is True
    assert provenance.assign is True
    assert provenance.missing_keys == ()
    assert provenance.unexpected_keys == ()
    assert provenance.meta_tensor_names == ()
    assert provenance.source_weights_loaded is False
    assert provenance.source_metadata_file_sha256 == _SOURCE_METADATA_HASHES
    assert len(provenance.metadata_sha256) == 64
    assert len(provenance.tensor_sha256) == 64
    assert len(provenance.bundle_sha256) == 64
    assert len(provenance.candidate_state_checksum) == 64
    assert provenance.as_dict()["source_weights_loaded"] is False


def test_source_weight_free_native_int8_reloads_repacks_and_matches_with_tolerance(
    config_factory: Callable[..., ExperimentConfig], tmp_path: Path
) -> None:
    capability = probe_native_dynamic_int8()
    if not capability.supported or capability.implementation is None:
        pytest.skip("this PyTorch build has no native dynamic INT8 CPU kernel")
    config = _source_weight_free_config(
        config_factory,
        backend="torch_native_dynamic_int8",
        engine=capability.implementation,
    )
    export, expected = _export_candidate(
        tmp_path / "native",
        config,
        backend="torch_native_dynamic_int8",
        engine=capability.implementation,
    )

    loaded, provenance = reload_exported_model_source_weight_free(
        export,
        _FakeEmptyHFAdapter(),
        config,
        TorchEagerCPURuntime(),
    )

    wrapper = loaded.loaded.model.projection
    assert isinstance(wrapper, NativeDynamicInt8Linear)
    assert wrapper._packed_weight is not None
    inputs = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 11.0
    assert torch.allclose(loaded.loaded.model(inputs), expected(inputs), atol=1e-5, rtol=1e-5)
    assert provenance.native_wrapper_count == 1
    assert provenance.quantized_module_names == ("projection",)
    assert provenance.tensor_count == len(loaded.loaded.model.state_dict())


@pytest.mark.parametrize(
    "adapter_name",
    ("local_fixture", TRUSTED_SOURCE_MODEL_RELOAD_ADAPTER),
)
def test_source_weight_free_reload_rejects_non_hf_adapter_before_shell_load(
    config_factory: Callable[..., ExperimentConfig], tmp_path: Path, adapter_name: str
) -> None:
    config = _source_weight_free_config(config_factory, backend="noop")
    export, _ = _export_candidate(tmp_path / adapter_name, config, backend="noop")
    _rewrite_recipe(export, lambda raw: raw["reload"].update({"adapter": adapter_name}))
    adapter = _FakeEmptyHFAdapter()

    with pytest.raises(ValueError, match="hf_causal_lm_source_weight_free_v1"):
        reload_exported_model_source_weight_free(
            export,
            adapter,
            config,
            TorchEagerCPURuntime(),
        )

    assert adapter.calls == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing", "tensor inventory"),
        ("extra", "tensor inventory"),
        ("shape", "tensor shape"),
        ("dtype", "tensor dtype"),
    ),
)
def test_source_weight_free_reload_rejects_tensor_contract_drift(
    config_factory: Callable[..., ExperimentConfig],
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    config = _source_weight_free_config(config_factory, backend="noop")
    export, _ = _export_candidate(tmp_path / mutation, config, backend="noop")
    tensor_path = export / "model.safetensors"
    state = load_file(str(tensor_path), device="cpu")
    if mutation == "missing":
        state.pop("projection.weight")
    elif mutation == "extra":
        state["unexpected"] = torch.zeros((1,), dtype=torch.float32)
    elif mutation == "shape":
        state["projection.weight"] = state["projection.weight"][:, :-1].contiguous()
    else:
        state["projection.weight"] = state["projection.weight"].to(torch.float16)
    save_file(state, str(tensor_path))
    tensor_sha256 = hashlib.sha256(tensor_path.read_bytes()).hexdigest()
    _rewrite_recipe(
        export,
        lambda raw: raw["reload"].update({"tensor_sha256": tensor_sha256}),
    )

    with pytest.raises(ValueError, match=message):
        reload_exported_model_source_weight_free(
            export,
            _FakeEmptyHFAdapter(),
            config,
            TorchEagerCPURuntime(),
        )


def test_source_weight_free_reload_rejects_config_model_identity_mismatch(
    config_factory: Callable[..., ExperimentConfig], tmp_path: Path
) -> None:
    config = _source_weight_free_config(config_factory, backend="noop")
    export, _ = _export_candidate(tmp_path / "identity", config, backend="noop")
    wrong = config.model_copy(
        update={"model": config.model.model_copy(update={"model_id": "other/model"})}
    )
    adapter = _FakeEmptyHFAdapter()

    with pytest.raises(ValueError, match="model identity"):
        reload_exported_model_source_weight_free(
            export,
            adapter,
            wrong,
            TorchEagerCPURuntime(),
        )

    assert adapter.calls == 0


@pytest.mark.parametrize("bias", (False, True))
def test_native_dynamic_int8_from_empty_defers_prepack_until_strict_load(
    bias: bool,
) -> None:
    capability = probe_native_dynamic_int8()
    if not capability.supported or capability.implementation is None:
        pytest.skip("this PyTorch build has no native dynamic INT8 CPU kernel")
    torch.manual_seed(303)
    source = NativeDynamicInt8Linear(nn.Linear(8, 6, bias=bias), engine=capability.implementation)
    empty = NativeDynamicInt8Linear.from_empty(
        in_features=8,
        out_features=6,
        bias=bias,
        engine=capability.implementation,
        device="meta",
    )

    assert empty._packed_weight is None
    assert all(tensor.is_meta for tensor in empty.state_dict().values())
    result = empty.load_state_dict(source.state_dict(), strict=True, assign=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert empty._packed_weight is not None
    assert not any(tensor.is_meta for tensor in empty.state_dict().values())
    inputs = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 7.0
    assert torch.equal(empty(inputs), source(inputs))
