from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from inkling_quant_lab.config import RuntimeConfig
from inkling_quant_lab.exceptions import CapabilityError
from inkling_quant_lab.mlx_contract import (
    MLXEnvironmentStatus,
    expected_float32_only_names,
    expected_quantized_leaf_names,
)
from inkling_quant_lab.models import mlx_lm_mixtral as mlx_model
from inkling_quant_lab.models.base import LoadedModel, ModelBatch
from inkling_quant_lab.models.mlx_lm_mixtral import (
    MLXMixtralAdapter,
    MLXModelHandle,
    MLXTokenizerAdapter,
)
from inkling_quant_lab.runtimes import mlx_metal
from inkling_quant_lab.runtimes.mlx_metal import MLXMetalRuntime
from tests.unit.test_mlx_registered_backend import _descriptor, _mlx_config

pytestmark = pytest.mark.unit


class _Tensor:
    def __init__(self, shape: tuple[int, ...], nbytes: int) -> None:
        self.shape = shape
        self.nbytes = nbytes


class _QuantizableLeaf:
    def __init__(self, *, with_bias: bool = False) -> None:
        self.weight = _Tensor((2, 3), 24)
        self.bias = _Tensor((2,), 8) if with_bias else None

    def to_quantized(self) -> None:
        return None


class _FloatLeaf:
    def __init__(self) -> None:
        self.weight = _Tensor((3,), 12)


class _RawTokenizer:
    eos_token_ids = (9,)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if text == "empty":
            return []
        if text == "long":
            return list(range(257))
        return [1, 2] if add_special_tokens else [2]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return ":".join(str(token) for token in token_ids)


def _model_handle() -> MLXModelHandle:
    named: dict[str, Any] = {
        name: _QuantizableLeaf(with_bias=name == "lm_head")
        for name in expected_quantized_leaf_names()
    }
    named.update({name: _FloatLeaf() for name in expected_float32_only_names()})
    layers = []
    for layer in range(6):
        gate_name = f"model.layers.{layer}.block_sparse_moe.gate"
        block = SimpleNamespace(
            gate=named[gate_name],
            num_experts=4,
            num_experts_per_tok=2,
        )
        layers.append(SimpleNamespace(block_sparse_moe=block))
    module = SimpleNamespace(
        named_modules=lambda: tuple(named.items()),
        model=SimpleNamespace(layers=layers),
    )
    return MLXModelHandle(
        module=module,
        config={},
        source_snapshot=Path("snapshot"),
        source_audit={"safe": True},
        seed=17,
    )


def _loaded(handle: Any) -> LoadedModel:
    return LoadedModel(
        model=handle,
        tokenizer=MLXTokenizerAdapter(_RawTokenizer()),
        descriptor=_descriptor(),
        load_time_seconds=0.01,
    )


def test_mlx_tokenizer_adapter_pads_and_rejects_unsafe_shapes() -> None:
    tokenizer = MLXTokenizerAdapter(_RawTokenizer())

    assert tokenizer.encode("short") == [1, 2]
    assert tokenizer.encode("short", special_tokens=False) == [2]
    assert tokenizer.decode((1, 2)) == "1:2"
    assert tokenizer.batch_encode(("short", "other")) == (
        ((1, 2), (1, 2)),
        ((1, 1), (1, 1)),
    )

    class _UnevenTokenizer(_RawTokenizer):
        eos_token_ids: tuple[int, ...] = ()

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            del add_special_tokens
            return [1] if text == "one" else [2, 3]

    uneven = MLXTokenizerAdapter(_UnevenTokenizer())
    assert uneven.batch_encode(("one", "two")) == (
        ((1, 0), (2, 3)),
        ((1, 0), (1, 1)),
    )

    with pytest.raises(ValueError, match="must not be empty"):
        tokenizer.batch_encode(())
    with pytest.raises(ValueError, match="at least one token"):
        tokenizer.batch_encode(("empty",))
    with pytest.raises(ValueError, match="256-token"):
        tokenizer.batch_encode(("long",))


def test_mlx_inventory_rows_moe_discovery_and_hook_cleanup() -> None:
    handle = _model_handle()
    adapter = MLXMixtralAdapter()
    loaded = _loaded(handle)

    inventory = adapter.enumerate_modules(loaded)
    assert len(inventory) == 63
    assert sum(item.is_expert for item in inventory) == 18
    assert sum(item.is_router for item in inventory) == 6
    assert next(item for item in inventory if item.name == "lm_head").parameter_count == 8
    assert mlx_model._layer_id("model.layers.3.self_attn.q_proj") == (
        "model.layers.3.block_sparse_moe"
    )
    assert mlx_model._layer_id("lm_head") is None

    moe = adapter.discover_moe(loaded)
    assert moe is not None
    assert len(moe.layers) == 6
    assert all(layer.expert_count == 4 and layer.top_k == 2 for layer in moe.layers)
    assert adapter.discover_moe(_loaded(object())) is None

    batch = ModelBatch(
        sample_ids=("sample",),
        input_ids=((1, 2),),
        attention_mask=((1, 1),),
    )
    assert mlx_model._rows(batch) == ((1, 2),)

    block = SimpleNamespace(gate="wrapped")
    hook = mlx_model._MLXHookHandle(((block, "original"),))
    hook.remove()
    hook.remove()
    assert block.gate == "original"

    with pytest.raises(CapabilityError, match="not owned by MLX"):
        adapter.enumerate_modules(_loaded(object()))

    broken_layers = list(handle.module.model.layers)
    handle.module.model.layers = broken_layers[:-1]
    with pytest.raises(CapabilityError, match="layer count"):
        adapter.discover_moe(loaded)
    handle.module.model.layers = broken_layers
    broken_layers[0].block_sparse_moe.num_experts = 3
    with pytest.raises(CapabilityError, match="routing dimensions"):
        adapter.discover_moe(loaded)


@pytest.mark.parametrize(
    "batch",
    (
        ModelBatch(sample_ids=("x",), input_ids="not-rows"),
        ModelBatch(sample_ids=("x",), input_ids=((),)),
        ModelBatch(sample_ids=("x",), input_ids=((-1,),)),
        ModelBatch(sample_ids=("x", "y"), input_ids=((1,),)),
        ModelBatch(sample_ids=("x",), input_ids=((1,),), attention_mask=()),
        ModelBatch(sample_ids=("x",), input_ids=((1,),), attention_mask=((0,),)),
    ),
)
def test_mlx_rows_reject_invalid_transport(batch: ModelBatch) -> None:
    with pytest.raises(ValueError):
        mlx_model._rows(batch)


def test_mlx_adapter_capabilities_and_load_use_audited_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _mlx_config(str(tmp_path))
    handle = _model_handle()
    tokenizer = MLXTokenizerAdapter(_RawTokenizer())
    audits: list[Path] = []
    monkeypatch.setattr(mlx_model, "audit_source_snapshot", lambda path: audits.append(path) or {})
    monkeypatch.setattr(mlx_model, "_require_environment", lambda: {"mlx": "0.32.0"})
    monkeypatch.setattr(
        mlx_model,
        "load_exact_source",
        lambda snapshot, seed: (handle, tokenizer, 0.25),
    )

    adapter = MLXMixtralAdapter()
    capabilities = adapter.capabilities(config)
    loaded = adapter.load(config, SimpleNamespace())

    assert capabilities.is_moe and capabilities.supports_token_level_routes
    assert loaded.model is handle
    assert loaded.load_time_seconds == 0.25
    assert loaded.descriptor.checksum == _descriptor().checksum
    assert audits and all(path == tmp_path.resolve() for path in audits)

    invalid = config.model_copy(
        update={"model": config.model.model_copy(update={"dtype": "float16"})}
    )
    with pytest.raises(CapabilityError, match="exact offline float32"):
        adapter.capabilities(invalid)


def _install_fake_mlx(monkeypatch: pytest.MonkeyPatch, *, metal_available: bool) -> Any:
    calls = SimpleNamespace(synchronized=0, cleared=0)
    core = ModuleType("mlx.core")
    core.metal = SimpleNamespace(is_available=lambda: metal_available)
    core.synchronize = lambda: setattr(calls, "synchronized", calls.synchronized + 1)
    core.clear_cache = lambda: setattr(calls, "cleared", calls.cleared + 1)
    core.get_active_memory = lambda: 456
    package = ModuleType("mlx")
    package.__path__ = []  # type: ignore[attr-defined]
    package.core = core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return calls


def test_mlx_runtime_probe_memory_context_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = MLXEnvironmentStatus(
        available=True,
        versions={"mlx": "0.32.0", "mlx-lm": "0.31.3"},
        reasons=(),
    )
    monkeypatch.setattr(mlx_metal, "mlx_environment_status", lambda: status)
    monkeypatch.setattr(mlx_metal, "current_process_rss_bytes", lambda: 123)
    calls = _install_fake_mlx(monkeypatch, metal_available=True)

    runtime = MLXMetalRuntime()
    capabilities = runtime.probe(_mlx_config().runtime)
    assert capabilities.available
    assert capabilities.devices == ("mps",)
    assert capabilities.supports_forward_loss and capabilities.supports_routing_hooks

    snapshot = runtime.memory_snapshot()
    assert snapshot.host_bytes == 123
    assert snapshot.device_bytes == 456
    assert snapshot.device_measurement_kind == "mlx_allocator_active_bytes_at_sample"
    with runtime.execution_context():
        pass
    runtime.synchronize()
    runtime.cleanup()
    assert calls.synchronized == 2
    assert calls.cleared == 1
    assert isinstance(mlx_metal.create_runtime(), MLXMetalRuntime)


def test_mlx_runtime_unavailable_and_uninitialized_memory_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = MLXEnvironmentStatus(
        available=False,
        versions={},
        reasons=("missing mlx",),
    )
    monkeypatch.setattr(mlx_metal, "mlx_environment_status", lambda: unavailable)
    runtime = MLXMetalRuntime()
    bad_config = RuntimeConfig(
        backend="mlx_metal",
        device="cpu",
        dtype="float16",
        device_map="auto",
        sharding={"ranks": 2},
    )
    capabilities = runtime.probe(bad_config)
    assert not capabilities.available
    assert capabilities.devices == ()
    assert len(capabilities.reasons) == 5
    assert capabilities.version == "unavailable"

    monkeypatch.setattr(mlx_metal, "current_process_rss_bytes", lambda: 321)
    snapshot = runtime.memory_snapshot()
    assert snapshot.host_available and not snapshot.device_available
    assert snapshot.host_measurement_kind == "process_current_rss"

    def _rss_failure() -> int:
        raise OSError("unsupported")

    monkeypatch.setattr(mlx_metal, "current_process_rss_bytes", _rss_failure)
    missing = runtime.memory_snapshot()
    assert missing.host_bytes is None
    assert not missing.host_available and not missing.device_available


def test_mlx_runtime_reports_imported_but_missing_metal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = MLXEnvironmentStatus(
        available=True,
        versions={"mlx": "0.32.0"},
        reasons=(),
    )
    monkeypatch.setattr(mlx_metal, "mlx_environment_status", lambda: status)
    _install_fake_mlx(monkeypatch, metal_available=False)

    capabilities = MLXMetalRuntime().probe(_mlx_config().runtime)

    assert not capabilities.available
    assert "no available Metal device" in capabilities.reasons[-1]
