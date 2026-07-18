from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import yaml
from torch import nn

from inkling_quant_lab.config import QuantizationConfig
from inkling_quant_lab.exceptions import CapabilityError, QuantizationError
from inkling_quant_lab.models.base import LoadedModel, ModelBatch
from inkling_quant_lab.models.local import LocalFixtureAdapter
from inkling_quant_lab.quantization import optional
from inkling_quant_lab.quantization.base import (
    PinnedSourceReloadQuantizer,
    QuantizationManifest,
    QuantizedModel,
)
from inkling_quant_lab.runtimes.base import RuntimeCapabilities
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.unit

_REVISION = "1" * 40
_STORIES15M_REVISION = "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _runtime() -> RuntimeCapabilities:
    return RuntimeCapabilities(
        backend="torch_eager_cuda",
        available=True,
        devices=("cuda",),
        supported_dtypes=("float32", "float16", "bfloat16"),
        supports_routing_hooks=True,
        supports_forward_loss=True,
        supports_memory_measurement=True,
        supports_energy_measurement=False,
        supports_sharding=False,
        version="2.8.0",
    )


def _cpu_runtime() -> RuntimeCapabilities:
    return RuntimeCapabilities(
        backend="torch_eager_cpu",
        available=True,
        devices=("cpu",),
        supported_dtypes=("float32", "float16", "bfloat16"),
        supports_routing_hooks=True,
        supports_forward_loss=True,
        supports_memory_measurement=True,
        supports_energy_measurement=False,
        supports_sharding=False,
        version="2.13.0",
    )


def _quantization(backend: str, *, calibration: bool, device: str = "cuda:0") -> QuantizationConfig:
    payload: dict[str, Any] = {
        "backend": backend,
        "method": backend,
        "policy": {
            "type": "uniform",
            "default_precision": "fp8" if backend == "fp8" else "int4",
            "preserve_router_precision": True,
            "preserve_output_head": True,
        },
        "parameters": {"local_files_only": True, "device": device},
        "export": {"format": "safetensors", "destination": "candidate"},
    }
    if calibration:
        payload["calibration"] = {
            "dataset": "local://fixtures/calibration",
            "revision": "fixture-data-v1",
            "split": "calibration",
            "sample_ids": ["calibration-001"],
            "max_samples": 4,
        }
    return QuantizationConfig.model_validate(payload)


def _source(config_factory) -> LoadedModel:
    config = config_factory()
    loaded = LocalFixtureAdapter().load(config, TorchEagerCPURuntime())
    loaded.descriptor = replace(
        loaded.descriptor,
        model_id="org/pinned-native-mixtral",
        revision=_REVISION,
        architecture="MixtralForCausalLM",
        resolved_class="transformers.models.mixtral.MixtralForCausalLM",
    )
    return loaded


def _stories15m_source(config_factory) -> LoadedModel:
    loaded = _source(config_factory)
    loaded.descriptor = replace(
        loaded.descriptor,
        model_id="ggml-org/stories15M_MOE",
        revision=_STORIES15M_REVISION,
    )
    return loaded


def _dependencies(kind: str) -> optional._DependencyStatus:
    versions = {
        "accelerate": "1.14.0",
        "safetensors": "0.8.0",
        "torch": "2.13.0",
        "transformers": "5.12.1",
    }
    if kind == "gptqmodel":
        versions.update(
            {
                "defuser": "0.0.23",
                "gptqmodel": "5.8.0",
                "huggingface-hub": "1.23.0",
                "kernels": "0.14.1",
            }
        )
    return optional._DependencyStatus(available=True, versions=versions, reasons=())


def _patch_dependencies(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    status = _dependencies(kind)
    monkeypatch.setattr(optional, "_dependency_status", lambda _requirements: status)


def _calibration_batch(source: LoadedModel) -> tuple[ModelBatch, ...]:
    input_ids, attention_mask = source.tokenizer.batch_encode(
        ("alpha beta gamma delta red blue green yellow one two three four",)
    )
    return (
        ModelBatch(
            sample_ids=("calibration-001",),
            input_ids=input_ids,
            attention_mask=attention_mask,
        ),
    )


def _policy(source: LoadedModel, target: str) -> SimpleNamespace:
    names = {name for name, _module in source.model.named_modules()}
    assert "lm_head" in names
    router_name = next(name for name in sorted(names) if name.endswith(".router"))
    target_name = next(
        name
        for name, module in source.model.named_modules()
        if name != "lm_head" and not name.endswith(".router") and isinstance(module, nn.Linear)
    )
    return SimpleNamespace(
        precision_map={
            target_name: target,
            router_name: "float32",
            "lm_head": "float32",
        }
    )


def _complete_linear_policy(source: LoadedModel, target: str) -> SimpleNamespace:
    linears = tuple(
        name
        for name, module in source.model.named_modules()
        if name and isinstance(module, nn.Linear)
    )
    selected = next(name for name in linears if name != "lm_head" and not name.endswith(".router"))
    return SimpleNamespace(
        precision_map={name: target if name == selected else "float32" for name in sorted(linears)}
    )


def test_factories_expose_explicit_pinned_source_reload_lifecycle() -> None:
    for quantizer in (
        optional.create_awq_quantizer(),
        optional.create_gptq_quantizer(),
        optional.create_fp8_quantizer(),
    ):
        assert isinstance(quantizer, PinnedSourceReloadQuantizer)
        assert quantizer.lifecycle == "pinned_source_reload"


def test_dependency_gate_enforces_audited_patch_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirement = optional._DependencyRequirement(
        "example-dist",
        "example_module",
        (5, 8, 0),
        (5, 8, 1),
    )
    monkeypatch.setattr(optional.importlib.util, "find_spec", lambda _module: object())
    monkeypatch.setattr(optional.importlib.metadata, "version", lambda _distribution: "5.8.1")

    status = optional._dependency_status((requirement,))

    assert not status.available
    assert status.versions == {"example-dist": "5.8.1"}
    assert "outside supported range >=5.8.0,<5.8.1" in status.reasons[0]


@pytest.mark.parametrize(
    ("backend", "filename"),
    (
        ("awq", "awq_gptqmodel_capability_probe.yaml"),
        ("gptq", "gptq_gptqmodel_capability_probe.yaml"),
        ("gptq", "gptq_gptqmodel_cpu_stories15m.yaml"),
        ("fp8", "finegrained_fp8_capability_probe.yaml"),
    ),
)
def test_checked_optional_yaml_is_a_valid_quantization_fragment(
    backend: str, filename: str
) -> None:
    path = _PROJECT_ROOT / "configs" / "quantization" / filename
    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert isinstance(payload, dict) and set(payload) == {"quantization"}
    config = QuantizationConfig.model_validate(payload["quantization"])
    assert config.backend == backend
    assert config.method == backend


def test_uninstalled_production_backend_is_actionable_without_importing_it(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
) -> None:
    monkeypatch.setattr(
        optional,
        "_dependency_status",
        lambda _requirements: optional._DependencyStatus(
            available=False,
            versions={},
            reasons=("missing Python package gptqmodel",),
        ),
    )
    source = _source(config_factory)
    quantizer = optional.create_awq_quantizer()
    config = _quantization("awq", calibration=True)

    support = quantizer.check_support(source.descriptor, _runtime(), config)

    assert not support.available
    assert not support.supported
    assert support.install_extra == "awq"
    assert "uv sync --extra awq" in support.message()


@pytest.mark.parametrize("backend", ["awq", "gptq"])
def test_awq_and_gptq_support_requires_calibration_and_explicit_source_settings(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
    backend: str,
) -> None:
    _patch_dependencies(monkeypatch, "gptqmodel")
    source = _source(config_factory)
    quantizer = optional.GPTQModelQuantizer(backend)

    supported = quantizer.check_support(
        source.descriptor, _runtime(), _quantization(backend, calibration=True)
    )
    missing_calibration = quantizer.check_support(
        source.descriptor, _runtime(), _quantization(backend, calibration=False)
    )

    assert supported.available and supported.supported
    assert supported.supported_precisions[0] == "int4"
    assert supported.message() == "supported"
    assert supported.install_extra is None
    assert supported.remediation is None
    assert not missing_calibration.supported
    assert "calibration" in missing_calibration.message().lower()


def test_cpu_gptq_support_is_limited_to_exact_executed_stories15m_pin(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
) -> None:
    _patch_dependencies(monkeypatch, "gptqmodel")
    exact_source = _stories15m_source(config_factory)
    other_source = _source(config_factory)
    quantizer = optional.create_gptq_quantizer()
    config = _quantization("gptq", calibration=True, device="cpu")

    exact = quantizer.check_support(exact_source.descriptor, _cpu_runtime(), config)
    other = quantizer.check_support(other_source.descriptor, _cpu_runtime(), config)

    assert exact.available and exact.supported
    assert not other.supported
    assert "pinned ggml-org/stories15M_MOE revision" in other.message()


def test_cpu_gptq_support_requires_the_exact_executed_software_matrix(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
) -> None:
    status = _dependencies("gptqmodel")
    versions = {**status.versions, "transformers": "5.12.2"}
    monkeypatch.setattr(
        optional,
        "_dependency_status",
        lambda _requirements: optional._DependencyStatus(
            available=True,
            versions=versions,
            reasons=(),
        ),
    )
    report = optional.create_gptq_quantizer().check_support(
        _stories15m_source(config_factory).descriptor,
        _cpu_runtime(),
        _quantization("gptq", calibration=True, device="cpu"),
    )

    assert report.available
    assert not report.supported
    assert "transformers==5.12.1" in report.message()


@pytest.mark.parametrize("backend", ["awq", "fp8"])
def test_cpu_remains_fail_closed_for_unvalidated_optional_backends(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
    backend: str,
) -> None:
    _patch_dependencies(monkeypatch, "gptqmodel" if backend == "awq" else "fp8")
    source = _stories15m_source(config_factory)

    report = (
        optional.create_awq_quantizer() if backend == "awq" else optional.create_fp8_quantizer()
    ).check_support(
        source.descriptor,
        _cpu_runtime(),
        _quantization(backend, calibration=backend == "awq", device="cpu"),
    )

    assert not report.supported
    assert "CUDA" in report.message()


def test_fp8_fails_closed_below_documented_compute_capability(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
) -> None:
    _patch_dependencies(monkeypatch, "fp8")
    monkeypatch.setattr(optional, "_cuda_compute_capability", lambda _device: (8, 9))
    source = _source(config_factory)

    support = optional.create_fp8_quantizer().check_support(
        source.descriptor, _runtime(), _quantization("fp8", calibration=False)
    )

    assert support.available
    assert not support.supported
    assert "compute capability >=9.0" in support.message()


@pytest.mark.parametrize(
    ("device", "capabilities", "expected_supported"),
    (
        ("cuda:0", {0: (8, 9), 1: (9, 0)}, False),
        ("cuda:1", {0: (8, 9), 1: (9, 0)}, True),
    ),
)
def test_fp8_capability_gate_queries_exact_configured_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
    device: str,
    capabilities: dict[int, tuple[int, int]],
    expected_supported: bool,
) -> None:
    _patch_dependencies(monkeypatch, "fp8")
    monkeypatch.setattr(optional.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(optional.torch.cuda, "device_count", lambda: 2)
    queried: list[int] = []

    def get_device_capability(index: int) -> tuple[int, int]:
        queried.append(index)
        return capabilities[index]

    monkeypatch.setattr(optional.torch.cuda, "get_device_capability", get_device_capability)
    source = _source(config_factory)

    support = optional.create_fp8_quantizer().check_support(
        source.descriptor,
        _runtime(),
        _quantization("fp8", calibration=False, device=device),
    )

    assert support.available
    assert support.supported is expected_supported
    assert queried == [int(device.removeprefix("cuda:"))]
    if not expected_supported:
        assert device in support.message()


def test_optional_preload_probe_allows_unresolved_but_conversion_requires_resolved_identity(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
) -> None:
    _patch_dependencies(monkeypatch, "fp8")
    monkeypatch.setattr(optional, "_cuda_compute_capability", lambda _device: (9, 0))
    source = _source(config_factory)
    source.descriptor = replace(
        source.descriptor,
        architecture="unresolved",
        resolved_class="unresolved",
    )
    quantizer = optional.create_fp8_quantizer()
    config = _quantization("fp8", calibration=False)

    support = quantizer.check_support(source.descriptor, _runtime(), config)

    assert support.supported
    with pytest.raises(CapabilityError, match="resolved native MixtralForCausalLM"):
        quantizer.quantize_from_pinned_source(source, _policy(source, "fp8"), None, config)


def test_optional_preload_probe_rejects_architecture_that_only_ends_with_mixtral(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
) -> None:
    _patch_dependencies(monkeypatch, "fp8")
    monkeypatch.setattr(optional, "_cuda_compute_capability", lambda _device: (9, 0))
    source = _source(config_factory)
    source.descriptor = replace(
        source.descriptor,
        architecture="NotMixtralForCausalLM",
        resolved_class="malicious.NotMixtralForCausalLM",
    )

    support = optional.create_fp8_quantizer().check_support(
        source.descriptor, _runtime(), _quantization("fp8", calibration=False)
    )

    assert not support.supported
    assert "validated MixtralForCausalLM adapter" in support.message()


def test_optional_conversion_rejects_non_mixtral_resolved_class(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
) -> None:
    _patch_dependencies(monkeypatch, "fp8")
    monkeypatch.setattr(optional, "_cuda_compute_capability", lambda _device: (9, 0))
    source = _source(config_factory)
    source.descriptor = replace(
        source.descriptor,
        resolved_class="malicious.NotMixtralForCausalLM",
    )
    quantizer = optional.create_fp8_quantizer()
    config = _quantization("fp8", calibration=False)

    assert quantizer.check_support(source.descriptor, _runtime(), config).supported
    with pytest.raises(CapabilityError, match="resolved class"):
        quantizer.quantize_from_pinned_source(source, _policy(source, "fp8"), None, config)


def test_calibration_keeps_tokens_ephemeral_and_binds_artifact_to_source(config_factory) -> None:
    source = _source(config_factory)
    quantizer = optional.create_gptq_quantizer()
    config = _quantization("gptq", calibration=True)

    artifact = quantizer.calibrate(source, _calibration_batch(source), config)

    assert artifact is not None
    assert artifact.sample_ids == ("calibration-001",)
    assert artifact.statistics["sample_count"] == 1.0
    serialized = json.dumps(artifact.model_dump(mode="json"))
    assert "input_ids" not in serialized
    assert "alpha" not in serialized


def test_gptqmodel_calibration_rejects_samples_upstream_would_filter(config_factory) -> None:
    source = _source(config_factory)
    quantizer = optional.create_gptq_quantizer()
    config = _quantization("gptq", calibration=True)
    input_ids, attention_mask = source.tokenizer.batch_encode(("alpha beta",))
    samples = (
        ModelBatch(
            sample_ids=("calibration-001",),
            input_ids=input_ids,
            attention_mask=attention_mask,
        ),
    )

    with pytest.raises(CapabilityError, match="at least 10 active tokens"):
        quantizer.calibrate(source, samples, config)


def test_source_dtype_rejects_mixed_floating_parameter_dtypes(config_factory) -> None:
    source = _source(config_factory)
    first, second = tuple(source.model.parameters())[:2]
    first.data = first.data.float()
    second.data = second.data.half()

    with pytest.raises(QuantizationError, match="one consistent floating-point dtype"):
        optional._source_dtype(source)


def test_gptqmodel_source_is_materialized_at_exact_safe_revision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = SimpleNamespace(
        model_id="org/pinned-native-mixtral",
        revision=_REVISION,
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    calls: dict[str, Any] = {}

    def snapshot_download(**kwargs: Any) -> str:
        calls.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr(
        optional,
        "_load_huggingface_hub_api",
        lambda: SimpleNamespace(snapshot_download=snapshot_download),
    )

    result = optional._materialize_pinned_source(source, local_files_only=False)

    assert result == snapshot.resolve()
    assert calls["repo_id"] == source.model_id
    assert calls["revision"] == _REVISION
    assert calls["local_files_only"] is False
    assert "*.safetensors" in calls["allow_patterns"]
    assert "*.bin" in calls["ignore_patterns"]
    assert any(fnmatch("config.json", pattern) for pattern in calls["allow_patterns"])
    assert any(fnmatch("tokenizer.json", pattern) for pattern in calls["allow_patterns"])
    assert any(fnmatch("tokenizer.model", pattern) for pattern in calls["allow_patterns"])
    assert any(fnmatch("vocab.txt", pattern) for pattern in calls["allow_patterns"])
    assert not any(fnmatch("tokenizer.py", pattern) for pattern in calls["allow_patterns"])
    assert "*.py" in calls["ignore_patterns"]


def test_stories15m_materialization_uses_exact_audited_file_allowlist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = SimpleNamespace(
        model_id="ggml-org/stories15M_MOE",
        revision=_STORIES15M_REVISION,
    )
    snapshot = tmp_path / _STORIES15M_REVISION
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    calls: dict[str, Any] = {}
    audited: list[Path] = []

    def snapshot_download(**kwargs: Any) -> str:
        calls.update(kwargs)
        return str(snapshot)

    monkeypatch.setattr(
        optional,
        "_load_huggingface_hub_api",
        lambda: SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setattr(
        optional,
        "_audit_exact_stories15m_snapshot",
        lambda path: audited.append(path),
    )

    result = optional._materialize_pinned_source(source, local_files_only=True)

    assert result == snapshot.resolve()
    assert audited == [snapshot.resolve()]
    assert "model.safetensors" in calls["allow_patterns"]
    assert "moe_shakespeare15M/checkpoint-400/trainer_state.json" in calls["allow_patterns"]
    assert "*.safetensors" not in calls["allow_patterns"]


@pytest.mark.parametrize(
    ("filename", "executable"),
    (
        ("tokenizer.py", False),
        ("quant_kernel.so", False),
        ("snapshot-launcher", True),
    ),
)
def test_gptqmodel_source_snapshot_rejects_executable_or_loadable_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    filename: str,
    executable: bool,
) -> None:
    source = SimpleNamespace(
        model_id="org/pinned-native-mixtral",
        revision=_REVISION,
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    unsafe = snapshot / filename
    unsafe.write_bytes(b"unsafe")
    if executable:
        unsafe.chmod(0o755)
    monkeypatch.setattr(
        optional,
        "_load_huggingface_hub_api",
        lambda: SimpleNamespace(snapshot_download=lambda **_kwargs: str(snapshot)),
    )

    with pytest.raises(QuantizationError, match="executable or loadable code"):
        optional._materialize_pinned_source(source, local_files_only=True)


@pytest.mark.parametrize(
    "magic",
    (
        b"\x7fELF",
        b"MZ\x90\x00",
        b"\xfe\xed\xfa\xcf",
        b"\x00asm",
    ),
)
def test_gptqmodel_source_snapshot_rejects_native_magic_under_tokenizer_data_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    magic: bytes,
) -> None:
    source = SimpleNamespace(
        model_id="org/pinned-native-mixtral",
        revision=_REVISION,
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"weights")
    (snapshot / "tokenizer.model").write_bytes(magic + b"native-payload")
    monkeypatch.setattr(
        optional,
        "_load_huggingface_hub_api",
        lambda: SimpleNamespace(snapshot_download=lambda **_kwargs: str(snapshot)),
    )

    with pytest.raises(QuantizationError, match="executable or loadable code"):
        optional._materialize_pinned_source(source, local_files_only=True)


def _write_minimal_optional_export(path: Path) -> None:
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"weights")
    (path / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "gptq",
                    "format": "gptq",
                    "checkpoint_format": "gptq",
                }
            }
        ),
        encoding="utf-8",
    )


def test_optional_export_accepts_inert_tokenizer_data_files(tmp_path: Path) -> None:
    export = tmp_path / "safe-export"
    _write_minimal_optional_export(export)
    (export / "tokenizer.json").write_text("{}", encoding="utf-8")
    (export / "tokenizer.model").write_bytes(b"sentencepiece-data")
    (export / "vocab.txt").write_text("token\n", encoding="utf-8")

    optional._validate_safe_export(export, "gptq")


@pytest.mark.parametrize(
    ("filename", "executable"),
    (
        ("tokenizer.py", False),
        ("quant_kernel.dylib", False),
        ("export-launcher", True),
    ),
)
def test_optional_export_rejects_executable_or_loadable_code(
    tmp_path: Path,
    filename: str,
    executable: bool,
) -> None:
    export = tmp_path / "unsafe-export"
    _write_minimal_optional_export(export)
    unsafe = export / filename
    unsafe.write_bytes(b"unsafe")
    if executable:
        unsafe.chmod(0o755)

    with pytest.raises(QuantizationError, match="executable or loadable code"):
        optional._validate_safe_export(export, "gptq")


class _FusedExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_experts = 4
        self.w1 = nn.Parameter(torch.zeros(4, 8, 4))
        self.w2 = nn.Parameter(torch.zeros(4, 4, 8))
        self.w3 = nn.Parameter(torch.zeros(4, 8, 4))


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = nn.Linear(4, 4, bias=False)
        self.experts = _FusedExperts()
        self.top_k = 2


def _fused_mixtral_source(source: LoadedModel) -> LoadedModel:
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([nn.Module()])
    root.model.layers[0].self_attn = nn.Module()
    root.model.layers[0].self_attn.q_proj = nn.Linear(4, 4, bias=False)
    root.model.layers[0].mlp = MixtralSparseMoeBlock()
    root.lm_head = nn.Linear(4, 4, bias=False)
    return LoadedModel(
        model=root,
        tokenizer=source.tokenizer,
        descriptor=source.descriptor,
        load_time_seconds=source.load_time_seconds,
    )


def _fused_policy(*, expert_precision: str = "float32") -> SimpleNamespace:
    precision_map = {
        "lm_head": "float32",
        "model.layers.0.mlp.gate": "float32",
        "model.layers.0.self_attn.q_proj": "int4",
    }
    precision_map.update(
        {f"model.layers.0.mlp.experts.{expert}": expert_precision for expert in range(4)}
    )
    return SimpleNamespace(precision_map=precision_map)


def test_gptqmodel_policy_expands_fused_experts_into_exact_defuser_exclusions(
    config_factory,
) -> None:
    source = _fused_mixtral_source(_stories15m_source(config_factory))

    precision_map, selected, excluded = optional._gptqmodel_policy_partition(
        source,
        _fused_policy(),
        "int4",
    )

    assert precision_map["model.layers.0.mlp.experts.0"] == "float32"
    assert selected == ("model.layers.0.self_attn.q_proj",)
    assert "model.layers.0.mlp.experts.0" not in excluded
    assert {
        f"model.layers.0.mlp.experts.{expert}.{projection}"
        for expert in range(4)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }.issubset(excluded)


def test_common_gptqmodel_policy_rejects_quantized_fused_expert_slices(
    config_factory,
) -> None:
    source = _fused_mixtral_source(_stories15m_source(config_factory))

    with pytest.raises(CapabilityError, match="keeps fused expert slices full precision"):
        optional._gptqmodel_policy_partition(
            source,
            _fused_policy(expert_precision="int4"),
            "int4",
        )


def test_gptqmodel_policy_rejects_missing_fused_expert_assignment(config_factory) -> None:
    source = _fused_mixtral_source(_stories15m_source(config_factory))
    policy = _fused_policy()
    del policy.precision_map["model.layers.0.mlp.experts.3"]

    with pytest.raises(CapabilityError, match="omits fused expert slices"):
        optional._gptqmodel_policy_partition(source, policy, "int4")


def test_gptqmodel_policy_rejects_unrealized_full_precision_cast(config_factory) -> None:
    source = _fused_mixtral_source(_stories15m_source(config_factory))
    policy = _fused_policy()
    policy.precision_map["model.layers.0.mlp.experts.0"] = "float16"

    with pytest.raises(CapabilityError, match="preserve the source reload dtype"):
        optional._gptqmodel_policy_partition(source, policy, "int4")


def test_gptqmodel_inventory_verifier_rejects_wrapper_flag_without_exact_modules() -> None:
    model = nn.Sequential(nn.Linear(4, 4), _FakeTorchQuantLinear(4, 4))

    with pytest.raises(QuantizationError, match="exact selected module inventory"):
        optional._verify_gptqmodel_inventory(
            model,
            selected=("0",),
            quantized_type=_FakeTorchQuantLinear,
            backend="gptq",
        )


def test_gptqmodel_inventory_verifier_accepts_exact_selected_and_excluded_names() -> None:
    model = nn.Sequential(nn.Linear(4, 4), _FakeTorchQuantLinear(4, 4))

    observed = optional._verify_gptqmodel_inventory(
        model,
        selected=("1",),
        runtime_excluded=("0",),
        quantized_type=_FakeTorchQuantLinear,
        base_quantized_type=_FakeBaseQuantLinear,
        backend="gptq",
    )

    assert observed == ("1",)


def test_gptqmodel_inventory_verifier_rejects_missing_defuser_runtime_name() -> None:
    model = nn.Sequential(_FakeTorchQuantLinear(4, 4))

    with pytest.raises(QuantizationError, match="missing policy-realized runtime modules"):
        optional._verify_gptqmodel_inventory(
            model,
            selected=("0",),
            runtime_excluded=("missing.expert.down_proj",),
            quantized_type=_FakeTorchQuantLinear,
            base_quantized_type=_FakeBaseQuantLinear,
            backend="gptq",
        )


def test_gptqmodel_exclusion_audit_binds_exact_source_state() -> None:
    source = nn.Sequential(nn.Linear(4, 4))
    candidate = copy.deepcopy(source)

    audit = optional._audit_gptqmodel_exclusions(
        source,
        candidate,
        ("0",),
        source_dtype=torch.float32,
        backend="gptq",
    )

    assert audit.module_count == 1
    assert audit.state_tensor_count == 2
    assert audit.expanded_linear_count == 0
    assert len(audit.state_sha256) == 64


def test_gptqmodel_exclusion_audit_rejects_changed_protected_tensor() -> None:
    source = nn.Sequential(nn.Linear(4, 4))
    candidate = copy.deepcopy(source)
    with torch.no_grad():
        candidate[0].weight[0, 0].add_(1.0)

    with pytest.raises(QuantizationError, match="changed protected runtime tensor"):
        optional._audit_gptqmodel_exclusions(
            source,
            candidate,
            ("0",),
            source_dtype=torch.float32,
            backend="gptq",
        )


def test_module_state_digest_binds_values_shapes_and_dtypes() -> None:
    first = nn.Sequential(nn.Linear(4, 4))
    second = copy.deepcopy(first)

    assert optional._module_state_sha256(first) == optional._module_state_sha256(second)
    with torch.no_grad():
        second[0].weight[0, 0].add_(1.0)
    assert optional._module_state_sha256(first) != optional._module_state_sha256(second)
    assert optional._module_state_sha256(first.double()) != optional._module_state_sha256(second)


def test_gptqmodel_cpu_post_init_failure_restores_compile_binding() -> None:
    class FailingTorchQuantLinear(_FakeTorchQuantLinear):
        def post_init(self) -> None:
            self.post_init_calls += 1
            torch_qlinear_module.torch_compile(lambda: None)
            raise RuntimeError("synthetic post-init failure")

    def original_compile(value: Any, **_kwargs: Any) -> Any:
        return value

    torch_qlinear_module = SimpleNamespace(torch_compile=original_compile)
    qlinear = FailingTorchQuantLinear(4, 4)
    qlinear.qzeros.add_(0x11111111)
    qlinear.qzero_format(2)
    module = nn.Sequential(qlinear)
    api = SimpleNamespace(
        TorchQLinearModule=torch_qlinear_module,
        TorchQuantLinear=FailingTorchQuantLinear,
    )

    with pytest.raises(QuantizationError, match="synthetic post-init failure"):
        optional._post_init_gptqmodel_580_cpu_torch(api, module, ("0",), "5.8.0")

    assert torch_qlinear_module.torch_compile is original_compile


def test_gptqmodel_580_awq_pack_shim_is_narrow_and_explicit() -> None:
    class AwqTorchQuantLinear:
        pass

    class AwqGEMMQuantLinear:
        def pack(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    api = SimpleNamespace(
        AwqTorchQuantLinear=AwqTorchQuantLinear,
        AwqGEMMQuantLinear=AwqGEMMQuantLinear,
    )

    applied = optional._apply_gptqmodel_580_awq_pack_shim(api, "5.8.0")

    assert applied
    assert AwqTorchQuantLinear.pack is AwqGEMMQuantLinear.pack
    optional._remove_gptqmodel_580_awq_pack_shim(api, applied)
    assert not hasattr(AwqTorchQuantLinear, "pack")


class _FakeConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.pack_dtype = kwargs.get("pack_dtype", torch.int32)


class _FakeBackend:
    TORCH_AWQ = SimpleNamespace(value="torch_awq")
    TORCH = SimpleNamespace(value="torch")


class _FakeMethod:
    AWQ = "awq"
    GPTQ = "gptq"


class _FakeBaseQuantLinear(nn.Linear):
    pass


_FAKE_TORCH_QLINEAR_MODULE = SimpleNamespace(torch_compile=lambda value, **_kwargs: value)


class _FakeTorchQuantLinear(_FakeBaseQuantLinear):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.optimized = False
        self.post_init_calls = 0
        self.enable_wf_unsqueeze = True
        self._qzero_format = 1
        self.register_buffer("qweight", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("qzeros", torch.zeros(1, dtype=torch.int32))
        self.register_buffer("scales", torch.ones(1, dtype=torch.float16))
        self.register_buffer("g_idx", torch.zeros(1, dtype=torch.int32))

    def qzero_format(self, value: int | None = None) -> int:
        if value is not None:
            self._qzero_format = value
        return self._qzero_format

    def post_init(self) -> None:
        self.post_init_calls += 1
        _FAKE_TORCH_QLINEAR_MODULE.torch_compile(lambda: None)
        self.register_buffer(
            "wf_unsqueeze_zero", torch.zeros(1, dtype=torch.int32), persistent=False
        )
        self.register_buffer(
            "wf_unsqueeze_neg_one", torch.zeros(1, dtype=torch.int32), persistent=False
        )
        self.optimized = True

    def _stream_reset_cache(self) -> None:
        return None

    def clear_weight_cache(self) -> None:
        return None

    def _reset_prefetch_state(self) -> None:
        return None


class _FakeAwqTorchQuantLinear(_FakeBaseQuantLinear):
    @classmethod
    def pack(cls, *_args: Any, **_kwargs: Any) -> None:
        return None


def _replace_with_fake_quant_linear(
    root: nn.Module,
    name: str,
    replacement_type: type[nn.Linear],
) -> None:
    original = root.get_submodule(name)
    assert isinstance(original, nn.Linear)
    replacement = replacement_type(
        original.in_features,
        original.out_features,
        bias=original.bias is not None,
    )
    replacement.weight.data.copy_(original.weight.data)
    if original.bias is not None and replacement.bias is not None:
        replacement.bias.data.copy_(original.bias.data)
    if replacement_type is _FakeTorchQuantLinear and original.bias is None:
        del replacement.bias
    parent_name, _, child_name = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    parent._modules[child_name] = replacement


class _FakeGPTQWrapper:
    def __init__(
        self,
        source: nn.Module,
        method: str,
        selected: tuple[str, ...],
        *,
        expect_python_packing: bool,
    ) -> None:
        self.model = copy.deepcopy(source)
        self.method = method
        self.selected = selected
        self.expect_python_packing = expect_python_packing
        self.quantized = False
        self.calibration: list[dict[str, list[int]]] | None = None
        self.load_kwargs: dict[str, Any] = {}

    def quantize(
        self,
        calibration: list[dict[str, list[int]]],
        *,
        batch_size: int,
        backend: Any,
        calibration_data_min_length: int,
    ) -> None:
        assert batch_size == 1
        assert backend in (_FakeBackend.TORCH_AWQ, _FakeBackend.TORCH)
        assert calibration_data_min_length == 10
        assert (os.environ.get("GPTQMODEL_DISABLE_PACK_EXT") == "1") is (self.expect_python_packing)
        self.calibration = calibration
        replacement = _FakeAwqTorchQuantLinear if self.method == "awq" else _FakeTorchQuantLinear
        for name in self.selected:
            _replace_with_fake_quant_linear(self.model, name, replacement)
        self.quantized = True

    def save(self, destination: str, *, max_shard_size: str) -> None:
        assert max_shard_size == "5GB"
        if self.method == "gptq" and self.expect_python_packing:
            assert all(self.model.get_submodule(name).qzero_format() == 1 for name in self.selected)
        path = Path(destination)
        (path / "model.safetensors").write_bytes(b"safe-weights")
        (path / "config.json").write_text(
            json.dumps(
                {
                    "quantization_config": {
                        "quant_method": self.method,
                        **(
                            {"format": "gptq", "checkpoint_format": "gptq"}
                            if self.method == "gptq"
                            else {}
                        ),
                    }
                }
            ),
            encoding="utf-8",
        )


class _FakeGPTQExportOwner:
    def __init__(self, module: nn.Module, *, fail: bool = False, mutate: bool = False) -> None:
        self.model = module
        self.fail = fail
        self.mutate = mutate

    def save(self, destination: str, *, max_shard_size: str) -> None:
        assert max_shard_size == "5GB"
        qlinear = self.model.get_submodule("0")
        assert qlinear.qzero_format() == 1
        if self.mutate:
            qlinear.qzeros.add_(1)
        if self.fail:
            raise RuntimeError("synthetic save failure")
        path = Path(destination)
        (path / "model.safetensors").write_bytes(b"safe-weights")
        (path / "config.json").write_text(
            json.dumps(
                {
                    "quantization_config": {
                        "quant_method": "gptq",
                        "format": "gptq",
                        "checkpoint_format": "gptq",
                    }
                }
            ),
            encoding="utf-8",
        )


def _gptq_export_failure_fixture(
    config_factory,
    *,
    fail: bool = False,
    mutate: bool = False,
) -> tuple[optional.GPTQModelQuantizer, QuantizedModel, QuantizationConfig, _FakeGPTQExportOwner]:
    source = _stories15m_source(config_factory)
    qlinear = _FakeTorchQuantLinear(4, 4)
    qlinear.qzeros.add_(0x11111111)
    qlinear.qzero_format(2)
    module = nn.Sequential(qlinear)
    api = SimpleNamespace(
        TorchQuantLinear=_FakeTorchQuantLinear,
        convert_gptq_v1_to_v2_format_module=lambda module, bits, pack_dtype: (
            module.qzeros.add_(0x11111111),
            module.qzero_format(2),
        ),
        convert_gptq_v2_to_v1_format_module=lambda module, quantize_config: (
            module.qzeros.sub_(0x11111111),
            module.qzero_format(1),
        ),
    )
    transition = optional._GPTQModelRuntimeTransition(
        api=api,
        module=module,
        quantize_config=SimpleNamespace(bits=4, pack_dtype=torch.int32),
        qlinear_names=("0",),
    )
    owner = _FakeGPTQExportOwner(module, fail=fail, mutate=mutate)
    manifest = QuantizationManifest(
        backend="gptq",
        backend_version="5.8.0",
        method="gptq",
        source_model_checksum=source.descriptor.checksum,
        module_precision_map={"0": "int4"},
        excluded_modules=(),
        quantization_parameters={
            "quantized_module_count": 1,
            "quantized_module_names_sha256": hashlib.sha256(b"0").hexdigest(),
            "candidate_state_sha256": optional._module_state_sha256(module),
            "excluded_state_sha256": hashlib.sha256(b"").hexdigest(),
            "software_matrix_sha256": hashlib.sha256(
                json.dumps(
                    optional._GPTQMODEL_CPU_EXACT_VERSIONS,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "accelerate_version": "1.14.0",
            "defuser_version": "0.0.23",
            "gptqmodel_version": "5.8.0",
            "huggingface_hub_version": "1.23.0",
            "kernels_version": "0.14.1",
            "safetensors_version": "0.8.0",
            "torch_version": "2.13.0",
            "transformers_version": "5.12.1",
        },
        serialized_size_bytes=1,
    )
    candidate = QuantizedModel(
        loaded=LoadedModel(
            model=module,
            tokenizer=source.tokenizer,
            descriptor=source.descriptor,
            load_time_seconds=0.01,
        ),
        manifest=manifest,
    )
    quantizer = optional.GPTQModelQuantizer("gptq")
    quantizer._exporters[id(module)] = optional._ExportState(
        owner=owner,
        save_kind="gptqmodel",
        runtime_transition=transition,
    )
    return quantizer, candidate, _quantization("gptq", calibration=True, device="cpu"), owner


@pytest.mark.parametrize(
    ("backend", "device", "python_packing"),
    (("awq", "cuda:0", False), ("gptq", "cuda:0", False), ("gptq", "cpu", True)),
)
def test_gptqmodel_adapter_invokes_safe_pinned_conversion_and_export(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
    tmp_path: Path,
    backend: str,
    device: str,
    python_packing: bool,
) -> None:
    monkeypatch.delenv("GPTQMODEL_DISABLE_PACK_EXT", raising=False)
    _patch_dependencies(monkeypatch, "gptqmodel")
    source = _stories15m_source(config_factory) if device == "cpu" else _source(config_factory)
    policy = _complete_linear_policy(source, "int4")
    selected = tuple(
        name for name, precision in policy.precision_map.items() if precision == "int4"
    )
    expected_bias_none_repairs = sum(
        source.model.get_submodule(name).bias is None for name in selected
    )
    wrapper = _FakeGPTQWrapper(
        source.model,
        backend,
        selected,
        expect_python_packing=python_packing,
    )
    snapshot = tmp_path / "pinned-source"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"safe-source")
    lifecycle_events: list[str] = []
    original_mixtral_class = object()
    patched_mixtral_class = object()
    mixtral_module = SimpleNamespace(MixtralSparseMoeBlock=original_mixtral_class)
    mixtral_bindings = object()
    monkeypatch.setattr(
        optional,
        "_capture_native_mixtral_class_binding",
        lambda: mixtral_bindings,
    )

    def restore_mixtral_bindings(value: Any) -> bool:
        assert value is mixtral_bindings
        mixtral_module.MixtralSparseMoeBlock = original_mixtral_class
        return True

    monkeypatch.setattr(
        optional,
        "_restore_native_mixtral_class_binding",
        restore_mixtral_bindings,
    )

    def materialize(_descriptor: Any, *, local_files_only: bool) -> Path:
        assert local_files_only
        lifecycle_events.append("source_audited")
        return snapshot

    monkeypatch.setattr(
        optional,
        "_materialize_pinned_source",
        materialize,
    )

    class FakeGPTQModel:
        @staticmethod
        def load(model_id: str, **kwargs: Any) -> _FakeGPTQWrapper:
            assert model_id == str(snapshot)
            mixtral_module.MixtralSparseMoeBlock = patched_mixtral_class
            wrapper.load_kwargs = kwargs
            return wrapper

    api = SimpleNamespace(
        BACKEND=_FakeBackend,
        FORMAT=SimpleNamespace(GEMM="gemm", GPTQ="gptq"),
        GPTQModel=FakeGPTQModel,
        METHOD=_FakeMethod,
        QuantizeConfig=_FakeConfig,
        AwqTorchQuantLinear=_FakeAwqTorchQuantLinear,
        BaseQuantLinear=_FakeBaseQuantLinear,
        TorchQLinearModule=_FAKE_TORCH_QLINEAR_MODULE,
        TorchQuantLinear=_FakeTorchQuantLinear,
        convert_gptq_v1_to_v2_format_module=lambda module, bits, pack_dtype: (
            module.qzeros.add_(0x11111111),
            module.qzero_format(2),
        ),
        convert_gptq_v2_to_v1_format_module=lambda module, quantize_config: (
            module.qzeros.sub_(0x11111111),
            module.qzero_format(1),
        ),
    )
    original_torch_compile = api.TorchQLinearModule.torch_compile

    def load_api() -> Any:
        lifecycle_events.append("gptqmodel_imported")
        return api

    monkeypatch.setattr(optional, "_load_gptqmodel_api", load_api)
    quantizer = optional.GPTQModelQuantizer(backend)
    config = _quantization(backend, calibration=True, device=device)
    runtime = _cpu_runtime() if device == "cpu" else _runtime()
    assert quantizer.check_support(source.descriptor, runtime, config).supported
    calibration = quantizer.calibrate(source, _calibration_batch(source), config)

    candidate = quantizer.quantize_from_pinned_source(source, policy, calibration, config)
    export = quantizer.export(candidate, tmp_path / backend, config)

    assert wrapper.quantized
    assert os.environ.get("GPTQMODEL_DISABLE_PACK_EXT") is None
    assert mixtral_module.MixtralSparseMoeBlock is original_mixtral_class
    assert lifecycle_events == ["source_audited", "gptqmodel_imported"]
    assert wrapper.calibration is not None
    assert "revision" not in wrapper.load_kwargs
    assert "device" not in wrapper.load_kwargs
    assert "device_map" not in wrapper.load_kwargs
    assert wrapper.load_kwargs["trust_remote_code"] is False
    assert wrapper.load_kwargs["use_safetensors"] is True
    assert wrapper.load_kwargs["local_files_only"] is True
    assert candidate.loaded.load_time_kind == "candidate_pinned_source_quantization"
    assert candidate.manifest.calibration_sample_ids == ("calibration-001",)
    assert calibration is not None
    assert candidate.manifest.calibration_checksum == calibration.checksum
    assert candidate.manifest.calibration_sample_count == int(
        calibration.statistics["sample_count"]
    )
    assert candidate.manifest.calibration_token_count == int(calibration.statistics["token_count"])
    assert candidate.manifest.quantization_parameters["candidate_lifecycle"] == (
        "pinned_source_reload"
    )
    assert len(candidate.manifest.quantization_parameters["candidate_state_sha256"]) == 64
    assert candidate.manifest.quantization_parameters["excluded_state_tensor_count"] > 0
    assert candidate.manifest.quantization_parameters["excluded_expanded_linear_count"] == 0
    assert candidate.manifest.quantization_parameters["excluded_state_verified_equal_to_source"]
    assert {
        name: candidate.manifest.quantization_parameters[f"{name}_version"]
        for name in (
            "accelerate",
            "defuser",
            "gptqmodel",
            "huggingface_hub",
            "kernels",
            "safetensors",
            "torch",
            "transformers",
        )
    } == {
        "accelerate": "1.14.0",
        "defuser": "0.0.23",
        "gptqmodel": "5.8.0",
        "huggingface_hub": "1.23.0",
        "kernels": "0.14.1",
        "safetensors": "0.8.0",
        "torch": "2.13.0",
        "transformers": "5.12.1",
    }
    assert len(candidate.manifest.quantization_parameters["software_matrix_sha256"]) == 64
    assert candidate.manifest.quantization_parameters["quantized_module_count"] == 1
    assert candidate.manifest.quantization_parameters["python_packing"] is python_packing
    assert candidate.manifest.quantization_parameters["cpu_eager_post_init"] is (
        backend == "gptq" and device == "cpu"
    )
    assert candidate.manifest.quantization_parameters["cpu_eager_post_init_module_count"] == (
        1 if backend == "gptq" and device == "cpu" else 0
    )
    assert candidate.manifest.quantization_parameters["cpu_post_init_compile_bypass_count"] == (
        1 if backend == "gptq" and device == "cpu" else 0
    )
    assert candidate.manifest.quantization_parameters["cpu_bias_none_repair_count"] == (
        expected_bias_none_repairs if backend == "gptq" and device == "cpu" else 0
    )
    assert candidate.manifest.quantization_parameters["runtime_qzero_conversion_count"] == (
        1 if backend == "gptq" and device == "cpu" else 0
    )
    assert candidate.manifest.quantization_parameters[
        "torch_compile_bypassed_during_cpu_post_init"
    ] is (backend == "gptq" and device == "cpu")
    selected_module = candidate.loaded.model.get_submodule(selected[0])
    if backend == "gptq":
        assert isinstance(selected_module, _FakeTorchQuantLinear)
        assert selected_module.post_init_calls == (1 if device == "cpu" else 0)
        if device == "cpu":
            assert selected_module.qzero_format() == 2
            assert (selected_module.bias is None) is (
                source.model.get_submodule(selected[0]).bias is None
            )
            assert torch.equal(selected_module.qzeros, torch.tensor([0x11111111]))
    assert api.TorchQLinearModule.torch_compile is original_torch_compile
    assert candidate.manifest.quantization_parameters["defuser_global_patch_restored"] is True
    assert candidate.manifest.quantization_parameters["policy_realization"] == (
        "exact_selected_qlinear_inventory"
    )
    pipeline_manifest = QuantizationManifest.model_validate(
        candidate.manifest.model_dump(mode="json")
    )
    assert pipeline_manifest.calibration_checksum == calibration.checksum
    assert pipeline_manifest.calibration_sample_count == 1
    assert pipeline_manifest.calibration_token_count == int(calibration.statistics["token_count"])
    assert export.format == "safetensors"
    assert export.reload_recipe["trust_remote_code"] is False
    if backend == "gptq" and device == "cpu":
        assert export.reload_recipe["adapter"] == "inkling_quant_lab_gptqmodel_cpu"
        assert export.reload_recipe["loader"].endswith("reload_gptqmodel_cpu_export")
        assert export.reload_recipe["checkpoint_qzero_format"] == "gptq_v1"
        assert export.reload_recipe["runtime_qzero_format"] == "gptq_v2"
        assert export.reload_recipe["policy_preserving_reload_supported"] is True
        exported_config = Path(export.path) / "config.json"
        assert export.reload_recipe["config_file"] == "config.json"
        assert export.reload_recipe["config_file_sha256"] == optional._file_sha256(exported_config)
        assert export.reload_recipe["config_file_size_bytes"] == exported_config.stat().st_size
        assert (
            candidate.manifest.quantization_parameters["export_config_sha256"]
            == (export.reload_recipe["config_file_sha256"])
        )
        assert candidate.manifest.quantization_parameters["export_config_size_bytes"] == (
            exported_config.stat().st_size
        )
    else:
        assert export.reload_recipe["adapter"] == "gptqmodel"
        assert export.reload_recipe["loader"] == "gptqmodel.GPTQModel.load"
        assert export.reload_recipe["device_map"] == {"": device}
        assert export.reload_recipe["policy_preserving_reload_supported"] is False
    persisted_reload = json.loads(
        (Path(export.path) / "inkling_quant_reload.json").read_text(encoding="utf-8")
    )
    assert persisted_reload == export.reload_recipe
    exported_manifest_path = Path(export.path) / "inkling_quant_manifest.json"
    exported_manifest = json.loads(exported_manifest_path.read_text(encoding="utf-8"))
    assert exported_manifest["calibration_checksum"] == calibration.checksum
    assert exported_manifest["calibration_sample_count"] == 1
    assert exported_manifest["calibration_token_count"] == int(
        calibration.statistics["token_count"]
    )
    serialized_export = json.dumps(exported_manifest)
    assert "input_ids" not in serialized_export
    assert "attention_mask" not in serialized_export
    assert "alpha beta" not in serialized_export


def test_gptqmodel_cpu_reload_rejects_behavior_config_tampering(
    config_factory,
    tmp_path: Path,
) -> None:
    quantizer, candidate, config, _owner = _gptq_export_failure_fixture(config_factory)
    destination = tmp_path / "candidate"
    export = quantizer.export(candidate, destination, config)
    recipe = cast(dict[str, Any], export.reload_recipe)

    optional._validate_gptqmodel_cpu_export_config(destination, recipe, candidate.manifest)
    config_path = destination / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["num_experts_per_tok"] = 1
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(QuantizationError, match=r"config\.json checksum or size"):
        optional._validate_gptqmodel_cpu_export_config(destination, recipe, candidate.manifest)


def test_gptqmodel_rejects_source_mutation_after_descriptor_creation(config_factory) -> None:
    source = _stories15m_source(config_factory)
    optional._validate_loaded_source_state(source, backend="gptq")
    assert isinstance(source.model, nn.Module)
    parameter = next(source.model.parameters())
    with torch.inference_mode():
        parameter.view(-1)[0].add_(1.0)

    with pytest.raises(QuantizationError, match="immutable descriptor checksum"):
        optional._validate_loaded_source_state(source, backend="gptq")


@pytest.mark.parametrize(
    ("precision_map", "excluded"),
    (
        ({"selected": "int4", "protected": "float32"}, ()),
        ({"selected": "int4", "protected": "int8"}, ("protected",)),
        ({"selected": "int4", "protected": "float32"}, ("selected", "protected")),
    ),
)
def test_gptqmodel_cpu_manifest_partition_rejects_exclusion_drift(
    config_factory,
    precision_map: dict[str, str],
    excluded: tuple[str, ...],
) -> None:
    source = _stories15m_source(config_factory)
    manifest = QuantizationManifest(
        backend="gptq",
        backend_version="5.8.0",
        method="gptq",
        source_model_checksum=source.descriptor.checksum,
        module_precision_map=precision_map,
        excluded_modules=excluded,
        serialized_size_bytes=1,
    )

    with pytest.raises(QuantizationError, match="INT4/float32 policy partition"):
        optional._gptqmodel_cpu_manifest_partition(manifest)


def test_gptqmodel_conversion_failure_restores_environment_and_mixtral_binding(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GPTQMODEL_DISABLE_PACK_EXT", "sentinel")
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", "original-allocator")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "original-order")
    _patch_dependencies(monkeypatch, "gptqmodel")
    source = _stories15m_source(config_factory)
    policy = _complete_linear_policy(source, "int4")
    selected = tuple(
        name for name, precision in policy.precision_map.items() if precision == "int4"
    )
    wrapper = _FakeGPTQWrapper(
        source.model,
        "gptq",
        selected,
        expect_python_packing=True,
    )
    snapshot = tmp_path / "pinned-source"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"safe-source")
    binding = object()
    restored: list[object] = []
    monkeypatch.setattr(optional, "_capture_native_mixtral_class_binding", lambda: binding)
    monkeypatch.setattr(
        optional,
        "_restore_native_mixtral_class_binding",
        lambda value: restored.append(value) is None or True,
    )
    monkeypatch.setattr(
        optional,
        "_materialize_pinned_source",
        lambda _descriptor, *, local_files_only: snapshot,
    )

    class FakeGPTQModel:
        @staticmethod
        def load(_model_id: str, **_kwargs: Any) -> _FakeGPTQWrapper:
            return wrapper

    api = SimpleNamespace(
        BACKEND=_FakeBackend,
        FORMAT=SimpleNamespace(GPTQ="gptq"),
        GPTQModel=FakeGPTQModel,
        METHOD=_FakeMethod,
        QuantizeConfig=_FakeConfig,
    )

    def load_api() -> Any:
        os.environ["PYTORCH_ALLOC_CONF"] = "mutated-allocator"
        os.environ["CUDA_DEVICE_ORDER"] = "mutated-order"
        return api

    def fail_quantize(*_args: Any, **_kwargs: Any) -> None:
        assert os.environ["GPTQMODEL_DISABLE_PACK_EXT"] == "1"
        raise RuntimeError("synthetic quantization failure")

    wrapper.quantize = fail_quantize  # type: ignore[method-assign]
    monkeypatch.setattr(optional, "_load_gptqmodel_api", load_api)
    quantizer = optional.GPTQModelQuantizer("gptq")
    config = _quantization("gptq", calibration=True, device="cpu")
    assert quantizer.check_support(source.descriptor, _cpu_runtime(), config).supported
    calibration = quantizer.calibrate(source, _calibration_batch(source), config)

    with pytest.raises(QuantizationError, match="synthetic quantization failure"):
        quantizer.quantize_from_pinned_source(source, policy, calibration, config)

    assert os.environ["GPTQMODEL_DISABLE_PACK_EXT"] == "sentinel"
    assert os.environ["PYTORCH_ALLOC_CONF"] == "original-allocator"
    assert os.environ["CUDA_DEVICE_ORDER"] == "original-order"
    assert restored == [binding]
    assert not quantizer._exporters


def test_gptqmodel_awq_failure_removes_temporary_pack_binding(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
    tmp_path: Path,
) -> None:
    _patch_dependencies(monkeypatch, "gptqmodel")
    source = _source(config_factory)
    policy = _complete_linear_policy(source, "int4")
    selected = tuple(
        name for name, precision in policy.precision_map.items() if precision == "int4"
    )
    wrapper = _FakeGPTQWrapper(
        source.model,
        "awq",
        selected,
        expect_python_packing=False,
    )
    snapshot = tmp_path / "pinned-source"
    snapshot.mkdir()
    (snapshot / "model.safetensors").write_bytes(b"safe-source")
    monkeypatch.setattr(optional, "_capture_native_mixtral_class_binding", object)
    monkeypatch.setattr(optional, "_restore_native_mixtral_class_binding", lambda _value: True)
    monkeypatch.setattr(
        optional,
        "_materialize_pinned_source",
        lambda _descriptor, *, local_files_only: snapshot,
    )

    class AwqTorchQuantLinear:
        pass

    class AwqGEMMQuantLinear:
        def pack(self, *_args: Any, **_kwargs: Any) -> None:
            return None

    class FakeGPTQModel:
        @staticmethod
        def load(_model_id: str, **_kwargs: Any) -> _FakeGPTQWrapper:
            return wrapper

    api = SimpleNamespace(
        AwqGEMMQuantLinear=AwqGEMMQuantLinear,
        AwqTorchQuantLinear=AwqTorchQuantLinear,
        BACKEND=_FakeBackend,
        FORMAT=SimpleNamespace(GEMM="gemm"),
        GPTQModel=FakeGPTQModel,
        METHOD=_FakeMethod,
        QuantizeConfig=_FakeConfig,
    )
    wrapper.quantize = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RuntimeError("synthetic AWQ failure")
    )
    monkeypatch.setattr(optional, "_load_gptqmodel_api", lambda: api)
    quantizer = optional.GPTQModelQuantizer("awq")
    config = _quantization("awq", calibration=True, device="cuda:0")
    assert quantizer.check_support(source.descriptor, _runtime(), config).supported
    calibration = quantizer.calibrate(source, _calibration_batch(source), config)

    with pytest.raises(QuantizationError, match="synthetic AWQ failure"):
        quantizer.quantize_from_pinned_source(source, policy, calibration, config)

    assert not hasattr(AwqTorchQuantLinear, "pack")


def test_gptqmodel_failed_save_repairs_runtime_qzeros_and_allows_retry(
    config_factory,
    tmp_path: Path,
) -> None:
    quantizer, candidate, config, owner = _gptq_export_failure_fixture(
        config_factory,
        fail=True,
    )
    qlinear = candidate.loaded.model.get_submodule("0")
    before = qlinear.qzeros.detach().clone()
    destination = tmp_path / "candidate"

    with pytest.raises(QuantizationError, match="synthetic save failure"):
        quantizer.export(candidate, destination, config)

    assert not destination.exists()
    assert qlinear.qzero_format() == 2
    assert torch.equal(qlinear.qzeros, before)
    owner.fail = False
    artifact = quantizer.export(candidate, destination, config)
    assert artifact.size_bytes > 0


def test_gptqmodel_mutating_save_is_refused_and_candidate_is_repaired(
    config_factory,
    tmp_path: Path,
) -> None:
    quantizer, candidate, config, owner = _gptq_export_failure_fixture(
        config_factory,
        mutate=True,
    )
    qlinear = candidate.loaded.model.get_submodule("0")
    before = qlinear.qzeros.detach().clone()
    destination = tmp_path / "candidate"

    with pytest.raises(QuantizationError, match="exact runtime qzero tensors"):
        quantizer.export(candidate, destination, config)

    assert not destination.exists()
    assert qlinear.qzero_format() == 2
    assert torch.equal(qlinear.qzeros, before)
    owner.mutate = False
    assert quantizer.export(candidate, destination, config).size_bytes > 0


class _FakeFP8Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.is_quantized = True

    def save_pretrained(
        self, destination: str, *, safe_serialization: bool, max_shard_size: str
    ) -> None:
        assert safe_serialization
        assert max_shard_size == "5GB"
        path = Path(destination)
        (path / "model.safetensors").write_bytes(b"fp8-safe-weights")
        (path / "config.json").write_text(
            json.dumps({"quantization_config": {"quant_method": "fp8"}}),
            encoding="utf-8",
        )


def test_fp8_adapter_invokes_transformers_config_with_protected_exclusions(
    monkeypatch: pytest.MonkeyPatch,
    config_factory,
    tmp_path: Path,
) -> None:
    _patch_dependencies(monkeypatch, "fp8")
    checked_devices: list[str] = []

    def compute_capability(device: str) -> tuple[int, int]:
        checked_devices.append(device)
        return (9, 0)

    monkeypatch.setattr(optional, "_cuda_compute_capability", compute_capability)
    source = _source(config_factory)
    calls: dict[str, Any] = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: Any) -> _FakeFP8Model:
            calls["model_id"] = model_id
            calls.update(kwargs)
            return _FakeFP8Model()

    class FakeFP8Config(_FakeConfig):
        pass

    monkeypatch.setattr(
        optional,
        "_load_transformers_fp8_api",
        lambda: SimpleNamespace(
            AutoModelForCausalLM=FakeAutoModel,
            FineGrainedFP8Config=FakeFP8Config,
        ),
    )
    quantizer = optional.create_fp8_quantizer()
    config = _quantization("fp8", calibration=False, device="cuda:1")
    assert quantizer.check_support(source.descriptor, _runtime(), config).supported

    candidate = quantizer.quantize_from_pinned_source(source, _policy(source, "fp8"), None, config)
    export = quantizer.export(candidate, tmp_path / "fp8", config)

    assert calls["model_id"] == source.descriptor.model_id
    assert calls["revision"] == _REVISION
    assert calls["trust_remote_code"] is False
    assert calls["use_safetensors"] is True
    assert checked_devices == ["cuda:1", "cuda:1"]
    assert calls["device_map"] == {"": "cuda:1"}
    exclusions = calls["quantization_config"].kwargs["modules_to_not_convert"]
    assert exclusions == ["lm_head", "moe_layers.0.router"]
    assert candidate.manifest.excluded_modules == tuple(exclusions)
    assert export.reload_recipe["backend"] == "fp8"


def test_non_addressable_exclusion_fails_before_upstream_invocation(config_factory) -> None:
    source = _source(config_factory)
    policy = SimpleNamespace(
        precision_map={"moe_layers.0.experts.synthetic": "float32", "lm_head": "int4"}
    )

    with pytest.raises(CapabilityError, match="non-addressable"):
        optional._policy_partition(source, policy, "int4")
