from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from inkling_quant_lab.config import ExperimentConfig, QuantizationConfig
from inkling_quant_lab.mlx_contract import (
    CANONICAL_MLX_MODEL_CARD_BYTES,
    MLX_CONVERSION_CONTRACTS,
    STORIES15M_SOURCE_CONTRACT,
    audit_source_snapshot,
    expected_quantized_leaf_names,
    validate_quantized_tensor_facts,
)
from inkling_quant_lab.models.base import ModelCapabilities, ModelDescriptor
from inkling_quant_lab.quantization.base import PinnedSourceReloadQuantizer
from inkling_quant_lab.quantization.mlx import MLXAffineQuantizer
from inkling_quant_lab.runtimes.base import RuntimeCapabilities

pytestmark = pytest.mark.unit


def _mlx_config(snapshot: str = "model-cache/stories15m") -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "mlx-stories15m-q8",
            "model": {
                "model_id": "ggml-org/stories15M_MOE",
                "revision": "b6dd737497465570b5f5e962dbc9d9454ed1e0eb",
                "adapter": "mlx_lm_mixtral",
                "local_snapshot_path": snapshot,
                "dtype": "float32",
                "local_files_only": True,
                "checkpoint_format": "safetensors",
            },
            "quantization": {
                "backend": "mlx_affine",
                "method": "mlx_affine",
                "policy": {
                    "type": "uniform",
                    "default_precision": "float32",
                    "module_class_rules": [
                        {
                            "name": "mlx-linear-q8",
                            "pattern": "mlx.nn.layers.linear.Linear",
                            "precision": "int8",
                        },
                        {
                            "name": "mlx-embedding-q8",
                            "pattern": "mlx.nn.layers.embedding.Embedding",
                            "precision": "int8",
                        },
                        {
                            "name": "mlx-switch-q8",
                            "pattern": "mlx_lm.models.switch_layers.SwitchLinear",
                            "precision": "int8",
                        },
                    ],
                    "preserve_router_precision": False,
                    "preserve_output_head": False,
                    "preserve_embeddings": False,
                },
                "export": {"enabled": True, "format": "safetensors"},
                "parameters": {"bits": 8, "group_size": 32, "mode": "affine"},
            },
            "evaluation": {
                "suites": [
                    {
                        "type": "perplexity",
                        "dataset": "local://fixtures/tiny-corpus",
                        "revision": "fixture-data-v1",
                        "split": "evaluation",
                    }
                ]
            },
            "routing": {"mode": "off"},
            "benchmark": {"enabled": False},
            "runtime": {
                "backend": "mlx_metal",
                "device": "mps",
                "dtype": "float32",
                "device_map": "single",
            },
            "reporting": {"markdown": False, "html": False, "plots": False},
        }
    )


def _descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        model_id="ggml-org/stories15M_MOE",
        revision="b6dd737497465570b5f5e962dbc9d9454ed1e0eb",
        resolved_class="mlx_lm.models.mixtral.Model",
        architecture="MixtralForCausalLM",
        checksum="dbfa0289f68a8dd721d10eb12d8bd82e098455682027f6f9986ba548913f9082",
        capabilities=ModelCapabilities(
            supports_text=True,
            supports_images=False,
            supports_audio=False,
            is_moe=True,
            supports_router_logits=True,
            supports_token_level_routes=True,
            supported_dtypes=("float32",),
            supported_device_maps=("single",),
            max_context_length=256,
        ),
    )


def _runtime() -> RuntimeCapabilities:
    return RuntimeCapabilities(
        backend="mlx_metal",
        available=True,
        devices=("mps",),
        supported_dtypes=("float32",),
        supports_routing_hooks=True,
        supports_forward_loss=True,
        supports_memory_measurement=True,
        supports_energy_measurement=False,
        supports_sharding=False,
        version="mlx=0.32.0;mlx-lm=0.31.3",
    )


def _quantized_facts(bits: int) -> dict[str, tuple[str, tuple[int, ...], int]]:
    facts: dict[str, tuple[str, tuple[int, ...], int]] = {}
    for name in expected_quantized_leaf_names():
        shape: tuple[int, ...]
        aux: tuple[int, ...]
        if ".switch_mlp." in name:
            projection = name.rpartition(".")[2]
            output_dims, input_dims = {
                "gate_proj": (768, 288),
                "up_proj": (768, 288),
                "down_proj": (288, 768),
            }[projection]
            shape = (4, output_dims, input_dims * bits // 32)
            aux = (4, output_dims, input_dims // 32)
        elif name in {"lm_head", "model.embed_tokens"}:
            shape = (32000, 288 * bits // 32)
            aux = (32000, 288 // 32)
        elif name.endswith(".gate"):
            shape = (4, 288 * bits // 32)
            aux = (4, 288 // 32)
        else:
            shape = (288, 288 * bits // 32)
            aux = (288, 288 // 32)
        facts[f"{name}.weight"] = ("uint32", shape, 4)
        facts[f"{name}.scales"] = ("float32", aux, 4)
        facts[f"{name}.biases"] = ("float32", aux, 4)
    return facts


def test_mlx_registry_metadata_is_lazy_and_has_a_real_extra() -> None:
    script = textwrap.dedent(
        """
        import sys
        from inkling_quant_lab.bootstrap import register_builtins
        from inkling_quant_lab.registry import MODEL_ADAPTERS, QUANTIZERS, RUNTIMES

        register_builtins()
        assert "mlx" not in sys.modules
        assert "mlx_lm" not in sys.modules
        descriptors = {
            descriptor.name: descriptor
            for registry in (MODEL_ADAPTERS, QUANTIZERS, RUNTIMES)
            for descriptor in registry.descriptors()
        }
        for name in ("mlx_lm_mixtral", "mlx_affine", "mlx_metal"):
            assert descriptors[name].optional_extra == "mlx"
            assert descriptors[name].available is None
        assert "mlx" not in sys.modules
        assert "mlx_lm" not in sys.modules
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_mlx_configuration_requires_the_exact_composable_triple() -> None:
    config = _mlx_config()
    assert config.model.local_snapshot_path == "model-cache/stories15m"

    payload = config.model_dump(mode="json")
    payload["runtime"]["backend"] = "torch_eager_mps"
    with pytest.raises(ValueError, match=r"mlx_lm_mixtral.*mlx_metal.*mlx_affine"):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model", "revision"), "unpinned", r"limited to the pinned"),
        (("model", "local_snapshot_path"), None, r"local_snapshot_path"),
        (("model", "local_files_only"), False, r"offline safetensors snapshot"),
        (("model", "tokenizer_id"), "other/tokenizer", r"tokenizer in the pinned snapshot"),
        (("runtime", "device"), "cpu", r"device=mps"),
        (("quantization", "method"), "not-mlx", r"method=mlx_affine"),
        (
            ("quantization", "calibration"),
            {
                "dataset": "local://fixtures/calibration",
                "revision": "fixture-data-v1",
                "split": "calibration",
            },
            r"calibration-free",
        ),
        (("quantization", "export", "format"), "recipe_json", r"only safetensors"),
        (("benchmark", "enabled"), True, r"benchmark is not implemented"),
    ],
)
def test_mlx_configuration_rejects_values_outside_the_registered_contract(
    path: tuple[str, ...], value: object, message: str
) -> None:
    payload = _mlx_config().model_dump(mode="json")
    target = payload
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=message):
        ExperimentConfig.model_validate(payload)


def test_mlx_configuration_rejects_expert_aware_policy() -> None:
    payload = _mlx_config().model_dump(mode="json")
    payload["quantization"]["policy"].update(
        {
            "type": "frequency_tiered",
            "frequency": {"quantiles": [0.5], "precisions": ["int4", "int8"]},
        }
    )

    with pytest.raises(ValueError, match=r"only a uniform policy"):
        ExperimentConfig.model_validate(payload)


def test_mlx_quantizer_exposes_the_pinned_source_reload_lifecycle() -> None:
    quantizer = MLXAffineQuantizer()

    assert quantizer.lifecycle == "pinned_source_reload"
    assert isinstance(quantizer, PinnedSourceReloadQuantizer)


def test_mlx_model_card_is_normalized_to_the_frozen_conversion_bytes() -> None:
    expected_size, expected_sha256 = MLX_CONVERSION_CONTRACTS[8].files["README.md"]

    assert len(CANONICAL_MLX_MODEL_CARD_BYTES) == expected_size
    assert hashlib.sha256(CANONICAL_MLX_MODEL_CARD_BYTES).hexdigest() == expected_sha256


@pytest.mark.parametrize(("bits", "precision"), [(4, "int4"), (8, "int8")])
def test_mlx_quantizer_support_is_exact_and_does_not_call_optional_runtime(
    monkeypatch: pytest.MonkeyPatch, bits: int, precision: str
) -> None:
    from inkling_quant_lab import mlx_contract
    from inkling_quant_lab.quantization import mlx as mlx_quantization

    monkeypatch.setattr(
        mlx_quantization,
        "mlx_environment_status",
        lambda: mlx_contract.MLXEnvironmentStatus(
            available=True,
            versions=dict(mlx_contract.EXPECTED_MLX_VERSIONS),
            reasons=(),
        ),
    )
    config = _mlx_config().quantization.model_copy(
        update={
            "parameters": {"bits": bits, "group_size": 32, "mode": "affine"},
            "policy": _mlx_config().quantization.policy.model_copy(
                update={
                    "module_class_rules": tuple(
                        rule.model_copy(update={"precision": precision})
                        for rule in _mlx_config().quantization.policy.module_class_rules
                    )
                }
            ),
        }
    )

    report = MLXAffineQuantizer().check_support(_descriptor(), _runtime(), config)

    assert report.available and report.supported
    assert report.supported_precisions == ("float32", "int4", "int8")
    assert report.install_extra == "mlx"


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_tensor_contract_covers_all_fused_expert_projections(bits: int) -> None:
    proof = validate_quantized_tensor_facts(_quantized_facts(bits), bits=bits)
    assert proof["quantized_leaf_count"] == 50
    assert proof["quantized_fused_expert_projection_count"] == 18
    assert proof["bits"] == bits


def test_source_snapshot_audit_precedes_optional_import_and_rejects_code(tmp_path: Path) -> None:
    snapshot = tmp_path / "cache" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    config = json.dumps(
        {
            "architectures": ["MixtralForCausalLM"],
            "model_type": "mixtral",
            "model_file": None,
            "auto_map": None,
        },
        sort_keys=True,
    ).encode()
    tokenizer_config = json.dumps({"tokenizer_class": "LlamaTokenizer"}).encode()
    tokenizer = b'{"version":"1.0"}'
    weight = b"safetensors-placeholder"
    files = {
        "config.json": config,
        "tokenizer_config.json": tokenizer_config,
        "tokenizer.json": tokenizer,
        "model.safetensors": weight,
    }
    for name, payload in files.items():
        (snapshot / name).write_bytes(payload)
    contract = replace(
        STORIES15M_SOURCE_CONTRACT,
        revision="revision",
        file_names=frozenset(files),
        config_sha256=hashlib.sha256(config).hexdigest(),
        tokenizer_config_sha256=hashlib.sha256(tokenizer_config).hexdigest(),
        tokenizer_sha256=hashlib.sha256(tokenizer).hexdigest(),
        weight_sha256=hashlib.sha256(weight).hexdigest(),
        weight_size_bytes=len(weight),
        tensor_count=None,
    )

    result = audit_source_snapshot(snapshot, contract=contract)
    assert result["python_file_count"] == 0
    assert result["weight_sha256"] == hashlib.sha256(weight).hexdigest()

    (snapshot / "modeling.py").write_text("raise RuntimeError", encoding="utf-8")
    with pytest.raises(ValueError, match=r"unsafe|unsupported"):
        audit_source_snapshot(snapshot, contract=contract)


def test_invalid_mlx_quantization_parameters_fail_closed() -> None:
    quantizer = MLXAffineQuantizer()
    base = _mlx_config().quantization
    for parameters in (
        {"bits": 6, "group_size": 32, "mode": "affine"},
        {"bits": 8, "group_size": 64, "mode": "affine"},
        {"bits": 8, "group_size": 32, "mode": "mxfp8"},
        {"bits": True, "group_size": 32, "mode": "affine"},
    ):
        config = QuantizationConfig.model_validate(
            {**base.model_dump(mode="json"), "parameters": parameters}
        )
        report = quantizer.check_support(_descriptor(), _runtime(), config)
        assert not report.supported
        assert report.reasons
