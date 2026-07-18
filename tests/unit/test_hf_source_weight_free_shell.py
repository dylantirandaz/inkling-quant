from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.models.base import ExportedModelIdentity
from inkling_quant_lab.models.hf_causal_lm import HFCausalLMAdapter
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.unit

_MODEL_ID = "example/tiny-offline-mixtral"
_REVISION = "4" * 40
_RESOLVED_CLASS = "transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM"
_ARCHITECTURE = "MixtralForCausalLM"
_SOURCE_CHECKSUM = "5" * 64
_FILENAMES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


class _RawTokenizer:
    vocab_size = 32
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "</s>"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del text, add_special_tokens
        return [1, 2]

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return " ".join(str(token) for token in token_ids)


def _hf_config(config_factory: Callable[..., ExperimentConfig]) -> ExperimentConfig:
    base = config_factory(backend="noop", precision="float32", routing_mode="off")
    return base.model_copy(
        update={
            "model": base.model.model_copy(
                update={
                    "model_id": _MODEL_ID,
                    "revision": _REVISION,
                    "adapter": "hf_causal_lm",
                    "checkpoint_format": "safetensors",
                    "local_files_only": True,
                }
            )
        }
    )


def _identity() -> ExportedModelIdentity:
    return ExportedModelIdentity(
        model_id=_MODEL_ID,
        revision=_REVISION,
        resolved_class=_RESOLVED_CLASS,
        architecture=_ARCHITECTURE,
        source_checksum=_SOURCE_CHECKSUM,
    )


def test_hf_empty_export_shell_constructs_meta_model_without_source_weights_and_rebuilds_rope(
    config_factory: Callable[..., ExperimentConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    transformers = pytest.importorskip("transformers")
    from transformers import MixtralConfig

    resolved = MixtralConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_local_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=32,
        architectures=[_ARCHITECTURE],
        torch_dtype="float32",
    )
    hashes = tuple((name, hashlib.sha256(name.encode()).hexdigest()) for name in _FILENAMES)
    adapter = HFCausalLMAdapter()
    monkeypatch.setattr(
        adapter,
        "_load_offline_config_and_tokenizer",
        lambda _config: (resolved, _RawTokenizer(), hashes),
    )

    def forbidden_from_pretrained(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("empty export shell must not load source model weights")

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        forbidden_from_pretrained,
    )
    shell = adapter.load_empty_export_shell(
        _hf_config(config_factory),
        TorchEagerCPURuntime(),
        _identity(),
    )

    assert type(shell.model).__name__ == _ARCHITECTURE
    assert shell.descriptor.model_id == _MODEL_ID
    assert shell.descriptor.revision == _REVISION
    assert shell.descriptor.checksum == _SOURCE_CHECKSUM
    assert shell.source_metadata_file_sha256 == hashes
    assert shell.model.state_dict()
    assert all(tensor.is_meta for tensor in shell.model.state_dict().values())
    rotary = {
        name: buffer
        for name, buffer in shell.model.named_buffers()
        if name.startswith("model.rotary_emb.")
    }
    assert set(rotary) == {
        "model.rotary_emb.inv_freq",
        "model.rotary_emb.original_inv_freq",
    }
    assert all(
        buffer.device.type == "cpu" and torch.isfinite(buffer).all() for buffer in rotary.values()
    )


def test_hf_source_metadata_resolver_hashes_only_exact_cached_nonweight_files(
    config_factory: Callable[..., ExperimentConfig],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = pytest.importorskip("huggingface_hub")
    snapshot = tmp_path / "snapshots" / _REVISION
    snapshot.mkdir(parents=True)
    expected: dict[str, str] = {}
    for index, name in enumerate(_FILENAMES):
        payload = f"safe-metadata-{index}".encode()
        (snapshot / name).write_bytes(payload)
        expected[name] = hashlib.sha256(payload).hexdigest()
    requested: list[str] = []

    def cached_file(
        repo_id: str,
        filename: str,
        *,
        revision: str,
        local_files_only: bool,
    ) -> str:
        assert repo_id == _MODEL_ID
        assert revision == _REVISION
        assert local_files_only is True
        requested.append(filename)
        return str(snapshot / filename)

    monkeypatch.setattr(hub, "hf_hub_download", cached_file)
    resolved = HFCausalLMAdapter()._resolve_source_metadata_files(_hf_config(config_factory))

    assert requested == list(_FILENAMES)
    assert "model.safetensors" not in requested
    assert dict(resolved.file_sha256) == expected
    assert resolved.snapshot_path == snapshot


def test_hf_empty_export_shell_requires_exact_export_identity(
    config_factory: Callable[..., ExperimentConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = HFCausalLMAdapter()
    monkeypatch.setattr(
        adapter,
        "_load_offline_config_and_tokenizer",
        lambda _config: (_ for _ in ()).throw(AssertionError("identity must fail first")),
    )
    wrong = ExportedModelIdentity(
        model_id="other/model",
        revision=_REVISION,
        resolved_class=_RESOLVED_CLASS,
        architecture=_ARCHITECTURE,
        source_checksum=_SOURCE_CHECKSUM,
    )

    with pytest.raises(ValueError, match="model identity"):
        adapter.load_empty_export_shell(
            _hf_config(config_factory),
            TorchEagerCPURuntime(),
            wrong,
        )
