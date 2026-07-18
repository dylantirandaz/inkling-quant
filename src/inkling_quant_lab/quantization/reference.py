"""Shared helpers and the exact no-op reference quantizer."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import load_file, save
from torch import Tensor, nn

from inkling_quant_lab.config import ExperimentConfig, QuantizationConfig
from inkling_quant_lab.models.base import (
    ExportedModelIdentity,
    LoadedModel,
    ModelBatch,
    ModelDescriptor,
    RuntimeForModel,
    SourceWeightFreeModelAdapter,
    SourceWeightFreeReloadProvenance,
)
from inkling_quant_lab.quantization.base import (
    CalibrationArtifact,
    ExportArtifact,
    QuantizationManifest,
    QuantizedModel,
    ResolvedPolicyLike,
    SupportReport,
)
from inkling_quant_lab.runtimes.base import RuntimeCapabilities

_SAFE_EXPORT_FORMATS = frozenset({"recipe_json", "safetensors"})
HF_CAUSAL_LM_SOURCE_WEIGHT_FREE_RELOAD_ADAPTER = "hf_causal_lm_source_weight_free_v1"
TRUSTED_SOURCE_MODEL_RELOAD_ADAPTER = "trusted_source_model_v1"
_VALIDATED_HF_MIXTRAL_CLASS = "transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM"
_SOURCE_WEIGHT_FREE_BACKENDS = frozenset({"noop", "torch_native_dynamic_int8"})
_SOURCE_METADATA_FILES = frozenset(
    {
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reload_adapter_for_descriptor(descriptor: ModelDescriptor) -> str:
    if descriptor.resolved_class == _VALIDATED_HF_MIXTRAL_CLASS:
        return HF_CAUSAL_LM_SOURCE_WEIGHT_FREE_RELOAD_ADAPTER
    return TRUSTED_SOURCE_MODEL_RELOAD_ADAPTER


def model_storage_bytes(model: nn.Module) -> int:
    """Count unique parameter and buffer storage bytes."""

    seen: set[int] = set()
    total = 0
    for value in (*tuple(model.parameters()), *tuple(model.buffers())):
        pointer = value.untyped_storage().data_ptr()
        if pointer in seen:
            continue
        seen.add(pointer)
        total += value.untyped_storage().nbytes()
    return total


def _calibration_fields(
    calibration: CalibrationArtifact | None,
) -> tuple[dict[str, str] | None, tuple[str, ...]]:
    if calibration is None:
        return None, ()
    config = calibration.config
    return (
        {"dataset": config.dataset, "revision": config.revision, "split": config.split},
        calibration.sample_ids,
    )


def build_manifest(
    *,
    backend: str,
    backend_version: str,
    method: str,
    source: ModelDescriptor,
    precision_map: dict[str, str],
    serialized_size_bytes: int,
    calibration: CalibrationArtifact | None,
    parameters: dict[str, str | int | float | bool],
    warnings: tuple[str, ...] = (),
) -> QuantizationManifest:
    """Build a complete reference quantization manifest."""

    dataset, sample_ids = _calibration_fields(calibration)
    excluded = tuple(
        sorted(
            name for name, precision in precision_map.items() if precision not in {"int8", "int4"}
        )
    )
    return QuantizationManifest(
        backend=backend,
        backend_version=backend_version,
        method=method,
        source_model_checksum=source.checksum,
        module_precision_map=dict(sorted(precision_map.items())),
        excluded_modules=excluded,
        calibration_dataset=dataset,
        calibration_sample_ids=sample_ids,
        quantization_parameters=dict(sorted(parameters.items())),
        serialized_size_bytes=serialized_size_bytes,
        warnings=warnings,
    )


def export_recipe(
    model: QuantizedModel,
    destination: Path,
    config: QuantizationConfig,
) -> ExportArtifact:
    """Atomically export a safe tensor-and-metadata bundle that can be reloaded."""

    updated_manifest, payloads, recipe = _export_payloads(model, config)
    model.manifest = updated_manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite quantizer export: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    )
    try:
        for name, payload in payloads.items():
            _write_durable_bytes(temporary / name, payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink(missing_ok=True)
            temporary.rmdir()
    return ExportArtifact(
        path=str(destination),
        format=config.export.format,
        sha256=_bundle_sha256(payloads),
        size_bytes=sum(len(payload) for payload in payloads.values()),
        files=tuple(sorted(payloads)),
        reload_recipe=cast(dict[str, Any], recipe["reload"]),
    )


def load_export_recipe(path: Path) -> dict[str, Any]:
    """Validate and return a portable quantizer reconstruction recipe."""

    metadata_path = _metadata_path(path)
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "2.0":
        raise ValueError("unsupported quantizer recipe schema")
    if not isinstance(raw.get("model"), dict) or not isinstance(raw.get("quantization"), dict):
        raise ValueError("quantizer recipe is incomplete")
    reload = raw.get("reload")
    if not isinstance(reload, dict):
        raise ValueError("quantizer recipe has no reload contract")
    export_format = reload.get("format")
    if export_format not in _SAFE_EXPORT_FORMATS:
        raise ValueError(f"unsupported safe quantizer export format: {export_format!r}")
    tensor_name = reload.get("tensor_file")
    if not isinstance(tensor_name, str) or Path(tensor_name).name != tensor_name:
        raise ValueError("quantizer recipe tensor path must be one bundle-local filename")
    tensor_path = metadata_path.parent / tensor_name
    if tensor_path.is_symlink() or not tensor_path.is_file():
        raise ValueError("quantizer recipe tensor payload is missing or unsafe")
    expected_files = {metadata_path.name, tensor_name}
    actual_files = {child.name for child in metadata_path.parent.iterdir()}
    if actual_files != expected_files or any(
        child.is_symlink() or not child.is_file() for child in metadata_path.parent.iterdir()
    ):
        raise ValueError("quantizer export bundle contains unexpected or unsafe entries")
    expected_sha256 = reload.get("tensor_sha256")
    actual_sha256 = _stream_sha256(tensor_path)
    if not isinstance(expected_sha256, str) or actual_sha256 != expected_sha256:
        raise ValueError("quantizer recipe tensor checksum does not match")
    expected_size = cast(dict[str, Any], raw["quantization"]).get("serialized_size_bytes")
    actual_size = sum(child.stat().st_size for child in metadata_path.parent.iterdir())
    if expected_size != actual_size:
        raise ValueError("quantizer export serialized size does not match its bundle")
    return raw


def reload_exported_model(path: Path, source: LoadedModel) -> QuantizedModel:
    """Safely reload a CPU reference export using a trusted source model constructor."""

    started = time.perf_counter()
    recipe = load_export_recipe(path)
    model_record = cast(dict[str, Any], recipe["model"])
    if model_record.get("source_checksum") != source.descriptor.checksum:
        raise ValueError("quantizer export source checksum does not match the supplied model")
    manifest = QuantizationManifest.model_validate(recipe["quantization"])
    backend = manifest.backend
    candidate = copy.deepcopy(cast(nn.Module, source.model))
    _install_reference_modules(
        candidate,
        backend,
        manifest.module_precision_map,
        manifest.quantization_parameters,
    )
    reload_record = cast(dict[str, Any], recipe["reload"])
    tensor_path = _metadata_path(path).parent / cast(str, reload_record["tensor_file"])
    state = load_file(str(tensor_path), device="cpu")
    candidate.load_state_dict(state, strict=True)
    candidate.eval()
    elapsed = time.perf_counter() - started
    if elapsed <= 0.0:
        raise RuntimeError("candidate export reload did not produce a measurable duration")
    loaded = LoadedModel(
        model=candidate,
        tokenizer=copy.deepcopy(source.tokenizer),
        descriptor=source.descriptor,
        load_time_seconds=elapsed,
        load_time_kind="candidate_export_reload",
    )
    return QuantizedModel(loaded=loaded, manifest=manifest)


def _exported_model_identity(record: Mapping[str, Any]) -> ExportedModelIdentity:
    expected_fields = {
        "model_id",
        "revision",
        "resolved_class",
        "architecture",
        "source_checksum",
    }
    if set(record) != expected_fields or any(
        not isinstance(record.get(field), str) or not cast(str, record[field])
        for field in expected_fields
    ):
        raise ValueError("source-weight-free export model identity is incomplete")
    return ExportedModelIdentity(
        model_id=cast(str, record["model_id"]),
        revision=cast(str, record["revision"]),
        resolved_class=cast(str, record["resolved_class"]),
        architecture=cast(str, record["architecture"]),
        source_checksum=cast(str, record["source_checksum"]),
    )


def _install_source_weight_free_modules(
    root: nn.Module, manifest: QuantizationManifest
) -> tuple[str, ...]:
    precision_map = manifest.module_precision_map
    if any(precision not in {"float32", "int8"} for precision in precision_map.values()):
        raise ValueError("source-weight-free reload supports only float32 and native INT8 state")
    if manifest.backend == "noop":
        if any(precision != "float32" for precision in precision_map.values()):
            raise ValueError("source-weight-free no-op reload cannot install quantized modules")
        return ()
    if manifest.backend != "torch_native_dynamic_int8":
        raise ValueError(f"source-weight-free reload does not support backend {manifest.backend!r}")

    from inkling_quant_lab.quantization.int8 import _replace_module
    from inkling_quant_lab.quantization.native_cpu import (
        NativeDynamicInt8Linear,
        probe_native_dynamic_int8,
    )

    engine = manifest.quantization_parameters.get("quantized_engine")
    if not isinstance(engine, str) or not engine:
        raise ValueError("source-weight-free native INT8 reload requires quantized_engine")
    capability = probe_native_dynamic_int8(engine)
    if not capability.supported or capability.implementation != engine:
        raise ValueError(
            "source-weight-free native INT8 reload requires its recorded quantized engine "
            f"{engine!r}: {'; '.join(capability.reasons) or 'probe mismatch'}"
        )
    targets = tuple(
        sorted(name for name, precision in precision_map.items() if precision == "int8")
    )
    if not targets:
        raise ValueError("source-weight-free native INT8 manifest selects no modules")
    if any(left != right and right.startswith(f"{left}.") for left in targets for right in targets):
        raise ValueError("source-weight-free native INT8 targets cannot overlap")
    named = dict(root.named_modules())
    invalid = tuple(name for name in targets if not isinstance(named.get(name), nn.Linear))
    if invalid:
        raise ValueError(
            "source-weight-free native INT8 targets are missing or are not linear: "
            + ", ".join(invalid)
        )
    for name in targets:
        source = cast(nn.Linear, named[name])
        _replace_module(
            root,
            name,
            NativeDynamicInt8Linear.from_empty(
                in_features=source.in_features,
                out_features=source.out_features,
                bias=source.bias is not None,
                engine=engine,
                device="meta",
            ),
        )
    installed = tuple(
        sorted(
            name
            for name, module in root.named_modules()
            if isinstance(module, NativeDynamicInt8Linear)
        )
    )
    if installed != targets:
        raise ValueError("source-weight-free native INT8 wrapper inventory differs from manifest")
    return installed


def _validate_assign_state(expected: Mapping[str, Tensor], state: Mapping[str, Tensor]) -> None:
    expected_names = set(expected)
    actual_names = set(state)
    if expected_names != actual_names:
        raise ValueError(
            "source-weight-free tensor inventory mismatch: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    for name in sorted(expected):
        expected_tensor = expected[name]
        actual_tensor = state[name]
        if tuple(actual_tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"source-weight-free tensor shape mismatch for {name}: "
                f"{tuple(actual_tensor.shape)} != {tuple(expected_tensor.shape)}"
            )
        if actual_tensor.dtype != expected_tensor.dtype:
            raise ValueError(
                f"source-weight-free tensor dtype mismatch for {name}: "
                f"{actual_tensor.dtype} != {expected_tensor.dtype}"
            )
        if actual_tensor.device.type != "cpu":
            raise ValueError(f"source-weight-free tensor {name} is not on CPU")


def _semantic_state_checksum(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        normalized = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(normalized.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(normalized.dtype).encode("ascii"))
        digest.update(b"\0")
        payload = normalized.view(torch.uint8).reshape(-1).numpy()
        digest.update(memoryview(payload))
    return digest.hexdigest()


def _bundle_digest_from_hashes(file_sha256: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name, sha256 in sorted(file_sha256.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256))
    return digest.hexdigest()


def _validated_source_metadata_hashes(
    values: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    normalized = tuple(sorted(values))
    if (
        len(normalized) != len(_SOURCE_METADATA_FILES)
        or {name for name, _ in normalized} != _SOURCE_METADATA_FILES
        or any(
            len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256)
            for _, sha256 in normalized
        )
    ):
        raise ValueError("source-weight-free shell metadata hashes are incomplete or invalid")
    return normalized


def reload_exported_model_source_weight_free(
    path: Path,
    adapter: SourceWeightFreeModelAdapter,
    config: ExperimentConfig,
    runtime: RuntimeForModel,
) -> tuple[QuantizedModel, SourceWeightFreeReloadProvenance]:
    """Load one HF candidate export without loading or copying source checkpoint weights."""

    started = time.perf_counter()
    metadata_path = _metadata_path(path)
    recipe = load_export_recipe(path)
    if set(recipe) != {"schema_version", "model", "quantization", "reload"}:
        raise ValueError("source-weight-free export recipe contains unexpected fields")
    model_record = recipe.get("model")
    reload_record = recipe.get("reload")
    if not isinstance(model_record, Mapping) or not isinstance(reload_record, Mapping):
        raise ValueError("source-weight-free export recipe is incomplete")
    expected_reload_fields = {
        "adapter",
        "backend",
        "format",
        "metadata_file",
        "tensor_file",
        "tensor_sha256",
    }
    if set(reload_record) != expected_reload_fields:
        raise ValueError("source-weight-free export reload contract contains unexpected fields")
    reload_adapter = reload_record.get("adapter")
    if reload_adapter != HF_CAUSAL_LM_SOURCE_WEIGHT_FREE_RELOAD_ADAPTER:
        raise ValueError(
            "source-weight-free reload requires adapter "
            f"{HF_CAUSAL_LM_SOURCE_WEIGHT_FREE_RELOAD_ADAPTER!r}"
        )
    identity = _exported_model_identity(model_record)
    if (
        config.model.adapter != "hf_causal_lm"
        or config.model.model_id != identity.model_id
        or config.model.revision != identity.revision
    ):
        raise ValueError("source-weight-free export model identity does not match config")
    manifest = QuantizationManifest.model_validate(recipe["quantization"])
    backend = reload_record.get("backend")
    if (
        backend not in _SOURCE_WEIGHT_FREE_BACKENDS
        or backend != manifest.backend
        or backend != config.quantization.backend
    ):
        raise ValueError("source-weight-free export backend does not match config and manifest")
    if manifest.method != config.quantization.method:
        raise ValueError("source-weight-free export method does not match config")
    if manifest.source_model_checksum != identity.source_checksum:
        raise ValueError("source-weight-free export source checksum is inconsistent")
    if any(
        manifest.quantization_parameters.get(name) != value
        for name, value in config.quantization.parameters.items()
    ):
        raise ValueError("source-weight-free quantization parameters do not match config")
    if (
        reload_record.get("metadata_file") != metadata_path.name
        or reload_record.get("format") != config.quantization.export.format
    ):
        raise ValueError("source-weight-free export format does not match config")
    tensor_name = reload_record.get("tensor_file")
    if not isinstance(tensor_name, str):
        raise ValueError("source-weight-free export tensor filename is invalid")
    tensor_path = metadata_path.parent / tensor_name
    tensor_sha256 = _stream_sha256(tensor_path)
    if tensor_sha256 != reload_record.get("tensor_sha256"):
        raise ValueError("source-weight-free export tensor checksum changed during validation")
    metadata_sha256 = _stream_sha256(metadata_path)

    shell = adapter.load_empty_export_shell(config, runtime, identity)
    descriptor_identity = (
        shell.descriptor.model_id,
        shell.descriptor.revision,
        shell.descriptor.resolved_class,
        shell.descriptor.architecture,
        shell.descriptor.checksum,
    )
    if descriptor_identity != (
        identity.model_id,
        identity.revision,
        identity.resolved_class,
        identity.architecture,
        identity.source_checksum,
    ):
        raise ValueError("source-weight-free shell identity differs from the export recipe")
    source_metadata_hashes = _validated_source_metadata_hashes(shell.source_metadata_file_sha256)
    candidate = cast(nn.Module, shell.model)
    initial_state = candidate.state_dict()
    if not initial_state or any(not tensor.is_meta for tensor in initial_state.values()):
        raise ValueError("source-weight-free model shell materialized persistent state")
    quantized_module_names = _install_source_weight_free_modules(candidate, manifest)
    expected_state = candidate.state_dict()
    if any(not tensor.is_meta for tensor in expected_state.values()):
        raise ValueError("source-weight-free wrapper installation materialized persistent state")

    state = load_file(str(tensor_path), device="cpu")
    _validate_assign_state(expected_state, state)
    incompatible = candidate.load_state_dict(state, strict=True, assign=True)
    missing_keys = tuple(incompatible.missing_keys)
    unexpected_keys = tuple(incompatible.unexpected_keys)
    if missing_keys or unexpected_keys:
        raise ValueError("source-weight-free strict state load returned incompatible keys")
    candidate.eval()
    meta_tensor_names = tuple(
        name
        for name, tensor in (*candidate.named_parameters(), *candidate.named_buffers())
        if tensor.is_meta
    )
    if meta_tensor_names:
        raise ValueError(
            "source-weight-free state load left meta tensors: " + ", ".join(meta_tensor_names)
        )
    non_cpu = tuple(
        name
        for name, tensor in (*candidate.named_parameters(), *candidate.named_buffers())
        if tensor.device.type != "cpu"
    )
    if non_cpu:
        raise ValueError(
            "source-weight-free state load placed tensors outside CPU: " + ", ".join(non_cpu)
        )
    if manifest.backend == "torch_native_dynamic_int8":
        from inkling_quant_lab.quantization.native_cpu import NativeDynamicInt8Linear

        unpacked = tuple(
            name
            for name, module in candidate.named_modules()
            if isinstance(module, NativeDynamicInt8Linear) and module._packed_weight is None
        )
        if unpacked:
            raise ValueError(
                "source-weight-free native INT8 wrappers were not repacked: " + ", ".join(unpacked)
            )
    tensor_count = len(state)
    candidate_state_checksum = _semantic_state_checksum(candidate)
    elapsed = time.perf_counter() - started
    if elapsed <= 0.0:
        raise RuntimeError("source-weight-free candidate load duration was not measurable")
    loaded = LoadedModel(
        model=candidate,
        tokenizer=shell.tokenizer,
        descriptor=shell.descriptor,
        load_time_seconds=elapsed,
        load_time_kind="candidate_source_weight_free_export_load",
    )
    provenance = SourceWeightFreeReloadProvenance(
        reload_adapter=cast(str, reload_adapter),
        backend=manifest.backend,
        model_id=identity.model_id,
        revision=identity.revision,
        resolved_class=identity.resolved_class,
        architecture=identity.architecture,
        source_model_checksum=identity.source_checksum,
        metadata_file=metadata_path.name,
        metadata_sha256=metadata_sha256,
        tensor_file=tensor_path.name,
        tensor_sha256=tensor_sha256,
        bundle_sha256=_bundle_digest_from_hashes(
            {metadata_path.name: metadata_sha256, tensor_path.name: tensor_sha256}
        ),
        source_metadata_file_sha256=source_metadata_hashes,
        candidate_state_checksum=candidate_state_checksum,
        quantized_module_names=quantized_module_names,
        native_wrapper_count=len(quantized_module_names),
        tensor_count=tensor_count,
        strict_load=True,
        assign=True,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        meta_tensor_names=meta_tensor_names,
        source_weights_loaded=False,
    )
    return QuantizedModel(loaded=loaded, manifest=manifest), provenance


def finalize_serialized_manifest(
    model: QuantizedModel, config: QuantizationConfig
) -> QuantizedModel:
    """Measure the exact safe bundle bytes and store that fact in the manifest."""

    manifest, _, _ = _export_payloads(model, config)
    model.manifest = manifest
    return model


def safe_model_serialized_size_bytes(model: LoadedModel) -> int:
    """Measure a reloadable safetensors-plus-metadata baseline representation."""

    tensor_payload = _tensor_payload(cast(nn.Module, model.model))
    tensor_sha256 = hashlib.sha256(tensor_payload).hexdigest()
    size = 0
    for _ in range(16):
        metadata = {
            "schema_version": "baseline-export-v1",
            "model": {
                "model_id": model.descriptor.model_id,
                "revision": model.descriptor.revision,
                "resolved_class": model.descriptor.resolved_class,
                "architecture": model.descriptor.architecture,
                "source_checksum": model.descriptor.checksum,
            },
            "reload": {
                "adapter": _reload_adapter_for_descriptor(model.descriptor),
                "format": "safetensors",
                "tensor_file": "model.safetensors",
                "tensor_sha256": tensor_sha256,
            },
            "serialized_size_bytes": size,
        }
        metadata_payload = _json_payload(metadata)
        measured = len(tensor_payload) + len(metadata_payload)
        if measured == size:
            return measured
        size = measured
    raise RuntimeError("baseline serialized-size metadata did not converge")


def _export_payloads(
    model: QuantizedModel, config: QuantizationConfig
) -> tuple[QuantizationManifest, dict[str, bytes], dict[str, Any]]:
    export_format = config.export.format
    if export_format not in _SAFE_EXPORT_FORMATS:
        raise ValueError(f"unsupported safe quantizer export format: {export_format!r}")
    metadata_name, tensor_name = _bundle_names(export_format)
    tensor_payload = _tensor_payload(cast(nn.Module, model.loaded.model))
    tensor_sha256 = hashlib.sha256(tensor_payload).hexdigest()
    manifest = model.manifest
    for _ in range(16):
        recipe: dict[str, Any] = {
            "schema_version": "2.0",
            "model": {
                "model_id": model.loaded.descriptor.model_id,
                "revision": model.loaded.descriptor.revision,
                "resolved_class": model.loaded.descriptor.resolved_class,
                "architecture": model.loaded.descriptor.architecture,
                "source_checksum": model.loaded.descriptor.checksum,
            },
            "quantization": manifest.model_dump(mode="json"),
            "reload": {
                "adapter": _reload_adapter_for_descriptor(model.loaded.descriptor),
                "backend": manifest.backend,
                "format": export_format,
                "metadata_file": metadata_name,
                "tensor_file": tensor_name,
                "tensor_sha256": tensor_sha256,
            },
        }
        metadata_payload = _json_payload(recipe)
        measured = len(tensor_payload) + len(metadata_payload)
        if measured == manifest.serialized_size_bytes:
            return manifest, {metadata_name: metadata_payload, tensor_name: tensor_payload}, recipe
        manifest = manifest.model_copy(update={"serialized_size_bytes": measured})
    raise RuntimeError("quantizer serialized-size metadata did not converge")


def _tensor_payload(model: nn.Module) -> bytes:
    state: dict[str, Tensor] = {
        name: tensor.detach().cpu().contiguous().clone()
        for name, tensor in sorted(model.state_dict().items())
    }
    return save(state)


def _bundle_names(export_format: str) -> tuple[str, str]:
    if export_format == "recipe_json":
        return "recipe.json", "weights.safetensors"
    if export_format == "safetensors":
        return "metadata.json", "model.safetensors"
    raise ValueError(f"unsupported safe quantizer export format: {export_format!r}")


def _metadata_path(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("quantizer export path must not be a symlink")
    if path.is_dir():
        candidates = tuple(
            candidate
            for candidate in (path / "recipe.json", path / "metadata.json")
            if candidate.is_file() and not candidate.is_symlink()
        )
        if len(candidates) != 1:
            raise ValueError("quantizer export must contain exactly one recognized metadata file")
        return candidates[0]
    if path.is_file() and not path.is_symlink() and path.name in {"recipe.json", "metadata.json"}:
        return path
    raise ValueError("quantizer export path must be a bundle or its recognized metadata file")


def _install_reference_modules(
    root: nn.Module,
    backend: str,
    precision_map: dict[str, str],
    parameters: dict[str, str | int | float | bool] | None = None,
) -> None:
    if backend in {"noop", "fake_optional_cpu"}:
        return
    if backend == "torch_dynamic_int8":
        from inkling_quant_lab.quantization.int8 import _replace_int8_linears

        if not _replace_int8_linears(root, precision_map):
            raise ValueError("dynamic INT8 export selected no reloadable linear modules")
        return
    if backend in {"torch_weight_only_int4", "torch_reference_mixed"}:
        from inkling_quant_lab.quantization.int8 import DynamicInt8Linear, _replace_module
        from inkling_quant_lab.quantization.weight_only import PackedInt4Linear

        named = dict(root.named_modules())
        replaced: list[str] = []
        for name in sorted(precision_map):
            module = named.get(name)
            if not isinstance(module, nn.Linear):
                continue
            precision = precision_map[name]
            if precision == "int4":
                _replace_module(root, name, PackedInt4Linear(module))
                replaced.append(name)
            elif backend == "torch_reference_mixed" and precision == "int8":
                _replace_module(root, name, DynamicInt8Linear(module))
                replaced.append(name)
        if not replaced:
            raise ValueError(f"{backend} export selected no reloadable linear modules")
        return
    if backend in {"torch_native_dynamic_int8", "torch_native_int4_kleidiai"}:
        from inkling_quant_lab.quantization.int8 import _replace_module
        from inkling_quant_lab.quantization.native_cpu import (
            NativeDynamicInt8Linear,
            NativeInt4KleidiAILinear,
            probe_native_dynamic_int8,
        )

        named = dict(root.named_modules())
        replaced = []
        engine_value = (parameters or {}).get("quantized_engine")
        if backend == "torch_native_dynamic_int8":
            if not isinstance(engine_value, str):
                raise ValueError("native dynamic INT8 export does not pin quantized_engine")
            capability = probe_native_dynamic_int8(engine_value)
            if not capability.supported or capability.implementation != engine_value:
                raise ValueError(
                    "native dynamic INT8 export requires its recorded quantized engine "
                    f"{engine_value!r}: {'; '.join(capability.reasons) or 'probe mismatch'}"
                )
        for name in sorted(precision_map):
            module = named.get(name)
            if not isinstance(module, nn.Linear):
                continue
            precision = precision_map[name]
            if backend == "torch_native_dynamic_int8" and precision == "int8":
                _replace_module(
                    root,
                    name,
                    NativeDynamicInt8Linear(module, engine=cast(str, engine_value)),
                )
                replaced.append(name)
            elif backend == "torch_native_int4_kleidiai" and precision == "int4":
                _replace_module(root, name, NativeInt4KleidiAILinear(module))
                replaced.append(name)
        if not replaced:
            raise ValueError(f"{backend} export selected no reloadable linear modules")
        return
    raise ValueError(f"quantizer backend {backend!r} has no safe CPU reload implementation")


def _write_durable_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _json_payload(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _bundle_sha256(payloads: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(payloads.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


class NoopQuantizer:
    """Deep-copy baseline quantizer used to prove exact comparison behavior."""

    name = "noop"
    version = "reference-v1"

    def check_support(
        self,
        model: ModelDescriptor,
        runtime: RuntimeCapabilities,
        config: QuantizationConfig,
    ) -> SupportReport:
        """No-op supports every model that the selected runtime can execute."""

        del model, config
        return SupportReport(
            available=runtime.available,
            supported=runtime.available,
            component=self.name,
            reasons=runtime.reasons,
            supported_precisions=("float32", "float16", "bfloat16"),
        )

    def calibrate(
        self,
        model: LoadedModel,
        samples: tuple[ModelBatch, ...],
        config: QuantizationConfig,
    ) -> CalibrationArtifact | None:
        """No-op requires no calibration."""

        del model, samples, config
        return None

    def quantize(
        self,
        model: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        """Deep-copy model state without numerical modification."""

        started = time.perf_counter()
        copied = LoadedModel(
            model=copy.deepcopy(model.model),
            tokenizer=copy.deepcopy(model.tokenizer),
            descriptor=model.descriptor,
            load_time_seconds=0.0,
            load_time_kind="candidate_reconstruction",
        )
        copied.load_time_seconds = time.perf_counter() - started
        if copied.load_time_seconds <= 0.0:
            raise RuntimeError("no-op candidate reconstruction duration was not measurable")
        size = model_storage_bytes(copied.model)
        manifest = build_manifest(
            backend=self.name,
            backend_version=self.version,
            method=config.method,
            source=model.descriptor,
            precision_map=policy.precision_map,
            serialized_size_bytes=size,
            calibration=calibration,
            parameters=config.parameters,
        )
        return finalize_serialized_manifest(
            QuantizedModel(loaded=copied, manifest=manifest), config
        )

    def export(
        self, model: QuantizedModel, destination: Path, config: QuantizationConfig
    ) -> ExportArtifact:
        """Export a safe reconstruction recipe."""

        return export_recipe(model, destination, config)


def create_quantizer() -> NoopQuantizer:
    """Registry factory for the no-op backend."""

    return NoopQuantizer()
