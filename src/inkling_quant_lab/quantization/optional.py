"""Capability-gated production integrations for optional quantizers.

AWQ and GPTQ use GPTQModel's maintained conversion API.  Fine-grained FP8
uses Transformers' ``FineGrainedFP8Config`` load-time conversion.  All three
own source loading, so they implement the explicit pinned-source-reload
lifecycle in addition to the common quantizer protocol.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Literal, NoReturn, cast

import torch
from torch import Tensor, nn

from inkling_quant_lab.config import QuantizationConfig
from inkling_quant_lab.exceptions import CapabilityError, QuantizationError
from inkling_quant_lab.models.base import LoadedModel, ModelBatch, ModelDescriptor
from inkling_quant_lab.models.mixtral_compat import (
    DefuserMixtralBindings,
    capture_defuser_mixtral_bindings,
    restore_defuser_mixtral_bindings,
)
from inkling_quant_lab.models.state import model_state_sha256
from inkling_quant_lab.quantization.base import (
    CalibrationArtifact,
    ExportArtifact,
    QuantizationManifest,
    QuantizedModel,
    ResolvedPolicyLike,
    SupportReport,
)
from inkling_quant_lab.quantization.reference import model_storage_bytes
from inkling_quant_lab.runtimes.base import RuntimeCapabilities

_COMMIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_CUDA_DEVICE = re.compile(r"^cuda:(0|[1-9][0-9]*)$")
_CPU_DEVICE = "cpu"
_FULL_PRECISIONS = frozenset({"float32", "float16", "bfloat16"})
_BANNED_WEIGHT_SUFFIXES = frozenset({".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth"})
_BANNED_CODE_SUFFIXES = frozenset(
    {
        ".a",
        ".bash",
        ".bat",
        ".bundle",
        ".class",
        ".cmd",
        ".com",
        ".dll",
        ".dylib",
        ".egg",
        ".exe",
        ".fish",
        ".jar",
        ".lib",
        ".node",
        ".o",
        ".ps1",
        ".psm1",
        ".py",
        ".pyc",
        ".pyd",
        ".pyi",
        ".pyo",
        ".pyw",
        ".sh",
        ".so",
        ".wasm",
        ".whl",
        ".zsh",
    }
)
_HF_REPOSITORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_SAFE_SNAPSHOT_PATTERNS = (
    "*.json",
    "*.safetensors",
    "merges.txt",
    "sentencepiece.bpe.model",
    "spiece.model",
    "tokenizer.model",
    "vocab.txt",
)
_UNSAFE_SNAPSHOT_PATTERNS = (
    *(f"*{suffix}" for suffix in sorted(_BANNED_CODE_SUFFIXES | _BANNED_WEIGHT_SUFFIXES)),
    "*.so.*",
)
_EXECUTABLE_MAGIC_PREFIXES = (
    b"\x00asm",  # WebAssembly
    b"\x7fELF",
    b"MZ",  # DOS/PE executable envelope
    b"\xbe\xba\xfe\xca",  # reverse-endian Mach-O universal
    b"\xbf\xba\xfe\xca",  # reverse-endian Mach-O universal 64-bit
    b"\xca\xfe\xba\xbe",  # Mach-O universal
    b"\xca\xfe\xba\xbf",  # Mach-O universal 64-bit
    b"\xce\xfa\xed\xfe",  # little-endian Mach-O 32-bit
    b"\xcf\xfa\xed\xfe",  # little-endian Mach-O 64-bit
    b"\xfe\xed\xfa\xce",  # big-endian Mach-O 32-bit
    b"\xfe\xed\xfa\xcf",  # big-endian Mach-O 64-bit
)
_GPTQMODEL_MIN_ACTIVE_TOKENS = 10
_NATIVE_MIXTRAL_ARCHITECTURE = "MixtralForCausalLM"
_STORIES15M_MODEL_ID = "ggml-org/stories15M_MOE"
_STORIES15M_REVISION = "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
_STORIES15M_SOURCE_FILES = (
    "config.json",
    "data.txt",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "moe_shakespeare15M/checkpoint-400/adapter_config.json",
    "moe_shakespeare15M/checkpoint-400/trainer_state.json",
    "moe_shakespeare15M/checkpoint-500/adapter_config.json",
    "moe_shakespeare15M/checkpoint-500/trainer_state.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
_DEFUSED_EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_GPTQMODEL_IMPORT_ENVIRONMENT = ("CUDA_DEVICE_ORDER", "PYTORCH_ALLOC_CONF")
_GPTQ_RELOAD_FILENAME = "inkling_quant_reload.json"
_MISSING = object()


@dataclass(frozen=True, slots=True)
class _DependencyRequirement:
    distribution: str
    module: str
    minimum: tuple[int, int, int]
    maximum: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _DependencyStatus:
    available: bool
    versions: dict[str, str]
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TokenSample:
    sample_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    def upstream_record(self) -> dict[str, list[int]]:
        return {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
        }


@dataclass(frozen=True, slots=True)
class _PreparedCalibration:
    source_checksum: str
    artifact_checksum: str
    samples: tuple[_TokenSample, ...]


@dataclass(frozen=True, slots=True)
class _CommonSettings:
    local_files_only: bool
    device: str


@dataclass(frozen=True, slots=True)
class _GPTQModelSettings(_CommonSettings):
    bits: int
    group_size: int
    sym: bool
    batch_size: int
    format: str
    desc_act: bool = False
    act_group_aware: bool = True
    damp_percent: float = 0.1


@dataclass(frozen=True, slots=True)
class _FP8Settings(_CommonSettings):
    activation_scheme: str
    block_rows: int
    block_columns: int
    scale_fmt: str


@dataclass(frozen=True, slots=True)
class _ExportState:
    owner: Any
    save_kind: str
    runtime_transition: _GPTQModelRuntimeTransition | None = None


@dataclass(frozen=True, slots=True)
class _GPTQModelRuntimeTransition:
    api: Any
    module: nn.Module
    quantize_config: Any
    qlinear_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _GPTQModelCPUPostInit:
    module_count: int
    compile_call_count: int


@dataclass(frozen=True, slots=True)
class _GPTQModelExclusionAudit:
    module_count: int
    state_tensor_count: int
    state_sha256: str
    expanded_linear_count: int


_GPTQMODEL_REQUIREMENTS = (
    _DependencyRequirement("gptqmodel", "gptqmodel", (5, 8, 0), (5, 8, 1)),
    _DependencyRequirement("defuser", "defuser", (0, 0, 23), (0, 0, 24)),
    _DependencyRequirement("kernels", "kernels", (0, 14, 1), (0, 14, 2)),
    _DependencyRequirement("huggingface-hub", "huggingface_hub", (0, 34, 4), (2, 0, 0)),
    _DependencyRequirement("transformers", "transformers", (5, 3, 0), (6, 0, 0)),
    _DependencyRequirement("accelerate", "accelerate", (1, 13, 0), (2, 0, 0)),
    _DependencyRequirement("safetensors", "safetensors", (0, 7, 0), (1, 0, 0)),
    _DependencyRequirement("torch", "torch", (2, 8, 0), (3, 0, 0)),
)
_GPTQMODEL_CPU_EXACT_VERSIONS = {
    "accelerate": "1.14.0",
    "defuser": "0.0.23",
    "gptqmodel": "5.8.0",
    "huggingface-hub": "1.23.0",
    "kernels": "0.14.1",
    "safetensors": "0.8.0",
    "torch": "2.13.0",
    "transformers": "5.12.1",
}
_FP8_REQUIREMENTS = (
    _DependencyRequirement("transformers", "transformers", (5, 4, 0), (6, 0, 0)),
    _DependencyRequirement("accelerate", "accelerate", (1, 13, 0), (2, 0, 0)),
    _DependencyRequirement("safetensors", "safetensors", (0, 7, 0), (1, 0, 0)),
    _DependencyRequirement("torch", "torch", (2, 8, 0), (3, 0, 0)),
)


def _release_triplet(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", value)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), int(match.group(3) or 0)


def _capture_environment_bindings(names: tuple[str, ...]) -> dict[str, str | object]:
    return {name: os.environ.get(name, _MISSING) for name in names}


def _restore_environment_bindings(bindings: dict[str, str | object]) -> bool:
    for name, value in bindings.items():
        if value is _MISSING:
            os.environ.pop(name, None)
        else:
            os.environ[name] = cast(str, value)
    return all((os.environ.get(name, _MISSING) == value) for name, value in bindings.items())


def _dependency_status(
    requirements: tuple[_DependencyRequirement, ...],
) -> _DependencyStatus:
    versions: dict[str, str] = {}
    reasons: list[str] = []
    for requirement in requirements:
        if importlib.util.find_spec(requirement.module) is None:
            reasons.append(f"missing Python package {requirement.distribution}")
            continue
        try:
            version = importlib.metadata.version(requirement.distribution)
        except importlib.metadata.PackageNotFoundError:
            reasons.append(
                f"package module {requirement.module} exists but distribution metadata is missing"
            )
            continue
        versions[requirement.distribution] = version
        release = _release_triplet(version)
        if release is None or not (requirement.minimum <= release < requirement.maximum):
            minimum = ".".join(str(part) for part in requirement.minimum)
            maximum = ".".join(str(part) for part in requirement.maximum)
            reasons.append(
                f"{requirement.distribution}=={version} is outside supported range "
                f">={minimum},<{maximum}"
            )
    return _DependencyStatus(
        available=not reasons,
        versions=dict(sorted(versions.items())),
        reasons=tuple(reasons),
    )


def _is_executable_or_loadable_code(path: Path) -> bool:
    name = path.name.lower()
    suffix = path.suffix.lower()
    mode = path.stat().st_mode
    with path.open("rb") as handle:
        prefix = handle.read(4)
    return (
        suffix in _BANNED_CODE_SUFFIXES
        or re.search(r"\.so(?:\.|$)", name) is not None
        or bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
        or prefix.startswith(_EXECUTABLE_MAGIC_PREFIXES)
    )


def _parameter(
    config: QuantizationConfig,
    name: str,
    expected: type[Any],
    *,
    default: Any = _MISSING,
) -> Any:
    value = config.parameters.get(name, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise ValueError(f"quantization.parameters.{name} is required")
        return default
    if expected is int:
        valid = type(value) is int
    elif expected is float:
        valid = type(value) in {int, float}
    else:
        valid = isinstance(value, expected)
    if not valid:
        raise ValueError(
            f"quantization.parameters.{name} must be {expected.__name__}, "
            f"not {type(value).__name__}"
        )
    if expected is float:
        return float(cast(int | float, value))
    return value


def _common_settings(
    config: QuantizationConfig,
    allowed: frozenset[str],
    *,
    allow_cpu: bool = False,
) -> _CommonSettings:
    unknown = sorted(set(config.parameters).difference(allowed))
    if unknown:
        raise ValueError("unsupported quantization parameter(s): " + ", ".join(unknown))
    local_files_only = cast(bool, _parameter(config, "local_files_only", bool))
    device = cast(str, _parameter(config, "device", str))
    match = _CUDA_DEVICE.fullmatch(device)
    if device == _CPU_DEVICE and allow_cpu:
        return _CommonSettings(local_files_only=local_files_only, device=device)
    if match is None:
        expected = (
            "'cpu' or an explicit CUDA device (cuda:N)"
            if allow_cpu
            else ("an explicit CUDA device (cuda:N)")
        )
        raise ValueError(f"quantization.parameters.device must be {expected}")
    if torch.cuda.is_available() and int(match.group(1)) >= torch.cuda.device_count():
        raise ValueError(f"configured CUDA device does not exist: {device}")
    return _CommonSettings(local_files_only=local_files_only, device=device)


def _gptqmodel_settings(config: QuantizationConfig, backend: str) -> _GPTQModelSettings:
    allowed = {
        "local_files_only",
        "device",
        "bits",
        "group_size",
        "sym",
        "batch_size",
        "format",
    }
    if backend == "gptq":
        allowed.update({"desc_act", "act_group_aware", "damp_percent"})
    common = _common_settings(config, frozenset(allowed), allow_cpu=backend == "gptq")
    bits = cast(int, _parameter(config, "bits", int, default=4))
    group_size = cast(int, _parameter(config, "group_size", int, default=128))
    sym = cast(bool, _parameter(config, "sym", bool, default=(backend == "gptq")))
    batch_size = cast(int, _parameter(config, "batch_size", int, default=1))
    output_format = cast(
        str, _parameter(config, "format", str, default="gemm" if backend == "awq" else "gptq")
    )
    if bits != 4:
        raise ValueError(f"the validated {backend.upper()} adapter currently requires bits=4")
    if backend == "gptq" and common.device == _CPU_DEVICE and not sym:
        raise ValueError("the validated GPTQ CPU runtime transition requires sym=true")
    if group_size != -1 and group_size <= 0:
        raise ValueError("group_size must be -1 or a positive integer")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    expected_format = "gemm" if backend == "awq" else "gptq"
    if output_format != expected_format:
        raise ValueError(
            f"the validated {backend.upper()} adapter requires format={expected_format!r}"
        )
    desc_act = False
    act_group_aware = True
    damp_percent = 0.1
    if backend == "gptq":
        desc_act = cast(bool, _parameter(config, "desc_act", bool, default=False))
        act_group_aware = cast(
            bool, _parameter(config, "act_group_aware", bool, default=not desc_act)
        )
        damp_percent = cast(float, _parameter(config, "damp_percent", float, default=0.1))
        if desc_act and act_group_aware:
            raise ValueError("desc_act=true is incompatible with act_group_aware=true")
        if not 0.0 < damp_percent < 1.0:
            raise ValueError("damp_percent must be between 0 and 1")
    return _GPTQModelSettings(
        local_files_only=common.local_files_only,
        device=common.device,
        bits=bits,
        group_size=group_size,
        sym=sym,
        batch_size=batch_size,
        format=output_format,
        desc_act=desc_act,
        act_group_aware=act_group_aware,
        damp_percent=damp_percent,
    )


def _fp8_settings(config: QuantizationConfig) -> _FP8Settings:
    common = _common_settings(
        config,
        frozenset(
            {
                "local_files_only",
                "device",
                "activation_scheme",
                "weight_block_rows",
                "weight_block_columns",
                "scale_fmt",
            }
        ),
    )
    activation = cast(str, _parameter(config, "activation_scheme", str, default="dynamic")).lower()
    rows = cast(int, _parameter(config, "weight_block_rows", int, default=128))
    columns = cast(int, _parameter(config, "weight_block_columns", int, default=128))
    scale_fmt = cast(str, _parameter(config, "scale_fmt", str, default="float"))
    if activation != "dynamic":
        raise ValueError("the validated fine-grained FP8 adapter requires dynamic activations")
    if rows <= 0 or columns <= 0:
        raise ValueError("FP8 weight block dimensions must be positive")
    if scale_fmt not in {"float", "ue8m0"}:
        raise ValueError("scale_fmt must be 'float' or 'ue8m0'")
    return _FP8Settings(
        local_files_only=common.local_files_only,
        device=common.device,
        activation_scheme=activation,
        block_rows=rows,
        block_columns=columns,
        scale_fmt=scale_fmt,
    )


def _model_reasons(model: ModelDescriptor) -> tuple[str, ...]:
    reasons: list[str] = []
    if model.revision is None or _COMMIT_REVISION.fullmatch(model.revision) is None:
        reasons.append("model.revision must be an immutable 40-character commit SHA")
    if model.capabilities.requires_remote_code:
        reasons.append("models requiring remote code are unsupported")
    if not model.capabilities.supports_text:
        reasons.append("the optional quantizers require a text causal language model")
    if _HF_REPOSITORY.fullmatch(model.model_id) is None:
        reasons.append("the production optional backends require a namespace/repository HF source")
    if model.architecture not in {"unresolved", _NATIVE_MIXTRAL_ARCHITECTURE}:
        reasons.append(
            "this release gate is intentionally limited to the validated MixtralForCausalLM adapter"
        )
    return tuple(reasons)


def _final_source_identity_reasons(model: ModelDescriptor) -> tuple[str, ...]:
    reasons: list[str] = []
    if model.architecture != _NATIVE_MIXTRAL_ARCHITECTURE:
        reasons.append(
            "loaded source architecture must be exactly MixtralForCausalLM, "
            f"not {model.architecture!r}"
        )
    resolved_leaf = model.resolved_class.rpartition(".")[2]
    if resolved_leaf != _NATIVE_MIXTRAL_ARCHITECTURE:
        reasons.append(
            "loaded source resolved class must have the exact MixtralForCausalLM class name, "
            f"not {model.resolved_class!r}"
        )
    return tuple(reasons)


def _runtime_reasons(
    runtime: RuntimeCapabilities,
    *,
    backend: str,
    device: str | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not runtime.available:
        reasons.extend(runtime.reasons or ("selected runtime is unavailable",))
    cpu_gptq = backend == "gptq" and device == _CPU_DEVICE
    if cpu_gptq:
        if runtime.backend != "torch_eager_cpu" or "cpu" not in runtime.devices:
            reasons.append(
                "CPU GPTQ requires the torch_eager_cpu runtime and explicit device='cpu'"
            )
    elif runtime.backend != "torch_eager_cuda" or "cuda" not in runtime.devices:
        reasons.append("this production adapter requires the torch_eager_cuda runtime")
    if (
        backend == "fp8"
        and device is not None
        and runtime.available
        and runtime.backend == "torch_eager_cuda"
        and "cuda" in runtime.devices
    ):
        capability = _cuda_compute_capability(device)
        if capability is None:
            reasons.append(f"unable to query CUDA compute capability for configured {device}")
        elif capability < (9, 0):
            reasons.append(
                "fine-grained FP8 is fail-closed to CUDA compute capability >=9.0; "
                f"configured {device} reports {capability[0]}.{capability[1]}"
            )
    return tuple(reasons)


def _backend_model_reasons(
    model: ModelDescriptor,
    *,
    backend: str,
    device: str | None,
) -> tuple[str, ...]:
    if backend != "gptq" or device != _CPU_DEVICE:
        return ()
    if (model.model_id, model.revision) == (_STORIES15M_MODEL_ID, _STORIES15M_REVISION):
        return ()
    return (
        "CPU GPTQ is validated only for the pinned ggml-org/stories15M_MOE revision "
        f"{_STORIES15M_REVISION}",
    )


def _cuda_compute_capability(device: str) -> tuple[int, int] | None:
    match = _CUDA_DEVICE.fullmatch(device)
    if match is None or not torch.cuda.is_available():
        return None
    device_index = int(match.group(1))
    try:
        major, minor = torch.cuda.get_device_capability(device_index)
    except (AssertionError, IndexError, RuntimeError, ValueError):
        return None
    return int(major), int(minor)


def _calibration_count(calibration: CalibrationArtifact, name: str) -> int:
    value = calibration.statistics.get(name)
    if value is None or value < 0.0 or not value.is_integer():
        raise QuantizationError(
            f"Calibration provenance requires a non-negative integral {name}",
            component="optional_quantizer",
        )
    return int(value)


def _source_dtype(source: LoadedModel) -> torch.dtype:
    module = source.model
    if not isinstance(module, nn.Module):
        raise QuantizationError(
            "Pinned-source quantization requires a torch.nn.Module baseline",
            component="optional_quantizer",
        )
    dtypes = {value.dtype for value in module.parameters() if value.is_floating_point()}
    if not dtypes:
        raise QuantizationError(
            "The baseline exposes no floating-point parameters", component="optional_quantizer"
        )
    supported = {torch.float32, torch.float16, torch.bfloat16}
    unsupported = dtypes.difference(supported)
    if unsupported:
        rendered = ", ".join(str(dtype) for dtype in sorted(unsupported, key=str))
        raise QuantizationError(
            f"Unsupported source dtype for optional quantization: {rendered}",
            component="optional_quantizer",
        )
    if len(dtypes) != 1:
        rendered = ", ".join(str(dtype) for dtype in sorted(dtypes, key=str))
        raise QuantizationError(
            "Pinned-source quantization requires one consistent floating-point dtype; "
            f"observed {rendered}",
            component="optional_quantizer",
        )
    return next(iter(dtypes))


def _module_state_sha256(module: nn.Module) -> str:
    """Hash the exact persistent tensor state without dtype coercion."""

    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        if not isinstance(value, Tensor):
            raise QuantizationError(
                f"Candidate state entry {name!r} is not a tensor",
                component="optional_quantizer",
            )
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "dtype": str(tensor.dtype),
                "name": name,
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = tensor.view(torch.uint8).reshape(-1).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _validate_loaded_source_state(source: LoadedModel, *, backend: str) -> None:
    """Reject stale descriptors or source mutation before conversion/reload."""

    if not isinstance(source.model, nn.Module):
        raise QuantizationError(
            "Optional quantization source is not a torch module",
            component=backend,
        )
    observed = model_state_sha256(source.model)
    if observed != source.descriptor.checksum:
        raise QuantizationError(
            "Optional quantization source state differs from its immutable descriptor checksum",
            component=backend,
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _gptqmodel_cpu_manifest_partition(
    manifest: QuantizationManifest,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Require one complete INT4-selected/float32-excluded policy partition."""

    precision_map = manifest.module_precision_map
    selected = tuple(
        sorted(name for name, precision in precision_map.items() if precision == "int4")
    )
    excluded = tuple(
        sorted(name for name, precision in precision_map.items() if precision != "int4")
    )
    if (
        not selected
        or any(precision not in {"int4", "float32"} for precision in precision_map.values())
        or manifest.excluded_modules != excluded
        or set(selected).intersection(excluded)
        or len(selected) + len(excluded) != len(precision_map)
    ):
        raise QuantizationError(
            "CPU GPTQ manifest does not contain one exact INT4/float32 policy partition",
            component="gptq",
        )
    return selected, excluded


def _validate_gptqmodel_cpu_export_config(
    path: Path,
    recipe: dict[str, Any],
    manifest: QuantizationManifest,
) -> None:
    """Bind the exact behavior-affecting Transformers config to governed metadata."""

    config_path = path / "config.json"
    parameters = manifest.quantization_parameters
    expected_sha256 = parameters.get("export_config_sha256")
    expected_size = parameters.get("export_config_size_bytes")
    if (
        not config_path.is_file()
        or config_path.is_symlink()
        or recipe.get("config_file") != "config.json"
        or recipe.get("config_file_sha256") != expected_sha256
        or recipe.get("config_file_size_bytes") != expected_size
        or not isinstance(expected_sha256, str)
        or not isinstance(expected_size, int)
        or _file_sha256(config_path) != expected_sha256
        or config_path.stat().st_size != expected_size
    ):
        raise QuantizationError(
            "CPU GPTQ export config.json checksum or size differs from governed metadata",
            component="gptq",
        )


def _gptqmodel_cpu_reload_recipe(path: Path, model: QuantizedModel) -> dict[str, Any]:
    weights = path / "model.safetensors"
    if not weights.is_file() or weights.is_symlink():
        raise QuantizationError(
            "The audited CPU GPTQ reload boundary requires one model.safetensors file",
            component="gptq",
        )
    config_path = path / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise QuantizationError(
            "The audited CPU GPTQ reload boundary requires one safe config.json file",
            component="gptq",
        )
    config_sha256 = _file_sha256(config_path)
    config_size_bytes = config_path.stat().st_size
    manifest = model.manifest
    parameters = {
        **manifest.quantization_parameters,
        "export_config_sha256": config_sha256,
        "export_config_size_bytes": config_size_bytes,
    }
    manifest = manifest.model_copy(update={"quantization_parameters": parameters})
    model.manifest = manifest
    selected, _excluded = _gptqmodel_cpu_manifest_partition(manifest)
    selected_digest = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
    if (
        not selected
        or parameters.get("quantized_module_count") != len(selected)
        or parameters.get("quantized_module_names_sha256") != selected_digest
    ):
        raise QuantizationError(
            "CPU GPTQ reload metadata cannot bind the selected module inventory",
            component="gptq",
        )
    versions = {
        distribution: parameters.get(f"{key}_version")
        for distribution, key in (
            ("accelerate", "accelerate"),
            ("defuser", "defuser"),
            ("gptqmodel", "gptqmodel"),
            ("huggingface-hub", "huggingface_hub"),
            ("kernels", "kernels"),
            ("safetensors", "safetensors"),
            ("torch", "torch"),
            ("transformers", "transformers"),
        )
    }
    if versions != _GPTQMODEL_CPU_EXACT_VERSIONS:
        raise QuantizationError(
            "CPU GPTQ reload metadata does not contain the exact audited software matrix",
            component="gptq",
        )
    descriptor = model.loaded.descriptor
    return {
        "schema_version": "1.0",
        "adapter": "inkling_quant_lab_gptqmodel_cpu",
        "loader": ("inkling_quant_lab.quantization.optional.reload_gptqmodel_cpu_export"),
        "backend": "gptq",
        "model_id": descriptor.model_id,
        "revision": descriptor.revision,
        "resolved_class": descriptor.resolved_class,
        "architecture": descriptor.architecture,
        "source_model_checksum": descriptor.checksum,
        "software_versions": versions,
        "software_matrix_sha256": parameters.get("software_matrix_sha256"),
        "source_dtype": "float32",
        "checkpoint_qzero_format": "gptq_v1",
        "runtime_qzero_format": "gptq_v2",
        "config_file": "config.json",
        "config_file_sha256": config_sha256,
        "config_file_size_bytes": config_size_bytes,
        "weight_file": "model.safetensors",
        "weight_file_sha256": _file_sha256(weights),
        "weight_file_size_bytes": weights.stat().st_size,
        "selected_module_names": list(selected),
        "selected_module_count": len(selected),
        "selected_module_names_sha256": selected_digest,
        "excluded_module_count": len(manifest.excluded_modules),
        "excluded_state_sha256": parameters.get("excluded_state_sha256"),
        "candidate_state_sha256": parameters.get("candidate_state_sha256"),
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "strict_state_assignment": True,
        "policy_preserving_reload_supported": True,
        "validation_required_on_reload": True,
    }


def _direct_tensor_state(module: nn.Module) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {
        f"parameter:{name}": value for name, value in module.named_parameters(recurse=False)
    }
    state.update({f"buffer:{name}": value for name, value in module.named_buffers(recurse=False)})
    return dict(sorted(state.items()))


def _audit_gptqmodel_exclusions(
    source_module: nn.Module,
    candidate_module: nn.Module,
    excluded: tuple[str, ...],
    *,
    source_dtype: torch.dtype,
    backend: str,
) -> _GPTQModelExclusionAudit:
    """Prove protected runtime leaves retained their source kind, dtype, and state."""

    source_named = dict(source_module.named_modules())
    candidate_named = dict(candidate_module.named_modules())
    digest = hashlib.sha256()
    state_tensor_count = 0
    expanded_linear_count = 0
    expanded_pattern = re.compile(r"\.experts\.[0-9]+\.(?:gate_proj|up_proj|down_proj)$")
    for name in excluded:
        candidate = candidate_named.get(name)
        if candidate is None:
            raise QuantizationError(
                f"GPTQModel candidate is missing protected runtime module {name!r}",
                component=backend,
            )
        candidate_state = _direct_tensor_state(candidate)
        if not candidate_state:
            raise QuantizationError(
                f"GPTQModel protected runtime module {name!r} has no direct tensor state",
                component=backend,
            )
        bad_dtypes = tuple(
            state_name
            for state_name, value in candidate_state.items()
            if value.is_floating_point() and value.dtype != source_dtype
        )
        if bad_dtypes:
            raise QuantizationError(
                f"GPTQModel changed protected runtime dtype for {name!r}: " + ", ".join(bad_dtypes),
                component=backend,
            )

        source = source_named.get(name)
        if source is None:
            if expanded_pattern.search(name) is None or not isinstance(candidate, nn.Linear):
                raise QuantizationError(
                    f"GPTQModel cannot bind protected runtime module {name!r} to source state",
                    component=backend,
                )
            expanded_linear_count += 1
        else:
            if isinstance(source, nn.Linear):
                kind_matches = isinstance(candidate, nn.Linear)
            elif isinstance(source, nn.Embedding):
                kind_matches = isinstance(candidate, nn.Embedding)
            else:
                kind_matches = type(candidate) is type(source)
            if not kind_matches:
                raise QuantizationError(
                    f"GPTQModel changed protected runtime module kind for {name!r}",
                    component=backend,
                )
            source_state = _direct_tensor_state(source)
            if candidate_state.keys() != source_state.keys():
                raise QuantizationError(
                    f"GPTQModel changed protected runtime state schema for {name!r}",
                    component=backend,
                )
            for state_name, candidate_value in candidate_state.items():
                source_value = source_state[state_name]
                if (
                    candidate_value.shape != source_value.shape
                    or candidate_value.dtype != source_value.dtype
                    or not torch.equal(candidate_value.detach().cpu(), source_value.detach().cpu())
                ):
                    raise QuantizationError(
                        f"GPTQModel changed protected runtime tensor {name}.{state_name}",
                        component=backend,
                    )

        for state_name, value in candidate_state.items():
            tensor = value.detach().cpu().contiguous()
            metadata = json.dumps(
                {
                    "dtype": str(tensor.dtype),
                    "name": f"{name}.{state_name}",
                    "shape": list(tensor.shape),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            payload = tensor.view(torch.uint8).reshape(-1).numpy().tobytes()
            digest.update(len(metadata).to_bytes(8, "big"))
            digest.update(metadata)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
            state_tensor_count += 1
    return _GPTQModelExclusionAudit(
        module_count=len(excluded),
        state_tensor_count=state_tensor_count,
        state_sha256=digest.hexdigest(),
        expanded_linear_count=expanded_linear_count,
    )


def _policy_partition(
    source: LoadedModel, policy: ResolvedPolicyLike, target: str
) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...]]:
    precision_map = dict(sorted(policy.precision_map.items()))
    unsupported = sorted(
        name
        for name, precision in precision_map.items()
        if precision not in _FULL_PRECISIONS and precision != target
    )
    if unsupported:
        details = ", ".join(f"{name}={precision_map[name]}" for name in unsupported)
        raise CapabilityError(
            f"The {target} backend cannot implement resolved assignments: {details}",
            component="precision_policy",
        )
    selected = tuple(name for name, precision in precision_map.items() if precision == target)
    if not selected:
        raise CapabilityError(
            f"Resolved policy selects no modules for {target}", component="precision_policy"
        )
    excluded = tuple(name for name, precision in precision_map.items() if precision != target)
    module = source.model
    if isinstance(module, nn.Module):
        actual = {name for name, _value in module.named_modules()}
        missing_exclusions = tuple(name for name in excluded if name not in actual)
        if missing_exclusions:
            raise CapabilityError(
                "The backend cannot prove exclusions for non-addressable module slices: "
                + ", ".join(missing_exclusions),
                component="precision_policy",
            )
    return precision_map, selected, excluded


def _fused_mixtral_expert_projection_expansions(
    source: LoadedModel,
) -> dict[str, tuple[str, str, str]]:
    module = source.model
    if not isinstance(module, nn.Module):
        raise CapabilityError(
            "GPTQModel policy realization requires a torch.nn.Module source",
            component="precision_policy",
        )
    expansions: dict[str, tuple[str, str, str]] = {}
    for block_name, block in module.named_modules():
        if block.__class__.__name__ != "MixtralSparseMoeBlock":
            continue
        experts = getattr(block, "experts", None)
        router = getattr(block, "gate", None)
        if not isinstance(experts, nn.Module) or not isinstance(router, nn.Module):
            raise CapabilityError(
                f"Mixtral block {block_name} lacks validated gate/expert modules",
                component="precision_policy",
            )
        expert_count = getattr(experts, "num_experts", None)
        if expert_count is None:
            expert_count = getattr(router, "num_experts", None)
        if not isinstance(expert_count, int) or expert_count <= 0:
            raise CapabilityError(
                f"Mixtral block {block_name} lacks a valid fused expert count",
                component="precision_policy",
            )
        indexed_children = tuple(
            name
            for name, child in experts.named_children()
            if name.isdigit() and isinstance(child, nn.Module)
        )
        if len(indexed_children) == expert_count:
            continue
        values = (
            *tuple(experts.parameters(recurse=False)),
            *tuple(experts.buffers(recurse=False)),
        )
        if not values or any(value.ndim == 0 or value.shape[0] != expert_count for value in values):
            raise CapabilityError(
                f"Fused Mixtral experts in {block_name}.experts cannot be mapped by expert ID",
                component="precision_policy",
            )
        for expert in range(expert_count):
            logical = f"{block_name}.experts.{expert}"
            expansions[logical] = cast(
                tuple[str, str, str],
                tuple(f"{logical}.{projection}" for projection in _DEFUSED_EXPERT_PROJECTIONS),
            )
    return dict(sorted(expansions.items()))


def _gptqmodel_policy_partition(
    source: LoadedModel,
    policy: ResolvedPolicyLike,
    target: str,
) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...]]:
    precision_map = dict(sorted(policy.precision_map.items()))
    unsupported = sorted(
        name
        for name, precision in precision_map.items()
        if precision not in _FULL_PRECISIONS and precision != target
    )
    if unsupported:
        details = ", ".join(f"{name}={precision_map[name]}" for name in unsupported)
        raise CapabilityError(
            f"The {target} backend cannot implement resolved assignments: {details}",
            component="precision_policy",
        )
    dtype_precision = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }[_source_dtype(source)]
    module = cast(nn.Module, source.model)
    actual = dict(module.named_modules())
    expansions = _fused_mixtral_expert_projection_expansions(source)
    missing_logical = tuple(name for name in expansions if name not in precision_map)
    if missing_logical:
        raise CapabilityError(
            "Resolved policy omits fused expert slices required for exact GPTQModel exclusions: "
            + ", ".join(missing_logical),
            component="precision_policy",
        )

    realized: dict[str, str] = {}
    unknown: list[str] = []
    for name, precision in precision_map.items():
        expanded = expansions.get(name)
        if expanded is not None:
            if precision == target:
                raise CapabilityError(
                    "The common GPTQModel adapter keeps fused expert slices full precision; "
                    f"resolved policy selected {name}={target}",
                    component="precision_policy",
                )
            if precision != dtype_precision:
                raise CapabilityError(
                    "GPTQModel exclusions preserve the source reload dtype and cannot implement "
                    f"{name}={precision}; source dtype is {dtype_precision}",
                    component="precision_policy",
                )
            for runtime_name in expanded:
                if runtime_name in realized:
                    raise CapabilityError(
                        f"GPTQModel policy expansion collides at {runtime_name}",
                        component="precision_policy",
                    )
                realized[runtime_name] = precision
            continue
        if name not in actual:
            unknown.append(name)
            continue
        if precision != target and precision != dtype_precision:
            raise CapabilityError(
                "GPTQModel exclusions preserve the source reload dtype and cannot implement "
                f"{name}={precision}; source dtype is {dtype_precision}",
                component="precision_policy",
            )
        realized[name] = precision
    if unknown:
        raise CapabilityError(
            "The backend cannot prove assignments for non-addressable modules: "
            + ", ".join(sorted(unknown)),
            component="precision_policy",
        )

    expected_linears = {
        name for name, value in actual.items() if name and isinstance(value, nn.Linear)
    }
    expected_linears.update(
        runtime_name for expanded in expansions.values() for runtime_name in expanded
    )
    missing_linears = tuple(sorted(expected_linears.difference(realized)))
    if missing_linears:
        raise CapabilityError(
            "Resolved policy is incomplete for GPTQModel-addressable linear modules: "
            + ", ".join(missing_linears),
            component="precision_policy",
        )
    selected = tuple(sorted(name for name, precision in realized.items() if precision == target))
    if not selected:
        raise CapabilityError(
            f"Resolved policy selects no modules for {target}", component="precision_policy"
        )
    non_linear_selected = tuple(
        name for name in selected if name not in actual or not isinstance(actual[name], nn.Linear)
    )
    if non_linear_selected:
        raise CapabilityError(
            "GPTQModel target assignments must select addressable source nn.Linear leaves: "
            + ", ".join(non_linear_selected),
            component="precision_policy",
        )
    excluded = tuple(sorted(name for name, precision in realized.items() if precision != target))
    return precision_map, selected, excluded


def _verify_gptqmodel_inventory(
    module: nn.Module,
    *,
    selected: tuple[str, ...],
    quantized_type: type[nn.Module],
    backend: str,
    runtime_excluded: tuple[str, ...] = (),
    base_quantized_type: type[nn.Module] | None = None,
) -> tuple[str, ...]:
    named = dict(module.named_modules())
    missing_runtime = tuple(sorted(set((*selected, *runtime_excluded)).difference(named)))
    if missing_runtime:
        raise QuantizationError(
            "GPTQModel candidate is missing policy-realized runtime modules: "
            + ", ".join(missing_runtime),
            component=backend,
        )
    base_type = base_quantized_type or quantized_type
    observed = tuple(
        sorted(name for name, value in named.items() if name and isinstance(value, base_type))
    )
    if observed != tuple(sorted(selected)) or any(
        type(named[name]) is not quantized_type for name in observed
    ):
        raise QuantizationError(
            "GPTQModel did not realize the exact selected module inventory; "
            f"expected {len(selected)}, observed {len(observed)}",
            component=backend,
        )
    quantized_exclusions = tuple(
        name for name in runtime_excluded if isinstance(named[name], base_type)
    )
    if quantized_exclusions:
        raise QuantizationError(
            "GPTQModel quantized protected runtime exclusions: " + ", ".join(quantized_exclusions),
            component=backend,
        )
    return observed


def _exact_dynamic_exclusions(excluded: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    return {f"-:^{re.escape(name)}$": {} for name in excluded}


def _load_gptqmodel_api() -> Any:
    from gptqmodel import (
        BACKEND,
        GPTQModel,
        QuantizeConfig,
    )
    from gptqmodel.nn_modules.qlinear import (
        BaseQuantLinear,
    )
    from gptqmodel.nn_modules.qlinear import (
        torch as torch_qlinear_module,
    )
    from gptqmodel.nn_modules.qlinear.gemm_awq import (
        AwqGEMMQuantLinear,
    )
    from gptqmodel.nn_modules.qlinear.torch import (
        TorchQuantLinear,
    )
    from gptqmodel.nn_modules.qlinear.torch_awq import (
        AwqTorchQuantLinear,
    )
    from gptqmodel.quantization import (
        FORMAT,
        METHOD,
    )
    from gptqmodel.utils.model import (
        convert_gptq_v1_to_v2_format_module,
        convert_gptq_v2_to_v1_format_module,
    )

    return SimpleNamespace(
        AwqGEMMQuantLinear=AwqGEMMQuantLinear,
        AwqTorchQuantLinear=AwqTorchQuantLinear,
        BACKEND=BACKEND,
        BaseQuantLinear=BaseQuantLinear,
        FORMAT=FORMAT,
        GPTQModel=GPTQModel,
        METHOD=METHOD,
        QuantizeConfig=QuantizeConfig,
        TorchQLinearModule=torch_qlinear_module,
        TorchQuantLinear=TorchQuantLinear,
        convert_gptq_v1_to_v2_format_module=convert_gptq_v1_to_v2_format_module,
        convert_gptq_v2_to_v1_format_module=convert_gptq_v2_to_v1_format_module,
    )


def _apply_gptqmodel_580_awq_pack_shim(api: Any, version: str) -> bool:
    awq_type = api.AwqTorchQuantLinear
    if hasattr(awq_type, "pack"):
        return False
    if version != "5.8.0":
        raise QuantizationError(
            "The selected GPTQModel AWQ Torch class has no pack method outside the audited "
            "5.8.0 compatibility boundary",
            component="awq",
        )
    pack = getattr(api.AwqGEMMQuantLinear, "pack", None)
    if not callable(pack):
        raise QuantizationError(
            "GPTQModel 5.8.0 does not expose the audited device-agnostic AWQ pack method",
            component="awq",
        )
    awq_type.pack = pack
    return True


def _remove_gptqmodel_580_awq_pack_shim(api: Any, applied: bool) -> None:
    if not applied:
        return
    awq_type = api.AwqTorchQuantLinear
    if "pack" not in awq_type.__dict__:
        raise QuantizationError(
            "GPTQModel AWQ pack compatibility binding disappeared before restoration",
            component="awq",
        )
    delattr(awq_type, "pack")
    if hasattr(awq_type, "pack"):
        raise QuantizationError(
            "GPTQModel AWQ pack compatibility binding was not restored",
            component="awq",
        )


def _normalize_gptqmodel_580_cpu_bias_contract(
    api: Any,
    source_module: nn.Module,
    candidate_module: nn.Module,
    selected: tuple[str, ...],
    version: str,
) -> int:
    """Restore explicit ``bias=None`` omitted by 5.8 in-memory CPU packing."""

    if version != "5.8.0":
        raise QuantizationError(
            "GPTQModel CPU bias normalization is only audited for version 5.8.0",
            component="gptq",
        )
    source_named = dict(source_module.named_modules())
    candidate_named = dict(candidate_module.named_modules())
    repaired = 0
    for name in selected:
        source_layer = source_named.get(name)
        candidate_layer = candidate_named.get(name)
        if (
            not isinstance(source_layer, nn.Linear)
            or type(candidate_layer) is not api.TorchQuantLinear
        ):
            raise QuantizationError(
                f"GPTQModel CPU bias audit cannot resolve selected layer {name!r}",
                component="gptq",
            )
        source_bias = getattr(source_layer, "bias", _MISSING)
        candidate_bias = getattr(candidate_layer, "bias", _MISSING)
        if source_bias is _MISSING:
            raise QuantizationError(
                f"Source linear {name!r} does not expose an auditable bias contract",
                component="gptq",
            )
        expects_bias = source_bias is not None
        if candidate_bias is _MISSING:
            if expects_bias:
                raise QuantizationError(
                    f"GPTQModel dropped the non-null source bias for {name!r}",
                    component="gptq",
                )
            cast(Any, candidate_layer).bias = None
            repaired += 1
            continue
        if (candidate_bias is not None) != expects_bias:
            raise QuantizationError(
                f"GPTQModel changed the source bias presence for {name!r}",
                component="gptq",
            )
    return repaired


def _reset_gptqmodel_torch_qlinear_caches(qlinear: nn.Module) -> None:
    for method_name in ("_stream_reset_cache", "clear_weight_cache", "_reset_prefetch_state"):
        method = getattr(qlinear, method_name, None)
        if callable(method):
            method()


def _transition_gptqmodel_qzero_format(
    transition: _GPTQModelRuntimeTransition,
    *,
    source_format: int,
    target_format: int,
) -> int:
    """Atomically cross GPTQModel's checkpoint/runtime qzero boundary."""

    if (source_format, target_format) not in {(1, 2), (2, 1)}:
        raise ValueError("GPTQModel qzero transition must be between formats 1 and 2")
    named = dict(transition.module.named_modules())
    qlinears: list[nn.Module] = []
    for name in transition.qlinear_names:
        value = named.get(name)
        if value is None or type(value) is not transition.api.TorchQuantLinear:
            raise QuantizationError(
                f"GPTQModel qzero transition cannot resolve selected layer {name!r}",
                component="gptq",
            )
        qzero_format = getattr(value, "qzero_format", None)
        if not callable(qzero_format) or qzero_format() != source_format:
            raise QuantizationError(
                f"GPTQModel layer {name!r} is not in required qzero format {source_format}",
                component="gptq",
            )
        qlinears.append(value)

    transitioned: list[nn.Module] = []
    try:
        for qlinear in qlinears:
            if target_format == 2:
                transition.api.convert_gptq_v1_to_v2_format_module(
                    module=qlinear,
                    bits=transition.quantize_config.bits,
                    pack_dtype=transition.quantize_config.pack_dtype,
                )
            else:
                transition.api.convert_gptq_v2_to_v1_format_module(
                    module=qlinear,
                    quantize_config=transition.quantize_config,
                )
            transitioned.append(qlinear)
            if cast(Any, qlinear).qzero_format() != target_format:
                raise QuantizationError(
                    f"GPTQModel did not realize qzero format {target_format}",
                    component="gptq",
                )
            _reset_gptqmodel_torch_qlinear_caches(qlinear)
    except Exception as error:
        rollback_errors: list[str] = []
        for qlinear in reversed(transitioned):
            try:
                if source_format == 1:
                    transition.api.convert_gptq_v2_to_v1_format_module(
                        module=qlinear,
                        quantize_config=transition.quantize_config,
                    )
                else:
                    transition.api.convert_gptq_v1_to_v2_format_module(
                        module=qlinear,
                        bits=transition.quantize_config.bits,
                        pack_dtype=transition.quantize_config.pack_dtype,
                    )
                _reset_gptqmodel_torch_qlinear_caches(qlinear)
            except Exception as rollback_error:
                rollback_errors.append(str(rollback_error))
        rollback = "" if not rollback_errors else "; rollback errors: " + "; ".join(rollback_errors)
        raise QuantizationError(
            f"GPTQModel qzero format transition failed: {error}{rollback}",
            component="gptq",
        ) from error
    return len(transitioned)


def _snapshot_gptqmodel_qzeros(
    transition: _GPTQModelRuntimeTransition,
    *,
    required_format: int,
) -> tuple[Tensor, ...]:
    named = dict(transition.module.named_modules())
    snapshots: list[Tensor] = []
    for name in transition.qlinear_names:
        qlinear = named.get(name)
        qzero_format = getattr(qlinear, "qzero_format", None)
        qzeros = getattr(qlinear, "qzeros", None)
        if (
            type(qlinear) is not transition.api.TorchQuantLinear
            or not callable(qzero_format)
            or qzero_format() != required_format
            or not isinstance(qzeros, Tensor)
        ):
            raise QuantizationError(
                f"GPTQModel cannot snapshot qzeros for selected layer {name!r}",
                component="gptq",
            )
        snapshots.append(qzeros.detach().clone())
    return tuple(snapshots)


def _verify_gptqmodel_qzero_snapshots(
    transition: _GPTQModelRuntimeTransition,
    snapshots: tuple[Tensor, ...],
    *,
    required_format: int,
) -> None:
    current = _snapshot_gptqmodel_qzeros(transition, required_format=required_format)
    if len(current) != len(snapshots) or any(
        not torch.equal(before, after) for before, after in zip(snapshots, current, strict=True)
    ):
        raise QuantizationError(
            "GPTQModel export did not restore the exact runtime qzero tensors",
            component="gptq",
        )


def _restore_gptqmodel_qzero_snapshots(
    transition: _GPTQModelRuntimeTransition,
    snapshots: tuple[Tensor, ...],
    *,
    required_format: int,
) -> None:
    """Repair runtime qzeros after a failed or state-mutating upstream save."""

    if len(snapshots) != len(transition.qlinear_names):
        raise QuantizationError(
            "GPTQModel cannot repair an incomplete qzero snapshot",
            component="gptq",
        )
    named = dict(transition.module.named_modules())
    try:
        for name, snapshot in zip(transition.qlinear_names, snapshots, strict=True):
            qlinear = named.get(name)
            if type(qlinear) is not transition.api.TorchQuantLinear:
                raise TypeError(f"selected layer {name!r} is no longer TorchQuantLinear")
            concrete_qlinear = cast(Any, qlinear)
            concrete_qlinear._buffers["qzeros"] = snapshot.detach().clone()
            qzero_format = getattr(qlinear, "qzero_format", None)
            if not callable(qzero_format):
                raise TypeError(f"selected layer {name!r} has no qzero format setter")
            qzero_format(required_format)
            _reset_gptqmodel_torch_qlinear_caches(cast(nn.Module, qlinear))
        _verify_gptqmodel_qzero_snapshots(
            transition,
            snapshots,
            required_format=required_format,
        )
    except Exception as error:
        if isinstance(error, QuantizationError):
            raise
        raise QuantizationError(
            f"GPTQModel could not repair runtime qzeros after export failure: {error}",
            component="gptq",
        ) from error


def _post_init_gptqmodel_580_cpu_torch(
    api: Any,
    module: nn.Module,
    selected: tuple[str, ...],
    version: str,
) -> _GPTQModelCPUPostInit:
    """Initialize GPTQModel's packed CPU layers without retaining a global patch."""

    if version != "5.8.0":
        raise QuantizationError(
            "GPTQModel CPU eager post-initialization is only audited for version 5.8.0",
            component="gptq",
        )
    torch_qlinear_module = getattr(api, "TorchQLinearModule", None)
    original_compile = getattr(torch_qlinear_module, "torch_compile", None)
    if not callable(original_compile):
        raise QuantizationError(
            "GPTQModel 5.8.0 does not expose the audited Torch compilation binding",
            component="gptq",
        )
    torch_qlinear_module = cast(Any, torch_qlinear_module)
    named = dict(module.named_modules())
    qlinears: list[nn.Module] = []
    for name in selected:
        value = named.get(name)
        if value is None or type(value) is not api.TorchQuantLinear:
            raise QuantizationError(
                f"GPTQModel CPU post-initialization cannot resolve selected layer {name!r}",
                component="gptq",
            )
        qzero_format = getattr(value, "qzero_format", None)
        buffers = tuple(
            getattr(value, buffer_name, None)
            for buffer_name in ("qweight", "qzeros", "scales", "g_idx")
        )
        if (
            bool(getattr(value, "optimized", False))
            or not bool(getattr(value, "enable_wf_unsqueeze", False))
            or not callable(qzero_format)
            or qzero_format() != 2
            or any(
                not isinstance(buffer, Tensor) or buffer.device.type != "cpu" for buffer in buffers
            )
        ):
            raise QuantizationError(
                f"GPTQModel CPU layer {name!r} is not in the audited pre-init runtime state",
                component="gptq",
            )
        qlinears.append(value)

    compile_calls = 0

    def eager_identity(target: Any, *_args: Any, **_kwargs: Any) -> Any:
        nonlocal compile_calls
        compile_calls += 1
        return target

    try:
        torch_qlinear_module.torch_compile = eager_identity
        for name, qlinear in zip(selected, qlinears, strict=True):
            post_init = getattr(qlinear, "post_init", None)
            if not callable(post_init):
                raise QuantizationError(
                    f"GPTQModel CPU layer {name!r} has no post_init method",
                    component="gptq",
                )
            post_init()
            if any(
                getattr(qlinear, buffer_name, None) is None
                for buffer_name in ("wf_unsqueeze_zero", "wf_unsqueeze_neg_one")
            ):
                raise QuantizationError(
                    f"GPTQModel CPU layer {name!r} did not initialize unpack buffers",
                    component="gptq",
                )
            for buffer_name in ("wf_unsqueeze_zero", "wf_unsqueeze_neg_one"):
                buffer = getattr(qlinear, buffer_name)
                if (
                    not isinstance(buffer, Tensor)
                    or buffer.dtype != torch.int32
                    or buffer.device != qlinear.qweight.device
                    or buffer_name not in qlinear._buffers
                    or buffer_name not in qlinear._non_persistent_buffers_set
                ):
                    raise QuantizationError(
                        f"GPTQModel CPU layer {name!r} has an invalid {buffer_name} buffer",
                        component="gptq",
                    )
            if not bool(getattr(qlinear, "optimized", False)):
                raise QuantizationError(
                    f"GPTQModel CPU layer {name!r} did not complete post-initialization",
                    component="gptq",
                )
    except QuantizationError:
        raise
    except Exception as error:
        raise QuantizationError(
            f"GPTQModel CPU eager post-initialization failed: {error}",
            component="gptq",
        ) from error
    finally:
        torch_qlinear_module.torch_compile = original_compile
    if compile_calls != len(qlinears):
        raise QuantizationError(
            "GPTQModel CPU eager post-initialization did not bypass compilation exactly once "
            "per selected layer",
            component="gptq",
        )
    return _GPTQModelCPUPostInit(
        module_count=len(qlinears),
        compile_call_count=compile_calls,
    )


def _capture_native_mixtral_class_binding() -> DefuserMixtralBindings:
    try:
        return capture_defuser_mixtral_bindings()
    except Exception as error:
        raise QuantizationError(
            f"Failed to capture Transformers globals before Defuser conversion: {error}",
            component="gptqmodel",
        ) from error


def _restore_native_mixtral_class_binding(binding: DefuserMixtralBindings) -> bool:
    try:
        return restore_defuser_mixtral_bindings(binding)
    except Exception as error:
        raise QuantizationError(
            f"Failed to restore Transformers globals after Defuser conversion: {error}",
            component="gptqmodel",
        ) from error


def _load_huggingface_hub_api() -> Any:
    from huggingface_hub import snapshot_download

    return SimpleNamespace(snapshot_download=snapshot_download)


def _audit_exact_stories15m_snapshot(path: Path) -> None:
    from inkling_quant_lab.public_moe_tensor_parallel import audit_stories15m_snapshot

    audit_stories15m_snapshot(path)


def _materialize_pinned_source(source: ModelDescriptor, *, local_files_only: bool) -> Path:
    revision = cast(str, source.revision)
    api = _load_huggingface_hub_api()
    exact_stories15m = (source.model_id, revision) == (
        _STORIES15M_MODEL_ID,
        _STORIES15M_REVISION,
    )
    allow_patterns = (
        list(_STORIES15M_SOURCE_FILES) if exact_stories15m else list(_SAFE_SNAPSHOT_PATTERNS)
    )
    try:
        snapshot = api.snapshot_download(
            repo_id=source.model_id,
            repo_type="model",
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=allow_patterns,
            ignore_patterns=list(_UNSAFE_SNAPSHOT_PATTERNS),
        )
    except Exception as error:
        raise QuantizationError(
            f"Unable to materialize immutable safetensors source snapshot: {error}",
            component="optional_quantizer",
            remediation=(
                "Verify the pinned Hugging Face commit, cache/network mode, credentials, and "
                "safetensors checkpoint inventory."
            ),
        ) from error
    path = Path(snapshot).resolve()
    if not path.is_dir():
        raise QuantizationError(
            "Pinned Hugging Face source snapshot is not a directory",
            component="optional_quantizer",
        )
    if exact_stories15m:
        try:
            _audit_exact_stories15m_snapshot(path)
        except (OSError, RuntimeError, ValueError) as error:
            raise QuantizationError(
                f"Pinned Stories15M source failed the exact checksum/tensor audit: {error}",
                component="optional_quantizer",
            ) from error
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() in _BANNED_WEIGHT_SUFFIXES:
            raise QuantizationError(
                "Pinned Hugging Face source snapshot contains a pickle-capable checkpoint",
                component="optional_quantizer",
            )
        if _is_executable_or_loadable_code(candidate):
            raise QuantizationError(
                "Pinned Hugging Face source snapshot contains executable or loadable code: "
                f"{candidate.name}",
                component="optional_quantizer",
            )
    if not any(candidate.suffix == ".safetensors" for candidate in path.rglob("*")):
        raise QuantizationError(
            "Pinned Hugging Face source snapshot contains no safetensors checkpoint",
            component="optional_quantizer",
        )
    return path


def _load_transformers_fp8_api() -> Any:
    from transformers import AutoModelForCausalLM, FineGrainedFP8Config

    return SimpleNamespace(
        AutoModelForCausalLM=AutoModelForCausalLM,
        FineGrainedFP8Config=FineGrainedFP8Config,
    )


class UnavailableOptionalQuantizer:
    """Compatibility fixture for callers testing a deliberately missing backend."""

    def __init__(self, name: str, install_extra: str, category: str) -> None:
        self.name = name
        self.install_extra = install_extra
        self.category = category

    def check_support(
        self,
        model: ModelDescriptor,
        runtime: RuntimeCapabilities,
        config: QuantizationConfig,
    ) -> SupportReport:
        del model, runtime, config
        return SupportReport(
            available=False,
            supported=False,
            component=self.name,
            reasons=(f"{self.category} backend is not installed or validated",),
            install_extra=self.install_extra,
            remediation=(
                "No validated implementation is bundled for this synthetic missing-backend fixture."
            ),
        )

    def _raise(self) -> NoReturn:
        raise CapabilityError(
            f"Optional quantizer '{self.name}' is unavailable",
            component=self.name,
            remediation="Use a registered production adapter or a CPU reference backend.",
        )

    def calibrate(
        self,
        model: LoadedModel,
        samples: tuple[ModelBatch, ...],
        config: QuantizationConfig,
    ) -> CalibrationArtifact | None:
        del model, samples, config
        self._raise()

    def quantize(
        self,
        model: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        del model, policy, calibration, config
        self._raise()

    def export(
        self, model: QuantizedModel, destination: Path, config: QuantizationConfig
    ) -> ExportArtifact:
        del model, destination, config
        self._raise()


class _PinnedSourceQuantizer:
    lifecycle: ClassVar[Literal["pinned_source_reload"]] = "pinned_source_reload"
    name: str
    target_precision: str
    install_extra: str
    required_method: str
    requirements: ClassVar[tuple[_DependencyRequirement, ...]]

    def __init__(self) -> None:
        self._runtime: RuntimeCapabilities | None = None
        self._prepared: _PreparedCalibration | None = None
        self._exporters: dict[int, _ExportState] = {}

    def _settings(self, config: QuantizationConfig) -> _CommonSettings:
        raise NotImplementedError

    def _requires_calibration(self) -> bool:
        return False

    def _partition_policy(
        self,
        source: LoadedModel,
        policy: ResolvedPolicyLike,
    ) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...]]:
        return _policy_partition(source, policy, self.target_precision)

    def check_support(
        self,
        model: ModelDescriptor,
        runtime: RuntimeCapabilities,
        config: QuantizationConfig,
    ) -> SupportReport:
        self._runtime = runtime
        dependencies = _dependency_status(self.requirements)
        settings: _CommonSettings | None = None
        settings_error: str | None = None
        try:
            settings = self._settings(config)
        except ValueError as error:
            settings_error = str(error)
        reasons = [
            *dependencies.reasons,
            *_runtime_reasons(
                runtime,
                backend=self.name,
                device=settings.device if settings is not None else None,
            ),
        ]
        if (
            self.name == "gptq"
            and settings is not None
            and settings.device == _CPU_DEVICE
            and dependencies.available
        ):
            reasons.extend(
                f"CPU GPTQ requires {distribution}=={expected}, found "
                f"{dependencies.versions.get(distribution, 'unavailable')}"
                for distribution, expected in _GPTQMODEL_CPU_EXACT_VERSIONS.items()
                if dependencies.versions.get(distribution) != expected
            )
        reasons.extend(_model_reasons(model))
        reasons.extend(
            _backend_model_reasons(
                model,
                backend=self.name,
                device=settings.device if settings is not None else None,
            )
        )
        if config.method != self.required_method:
            reasons.append(
                f"quantization.method must be {self.required_method!r} for backend {self.name!r}"
            )
        if self._requires_calibration() and config.calibration is None:
            reasons.append(f"{self.name.upper()} requires an explicit calibration dataset")
        if not self._requires_calibration() and config.calibration is not None:
            reasons.append(f"{self.name.upper()} does not consume calibration data")
        if settings_error is not None:
            reasons.append(settings_error)
        supported = not reasons
        return SupportReport(
            available=dependencies.available,
            supported=supported,
            component=self.name,
            reasons=tuple(reasons),
            warnings=(
                "candidate lifecycle reloads the exact pinned source; the measured baseline "
                "object is not mutated",
            ),
            install_extra=None if supported else self.install_extra,
            remediation=(
                None
                if supported
                else (
                    f"Install `uv sync --extra {self.install_extra}` and use the documented "
                    "backend/model/runtime matrix with an immutable safetensors checkpoint."
                )
            ),
            supported_precisions=(
                self.target_precision,
                "float32",
                "float16",
                "bfloat16",
            ),
        )

    def _require_support(self, source: LoadedModel, config: QuantizationConfig) -> None:
        if self._runtime is None:
            raise CapabilityError(
                "check_support must run before optional quantization",
                component=self.name,
            )
        report = self.check_support(source.descriptor, self._runtime, config)
        if not report.available or not report.supported:
            raise CapabilityError(
                f"Optional quantizer {self.name!r} is unavailable or unsupported: "
                + report.message(),
                component=self.name,
                remediation=report.remediation,
            )

    def calibrate(
        self,
        model: LoadedModel,
        samples: tuple[ModelBatch, ...],
        config: QuantizationConfig,
    ) -> CalibrationArtifact | None:
        if not self._requires_calibration():
            if samples:
                raise CapabilityError(
                    f"{self.name.upper()} is weight-only and does not consume calibration data",
                    component=self.name,
                )
            self._prepared = None
            return None
        calibration = config.calibration
        if calibration is None:
            raise CapabilityError(
                f"{self.name.upper()} requires quantization.calibration",
                component=self.name,
            )
        token_samples = _token_samples(samples)
        if not token_samples:
            raise CapabilityError(
                f"{self.name.upper()} calibration received no samples", component=self.name
            )
        if len(token_samples) > calibration.max_samples:
            raise CapabilityError(
                f"{self.name.upper()} calibration exceeds max_samples={calibration.max_samples}",
                component=self.name,
            )
        short_samples = tuple(
            sample.sample_id
            for sample in token_samples
            if sum(sample.attention_mask) < _GPTQMODEL_MIN_ACTIVE_TOKENS
        )
        if short_samples:
            raise CapabilityError(
                "GPTQModel calibration samples must contain at least "
                f"{_GPTQMODEL_MIN_ACTIVE_TOKENS} active tokens: " + ", ".join(short_samples),
                component=self.name,
            )
        canonical = {
            "config": calibration.model_dump(mode="json"),
            "source_checksum": model.descriptor.checksum,
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    "input_ids": sample.input_ids,
                    "attention_mask": sample.attention_mask,
                }
                for sample in token_samples
            ],
        }
        checksum = hashlib.sha256(
            json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        artifact = CalibrationArtifact(
            config=calibration,
            sample_ids=tuple(sample.sample_id for sample in token_samples),
            statistics={
                "sample_count": float(len(token_samples)),
                "token_count": float(sum(sum(sample.attention_mask) for sample in token_samples)),
            },
            checksum=checksum,
        )
        self._prepared = _PreparedCalibration(
            source_checksum=model.descriptor.checksum,
            artifact_checksum=checksum,
            samples=token_samples,
        )
        return artifact

    def quantize(
        self,
        model: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        """Protocol-compatible entry point that preserves the explicit lifecycle."""

        return self.quantize_from_pinned_source(model, policy, calibration, config)

    def quantize_from_pinned_source(
        self,
        source: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        self._require_support(source, config)
        identity_reasons = _final_source_identity_reasons(source.descriptor)
        if identity_reasons:
            raise CapabilityError(
                "Pinned-source conversion requires a resolved native MixtralForCausalLM: "
                + "; ".join(identity_reasons),
                component=self.name,
            )
        _validate_loaded_source_state(source, backend=self.name)
        if self._requires_calibration():
            prepared = self._prepared
            if calibration is None or prepared is None:
                raise CapabilityError(
                    f"{self.name.upper()} quantization requires its prepared calibration",
                    component=self.name,
                )
            if (
                prepared.source_checksum != source.descriptor.checksum
                or prepared.artifact_checksum != calibration.checksum
            ):
                raise CapabilityError(
                    f"{self.name.upper()} calibration does not match the pinned source",
                    component=self.name,
                )
        precision_map, selected, excluded = self._partition_policy(source, policy)
        return self._quantize_source(
            source,
            precision_map,
            selected,
            excluded,
            calibration,
            config,
        )

    def _quantize_source(
        self,
        source: LoadedModel,
        precision_map: dict[str, str],
        selected: tuple[str, ...],
        excluded: tuple[str, ...],
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        raise NotImplementedError

    def _candidate(
        self,
        *,
        source: LoadedModel,
        module: nn.Module,
        precision_map: dict[str, str],
        excluded: tuple[str, ...],
        calibration: CalibrationArtifact | None,
        parameters: dict[str, str | int | float | bool],
        backend_version: str,
        elapsed: float,
        exporter: _ExportState,
    ) -> QuantizedModel:
        if elapsed <= 0.0:
            raise QuantizationError(
                "Optional quantization did not produce a measurable duration", component=self.name
            )
        module.eval()
        calibration_dataset = None
        sample_ids: tuple[str, ...] = ()
        calibration_checksum = None
        calibration_sample_count = None
        calibration_token_count = None
        if calibration is not None:
            calibration_dataset = {
                "dataset": calibration.config.dataset,
                "revision": calibration.config.revision,
                "split": calibration.config.split,
            }
            sample_ids = calibration.sample_ids
            calibration_checksum = calibration.checksum
            calibration_sample_count = _calibration_count(calibration, "sample_count")
            calibration_token_count = _calibration_count(calibration, "token_count")
            if calibration_sample_count != len(sample_ids):
                raise QuantizationError(
                    "Calibration sample_count does not match persisted stable sample IDs",
                    component=self.name,
                )
        manifest = QuantizationManifest(
            backend=self.name,
            backend_version=backend_version,
            method=self.required_method,
            source_model_checksum=source.descriptor.checksum,
            module_precision_map=precision_map,
            excluded_modules=excluded,
            calibration_dataset=calibration_dataset,
            calibration_sample_ids=sample_ids,
            calibration_checksum=calibration_checksum,
            calibration_sample_count=calibration_sample_count,
            calibration_token_count=calibration_token_count,
            quantization_parameters=dict(sorted(parameters.items())),
            serialized_size_bytes=model_storage_bytes(module),
            warnings=(
                "candidate was produced by reloading the exact pinned source through the "
                "upstream quantization lifecycle",
            ),
        )
        candidate = QuantizedModel(
            loaded=LoadedModel(
                model=module,
                tokenizer=source.tokenizer,
                descriptor=source.descriptor,
                load_time_seconds=elapsed,
                load_time_kind="candidate_pinned_source_quantization",
            ),
            manifest=manifest,
        )
        self._exporters[id(module)] = exporter
        return candidate

    def export(
        self, model: QuantizedModel, destination: Path, config: QuantizationConfig
    ) -> ExportArtifact:
        state = self._exporters.get(id(model.loaded.model))
        if state is None:
            raise QuantizationError(
                "Candidate is not owned by this optional quantizer instance",
                component=self.name,
            )
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite quantizer export: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        )
        reload_recipe: dict[str, Any]
        try:
            if state.save_kind == "gptqmodel":
                transitioned = False
                runtime_qzero_snapshots: tuple[Tensor, ...] = ()
                transition = state.runtime_transition
                if transition is not None:
                    runtime_qzero_snapshots = _snapshot_gptqmodel_qzeros(
                        transition,
                        required_format=2,
                    )
                    _transition_gptqmodel_qzero_format(
                        transition,
                        source_format=2,
                        target_format=1,
                    )
                    transitioned = True
                try:
                    state.owner.save(str(temporary), max_shard_size="5GB")
                finally:
                    if transitioned and transition is not None:
                        try:
                            _transition_gptqmodel_qzero_format(
                                transition,
                                source_format=1,
                                target_format=2,
                            )
                            _verify_gptqmodel_qzero_snapshots(
                                transition,
                                runtime_qzero_snapshots,
                                required_format=2,
                            )
                        except Exception:
                            _restore_gptqmodel_qzero_snapshots(
                                transition,
                                runtime_qzero_snapshots,
                                required_format=2,
                            )
                            raise
            elif state.save_kind == "transformers_fp8":
                state.owner.save_pretrained(
                    str(temporary), safe_serialization=True, max_shard_size="5GB"
                )
            else:
                raise QuantizationError(
                    f"Unknown optional export lifecycle {state.save_kind!r}", component=self.name
                )
            tokenizer = getattr(model.loaded.tokenizer, "raw", None)
            save_tokenizer = getattr(tokenizer, "save_pretrained", None)
            if callable(save_tokenizer):
                save_tokenizer(str(temporary))
            _validate_safe_export(temporary, self.name)
            settings = self._settings(config)
            if (
                state.save_kind == "gptqmodel"
                and self.name == "gptq"
                and settings.device == _CPU_DEVICE
            ):
                reload_recipe = _gptqmodel_cpu_reload_recipe(temporary, model)
            elif state.save_kind == "gptqmodel":
                execution_backend = "torch_awq" if self.name == "awq" else "torch"
                reload_recipe = {
                    "schema_version": "1.0",
                    "adapter": "gptqmodel",
                    "loader": "gptqmodel.GPTQModel.load",
                    "backend": self.name,
                    "execution_backend": execution_backend,
                    "device_map": {"": settings.device},
                    "dtype": model.manifest.quantization_parameters.get("source_dtype"),
                    "format": "safetensors",
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "use_safetensors": True,
                    "source_model_id": model.loaded.descriptor.model_id,
                    "source_revision": model.loaded.descriptor.revision,
                    "policy_preserving_reload_supported": False,
                    "validation_required_on_reload": True,
                }
            else:
                reload_recipe = {
                    "schema_version": "1.0",
                    "adapter": "hf_causal_lm",
                    "backend": self.name,
                    "device_map": {"": settings.device},
                    "format": "safetensors",
                    "local_files_only": True,
                    "trust_remote_code": False,
                    "use_safetensors": True,
                    "source_model_id": model.loaded.descriptor.model_id,
                    "source_revision": model.loaded.descriptor.revision,
                    "validation_required_on_reload": True,
                }
            (temporary / _GPTQ_RELOAD_FILENAME).write_text(
                json.dumps(reload_recipe, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            _write_manifest(temporary, model)
            _validate_safe_export(temporary, self.name)
            os.replace(temporary, destination)
        except Exception as error:
            if temporary.exists():
                shutil.rmtree(temporary)
            if isinstance(error, (FileExistsError, QuantizationError)):
                raise
            raise QuantizationError(
                f"Failed to safely export {self.name.upper()} candidate: {error}",
                component=self.name,
            ) from error
        self._exporters.pop(id(model.loaded.model), None)
        files = _bundle_files(destination)
        size = sum((destination / name).stat().st_size for name in files)
        return ExportArtifact(
            path=str(destination),
            format="safetensors",
            sha256=_directory_sha256(destination, files),
            size_bytes=size,
            files=files,
            reload_recipe=reload_recipe,
        )


class GPTQModelQuantizer(_PinnedSourceQuantizer):
    """AWQ or GPTQ conversion through the audited GPTQModel 5.8.0 boundary."""

    requirements = _GPTQMODEL_REQUIREMENTS

    def __init__(self, backend: str) -> None:
        if backend not in {"awq", "gptq"}:
            raise ValueError("GPTQModel backend must be 'awq' or 'gptq'")
        self.name = backend
        self.install_extra = backend
        self.required_method = backend
        self.target_precision = "int4"
        super().__init__()

    def _requires_calibration(self) -> bool:
        return True

    def _settings(self, config: QuantizationConfig) -> _GPTQModelSettings:
        return _gptqmodel_settings(config, self.name)

    def _partition_policy(
        self,
        source: LoadedModel,
        policy: ResolvedPolicyLike,
    ) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...]]:
        return _gptqmodel_policy_partition(source, policy, self.target_precision)

    def _quantize_source(
        self,
        source: LoadedModel,
        precision_map: dict[str, str],
        selected: tuple[str, ...],
        excluded: tuple[str, ...],
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        settings = self._settings(config)
        prepared = self._prepared
        if prepared is None:
            raise CapabilityError(
                f"{self.name.upper()} calibration was not prepared", component=self.name
            )
        pinned_source = _materialize_pinned_source(
            source.descriptor, local_files_only=settings.local_files_only
        )
        source_dtype = _source_dtype(source)
        versions = _dependency_status(self.requirements).versions
        import_environment = _capture_environment_bindings(_GPTQMODEL_IMPORT_ENVIRONMENT)
        import_environment_restored = False
        mixtral_binding = _capture_native_mixtral_class_binding()
        api: Any
        quantize_config: Any
        execution_backend: Any
        wrapper: Any
        awq_pack_shim = False
        python_packing = settings.device == _CPU_DEVICE
        defuser_patch_restored = False
        started = time.perf_counter()
        try:
            try:
                api = _load_gptqmodel_api()
                dynamic = _exact_dynamic_exclusions(excluded)
                common: dict[str, Any] = {
                    "bits": settings.bits,
                    "group_size": settings.group_size,
                    "sym": settings.sym,
                    "dynamic": dynamic,
                    "device": settings.device,
                    "format": api.FORMAT.GEMM if self.name == "awq" else api.FORMAT.GPTQ,
                    "offload_to_disk": False,
                }
                if self.name == "awq":
                    quantize_config = api.QuantizeConfig(
                        **common,
                        quant_method=api.METHOD.AWQ,
                    )
                    execution_backend = api.BACKEND.TORCH_AWQ
                else:
                    quantize_config = api.QuantizeConfig(
                        **common,
                        quant_method=api.METHOD.GPTQ,
                        desc_act=settings.desc_act,
                        act_group_aware=settings.act_group_aware,
                        damp_percent=settings.damp_percent,
                    )
                    execution_backend = api.BACKEND.TORCH
                wrapper = api.GPTQModel.load(
                    str(pinned_source),
                    quantize_config=quantize_config,
                    trust_remote_code=False,
                    local_files_only=True,
                    use_safetensors=True,
                    weights_only=True,
                    dtype=source_dtype,
                )
                previous_pack_setting = os.environ.get("GPTQMODEL_DISABLE_PACK_EXT")
                if python_packing:
                    os.environ["GPTQMODEL_DISABLE_PACK_EXT"] = "1"
                try:
                    if self.name == "awq":
                        awq_pack_shim = _apply_gptqmodel_580_awq_pack_shim(
                            api, versions["gptqmodel"]
                        )
                    wrapper.quantize(
                        [sample.upstream_record() for sample in prepared.samples],
                        batch_size=settings.batch_size,
                        backend=execution_backend,
                        calibration_data_min_length=_GPTQMODEL_MIN_ACTIVE_TOKENS,
                    )
                finally:
                    try:
                        _remove_gptqmodel_580_awq_pack_shim(api, awq_pack_shim)
                    finally:
                        if python_packing:
                            if previous_pack_setting is None:
                                os.environ.pop("GPTQMODEL_DISABLE_PACK_EXT", None)
                            else:
                                os.environ["GPTQMODEL_DISABLE_PACK_EXT"] = previous_pack_setting
            finally:
                try:
                    defuser_patch_restored = _restore_native_mixtral_class_binding(mixtral_binding)
                finally:
                    import_environment_restored = _restore_environment_bindings(import_environment)
                if not import_environment_restored:
                    raise QuantizationError(
                        "GPTQModel import environment was not restored",
                        component=self.name,
                    )
        except Exception as error:
            raise QuantizationError(
                f"GPTQModel {self.name.upper()} conversion failed: {error}",
                component=self.name,
                remediation=(
                    "Verify the pinned native Transformers checkpoint, configured execution "
                    "device, calibration coverage, and documented GPTQModel software matrix."
                ),
            ) from error
        module = getattr(wrapper, "model", None)
        if not isinstance(module, nn.Module) or not bool(getattr(wrapper, "quantized", False)):
            raise QuantizationError(
                f"GPTQModel did not return a quantized torch module for {self.name.upper()}",
                component=self.name,
            )
        quantized_type = api.AwqTorchQuantLinear if self.name == "awq" else api.TorchQuantLinear
        observed = _verify_gptqmodel_inventory(
            module,
            selected=selected,
            quantized_type=quantized_type,
            backend=self.name,
            runtime_excluded=excluded,
            base_quantized_type=api.BaseQuantLinear,
        )
        exclusion_audit = _audit_gptqmodel_exclusions(
            cast(nn.Module, source.model),
            module,
            excluded,
            source_dtype=source_dtype,
            backend=self.name,
        )
        cpu_bias_none_repair_count = 0
        cpu_eager_post_init = _GPTQModelCPUPostInit(module_count=0, compile_call_count=0)
        runtime_qzero_conversion_count = 0
        runtime_transition: _GPTQModelRuntimeTransition | None = None
        if self.name == "gptq" and settings.device == _CPU_DEVICE:
            cpu_bias_none_repair_count = _normalize_gptqmodel_580_cpu_bias_contract(
                api,
                cast(nn.Module, source.model),
                module,
                observed,
                versions["gptqmodel"],
            )
            runtime_transition = _GPTQModelRuntimeTransition(
                api=api,
                module=module,
                quantize_config=quantize_config,
                qlinear_names=observed,
            )
            runtime_qzero_conversion_count = _transition_gptqmodel_qzero_format(
                runtime_transition,
                source_format=1,
                target_format=2,
            )
            cpu_eager_post_init = _post_init_gptqmodel_580_cpu_torch(
                api,
                module,
                observed,
                versions["gptqmodel"],
            )
        observed_digest = hashlib.sha256("\n".join(observed).encode("utf-8")).hexdigest()
        candidate_state_sha256 = _module_state_sha256(module)
        software_matrix_sha256 = hashlib.sha256(
            json.dumps(versions, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        parameters: dict[str, str | int | float | bool] = {
            **config.parameters,
            "awq_pack_shim_applied": awq_pack_shim,
            "candidate_lifecycle": self.lifecycle,
            "candidate_state_sha256": candidate_state_sha256,
            "calibration_min_active_tokens": _GPTQMODEL_MIN_ACTIVE_TOKENS,
            "cpu_bias_none_repair_count": cpu_bias_none_repair_count,
            "cpu_eager_post_init": cpu_eager_post_init.module_count > 0,
            "cpu_eager_post_init_module_count": cpu_eager_post_init.module_count,
            "cpu_post_init_compile_bypass_count": cpu_eager_post_init.compile_call_count,
            "defuser_global_patch_restored": defuser_patch_restored,
            "execution_backend": str(getattr(execution_backend, "value", execution_backend)),
            "excluded_expanded_linear_count": exclusion_audit.expanded_linear_count,
            "excluded_state_sha256": exclusion_audit.state_sha256,
            "excluded_state_tensor_count": exclusion_audit.state_tensor_count,
            "excluded_state_verified_equal_to_source": (exclusion_audit.expanded_linear_count == 0),
            "accelerate_version": versions["accelerate"],
            "defuser_version": versions["defuser"],
            "gptqmodel_version": versions["gptqmodel"],
            "gptqmodel_import_environment_restored": import_environment_restored,
            "huggingface_hub_version": versions["huggingface-hub"],
            "kernels_version": versions["kernels"],
            "policy_realization": "exact_selected_qlinear_inventory",
            "python_packing": python_packing,
            "quantized_module_count": len(observed),
            "quantized_module_names_sha256": observed_digest,
            "runtime_exclusion_count": len(excluded),
            "runtime_qzero_conversion_count": runtime_qzero_conversion_count,
            "runtime_qzero_format": "gptq_v2" if runtime_qzero_conversion_count else "upstream",
            "safetensors_version": versions["safetensors"],
            "software_matrix_sha256": software_matrix_sha256,
            "source_dtype": str(source_dtype).removeprefix("torch."),
            "torch_version": versions["torch"],
            "transformers_version": versions["transformers"],
            "torch_compile_bypassed_during_cpu_post_init": cpu_eager_post_init.module_count > 0,
            "trust_remote_code": False,
            "use_safetensors": True,
        }
        return self._candidate(
            source=source,
            module=module,
            precision_map=precision_map,
            excluded=excluded,
            calibration=calibration,
            parameters=parameters,
            backend_version=versions["gptqmodel"],
            elapsed=time.perf_counter() - started,
            exporter=_ExportState(
                owner=wrapper,
                save_kind="gptqmodel",
                runtime_transition=runtime_transition,
            ),
        )


class FineGrainedFP8Quantizer(_PinnedSourceQuantizer):
    """Transformers fine-grained FP8 load-time conversion for Hopper or newer."""

    name = "fp8"
    target_precision = "fp8"
    install_extra = "fp8"
    required_method = "fp8"
    requirements = _FP8_REQUIREMENTS

    def _settings(self, config: QuantizationConfig) -> _FP8Settings:
        return _fp8_settings(config)

    def _quantize_source(
        self,
        source: LoadedModel,
        precision_map: dict[str, str],
        selected: tuple[str, ...],
        excluded: tuple[str, ...],
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        del selected
        if calibration is not None:
            raise CapabilityError(
                "Fine-grained FP8 does not consume calibration data", component=self.name
            )
        settings = self._settings(config)
        api = _load_transformers_fp8_api()
        quantize_config = api.FineGrainedFP8Config(
            activation_scheme=settings.activation_scheme,
            weight_block_size=(settings.block_rows, settings.block_columns),
            dequantize=False,
            modules_to_not_convert=list(excluded),
            scale_fmt=settings.scale_fmt,
        )
        started = time.perf_counter()
        try:
            module = api.AutoModelForCausalLM.from_pretrained(
                source.descriptor.model_id,
                revision=cast(str, source.descriptor.revision),
                local_files_only=settings.local_files_only,
                trust_remote_code=False,
                use_safetensors=True,
                weights_only=True,
                dtype=_source_dtype(source),
                device_map={"": settings.device},
                quantization_config=quantize_config,
            )
        except Exception as error:
            raise QuantizationError(
                f"Transformers fine-grained FP8 conversion failed: {error}",
                component=self.name,
                remediation=(
                    "Use CUDA compute capability >=9.0 with compatible PyTorch/CUDA and "
                    "Transformers versions, and verify the pinned native checkpoint."
                ),
            ) from error
        if not isinstance(module, nn.Module) or not bool(getattr(module, "is_quantized", False)):
            raise QuantizationError(
                "Transformers did not return a fine-grained FP8 quantized torch module",
                component=self.name,
            )
        versions = _dependency_status(self.requirements).versions
        parameters: dict[str, str | int | float | bool] = {
            **config.parameters,
            "candidate_lifecycle": self.lifecycle,
            "transformers_version": versions["transformers"],
            "trust_remote_code": False,
            "use_safetensors": True,
        }
        return self._candidate(
            source=source,
            module=module,
            precision_map=precision_map,
            excluded=excluded,
            calibration=None,
            parameters=parameters,
            backend_version=versions["transformers"],
            elapsed=time.perf_counter() - started,
            exporter=_ExportState(owner=module, save_kind="transformers_fp8"),
        )


def _token_samples(samples: tuple[ModelBatch, ...]) -> tuple[_TokenSample, ...]:
    records: list[_TokenSample] = []
    seen: set[str] = set()
    for batch in samples:
        input_ids = batch.input_ids
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            raise CapabilityError(
                "Calibration input_ids must be a rank-2 torch tensor",
                component="optional_quantizer",
            )
        attention_mask = batch.attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if not isinstance(attention_mask, Tensor) or attention_mask.shape != input_ids.shape:
            raise CapabilityError(
                "Calibration attention_mask must match input_ids",
                component="optional_quantizer",
            )
        if len(batch.sample_ids) != input_ids.shape[0]:
            raise CapabilityError(
                "Calibration stable IDs do not match the batch dimension",
                component="optional_quantizer",
            )
        for index, sample_id in enumerate(batch.sample_ids):
            if sample_id in seen:
                raise CapabilityError(
                    f"Duplicate calibration sample ID: {sample_id}",
                    component="optional_quantizer",
                )
            seen.add(sample_id)
            ids = tuple(int(value) for value in input_ids[index].detach().cpu().tolist())
            mask = tuple(int(value) for value in attention_mask[index].detach().cpu().tolist())
            if not ids or not any(mask):
                raise CapabilityError(
                    f"Calibration sample {sample_id} contains no active tokens",
                    component="optional_quantizer",
                )
            if any(value not in {0, 1} for value in mask):
                raise CapabilityError(
                    f"Calibration sample {sample_id} has a non-binary attention mask",
                    component="optional_quantizer",
                )
            records.append(_TokenSample(sample_id, ids, mask))
    return tuple(records)


def _bundle_files(path: Path) -> tuple[str, ...]:
    return tuple(
        sorted(
            str(candidate.relative_to(path)) for candidate in path.rglob("*") if candidate.is_file()
        )
    )


def _validate_safe_export(path: Path, backend: str) -> None:
    files = tuple(path.rglob("*"))
    if not files:
        raise QuantizationError("Optional backend wrote an empty export", component=backend)
    for candidate in files:
        if candidate.is_symlink():
            raise QuantizationError("Optional export contains a symlink", component=backend)
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() in _BANNED_WEIGHT_SUFFIXES:
            raise QuantizationError(
                f"Optional export contains forbidden pickle-capable file: {candidate.name}",
                component=backend,
            )
        if _is_executable_or_loadable_code(candidate):
            raise QuantizationError(
                f"Optional export contains executable or loadable code: {candidate.name}",
                component=backend,
            )
    if not any(candidate.is_file() and candidate.suffix == ".safetensors" for candidate in files):
        raise QuantizationError(
            "Optional export contains no safetensors checkpoint", component=backend
        )
    config_path = path / "config.json"
    if not config_path.is_file() or config_path.is_symlink():
        raise QuantizationError("Optional export is missing safe config.json", component=backend)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuantizationError(
            f"Optional export config.json is invalid: {error}", component=backend
        ) from error
    quantization = payload.get("quantization_config") if isinstance(payload, dict) else None
    method = quantization.get("quant_method") if isinstance(quantization, dict) else None
    accepted = {"awq"} if backend == "awq" else {"gptq"}
    if backend == "fp8":
        accepted = {"fp8"}
    if method not in accepted:
        raise QuantizationError(
            f"Optional export does not declare expected {backend} quantization metadata",
            component=backend,
        )
    if (
        backend == "gptq"
        and isinstance(quantization, dict)
        and (
            quantization.get("format") != "gptq" or quantization.get("checkpoint_format") != "gptq"
        )
    ):
        raise QuantizationError(
            "GPTQ export is not serialized in the audited V1 checkpoint format",
            component=backend,
        )


def _write_manifest(path: Path, model: QuantizedModel) -> None:
    manifest_path = path / "inkling_quant_manifest.json"
    manifest = model.manifest
    for _attempt in range(16):
        payload = (
            json.dumps(manifest.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        manifest_path.write_bytes(payload)
        measured = sum(
            candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file()
        )
        if measured == manifest.serialized_size_bytes:
            model.manifest = manifest
            return
        manifest = manifest.model_copy(update={"serialized_size_bytes": measured})
    raise QuantizationError(
        "Optional export manifest size did not converge", component=model.manifest.backend
    )


def _directory_sha256(path: Path, files: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256((path / name).read_bytes()).digest())
    return digest.hexdigest()


def _replace_named_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, separator, child_name = name.rpartition(".")
    if not separator or not child_name:
        raise QuantizationError(
            f"Cannot replace root-level GPTQ module {name!r}",
            component="gptq",
        )
    parent = root.get_submodule(parent_name)
    if child_name not in parent._modules:
        raise QuantizationError(
            f"Cannot resolve GPTQ module parent for {name!r}",
            component="gptq",
        )
    parent._modules[child_name] = replacement


def _read_gptqmodel_cpu_reload_recipe(path: Path) -> dict[str, Any]:
    recipe_path = path / _GPTQ_RELOAD_FILENAME
    if not recipe_path.is_file() or recipe_path.is_symlink():
        raise QuantizationError(
            "CPU GPTQ export is missing its strict reload metadata",
            component="gptq",
        )
    try:
        payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuantizationError(
            f"CPU GPTQ reload metadata is invalid: {error}",
            component="gptq",
        ) from error
    if not isinstance(payload, dict):
        raise QuantizationError("CPU GPTQ reload metadata is not an object", component="gptq")
    return cast(dict[str, Any], payload)


def reload_gptqmodel_cpu_export(
    path: Path,
    source: LoadedModel,
    manifest: QuantizationManifest,
    config: QuantizationConfig,
) -> QuantizedModel:
    """Strictly reconstruct the exact policy-preserving GPTQModel CPU candidate."""

    started = time.perf_counter()
    export_path = path.resolve()
    settings = _gptqmodel_settings(config, "gptq")
    if settings.device != _CPU_DEVICE:
        raise CapabilityError(
            "The strict GPTQModel export reload is registered only for CPU",
            component="gptq",
        )
    if (source.descriptor.model_id, source.descriptor.revision) != (
        _STORIES15M_MODEL_ID,
        _STORIES15M_REVISION,
    ):
        raise CapabilityError(
            "The strict GPTQModel export reload is registered only for pinned Stories15M",
            component="gptq",
        )
    _validate_loaded_source_state(source, backend="gptq")
    if (
        manifest.backend != "gptq"
        or manifest.backend_version != "5.8.0"
        or manifest.method != "gptq"
        or manifest.source_model_checksum != source.descriptor.checksum
    ):
        raise QuantizationError(
            "CPU GPTQ export manifest does not match the pinned source contract",
            component="gptq",
        )
    _validate_safe_export(export_path, "gptq")
    recipe = _read_gptqmodel_cpu_reload_recipe(export_path)
    selected, _excluded = _gptqmodel_cpu_manifest_partition(manifest)
    selected_digest = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
    parameters = manifest.quantization_parameters
    identity = source.descriptor
    required_recipe = {
        "schema_version": "1.0",
        "adapter": "inkling_quant_lab_gptqmodel_cpu",
        "loader": "inkling_quant_lab.quantization.optional.reload_gptqmodel_cpu_export",
        "backend": "gptq",
        "model_id": identity.model_id,
        "revision": identity.revision,
        "resolved_class": identity.resolved_class,
        "architecture": identity.architecture,
        "source_model_checksum": identity.checksum,
        "source_dtype": "float32",
        "checkpoint_qzero_format": "gptq_v1",
        "runtime_qzero_format": "gptq_v2",
        "config_file": "config.json",
        "config_file_sha256": parameters.get("export_config_sha256"),
        "config_file_size_bytes": parameters.get("export_config_size_bytes"),
        "weight_file": "model.safetensors",
        "selected_module_count": len(selected),
        "selected_module_names_sha256": selected_digest,
        "excluded_module_count": len(manifest.excluded_modules),
        "excluded_state_sha256": parameters.get("excluded_state_sha256"),
        "candidate_state_sha256": parameters.get("candidate_state_sha256"),
        "local_files_only": True,
        "trust_remote_code": False,
        "use_safetensors": True,
        "strict_state_assignment": True,
        "policy_preserving_reload_supported": True,
        "validation_required_on_reload": True,
    }
    mismatched_recipe = tuple(
        key for key, expected in required_recipe.items() if recipe.get(key) != expected
    )
    if mismatched_recipe or tuple(recipe.get("selected_module_names", ())) != selected:
        raise QuantizationError(
            "CPU GPTQ reload metadata differs from the governed manifest: "
            + ", ".join(
                (*mismatched_recipe, "selected_module_names")
                if tuple(recipe.get("selected_module_names", ())) != selected
                else mismatched_recipe
            ),
            component="gptq",
        )
    _validate_gptqmodel_cpu_export_config(export_path, recipe, manifest)
    versions = _dependency_status(_GPTQMODEL_REQUIREMENTS).versions
    if versions != _GPTQMODEL_CPU_EXACT_VERSIONS or recipe.get("software_versions") != versions:
        raise CapabilityError(
            "CPU GPTQ reload requires the exact audited software matrix",
            component="gptq",
        )
    matrix_digest = hashlib.sha256(
        json.dumps(versions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        recipe.get("software_matrix_sha256") != matrix_digest
        or parameters.get("software_matrix_sha256") != matrix_digest
    ):
        raise QuantizationError(
            "CPU GPTQ reload software matrix digest differs from the manifest",
            component="gptq",
        )
    weights = export_path / "model.safetensors"
    if (
        recipe.get("weight_file_sha256") != _file_sha256(weights)
        or recipe.get("weight_file_size_bytes") != weights.stat().st_size
    ):
        raise QuantizationError(
            "CPU GPTQ export tensor checksum or size differs from reload metadata",
            component="gptq",
        )

    import_environment = _capture_environment_bindings(_GPTQMODEL_IMPORT_ENVIRONMENT)
    mixtral_binding = _capture_native_mixtral_class_binding()
    import_environment_restored = False
    defuser_patch_restored = False
    try:
        try:
            from safetensors.torch import load_file
            from transformers import AutoConfig, AutoModelForCausalLM

            from inkling_quant_lab.models.mixtral_compat import scoped_defuser_linear_mixtral

            api = _load_gptqmodel_api()
            resolved = AutoConfig.from_pretrained(
                str(export_path),
                local_files_only=True,
                trust_remote_code=False,
            )
            if getattr(
                resolved, "model_type", None
            ) != "mixtral" or "MixtralForCausalLM" not in tuple(
                getattr(resolved, "architectures", ()) or ()
            ):
                raise QuantizationError(
                    "CPU GPTQ export config is not the validated Mixtral architecture",
                    component="gptq",
                )
            with scoped_defuser_linear_mixtral(), torch.device("meta"):
                module = cast(Any, AutoModelForCausalLM).from_config(
                    resolved,
                    trust_remote_code=False,
                    dtype=torch.float32,
                )
            if not isinstance(module, nn.Module):
                raise QuantizationError(
                    "CPU GPTQ meta construction did not return a torch module",
                    component="gptq",
                )
            resolved_class = f"{module.__class__.__module__}.{module.__class__.__qualname__}"
            if resolved_class != identity.resolved_class:
                raise QuantizationError(
                    "CPU GPTQ meta shell resolved class differs from the pinned source",
                    component="gptq",
                )
            for name in selected:
                source_linear = module.get_submodule(name)
                if not isinstance(source_linear, nn.Linear):
                    raise QuantizationError(
                        f"CPU GPTQ selected module {name!r} is not an nn.Linear shell",
                        component="gptq",
                    )
                with torch.device("meta"):
                    qlinear = api.TorchQuantLinear(
                        bits=settings.bits,
                        group_size=settings.group_size,
                        sym=settings.sym,
                        desc_act=settings.desc_act,
                        in_features=source_linear.in_features,
                        out_features=source_linear.out_features,
                        bias=source_linear.bias is not None,
                        pack_dtype=torch.int32,
                        register_buffers=True,
                        backend=api.BACKEND.TORCH,
                        name=name,
                    )
                _replace_named_module(module, name, qlinear)
            model_body = getattr(module, "model", None)
            rotary = getattr(model_body, "rotary_emb", None)
            if not isinstance(model_body, nn.Module) or not isinstance(rotary, nn.Module):
                raise QuantizationError(
                    "CPU GPTQ meta shell has no validated rotary embedding",
                    component="gptq",
                )
            model_body.rotary_emb = rotary.__class__(
                config=resolved,
                device=torch.device("cpu"),
            )
            expected_state = module.state_dict()
            state = load_file(str(weights), device="cpu")
            if state.keys() != expected_state.keys():
                missing = tuple(sorted(expected_state.keys() - state.keys()))
                unexpected = tuple(sorted(state.keys() - expected_state.keys()))
                raise QuantizationError(
                    "CPU GPTQ export state keys differ from the strict meta shell; "
                    f"missing={missing}, unexpected={unexpected}",
                    component="gptq",
                )
            invalid_state = tuple(
                name
                for name, value in state.items()
                if value.shape != expected_state[name].shape
                or value.dtype != expected_state[name].dtype
            )
            if invalid_state:
                raise QuantizationError(
                    "CPU GPTQ export state shape/dtype differs from the strict meta shell: "
                    + ", ".join(invalid_state),
                    component="gptq",
                )
            incompatible = module.load_state_dict(state, strict=True, assign=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise QuantizationError(
                    "CPU GPTQ strict state assignment returned incompatible keys",
                    component="gptq",
                )
            remaining_meta = tuple(
                name
                for name, value in (*module.named_parameters(), *module.named_buffers())
                if value.is_meta or value.device.type != "cpu"
            )
            if remaining_meta:
                raise QuantizationError(
                    "CPU GPTQ reload left non-CPU or meta tensors: " + ", ".join(remaining_meta),
                    component="gptq",
                )
            transition = _GPTQModelRuntimeTransition(
                api=api,
                module=module,
                quantize_config=SimpleNamespace(bits=settings.bits, pack_dtype=torch.int32),
                qlinear_names=selected,
            )
            _transition_gptqmodel_qzero_format(
                transition,
                source_format=1,
                target_format=2,
            )
            _post_init_gptqmodel_580_cpu_torch(api, module, selected, versions["gptqmodel"])
            observed = _verify_gptqmodel_inventory(
                module,
                selected=selected,
                quantized_type=api.TorchQuantLinear,
                backend="gptq",
                runtime_excluded=manifest.excluded_modules,
                base_quantized_type=api.BaseQuantLinear,
            )
            if observed != selected:
                raise QuantizationError(
                    "CPU GPTQ reload did not realize its exact selected inventory",
                    component="gptq",
                )
            exclusion_audit = _audit_gptqmodel_exclusions(
                cast(nn.Module, source.model),
                module,
                manifest.excluded_modules,
                source_dtype=torch.float32,
                backend="gptq",
            )
            if (
                exclusion_audit.state_sha256 != parameters.get("excluded_state_sha256")
                or exclusion_audit.state_sha256 != recipe.get("excluded_state_sha256")
                or exclusion_audit.module_count != len(manifest.excluded_modules)
            ):
                raise QuantizationError(
                    "CPU GPTQ reload changed protected full-precision state",
                    component="gptq",
                )
            state_sha256 = _module_state_sha256(module)
            if state_sha256 != parameters.get(
                "candidate_state_sha256"
            ) or state_sha256 != recipe.get("candidate_state_sha256"):
                raise QuantizationError(
                    "CPU GPTQ reloaded candidate state differs from the quantized candidate",
                    component="gptq",
                )
            module.eval()
        finally:
            try:
                _restore_native_mixtral_class_binding(mixtral_binding)
                defuser_patch_restored = True
            finally:
                import_environment_restored = _restore_environment_bindings(import_environment)
            if not defuser_patch_restored or not import_environment_restored:
                raise QuantizationError(
                    "CPU GPTQ reload did not restore global compatibility bindings",
                    component="gptq",
                )
    except (CapabilityError, QuantizationError):
        raise
    except Exception as error:
        raise QuantizationError(
            f"Strict CPU GPTQ export reload failed: {error}",
            component="gptq",
        ) from error
    return QuantizedModel(
        loaded=LoadedModel(
            model=module,
            tokenizer=source.tokenizer,
            descriptor=source.descriptor,
            load_time_seconds=time.perf_counter() - started,
            load_time_kind="candidate_export_reload",
        ),
        manifest=manifest,
    )


def create_awq_quantizer() -> GPTQModelQuantizer:
    """Create the GPTQModel-backed AWQ converter."""

    return GPTQModelQuantizer("awq")


def create_gptq_quantizer() -> GPTQModelQuantizer:
    """Create the GPTQModel-backed GPTQ converter."""

    return GPTQModelQuantizer("gptq")


def create_fp8_quantizer() -> FineGrainedFP8Quantizer:
    """Create the Transformers fine-grained FP8 converter."""

    return FineGrainedFP8Quantizer()


__all__ = [
    "FineGrainedFP8Quantizer",
    "GPTQModelQuantizer",
    "UnavailableOptionalQuantizer",
    "create_awq_quantizer",
    "create_fp8_quantizer",
    "create_gptq_quantizer",
    "reload_gptqmodel_cpu_export",
]
