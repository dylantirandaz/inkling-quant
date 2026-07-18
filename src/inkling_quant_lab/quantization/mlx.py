"""Registered, exact MLX-LM affine q4/q8 quantizer for Stories15M MoE."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from inkling_quant_lab.config import QuantizationConfig
from inkling_quant_lab.exceptions import CapabilityError, QuantizationError
from inkling_quant_lab.mlx_contract import (
    CANONICAL_MLX_MODEL_CARD_BYTES,
    EXPECTED_MLX_VERSIONS,
    MLX_CONVERSION_CONTRACTS,
    MLX_MODEL_CLASS,
    MODEL_ARCHITECTURE,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_WEIGHT_SHA256,
    audit_converted_bundle,
    directory_sha256,
    expected_float32_only_names,
    expected_quantized_expert_names,
    expected_quantized_leaf_names,
    mlx_environment_status,
    runtime_quantization_proof,
)
from inkling_quant_lab.models.base import LoadedModel, ModelBatch, ModelDescriptor
from inkling_quant_lab.models.mlx_lm_mixtral import (
    MLXModelHandle,
    load_exact_converted_bundle,
    load_exact_source,
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


@dataclass(frozen=True, slots=True)
class _Settings:
    bits: int
    group_size: int
    mode: str

    @property
    def precision(self) -> str:
        return f"int{self.bits}"

    @property
    def label(self) -> str:
        return MLX_CONVERSION_CONTRACTS[self.bits].label


def _settings(config: QuantizationConfig) -> _Settings:
    unknown = sorted(set(config.parameters) - {"bits", "group_size", "mode"})
    if unknown:
        raise ValueError("unsupported MLX quantization parameter(s): " + ", ".join(unknown))
    bits = config.parameters.get("bits")
    group_size = config.parameters.get("group_size")
    mode = config.parameters.get("mode")
    if type(bits) is not int or bits not in {4, 8}:
        raise ValueError("quantization.parameters.bits must be exactly integer 4 or 8")
    if type(group_size) is not int or group_size != 32:
        raise ValueError("quantization.parameters.group_size must be exactly integer 32")
    if mode != "affine":
        raise ValueError("quantization.parameters.mode must be exactly 'affine'")
    return _Settings(bits=bits, group_size=group_size, mode=mode)


def _model_reasons(model: ModelDescriptor) -> tuple[str, ...]:
    reasons: list[str] = []
    if model.model_id != MODEL_ID or model.revision != MODEL_REVISION:
        reasons.append("mlx_affine supports only the exact pinned Stories15M repository/revision")
    if model.architecture not in {"unresolved", MODEL_ARCHITECTURE}:
        reasons.append("mlx_affine supports only the built-in MixtralForCausalLM architecture")
    if model.resolved_class not in {"unresolved", MLX_MODEL_CLASS}:
        reasons.append("mlx_affine requires mlx_lm.models.mixtral.Model")
    if model.capabilities.requires_remote_code:
        reasons.append("models requiring remote code are unsupported")
    if not model.capabilities.supports_text or not model.capabilities.is_moe:
        reasons.append("mlx_affine requires the validated text MoE model")
    return tuple(reasons)


def _runtime_reasons(runtime: RuntimeCapabilities) -> tuple[str, ...]:
    reasons: list[str] = []
    if runtime.backend != "mlx_metal":
        reasons.append("mlx_affine requires runtime.backend=mlx_metal")
    if not runtime.available:
        reasons.append("mlx_metal is unavailable")
    if "mps" not in runtime.devices:
        reasons.append("mlx_metal does not expose the Apple Metal device")
    if "float32" not in runtime.supported_dtypes:
        reasons.append("mlx_metal does not expose the validated float32 execution path")
    if runtime.supports_sharding:
        reasons.append("the exact single-device MLX contract must not advertise sharding")
    return tuple(reasons)


def _config_reasons(config: QuantizationConfig) -> tuple[str, ...]:
    reasons: list[str] = []
    try:
        _settings(config)
    except ValueError as error:
        reasons.append(str(error))
    if config.method != "mlx_affine":
        reasons.append("quantization.method must be mlx_affine")
    if config.calibration is not None:
        reasons.append("mlx_affine is calibration-free")
    if config.policy.type != "uniform":
        reasons.append("fused MLX expert projections currently require a uniform policy")
    if (
        config.policy.preserve_router_precision
        or config.policy.preserve_output_head
        or config.policy.preserve_embeddings
    ):
        reasons.append(
            "the evidenced all-eligible-leaf contract requires router/head/embedding preservation "
            "flags to be false"
        )
    if config.export.format != "safetensors":
        reasons.append("mlx_affine exports only safetensors")
    return tuple(reasons)


def _converged_manifest(
    manifest: QuantizationManifest, *, base_bundle_size_bytes: int
) -> tuple[QuantizationManifest, bytes]:
    current = manifest
    for _attempt in range(16):
        payload = (
            json.dumps(current.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        measured = base_bundle_size_bytes + len(payload)
        if measured == current.serialized_size_bytes:
            return current, payload
        current = current.model_copy(update={"serialized_size_bytes": measured})
    raise QuantizationError("MLX embedded manifest size did not converge", component="mlx_affine")


def _policy_partition(
    source: LoadedModel, policy: ResolvedPolicyLike, settings: _Settings
) -> tuple[dict[str, str], tuple[str, ...]]:
    handle = source.model
    if not isinstance(handle, MLXModelHandle):
        raise CapabilityError("mlx_affine source is not an MLXModelHandle", component="mlx_affine")
    precision_map = dict(sorted(policy.precision_map.items()))
    eligible = expected_quantized_leaf_names()
    excluded = expected_float32_only_names()
    expected_names = set(eligible | excluded)
    if set(precision_map) != expected_names:
        raise QuantizationError(
            "resolved MLX policy does not cover the exact 63 parameter leaves; "
            f"missing={sorted(expected_names - set(precision_map))}, "
            f"extra={sorted(set(precision_map) - expected_names)}",
            component="mlx_affine",
        )
    selected = {
        name for name, precision in precision_map.items() if precision == settings.precision
    }
    if selected != set(eligible):
        raise QuantizationError(
            "resolved MLX policy must select all and only the 50 eligible leaves; "
            f"missing={sorted(set(eligible) - selected)}, "
            f"extra={sorted(selected - set(eligible))}",
            component="mlx_affine",
        )
    invalid_exclusions = {name for name in excluded if precision_map.get(name) != "float32"}
    other_precisions = {
        name
        for name, precision in precision_map.items()
        if name not in eligible and precision != "float32"
    }
    if invalid_exclusions or other_precisions:
        raise QuantizationError(
            "every non-quantizable MLX parameter leaf must remain float32",
            component="mlx_affine",
        )
    return precision_map, tuple(sorted(excluded))


def _write_embedded_manifest(path: Path, model: QuantizedModel) -> None:
    contract = MLX_CONVERSION_CONTRACTS[cast(MLXModelHandle, model.loaded.model).bits or 0]
    manifest, payload = _converged_manifest(
        model.manifest,
        base_bundle_size_bytes=contract.base_bundle_size_bytes,
    )
    (path / "inkling_quant_manifest.json").write_bytes(payload)
    measured = sum(candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file())
    if measured != manifest.serialized_size_bytes:
        raise QuantizationError(
            "MLX serialized bundle size differs from its converged manifest",
            component="mlx_affine",
        )
    model.manifest = manifest


def _reload_execution_probe(handle: MLXModelHandle, tokenizer: Any) -> None:
    """Execute the reloaded packed model, including fused gather_qmm kernels."""

    import mlx.core as mx

    prompt = tuple(
        int(token) for token in tokenizer.encode("Once upon a time", special_tokens=True)
    )
    if len(prompt) < 2:
        raise QuantizationError(
            "MLX reload probe tokenizer returned no prompt", component="mlx_affine"
        )
    logits = handle.module(mx.array([prompt[:8]]))
    mx.eval(logits)
    if tuple(int(value) for value in logits.shape[:2]) != (1, min(len(prompt), 8)):
        raise QuantizationError(
            "MLX reload probe returned an invalid logits shape", component="mlx_affine"
        )
    if int(logits.shape[-1]) != 32_000 or not bool(mx.all(mx.isfinite(logits)).item()):
        raise QuantizationError("MLX reload probe logits are invalid", component="mlx_affine")
    mx.synchronize()


class MLXAffineQuantizer:
    """Calibration-free MLX affine q4/q8 conversion for the exact public MoE."""

    name = "mlx_affine"
    lifecycle: Literal["pinned_source_reload"] = "pinned_source_reload"

    def check_support(
        self,
        model: ModelDescriptor,
        runtime: RuntimeCapabilities,
        config: QuantizationConfig,
    ) -> SupportReport:
        """Return exact dependency, model, runtime, and format diagnostics."""

        environment = mlx_environment_status()
        reasons = (
            *environment.reasons,
            *_model_reasons(model),
            *_runtime_reasons(runtime),
            *_config_reasons(config),
        )
        return SupportReport(
            available=environment.available,
            supported=not reasons,
            component=self.name,
            reasons=tuple(reasons),
            warnings=(
                "affine q8 is integer weight quantization and is not FP8",
                "the exact model's publisher-randomized router is not "
                "learned-specialization evidence",
            ),
            install_extra="mlx",
            remediation=(
                None
                if not reasons
                else "Use the exact pinned Stories15M/MLX/Apple-Metal matrix and checked config."
            ),
            supported_precisions=("float32", "int4", "int8"),
        )

    def calibrate(
        self,
        model: LoadedModel,
        samples: tuple[ModelBatch, ...],
        config: QuantizationConfig,
    ) -> CalibrationArtifact | None:
        """Reject calibration rather than silently ignoring scientific inputs."""

        del model
        if samples or config.calibration is not None:
            raise CapabilityError(
                "MLX affine q4/q8 conversion is calibration-free",
                component=self.name,
            )
        return None

    def quantize(
        self,
        model: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        """Preserve the explicit source-reload lifecycle through the common protocol."""

        return self.quantize_from_pinned_source(model, policy, calibration, config)

    def quantize_from_pinned_source(
        self,
        source: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        """Reload exact source, quantize all eligible leaves, and prove runtime classes."""

        if calibration is not None:
            raise CapabilityError("mlx_affine does not consume calibration", component=self.name)
        settings = _settings(config)
        if _config_reasons(config):
            raise CapabilityError("; ".join(_config_reasons(config)), component=self.name)
        if _model_reasons(source.descriptor):
            raise CapabilityError("; ".join(_model_reasons(source.descriptor)), component=self.name)
        if source.descriptor.checksum != SOURCE_WEIGHT_SHA256:
            raise CapabilityError(
                "MLX source checksum does not match the pinned model", component=self.name
            )
        precision_map, excluded = _policy_partition(source, policy, settings)
        source_handle = source.model
        if not isinstance(source_handle, MLXModelHandle) or source_handle.bits is not None:
            raise CapabilityError(
                "mlx_affine requires the fresh all-float32 source handle",
                component=self.name,
            )

        started = time.perf_counter()
        fresh, tokenizer, _source_elapsed = load_exact_source(
            source_handle.source_snapshot,
            seed=source_handle.seed,
        )
        try:
            import mlx.core as mx
            from mlx.utils import tree_flatten
            from mlx_lm.utils import quantize_model

            module, quantized_config = quantize_model(
                fresh.module,
                fresh.config,
                settings.group_size,
                settings.bits,
                mode=settings.mode,
            )
            mx.eval(module.parameters())
            mx.synchronize()
        except Exception as error:
            raise QuantizationError(
                f"MLX-LM affine conversion failed: {error}",
                component=self.name,
                remediation="Verify the exact MLX package matrix and Apple Metal availability.",
            ) from error
        proof = runtime_quantization_proof(module, bits=settings.bits)
        flat_parameters = cast(
            list[tuple[str, Any]],
            tree_flatten(module.parameters()),
        )
        tensor_bytes = sum(int(value.nbytes) for _name, value in flat_parameters)
        contract = MLX_CONVERSION_CONTRACTS[settings.bits]
        if tensor_bytes != contract.tensor_bytes:
            raise QuantizationError(
                "MLX runtime tensor bytes differ from the deterministic conversion contract",
                component=self.name,
            )
        elapsed = time.perf_counter() - started
        if elapsed <= 0.0 or not math.isfinite(elapsed):
            raise QuantizationError("MLX conversion duration is invalid", component=self.name)
        parameters: dict[str, str | int | float | bool] = {
            "bits": settings.bits,
            "group_size": settings.group_size,
            "mode": settings.mode,
            "candidate_lifecycle": self.lifecycle,
            "quantized_leaf_count": int(proof["quantized_leaf_count"]),
            "quantized_fused_expert_projection_count": len(expected_quantized_expert_names()),
            "runtime_tensor_bytes": tensor_bytes,
            "source_revision": MODEL_REVISION,
            "expected_weight_sha256": contract.files["model.safetensors"][1],
            "expert_kernel": "mlx.core.gather_qmm",
        }
        manifest = QuantizationManifest(
            backend=self.name,
            backend_version=EXPECTED_MLX_VERSIONS["mlx-lm"],
            method=settings.label,
            source_model_checksum=SOURCE_WEIGHT_SHA256,
            module_precision_map=precision_map,
            excluded_modules=excluded,
            quantization_parameters=parameters,
            serialized_size_bytes=contract.base_bundle_size_bytes,
            warnings=(
                "candidate reloads the exact audited source before upstream MLX-LM conversion",
                "affine q8 is not FP8",
                "the uniform fused-expert format cannot express per-expert precision",
            ),
        )
        manifest, _payload = _converged_manifest(
            manifest,
            base_bundle_size_bytes=contract.base_bundle_size_bytes,
        )
        handle = MLXModelHandle(
            module=module,
            config=cast(dict[str, Any], quantized_config),
            source_snapshot=fresh.source_snapshot,
            source_audit=fresh.source_audit,
            seed=fresh.seed,
            bits=settings.bits,
        )
        return QuantizedModel(
            loaded=LoadedModel(
                model=handle,
                tokenizer=tokenizer,
                descriptor=source.descriptor,
                load_time_seconds=elapsed,
                load_time_kind="candidate_pinned_source_quantization",
            ),
            manifest=manifest,
        )

    def export(
        self,
        model: QuantizedModel,
        destination: Path,
        config: QuantizationConfig,
    ) -> ExportArtifact:
        """Atomically save, byte-audit, reload, and execute the exact safe bundle."""

        settings = _settings(config)
        handle = model.loaded.model
        if not isinstance(handle, MLXModelHandle) or handle.bits != settings.bits:
            raise QuantizationError(
                "candidate is not owned by this MLX quantizer", component=self.name
            )
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite MLX export: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
            )
        )
        try:
            from mlx_lm.utils import save

            save(
                temporary,
                handle.source_snapshot,
                handle.module,
                model.loaded.tokenizer.raw,
                handle.config,
                donate_model=False,
            )
            # ModelCard YAML key ordering is not stable even within one declared
            # huggingface-hub version.  Normalize this non-executable metadata to
            # the exact byte sequence in the frozen v3 conversion contract.
            (temporary / "README.md").write_bytes(CANONICAL_MLX_MODEL_CARD_BYTES)
            audit_converted_bundle(temporary, bits=settings.bits)
            _write_embedded_manifest(temporary, model)
            audit_converted_bundle(temporary, bits=settings.bits)
            reloaded, tokenizer, _elapsed, proof = load_exact_converted_bundle(
                temporary,
                bits=settings.bits,
                seed=handle.seed,
                source_snapshot=handle.source_snapshot,
            )
            if int(proof["quantized_leaf_count"]) != 50:
                raise QuantizationError(
                    "reloaded MLX bundle does not expose 50 quantized leaves",
                    component=self.name,
                )
            _reload_execution_probe(reloaded, tokenizer)
            os.replace(temporary, destination)
        except Exception as error:
            if temporary.exists():
                shutil.rmtree(temporary)
            if isinstance(error, (FileExistsError, QuantizationError)):
                raise
            raise QuantizationError(
                f"failed to safely export and reload MLX candidate: {error}",
                component=self.name,
            ) from error
        digest, files = directory_sha256(destination)
        measured = sum((destination / name).stat().st_size for name in files)
        if measured != model.manifest.serialized_size_bytes:
            raise QuantizationError(
                "published MLX bundle size differs from the canonical manifest",
                component=self.name,
            )
        return ExportArtifact(
            path=str(destination),
            format="safetensors",
            sha256=digest,
            size_bytes=measured,
            files=files,
            reload_recipe={
                "adapter": "mlx_lm_mixtral",
                "runtime": "mlx_metal",
                "backend": self.name,
                "format": settings.label,
                "local_files_only": True,
                "trust_remote_code": False,
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "mlx": EXPECTED_MLX_VERSIONS["mlx"],
                "mlx_lm": EXPECTED_MLX_VERSIONS["mlx-lm"],
            },
        )


def create_quantizer() -> MLXAffineQuantizer:
    """Lazy registry factory."""

    return MLXAffineQuantizer()


__all__ = ["MLXAffineQuantizer", "create_quantizer"]
