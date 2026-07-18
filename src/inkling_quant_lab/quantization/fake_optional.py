"""Deterministic CPU fixture for optional-backend contract and failure tests.

This backend intentionally performs no numeric quantization.  It exists so the
pipeline can exercise available/unavailable, supported/unsupported, calibration,
export, and injected-failure paths without installing a GPU quantization package.
Production measurements must never present it as a real quantization backend.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from torch import nn

from inkling_quant_lab.config import QuantizationConfig
from inkling_quant_lab.exceptions import CapabilityError, QuantizationError
from inkling_quant_lab.models.base import LoadedModel, ModelBatch, ModelDescriptor
from inkling_quant_lab.quantization.base import (
    CalibrationArtifact,
    ExportArtifact,
    QuantizedModel,
    ResolvedPolicyLike,
    SupportReport,
)
from inkling_quant_lab.quantization.reference import (
    build_manifest,
    export_recipe,
    finalize_serialized_manifest,
    model_storage_bytes,
)
from inkling_quant_lab.runtimes.base import RuntimeCapabilities

FakeFailurePoint: TypeAlias = Literal["none", "check_support", "calibrate", "quantize", "export"]

_FAILURE_POINTS = frozenset({"none", "check_support", "calibrate", "quantize", "export"})
_WARNING = (
    "Test-only fake optional backend: no weights are quantized and model bytes remain unchanged."
)


@dataclass(frozen=True, slots=True)
class _Controls:
    available: bool
    supported: bool
    fail_at: FakeFailurePoint
    requires_calibration: bool


def _configuration_error(parameter: str, expected: str, value: object) -> QuantizationError:
    return QuantizationError(
        f"Invalid fake optional backend parameter '{parameter}': expected {expected}",
        component=FakeOptionalQuantizer.name,
        remediation="Correct quantization.parameters for the test fixture.",
        details={"parameter": parameter, "value": repr(value), "expected": expected},
    )


def _boolean_parameter(
    config: QuantizationConfig,
    name: str,
    default: bool,
) -> bool:
    value = config.parameters.get(name, default)
    if not isinstance(value, bool):
        raise _configuration_error(name, "a boolean", value)
    return value


def _failure_parameter(
    config: QuantizationConfig,
    default: FakeFailurePoint,
) -> FakeFailurePoint:
    value = config.parameters.get("fake_fail_at", default)
    if not isinstance(value, str) or value not in _FAILURE_POINTS:
        allowed = ", ".join(sorted(_FAILURE_POINTS))
        raise _configuration_error("fake_fail_at", f"one of: {allowed}", value)
    return cast(FakeFailurePoint, value)


def _requested_precisions(config: QuantizationConfig) -> tuple[str, ...]:
    """Conservatively list every precision that the configured policy may select."""

    policy = config.policy
    requested = {
        policy.default_precision,
        *policy.explicit_overrides.values(),
        *(rule.precision for rule in policy.module_class_rules),
        *(rule.precision for rule in policy.layer_rules),
    }
    if policy.preserve_router_precision:
        requested.add(policy.router_precision)
    if policy.preserve_output_head:
        requested.add(policy.output_head_precision)
    if policy.preserve_embeddings:
        requested.add(policy.embedding_precision)
    if policy.preserve_multimodal_projectors:
        requested.add(policy.multimodal_projector_precision)
    if policy.type in {"frequency_tiered", "hybrid"} and policy.frequency is not None:
        requested.update(policy.frequency.precisions)
        if policy.frequency.fallback_precision is not None:
            requested.add(policy.frequency.fallback_precision)
    if policy.type in {"sensitivity_tiered", "hybrid"} and policy.sensitivity is not None:
        requested.update(policy.sensitivity.precisions)
        if policy.sensitivity.fallback_precision is not None:
            requested.add(policy.sensitivity.fallback_precision)
    return tuple(sorted(requested))


class FakeOptionalQuantizer:
    """Configurable, test-only no-op implementing the full quantizer contract.

    Constructor defaults can be overridden per experiment with the parameters
    ``fake_available``, ``fake_supported``, ``fake_fail_at``, and
    ``fake_requires_calibration``.
    """

    name = "fake_optional_cpu"
    version = "fixture-v1"

    def __init__(
        self,
        *,
        available: bool = True,
        supported: bool = True,
        fail_at: FakeFailurePoint = "none",
        requires_calibration: bool = False,
    ) -> None:
        if not isinstance(available, bool):
            raise TypeError("available must be a boolean")
        if not isinstance(supported, bool):
            raise TypeError("supported must be a boolean")
        if fail_at not in _FAILURE_POINTS:
            allowed = ", ".join(sorted(_FAILURE_POINTS))
            raise ValueError(f"fail_at must be one of: {allowed}")
        if not isinstance(requires_calibration, bool):
            raise TypeError("requires_calibration must be a boolean")
        self._defaults = _Controls(
            available=available,
            supported=supported,
            fail_at=fail_at,
            requires_calibration=requires_calibration,
        )

    def _controls(self, config: QuantizationConfig) -> _Controls:
        return _Controls(
            available=_boolean_parameter(config, "fake_available", self._defaults.available),
            supported=_boolean_parameter(config, "fake_supported", self._defaults.supported),
            fail_at=_failure_parameter(config, self._defaults.fail_at),
            requires_calibration=_boolean_parameter(
                config,
                "fake_requires_calibration",
                self._defaults.requires_calibration,
            ),
        )

    def _inject_failure(self, operation: FakeFailurePoint, controls: _Controls) -> None:
        if controls.fail_at != operation:
            return
        raise QuantizationError(
            f"Injected fake optional backend failure during {operation}",
            stage=operation,
            component=self.name,
            remediation="Set quantization.parameters.fake_fail_at to 'none'.",
            details={"injected": True, "operation": operation},
        )

    def _require_enabled(self, controls: _Controls) -> None:
        if not controls.available:
            raise CapabilityError(
                "Fake optional backend is configured unavailable",
                component=self.name,
                remediation="Set quantization.parameters.fake_available=true.",
            )
        if not controls.supported:
            raise CapabilityError(
                "Fake optional backend is configured unsupported",
                component=self.name,
                remediation="Set quantization.parameters.fake_supported=true.",
            )

    def check_support(
        self,
        model: ModelDescriptor,
        runtime: RuntimeCapabilities,
        config: QuantizationConfig,
    ) -> SupportReport:
        """Report configured fixture state plus real CPU/float32 constraints."""

        del model
        controls = self._controls(config)
        self._inject_failure("check_support", controls)

        reasons: list[str] = []
        if not controls.available:
            reasons.append(
                "fake optional backend is configured unavailable; set "
                "quantization.parameters.fake_available=true"
            )
        if not controls.supported:
            reasons.append(
                "fake optional backend is configured unsupported; set "
                "quantization.parameters.fake_supported=true"
            )

        runtime_supported = runtime.available and "cpu" in runtime.devices and not runtime.reasons
        if not runtime_supported:
            reasons.extend(runtime.reasons)
            reasons.append("fake optional backend requires an available CPU runtime")

        requested = _requested_precisions(config)
        precision_supported = set(requested) <= {"float32"}
        if not precision_supported:
            unsupported = ", ".join(precision for precision in requested if precision != "float32")
            reasons.append(
                "fake optional backend preserves model bytes and only supports float32 policies; "
                f"requested unsupported precision(s): {unsupported}"
            )

        supported = (
            controls.available and controls.supported and runtime_supported and precision_supported
        )
        return SupportReport(
            available=controls.available,
            supported=supported,
            component=self.name,
            reasons=tuple(reasons),
            warnings=(_WARNING,),
            supported_precisions=("float32",) if supported else (),
        )

    def calibrate(
        self,
        model: LoadedModel,
        samples: tuple[ModelBatch, ...],
        config: QuantizationConfig,
    ) -> CalibrationArtifact | None:
        """Optionally emit deterministic provenance without inspecting sample content."""

        del model
        controls = self._controls(config)
        self._require_enabled(controls)
        self._inject_failure("calibrate", controls)
        if not controls.requires_calibration:
            return None
        if config.calibration is None:
            raise QuantizationError(
                "Fake optional backend requires an explicit calibration configuration",
                component=self.name,
                remediation="Configure quantization.calibration or disable fake calibration.",
            )
        sample_ids = tuple(sample_id for batch in samples for sample_id in batch.sample_ids)
        if not sample_ids:
            raise QuantizationError(
                "Fake optional backend calibration requires at least one sample",
                component=self.name,
                remediation="Provide a non-empty calibration batch.",
            )
        if len(sample_ids) > config.calibration.max_samples:
            raise QuantizationError(
                "Fake optional backend received more calibration samples than max_samples",
                component=self.name,
                details={
                    "sample_count": len(sample_ids),
                    "max_samples": config.calibration.max_samples,
                },
            )
        statistics = {"fake_sample_count": float(len(sample_ids))}
        payload = {
            "config": config.calibration.model_dump(mode="json"),
            "sample_ids": sample_ids,
            "statistics": statistics,
        }
        checksum = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        return CalibrationArtifact(
            config=config.calibration,
            sample_ids=sample_ids,
            statistics=statistics,
            checksum=checksum,
        )

    def quantize(
        self,
        model: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel:
        """Deep-copy the model while explicitly recording that no quantization occurred."""

        controls = self._controls(config)
        self._require_enabled(controls)
        self._inject_failure("quantize", controls)
        unsupported = sorted(set(policy.precision_map.values()).difference({"float32"}))
        if unsupported:
            raise CapabilityError(
                "Fake optional backend only supports float32 policies",
                component=self.name,
                remediation="Use a float32 precision policy for this no-op test fixture.",
                details={"unsupported_precisions": unsupported},
            )
        if controls.requires_calibration and calibration is None:
            raise QuantizationError(
                "Fake optional backend requires its calibration artifact before quantization",
                component=self.name,
                remediation="Run the calibrate operation before quantize.",
            )
        started = time.perf_counter()
        source = cast(nn.Module, model.model)
        candidate = copy.deepcopy(source)
        loaded = LoadedModel(
            model=candidate,
            tokenizer=copy.deepcopy(model.tokenizer),
            descriptor=model.descriptor,
            load_time_seconds=time.perf_counter() - started,
            load_time_kind="candidate_reconstruction",
        )
        if loaded.load_time_seconds <= 0.0:
            raise RuntimeError("fake candidate reconstruction duration was not measurable")
        parameters: dict[str, str | int | float | bool] = {
            **config.parameters,
            "fake_available": controls.available,
            "fake_supported": controls.supported,
            "fake_fail_at": controls.fail_at,
            "fake_requires_calibration": controls.requires_calibration,
            "performs_real_quantization": False,
        }
        manifest = build_manifest(
            backend=self.name,
            backend_version=self.version,
            method=config.method,
            source=model.descriptor,
            precision_map=policy.precision_map,
            serialized_size_bytes=model_storage_bytes(candidate),
            calibration=calibration,
            parameters=parameters,
            warnings=(_WARNING,),
        )
        return finalize_serialized_manifest(
            QuantizedModel(loaded=loaded, manifest=manifest), config
        )

    def export(
        self,
        model: QuantizedModel,
        destination: Path,
        config: QuantizationConfig,
    ) -> ExportArtifact:
        """Export the standard safe reconstruction recipe or inject a controlled failure."""

        controls = self._controls(config)
        self._require_enabled(controls)
        self._inject_failure("export", controls)
        return export_recipe(model, destination, config)


def create_quantizer(
    *,
    available: bool = True,
    supported: bool = True,
    fail_at: FakeFailurePoint = "none",
    requires_calibration: bool = False,
) -> FakeOptionalQuantizer:
    """Registry factory for the configurable fake optional backend."""

    return FakeOptionalQuantizer(
        available=available,
        supported=supported,
        fail_at=fail_at,
        requires_calibration=requires_calibration,
    )


__all__ = ["FakeFailurePoint", "FakeOptionalQuantizer", "create_quantizer"]
