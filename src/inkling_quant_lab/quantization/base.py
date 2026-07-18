"""Quantization capability, artifact, and backend contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from inkling_quant_lab.config import CalibrationConfig, QuantizationConfig
from inkling_quant_lab.models.base import LoadedModel, ModelBatch, ModelDescriptor
from inkling_quant_lab.runtimes.base import RuntimeCapabilities


class ImmutableArtifact(BaseModel):
    """Strict immutable artifact schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SupportReport(ImmutableArtifact):
    """Actionable pre-load/pre-quantization capability result."""

    available: bool
    supported: bool
    component: str
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    install_extra: str | None = None
    remediation: str | None = None
    supported_precisions: tuple[str, ...] = ()

    def message(self) -> str:
        """Return a concise actionable human-readable summary."""

        parts = list(self.reasons)
        if self.remediation:
            parts.append(self.remediation)
        elif self.install_extra:
            parts.append(f"install with `uv sync --extra {self.install_extra}`")
        return "; ".join(parts) if parts else "supported"


class CalibrationArtifact(ImmutableArtifact):
    """Calibration provenance and optional computed statistics."""

    config: CalibrationConfig
    sample_ids: tuple[str, ...]
    statistics: dict[str, float] = Field(default_factory=dict)
    checksum: str


class QuantizationManifest(ImmutableArtifact):
    """Complete quantization provenance required for a candidate."""

    schema_version: str = "1.0"
    backend: str
    backend_version: str
    method: str
    source_model_checksum: str
    module_precision_map: dict[str, str]
    excluded_modules: tuple[str, ...]
    calibration_dataset: dict[str, str] | None = None
    calibration_sample_ids: tuple[str, ...] = ()
    calibration_checksum: str | None = None
    calibration_sample_count: int | None = Field(default=None, ge=0)
    calibration_token_count: int | None = Field(default=None, ge=0)
    quantization_parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    serialized_size_bytes: int = Field(ge=0)
    warnings: tuple[str, ...] = ()


@dataclass(slots=True)
class QuantizedModel:
    """In-memory candidate and its quantization manifest."""

    loaded: LoadedModel
    manifest: QuantizationManifest


class ExportArtifact(ImmutableArtifact):
    """Portable safe export metadata."""

    path: str
    format: str
    sha256: str
    size_bytes: int = Field(ge=0)
    files: tuple[str, ...] = ()
    reload_recipe: dict[str, Any] = Field(default_factory=dict)


class ResolvedPolicyLike(Protocol):
    """Narrow policy surface consumed by quantizers."""

    @property
    def precision_map(self) -> dict[str, str]: ...


class Quantizer(Protocol):
    """Shared backend contract."""

    name: str

    def check_support(
        self,
        model: ModelDescriptor,
        runtime: RuntimeCapabilities,
        config: QuantizationConfig,
    ) -> SupportReport: ...

    def calibrate(
        self,
        model: LoadedModel,
        samples: tuple[ModelBatch, ...],
        config: QuantizationConfig,
    ) -> CalibrationArtifact | None: ...

    def quantize(
        self,
        model: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel: ...

    def export(
        self, model: QuantizedModel, destination: Path, config: QuantizationConfig
    ) -> ExportArtifact: ...


@runtime_checkable
class PinnedSourceReloadQuantizer(Protocol):
    """Quantizer whose upstream integration must reload the immutable source.

    Some production libraries own both model loading and quantization.  This
    separate protocol keeps that lifecycle explicit instead of pretending the
    already-loaded baseline module was transformed in place.
    """

    lifecycle: Literal["pinned_source_reload"]

    def quantize_from_pinned_source(
        self,
        source: LoadedModel,
        policy: ResolvedPolicyLike,
        calibration: CalibrationArtifact | None,
        config: QuantizationConfig,
    ) -> QuantizedModel: ...
