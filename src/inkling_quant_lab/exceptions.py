"""Domain-specific errors with stable machine-readable codes."""

from __future__ import annotations

from typing import Any


class InklingQuantError(Exception):
    """Base class for errors safe to present through the CLI."""

    code = "IQL_ERROR"

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        component: str | None = None,
        remediation: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.stage = stage
        self.component = component
        self.remediation = remediation
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable error representation."""

        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "component": self.component,
            "remediation": self.remediation,
            "details": self.details,
        }


class ConfigurationError(InklingQuantError):
    """The resolved configuration violates a schema or invariant."""

    code = "CONFIGURATION_ERROR"


class CapabilityError(InklingQuantError):
    """A requested component combination is unavailable or unsupported."""

    code = "CAPABILITY_ERROR"


class ModelLoadError(InklingQuantError):
    """A model could not be loaded safely."""

    code = "MODEL_LOAD_ERROR"


class QuantizationError(InklingQuantError):
    """Quantization or export failed."""

    code = "QUANTIZATION_ERROR"


class RoutingInstrumentationError(InklingQuantError):
    """Routing hooks or trace processing failed."""

    code = "ROUTING_INSTRUMENTATION_ERROR"


class EvaluationError(InklingQuantError):
    """An evaluator failed outside its configured partial-failure policy."""

    code = "EVALUATION_ERROR"


class BenchmarkError(InklingQuantError):
    """A benchmark protocol could not be completed."""

    code = "BENCHMARK_ERROR"


class ArtifactIntegrityError(InklingQuantError):
    """An artifact path, checksum, fingerprint, or lifecycle is invalid."""

    code = "ARTIFACT_INTEGRITY_ERROR"


class ComparisonCompatibilityError(InklingQuantError):
    """Runs cannot be compared under the requested contract."""

    code = "COMPARISON_COMPATIBILITY_ERROR"
