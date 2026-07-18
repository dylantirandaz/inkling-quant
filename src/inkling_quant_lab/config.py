"""Strict experiment configuration, composition, overrides, and hashing."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.security import sensitive_literal_path

Precision: TypeAlias = Literal["float32", "float16", "bfloat16", "fp8", "int8", "int4"]
DType: TypeAlias = Literal["float32", "float16", "bfloat16"]


class StrictFrozenModel(BaseModel):
    """Base for versioned configuration records."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SecretReference(StrictFrozenModel):
    """Reference to a secret held outside persisted artifacts."""

    env: str = Field(min_length=1, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    required: bool = True

    def resolve(self, environment: Mapping[str, str] | None = None) -> str | None:
        """Resolve the secret for in-memory use without mutating the config."""

        source = os.environ if environment is None else environment
        value = source.get(self.env)
        if self.required and not value:
            raise ConfigurationError(
                f"Required secret environment variable is not set: {self.env}",
                component="config",
            )
        return value


class ModelConfig(StrictFrozenModel):
    """Model identity and safe loading settings."""

    model_id: str = Field(min_length=1)
    revision: str | None = None
    adapter: str = "local_fixture"
    tokenizer_id: str | None = None
    local_snapshot_path: str | None = None
    dtype: DType = "float32"
    trust_remote_code: bool = False
    local_files_only: bool = True
    checkpoint_format: Literal["safetensors", "pickle", "fixture"] = "safetensors"

    @field_validator("local_snapshot_path")
    @classmethod
    def local_snapshot_path_is_explicit(cls, value: str | None) -> str | None:
        """Reject empty or NUL-containing local snapshot references."""

        if value is None:
            return None
        if not value.strip() or "\x00" in value:
            raise ValueError("local_snapshot_path must be a non-empty filesystem path")
        return value


class DecodeConfig(StrictFrozenModel):
    """Deterministic generation parameters shared by baseline and candidate."""

    max_new_tokens: int = Field(default=4, ge=1, le=256)
    do_sample: bool = False
    temperature: float = Field(default=1.0, gt=0.0)
    top_k: int | None = Field(default=None, ge=1)


class EvaluationSuiteConfig(StrictFrozenModel):
    """One evaluator and its immutable local dataset identity."""

    type: Literal[
        "forward_loss",
        "perplexity",
        "generation_regression",
        "exact_match",
        "behavioral_retention",
        "multimodal_contract",
    ]
    dataset: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    split: str = Field(min_length=1)
    prompt_template: str = "{text}"
    decode: DecodeConfig = Field(default_factory=DecodeConfig)
    sample_ids: tuple[str, ...] = ()
    optional: bool = False


class EvaluationConfig(StrictFrozenModel):
    """Evaluation suites and partial-sample failure behavior."""

    suites: tuple[EvaluationSuiteConfig, ...] = ()
    allow_partial: bool = True


class CalibrationConfig(StrictFrozenModel):
    """Calibration dataset provenance and sampling limits."""

    dataset: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    split: str = "calibration"
    sample_ids: tuple[str, ...] = ()
    max_samples: int = Field(default=32, ge=1)


class PolicyRuleConfig(StrictFrozenModel):
    """Named deterministic policy rule."""

    name: str = Field(min_length=1)
    pattern: str = Field(min_length=1)
    precision: Precision


class FrequencyTierConfig(StrictFrozenModel):
    """Frequency quantiles mapped from least to most frequently selected."""

    quantiles: tuple[float, ...] = (0.5,)
    precisions: tuple[Precision, ...] = ("int4", "int8")
    fallback_precision: Precision | None = None

    @model_validator(mode="after")
    def validate_tiers(self) -> FrequencyTierConfig:
        """Validate strictly increasing quantile boundaries and tier count."""

        if tuple(sorted(self.quantiles)) != self.quantiles or any(
            boundary <= 0.0 or boundary >= 1.0 for boundary in self.quantiles
        ):
            raise ValueError("frequency quantiles must be strictly increasing inside (0, 1)")
        if len(set(self.quantiles)) != len(self.quantiles):
            raise ValueError("frequency quantiles must not contain duplicates")
        if len(self.precisions) != len(self.quantiles) + 1:
            raise ValueError("frequency precisions must have exactly len(quantiles) + 1 entries")
        return self


class SensitivityTierConfig(StrictFrozenModel):
    """Sensitivity thresholds and required statistic provenance."""

    method: Literal["reconstruction_error", "loss_impact", "supplied"] = "loss_impact"
    thresholds: tuple[float, ...] = (0.5,)
    precisions: tuple[Precision, ...] = ("int4", "int8")
    fallback_precision: Precision | None = None

    @model_validator(mode="after")
    def validate_tiers(self) -> SensitivityTierConfig:
        """Validate sensitivity thresholds and precision tier count."""

        if tuple(sorted(self.thresholds)) != self.thresholds:
            raise ValueError("sensitivity thresholds must be sorted")
        if len(self.precisions) != len(self.thresholds) + 1:
            raise ValueError("sensitivity precisions must have len(thresholds) + 1 entries")
        return self


class PrecisionPolicyConfig(StrictFrozenModel):
    """Rules used to build a complete module-to-precision assignment."""

    type: Literal["uniform", "frequency_tiered", "sensitivity_tiered", "hybrid"] = "uniform"
    default_precision: Precision = "float32"
    explicit_overrides: dict[str, Precision] = Field(default_factory=dict)
    module_class_rules: tuple[PolicyRuleConfig, ...] = ()
    layer_rules: tuple[PolicyRuleConfig, ...] = ()
    preserve_router_precision: bool = True
    router_precision: Precision = "float32"
    preserve_output_head: bool = True
    output_head_precision: Precision = "float32"
    preserve_embeddings: bool = True
    embedding_precision: Precision = "float32"
    preserve_multimodal_projectors: bool = True
    multimodal_projector_precision: Precision = "float32"
    frequency: FrequencyTierConfig | None = None
    sensitivity: SensitivityTierConfig | None = None
    memory_budget_bytes: int | None = Field(default=None, ge=1)
    memory_budget_tolerance: float = Field(default=0.02, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def required_tier_settings(self) -> PrecisionPolicyConfig:
        """Require statistics configuration for expert-aware policies."""

        if self.type in {"frequency_tiered", "hybrid"} and self.frequency is None:
            raise ValueError(f"policy type {self.type} requires frequency settings")
        if self.type in {"sensitivity_tiered", "hybrid"} and self.sensitivity is None:
            raise ValueError(f"policy type {self.type} requires sensitivity settings")
        return self


class ExportConfig(StrictFrozenModel):
    """Safe candidate export settings."""

    enabled: bool = True
    format: Literal["recipe_json", "safetensors"] = "recipe_json"
    destination: str = "candidate"

    @field_validator("destination")
    @classmethod
    def destination_is_relative(cls, value: str) -> str:
        """Reject absolute or traversing export destinations."""

        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not value:
            raise ValueError("export destination must be a non-empty relative path without '..'")
        return value


class QuantizationConfig(StrictFrozenModel):
    """Quantizer selection, calibration, policy, and export."""

    backend: str = "noop"
    method: str = "none"
    calibration: CalibrationConfig | None = None
    policy: PrecisionPolicyConfig = Field(default_factory=PrecisionPolicyConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    parameters: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RoutingConfig(StrictFrozenModel):
    """Routing storage and privacy mode."""

    mode: Literal["off", "aggregate", "sampled_tokens", "full_trace"] = "off"
    required: bool = False
    capture_router_logits: bool = False
    sampled_token_positions: tuple[int, ...] = ()
    dataset: str = "local://fixtures/routing-prompts"
    dataset_revision: str = "fixture-data-v1"
    dataset_split: str = "routing-evaluation"

    @model_validator(mode="after")
    def sampled_positions_required(self) -> RoutingConfig:
        """Require explicit positions for sampled-token capture."""

        if self.mode == "sampled_tokens" and not self.sampled_token_positions:
            raise ValueError("sampled_tokens mode requires sampled_token_positions")
        if any(position < 0 for position in self.sampled_token_positions):
            raise ValueError("sampled token positions must be non-negative")
        return self


class BenchmarkConfig(StrictFrozenModel):
    """Documented warm-up and repeated-trial protocol."""

    enabled: bool = True
    protocol_version: str = "cpu-v1"
    warmup_iterations: int = Field(default=2, ge=0)
    repetitions: int = Field(default=5, ge=1)
    synchronize: bool = True
    measure_energy: bool = False
    host_memory_mode: Literal[
        "boundary_samples",
        "isolated_stage_worker_peak_rss",
        "isolated_subject_artifact_peak_rss",
    ] = "boundary_samples"
    worker_timeout_seconds: float = Field(default=600.0, gt=0.0, allow_inf_nan=False)


class RuntimeConfig(StrictFrozenModel):
    """Explicit runtime, device, dtype, and placement choices."""

    backend: str = "torch_eager_cpu"
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    dtype: DType = "float32"
    device_map: str = "single"
    sharding: dict[str, str | int] | None = None


class OutputConfig(StrictFrozenModel):
    """Local artifact-store root and run naming."""

    root: str = "artifacts"
    run_prefix: str = "run"

    @field_validator("root", "run_prefix")
    @classmethod
    def reject_parent_traversal(cls, value: str) -> str:
        """Reject ambiguous traversal in configured output fragments."""

        if not value or ".." in Path(value).parts:
            raise ValueError("output paths must be non-empty and cannot contain '..'")
        return value


class SecurityConfig(StrictFrozenModel):
    """Explicit opt-ins for otherwise disabled unsafe behavior."""

    allow_remote_code: bool = False
    allow_pickle_weights: bool = False
    allow_uploads: bool = False
    log_prompts: bool = False
    log_model_outputs: bool = False
    secrets: dict[str, SecretReference] = Field(default_factory=dict)


class ReportingConfig(StrictFrozenModel):
    """Machine and human-readable report outputs."""

    markdown: bool = True
    html: bool = False
    plots: bool = True
    include_interpretation: bool = True


class ExperimentConfig(StrictFrozenModel):
    """Fully resolved versioned experiment configuration."""

    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(min_length=1)
    seed: int = Field(default=17, ge=0)
    model: ModelConfig
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @model_validator(mode="after")
    def validate_cross_component_invariants(self) -> ExperimentConfig:
        """Reject unsupported or unsafe combinations before model loading."""

        reporting_enabled = self.reporting.markdown or self.reporting.html or self.reporting.plots
        if not self.evaluation.suites and (self.benchmark.enabled or reporting_enabled):
            raise ValueError(
                "evaluation.suites must contain at least one suite when benchmark.enabled=true "
                "or reporting output is enabled; add an evaluation suite or disable benchmarking "
                "and reporting.markdown/html/plots for inspection-only use"
            )
        if self.model.trust_remote_code and not self.security.allow_remote_code:
            raise ValueError(
                "model.trust_remote_code=true requires security.allow_remote_code=true"
            )
        if self.model.checkpoint_format == "pickle" and not self.security.allow_pickle_weights:
            raise ValueError("pickle checkpoint format requires security.allow_pickle_weights=true")
        if self.benchmark.host_memory_mode == "isolated_subject_artifact_peak_rss":
            if not self.benchmark.enabled:
                raise ValueError(
                    "isolated_subject_artifact_peak_rss requires benchmark.enabled=true"
                )
            if self.model.adapter != "hf_causal_lm":
                raise ValueError(
                    "isolated_subject_artifact_peak_rss requires model.adapter=hf_causal_lm"
                )
            if (
                self.model.model_id != "ggml-org/stories15M_MOE"
                or self.model.revision != "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
            ):
                raise ValueError(
                    "isolated_subject_artifact_peak_rss is limited to the pinned Stories15M "
                    "revision"
                )
            if not self.model.local_files_only or self.model.checkpoint_format != "safetensors":
                raise ValueError(
                    "isolated_subject_artifact_peak_rss requires offline safetensors model "
                    "metadata and tokenizer files"
                )
            if self.model.trust_remote_code or self.security.allow_remote_code:
                raise ValueError(
                    "isolated_subject_artifact_peak_rss requires both remote-code opt-ins false"
                )
            if (
                self.runtime.backend != "torch_eager_cpu"
                or self.runtime.device != "cpu"
                or self.runtime.dtype != "float32"
                or self.runtime.device_map != "single"
                or self.runtime.sharding is not None
            ):
                raise ValueError(
                    "isolated_subject_artifact_peak_rss requires unsharded single-CPU float32 "
                    "execution"
                )
            if (
                self.quantization.backend != "torch_native_dynamic_int8"
                or self.quantization.method != "native_dynamic_w8a8"
            ):
                raise ValueError(
                    "isolated_subject_artifact_peak_rss requires torch_native_dynamic_int8 "
                    "with native_dynamic_w8a8"
                )
            if (
                not self.quantization.export.enabled
                or self.quantization.export.format != "safetensors"
                or self.quantization.export.destination != "candidate"
            ):
                raise ValueError(
                    "isolated_subject_artifact_peak_rss requires an enabled safetensors export "
                    "at destination=candidate"
                )
        mlx_triple = (
            self.model.adapter,
            self.runtime.backend,
            self.quantization.backend,
        )
        mlx_selected = (
            "mlx_lm_mixtral" in mlx_triple
            or "mlx_metal" in mlx_triple
            or "mlx_affine" in mlx_triple
        )
        if mlx_selected:
            expected = ("mlx_lm_mixtral", "mlx_metal", "mlx_affine")
            if mlx_triple != expected:
                raise ValueError(
                    "the validated MLX path requires the exact composable triple "
                    "model.adapter=mlx_lm_mixtral, runtime.backend=mlx_metal, and "
                    "quantization.backend=mlx_affine"
                )
            if (
                self.model.model_id != "ggml-org/stories15M_MOE"
                or self.model.revision != "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
            ):
                raise ValueError(
                    "the registered MLX contract is limited to the pinned "
                    "ggml-org/stories15M_MOE revision"
                )
            if self.model.local_snapshot_path is None:
                raise ValueError("mlx_lm_mixtral requires model.local_snapshot_path")
            if (
                not self.model.local_files_only
                or self.model.checkpoint_format != "safetensors"
                or self.model.trust_remote_code
                or self.security.allow_remote_code
            ):
                raise ValueError(
                    "mlx_lm_mixtral requires an offline safetensors snapshot with both "
                    "remote-code opt-ins false"
                )
            if self.model.tokenizer_id not in {None, self.model.model_id}:
                raise ValueError("mlx_lm_mixtral requires the tokenizer in the pinned snapshot")
            if (
                self.model.dtype != "float32"
                or self.runtime.dtype != "float32"
                or self.runtime.device != "mps"
                or self.runtime.device_map != "single"
                or self.runtime.sharding is not None
            ):
                raise ValueError(
                    "mlx_metal requires matching float32 model/runtime dtypes, device=mps, "
                    "device_map=single, and no sharding"
                )
            if self.quantization.method != "mlx_affine":
                raise ValueError("mlx_affine requires quantization.method=mlx_affine")
            if self.quantization.calibration is not None:
                raise ValueError("mlx_affine is calibration-free")
            if self.quantization.policy.type != "uniform":
                raise ValueError(
                    "mlx_affine currently supports only a uniform policy over fused expert leaves"
                )
            if self.quantization.export.format != "safetensors":
                raise ValueError("mlx_affine exports only safetensors bundles")
            if self.benchmark.enabled:
                raise ValueError(
                    "the registered MLX pipeline benchmark is not implemented; disable benchmark "
                    "rather than executing the PyTorch-specific benchmark loop"
                )
        if self.runtime.device == "cpu" and (
            self.quantization.backend in {"fp8", "torch_fp8"} or self.quantization.method == "fp8"
        ):
            raise ValueError("FP8 quantization is unsupported on the CPU runtime")
        cpu_quantizers = {
            "torch_dynamic_int8",
            "torch_weight_only_int4",
            "torch_reference_mixed",
            "torch_native_dynamic_int8",
            "torch_native_int4_kleidiai",
        }
        if self.quantization.backend in cpu_quantizers and self.runtime.device != "cpu":
            raise ValueError(f"the {self.quantization.backend} backend requires runtime.device=cpu")
        native_cpu_quantizers = {
            "torch_native_dynamic_int8",
            "torch_native_int4_kleidiai",
        }
        if self.quantization.backend in native_cpu_quantizers and (
            self.model.dtype != "float32" or self.runtime.dtype != "float32"
        ):
            raise ValueError(
                f"the {self.quantization.backend} backend requires explicit float32 model and "
                "runtime dtypes"
            )
        calibration = self.quantization.calibration
        if self.quantization.policy.type != "uniform" and calibration is None:
            raise ValueError("expert-aware precision policies require an explicit calibration set")
        if calibration is not None:
            for suite in self.evaluation.suites:
                same_dataset = (
                    calibration.dataset == suite.dataset
                    and calibration.revision == suite.revision
                    and calibration.split == suite.split
                )
                overlap = set(calibration.sample_ids).intersection(suite.sample_ids)
                if same_dataset or overlap:
                    raise ValueError(
                        "calibration and evaluation data must be disjoint; "
                        f"conflict with evaluation suite {suite.type}"
                    )
        literal_secret = sensitive_literal_path(self.model_dump(mode="json"))
        if literal_secret is not None:
            raise ValueError(
                "configuration contains literal credential material at "
                + ".".join(literal_secret)
                + "; use security.secrets with an environment-variable reference"
            )
        return self

    def canonical_dict(self) -> dict[str, Any]:
        """Return the complete secret-safe JSON representation."""

        return self.model_dump(mode="json", exclude_none=False)

    def canonical_json(self) -> str:
        """Serialize deterministically for hashing and equality checks."""

        return json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def config_hash(self) -> str:
        """Return the SHA-256 digest of canonical resolved configuration."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def resolve_secrets(self, environment: Mapping[str, str] | None = None) -> dict[str, str]:
        """Resolve configured secret references into an ephemeral mapping."""

        resolved: dict[str, str] = {}
        for name, reference in sorted(self.security.secrets.items()):
            value = reference.resolve(environment)
            if value is not None:
                resolved[name] = value
        return resolved


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and sequence values."""

    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping with domain-specific diagnostics."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigurationError(
            f"Unable to read config {path}: {error}", component="config"
        ) from error
    except yaml.YAMLError as error:
        raise ConfigurationError(f"Invalid YAML in {path}: {error}", component="config") from error
    if not isinstance(raw, dict):
        raise ConfigurationError(f"Config root must be a mapping: {path}", component="config")
    return {str(key): value for key, value in raw.items()}


def _load_composed_mapping(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Resolve ``extends`` files from oldest base to the current experiment."""

    resolved_path = path.expanduser().resolve()
    if resolved_path in stack:
        chain = " -> ".join(str(item) for item in (*stack, resolved_path))
        raise ConfigurationError(f"Cyclic config composition: {chain}", component="config")
    raw = _load_yaml_mapping(resolved_path)
    extends_value = raw.pop("extends", ())
    if isinstance(extends_value, str):
        extends: Sequence[str] = (extends_value,)
    elif isinstance(extends_value, list) and all(isinstance(item, str) for item in extends_value):
        extends = extends_value
    elif extends_value in (None, ()):
        extends = ()
    else:
        raise ConfigurationError("extends must be a path or list of paths", component="config")
    merged: dict[str, Any] = {}
    for reference in extends:
        component_path = (resolved_path.parent / reference).resolve()
        merged = _deep_merge(
            merged, _load_composed_mapping(component_path, (*stack, resolved_path))
        )
    return _deep_merge(merged, raw)


def _assign_dotted(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Assign a parsed CLI override using dotted mapping keys."""

    parts = dotted_key.split(".")
    if not parts or any(not part for part in parts):
        raise ConfigurationError(f"Invalid override key: {dotted_key}", component="config")
    cursor = mapping
    for part in parts[:-1]:
        existing = cursor.get(part)
        if existing is None:
            nested: dict[str, Any] = {}
            cursor[part] = nested
            cursor = nested
        elif isinstance(existing, dict):
            cursor = existing
        else:
            raise ConfigurationError(
                f"Override path crosses a non-mapping field: {dotted_key}", component="config"
            )
    cursor[parts[-1]] = value


def parse_overrides(overrides: Sequence[str]) -> dict[str, Any]:
    """Parse ``key=value`` overrides with YAML scalar semantics."""

    result: dict[str, Any] = {}
    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator:
            raise ConfigurationError(
                f"Override must use key=value syntax: {override}", component="config"
            )
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as error:
            raise ConfigurationError(
                f"Invalid override {override}: {error}", component="config"
            ) from error
        _assign_dotted(result, key, value)
    return result


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Load component YAML, apply CLI overrides, and validate the full config."""

    config_path = Path(path)
    raw = _load_composed_mapping(config_path)
    raw = _deep_merge(raw, parse_overrides(overrides))
    return _validate_experiment_config(raw)


def load_inspection_config(target: str | Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Load a model fragment or identifier as a safe inspection configuration.

    Model files under ``configs/models`` intentionally contain only the ``model``
    section. Inspection still uses the complete experiment schema so runtime and
    security defaults remain explicit and every ordinary cross-component
    invariant is enforced. Direct identifiers select only conservative adapters:
    built-in fixture URIs use the local fixture adapter, while all other IDs use
    the safe, local-files-only Hugging Face extension point.
    """

    text = str(target)
    config_path = Path(target).expanduser()
    if config_path.exists() or config_path.suffix.lower() in {".yaml", ".yml"}:
        raw = _load_composed_mapping(config_path)
    elif text.startswith("local://fixtures/"):
        raw = {
            "model": {
                "model_id": text,
                "adapter": "local_fixture",
                "checkpoint_format": "fixture",
                "trust_remote_code": False,
                "local_files_only": True,
            }
        }
    else:
        raw = {
            "model": {
                "model_id": text,
                "adapter": "hf_causal_lm",
                "checkpoint_format": "safetensors",
                "trust_remote_code": False,
                "local_files_only": True,
            }
        }
    raw.setdefault("benchmark", {"enabled": False})
    raw.setdefault(
        "reporting",
        {
            "markdown": False,
            "html": False,
            "plots": False,
            "include_interpretation": False,
        },
    )
    raw = _deep_merge(raw, parse_overrides(overrides))
    raw.setdefault("name", "model-inspection")
    return _validate_experiment_config(raw)


def _validate_experiment_config(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Validate one resolved mapping with stable, field-specific diagnostics."""

    try:
        return ExperimentConfig.model_validate(raw)
    except ValidationError as error:
        details = [
            {"field": ".".join(str(item) for item in issue["loc"]), "message": issue["msg"]}
            for issue in error.errors(include_url=False)
        ]
        summary = "; ".join(f"{item['field']}: {item['message']}" for item in details)
        raise ConfigurationError(
            f"Configuration validation failed: {summary}",
            component="config",
            details={"errors": details},
        ) from error


def dump_resolved_config(config: ExperimentConfig) -> str:
    """Serialize resolved config deterministically as human-readable YAML."""

    return yaml.safe_dump(config.canonical_dict(), sort_keys=True, allow_unicode=True)
