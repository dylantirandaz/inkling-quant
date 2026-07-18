#!/usr/bin/env python3
"""Compare native CPU linear quantization on the pinned public Mixtral model.

This is deliberately a narrow quality experiment.  Transformers' fused
Mixtral expert tensors are not addressable ``nn.Linear`` modules, so they stay
float32.  Only ordinary linear leaves outside the protected router and output
head are eligible for the capability-gated native CPU kernels.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from torch import nn

from inkling_quant_lab.config import EvaluationSuiteConfig, ExperimentConfig, load_config
from inkling_quant_lab.evaluation.noninferiority import (
    PairedNLLObservation,
    StratumDesign,
    paired_stratified_nll_noninferiority,
)
from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.hardware import probe_environment
from inkling_quant_lab.models.base import LoadedModel, ModelBatch, ModuleInfo
from inkling_quant_lab.models.hf_causal_lm import HFCausalLMAdapter
from inkling_quant_lab.quantization.base import QuantizedModel
from inkling_quant_lab.quantization.native_cpu import (
    NativeDynamicInt8Linear,
    NativeInt4KleidiAILinear,
    TorchNativeDynamicInt8Quantizer,
    TorchNativeInt4KleidiAIQuantizer,
)
from inkling_quant_lab.quantization.policies import (
    ResolvedPrecisionPolicy,
    resolve_precision_policy,
)
from inkling_quant_lab.quantization.reference import model_storage_bytes
from inkling_quant_lab.routing import BatchMeta, InMemoryRoutingSink, compare_routing
from inkling_quant_lab.routing.traces import RoutingArtifact
from inkling_quant_lab.runtimes.base import RuntimeCapabilities
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime
from inkling_quant_lab.security import safe_path

_MODEL_ID = "ggml-org/stories15M_MOE"
_MODEL_REVISION = "b6dd737497465570b5f5e962dbc9d9454ed1e0eb"
_MODEL_WEIGHT_FILE = "model.safetensors"
_MODEL_WEIGHT_SHA256 = "dbfa0289f68a8dd721d10eb12d8bd82e098455682027f6f9986ba548913f9082"
_MODEL_WEIGHT_SIZE_BYTES = 72_744_704
OFFICIAL_TINYSTORIES_VALID_SHA256 = (
    "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
)
_DATASET_LICENSE = "cdla-sharing-1.0"
_MODEL_LICENSE = "MIT"
_STORY_SEPARATOR = "<|endoftext|>"
_SAMPLE_PREFIX = "story-"
_CONFIRMATORY_PROTOCOL_ID = "stories15m-native-int8-confirmatory-quality-v1"
_CONFIRMATORY_QUALITY_RANK_PROTOCOL = "tinystories-confirmatory-length-stratified-v1"
_CONFIRMATORY_ROUTING_RANK_PROTOCOL = "tinystories-confirmatory-generation-routing-subset-v1"
_TINYSTORIES_DATASET_ID = "hf://datasets/roneneldan/TinyStories/TinyStories-valid.txt"
_TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
_CONFIRMATORY_SEED = 20260715
_CONFIRMATORY_STORY_COUNT = 21_990
_CONFIRMATORY_EXCLUDED_COUNT = 32
_CONFIRMATORY_ELIGIBLE_COUNT = 21_958
_CONFIRMATORY_MAX_CONTEXT = 256
_CONFIRMATORY_STRATUM_COUNT = 4
_CONFIRMATORY_QUALITY_PER_STRATUM = 64
_CONFIRMATORY_ROUTING_PER_STRATUM = 4
_CONFIRMATORY_EXPECTED_ROUTING_EVENTS = 19_764
_TOKENIZER_FILE_SHA256 = {
    "config.json": "e901e012953d1df93574b2cc3d7db5ed4758d52f8bd4a7dd4b647936e32261be",
    "generation_config.json": ("295aa491adda22ab9fbdecdda9e8121e8348fd0eea0529d8802993426ab0892c"),
    "special_tokens_map.json": ("ff3b4a612c4e447acb02d40071bddd989fe0da87eb5b7fe0dbadfc4f74de7531"),
    "tokenizer.json": "8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1",
    "tokenizer_config.json": ("33d29c87e41f7dd1efb0434d852730320c82970f292be452d820539bce417052"),
}

BackendName = Literal["torch_native_dynamic_int8", "torch_native_int4_kleidiai"]


class _StrictProtocolModel(BaseModel):
    """Immutable, unknown-field-rejecting prospective protocol record."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class ConfirmatoryModelIdentity(_StrictProtocolModel):
    model_id: Literal["ggml-org/stories15M_MOE"]
    revision: Literal["b6dd737497465570b5f5e962dbc9d9454ed1e0eb"]
    max_context_tokens: Literal[256]
    source_weight_file: Literal["model.safetensors"]
    source_weight_size_bytes: Literal[72744704]
    source_weight_sha256: Literal[
        "dbfa0289f68a8dd721d10eb12d8bd82e098455682027f6f9986ba548913f9082"
    ]
    config_json_sha256: Literal["e901e012953d1df93574b2cc3d7db5ed4758d52f8bd4a7dd4b647936e32261be"]
    generation_config_json_sha256: Literal[
        "295aa491adda22ab9fbdecdda9e8121e8348fd0eea0529d8802993426ab0892c"
    ]


class ConfirmatoryDatasetIdentity(_StrictProtocolModel):
    dataset_id: Literal["hf://datasets/roneneldan/TinyStories/TinyStories-valid.txt"]
    revision: Literal["f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"]
    split: Literal["validation"]
    file_sha256: Literal["94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"]
    story_count: Literal[21990]
    license: Literal["cdla-sharing-1.0"]


class ConfirmatoryTokenizerIdentity(_StrictProtocolModel):
    add_special_tokens: Literal[True]
    special_tokens_map_json_sha256: Literal[
        "ff3b4a612c4e447acb02d40071bddd989fe0da87eb5b7fe0dbadfc4f74de7531"
    ]
    tokenizer_json_sha256: Literal[
        "8eea70c4866c4f1320ba096fc986ac82038a8374dbe135212ba7628835b4a6f1"
    ]
    tokenizer_config_json_sha256: Literal[
        "33d29c87e41f7dd1efb0434d852730320c82970f292be452d820539bce417052"
    ]


class ConfirmatoryGenerationDecode(_StrictProtocolModel):
    max_new_tokens: Literal[8]
    do_sample: Literal[False]
    temperature: float
    top_k: None

    @model_validator(mode="after")
    def exact_temperature(self) -> ConfirmatoryGenerationDecode:
        if self.temperature != 1.0:
            raise ValueError("confirmatory generation temperature must be exactly 1.0")
        return self


class ConfirmatoryExecutionContract(_StrictProtocolModel):
    baseline_resolved_config_sha256: Literal[
        "e6db2959221babf9aaba1f529a20349fe3462d4309b6929caf3a7d3331d668f2"
    ]
    candidate_resolved_config_sha256: Literal[
        "d6ee2054d6db801c56596042df7e9eca130109ab922ee46735b61953da8d7912"
    ]
    candidate_backend: Literal["torch_native_dynamic_int8"]
    candidate_method: Literal["native_dynamic_w8a8"]
    quantized_engine: Literal["qnnpack"]
    activation_granularity: Literal["per_call_tensor"]
    weight_granularity: Literal["per_output_channel"]
    generation_decode: ConfirmatoryGenerationDecode
    runtime_backend: Literal["torch_eager_cpu"]
    runtime_device: Literal["cpu"]
    runtime_dtype: Literal["float32"]
    runtime_device_map: Literal["single"]


class ConfirmatoryStratum(_StrictProtocolModel):
    stratum: int = Field(ge=0, le=3)
    population_story_count: int = Field(gt=0)
    population_evaluated_token_count: int = Field(gt=0)
    minimum_evaluated_tokens: int = Field(gt=0, le=255)
    maximum_evaluated_tokens: int = Field(gt=0, le=255)
    selected_story_count: Literal[64]
    selected_evaluated_token_count: int = Field(gt=0)


class ConfirmatoryHoldoutSelection(_StrictProtocolModel):
    seed: Literal[20260715]
    excluded_sample_ids_first: Literal["story-000001"]
    excluded_sample_ids_last: Literal["story-000032"]
    excluded_sample_count: Literal[32]
    excluded_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    eligible_story_count: Literal[21958]
    eligible_evaluated_token_count: int = Field(gt=0)
    length_stratum_assignment: Literal[
        "sorted_by_evaluated_token_count_then_sample_id_floor_4i_over_N"
    ]
    quality_rank_protocol: Literal["tinystories-confirmatory-length-stratified-v1"]
    quality_sample_count_per_stratum: Literal[64]
    quality_sample_count: Literal[256]
    quality_sample_id_list_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_ids_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_input_token_ids_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_evaluated_token_count: int = Field(gt=0)
    quality_source_token_count: int = Field(gt=0)
    quality_truncated_sample_count: int = Field(ge=0, le=256)
    strata: tuple[ConfirmatoryStratum, ...]

    @field_validator("strata", mode="before")
    @classmethod
    def normalize_yaml_sequence(cls, value: Any) -> Any:
        """Normalize only YAML's native sequence representation; keep scalars strict."""

        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def exact_strata(self) -> ConfirmatoryHoldoutSelection:
        if tuple(item.stratum for item in self.strata) != (0, 1, 2, 3):
            raise ValueError("confirmatory holdout strata must be exactly 0, 1, 2, 3")
        if sum(item.population_story_count for item in self.strata) != 21_958:
            raise ValueError("confirmatory stratum populations must sum to 21,958")
        if sum(int(item.selected_story_count) for item in self.strata) != 256:
            raise ValueError("confirmatory stratum selections must sum to 256")
        return self


class ConfirmatoryRoutingSelection(_StrictProtocolModel):
    selection_protocol: Literal["tinystories-confirmatory-generation-routing-subset-v1"]
    sample_count_per_stratum: Literal[4]
    sample_count: Literal[16]
    sample_id_list_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ids_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    routing_input_token_ids_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generation_prompt_token_ids_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluated_token_count: int = Field(gt=0)
    routed_input_token_count: int = Field(gt=0)
    expected_layer_token_event_count: Literal[19764]
    inference_claim: Literal["descriptive_only"]


class ConfirmatoryBootstrap(_StrictProtocolModel):
    method: Literal["paired_story_cluster_within_length_strata_srswor_moment_matching"]
    replicates: Literal[100000]
    rng: Literal["numpy.random.Generator(numpy.random.PCG64)"]
    seed: Literal[20260715]
    quantile_method: Literal["linear"]
    finite_population_correction: Literal["per_stratum_moment_matched"]


class ConfirmatoryNoninferiority(_StrictProtocolModel):
    target: Literal["finite_holdout_population_token_weighted_candidate_minus_baseline_nll"]
    estimator: Literal["stratified_horvitz_thompson_numerator_over_known_token_denominator"]
    margin_relative_perplexity: float = Field(gt=0.0)
    margin_nats_per_token: float = Field(gt=0.0)
    decision_rule: Literal["one_sided_upper_bound_strictly_less_than_margin"]
    confidence_level: float = Field(gt=0.5, lt=1.0)
    bootstrap: ConfirmatoryBootstrap
    analytic_standard_error: Literal["stratified_srswor_design_based"]

    @model_validator(mode="after")
    def exact_frozen_thresholds(self) -> ConfirmatoryNoninferiority:
        expected = (0.005, 0.004987541511039074, 0.95)
        observed = (
            self.margin_relative_perplexity,
            self.margin_nats_per_token,
            self.confidence_level,
        )
        if observed != expected:
            raise ValueError(
                "confirmatory noninferiority thresholds must be exactly +0.5% PPL, "
                "log1p(0.005) nats, and 95% confidence"
            )
        return self


class ConfirmatoryRepeatability(_StrictProtocolModel):
    required_clean_process_executions: Literal[2]
    deterministic_decode: Literal[True]
    calibration_data: Literal["not_applicable_dynamic_int8"]


class ConfirmatoryClaims(_StrictProtocolModel):
    domain: Literal["pinned_tinystories_validation_holdout_only"]
    routing_and_generation_inferential: Literal[False]
    model_seed_uncertainty_covered: Literal[False]
    hardware_uncertainty_covered: Literal[False]
    cross_domain_generalization: Literal[False]


class ConfirmatoryProtocol(_StrictProtocolModel):
    """The exact, prospectively frozen Stories15M native-INT8 quality protocol."""

    schema_version: Literal["1.0"]
    protocol_id: Literal["stories15m-native-int8-confirmatory-quality-v1"]
    model: ConfirmatoryModelIdentity
    dataset: ConfirmatoryDatasetIdentity
    tokenizer: ConfirmatoryTokenizerIdentity
    execution_contract: ConfirmatoryExecutionContract
    holdout_selection: ConfirmatoryHoldoutSelection
    generation_and_routing: ConfirmatoryRoutingSelection
    noninferiority: ConfirmatoryNoninferiority
    repeatability: ConfirmatoryRepeatability
    claims: ConfirmatoryClaims


@dataclass(frozen=True, slots=True)
class ConfirmatorySample:
    """Ephemeral text/tokens plus the content-redacted selection manifest fields."""

    sample_id: str
    story: str
    content_sha256: str
    token_ids: tuple[int, ...]
    source_token_count: int
    evaluated_token_count: int
    truncated: bool
    length_stratum: int

    def manifest_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "content_sha256": self.content_sha256,
            "source_token_count": self.source_token_count,
            "evaluated_token_count": self.evaluated_token_count,
            "truncated": self.truncated,
            "length_stratum": self.length_stratum,
        }


@dataclass(frozen=True, slots=True)
class ConfirmatorySelection:
    """Fully re-derived prospective holdout and descriptive subset selection."""

    quality: tuple[ConfirmatorySample, ...]
    generation_and_routing: tuple[ConfirmatorySample, ...]
    excluded_selection_sha256: str
    eligible_evaluated_token_count: int
    quality_sample_id_list_sha256: str
    quality_ids_content_sha256: str
    quality_manifest_sha256: str
    quality_input_token_ids_manifest_sha256: str
    generation_sample_id_list_sha256: str
    generation_ids_content_sha256: str
    generation_manifest_sha256: str
    routing_input_token_ids_manifest_sha256: str
    generation_prompt_token_ids_manifest_sha256: str
    strata: tuple[dict[str, int], ...]


@dataclass(frozen=True, slots=True)
class PreparedPublicData:
    """Checksum-verified content selected by stable IDs from the source file."""

    dataset_sha256: str
    story_count: int
    perplexity_ids: tuple[str, ...]
    perplexity_stories: tuple[str, ...]
    perplexity_selection_sha256: str
    generation_ids: tuple[str, ...]
    generation_stories: tuple[str, ...]
    generation_selection_sha256: str
    confirmatory_selection: ConfirmatorySelection | None = None


@dataclass(frozen=True, slots=True)
class ModelMeasurement:
    """Measured quality plus ephemeral generation tokens and routing traces."""

    quality: dict[str, Any]
    generation: dict[str, Any]
    generation_tokens: dict[str, tuple[int, ...]]
    routing: RoutingArtifact
    elapsed_seconds: float
    routing_observer_proof: dict[str, Any] | None = None


def file_sha256(path: Path) -> str:
    """Hash one local source file without retaining its contents in evidence."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_official_dataset_sha256(value: str) -> str:
    """Accept only the pinned official TinyStories validation-file digest."""

    if value != OFFICIAL_TINYSTORIES_VALID_SHA256:
        raise ValueError(
            "public native quality evidence requires the official TinyStories validation "
            f"SHA-256 {OFFICIAL_TINYSTORIES_VALID_SHA256}"
        )
    return value


def require_hardware_label(value: str) -> str:
    """Require explicit, whitespace-exact operator hardware identity metadata."""

    if not value or value != value.strip():
        raise ValueError(
            "public native quality evidence requires an exact non-empty hardware label"
        )
    return value


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_confirmatory_protocol(path: Path) -> ConfirmatoryProtocol:
    """Safely load the exact prospective protocol and reject unknown fields."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"unable to read confirmatory protocol {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("confirmatory protocol YAML must contain one mapping")
    try:
        return ConfirmatoryProtocol.model_validate(raw)
    except ValidationError as error:
        raise ValueError(f"invalid confirmatory protocol {path}: {error}") from error


def _verified_cached_tokenizer_files(config: ExperimentConfig) -> dict[str, str]:
    """Hash all five config/tokenizer files frozen by the prospective protocol."""

    from huggingface_hub import hf_hub_download

    observed: dict[str, str] = {}
    for filename, expected in _TOKENIZER_FILE_SHA256.items():
        resolved = Path(
            hf_hub_download(
                repo_id=config.model.model_id,
                filename=filename,
                revision=config.model.revision,
                local_files_only=config.model.local_files_only,
            )
        )
        actual = file_sha256(resolved)
        if actual != expected:
            raise ValueError(
                f"cached pinned {filename} does not match confirmatory tokenizer identity"
            )
        observed[filename] = actual
    return observed


def _content_identity_records(
    samples: tuple[ConfirmatorySample, ...],
) -> list[dict[str, str]]:
    return [
        {"sample_id": sample.sample_id, "content_sha256": sample.content_sha256}
        for sample in samples
    ]


def _token_ids_manifest_sha256(
    samples: tuple[ConfirmatorySample, ...], *, maximum_tokens: int
) -> str:
    records = [
        {
            "sample_id": sample.sample_id,
            "token_ids_sha256": _canonical_json_sha256(list(sample.token_ids[:maximum_tokens])),
        }
        for sample in samples
    ]
    return _canonical_json_sha256(records)


def _assert_protocol_value(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(
            f"confirmatory protocol mismatch for {name}: expected {expected!r}, "
            f"re-derived {actual!r}"
        )


def derive_confirmatory_selection(
    stories: tuple[str, ...],
    tokenizer: Any,
    protocol: ConfirmatoryProtocol,
) -> ConfirmatorySelection:
    """Re-derive the frozen length strata and both hash-ranked sample sets."""

    _assert_protocol_value("dataset.story_count", len(stories), protocol.dataset.story_count)
    excluded_ids = tuple(f"story-{index:06d}" for index in range(1, 33))
    excluded_stories = select_stories(stories, excluded_ids)
    excluded_sha256 = _selection_sha256(excluded_ids, excluded_stories)
    _assert_protocol_value(
        "holdout_selection.excluded_selection_sha256",
        excluded_sha256,
        protocol.holdout_selection.excluded_selection_sha256,
    )

    excluded = set(excluded_ids)
    eligible: list[ConfirmatorySample] = []
    for index, story in enumerate(stories):
        sample_id = f"story-{index:06d}"
        if sample_id in excluded:
            continue
        encoded = tuple(int(token) for token in tokenizer.encode(story, special_tokens=True))
        token_ids = encoded[: protocol.model.max_context_tokens]
        if len(token_ids) < 2:
            raise ValueError(f"TinyStories sample {sample_id} has fewer than two tokens")
        eligible.append(
            ConfirmatorySample(
                sample_id=sample_id,
                story=story,
                content_sha256=hashlib.sha256(story.encode("utf-8")).hexdigest(),
                token_ids=token_ids,
                source_token_count=len(encoded),
                evaluated_token_count=len(token_ids) - 1,
                truncated=len(encoded) > protocol.model.max_context_tokens,
                length_stratum=-1,
            )
        )
    _assert_protocol_value(
        "holdout_selection.eligible_story_count",
        len(eligible),
        protocol.holdout_selection.eligible_story_count,
    )

    ordered_by_length = sorted(
        eligible, key=lambda sample: (sample.evaluated_token_count, sample.sample_id)
    )
    assigned: list[ConfirmatorySample] = []
    for position, sample in enumerate(ordered_by_length):
        stratum = min(
            _CONFIRMATORY_STRATUM_COUNT - 1,
            (_CONFIRMATORY_STRATUM_COUNT * position) // _CONFIRMATORY_ELIGIBLE_COUNT,
        )
        assigned.append(
            ConfirmatorySample(
                sample_id=sample.sample_id,
                story=sample.story,
                content_sha256=sample.content_sha256,
                token_ids=sample.token_ids,
                source_token_count=sample.source_token_count,
                evaluated_token_count=sample.evaluated_token_count,
                truncated=sample.truncated,
                length_stratum=stratum,
            )
        )

    quality: list[ConfirmatorySample] = []
    strata_records: list[dict[str, int]] = []
    for stratum in range(_CONFIRMATORY_STRATUM_COUNT):
        population = tuple(sample for sample in assigned if sample.length_stratum == stratum)

        def quality_rank(sample: ConfirmatorySample) -> tuple[str, str]:
            rank = _canonical_json_sha256(
                {
                    "content_sha256": sample.content_sha256,
                    "dataset_sha256": protocol.dataset.file_sha256,
                    "excluded_prior_selection_sha256": excluded_sha256,
                    "protocol": _CONFIRMATORY_QUALITY_RANK_PROTOCOL,
                    "sample_id": sample.sample_id,
                    "seed": protocol.holdout_selection.seed,
                }
            )
            return (rank, sample.sample_id)

        selected = tuple(sorted(population, key=quality_rank)[:_CONFIRMATORY_QUALITY_PER_STRATUM])
        quality.extend(selected)
        strata_records.append(
            {
                "stratum": stratum,
                "population_story_count": len(population),
                "population_evaluated_token_count": sum(
                    sample.evaluated_token_count for sample in population
                ),
                "minimum_evaluated_tokens": min(
                    sample.evaluated_token_count for sample in population
                ),
                "maximum_evaluated_tokens": max(
                    sample.evaluated_token_count for sample in population
                ),
                "selected_story_count": len(selected),
                "selected_evaluated_token_count": sum(
                    sample.evaluated_token_count for sample in selected
                ),
            }
        )

    quality_ordered = tuple(sorted(quality, key=lambda sample: sample.sample_id))
    quality_id_sha256 = _canonical_json_sha256([sample.sample_id for sample in quality_ordered])
    quality_content_sha256 = _canonical_json_sha256(_content_identity_records(quality_ordered))
    quality_manifest_sha256 = _canonical_json_sha256(
        [sample.manifest_record() for sample in quality_ordered]
    )

    generation: list[ConfirmatorySample] = []
    for stratum in range(_CONFIRMATORY_STRATUM_COUNT):
        population = tuple(sample for sample in quality_ordered if sample.length_stratum == stratum)

        def generation_rank(sample: ConfirmatorySample) -> tuple[str, str]:
            rank = _canonical_json_sha256(
                {
                    "content_sha256": sample.content_sha256,
                    "dataset_sha256": protocol.dataset.file_sha256,
                    "quality_manifest_sha256": quality_manifest_sha256,
                    "protocol": _CONFIRMATORY_ROUTING_RANK_PROTOCOL,
                    "sample_id": sample.sample_id,
                    "seed": protocol.holdout_selection.seed,
                }
            )
            return (rank, sample.sample_id)

        generation.extend(
            sorted(population, key=generation_rank)[:_CONFIRMATORY_ROUTING_PER_STRATUM]
        )
    generation_ordered = tuple(sorted(generation, key=lambda sample: sample.sample_id))

    selection = ConfirmatorySelection(
        quality=quality_ordered,
        generation_and_routing=generation_ordered,
        excluded_selection_sha256=excluded_sha256,
        eligible_evaluated_token_count=sum(sample.evaluated_token_count for sample in assigned),
        quality_sample_id_list_sha256=quality_id_sha256,
        quality_ids_content_sha256=quality_content_sha256,
        quality_manifest_sha256=quality_manifest_sha256,
        quality_input_token_ids_manifest_sha256=_token_ids_manifest_sha256(
            quality_ordered, maximum_tokens=protocol.model.max_context_tokens
        ),
        generation_sample_id_list_sha256=_canonical_json_sha256(
            [sample.sample_id for sample in generation_ordered]
        ),
        generation_ids_content_sha256=_canonical_json_sha256(
            _content_identity_records(generation_ordered)
        ),
        generation_manifest_sha256=_canonical_json_sha256(
            [sample.manifest_record() for sample in generation_ordered]
        ),
        routing_input_token_ids_manifest_sha256=_token_ids_manifest_sha256(
            generation_ordered, maximum_tokens=protocol.model.max_context_tokens
        ),
        generation_prompt_token_ids_manifest_sha256=_token_ids_manifest_sha256(
            generation_ordered,
            maximum_tokens=(
                protocol.model.max_context_tokens
                - protocol.execution_contract.generation_decode.max_new_tokens
            ),
        ),
        strata=tuple(strata_records),
    )
    validate_confirmatory_selection(selection, protocol)
    return selection


def validate_confirmatory_selection(
    selection: ConfirmatorySelection, protocol: ConfirmatoryProtocol
) -> None:
    """Fail closed unless every re-derived count and digest matches the checked YAML."""

    holdout = protocol.holdout_selection
    routing = protocol.generation_and_routing
    observed = {
        "holdout_selection.excluded_selection_sha256": selection.excluded_selection_sha256,
        "holdout_selection.eligible_evaluated_token_count": (
            selection.eligible_evaluated_token_count
        ),
        "holdout_selection.quality_sample_id_list_sha256": (
            selection.quality_sample_id_list_sha256
        ),
        "holdout_selection.quality_ids_content_sha256": selection.quality_ids_content_sha256,
        "holdout_selection.quality_manifest_sha256": selection.quality_manifest_sha256,
        "holdout_selection.quality_input_token_ids_manifest_sha256": (
            selection.quality_input_token_ids_manifest_sha256
        ),
        "holdout_selection.quality_evaluated_token_count": sum(
            sample.evaluated_token_count for sample in selection.quality
        ),
        "holdout_selection.quality_source_token_count": sum(
            sample.source_token_count for sample in selection.quality
        ),
        "holdout_selection.quality_truncated_sample_count": sum(
            sample.truncated for sample in selection.quality
        ),
        "holdout_selection.strata": list(selection.strata),
        "generation_and_routing.sample_id_list_sha256": (
            selection.generation_sample_id_list_sha256
        ),
        "generation_and_routing.ids_content_sha256": (selection.generation_ids_content_sha256),
        "generation_and_routing.manifest_sha256": selection.generation_manifest_sha256,
        "generation_and_routing.routing_input_token_ids_manifest_sha256": (
            selection.routing_input_token_ids_manifest_sha256
        ),
        "generation_and_routing.generation_prompt_token_ids_manifest_sha256": (
            selection.generation_prompt_token_ids_manifest_sha256
        ),
        "generation_and_routing.evaluated_token_count": sum(
            sample.evaluated_token_count for sample in selection.generation_and_routing
        ),
        "generation_and_routing.routed_input_token_count": sum(
            len(sample.token_ids) for sample in selection.generation_and_routing
        ),
    }
    expected = {
        "holdout_selection.excluded_selection_sha256": holdout.excluded_selection_sha256,
        "holdout_selection.eligible_evaluated_token_count": (
            holdout.eligible_evaluated_token_count
        ),
        "holdout_selection.quality_sample_id_list_sha256": (holdout.quality_sample_id_list_sha256),
        "holdout_selection.quality_ids_content_sha256": holdout.quality_ids_content_sha256,
        "holdout_selection.quality_manifest_sha256": holdout.quality_manifest_sha256,
        "holdout_selection.quality_input_token_ids_manifest_sha256": (
            holdout.quality_input_token_ids_manifest_sha256
        ),
        "holdout_selection.quality_evaluated_token_count": (holdout.quality_evaluated_token_count),
        "holdout_selection.quality_source_token_count": holdout.quality_source_token_count,
        "holdout_selection.quality_truncated_sample_count": (
            holdout.quality_truncated_sample_count
        ),
        "holdout_selection.strata": [item.model_dump(mode="json") for item in holdout.strata],
        "generation_and_routing.sample_id_list_sha256": routing.sample_id_list_sha256,
        "generation_and_routing.ids_content_sha256": routing.ids_content_sha256,
        "generation_and_routing.manifest_sha256": routing.manifest_sha256,
        "generation_and_routing.routing_input_token_ids_manifest_sha256": (
            routing.routing_input_token_ids_manifest_sha256
        ),
        "generation_and_routing.generation_prompt_token_ids_manifest_sha256": (
            routing.generation_prompt_token_ids_manifest_sha256
        ),
        "generation_and_routing.evaluated_token_count": routing.evaluated_token_count,
        "generation_and_routing.routed_input_token_count": routing.routed_input_token_count,
    }
    for name, actual in observed.items():
        _assert_protocol_value(name, actual, expected[name])


def prepare_confirmatory_data(
    config: ExperimentConfig,
    dataset_path: Path,
    *,
    expected_dataset_sha256: str,
    tokenizer: Any,
    protocol: ConfirmatoryProtocol,
) -> PreparedPublicData:
    """Verify the file, re-derive selection, and require checked execution order."""

    actual_sha256 = file_sha256(dataset_path)
    _assert_protocol_value("dataset.file_sha256", actual_sha256, expected_dataset_sha256)
    _assert_protocol_value(
        "protocol.dataset.file_sha256", actual_sha256, protocol.dataset.file_sha256
    )
    stories = split_stories(dataset_path.read_text(encoding="utf-8"))
    selection = derive_confirmatory_selection(stories, tokenizer, protocol)
    quality_suite = require_suite(config, "perplexity")
    generation_suite = require_suite(config, "generation_regression")
    quality_ids = tuple(sample.sample_id for sample in selection.quality)
    generation_ids = tuple(sample.sample_id for sample in selection.generation_and_routing)
    _assert_protocol_value("baseline quality sample IDs", quality_suite.sample_ids, quality_ids)
    _assert_protocol_value(
        "baseline generation/routing sample IDs", generation_suite.sample_ids, generation_ids
    )
    if quality_ids != tuple(sorted(quality_ids)) or generation_ids != tuple(sorted(generation_ids)):
        raise ValueError("confirmatory execution order must be ascending stable sample ID")
    return PreparedPublicData(
        dataset_sha256=actual_sha256,
        story_count=len(stories),
        perplexity_ids=quality_ids,
        perplexity_stories=tuple(sample.story for sample in selection.quality),
        perplexity_selection_sha256=selection.quality_ids_content_sha256,
        generation_ids=generation_ids,
        generation_stories=tuple(sample.story for sample in selection.generation_and_routing),
        generation_selection_sha256=selection.generation_ids_content_sha256,
        confirmatory_selection=selection,
    )


def _confirmatory_selection_record(selection: ConfirmatorySelection) -> dict[str, Any]:
    """Return only hashes, IDs, counts, and strata—not source text or token IDs."""

    return {
        "quality_sample_ids": [sample.sample_id for sample in selection.quality],
        "generation_and_routing_sample_ids": [
            sample.sample_id for sample in selection.generation_and_routing
        ],
        "excluded_selection_sha256": selection.excluded_selection_sha256,
        "eligible_evaluated_token_count": selection.eligible_evaluated_token_count,
        "quality_sample_id_list_sha256": selection.quality_sample_id_list_sha256,
        "quality_ids_content_sha256": selection.quality_ids_content_sha256,
        "quality_manifest_sha256": selection.quality_manifest_sha256,
        "quality_input_token_ids_manifest_sha256": (
            selection.quality_input_token_ids_manifest_sha256
        ),
        "generation_sample_id_list_sha256": selection.generation_sample_id_list_sha256,
        "generation_ids_content_sha256": selection.generation_ids_content_sha256,
        "generation_manifest_sha256": selection.generation_manifest_sha256,
        "routing_input_token_ids_manifest_sha256": (
            selection.routing_input_token_ids_manifest_sha256
        ),
        "generation_prompt_token_ids_manifest_sha256": (
            selection.generation_prompt_token_ids_manifest_sha256
        ),
        "quality_evaluated_token_count": sum(
            sample.evaluated_token_count for sample in selection.quality
        ),
        "quality_source_token_count": sum(
            sample.source_token_count for sample in selection.quality
        ),
        "quality_truncated_sample_count": sum(sample.truncated for sample in selection.quality),
        "generation_and_routing_evaluated_token_count": sum(
            sample.evaluated_token_count for sample in selection.generation_and_routing
        ),
        "generation_and_routing_input_token_count": sum(
            len(sample.token_ids) for sample in selection.generation_and_routing
        ),
        "expected_layer_token_event_count": _CONFIRMATORY_EXPECTED_ROUTING_EVENTS,
        "strata": list(selection.strata),
        "source_text_or_token_ids_persisted": False,
    }


def _confirmatory_status_record(
    protocol: ConfirmatoryProtocol,
    noninferiority_result: dict[str, object],
    execution_ordinal: Literal[1, 2],
) -> dict[str, object]:
    """Keep one execution provisional even when its within-run gate passes."""

    passed = noninferiority_result.get("passed")
    if not isinstance(passed, bool):
        raise ValueError("within-execution noninferiority result is missing a boolean decision")
    return {
        "status": "provisional_until_2_clean_executions",
        "confirmatory_claim_ready": False,
        "required_clean_process_executions": (
            protocol.repeatability.required_clean_process_executions
        ),
        "execution_ordinal": execution_ordinal,
        "within_execution_noninferiority_passed": passed,
        "within_execution_decision_is_overall_confirmatory_claim": False,
        "pair_verification_required": True,
    }


def resolve_quality_output_path(
    config: ExperimentConfig,
    *,
    project_root: Path,
    requested: Path,
) -> Path:
    """Resolve a project-relative record path below the configured artifact root."""

    project = project_root.expanduser().resolve()
    configured = Path(config.output.root).expanduser()
    artifact_root = (configured if configured.is_absolute() else project / configured).resolve(
        strict=False
    )
    if artifact_root == project or not artifact_root.is_relative_to(project):
        raise ArtifactIntegrityError(
            "Configured artifact root must resolve to a directory below the project root",
            component="public_native_quality",
        )
    if requested.is_absolute() or any(part == ".." for part in requested.parts):
        raise ArtifactIntegrityError(
            f"Quality output must be project-relative and cannot contain '..': {requested}",
            component="public_native_quality",
        )
    candidate = (project / requested).resolve(strict=False)
    if candidate == artifact_root or not candidate.is_relative_to(artifact_root):
        raise ArtifactIntegrityError(
            f"Quality output resolves outside the configured artifact root: {requested}",
            component="public_native_quality",
        )
    return safe_path(artifact_root, candidate.relative_to(artifact_root))


def split_stories(text: str) -> tuple[str, ...]:
    """Split the official TinyStories text format into non-empty segments."""

    return tuple(story.strip() for story in text.split(_STORY_SEPARATOR) if story.strip())


def select_stories(stories: tuple[str, ...], sample_ids: tuple[str, ...]) -> tuple[str, ...]:
    """Select exact stable story IDs in their declared order."""

    indexes = tuple(_sample_index(sample_id) for sample_id in sample_ids)
    if len(set(indexes)) != len(indexes):
        raise ValueError("TinyStories sample IDs must be unique")
    if not indexes or max(indexes) >= len(stories):
        raise ValueError("TinyStories sample selection is empty or outside the dataset file")
    return tuple(stories[index] for index in indexes)


def _sample_index(sample_id: str) -> int:
    if not sample_id.startswith(_SAMPLE_PREFIX):
        raise ValueError(f"unsupported TinyStories sample ID: {sample_id}")
    suffix = sample_id.removeprefix(_SAMPLE_PREFIX)
    if len(suffix) != 6 or not suffix.isdigit():
        raise ValueError(f"unsupported TinyStories sample ID: {sample_id}")
    return int(suffix)


def _selection_sha256(sample_ids: tuple[str, ...], stories: tuple[str, ...]) -> str:
    records = [
        {
            "sample_id": sample_id,
            "content_sha256": hashlib.sha256(story.encode("utf-8")).hexdigest(),
        }
        for sample_id, story in zip(sample_ids, stories, strict=True)
    ]
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_suite(config: ExperimentConfig, suite_type: str) -> EvaluationSuiteConfig:
    """Return the unique configured suite of one type."""

    suites = tuple(suite for suite in config.evaluation.suites if suite.type == suite_type)
    if len(suites) != 1:
        raise ValueError(f"quality config requires exactly one {suite_type} suite")
    return suites[0]


def validate_experiment_contracts(
    baseline: ExperimentConfig, candidates: tuple[ExperimentConfig, ...]
) -> None:
    """Require one identical scientific contract and only declared quantizer changes."""

    if baseline.quantization.backend != "noop":
        raise ValueError("baseline quality config must use the noop backend")
    if not candidates:
        raise ValueError("at least one native quantized candidate config is required")
    expected_backends = {
        "torch_native_dynamic_int8",
        "torch_native_int4_kleidiai",
    }
    candidate_backends = {candidate.quantization.backend for candidate in candidates}
    if not candidate_backends.issubset(expected_backends):
        raise ValueError("candidate configs must use only native dynamic INT8 or KleidiAI INT4")
    if len(candidate_backends) != len(candidates):
        raise ValueError("candidate backend configs must be unique")

    baseline_contract = _scientific_contract(baseline)
    for candidate in candidates:
        if _scientific_contract(candidate) != baseline_contract:
            raise ValueError(
                f"{candidate.name} changes the model, data, seed, runtime, routing, or privacy "
                "contract relative to the baseline"
            )
        if candidate.quantization.calibration is not None:
            raise ValueError("native dynamic CPU candidates must not declare calibration data")

    if baseline.model.model_id != _MODEL_ID or baseline.model.revision != _MODEL_REVISION:
        raise ValueError("quality experiment must use the checked pinned stories15M_MOE revision")
    if baseline.model.trust_remote_code or baseline.security.allow_remote_code:
        raise ValueError("public quality experiment must keep remote code disabled")
    if baseline.security.log_prompts or baseline.security.log_model_outputs:
        raise ValueError("public quality experiment must not persist prompts or model outputs")
    if (
        baseline.runtime.backend != "torch_eager_cpu"
        or baseline.runtime.device != "cpu"
        or baseline.runtime.dtype != "float32"
        or baseline.runtime.device_map != "single"
        or baseline.runtime.sharding is not None
    ):
        raise ValueError("public native quality evidence requires unsharded float32 eager CPU")
    if baseline.routing.mode != "full_trace" or not baseline.routing.required:
        raise ValueError("public native quality evidence requires aligned full routing traces")
    if baseline.routing.capture_router_logits:
        raise ValueError("router logits are unnecessary and must not be retained for this evidence")

    perplexity = require_suite(baseline, "perplexity")
    generation = require_suite(baseline, "generation_regression")
    if not perplexity.sample_ids or not generation.sample_ids:
        raise ValueError("quality and generation suites require declared stable sample IDs")
    if not set(generation.sample_ids).issubset(perplexity.sample_ids):
        raise ValueError("generation regression samples must be a subset of quality samples")
    if generation.decode.do_sample:
        raise ValueError("generation retention evidence must use deterministic decoding")
    if _suite_dataset_identity(perplexity) != _suite_dataset_identity(generation):
        raise ValueError("quality and generation suites must use the same pinned dataset")
    if perplexity.prompt_template != "{text}" or generation.prompt_template != "{text}":
        raise ValueError(
            "raw-story loss, generation, and routing require the exact '{text}' prompt template"
        )
    if (
        baseline.routing.dataset,
        baseline.routing.dataset_revision,
        baseline.routing.dataset_split,
    ) != _suite_dataset_identity(perplexity):
        raise ValueError("routing identity must match the exact quality dataset identity")


def validate_confirmatory_experiment_contracts(
    baseline: ExperimentConfig,
    candidates: tuple[ExperimentConfig, ...],
    protocol: ConfirmatoryProtocol,
) -> None:
    """Require the one checked baseline/INT8 design before loading or forwarding."""

    validate_experiment_contracts(baseline, candidates)
    if len(candidates) != 1 or candidates[0].quantization.backend != "torch_native_dynamic_int8":
        raise ValueError(
            "confirmatory quality protocol requires exactly one native dynamic INT8 candidate"
        )
    if baseline.seed != protocol.holdout_selection.seed:
        raise ValueError("confirmatory config seed does not match the prospective protocol")
    if baseline.evaluation.allow_partial or candidates[0].evaluation.allow_partial:
        raise ValueError("confirmatory quality protocol forbids partial evaluation")
    if (
        baseline.model.model_id != protocol.model.model_id
        or baseline.model.revision != protocol.model.revision
    ):
        raise ValueError("confirmatory config model identity does not match the protocol")
    if not baseline.model.local_files_only or not candidates[0].model.local_files_only:
        raise ValueError("confirmatory quality protocol requires offline pinned model loading")
    quality = require_suite(baseline, "perplexity")
    generation = require_suite(baseline, "generation_regression")
    identity = (protocol.dataset.dataset_id, protocol.dataset.revision, protocol.dataset.split)
    if (
        _suite_dataset_identity(quality) != identity
        or _suite_dataset_identity(generation) != identity
    ):
        raise ValueError("confirmatory config dataset identity does not match the protocol")
    if len(quality.sample_ids) != 256 or len(set(quality.sample_ids)) != 256:
        raise ValueError("confirmatory quality config must declare 256 unique sample IDs")
    if len(generation.sample_ids) != 16 or len(set(generation.sample_ids)) != 16:
        raise ValueError("confirmatory generation/routing config must declare 16 unique sample IDs")
    if quality.sample_ids != tuple(sorted(quality.sample_ids)) or generation.sample_ids != tuple(
        sorted(generation.sample_ids)
    ):
        raise ValueError("confirmatory config sample execution order must be ascending ID")
    excluded = {f"story-{index:06d}" for index in range(1, 33)}
    if excluded.intersection(quality.sample_ids):
        raise ValueError("confirmatory holdout must exclude all prior story-000001..000032 IDs")
    if not set(generation.sample_ids).issubset(quality.sample_ids):
        raise ValueError("confirmatory descriptive subset must be contained in the holdout")
    contract = protocol.execution_contract
    if baseline.config_hash() != contract.baseline_resolved_config_sha256:
        raise ValueError("confirmatory baseline resolved config hash does not match the protocol")
    candidate = candidates[0]
    if candidate.config_hash() != contract.candidate_resolved_config_sha256:
        raise ValueError("confirmatory candidate resolved config hash does not match the protocol")
    parameters = candidate.quantization.parameters
    quantization_contract = {
        "backend": candidate.quantization.backend,
        "method": candidate.quantization.method,
        "quantized_engine": parameters.get("quantized_engine"),
        "activation_granularity": parameters.get("activation_granularity"),
        "weight_granularity": parameters.get("weight_granularity"),
    }
    expected_quantization_contract = {
        "backend": contract.candidate_backend,
        "method": contract.candidate_method,
        "quantized_engine": contract.quantized_engine,
        "activation_granularity": contract.activation_granularity,
        "weight_granularity": contract.weight_granularity,
    }
    if quantization_contract != expected_quantization_contract:
        raise ValueError("confirmatory candidate quantization contract does not match the protocol")
    if generation.decode.model_dump(mode="json") != contract.generation_decode.model_dump(
        mode="json"
    ):
        raise ValueError("confirmatory generation decode does not match the protocol")
    runtime_contract = {
        "backend": baseline.runtime.backend,
        "device": baseline.runtime.device,
        "dtype": baseline.runtime.dtype,
        "device_map": baseline.runtime.device_map,
    }
    if runtime_contract != {
        "backend": contract.runtime_backend,
        "device": contract.runtime_device,
        "dtype": contract.runtime_dtype,
        "device_map": contract.runtime_device_map,
    }:
        raise ValueError("confirmatory runtime contract does not match the protocol")


def prepare_public_data(
    config: ExperimentConfig,
    dataset_path: Path,
    *,
    expected_dataset_sha256: str,
) -> PreparedPublicData:
    """Verify the full dataset file and select both configured sample sets."""

    actual_sha256 = file_sha256(dataset_path)
    if actual_sha256 != expected_dataset_sha256:
        raise ValueError(
            "TinyStories file checksum mismatch: "
            f"expected {expected_dataset_sha256}, received {actual_sha256}"
        )
    stories = split_stories(dataset_path.read_text(encoding="utf-8"))
    perplexity = require_suite(config, "perplexity")
    generation = require_suite(config, "generation_regression")
    perplexity_stories = select_stories(stories, perplexity.sample_ids)
    generation_stories = select_stories(stories, generation.sample_ids)
    return PreparedPublicData(
        dataset_sha256=actual_sha256,
        story_count=len(stories),
        perplexity_ids=perplexity.sample_ids,
        perplexity_stories=perplexity_stories,
        perplexity_selection_sha256=_selection_sha256(perplexity.sample_ids, perplexity_stories),
        generation_ids=generation.sample_ids,
        generation_stories=generation_stories,
        generation_selection_sha256=_selection_sha256(generation.sample_ids, generation_stories),
    )


def validate_native_policy(
    root: nn.Module,
    inventory: tuple[ModuleInfo, ...],
    policy: ResolvedPrecisionPolicy,
    *,
    target_precision: Literal["int8", "int4"],
) -> tuple[str, ...]:
    """Prove that only concrete, unprotected ``nn.Linear`` leaves were selected."""

    named_modules = dict(root.named_modules())
    inventory_by_name = {module.name: module for module in inventory}
    for module in inventory:
        concrete = named_modules.get(module.name)
        if isinstance(concrete, nn.Linear):
            if not {"float32", "int8", "int4"}.issubset(module.supported_precisions):
                raise ValueError(f"addressable linear {module.name} lacks native precision support")
        elif module.supported_precisions != ("float32",):
            raise ValueError(f"nonlinear inventory entry {module.name} advertises quantization")

    selected = tuple(
        sorted(
            name
            for name, precision in policy.precision_map.items()
            if precision == target_precision
        )
    )
    if not selected:
        raise ValueError(f"native {target_precision} policy selected no modules")
    expected = tuple(
        sorted(
            module.name
            for module in inventory
            if isinstance(named_modules.get(module.name), nn.Linear)
            and not module.is_router
            and not module.is_output_head
        )
    )
    if selected != expected:
        raise ValueError(
            f"native {target_precision} policy must select exactly unprotected concrete linears: "
            f"expected {expected}, received {selected}"
        )
    for name, precision in policy.precision_map.items():
        module = inventory_by_name[name]
        if module.class_name.endswith(".ExpertSlice") and precision != "float32":
            raise ValueError(f"fused expert slice {name} must remain float32")
        if name not in selected and precision != "float32":
            raise ValueError(f"non-selected module {name} must remain float32")
    return selected


def generation_retention(
    baseline: dict[str, tuple[int, ...]], candidate: dict[str, tuple[int, ...]]
) -> dict[str, Any]:
    """Compare deterministic generations without returning output token content."""

    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate generation sample IDs are not aligned")
    samples = []
    exact = 0
    for sample_id in baseline:
        baseline_tokens = baseline[sample_id]
        candidate_tokens = candidate[sample_id]
        matches = baseline_tokens == candidate_tokens
        exact += int(matches)
        samples.append(
            {
                "sample_id": sample_id,
                "exact_match": matches,
                "baseline_generated_token_count": len(baseline_tokens),
                "candidate_generated_token_count": len(candidate_tokens),
                "baseline_output_sha256": _token_ids_sha256(baseline_tokens),
                "candidate_output_sha256": _token_ids_sha256(candidate_tokens),
            }
        )
    return {
        "sample_count": len(samples),
        "exact_match_count": exact,
        "exact_match_rate": exact / len(samples),
        "samples": samples,
        "output_content_persisted": False,
    }


def evaluate(
    baseline_config: ExperimentConfig,
    candidate_configs: tuple[ExperimentConfig, ...],
    dataset_path: Path,
    *,
    expected_dataset_sha256: str,
    project_root: Path,
    hardware_label: str,
    confirmatory_protocol: ConfirmatoryProtocol | None = None,
    execution_ordinal: Literal[1, 2] | None = None,
) -> dict[str, Any]:
    """Run the baseline and every native candidate under one aligned contract."""

    expected_dataset_sha256 = require_official_dataset_sha256(expected_dataset_sha256)
    hardware_label = require_hardware_label(hardware_label)
    if confirmatory_protocol is None:
        if execution_ordinal is not None:
            raise ValueError("execution ordinal is valid only for confirmatory protocol records")
        validate_experiment_contracts(baseline_config, candidate_configs)
        data: PreparedPublicData | None = prepare_public_data(
            baseline_config,
            dataset_path,
            expected_dataset_sha256=expected_dataset_sha256,
        )
    else:
        if execution_ordinal not in (1, 2):
            raise ValueError("confirmatory protocol records require execution ordinal 1 or 2")
        validate_confirmatory_experiment_contracts(
            baseline_config, candidate_configs, confirmatory_protocol
        )
        actual_dataset_sha256 = file_sha256(dataset_path)
        _assert_protocol_value(
            "dataset.file_sha256", actual_dataset_sha256, confirmatory_protocol.dataset.file_sha256
        )
        initial_stories = split_stories(dataset_path.read_text(encoding="utf-8"))
        _assert_protocol_value(
            "dataset.story_count", len(initial_stories), confirmatory_protocol.dataset.story_count
        )
        data = None
    source_weight = _verified_cached_source_weight(baseline_config)
    source_metadata_files: dict[str, str] | None = None
    if confirmatory_protocol is not None:
        _assert_protocol_value(
            "model.source_weight_file",
            source_weight["file"],
            confirmatory_protocol.model.source_weight_file,
        )
        _assert_protocol_value(
            "model.source_weight_size_bytes",
            source_weight["size_bytes"],
            confirmatory_protocol.model.source_weight_size_bytes,
        )
        _assert_protocol_value(
            "model.source_weight_sha256",
            source_weight["sha256"],
            confirmatory_protocol.model.source_weight_sha256,
        )
        # Resolve and hash every architecture/tokenizer input before model construction.
        source_metadata_files = _verified_cached_tokenizer_files(baseline_config)
        expected_metadata_files = {
            "config.json": confirmatory_protocol.model.config_json_sha256,
            "generation_config.json": (confirmatory_protocol.model.generation_config_json_sha256),
            "special_tokens_map.json": (
                confirmatory_protocol.tokenizer.special_tokens_map_json_sha256
            ),
            "tokenizer.json": confirmatory_protocol.tokenizer.tokenizer_json_sha256,
            "tokenizer_config.json": (confirmatory_protocol.tokenizer.tokenizer_config_json_sha256),
        }
        _assert_protocol_value(
            "model/tokenizer source metadata files",
            source_metadata_files,
            expected_metadata_files,
        )
    runtime = TorchEagerCPURuntime()
    runtime_capability = runtime.probe(baseline_config.runtime)
    adapter = HFCausalLMAdapter()
    torch.manual_seed(baseline_config.seed)
    total_started = time.perf_counter()
    baseline_loaded = adapter.load(baseline_config, runtime)
    if confirmatory_protocol is not None:
        _assert_protocol_value(
            "loaded model maximum context",
            baseline_loaded.descriptor.capabilities.max_context_length,
            confirmatory_protocol.model.max_context_tokens,
        )
        data = prepare_confirmatory_data(
            baseline_config,
            dataset_path,
            expected_dataset_sha256=expected_dataset_sha256,
            tokenizer=baseline_loaded.tokenizer,
            protocol=confirmatory_protocol,
        )
    if data is None:  # Defensive: every branch above must prepare data before a forward.
        raise RuntimeError("quality data preparation did not complete")
    inventory = adapter.enumerate_modules(baseline_loaded)
    baseline_storage_bytes = model_storage_bytes(cast(nn.Module, baseline_loaded.model))
    baseline_measurement = _measure_loaded_model(
        baseline_config,
        data,
        adapter,
        baseline_loaded,
        confirmatory_protocol=confirmatory_protocol,
    )
    baseline_alignment_sha256 = _routing_alignment_sha256(baseline_measurement.routing)

    candidates: list[dict[str, Any]] = []
    try:
        for config in candidate_configs:
            record = _measure_candidate(
                config=config,
                data=data,
                adapter=adapter,
                runtime=runtime,
                runtime_capability=runtime_capability,
                baseline=baseline_loaded,
                inventory=inventory,
                baseline_storage_bytes=baseline_storage_bytes,
                baseline_measurement=baseline_measurement,
                baseline_alignment_sha256=baseline_alignment_sha256,
                confirmatory_protocol=confirmatory_protocol,
            )
            candidates.append(record)
            gc.collect()
    finally:
        runtime.cleanup()

    perplexity_suite = require_suite(baseline_config, "perplexity")
    generation_suite = require_suite(baseline_config, "generation_regression")
    record = {
        "schema_version": (
            "public-moe-native-linear-quality-v1"
            if confirmatory_protocol is None
            else "public-moe-native-linear-quality-v2"
        ),
        "created_at_utc": _utc_now(),
        "baseline_config": baseline_config.name,
        "baseline_config_hash": baseline_config.config_hash(),
        "candidate_configs": [
            {"name": config.name, "config_hash": config.config_hash()}
            for config in candidate_configs
        ],
        "model": {
            "model_id": baseline_loaded.descriptor.model_id,
            "revision": baseline_loaded.descriptor.revision,
            "resolved_class": baseline_loaded.descriptor.resolved_class,
            "architecture": baseline_loaded.descriptor.architecture,
            "loaded_float32_state_sha256": baseline_loaded.descriptor.checksum,
            "source_weight_file": source_weight["file"],
            "source_weight_sha256": source_weight["sha256"],
            "source_weight_size_bytes": source_weight["size_bytes"],
            "license": _MODEL_LICENSE,
            "expert_weight_provenance": "repeated_trained_TinyStories_weights",
            "router_provenance": "randomly_initialized_by_model_publisher",
        },
        "dataset": {
            "dataset_id": perplexity_suite.dataset,
            "revision": perplexity_suite.revision,
            "split": perplexity_suite.split,
            "file": dataset_path.name,
            "file_sha256": data.dataset_sha256,
            "full_file_story_count": data.story_count,
            "license": _DATASET_LICENSE,
            "perplexity_sample_ids": list(data.perplexity_ids),
            "perplexity_selection_sha256": data.perplexity_selection_sha256,
            "generation_sample_ids": list(data.generation_ids),
            "generation_selection_sha256": data.generation_selection_sha256,
        },
        "protocol": {
            "quality_metric": "token_weighted_causal_nll_and_perplexity",
            "loss_special_tokens": True,
            "loss_truncation": "right_to_model_max_context",
            "max_context_tokens": baseline_loaded.descriptor.capabilities.max_context_length,
            "generation_decode": generation_suite.decode.model_dump(mode="json"),
            "generation_input_truncation": "right_to_max_context_minus_max_new_tokens",
            "generation_reference": "same_loaded_float32_baseline",
            "routing_capture": "full_trace_during_the_exact_32_loss_forwards",
            "routing_comparison": "stable_sample_id_token_position_and_layer",
            "seed": baseline_config.seed,
            "prompt_or_output_content_persisted": False,
            "quantization_scope": (
                "concrete_unprotected_nn.Linear_leaves_only; fused_experts_float32"
            ),
        },
        "inventory": _inventory_summary(inventory),
        "baseline": {
            "runtime_tensor_storage_bytes": baseline_storage_bytes,
            "load_time_seconds": baseline_loaded.load_time_seconds,
            "measurement_elapsed_seconds": baseline_measurement.elapsed_seconds,
            "quality": baseline_measurement.quality,
            "generation": baseline_measurement.generation,
            "routing": {
                "observed_event_count": baseline_measurement.routing.observed_event_count,
                "recorded_event_count": baseline_measurement.routing.recorded_event_count,
                "batch_count": baseline_measurement.routing.batch_count,
                "alignment_sha256": baseline_alignment_sha256,
            },
            **(
                {}
                if baseline_measurement.routing_observer_proof is None
                else {"routing_observer_proof": baseline_measurement.routing_observer_proof}
            ),
        },
        "candidates": candidates,
        "environment": {
            **probe_environment(project_root),
            "operator_declared_hardware_label": hardware_label,
            "operator_hardware_label_scope": (
                "explicit CLI metadata; CPU workload host, not inferred from sandbox probes"
            ),
            "runtime_capability": asdict(runtime_capability),
            "software": _software_versions(),
            "torch_cpu_execution": {
                "intraop_threads": torch.get_num_threads(),
                "interop_threads": torch.get_num_interop_threads(),
                "quantized_engine": torch.backends.quantized.engine,
                "supported_quantized_engines": list(torch.backends.quantized.supported_engines),
                "kleidiai_available": bool(cast(Any, torch.backends.kleidiai).is_available()),
            },
        },
        "elapsed_seconds_including_model_load_and_all_candidates": (
            time.perf_counter() - total_started
        ),
        "limitations": [
            (
                "This is one deterministic 32-story subset and one hardware/software environment."
                if confirmatory_protocol is None
                else "This is one prospectively selected 256-story holdout and one "
                "hardware/software environment."
            ),
            "Only attention/ordinary concrete linear leaves are quantized; fused MoE expert "
            "tensors, routers, embeddings, normalization, and the output head remain float32.",
            "The model publisher states that expert weights repeat one trained TinyStories model.",
            "The model publisher states that routers were randomly initialized, so routing drift "
            "does not measure preservation of learned expert specialization.",
            (
                "Greedy generation exact match on four declared prompts is a regression signal, "
                "not a general generation-quality score."
                if confirmatory_protocol is None
                else "Greedy generation and routing on the declared 16-story stratified subset "
                "are descriptive regression signals, not inferential quality endpoints."
            ),
            "The candidate memory values are tensor-state storage, not measured process peak RSS.",
        ],
    }
    if confirmatory_protocol is not None:
        if data.confirmatory_selection is None or len(candidates) != 1 or execution_ordinal is None:
            raise RuntimeError("confirmatory quality record is missing its complete paired sample")
        if source_metadata_files is None:
            raise RuntimeError("confirmatory quality record is missing verified source metadata")
        noninferiority_result = candidates[0]["noninferiority"]
        record["model"]["verified_source_metadata_file_sha256"] = source_metadata_files
        del record["protocol"]["prompt_or_output_content_persisted"]
        record["protocol"]["raw_prompt_text_or_output_token_ids_persisted"] = False
        record["protocol"]["public_sample_ids_persisted"] = True
        record["protocol"]["generated_output_sha256_fingerprints_persisted"] = True
        record["protocol"]["routing_capture"] = (
            "all_256_quality_loss_forwards_hook_free_then_full_trace_on_exact_16_sample_reforward"
        )
        record["protocol"]["generation_and_routing_inference_claim"] = "descriptive_only"
        record["protocol"]["confirmatory"] = {
            "definition": confirmatory_protocol.model_dump(mode="json"),
            "definition_sha256": _canonical_json_sha256(
                confirmatory_protocol.model_dump(mode="json")
            ),
            "selection": _confirmatory_selection_record(data.confirmatory_selection),
            "noninferiority_result": noninferiority_result,
            "noninferiority_result_scope": (
                "within_execution_statistical_gate_only_not_overall_confirmatory_claim"
            ),
            "upper_bound_label": ("nominal_one_sided_95_percent_bootstrap_upper_bound"),
            "coverage_interpretation": (
                "nominal_approximate_coverage_under_the_predeclared_stratified_"
                "finite_population_bootstrap_design"
            ),
        }
        record["confirmatory_status"] = _confirmatory_status_record(
            confirmatory_protocol, noninferiority_result, execution_ordinal
        )
    return record


def _measure_candidate(
    *,
    config: ExperimentConfig,
    data: PreparedPublicData,
    adapter: HFCausalLMAdapter,
    runtime: TorchEagerCPURuntime,
    runtime_capability: RuntimeCapabilities,
    baseline: LoadedModel,
    inventory: tuple[ModuleInfo, ...],
    baseline_storage_bytes: int,
    baseline_measurement: ModelMeasurement,
    baseline_alignment_sha256: str,
    confirmatory_protocol: ConfirmatoryProtocol | None = None,
) -> dict[str, Any]:
    backend = cast(BackendName, config.quantization.backend)
    if backend == "torch_native_dynamic_int8":
        quantizer: TorchNativeDynamicInt8Quantizer | TorchNativeInt4KleidiAIQuantizer = (
            TorchNativeDynamicInt8Quantizer()
        )
        precision: Literal["int8", "int4"] = "int8"
        wrapper: type[nn.Module] = NativeDynamicInt8Linear
    else:
        quantizer = TorchNativeInt4KleidiAIQuantizer()
        precision = "int4"
        wrapper = NativeInt4KleidiAILinear
    support = quantizer.check_support(baseline.descriptor, runtime_capability, config.quantization)
    if not support.supported:
        raise RuntimeError(f"{backend} is unsupported on this host: {support.message()}")
    policy = resolve_precision_policy(inventory, config.quantization.policy)
    selected = validate_native_policy(
        cast(nn.Module, baseline.model), inventory, policy, target_precision=precision
    )
    torch.manual_seed(config.seed)
    quantize_started = time.perf_counter()
    quantized = quantizer.quantize(baseline, policy, None, config.quantization)
    quantize_elapsed = time.perf_counter() - quantize_started
    try:
        if (
            confirmatory_protocol is not None
            and torch.backends.quantized.engine
            != confirmatory_protocol.execution_contract.quantized_engine
        ):
            raise RuntimeError(
                "native INT8 execution did not activate the predeclared qnnpack engine"
            )
        candidate_modules = dict(cast(nn.Module, quantized.loaded.model).named_modules())
        if any(not isinstance(candidate_modules.get(name), wrapper) for name in selected):
            raise RuntimeError(
                f"{backend} did not install its native wrapper at every selected leaf"
            )
        measurement = _measure_loaded_model(
            config,
            data,
            adapter,
            quantized.loaded,
            confirmatory_protocol=confirmatory_protocol,
        )
        alignment_sha256 = _routing_alignment_sha256(measurement.routing)
        if alignment_sha256 != baseline_alignment_sha256:
            raise RuntimeError(f"{backend} routing traces are not aligned with the baseline")
        routing = compare_routing(baseline_measurement.routing, measurement.routing)
        candidate_storage_bytes = model_storage_bytes(cast(nn.Module, quantized.loaded.model))
        baseline_perplexity = float(baseline_measurement.quality["perplexity"])
        candidate_perplexity = float(measurement.quality["perplexity"])
        record: dict[str, Any] = {
            "config_name": config.name,
            "config_hash": config.config_hash(),
            "backend": backend,
            "method": config.quantization.method,
            "support": support.model_dump(mode="json"),
            "quantization": {
                "scope": "attention_and_other_unprotected_concrete_nn.Linear_leaves_only",
                "target_precision": precision,
                "quantized_module_count": len(selected),
                "quantized_modules": list(selected),
                "float32_module_count": len(policy.precision_map) - len(selected),
                "fused_expert_slices_quantized": 0,
                "fused_expert_slices_float32": sum(
                    module.class_name.endswith(".ExpertSlice") for module in inventory
                ),
                "quantization_elapsed_seconds": quantize_elapsed,
                "runtime_tensor_storage_bytes": candidate_storage_bytes,
                "runtime_tensor_storage_ratio_vs_float32": (
                    candidate_storage_bytes / baseline_storage_bytes
                ),
                "safe_bundle_serialized_size_bytes": quantized.manifest.serialized_size_bytes,
                "candidate_state_sha256": _model_state_sha256(
                    cast(nn.Module, quantized.loaded.model)
                ),
                "manifest": quantized.manifest.model_dump(mode="json"),
            },
            "quality": {
                **measurement.quality,
                "mean_nll_delta_vs_float32": (
                    float(measurement.quality["mean_nll"])
                    - float(baseline_measurement.quality["mean_nll"])
                ),
                "perplexity_delta_vs_float32": candidate_perplexity - baseline_perplexity,
                "perplexity_relative_change": (candidate_perplexity / baseline_perplexity - 1.0),
            },
            "generation_retention": generation_retention(
                baseline_measurement.generation_tokens, measurement.generation_tokens
            ),
            "routing": {
                "alignment_sha256": alignment_sha256,
                "observed_event_count": measurement.routing.observed_event_count,
                "recorded_event_count": measurement.routing.recorded_event_count,
                "comparison": routing.as_dict(),
            },
            "measurement_elapsed_seconds": measurement.elapsed_seconds,
        }
        if measurement.routing_observer_proof is not None:
            record["routing_observer_proof"] = measurement.routing_observer_proof
        if confirmatory_protocol is not None:
            record["noninferiority"] = _confirmatory_noninferiority(
                baseline_measurement.quality,
                measurement.quality,
                confirmatory_protocol,
            )
            record["noninferiority_decision_scope"] = (
                "within_execution_statistical_gate_only_not_overall_confirmatory_claim"
            )
            record["noninferiority_passed_is_overall_confirmatory_claim"] = False
            record["noninferiority_upper_bound_label"] = (
                "nominal_one_sided_95_percent_bootstrap_upper_bound"
            )
            record["noninferiority_coverage_interpretation"] = (
                "nominal_approximate_coverage_under_the_predeclared_stratified_"
                "finite_population_bootstrap_design"
            )
        return record
    finally:
        _release_quantized(quantized)


def _confirmatory_noninferiority(
    baseline_quality: dict[str, Any],
    candidate_quality: dict[str, Any],
    protocol: ConfirmatoryProtocol,
) -> dict[str, object]:
    """Evaluate the predeclared paired finite-population +0.5% PPL gate."""

    baseline_samples = baseline_quality.get("samples")
    candidate_samples = candidate_quality.get("samples")
    if not isinstance(baseline_samples, list) or not isinstance(candidate_samples, list):
        raise ValueError("confirmatory quality result is missing per-story observations")
    if len(baseline_samples) != 256 or len(candidate_samples) != 256:
        raise ValueError("confirmatory noninferiority requires all 256 paired story results")
    baseline_by_id = {str(item["sample_id"]): item for item in baseline_samples}
    candidate_by_id = {str(item["sample_id"]): item for item in candidate_samples}
    if len(baseline_by_id) != 256 or len(candidate_by_id) != 256:
        raise ValueError("confirmatory noninferiority rejects duplicate or missing sample IDs")
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("confirmatory baseline and candidate samples are not exactly aligned")

    observations: list[PairedNLLObservation] = []
    for sample_id in sorted(baseline_by_id):
        baseline = baseline_by_id[sample_id]
        candidate = candidate_by_id[sample_id]
        for field in ("content_sha256", "evaluated_token_count", "length_stratum"):
            if baseline[field] != candidate[field]:
                raise ValueError(
                    f"confirmatory paired sample {sample_id} differs in immutable {field}"
                )
        observations.append(
            PairedNLLObservation(
                sample_id=sample_id,
                content_sha256=str(baseline["content_sha256"]),
                stratum=str(baseline["length_stratum"]),
                token_count=int(baseline["evaluated_token_count"]),
                baseline_mean_nll=float(baseline["mean_nll"]),
                candidate_mean_nll=float(candidate["mean_nll"]),
            )
        )
    designs = tuple(
        StratumDesign(
            stratum=str(item.stratum),
            population_size=item.population_story_count,
            selected_size=item.selected_story_count,
        )
        for item in protocol.holdout_selection.strata
    )
    noninferiority = protocol.noninferiority
    result = paired_stratified_nll_noninferiority(
        observations,
        designs,
        exact_population_token_total=(protocol.holdout_selection.eligible_evaluated_token_count),
        margin_nll=noninferiority.margin_nats_per_token,
        confidence=noninferiority.confidence_level,
        bootstrap_replicates=noninferiority.bootstrap.replicates,
        seed=noninferiority.bootstrap.seed,
    )
    return result.as_dict()


def _measure_loaded_model(
    config: ExperimentConfig,
    data: PreparedPublicData,
    adapter: HFCausalLMAdapter,
    loaded: LoadedModel,
    *,
    confirmatory_protocol: ConfirmatoryProtocol | None = None,
) -> ModelMeasurement:
    if confirmatory_protocol is not None:
        return _measure_confirmatory_loaded_model(
            config, data, adapter, loaded, confirmatory_protocol
        )
    started = time.perf_counter()
    moe = adapter.discover_moe(loaded)
    if moe is None:
        raise ValueError("pinned public quality model did not expose validated MoE routing")
    sink = InMemoryRoutingSink(
        "full_trace",
        expert_counts={layer.layer_id: layer.expert_count for layer in moe.layers},
    )
    handle = adapter.attach_routing_hooks(loaded, sink, config)
    sample_records: list[dict[str, Any]] = []
    weighted_nll = 0.0
    evaluated_token_count = 0
    truncated_count = 0
    try:
        maximum = loaded.descriptor.capabilities.max_context_length
        for sample_id, story in zip(data.perplexity_ids, data.perplexity_stories, strict=True):
            encoded = loaded.tokenizer.encode(story, special_tokens=True)
            truncated = len(encoded) > maximum
            token_ids = encoded[:maximum]
            if len(token_ids) < 2:
                raise ValueError(f"TinyStories sample {sample_id} has fewer than two tokens")
            input_ids = torch.tensor((token_ids,), dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            sink.start_batch(BatchMeta(sample_ids=(sample_id,), batch_id=f"loss-{sample_id}"))
            try:
                loss = adapter.forward_loss(
                    loaded,
                    ModelBatch(
                        sample_ids=(sample_id,),
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ),
                )
            finally:
                sink.end_batch()
            count = loss.token_counts[0]
            mean_nll = loss.negative_log_likelihoods[0]
            weighted_nll += mean_nll * count
            evaluated_token_count += count
            truncated_count += int(truncated)
            sample_records.append(
                {
                    "sample_id": sample_id,
                    "content_sha256": hashlib.sha256(story.encode("utf-8")).hexdigest(),
                    "source_token_count": len(encoded),
                    "evaluated_token_count": count,
                    "truncated": truncated,
                    "mean_nll": mean_nll,
                }
            )
    finally:
        handle.remove()
    routing = sink.close()
    if evaluated_token_count <= 0:
        raise ValueError("public native quality evaluation produced no causal-loss tokens")
    mean_nll = weighted_nll / evaluated_token_count
    generation_tokens, generation_record = _measure_generation(config, data, adapter, loaded)
    return ModelMeasurement(
        quality={
            "sample_count": len(sample_records),
            "evaluated_token_count": evaluated_token_count,
            "truncated_sample_count": truncated_count,
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "samples": sample_records,
        },
        generation=generation_record,
        generation_tokens=generation_tokens,
        routing=routing,
        elapsed_seconds=time.perf_counter() - started,
    )


def _measure_confirmatory_loaded_model(
    config: ExperimentConfig,
    data: PreparedPublicData,
    adapter: HFCausalLMAdapter,
    loaded: LoadedModel,
    protocol: ConfirmatoryProtocol,
) -> ModelMeasurement:
    """Measure 256 losses hook-free, then route only an exact 16-story re-forward."""

    started = time.perf_counter()
    selection = data.confirmatory_selection
    if selection is None or len(selection.quality) != 256:
        raise ValueError("confirmatory measurement requires the complete 256-story selection")
    if len(selection.generation_and_routing) != 16:
        raise ValueError("confirmatory measurement requires the complete 16-story subset")
    moe = adapter.discover_moe(loaded)
    if moe is None:
        raise ValueError("pinned public quality model did not expose validated MoE routing")

    quality_records: list[dict[str, Any]] = []
    quality_by_id: dict[str, dict[str, Any]] = {}
    weighted_nll = 0.0
    evaluated_token_count = 0
    for sample in selection.quality:
        input_ids = torch.tensor((sample.token_ids,), dtype=torch.long)
        loss = adapter.forward_loss(
            loaded,
            ModelBatch(
                sample_ids=(sample.sample_id,),
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
            ),
        )
        if loss.sample_ids != (sample.sample_id,) or len(loss.token_counts) != 1:
            raise ValueError(f"loss output is incomplete for {sample.sample_id}")
        token_count = loss.token_counts[0]
        mean_nll = loss.negative_log_likelihoods[0]
        if token_count != sample.evaluated_token_count or not math.isfinite(mean_nll):
            raise ValueError(
                f"loss output does not match the frozen token manifest for {sample.sample_id}"
            )
        record = {
            **sample.manifest_record(),
            "mean_nll": mean_nll,
        }
        quality_records.append(record)
        quality_by_id[sample.sample_id] = record
        weighted_nll += mean_nll * token_count
        evaluated_token_count += token_count
    if len(quality_records) != 256 or len(quality_by_id) != 256:
        raise ValueError("confirmatory loss pass did not produce all 256 unique samples")
    _assert_protocol_value(
        "measured quality evaluated token count",
        evaluated_token_count,
        protocol.holdout_selection.quality_evaluated_token_count,
    )

    state_before_hooks = _model_state_sha256(cast(nn.Module, loaded.model))
    sink = InMemoryRoutingSink(
        "full_trace",
        expert_counts={layer.layer_id: layer.expert_count for layer in moe.layers},
    )
    handle = adapter.attach_routing_hooks(loaded, sink, config)
    reforward_records: list[dict[str, Any]] = []
    try:
        for sample in selection.generation_and_routing:
            input_ids = torch.tensor((sample.token_ids,), dtype=torch.long)
            sink.start_batch(
                BatchMeta(
                    sample_ids=(sample.sample_id,),
                    batch_id=f"routing-reforward-{sample.sample_id}",
                )
            )
            try:
                loss = adapter.forward_loss(
                    loaded,
                    ModelBatch(
                        sample_ids=(sample.sample_id,),
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                    ),
                )
            finally:
                sink.end_batch()
            original = quality_by_id[sample.sample_id]
            if (
                loss.sample_ids != (sample.sample_id,)
                or loss.token_counts != (original["evaluated_token_count"],)
                or loss.negative_log_likelihoods != (original["mean_nll"],)
            ):
                raise RuntimeError(
                    "routing observer changed the exact token count or mean NLL for "
                    f"{sample.sample_id}"
                )
            reforward_records.append(
                {
                    "sample_id": sample.sample_id,
                    "content_sha256": sample.content_sha256,
                    "evaluated_token_count": loss.token_counts[0],
                    "mean_nll": loss.negative_log_likelihoods[0],
                    "matches_hook_free_forward_exactly": True,
                }
            )
    finally:
        handle.remove()
    state_after_hooks = _model_state_sha256(cast(nn.Module, loaded.model))
    if state_after_hooks != state_before_hooks:
        raise RuntimeError("routing observer changed the model state")
    if len(reforward_records) != 16:
        raise ValueError("routing observer proof is missing one or more re-forward samples")
    routing = sink.close()
    expected_events = protocol.generation_and_routing.expected_layer_token_event_count
    descriptor_expected_events = sum(
        len(sample.token_ids) for sample in selection.generation_and_routing
    ) * len(moe.layers)
    if descriptor_expected_events != expected_events:
        raise RuntimeError(
            "discovered MoE layers and frozen routing inputs do not imply exactly 19,764 events"
        )
    if (
        routing.observed_event_count != expected_events
        or routing.recorded_event_count != expected_events
    ):
        raise RuntimeError(
            "routing observer did not produce the exact predeclared 19,764 layer-token events"
        )
    expected_alignment_keys = {
        (sample.sample_id, token_position, layer.layer_id)
        for sample in selection.generation_and_routing
        for token_position in range(len(sample.token_ids))
        for layer in moe.layers
    }
    observed_alignment_keys = tuple(event.alignment_key for event in routing.raw_traces)
    if (
        len(observed_alignment_keys) != expected_events
        or len(set(observed_alignment_keys)) != expected_events
        or set(observed_alignment_keys) != expected_alignment_keys
    ):
        raise RuntimeError(
            "routing observer traces are missing, duplicate, or outside the exact 16-story "
            "token/layer alignment"
        )

    # This call is intentionally after the finally block that removes routing hooks.
    generation_tokens, generation_record = _measure_generation(config, data, adapter, loaded)
    final_state = _model_state_sha256(cast(nn.Module, loaded.model))
    if final_state != state_before_hooks:
        raise RuntimeError("descriptive generation changed the model state")
    mean_nll = weighted_nll / evaluated_token_count
    return ModelMeasurement(
        quality={
            "sample_count": len(quality_records),
            "evaluated_token_count": evaluated_token_count,
            "truncated_sample_count": sum(sample.truncated for sample in selection.quality),
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "samples": quality_records,
        },
        generation=generation_record,
        generation_tokens=generation_tokens,
        routing=routing,
        elapsed_seconds=time.perf_counter() - started,
        routing_observer_proof={
            "schema_version": "routing-observer-proof-v1",
            "quality_forward_scope": "all_256_loss_forwards_hook_free",
            "quality_forward_count": 256,
            "routing_reforward_scope": "exact_16_generation_and_routing_subset_only",
            "routing_reforward_count": 16,
            "routing_hooks_removed_before_generation": True,
            "all_reforward_token_counts_and_mean_nll_match_hook_free_exactly": True,
            "reforward_observations": reforward_records,
            "model_state_before_hooks_sha256": state_before_hooks,
            "model_state_after_hook_removal_sha256": state_after_hooks,
            "model_state_after_generation_sha256": final_state,
            "model_state_unchanged": True,
            "expected_layer_token_event_count": expected_events,
            "observed_layer_token_event_count": routing.observed_event_count,
            "recorded_layer_token_event_count": routing.recorded_event_count,
            "routing_and_generation_inference_claim": "descriptive_only",
        },
    )


def _measure_generation(
    config: ExperimentConfig,
    data: PreparedPublicData,
    adapter: HFCausalLMAdapter,
    loaded: LoadedModel,
) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    suite = require_suite(config, "generation_regression")
    maximum = loaded.descriptor.capabilities.max_context_length
    maximum_input = maximum - suite.decode.max_new_tokens
    if maximum_input < 1:
        raise ValueError("generation decode leaves no model input context")
    outputs: dict[str, tuple[int, ...]] = {}
    records: list[dict[str, Any]] = []
    for sample_id, story in zip(data.generation_ids, data.generation_stories, strict=True):
        encoded = loaded.tokenizer.encode(story, special_tokens=True)
        prompt_ids = encoded[:maximum_input]
        input_ids = torch.tensor((prompt_ids,), dtype=torch.long)
        output = adapter.generate(
            loaded,
            ModelBatch(
                sample_ids=(sample_id,),
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
            ),
            suite.decode,
        )
        tokens = output.token_ids[0]
        outputs[sample_id] = tokens
        records.append(
            {
                "sample_id": sample_id,
                "content_sha256": hashlib.sha256(story.encode("utf-8")).hexdigest(),
                "source_token_count": len(encoded),
                "input_token_count": len(prompt_ids),
                "input_truncated": len(prompt_ids) < len(encoded),
                "generated_token_count": len(tokens),
                "output_sha256": _token_ids_sha256(tokens),
            }
        )
    return outputs, {
        "sample_count": len(records),
        "decode": suite.decode.model_dump(mode="json"),
        "samples": records,
        "output_content_persisted": False,
    }


def _verified_cached_source_weight(config: ExperimentConfig) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    weight = Path(
        hf_hub_download(
            repo_id=config.model.model_id,
            filename=_MODEL_WEIGHT_FILE,
            revision=config.model.revision,
            local_files_only=config.model.local_files_only,
        )
    )
    size = weight.stat().st_size
    sha256 = file_sha256(weight)
    if size != _MODEL_WEIGHT_SIZE_BYTES or sha256 != _MODEL_WEIGHT_SHA256:
        raise ValueError("cached pinned model.safetensors does not match the recorded source")
    return {"file": _MODEL_WEIGHT_FILE, "size_bytes": size, "sha256": sha256}


def _inventory_summary(inventory: tuple[ModuleInfo, ...]) -> dict[str, Any]:
    linears = tuple(
        module for module in inventory if module.supported_precisions == ("float32", "int8", "int4")
    )
    fused = tuple(module for module in inventory if module.class_name.endswith(".ExpertSlice"))
    return {
        "entry_count": len(inventory),
        "addressable_linear_count": len(linears),
        "addressable_linear_names": [module.name for module in linears],
        "fused_expert_slice_count": len(fused),
        "fused_expert_slice_names": [module.name for module in fused],
        "all_fused_expert_slices_float32_only": all(
            module.supported_precisions == ("float32",) for module in fused
        ),
        "all_nonlinear_entries_float32_only": all(
            module.supported_precisions == ("float32",)
            for module in inventory
            if module not in linears
        ),
    }


def _scientific_contract(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "model": config.model.model_dump(mode="json"),
        "evaluation": config.evaluation.model_dump(mode="json"),
        "routing": config.routing.model_dump(mode="json"),
        "runtime": config.runtime.model_dump(mode="json"),
        "security": config.security.model_dump(mode="json"),
    }


def _suite_dataset_identity(suite: EvaluationSuiteConfig) -> tuple[str, str, str]:
    return (suite.dataset, suite.revision, suite.split)


def _token_ids_sha256(tokens: tuple[int, ...]) -> str:
    payload = json.dumps(list(tokens), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _routing_alignment_sha256(artifact: RoutingArtifact) -> str:
    keys = sorted(event.alignment_key for event in artifact.raw_traces)
    payload = json.dumps(keys, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        normalized = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(normalized.shape)).encode("ascii"))
        digest.update(str(normalized.dtype).encode("ascii"))
        digest.update(normalized.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _software_versions() -> dict[str, str]:
    packages = (
        "inkling-quant-lab",
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "tokenizers",
        "huggingface-hub",
        "numpy",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def _release_quantized(model: QuantizedModel) -> None:
    del model.loaded.model
    gc.collect()


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _write_record(path: Path, record: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite quality record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite quality record: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_config", type=Path)
    parser.add_argument("candidate_configs", type=Path, nargs="+")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--expected-dataset-sha256",
        required=True,
        type=require_official_dataset_sha256,
    )
    parser.add_argument("--hardware-label", required=True, type=require_hardware_label)
    parser.add_argument(
        "--confirmatory-protocol",
        type=Path,
        help="Prospectively frozen confirmatory protocol YAML; omitted for legacy v1 mode.",
    )
    parser.add_argument("--execution-ordinal", type=int, choices=(1, 2))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.confirmatory_protocol is not None and len(arguments.candidate_configs) != 1:
        parser.error("--confirmatory-protocol requires exactly one candidate config")
    if arguments.confirmatory_protocol is not None and arguments.execution_ordinal is None:
        parser.error("--confirmatory-protocol requires --execution-ordinal {1,2}")
    if arguments.confirmatory_protocol is None and arguments.execution_ordinal is not None:
        parser.error("--execution-ordinal requires --confirmatory-protocol")
    return arguments


def main() -> None:
    """Load checked configs, execute the comparison, and emit one JSON record."""

    arguments = _arguments()
    baseline = load_config(arguments.baseline_config)
    candidates = tuple(load_config(path) for path in arguments.candidate_configs)
    confirmatory_protocol_path = getattr(arguments, "confirmatory_protocol", None)
    confirmatory_protocol = (
        None
        if confirmatory_protocol_path is None
        else load_confirmatory_protocol(confirmatory_protocol_path)
    )
    output = (
        None
        if arguments.output is None
        else resolve_quality_output_path(
            baseline,
            project_root=Path.cwd(),
            requested=arguments.output,
        )
    )
    record = evaluate(
        baseline,
        candidates,
        arguments.dataset,
        expected_dataset_sha256=arguments.expected_dataset_sha256,
        project_root=Path.cwd().resolve(),
        hardware_label=arguments.hardware_label,
        confirmatory_protocol=confirmatory_protocol,
        execution_ordinal=getattr(arguments, "execution_ordinal", None),
    )
    if output is None:
        print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    else:
        _write_record(output, record)
        print(output)


if __name__ == "__main__":
    main()
