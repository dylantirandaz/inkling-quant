"""Paired finite-population noninferiority inference for token-mean NLL.

The estimator treats stories as clusters.  Within each declared stratum it
uses the Horvitz--Thompson estimate of the token-weighted NLL numerator, then
divides the summed numerator by the exact, externally known population token
total.  The bootstrap is a deterministic, finite-population-corrected paired
cluster bootstrap; it is not an IID token bootstrap.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_MAX_EXACT_INTEGER = 2**63 - 1
_DEFAULT_BOOTSTRAP_CHUNK_SIZE = 1_024


class ImmutableNoninferiorityRecord(BaseModel):
    """Strict, immutable, finite record with canonical JSON serialization."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False, strict=True)

    def canonical_json(self) -> str:
        """Return deterministic JSON that rejects non-finite floating-point values."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )


class StoryNLLObservation(ImmutableNoninferiorityRecord):
    """One story's identity and token-mean NLL from one model condition."""

    sample_id: str = Field(min_length=1)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    stratum: str = Field(min_length=1)
    token_count: int = Field(gt=0, le=_MAX_EXACT_INTEGER)
    mean_nll: float = Field(ge=0.0)

    @field_validator("sample_id", "stratum")
    @classmethod
    def identity_is_whitespace_canonical(cls, value: str) -> str:
        """Reject empty-looking or ambiguously padded identifiers."""

        if value != value.strip():
            raise ValueError("sample_id and stratum must not have surrounding whitespace")
        if not value:
            raise ValueError("sample_id and stratum must not be empty")
        return value


class PairedNLLObservation(ImmutableNoninferiorityRecord):
    """One already-aligned story cluster with both model-condition NLLs."""

    sample_id: str = Field(min_length=1)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    stratum: str = Field(min_length=1)
    token_count: int = Field(gt=0, le=_MAX_EXACT_INTEGER)
    baseline_mean_nll: float = Field(ge=0.0)
    candidate_mean_nll: float = Field(ge=0.0)

    @field_validator("sample_id", "stratum")
    @classmethod
    def identity_is_whitespace_canonical(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("sample_id and stratum must not have surrounding whitespace")
        return value


class StratumDesign(ImmutableNoninferiorityRecord):
    """Exact finite-population and selected sample sizes for one stratum."""

    stratum: str = Field(min_length=1)
    population_size: int = Field(ge=2, le=_MAX_EXACT_INTEGER)
    selected_size: int = Field(ge=2, le=_MAX_EXACT_INTEGER)

    @field_validator("stratum")
    @classmethod
    def stratum_is_whitespace_canonical(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("stratum must not have surrounding whitespace")
        return value

    @model_validator(mode="after")
    def selected_size_fits_population(self) -> Self:
        if self.selected_size > self.population_size:
            raise ValueError("selected_size must not exceed population_size")
        return self


class PairedNoninferiorityDesign(ImmutableNoninferiorityRecord):
    """Predeclared sampling design and noninferiority decision parameters."""

    strata: tuple[StratumDesign, ...]
    exact_population_token_total: int = Field(gt=0, le=_MAX_EXACT_INTEGER)
    margin_nll: float = Field(gt=0.0)
    confidence: float = Field(gt=0.5, lt=1.0)
    bootstrap_replicates: int = Field(ge=2, le=_MAX_EXACT_INTEGER)
    seed: int = Field(ge=0, le=2**64 - 1)

    @model_validator(mode="after")
    def strata_are_nonempty_and_unique(self) -> Self:
        if not self.strata:
            raise ValueError("at least one stratum sampling design is required")
        names = tuple(item.stratum for item in self.strata)
        if len(set(names)) != len(names):
            raise ValueError("stratum sampling designs must have unique names")
        return self


class NumericInterval(ImmutableNoninferiorityRecord):
    """Closed numerical interval represented as JSON-safe finite floats."""

    lower: float
    upper: float

    @model_validator(mode="after")
    def endpoints_are_ordered(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("interval lower endpoint must not exceed upper endpoint")
        return self


class RelativePerplexityChange(ImmutableNoninferiorityRecord):
    """Candidate/baseline perplexity ratio minus one, derived with ``expm1``."""

    point: float
    upper_confidence_bound: float
    margin: float
    central_95_interval: NumericInterval


class StratumPointSummary(ImmutableNoninferiorityRecord):
    """Auditable point-estimator and variance inputs for one sampling stratum."""

    stratum: str
    population_size: int
    selected_size: int
    sampling_fraction: float
    finite_population_correction: float
    bootstrap_deviation_scale: float
    sampled_token_count: int
    horvitz_thompson_token_total: float
    baseline_sample_token_nll_sum: float
    candidate_sample_token_nll_sum: float
    baseline_horvitz_thompson_numerator: float
    candidate_horvitz_thompson_numerator: float
    difference_horvitz_thompson_numerator: float
    difference_y_sample_mean: float
    difference_y_sample_variance: float
    analytic_numerator_variance: float


class NoninferiorityMethodMetadata(ImmutableNoninferiorityRecord):
    """Stable, explicit description of the estimator and inference procedure."""

    estimand: Literal["finite_population_token_weighted_candidate_minus_baseline_mean_nll"] = (
        "finite_population_token_weighted_candidate_minus_baseline_mean_nll"
    )
    estimator: Literal["stratified_horvitz_thompson_nll_numerator_over_known_token_denominator"] = (
        "stratified_horvitz_thompson_nll_numerator_over_known_token_denominator"
    )
    sampling_design: Literal["stratified_srswor"] = "stratified_srswor"
    resampling_unit: Literal["paired_story_cluster"] = "paired_story_cluster"
    bootstrap: Literal["within_stratum_paired_empirical_resampling_with_srswor_moment_matching"] = (
        "within_stratum_paired_empirical_resampling_with_srswor_moment_matching"
    )
    bootstrap_deviation_scale: Literal["sqrt(n_h*(1-n_h/N_h)/(n_h-1))"] = (
        "sqrt(n_h*(1-n_h/N_h)/(n_h-1))"
    )
    analytic_variance: Literal["sum_h(N_h^2*(1-n_h/N_h)*s_y_h^2/n_h)/known_token_total^2"] = (
        "sum_h(N_h^2*(1-n_h/N_h)*s_y_h^2/n_h)/known_token_total^2"
    )
    rng: Literal["numpy.random.Generator(numpy.random.PCG64(seed))"] = (
        "numpy.random.Generator(numpy.random.PCG64(seed))"
    )
    stratum_order: tuple[str, ...]
    within_stratum_order: Literal["sample_id_ascending"] = "sample_id_ascending"
    quantile_method: Literal["linear"] = "linear"
    upper_quantile: Literal["q=confidence"] = "q=confidence"
    central_interval: Literal["q=(0.025,0.975)"] = "q=(0.025,0.975)"
    bootstrap_standard_error: Literal["sample_sd_ddof_1"] = "sample_sd_ddof_1"
    distribution_sha256_encoding: Literal[
        "sha256(contiguous_little_endian_float64_replicates_in_generation_order)"
    ] = "sha256(contiguous_little_endian_float64_replicates_in_generation_order)"
    decision_rule: Literal["upper_bound_strictly_less_than_margin"] = (
        "upper_bound_strictly_less_than_margin"
    )


class PairedNoninferiorityResult(ImmutableNoninferiorityRecord):
    """Complete point estimate, uncertainty, decision, and method provenance."""

    schema_version: Literal["paired-stratified-ht-noninferiority-v1"] = (
        "paired-stratified-ht-noninferiority-v1"
    )
    exact_population_token_total: int
    baseline_mean_nll_ht: float
    candidate_mean_nll_ht: float
    candidate_minus_baseline_mean_nll_ht: float
    margin_nll: float
    confidence: float
    upper_confidence_bound_nll: float
    passed: bool
    analytic_design_standard_error_nll: float
    bootstrap_standard_error_nll: float
    bootstrap_central_95_interval_nll: NumericInterval
    relative_perplexity_change: RelativePerplexityChange
    bootstrap_replicates: int
    bootstrap_seed: int
    bootstrap_distribution_sha256: str = Field(pattern=_SHA256_PATTERN)
    strata: tuple[StratumPointSummary, ...]
    method: NoninferiorityMethodMetadata

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-mode mapping suitable for immutable evidence records."""

        return self.model_dump(mode="json")


@dataclass(frozen=True, slots=True)
class _AlignedPair:
    sample_id: str
    stratum: str
    token_count: int
    baseline_mean_nll: float
    candidate_mean_nll: float


@dataclass(frozen=True, slots=True)
class _StratumComputation:
    summary: StratumPointSummary
    differences: npt.NDArray[np.float64]


def _index_observations(
    observations: Sequence[StoryNLLObservation], *, condition: str
) -> dict[str, StoryNLLObservation]:
    if not observations:
        raise ValueError(f"{condition} observations must not be empty")
    sample_ids = tuple(item.sample_id for item in observations)
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"{condition} observations contain duplicate sample_id values")
    content_hashes = tuple(item.content_sha256 for item in observations)
    if len(set(content_hashes)) != len(content_hashes):
        raise ValueError(f"{condition} observations contain duplicate content_sha256 values")
    return {item.sample_id: item for item in observations}


def _align_observations(
    baseline: Sequence[StoryNLLObservation],
    candidate: Sequence[StoryNLLObservation],
) -> tuple[_AlignedPair, ...]:
    baseline_by_id = _index_observations(baseline, condition="baseline")
    candidate_by_id = _index_observations(candidate, condition="candidate")
    baseline_ids = set(baseline_by_id)
    candidate_ids = set(candidate_by_id)
    if baseline_ids != candidate_ids:
        missing_candidate = sorted(baseline_ids - candidate_ids)
        missing_baseline = sorted(candidate_ids - baseline_ids)
        raise ValueError(
            "baseline/candidate sample_id sets are misaligned: "
            f"missing from candidate={missing_candidate!r}; "
            f"missing from baseline={missing_baseline!r}"
        )

    aligned: list[_AlignedPair] = []
    for sample_id in sorted(baseline_ids):
        baseline_item = baseline_by_id[sample_id]
        candidate_item = candidate_by_id[sample_id]
        if baseline_item.content_sha256 != candidate_item.content_sha256:
            raise ValueError(f"content_sha256 is misaligned for sample_id {sample_id!r}")
        if baseline_item.token_count != candidate_item.token_count:
            raise ValueError(f"token_count is misaligned for sample_id {sample_id!r}")
        if baseline_item.stratum != candidate_item.stratum:
            raise ValueError(f"stratum is misaligned for sample_id {sample_id!r}")
        aligned.append(
            _AlignedPair(
                sample_id=sample_id,
                stratum=baseline_item.stratum,
                token_count=baseline_item.token_count,
                baseline_mean_nll=baseline_item.mean_nll,
                candidate_mean_nll=candidate_item.mean_nll,
            )
        )
    return tuple(aligned)


def _require_finite(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"derived {name} is not finite")
    return value


def _relative_perplexity_change(delta_nll: float, *, name: str) -> float:
    try:
        value = math.expm1(delta_nll)
    except OverflowError as error:
        raise ValueError(f"derived {name} overflows finite float64") from error
    return _require_finite(value, name=name)


def _distribution_sha256(distribution: npt.NDArray[np.float64]) -> str:
    canonical = np.ascontiguousarray(distribution, dtype=np.dtype("<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _compute_stratum(pairs: tuple[_AlignedPair, ...], design: StratumDesign) -> _StratumComputation:
    selected = tuple(pair for pair in pairs if pair.stratum == design.stratum)
    if len(selected) != design.selected_size:
        raise ValueError(
            f"stratum {design.stratum!r} declares selected_size={design.selected_size}, "
            f"but {len(selected)} aligned observations were provided"
        )
    if len(selected) < 2:
        raise ValueError(f"stratum {design.stratum!r} requires at least two selected samples")

    population_size = design.population_size
    selected_size = design.selected_size
    sampling_fraction = selected_size / population_size
    finite_population_correction = 1.0 - sampling_fraction
    deviation_scale = math.sqrt(selected_size * finite_population_correction / (selected_size - 1))
    sampled_token_count = sum(pair.token_count for pair in selected)

    baseline_values = tuple(pair.token_count * pair.baseline_mean_nll for pair in selected)
    candidate_values = tuple(pair.token_count * pair.candidate_mean_nll for pair in selected)
    difference_values = tuple(
        candidate_value - baseline_value
        for baseline_value, candidate_value in zip(baseline_values, candidate_values, strict=True)
    )
    baseline_sum = _require_finite(math.fsum(baseline_values), name="baseline sample sum")
    candidate_sum = _require_finite(math.fsum(candidate_values), name="candidate sample sum")
    difference_mean = _require_finite(
        math.fsum(difference_values) / selected_size,
        name="difference y sample mean",
    )
    difference_variance = _require_finite(
        math.fsum((value - difference_mean) ** 2 for value in difference_values)
        / (selected_size - 1),
        name="difference y sample variance",
    )
    expansion = population_size / selected_size
    baseline_ht = _require_finite(
        expansion * baseline_sum, name="baseline Horvitz-Thompson numerator"
    )
    candidate_ht = _require_finite(
        expansion * candidate_sum, name="candidate Horvitz-Thompson numerator"
    )
    difference_ht = _require_finite(
        population_size * difference_mean,
        name="difference Horvitz-Thompson numerator",
    )
    analytic_variance = _require_finite(
        population_size**2 * finite_population_correction * difference_variance / selected_size,
        name="analytic numerator variance",
    )
    summary = StratumPointSummary(
        stratum=design.stratum,
        population_size=population_size,
        selected_size=selected_size,
        sampling_fraction=sampling_fraction,
        finite_population_correction=finite_population_correction,
        bootstrap_deviation_scale=deviation_scale,
        sampled_token_count=sampled_token_count,
        horvitz_thompson_token_total=expansion * sampled_token_count,
        baseline_sample_token_nll_sum=baseline_sum,
        candidate_sample_token_nll_sum=candidate_sum,
        baseline_horvitz_thompson_numerator=baseline_ht,
        candidate_horvitz_thompson_numerator=candidate_ht,
        difference_horvitz_thompson_numerator=difference_ht,
        difference_y_sample_mean=difference_mean,
        difference_y_sample_variance=difference_variance,
        analytic_numerator_variance=analytic_variance,
    )
    return _StratumComputation(
        summary=summary,
        differences=np.asarray(difference_values, dtype=np.float64),
    )


def _bootstrap_distribution(
    computations: tuple[_StratumComputation, ...],
    *,
    point_estimate: float,
    exact_population_token_total: int,
    bootstrap_replicates: int,
    seed: int,
    chunk_size: int,
) -> npt.NDArray[np.float64]:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("bootstrap_chunk_size must be a positive integer")
    distribution = np.full(bootstrap_replicates, point_estimate, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))

    # All B draws for one stratum are consumed before advancing to the next
    # lexicographically ordered stratum.  Chunking changes only allocation size,
    # not the flattened PCG64 draw order or floating-point reduction shape.
    for computation in computations:
        values = computation.differences
        selected_size = computation.summary.selected_size
        coefficient = (
            computation.summary.population_size
            * computation.summary.bootstrap_deviation_scale
            / exact_population_token_total
        )
        center = computation.summary.difference_y_sample_mean
        for start in range(0, bootstrap_replicates, chunk_size):
            stop = min(start + chunk_size, bootstrap_replicates)
            indices = rng.integers(
                0,
                selected_size,
                size=(stop - start, selected_size),
                dtype=np.int64,
            )
            resampled_means = np.mean(values[indices], axis=1, dtype=np.float64)
            distribution[start:stop] += coefficient * (resampled_means - center)
    if not bool(np.all(np.isfinite(distribution))):
        raise ValueError("derived bootstrap distribution contains non-finite values")
    return distribution


def evaluate_paired_noninferiority(
    baseline: Sequence[StoryNLLObservation],
    candidate: Sequence[StoryNLLObservation],
    design: PairedNoninferiorityDesign,
    *,
    bootstrap_chunk_size: int = _DEFAULT_BOOTSTRAP_CHUNK_SIZE,
) -> PairedNoninferiorityResult:
    """Evaluate strict paired noninferiority under a stratified SRSWOR design.

    The one-sided percentile upper bound is the bootstrap quantile at
    ``design.confidence``.  Passing requires strict inequality: equality with
    the predeclared NLL margin fails.
    """

    aligned = _align_observations(baseline, candidate)
    designs = tuple(sorted(design.strata, key=lambda item: item.stratum))
    declared_strata = {item.stratum for item in designs}
    observed_strata = {pair.stratum for pair in aligned}
    if observed_strata != declared_strata:
        missing_observations = sorted(declared_strata - observed_strata)
        undeclared_observations = sorted(observed_strata - declared_strata)
        raise ValueError(
            "observed and declared strata are misaligned: "
            f"without observations={missing_observations!r}; "
            f"undeclared observations={undeclared_observations!r}"
        )

    minimum_possible_population_tokens = sum(pair.token_count for pair in aligned) + sum(
        item.population_size - item.selected_size for item in designs
    )
    if design.exact_population_token_total < minimum_possible_population_tokens:
        raise ValueError(
            "exact_population_token_total is smaller than the sampled tokens plus one "
            "positive token for every unsampled population member"
        )

    computations = tuple(_compute_stratum(aligned, item) for item in designs)
    summaries = tuple(item.summary for item in computations)
    denominator = design.exact_population_token_total
    baseline_mean_nll = _require_finite(
        math.fsum(item.baseline_horvitz_thompson_numerator for item in summaries) / denominator,
        name="baseline HT mean NLL",
    )
    candidate_mean_nll = _require_finite(
        math.fsum(item.candidate_horvitz_thompson_numerator for item in summaries) / denominator,
        name="candidate HT mean NLL",
    )
    delta = _require_finite(
        math.fsum(item.difference_horvitz_thompson_numerator for item in summaries) / denominator,
        name="candidate-minus-baseline HT mean NLL",
    )
    analytic_variance = _require_finite(
        math.fsum(item.analytic_numerator_variance for item in summaries) / denominator**2,
        name="analytic design variance",
    )
    analytic_standard_error = _require_finite(
        math.sqrt(analytic_variance), name="analytic design standard error"
    )

    distribution = _bootstrap_distribution(
        computations,
        point_estimate=delta,
        exact_population_token_total=denominator,
        bootstrap_replicates=design.bootstrap_replicates,
        seed=design.seed,
        chunk_size=bootstrap_chunk_size,
    )
    upper_bound = _require_finite(
        float(np.quantile(distribution, q=design.confidence, method="linear")),
        name="upper bootstrap quantile",
    )
    central_values = np.quantile(
        distribution,
        q=np.asarray((0.025, 0.975), dtype=np.float64),
        method="linear",
    )
    central_interval = NumericInterval(
        lower=_require_finite(float(central_values[0]), name="central interval lower endpoint"),
        upper=_require_finite(float(central_values[1]), name="central interval upper endpoint"),
    )
    bootstrap_standard_error = _require_finite(
        float(np.std(distribution, ddof=1, dtype=np.float64)),
        name="bootstrap standard error",
    )
    relative_interval = NumericInterval(
        lower=_relative_perplexity_change(
            central_interval.lower, name="relative perplexity central lower endpoint"
        ),
        upper=_relative_perplexity_change(
            central_interval.upper, name="relative perplexity central upper endpoint"
        ),
    )

    return PairedNoninferiorityResult(
        exact_population_token_total=denominator,
        baseline_mean_nll_ht=baseline_mean_nll,
        candidate_mean_nll_ht=candidate_mean_nll,
        candidate_minus_baseline_mean_nll_ht=delta,
        margin_nll=design.margin_nll,
        confidence=design.confidence,
        upper_confidence_bound_nll=upper_bound,
        passed=upper_bound < design.margin_nll,
        analytic_design_standard_error_nll=analytic_standard_error,
        bootstrap_standard_error_nll=bootstrap_standard_error,
        bootstrap_central_95_interval_nll=central_interval,
        relative_perplexity_change=RelativePerplexityChange(
            point=_relative_perplexity_change(delta, name="relative perplexity point estimate"),
            upper_confidence_bound=_relative_perplexity_change(
                upper_bound, name="relative perplexity upper confidence bound"
            ),
            margin=_relative_perplexity_change(
                design.margin_nll, name="relative perplexity margin"
            ),
            central_95_interval=relative_interval,
        ),
        bootstrap_replicates=design.bootstrap_replicates,
        bootstrap_seed=design.seed,
        bootstrap_distribution_sha256=_distribution_sha256(distribution),
        strata=summaries,
        method=NoninferiorityMethodMetadata(stratum_order=tuple(item.stratum for item in designs)),
    )


def paired_stratified_nll_noninferiority(
    observations: Sequence[PairedNLLObservation],
    strata: Sequence[StratumDesign],
    *,
    exact_population_token_total: int,
    margin_nll: float,
    confidence: float,
    bootstrap_replicates: int,
    seed: int,
    bootstrap_chunk_size: int = _DEFAULT_BOOTSTRAP_CHUNK_SIZE,
) -> PairedNoninferiorityResult:
    """Convenience API for already-aligned paired story observations.

    Use :func:`evaluate_paired_noninferiority` when baseline and candidate
    records originate independently and the module must validate their exact
    IDs, content hashes, token counts, and strata before pairing them.
    """

    baseline = tuple(
        StoryNLLObservation(
            sample_id=item.sample_id,
            content_sha256=item.content_sha256,
            stratum=item.stratum,
            token_count=item.token_count,
            mean_nll=item.baseline_mean_nll,
        )
        for item in observations
    )
    candidate = tuple(
        StoryNLLObservation(
            sample_id=item.sample_id,
            content_sha256=item.content_sha256,
            stratum=item.stratum,
            token_count=item.token_count,
            mean_nll=item.candidate_mean_nll,
        )
        for item in observations
    )
    design = PairedNoninferiorityDesign(
        strata=tuple(strata),
        exact_population_token_total=exact_population_token_total,
        margin_nll=margin_nll,
        confidence=confidence,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    return evaluate_paired_noninferiority(
        baseline,
        candidate,
        design,
        bootstrap_chunk_size=bootstrap_chunk_size,
    )


__all__ = [
    "NoninferiorityMethodMetadata",
    "NumericInterval",
    "PairedNLLObservation",
    "PairedNoninferiorityDesign",
    "PairedNoninferiorityResult",
    "RelativePerplexityChange",
    "StoryNLLObservation",
    "StratumDesign",
    "StratumPointSummary",
    "evaluate_paired_noninferiority",
    "paired_stratified_nll_noninferiority",
]
