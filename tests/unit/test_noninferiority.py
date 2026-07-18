"""Deterministic contracts for paired finite-population noninferiority inference."""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import pytest
from pydantic import ValidationError

from inkling_quant_lab.evaluation.noninferiority import (
    PairedNLLObservation,
    PairedNoninferiorityDesign,
    PairedNoninferiorityResult,
    StoryNLLObservation,
    StratumDesign,
    evaluate_paired_noninferiority,
    paired_stratified_nll_noninferiority,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _paired_observations() -> tuple[PairedNLLObservation, ...]:
    # Deliberately not grouped or sorted: scientific order must be canonical.
    return (
        PairedNLLObservation(
            sample_id="b-2",
            content_sha256=_digest(4),
            stratum="long",
            token_count=5,
            baseline_mean_nll=1.5,
            candidate_mean_nll=1.55,
        ),
        PairedNLLObservation(
            sample_id="a-2",
            content_sha256=_digest(2),
            stratum="short",
            token_count=4,
            baseline_mean_nll=2.0,
            candidate_mean_nll=1.9,
        ),
        PairedNLLObservation(
            sample_id="b-1",
            content_sha256=_digest(3),
            stratum="long",
            token_count=3,
            baseline_mean_nll=0.5,
            candidate_mean_nll=0.55,
        ),
        PairedNLLObservation(
            sample_id="a-1",
            content_sha256=_digest(1),
            stratum="short",
            token_count=2,
            baseline_mean_nll=1.0,
            candidate_mean_nll=1.1,
        ),
        PairedNLLObservation(
            sample_id="b-3",
            content_sha256=_digest(5),
            stratum="long",
            token_count=2,
            baseline_mean_nll=2.5,
            candidate_mean_nll=2.4,
        ),
    )


def _strata() -> tuple[StratumDesign, ...]:
    # Reverse lexical order to prove the implementation canonicalizes h order.
    return (
        StratumDesign(stratum="short", population_size=5, selected_size=2),
        StratumDesign(stratum="long", population_size=4, selected_size=3),
    )


def _evaluate(
    *, margin: float = 0.05, chunk_size: int = 7, bootstrap_replicates: int = 1_000
) -> PairedNoninferiorityResult:
    return paired_stratified_nll_noninferiority(
        _paired_observations(),
        _strata(),
        exact_population_token_total=40,
        margin_nll=margin,
        confidence=0.95,
        bootstrap_replicates=bootstrap_replicates,
        seed=42,
        bootstrap_chunk_size=chunk_size,
    )


@pytest.mark.unit
def test_golden_deterministic_stratified_result() -> None:
    """A fixed PCG64 design has stable estimates, quantiles, and byte digest."""

    result = _evaluate()

    assert result.baseline_mean_nll_ht == pytest.approx(1.0916666666666666)
    assert result.candidate_mean_nll_ht == pytest.approx(1.0858333333333334)
    assert result.candidate_minus_baseline_mean_nll_ht == pytest.approx(-0.005833333333333329)
    assert result.analytic_design_standard_error_nll == pytest.approx(
        math.sqrt((1.35 + 0.07444444444444445) / 40**2)
    )
    assert result.bootstrap_standard_error_nll == pytest.approx(0.029769327871141695)
    assert result.upper_confidence_bound_nll == pytest.approx(0.04239020356267176)
    assert result.bootstrap_central_95_interval_nll.lower == pytest.approx(-0.056098111681657764)
    assert result.bootstrap_central_95_interval_nll.upper == pytest.approx(0.04443144501499107)
    assert result.bootstrap_distribution_sha256 == (
        "e7423fa2376f0f787ff77810b076473228d8a952f77e0221d9274ef850217e12"
    )
    assert result.passed is True
    assert result.method.stratum_order == ("long", "short")
    assert result.method.resampling_unit == "paired_story_cluster"
    assert result.method.quantile_method == "linear"
    assert result.relative_perplexity_change.point == pytest.approx(
        math.expm1(result.candidate_minus_baseline_mean_nll_ht)
    )
    assert result.relative_perplexity_change.margin == pytest.approx(math.expm1(0.05))


@pytest.mark.unit
def test_exact_ht_arithmetic_and_stratum_fpc_inputs() -> None:
    """Point and analytic estimates use y_i=tokens_i*(candidate-baseline NLL)."""

    result = _evaluate(bootstrap_replicates=20)
    long, short = result.strata

    assert long.stratum == "long"
    assert long.sampling_fraction == 3 / 4
    assert long.finite_population_correction == 1 / 4
    assert long.bootstrap_deviation_scale == pytest.approx(math.sqrt(3 * 0.25 / 2))
    assert long.sampled_token_count == 10
    assert long.horvitz_thompson_token_total == pytest.approx(40 / 3)
    assert long.baseline_sample_token_nll_sum == pytest.approx(14.0)
    assert long.candidate_sample_token_nll_sum == pytest.approx(14.2)
    assert long.difference_y_sample_mean == pytest.approx(1 / 15)
    assert long.difference_y_sample_variance == pytest.approx(0.05583333333333335)
    assert long.difference_horvitz_thompson_numerator == pytest.approx(4 / 15)
    assert long.analytic_numerator_variance == pytest.approx(0.07444444444444447)

    assert short.sampling_fraction == 2 / 5
    assert short.finite_population_correction == 3 / 5
    assert short.bootstrap_deviation_scale == pytest.approx(math.sqrt(1.2))
    assert short.difference_y_sample_mean == pytest.approx(-0.1)
    assert short.difference_y_sample_variance == pytest.approx(0.18)
    assert short.difference_horvitz_thompson_numerator == pytest.approx(-0.5)
    assert short.analytic_numerator_variance == pytest.approx(1.35)


@pytest.mark.unit
def test_bootstrap_chunk_size_does_not_change_rng_or_float_order() -> None:
    """Chunking limits allocation while preserving exact PCG64 draw consumption."""

    one = _evaluate(chunk_size=1)
    uneven = _evaluate(chunk_size=13)
    whole = _evaluate(chunk_size=10_000)

    assert one == uneven == whole
    assert one.canonical_json() == whole.canonical_json()


@pytest.mark.unit
def test_noninferiority_pass_is_strict_at_equality() -> None:
    """An upper bound equal to the margin fails; only a smaller bound passes."""

    reference = _evaluate()
    boundary = reference.upper_confidence_bound_nll

    assert _evaluate(margin=math.nextafter(boundary, math.inf)).passed is True
    equal = _evaluate(margin=boundary)
    assert equal.upper_confidence_bound_nll == boundary
    assert equal.passed is False
    assert _evaluate(margin=math.nextafter(boundary, 0.0)).passed is False


@pytest.mark.unit
def test_result_is_round_trip_json_safe() -> None:
    """Evidence serialization contains only finite JSON values and validates on reload."""

    result = _evaluate(bootstrap_replicates=20)
    payload = result.as_dict()
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True)

    assert PairedNoninferiorityResult.model_validate_json(encoded) == result
    assert json.loads(result.canonical_json()) == payload
    assert "bootstrap_chunk_size" not in payload


def _condition_observations() -> tuple[
    tuple[StoryNLLObservation, ...], tuple[StoryNLLObservation, ...]
]:
    pairs = _paired_observations()
    baseline = tuple(
        StoryNLLObservation(
            sample_id=item.sample_id,
            content_sha256=item.content_sha256,
            stratum=item.stratum,
            token_count=item.token_count,
            mean_nll=item.baseline_mean_nll,
        )
        for item in pairs
    )
    candidate = tuple(
        StoryNLLObservation(
            sample_id=item.sample_id,
            content_sha256=item.content_sha256,
            stratum=item.stratum,
            token_count=item.token_count,
            mean_nll=item.candidate_mean_nll,
        )
        for item in pairs
    )
    return baseline, candidate


def _design(**updates: object) -> PairedNoninferiorityDesign:
    values: dict[str, object] = {
        "strata": _strata(),
        "exact_population_token_total": 40,
        "margin_nll": 0.05,
        "confidence": 0.95,
        "bootstrap_replicates": 20,
        "seed": 42,
    }
    values.update(updates)
    return PairedNoninferiorityDesign.model_validate(values)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("change_candidate", "message"),
    (
        (
            lambda values: values[:-1],
            "sample_id sets are misaligned",
        ),
        (
            lambda values: (
                *values[:-1],
                values[-1].model_copy(update={"content_sha256": _digest(63)}),
            ),
            "content_sha256 is misaligned",
        ),
        (
            lambda values: (
                *values[:-1],
                values[-1].model_copy(update={"token_count": 99}),
            ),
            "token_count is misaligned",
        ),
        (
            lambda values: (
                *values[:-1],
                values[-1].model_copy(update={"stratum": "short"}),
            ),
            "stratum is misaligned",
        ),
    ),
)
def test_independent_condition_alignment_fails_closed(
    change_candidate: Callable[[tuple[StoryNLLObservation, ...]], tuple[StoryNLLObservation, ...]],
    message: str,
) -> None:
    """Independent model outputs cannot silently lose or alter paired provenance."""

    baseline, candidate = _condition_observations()
    with pytest.raises(ValueError, match=message):
        evaluate_paired_noninferiority(baseline, change_candidate(candidate), _design())


@pytest.mark.unit
@pytest.mark.parametrize("condition", ("baseline", "candidate"))
@pytest.mark.parametrize("duplicate", ("sample_id", "content_sha256"))
def test_duplicate_condition_identity_fails_closed(condition: str, duplicate: str) -> None:
    baseline, candidate = _condition_observations()
    selected = list(baseline if condition == "baseline" else candidate)
    selected[1] = selected[1].model_copy(update={duplicate: getattr(selected[0], duplicate)})

    with pytest.raises(ValueError, match=f"{condition} observations contain duplicate {duplicate}"):
        evaluate_paired_noninferiority(
            tuple(selected) if condition == "baseline" else baseline,
            tuple(selected) if condition == "candidate" else candidate,
            _design(),
        )


@pytest.mark.unit
def test_population_and_stratum_contracts_fail_closed() -> None:
    baseline, candidate = _condition_observations()

    with pytest.raises(ValueError, match="selected_size=3"):
        evaluate_paired_noninferiority(
            baseline,
            candidate,
            _design(
                strata=(
                    StratumDesign(stratum="long", population_size=4, selected_size=3),
                    StratumDesign(stratum="short", population_size=5, selected_size=3),
                )
            ),
        )
    with pytest.raises(ValueError, match="observed and declared strata are misaligned"):
        evaluate_paired_noninferiority(
            baseline,
            candidate,
            _design(
                strata=(
                    StratumDesign(stratum="long", population_size=4, selected_size=3),
                    StratumDesign(stratum="other", population_size=5, selected_size=2),
                )
            ),
        )
    with pytest.raises(ValueError, match="smaller than the sampled tokens"):
        evaluate_paired_noninferiority(
            baseline,
            candidate,
            _design(exact_population_token_total=19),
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (
            lambda: PairedNLLObservation(
                sample_id="x",
                content_sha256=_digest(1),
                stratum="h",
                token_count=0,
                baseline_mean_nll=1.0,
                candidate_mean_nll=1.0,
            ),
            "greater than 0",
        ),
        (
            lambda: PairedNLLObservation(
                sample_id="x",
                content_sha256=_digest(1),
                stratum="h",
                token_count=1,
                baseline_mean_nll=float("nan"),
                candidate_mean_nll=1.0,
            ),
            "finite number",
        ),
        (
            lambda: StratumDesign(stratum="h", population_size=2, selected_size=1),
            "greater than or equal to 2",
        ),
        (
            lambda: StratumDesign(stratum="h", population_size=2, selected_size=3),
            "selected_size must not exceed",
        ),
        (
            lambda: _design(confidence=0.5),
            "greater than 0.5",
        ),
        (
            lambda: _design(bootstrap_replicates=1),
            "greater than or equal to 2",
        ),
        (
            lambda: _design(margin_nll=0.0),
            "greater than 0",
        ),
        (
            lambda: _design(strata=()),
            "at least one stratum",
        ),
        (
            lambda: _design(strata=(_strata()[0], _strata()[0])),
            "unique names",
        ),
    ),
)
def test_invalid_records_and_design_parameters_fail_closed(
    factory: Callable[[], object], message: str
) -> None:
    with pytest.raises((ValueError, ValidationError), match=message):
        factory()


@pytest.mark.unit
def test_empty_inputs_and_invalid_chunk_fail_closed() -> None:
    baseline, candidate = _condition_observations()

    with pytest.raises(ValueError, match="baseline observations must not be empty"):
        evaluate_paired_noninferiority((), candidate, _design())
    with pytest.raises(ValueError, match="candidate observations must not be empty"):
        evaluate_paired_noninferiority(baseline, (), _design())
    with pytest.raises(ValueError, match="bootstrap_chunk_size must be a positive integer"):
        evaluate_paired_noninferiority(baseline, candidate, _design(), bootstrap_chunk_size=0)
    with pytest.raises(ValueError, match="bootstrap_chunk_size must be a positive integer"):
        evaluate_paired_noninferiority(baseline, candidate, _design(), bootstrap_chunk_size=True)
