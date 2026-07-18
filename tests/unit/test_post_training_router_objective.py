"""Stable CPU reference semantics for domain-pair router training."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from inkling_quant_lab.post_training.router import (
    DomainExpertPair,
    DomainPairObjectiveResult,
    RouterObjectiveError,
    domain_pair_cross_entropy,
)

pytestmark = pytest.mark.unit


def test_uniform_logits_match_soft_target_cross_entropy_and_deterministic_top2() -> None:
    result = domain_pair_cross_entropy(
        logits=((0.0, 0.0, 0.0, 0.0),),
        domain_ids=("books",),
        domain_pairs={"books": (0, 1)},
    )

    assert result.objective_version == "top2-domain-pair-ce-v1"
    assert result.loss == pytest.approx(math.log(4.0))
    assert result.token_mean_cross_entropy == pytest.approx(math.log(4.0))
    assert result.exact_top2_pair_accuracy == 1.0
    assert result.per_domain[0].target_expert_indices == (0, 1)


def test_large_logits_remain_finite_and_pair_accuracy_is_unordered() -> None:
    result = domain_pair_cross_entropy(
        logits=((9_999.0, -10_000.0, 10_000.0, -9_999.0),),
        domain_ids=("sonnets",),
        domain_pairs=(DomainExpertPair(domain_id="sonnets", expert_indices=(0, 2)),),
    )

    assert math.isfinite(result.loss)
    assert result.loss == pytest.approx(0.8132616875182228)
    assert result.exact_top2_pair_accuracy == 1.0


def test_domain_mean_reduction_does_not_silently_weight_by_token_count() -> None:
    result = domain_pair_cross_entropy(
        logits=(
            (8.0, 8.0, 0.0, 0.0),
            (8.0, 8.0, 0.0, 0.0),
            (8.0, 8.0, 0.0, 0.0),
            (0.0, 0.0, 8.0, 8.0),
        ),
        domain_ids=("majority", "majority", "majority", "minority"),
        domain_pairs={"majority": (0, 1), "minority": (0, 1)},
    )

    by_domain = {item.domain_id: item for item in result.per_domain}
    expected_domain_mean = (
        by_domain["majority"].mean_cross_entropy + by_domain["minority"].mean_cross_entropy
    ) / 2.0
    assert result.loss == pytest.approx(expected_domain_mean)
    assert result.loss != pytest.approx(result.token_mean_cross_entropy)
    assert result.exact_top2_pair_accuracy == 0.75


def test_mapping_and_record_inputs_are_canonical_and_equivalent() -> None:
    logits = ((4.0, 3.0, 2.0), (0.0, 2.0, 3.0))
    domains = ("b", "a")
    mapping_result = domain_pair_cross_entropy(logits, domains, {"b": (0, 1), "a": (1, 2)})
    record_result = domain_pair_cross_entropy(
        logits,
        domains,
        (
            DomainExpertPair(domain_id="b", expert_indices=(0, 1)),
            DomainExpertPair(domain_id="a", expert_indices=(1, 2)),
        ),
    )

    assert mapping_result == record_result
    assert tuple(item.domain_id for item in mapping_result.per_domain) == ("a", "b")
    assert mapping_result.exact_top2_pair_accuracy == 1.0


def test_objective_result_is_frozen_strict_and_count_validated() -> None:
    result = domain_pair_cross_entropy(((1.0, 0.0),), ("a",), {"a": (0, 1)})
    with pytest.raises(ValidationError):
        result.loss = 0.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        DomainPairObjectiveResult.model_validate({**result.model_dump(), "extra": True})
    payload = result.model_dump()
    payload["token_count"] = 2
    with pytest.raises(ValidationError, match="token count"):
        DomainPairObjectiveResult.model_validate(payload)


@pytest.mark.parametrize(
    ("logits", "domains", "pairs", "match"),
    (
        ((), (), {"a": (0, 1)}, "at least one"),
        (((1.0, 0.0),), (), {"a": (0, 1)}, "same token count"),
        (((1.0, 0.0),), ("a",), {}, "at least one domain pair"),
        (((1.0,),), ("a",), {"a": (0, 1)}, "at least two experts"),
        (((1.0, 0.0), (1.0, 0.0, -1.0)), ("a", "a"), {"a": (0, 1)}, "same expert"),
        (((math.nan, 0.0),), ("a",), {"a": (0, 1)}, "must be finite"),
        (((math.inf, 0.0),), ("a",), {"a": (0, 1)}, "must be finite"),
        (((1.0, 0.0),), ("missing",), {"a": (0, 1)}, "coverage differs"),
        (((1.0, 0.0),), ("a",), {"a": (0, 1), "unused": (0, 1)}, "coverage differs"),
        (((1.0, 0.0),), ("a",), {"a": (0, 2)}, "outside"),
    ),
)
def test_objective_rejects_incomplete_or_invalid_inputs(
    logits: tuple[tuple[float, ...], ...],
    domains: tuple[str, ...],
    pairs: dict[str, tuple[int, int]],
    match: str,
) -> None:
    with pytest.raises(RouterObjectiveError, match=match):
        domain_pair_cross_entropy(logits, domains, pairs)


@pytest.mark.parametrize("pair", ((0, 0), (2, 1), (-1, 1)))
def test_domain_pair_requires_two_distinct_canonically_sorted_experts(
    pair: tuple[int, int],
) -> None:
    with pytest.raises(ValidationError):
        DomainExpertPair(domain_id="domain", expert_indices=pair)


def test_duplicate_domain_pair_records_are_rejected() -> None:
    pairs = (
        DomainExpertPair(domain_id="same", expert_indices=(0, 1)),
        DomainExpertPair(domain_id="same", expert_indices=(1, 2)),
    )
    with pytest.raises(RouterObjectiveError, match="unique"):
        domain_pair_cross_entropy(((1.0, 0.0, 0.0),), ("same",), pairs)


def test_extreme_finite_logits_fail_if_the_reduced_loss_overflows() -> None:
    with pytest.raises(RouterObjectiveError, match=r"finite|overflowed"):
        domain_pair_cross_entropy(
            ((1e308, -1e308, -1e308),),
            ("domain",),
            {"domain": (1, 2)},
        )
