"""Backend-independent reference objective for domain-conditioned top-2 routing."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RouterObjectiveError(ValueError):
    """Reference routing objective inputs are incomplete or numerically invalid."""


class ImmutableObjectiveRecord(BaseModel):
    """Strict immutable record for deterministic objective inputs and outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DomainExpertPair(ImmutableObjectiveRecord):
    """Canonical unordered pair of experts assigned to one domain."""

    domain_id: str = Field(min_length=1)
    expert_indices: tuple[int, int]

    @model_validator(mode="after")
    def pair_is_canonical(self) -> Self:
        left, right = self.expert_indices
        if left < 0 or right < 0:
            raise ValueError("expert indices must be non-negative")
        if left >= right:
            raise ValueError("domain expert pair must contain two distinct sorted indices")
        return self


class DomainRoutingResult(ImmutableObjectiveRecord):
    """Reference loss and exact unordered top-2 accuracy for one domain."""

    domain_id: str = Field(min_length=1)
    target_expert_indices: tuple[int, int]
    token_count: int = Field(ge=1)
    mean_cross_entropy: float = Field(ge=0.0)
    exact_top2_pair_accuracy: float = Field(ge=0.0, le=1.0)


class DomainPairObjectiveResult(ImmutableObjectiveRecord):
    """Stable soft-target cross-entropy with domain-balanced reduction."""

    objective_name: Literal["domain_pair_soft_target_cross_entropy"] = (
        "domain_pair_soft_target_cross_entropy"
    )
    objective_version: Literal["top2-domain-pair-ce-v1"] = "top2-domain-pair-ce-v1"
    reduction: Literal["domain_mean"] = "domain_mean"
    token_count: int = Field(ge=1)
    expert_count: int = Field(ge=2)
    domain_count: int = Field(ge=1)
    loss: float = Field(ge=0.0)
    token_mean_cross_entropy: float = Field(ge=0.0)
    exact_top2_pair_accuracy: float = Field(ge=0.0, le=1.0)
    per_domain: tuple[DomainRoutingResult, ...]

    @model_validator(mode="after")
    def aggregate_counts_are_consistent(self) -> Self:
        if len(self.per_domain) != self.domain_count:
            raise ValueError("domain count differs from per-domain objective records")
        if sum(item.token_count for item in self.per_domain) != self.token_count:
            raise ValueError("token count differs from per-domain objective records")
        if tuple(item.domain_id for item in self.per_domain) != tuple(
            sorted(item.domain_id for item in self.per_domain)
        ):
            raise ValueError("per-domain objective records must be canonically ordered")
        return self


def _stable_pair_cross_entropy(logits: tuple[float, ...], pair: tuple[int, int]) -> float:
    maximum = max(logits)
    log_partition = maximum + math.log(math.fsum(math.exp(value - maximum) for value in logits))
    target_mean_logit = 0.5 * logits[pair[0]] + 0.5 * logits[pair[1]]
    loss = log_partition - target_mean_logit
    if not math.isfinite(loss) or loss < 0.0:
        raise RouterObjectiveError("domain-pair cross-entropy was not finite and non-negative")
    return loss


def _top2_pair(logits: tuple[float, ...]) -> tuple[int, int]:
    selected = sorted(range(len(logits)), key=lambda index: (-logits[index], index))[:2]
    return tuple(sorted(selected))  # type: ignore[return-value]


def domain_pair_cross_entropy(
    logits: Sequence[Sequence[float]],
    domain_ids: Sequence[str],
    domain_pairs: Sequence[DomainExpertPair] | Mapping[str, tuple[int, int]],
) -> DomainPairObjectiveResult:
    """Compute the domain-balanced reference objective for unordered top-2 targets.

    Each token uses a uniform soft target over its domain's two experts. Stable
    log-sum-exp avoids overflow, and ties in exact top-2 accuracy prefer the
    lower expert index. The returned optimization loss is the mean of per-domain
    means so corpus-size imbalance does not silently reweight domains.
    """

    if not logits:
        raise RouterObjectiveError("routing objective requires at least one logits row")
    if len(logits) != len(domain_ids):
        raise RouterObjectiveError("logits and domain IDs must have the same token count")
    if isinstance(domain_pairs, Mapping):
        try:
            pairs = tuple(
                DomainExpertPair(domain_id=domain, expert_indices=pair)
                for domain, pair in sorted(domain_pairs.items())
            )
        except ValueError as error:
            raise RouterObjectiveError(f"invalid domain-pair mapping: {error}") from error
    else:
        pairs = tuple(domain_pairs)
    if not pairs:
        raise RouterObjectiveError("routing objective requires at least one domain pair")
    pair_by_domain = {item.domain_id: item.expert_indices for item in pairs}
    if len(pair_by_domain) != len(pairs):
        raise RouterObjectiveError("domain pair IDs must be unique")

    normalized_logits: list[tuple[float, ...]] = []
    expert_count: int | None = None
    for row in logits:
        try:
            values = tuple(float(value) for value in row)
        except (TypeError, ValueError) as error:
            raise RouterObjectiveError("router logits must contain only numeric values") from error
        if expert_count is None:
            expert_count = len(values)
            if expert_count < 2:
                raise RouterObjectiveError("routing objective requires at least two experts")
        if len(values) != expert_count:
            raise RouterObjectiveError("all router-logit rows must have the same expert count")
        if any(not math.isfinite(value) for value in values):
            raise RouterObjectiveError("router logits must be finite")
        normalized_logits.append(values)
    assert expert_count is not None

    observed_domains = set(domain_ids)
    configured_domains = set(pair_by_domain)
    if observed_domains != configured_domains:
        missing = sorted(observed_domains - configured_domains)
        unused = sorted(configured_domains - observed_domains)
        raise RouterObjectiveError(
            f"domain-pair coverage differs; missing={missing!r}, unused={unused!r}"
        )
    for domain, pair in pair_by_domain.items():
        if pair[1] >= expert_count:
            raise RouterObjectiveError(
                f"domain {domain!r} references expert {pair[1]} outside {expert_count} experts"
            )

    losses_by_domain: defaultdict[str, list[float]] = defaultdict(list)
    correct_by_domain: defaultdict[str, int] = defaultdict(int)
    all_losses: list[float] = []
    total_correct = 0
    for values, domain in zip(normalized_logits, domain_ids, strict=True):
        pair = pair_by_domain[domain]
        loss = _stable_pair_cross_entropy(values, pair)
        correct = _top2_pair(values) == pair
        losses_by_domain[domain].append(loss)
        correct_by_domain[domain] += int(correct)
        all_losses.append(loss)
        total_correct += int(correct)

    per_domain = tuple(
        DomainRoutingResult(
            domain_id=domain,
            target_expert_indices=pair_by_domain[domain],
            token_count=len(losses_by_domain[domain]),
            mean_cross_entropy=math.fsum(losses_by_domain[domain]) / len(losses_by_domain[domain]),
            exact_top2_pair_accuracy=correct_by_domain[domain] / len(losses_by_domain[domain]),
        )
        for domain in sorted(pair_by_domain)
    )
    domain_mean = math.fsum(item.mean_cross_entropy for item in per_domain) / len(per_domain)
    token_mean = math.fsum(all_losses) / len(all_losses)
    if not math.isfinite(domain_mean) or not math.isfinite(token_mean):
        raise RouterObjectiveError("routing objective reduction overflowed")
    return DomainPairObjectiveResult(
        token_count=len(all_losses),
        expert_count=expert_count,
        domain_count=len(per_domain),
        loss=domain_mean,
        token_mean_cross_entropy=token_mean,
        exact_top2_pair_accuracy=total_correct / len(all_losses),
        per_domain=per_domain,
    )


__all__ = [
    "DomainExpertPair",
    "DomainPairObjectiveResult",
    "DomainRoutingResult",
    "RouterObjectiveError",
    "domain_pair_cross_entropy",
]
