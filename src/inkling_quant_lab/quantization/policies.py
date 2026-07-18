"""Deterministic precision-policy resolution for quantizable module inventories."""

from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Literal, NoReturn, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from inkling_quant_lab.config import PolicyRuleConfig, Precision, PrecisionPolicyConfig
from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.models.base import ModuleInfo

AssignmentSource: TypeAlias = Literal[
    "explicit", "invariant", "module_rule", "expert_tier", "default"
]
DecisionAction: TypeAlias = Literal["assign", "budget_adjust"]

_PRECISION_BITS: dict[Precision, int] = {
    "float32": 32,
    "float16": 16,
    "bfloat16": 16,
    "fp8": 8,
    "int8": 8,
    "int4": 4,
}

# This order is only a deterministic tie-break for formats with equal storage.
# Backends still decide whether a precision is actually supported per module.
_PRECISION_PREFERENCE: dict[Precision, int] = {
    "int4": 0,
    "int8": 1,
    "fp8": 2,
    "float16": 3,
    "bfloat16": 4,
    "float32": 5,
}


class _ImmutablePolicyModel(BaseModel):
    """Strict immutable base for persisted policy decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PrecisionAssignment(_ImmutablePolicyModel):
    """One resolved precision with its provenance and size estimate."""

    module_name: str
    precision: Precision
    source: AssignmentSource
    rule_name: str
    precedence: int = Field(ge=1, le=5)
    estimated_size_bytes: int = Field(ge=0)
    adjusted_for_budget: bool = False


class PolicyDecision(_ImmutablePolicyModel):
    """One deterministic, human-readable resolution-log entry."""

    module_name: str
    action: DecisionAction
    precision: Precision
    source: AssignmentSource
    rule_name: str
    reason: str
    previous_precision: Precision | None = None


class ResolvedPrecisionPolicy(_ImmutablePolicyModel):
    """Complete module assignment map produced before quantization."""

    assignments: dict[str, PrecisionAssignment]
    estimated_size_bytes: int = Field(ge=0)
    estimated_budget_error: float = Field(ge=0.0)
    resolution_log: tuple[PolicyDecision, ...]

    @property
    def precision_map(self) -> dict[str, str]:
        """Return the quantizer-facing module-to-precision map in stable order."""

        result: dict[str, str] = {}
        for name in sorted(self.assignments):
            result[name] = self.assignments[name].precision
        return result

    def canonical_json(self) -> str:
        """Return a byte-stable serialized policy suitable for artifacts and hashing."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )


@dataclass(frozen=True, slots=True)
class _Selection:
    precision: Precision
    source: AssignmentSource
    rule_name: str
    precedence: int
    reason: str
    importance: float


@dataclass(frozen=True, slots=True)
class _TierChoice:
    precision: Precision
    importance: float
    reason: str


def resolve_precision_policy(
    modules: Sequence[ModuleInfo],
    config: PrecisionPolicyConfig,
    *,
    usage_statistics: Mapping[str, float] | None = None,
    sensitivity_statistics: Mapping[str, float] | None = None,
) -> ResolvedPrecisionPolicy:
    """Resolve a complete precision map using documented rule precedence.

    Statistics are keyed by normalized module name. Frequency ranks are computed
    independently within each ``layer_id``. Rules use shell-style, case-sensitive
    patterns: class rules match ``class_name`` and layer rules match ``layer_id``.
    """

    ordered_modules = _validated_inventory(modules)
    known_names = {module.name for module in ordered_modules}
    unknown_overrides = sorted(set(config.explicit_overrides).difference(known_names))
    if unknown_overrides:
        names = ", ".join(unknown_overrides)
        raise ConfigurationError(
            f"precision policy contains overrides for unknown modules: {names}",
            component="precision_policy",
            details={"unknown_modules": unknown_overrides},
        )

    usage = usage_statistics or {}
    sensitivity = sensitivity_statistics or {}
    frequency_choices = _frequency_choices(ordered_modules, config, usage)

    assignments: dict[str, PrecisionAssignment] = {}
    importance: dict[str, float] = {}
    decisions: list[PolicyDecision] = []

    for module in ordered_modules:
        selection = _select_precision(
            module,
            config,
            frequency_choices=frequency_choices,
            sensitivity_statistics=sensitivity,
        )
        _require_supported(module, selection.precision, selection.rule_name)
        assignment = PrecisionAssignment(
            module_name=module.name,
            precision=selection.precision,
            source=selection.source,
            rule_name=selection.rule_name,
            precedence=selection.precedence,
            estimated_size_bytes=_estimate_size(module, selection.precision),
        )
        assignments[module.name] = assignment
        importance[module.name] = selection.importance
        decisions.append(
            PolicyDecision(
                module_name=module.name,
                action="assign",
                precision=selection.precision,
                source=selection.source,
                rule_name=selection.rule_name,
                reason=selection.reason,
            )
        )

    assignments, budget_decisions = _fit_memory_budget(
        ordered_modules,
        assignments,
        importance,
        config,
    )
    decisions.extend(budget_decisions)
    estimated_size = sum(item.estimated_size_bytes for item in assignments.values())
    budget_error = _budget_overage(estimated_size, config.memory_budget_bytes)

    return ResolvedPrecisionPolicy(
        assignments={name: assignments[name] for name in sorted(assignments)},
        estimated_size_bytes=estimated_size,
        estimated_budget_error=budget_error,
        resolution_log=tuple(decisions),
    )


def _validated_inventory(modules: Sequence[ModuleInfo]) -> tuple[ModuleInfo, ...]:
    ordered = tuple(sorted(modules, key=lambda item: item.name))
    duplicates = sorted(
        name
        for name in {module.name for module in ordered}
        if sum(module.name == name for module in ordered) > 1
    )
    if duplicates:
        names = ", ".join(duplicates)
        raise ConfigurationError(
            f"module inventory contains duplicate names: {names}",
            component="precision_policy",
            details={"duplicate_modules": duplicates},
        )
    return ordered


def _select_precision(
    module: ModuleInfo,
    config: PrecisionPolicyConfig,
    *,
    frequency_choices: Mapping[str, _TierChoice],
    sensitivity_statistics: Mapping[str, float],
) -> _Selection:
    explicit = config.explicit_overrides.get(module.name)
    if explicit is not None:
        return _Selection(
            precision=explicit,
            source="explicit",
            rule_name=f"explicit:{module.name}",
            precedence=5,
            reason="exact module-name override",
            importance=math.inf,
        )

    invariant = _invariant_selection(module, config)
    if invariant is not None:
        return invariant

    rule_selection = _module_rule_selection(module, config)
    if rule_selection is not None:
        return rule_selection

    if module.is_expert and config.type != "uniform":
        return _expert_tier_selection(
            module,
            config,
            frequency_choices=frequency_choices,
            sensitivity_statistics=sensitivity_statistics,
        )

    return _Selection(
        precision=config.default_precision,
        source="default",
        rule_name="default",
        precedence=1,
        reason="policy default precision",
        importance=0.0,
    )


def _invariant_selection(module: ModuleInfo, config: PrecisionPolicyConfig) -> _Selection | None:
    matches: list[tuple[str, Precision]] = []
    if module.is_router and config.preserve_router_precision:
        matches.append(("preserve_router_precision", config.router_precision))
    if module.is_output_head and config.preserve_output_head:
        matches.append(("preserve_output_head", config.output_head_precision))
    if module.is_embedding and config.preserve_embeddings:
        matches.append(("preserve_embeddings", config.embedding_precision))
    if module.is_multimodal_projector and config.preserve_multimodal_projectors:
        matches.append(("preserve_multimodal_projectors", config.multimodal_projector_precision))

    if not matches:
        return None
    precisions = {precision for _, precision in matches}
    if len(precisions) > 1:
        _raise_rule_conflict(module.name, 4, [name for name, _ in matches])

    names = sorted(name for name, _ in matches)
    return _Selection(
        precision=matches[0][1],
        source="invariant",
        rule_name="+".join(names),
        precedence=4,
        reason="protected-module correctness invariant",
        importance=math.inf,
    )


def _module_rule_selection(module: ModuleInfo, config: PrecisionPolicyConfig) -> _Selection | None:
    matches: list[tuple[str, PolicyRuleConfig]] = []
    for rule in config.module_class_rules:
        if fnmatchcase(module.class_name, rule.pattern):
            matches.append((f"module_class:{rule.name}", rule))
    if module.layer_id is not None:
        for rule in config.layer_rules:
            if fnmatchcase(module.layer_id, rule.pattern):
                matches.append((f"layer:{rule.name}", rule))

    if not matches:
        return None
    if len(matches) > 1:
        _raise_rule_conflict(module.name, 3, [name for name, _ in matches])

    display_name, rule = matches[0]
    return _Selection(
        precision=rule.precision,
        source="module_rule",
        rule_name=display_name,
        precedence=3,
        reason=f"matched {display_name} pattern {rule.pattern!r}",
        importance=math.inf,
    )


def _expert_tier_selection(
    module: ModuleInfo,
    config: PrecisionPolicyConfig,
    *,
    frequency_choices: Mapping[str, _TierChoice],
    sensitivity_statistics: Mapping[str, float],
) -> _Selection:
    frequency: _TierChoice | None = None
    sensitivity: _TierChoice | None = None

    if config.type in {"frequency_tiered", "hybrid"}:
        frequency = frequency_choices.get(module.name)
        if frequency is None:
            frequency_config = config.frequency
            if frequency_config is None:  # guarded by PrecisionPolicyConfig validation
                raise AssertionError("frequency configuration is required")
            fallback = frequency_config.fallback_precision
            if fallback is None:
                _raise_missing_statistic(module.name, "usage")
            frequency = _TierChoice(
                precision=fallback,
                importance=0.0,
                reason="missing usage statistic; configured frequency fallback",
            )

    if config.type in {"sensitivity_tiered", "hybrid"}:
        sensitivity_config = config.sensitivity
        if sensitivity_config is None:  # guarded by PrecisionPolicyConfig validation
            raise AssertionError("sensitivity configuration is required")
        value = _optional_statistic(sensitivity_statistics, module.name, "sensitivity")
        if value is None:
            fallback = sensitivity_config.fallback_precision
            if fallback is None:
                _raise_missing_statistic(module.name, "sensitivity")
            sensitivity = _TierChoice(
                precision=fallback,
                importance=0.0,
                reason="missing sensitivity statistic; configured sensitivity fallback",
            )
        else:
            tier = bisect_right(sensitivity_config.thresholds, value)
            sensitivity = _TierChoice(
                precision=sensitivity_config.precisions[tier],
                importance=value,
                reason=(
                    f"sensitivity={value:.17g}, method={sensitivity_config.method}, tier={tier}"
                ),
            )

    if config.type == "frequency_tiered" and frequency is not None:
        choice = frequency
        name = "frequency_tier"
    elif config.type == "sensitivity_tiered" and sensitivity is not None:
        choice = sensitivity
        name = "sensitivity_tier"
    elif config.type == "hybrid" and frequency is not None and sensitivity is not None:
        choice = max(
            (frequency, sensitivity),
            key=lambda item: (
                _PRECISION_BITS[item.precision],
                _PRECISION_PREFERENCE[item.precision],
            ),
        )
        choice = _TierChoice(
            precision=choice.precision,
            importance=frequency.importance + sensitivity.importance,
            reason=f"hybrid maximum-protection choice ({frequency.reason}; {sensitivity.reason})",
        )
        name = "hybrid_tier"
    else:  # the caller excludes uniform and the config validator excludes missing settings
        raise AssertionError(f"unsupported expert policy type: {config.type}")

    return _Selection(
        precision=choice.precision,
        source="expert_tier",
        rule_name=name,
        precedence=2,
        reason=choice.reason,
        importance=choice.importance,
    )


def _frequency_choices(
    modules: Sequence[ModuleInfo],
    config: PrecisionPolicyConfig,
    statistics: Mapping[str, float],
) -> dict[str, _TierChoice]:
    if config.type not in {"frequency_tiered", "hybrid"}:
        return {}
    frequency_config = config.frequency
    if frequency_config is None:  # guarded by PrecisionPolicyConfig validation
        raise AssertionError("frequency configuration is required")

    layers: dict[str, list[tuple[ModuleInfo, float]]] = {}
    for module in modules:
        if not module.is_expert:
            continue
        count = _optional_statistic(statistics, module.name, "usage")
        if count is None:
            continue
        layer = module.layer_id if module.layer_id is not None else "<unassigned>"
        layers.setdefault(layer, []).append((module, count))

    result: dict[str, _TierChoice] = {}
    for layer_id in sorted(layers):
        experts = sorted(layers[layer_id], key=lambda item: (item[1], item[0].name))
        total = math.fsum(count for _, count in experts)
        expert_count = len(experts)
        for rank, (module, count) in enumerate(experts, start=1):
            quantile = rank / expert_count
            tier = bisect_left(frequency_config.quantiles, quantile)
            normalized = count / total if total > 0.0 else 0.0
            result[module.name] = _TierChoice(
                precision=frequency_config.precisions[tier],
                importance=normalized,
                reason=(
                    f"layer={layer_id}, normalized_usage={normalized:.17g}, "
                    f"quantile_rank={quantile:.17g}, tier={tier}"
                ),
            )
    return result


def _optional_statistic(
    statistics: Mapping[str, float], module_name: str, statistic_name: str
) -> float | None:
    if module_name not in statistics:
        return None
    value = float(statistics[module_name])
    if not math.isfinite(value) or value < 0.0:
        raise ConfigurationError(
            f"{statistic_name} statistic for module {module_name!r} "
            "must be finite and non-negative",
            component="precision_policy",
            details={"module": module_name, "statistic": statistic_name, "value": value},
        )
    return value


def _raise_missing_statistic(module_name: str, statistic_name: str) -> NoReturn:
    raise ConfigurationError(
        f"expert module {module_name!r} is missing required {statistic_name} statistics "
        "and no fallback precision is configured",
        component="precision_policy",
        remediation=(
            f"provide complete {statistic_name} statistics or configure a fallback precision"
        ),
        details={"module": module_name, "statistic": statistic_name},
    )


def _raise_rule_conflict(module_name: str, precedence: int, rule_names: list[str]) -> None:
    names = sorted(rule_names)
    raise ConfigurationError(
        f"equal-precedence precision rules conflict for module {module_name!r}: "
        + ", ".join(names),
        component="precision_policy",
        details={"module": module_name, "precedence": precedence, "rules": names},
    )


def _require_supported(module: ModuleInfo, precision: Precision, rule_name: str) -> None:
    if precision in module.supported_precisions:
        return
    supported = ", ".join(module.supported_precisions) or "none"
    raise ConfigurationError(
        f"policy rule {rule_name!r} assigns unsupported precision {precision!r} "
        f"to module {module.name!r}; supported precisions: {supported}",
        component="precision_policy",
        details={
            "module": module.name,
            "precision": precision,
            "supported_precisions": list(module.supported_precisions),
            "rule": rule_name,
        },
    )


def _estimate_size(module: ModuleInfo, precision: Precision) -> int:
    """Scale the inventory's full-precision byte size to the selected bit width."""

    return math.ceil(module.size_bytes * _PRECISION_BITS[precision] / 32)


def _fit_memory_budget(
    modules: Sequence[ModuleInfo],
    assignments: Mapping[str, PrecisionAssignment],
    importance: Mapping[str, float],
    config: PrecisionPolicyConfig,
) -> tuple[dict[str, PrecisionAssignment], list[PolicyDecision]]:
    budget = config.memory_budget_bytes
    resolved = dict(assignments)
    if budget is None:
        return resolved, []

    allowed = budget * (1.0 + config.memory_budget_tolerance)
    module_by_name = {module.name: module for module in modules}
    decisions: list[PolicyDecision] = []

    while sum(item.estimated_size_bytes for item in resolved.values()) > allowed:
        candidates: list[tuple[float, str, Precision]] = []
        for name in sorted(resolved):
            assignment = resolved[name]
            if assignment.source not in {"default", "expert_tier"}:
                continue
            lower = _next_lower_precision(
                assignment.precision, module_by_name[name].supported_precisions
            )
            if lower is not None:
                candidates.append((importance[name], name, lower))

        if not candidates:
            current = sum(item.estimated_size_bytes for item in resolved.values())
            raise ConfigurationError(
                f"memory budget is infeasible: estimated {current} bytes exceeds "
                f"allowed {allowed:.17g} bytes and no eligible assignment can be lowered",
                component="precision_policy",
                remediation="increase the memory budget or permit lower module precisions",
                details={
                    "budget_bytes": budget,
                    "tolerance": config.memory_budget_tolerance,
                    "estimated_size_bytes": current,
                },
            )

        _, name, lower = min(candidates, key=lambda item: (item[0], item[1]))
        previous = resolved[name]
        updated = previous.model_copy(
            update={
                "precision": lower,
                "estimated_size_bytes": _estimate_size(module_by_name[name], lower),
                "adjusted_for_budget": True,
            }
        )
        resolved[name] = updated
        decisions.append(
            PolicyDecision(
                module_name=name,
                action="budget_adjust",
                precision=lower,
                previous_precision=previous.precision,
                source=previous.source,
                rule_name=previous.rule_name,
                reason="lowered the least-impact eligible assignment to satisfy memory budget",
            )
        )

    return resolved, decisions


def _next_lower_precision(current: Precision, supported: Sequence[Precision]) -> Precision | None:
    current_bits = _PRECISION_BITS[current]
    lower = [precision for precision in supported if _PRECISION_BITS[precision] < current_bits]
    if not lower:
        return None
    return max(lower, key=_precision_sort_key)


def _precision_sort_key(precision: Precision) -> tuple[int, int]:
    return _PRECISION_BITS[precision], _PRECISION_PREFERENCE[precision]


def _budget_overage(estimated_size: int, budget: int | None) -> float:
    if budget is None or estimated_size <= budget:
        return 0.0
    return (estimated_size - budget) / budget
