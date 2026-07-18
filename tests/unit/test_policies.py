"""Focused tests for deterministic precision-policy resolution."""

from __future__ import annotations

import json

import pytest

from inkling_quant_lab.config import (
    FrequencyTierConfig,
    PolicyRuleConfig,
    Precision,
    PrecisionPolicyConfig,
    SensitivityTierConfig,
)
from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.models.base import ModuleInfo
from inkling_quant_lab.quantization.policies import resolve_precision_policy

pytestmark = pytest.mark.unit


def _module(
    name: str,
    *,
    class_name: str = "Linear",
    layer_id: str | None = None,
    is_router: bool = False,
    is_output_head: bool = False,
    is_expert: bool = False,
    size_bytes: int = 400,
    supported_precisions: tuple[Precision, ...] = (
        "float32",
        "float16",
        "bfloat16",
        "int8",
        "int4",
    ),
) -> ModuleInfo:
    return ModuleInfo(
        name=name,
        class_name=class_name,
        parameter_count=size_bytes // 4,
        size_bytes=size_bytes,
        is_router=is_router,
        is_output_head=is_output_head,
        is_expert=is_expert,
        layer_id=layer_id,
        supported_precisions=supported_precisions,
    )


def test_rule_precedence_is_complete_and_documented() -> None:
    """TC-POLICY-001: explicit > invariant > class/layer > tier > default."""

    modules = (
        _module(
            "block.7.router",
            class_name="Projection",
            layer_id="block.7",
            is_router=True,
        ),
        _module("lm_head", class_name="Projection", is_output_head=True),
        _module("block.1.proj", class_name="Projection", layer_id="block.1"),
        _module("block.7.norm", class_name="Norm", layer_id="block.7"),
        _module("block.2.expert.0", class_name="Expert", layer_id="block.2", is_expert=True),
        _module("embed", class_name="Embedding"),
    )
    config = PrecisionPolicyConfig(
        type="frequency_tiered",
        default_precision="float16",
        explicit_overrides={"block.7.router": "int8"},
        module_class_rules=(
            PolicyRuleConfig(name="projection", pattern="Projection", precision="int8"),
        ),
        layer_rules=(PolicyRuleConfig(name="block-seven", pattern="block.7", precision="int4"),),
        router_precision="float32",
        output_head_precision="float32",
        frequency=FrequencyTierConfig(quantiles=(), precisions=("int8",)),
    )

    resolved = resolve_precision_policy(
        modules,
        config,
        usage_statistics={"block.2.expert.0": 10.0},
    )

    assert resolved.precision_map == {
        "block.1.proj": "int8",
        "block.2.expert.0": "int8",
        "block.7.norm": "int4",
        "block.7.router": "int8",
        "embed": "float16",
        "lm_head": "float32",
    }
    assert resolved.assignments["block.7.router"].source == "explicit"
    assert resolved.assignments["lm_head"].source == "invariant"
    assert resolved.assignments["block.1.proj"].source == "module_rule"
    assert resolved.assignments["block.7.norm"].source == "module_rule"
    assert resolved.assignments["block.2.expert.0"].source == "expert_tier"
    assert resolved.assignments["embed"].source == "default"


def test_equal_precedence_conflict_names_module_and_both_rules() -> None:
    """TC-POLICY-002: class and layer matches share a conflict level."""

    config = PrecisionPolicyConfig(
        module_class_rules=(
            PolicyRuleConfig(name="all-linear", pattern="Linear", precision="int8"),
        ),
        layer_rules=(PolicyRuleConfig(name="first-layer", pattern="block.0", precision="int4"),),
    )

    with pytest.raises(ConfigurationError) as captured:
        resolve_precision_policy((_module("block.0.proj", layer_id="block.0"),), config)

    message = str(captured.value)
    assert "block.0.proj" in message
    assert "all-linear" in message
    assert "first-layer" in message
    assert captured.value.details == {
        "module": "block.0.proj",
        "precedence": 3,
        "rules": ["layer:first-layer", "module_class:all-linear"],
    }


def test_conflicting_router_and_output_invariants_are_rejected() -> None:
    module = _module("shared_head", is_router=True, is_output_head=True)
    config = PrecisionPolicyConfig(
        router_precision="float32",
        output_head_precision="float16",
    )

    with pytest.raises(ConfigurationError) as captured:
        resolve_precision_policy((module,), config)

    assert captured.value.details == {
        "module": "shared_head",
        "precedence": 4,
        "rules": ["preserve_output_head", "preserve_router_precision"],
    }


def test_unknown_explicit_override_is_not_silently_ignored() -> None:
    config = PrecisionPolicyConfig(explicit_overrides={"missing.module": "int8"})

    with pytest.raises(ConfigurationError, match=r"missing\.module") as captured:
        resolve_precision_policy((_module("present.module"),), config)

    assert captured.value.details["unknown_modules"] == ["missing.module"]


def test_frequency_tiers_are_ranked_within_each_layer() -> None:
    """TC-POLICY-003: known counts map to deterministic per-layer quantiles."""

    modules = tuple(
        _module(f"layer.0.expert.{expert}", layer_id="layer.0", is_expert=True)
        for expert in range(4)
    ) + tuple(
        _module(f"layer.1.expert.{expert}", layer_id="layer.1", is_expert=True)
        for expert in range(2)
    )
    config = PrecisionPolicyConfig(
        type="frequency_tiered",
        frequency=FrequencyTierConfig(quantiles=(0.5,), precisions=("int4", "int8")),
    )
    usage = {
        "layer.0.expert.0": 10.0,
        "layer.0.expert.1": 20.0,
        "layer.0.expert.2": 30.0,
        "layer.0.expert.3": 40.0,
        "layer.1.expert.0": 1_000.0,
        "layer.1.expert.1": 2_000.0,
    }

    resolved = resolve_precision_policy(tuple(reversed(modules)), config, usage_statistics=usage)

    assert resolved.precision_map == {
        "layer.0.expert.0": "int4",
        "layer.0.expert.1": "int4",
        "layer.0.expert.2": "int8",
        "layer.0.expert.3": "int8",
        "layer.1.expert.0": "int4",
        "layer.1.expert.1": "int8",
    }


def test_frequency_missing_statistics_fail_without_fallback() -> None:
    """TC-POLICY-004: required expert usage statistics fail loudly."""

    module = _module("layer.0.expert.0", layer_id="layer.0", is_expert=True)
    config = PrecisionPolicyConfig(
        type="frequency_tiered",
        frequency=FrequencyTierConfig(fallback_precision=None),
    )

    with pytest.raises(ConfigurationError, match=r"layer\.0\.expert\.0.*usage"):
        resolve_precision_policy((module,), config)


def test_frequency_missing_statistics_use_configured_fallback() -> None:
    module = _module("layer.0.expert.0", layer_id="layer.0", is_expert=True)
    config = PrecisionPolicyConfig(
        type="frequency_tiered",
        frequency=FrequencyTierConfig(fallback_precision="bfloat16"),
    )

    resolved = resolve_precision_policy((module,), config)

    assert resolved.precision_map == {"layer.0.expert.0": "bfloat16"}
    assert "fallback" in resolved.resolution_log[0].reason


def test_sensitivity_tiers_and_missing_statistic_fallback() -> None:
    modules = tuple(
        _module(f"layer.0.expert.{expert}", layer_id="layer.0", is_expert=True)
        for expert in range(3)
    )
    config = PrecisionPolicyConfig(
        type="sensitivity_tiered",
        sensitivity=SensitivityTierConfig(
            method="loss_impact",
            thresholds=(0.5,),
            precisions=("int4", "int8"),
            fallback_precision="float16",
        ),
    )

    resolved = resolve_precision_policy(
        modules,
        config,
        sensitivity_statistics={
            "layer.0.expert.0": 0.1,
            "layer.0.expert.1": 0.5,
        },
    )

    assert resolved.precision_map == {
        "layer.0.expert.0": "int4",
        "layer.0.expert.1": "int8",
        "layer.0.expert.2": "float16",
    }


def test_sensitivity_missing_statistics_fail_without_fallback() -> None:
    module = _module("layer.0.expert.0", layer_id="layer.0", is_expert=True)
    config = PrecisionPolicyConfig(
        type="sensitivity_tiered",
        sensitivity=SensitivityTierConfig(fallback_precision=None),
    )

    with pytest.raises(ConfigurationError, match="sensitivity"):
        resolve_precision_policy((module,), config)


def test_hybrid_uses_maximum_protection_and_independent_fallbacks() -> None:
    modules = tuple(
        _module(f"layer.0.expert.{expert}", layer_id="layer.0", is_expert=True)
        for expert in range(3)
    )
    config = PrecisionPolicyConfig(
        type="hybrid",
        frequency=FrequencyTierConfig(
            quantiles=(0.5,),
            precisions=("int4", "int8"),
            fallback_precision="int8",
        ),
        sensitivity=SensitivityTierConfig(
            thresholds=(0.5,),
            precisions=("int4", "float16"),
            fallback_precision="float16",
        ),
    )

    resolved = resolve_precision_policy(
        modules,
        config,
        usage_statistics={"layer.0.expert.0": 1.0, "layer.0.expert.1": 2.0},
        sensitivity_statistics={"layer.0.expert.0": 0.9, "layer.0.expert.1": 0.1},
    )

    assert resolved.precision_map == {
        "layer.0.expert.0": "float16",
        "layer.0.expert.1": "int8",
        "layer.0.expert.2": "float16",
    }


def test_assignment_rejects_unsupported_module_precision() -> None:
    module = _module(
        "proj",
        supported_precisions=("float32", "int8"),
    )
    config = PrecisionPolicyConfig(
        default_precision="int4",
        preserve_router_precision=False,
        preserve_output_head=False,
    )

    with pytest.raises(ConfigurationError) as captured:
        resolve_precision_policy((module,), config)

    assert "proj" in str(captured.value)
    assert "int4" in str(captured.value)
    assert captured.value.details["supported_precisions"] == ["float32", "int8"]


def test_memory_budget_lowers_least_used_expert_deterministically() -> None:
    """TC-POLICY-005/006: budget fitting and serialization are deterministic."""

    modules = tuple(
        _module(
            f"layer.0.expert.{name}",
            layer_id="layer.0",
            is_expert=True,
            supported_precisions=("int8", "int4"),
        )
        for name in ("low", "middle", "high")
    )
    config = PrecisionPolicyConfig(
        type="frequency_tiered",
        frequency=FrequencyTierConfig(
            quantiles=(0.5,),
            precisions=("int8", "int8"),
        ),
        memory_budget_bytes=250,
        memory_budget_tolerance=0.0,
    )
    usage = {
        "layer.0.expert.low": 1.0,
        "layer.0.expert.middle": 2.0,
        "layer.0.expert.high": 3.0,
    }

    first = resolve_precision_policy(modules, config, usage_statistics=usage)
    second = resolve_precision_policy(
        tuple(reversed(modules)),
        config,
        usage_statistics=dict(reversed(tuple(usage.items()))),
    )

    assert first.precision_map == {
        "layer.0.expert.high": "int8",
        "layer.0.expert.low": "int4",
        "layer.0.expert.middle": "int8",
    }
    assert first.estimated_size_bytes == 250
    assert first.estimated_budget_error == 0.0
    assert first.assignments["layer.0.expert.low"].adjusted_for_budget is True
    assert [
        decision.module_name
        for decision in first.resolution_log
        if decision.action == "budget_adjust"
    ] == ["layer.0.expert.low"]
    assert first.canonical_json() == second.canonical_json()
    assert json.loads(first.canonical_json())["estimated_size_bytes"] == 250


def test_infeasible_budget_does_not_violate_protected_assignments() -> None:
    module = _module("router", is_router=True, supported_precisions=("float32", "int4"))
    config = PrecisionPolicyConfig(
        router_precision="float32",
        memory_budget_bytes=399,
        memory_budget_tolerance=0.0,
    )

    with pytest.raises(ConfigurationError, match="memory budget is infeasible"):
        resolve_precision_policy((module,), config)


def test_infeasible_budget_reports_when_eligible_module_has_no_lower_precision() -> None:
    module = _module("proj", supported_precisions=("int4",))
    config = PrecisionPolicyConfig(
        default_precision="int4",
        memory_budget_bytes=49,
        memory_budget_tolerance=0.0,
    )

    with pytest.raises(ConfigurationError, match="memory budget is infeasible"):
        resolve_precision_policy((module,), config)


def test_budget_tolerance_records_permitted_fractional_overage() -> None:
    module = _module("proj", supported_precisions=("int8", "int4"))
    config = PrecisionPolicyConfig(
        default_precision="int8",
        memory_budget_bytes=95,
        memory_budget_tolerance=0.1,
    )

    resolved = resolve_precision_policy((module,), config)

    assert resolved.estimated_size_bytes == 100
    assert resolved.estimated_budget_error == pytest.approx(5 / 95)
    assert resolved.assignments["proj"].adjusted_for_budget is False


@pytest.mark.parametrize("invalid", [-1.0, float("inf"), float("nan")])
def test_statistics_must_be_finite_and_non_negative(invalid: float) -> None:
    module = _module("layer.0.expert.0", layer_id="layer.0", is_expert=True)
    config = PrecisionPolicyConfig(
        type="frequency_tiered",
        frequency=FrequencyTierConfig(),
    )

    with pytest.raises(ConfigurationError, match="finite and non-negative"):
        resolve_precision_policy(
            (module,),
            config,
            usage_statistics={module.name: invalid},
        )
