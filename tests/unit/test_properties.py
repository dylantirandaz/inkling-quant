"""Bounded property tests required by TDD section 8."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from inkling_quant_lab.comparison import (
    DatasetIdentity,
    MetricValue,
    NormalizedRunSummary,
    ParetoObjective,
    ParetoPoint,
    compare_summaries,
    dominates,
    pareto_frontier,
)
from inkling_quant_lab.config import PrecisionPolicyConfig
from inkling_quant_lab.exceptions import ArtifactIntegrityError, ComparisonCompatibilityError
from inkling_quant_lab.manifests import (
    ModelProvenance,
    RunManifest,
    RunStatus,
    StageError,
    StageStatus,
)
from inkling_quant_lab.models.base import ModuleInfo
from inkling_quant_lab.quantization.policies import resolve_precision_policy
from inkling_quant_lab.routing.metrics import (
    expert_selection_frequency,
    jensen_shannon_divergence,
    routing_entropy,
)

pytestmark = pytest.mark.unit

NOW = datetime(2026, 1, 1, tzinfo=UTC)
COUNT_VECTORS = st.lists(st.integers(min_value=0, max_value=1_000_000), max_size=8)
MANIFEST_ACTIONS = st.sampled_from(
    (
        "start_run",
        "start_stage",
        "finish_stage",
        "fail_stage",
        "skip_stage",
        "unsupported_stage",
        "invalidate_stage",
        "succeed_run",
        "fail_run",
    )
)
PRECISION_BITS = {
    "float32": 32,
    "float16": 16,
    "bfloat16": 16,
    "fp8": 8,
    "int8": 8,
    "int4": 4,
}
PRECISION_ORDER = tuple(PRECISION_BITS)


def _comparison_summary(run_id: str, *, seed: int, digest: str) -> NormalizedRunSummary:
    return NormalizedRunSummary(
        run_id=run_id,
        artifact_path=f"artifacts/{run_id}",
        model_id="fixture://property-model",
        model_revision="fixture-v1",
        datasets=(
            DatasetIdentity(
                dataset_id="fixture://property-data",
                dataset_revision="fixture-v1",
                split="test",
                dataset_sha256=digest,
            ),
        ),
        seed_set=(seed,),
        sample_ids=("sample-1",),
        prompt_template_hash="a" * 64,
        decode_config={"do_sample": False},
        benchmark_protocol_version="cpu-v1",
        hardware_environment={
            "hardware": {"device": "cpu"},
            "runtime": {"backend": "torch_eager_cpu", "device": "cpu"},
        },
        metrics={"quality": MetricValue(value=1.0, category="quality")},
    )


@settings(max_examples=40, deadline=None)
@given(
    st.integers(min_value=0, max_value=2**31 - 2),
    st.binary(min_size=32, max_size=32),
)
def test_distinct_seed_and_exact_dataset_bytes_always_require_named_overrides(
    seed: int, dataset_bytes: bytes
) -> None:
    """Labels cannot mask different RNG contracts or exact dataset content."""

    baseline_digest = dataset_bytes.hex()
    candidate_digest = (bytes((dataset_bytes[0] ^ 1,)) + dataset_bytes[1:]).hex()
    baseline = _comparison_summary("baseline", seed=seed, digest=baseline_digest)
    candidate = _comparison_summary("candidate", seed=seed + 1, digest=candidate_digest)

    with pytest.raises(ComparisonCompatibilityError) as captured:
        compare_summaries(baseline, candidate)
    mismatches = captured.value.details["mismatches"]
    assert isinstance(mismatches, list)
    assert {item["dimension"] for item in mismatches} == {"datasets", "seed_set"}

    result = compare_summaries(
        baseline,
        candidate,
        unsafe_overrides={"datasets", "seed_set"},
    )
    assert result.overridden_dimensions == ("datasets", "seed_set")


@settings(max_examples=80, deadline=None)
@given(COUNT_VECTORS, COUNT_VECTORS, st.integers(min_value=1, max_value=100))
def test_routing_normalization_and_divergence_remain_finite(
    baseline: list[int], candidate: list[int], scale: int
) -> None:
    """Normalized traffic is a distribution and JS divergence is finite and symmetric."""

    for counts in (baseline, candidate):
        frequency = expert_selection_frequency(counts)
        assert len(frequency) == len(counts)
        assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in frequency)
        expected_total = 0.0 if sum(counts) == 0 else 1.0
        assert math.fsum(frequency) == pytest.approx(expected_total, abs=1e-12)
        assert expert_selection_frequency([value * scale for value in counts]) == pytest.approx(
            frequency, abs=1e-12
        )
        normalized_entropy = routing_entropy(counts, normalized=True)
        assert math.isfinite(normalized_entropy)
        assert 0.0 <= normalized_entropy <= 1.0 + 1e-12

    divergence = jensen_shannon_divergence(baseline, candidate)
    reverse = jensen_shannon_divergence(candidate, baseline)
    assert math.isfinite(divergence)
    assert 0.0 <= divergence <= math.log(2.0)
    assert divergence == pytest.approx(reverse, abs=1e-12)
    assert jensen_shannon_divergence(baseline, baseline) == pytest.approx(0.0, abs=1e-12)


@st.composite
def _pareto_cases(draw):
    dimension = draw(st.integers(min_value=1, max_value=4))
    point_count = draw(st.integers(min_value=1, max_value=7))
    vectors = draw(
        st.lists(
            st.lists(
                st.integers(min_value=-100, max_value=100),
                min_size=dimension,
                max_size=dimension,
            ),
            min_size=point_count,
            max_size=point_count,
        )
    )
    directions = draw(
        st.lists(
            st.sampled_from(("maximize", "minimize")),
            min_size=dimension,
            max_size=dimension,
        )
    )
    tolerances = draw(
        st.lists(
            st.integers(min_value=0, max_value=3),
            min_size=dimension,
            max_size=dimension,
        )
    )
    objectives = tuple(
        ParetoObjective(
            metric=f"metric-{index}",
            direction=direction,
            tolerance=float(tolerance),
        )
        for index, (direction, tolerance) in enumerate(zip(directions, tolerances, strict=True))
    )
    points = tuple(
        ParetoPoint(
            point_id=f"point-{index:02d}",
            values={
                objective.metric: float(value)
                for objective, value in zip(objectives, vector, strict=True)
            },
        )
        for index, vector in enumerate(vectors)
    )
    return points, objectives


@settings(max_examples=60, deadline=None)
@given(_pareto_cases())
def test_pareto_frontier_is_order_independent_and_matches_dominance(case) -> None:
    """Membership is permutation invariant and a coordinate-wise ideal is uniquely optimal."""

    points, objectives = case
    result = pareto_frontier(points, objectives)
    reversed_result = pareto_frontier(reversed(points), objectives)

    assert result == reversed_result
    assert tuple(item.point_id for item in result.memberships) == tuple(
        sorted(point.point_id for point in points)
    )
    for point in points:
        expected_dominators = tuple(
            other.point_id
            for other in points
            if other.point_id != point.point_id and dominates(other, point, objectives)
        )
        membership = result.membership_for(point.point_id)
        assert membership.dominated_by == expected_dominators
        assert membership.pareto_optimal is (not expected_dominators)

    ideal_values: dict[str, float] = {}
    for objective in objectives:
        values = [point.values[objective.metric] for point in points]
        if objective.direction == "maximize":
            ideal_values[objective.metric] = max(values) + objective.tolerance + 1.0
        else:
            ideal_values[objective.metric] = min(values) - objective.tolerance - 1.0
    ideal = ParetoPoint(point_id="ideal", values=ideal_values)
    augmented = pareto_frontier((*points, ideal), objectives)

    assert augmented.optimal_ids == ("ideal",)
    assert all("ideal" in augmented.membership_for(point.point_id).dominated_by for point in points)


@st.composite
def _feasible_policy_cases(draw):
    module_count = draw(st.integers(min_value=1, max_value=7))
    modules: list[ModuleInfo] = []
    minimum_size = 0
    full_size = 0
    for index in range(module_count):
        size_bytes = draw(st.integers(min_value=8, max_value=2_048))
        lower_precisions = draw(
            st.sets(
                st.sampled_from(PRECISION_ORDER[1:]),
                min_size=1,
                max_size=len(PRECISION_ORDER) - 1,
            )
        )
        supported = tuple(
            precision
            for precision in PRECISION_ORDER
            if precision == "float32" or precision in lower_precisions
        )
        modules.append(
            ModuleInfo(
                name=f"module.{index:02d}",
                class_name="Linear",
                parameter_count=max(1, size_bytes // 4),
                size_bytes=size_bytes,
                supported_precisions=supported,
            )
        )
        minimum_bits = min(PRECISION_BITS[precision] for precision in supported)
        minimum_size += math.ceil(size_bytes * minimum_bits / 32)
        full_size += size_bytes

    budget = draw(st.integers(min_value=minimum_size, max_value=full_size))
    config = PrecisionPolicyConfig(
        default_precision="float32",
        preserve_router_precision=False,
        preserve_output_head=False,
        preserve_embeddings=False,
        preserve_multimodal_projectors=False,
        memory_budget_bytes=budget,
        memory_budget_tolerance=0.0,
    )
    return tuple(modules), config


@settings(max_examples=60, deadline=None)
@given(_feasible_policy_cases())
def test_policy_assignments_always_respect_module_supported_precisions(case) -> None:
    """Initial and budget-adjusted decisions stay within every backend support set."""

    modules, config = case
    resolved = resolve_precision_policy(modules, config)
    by_name = {module.name: module for module in modules}

    assert set(resolved.assignments) == set(by_name)
    assert resolved.estimated_size_bytes <= config.memory_budget_bytes
    assert resolved.estimated_size_bytes == sum(
        assignment.estimated_size_bytes for assignment in resolved.assignments.values()
    )
    for name, assignment in resolved.assignments.items():
        module = by_name[name]
        assert assignment.precision in module.supported_precisions
        assert assignment.estimated_size_bytes == math.ceil(
            module.size_bytes * PRECISION_BITS[assignment.precision] / 32
        )
    for decision in resolved.resolution_log:
        assert decision.precision in by_name[decision.module_name].supported_precisions
        if decision.previous_precision is not None:
            assert decision.previous_precision in by_name[decision.module_name].supported_precisions

    reordered = resolve_precision_policy(tuple(reversed(modules)), config)
    assert reordered.canonical_json() == resolved.canonical_json()


def _new_manifest(*, required: bool) -> RunManifest:
    return RunManifest(
        run_id="property-run",
        config_hash="a" * 64,
        model=ModelProvenance(id="local://fixture", revision="v1"),
        stages={"stage": {"name": "stage", "required": required}},
    )


def _transition_is_allowed(manifest: RunManifest, action: str) -> bool:
    stage = manifest.stages["stage"]
    if action == "start_run":
        return manifest.status is RunStatus.PENDING
    if action == "start_stage":
        return manifest.status is RunStatus.RUNNING and stage.status in {
            StageStatus.PENDING,
            StageStatus.FAILED,
            StageStatus.INVALIDATED,
        }
    if action in {"finish_stage", "fail_stage"}:
        return stage.status is StageStatus.RUNNING
    if action in {"skip_stage", "unsupported_stage"}:
        return stage.status in {StageStatus.PENDING, StageStatus.RUNNING}
    if action == "invalidate_stage":
        return True
    if action == "succeed_run":
        required_stages_complete = not stage.required or stage.status is StageStatus.SUCCESS
        return manifest.status is RunStatus.RUNNING and required_stages_complete
    if action == "fail_run":
        return manifest.status is RunStatus.RUNNING
    raise AssertionError(f"unknown manifest action: {action}")


def _apply_transition(manifest: RunManifest, action: str, step: int) -> RunManifest:
    if action == "start_run":
        return manifest.start(at=NOW)
    if action == "start_stage":
        return manifest.start_stage("stage", f"fingerprint-{step}", at=NOW)
    if action == "finish_stage":
        return manifest.finish_stage("stage", (), at=NOW)
    if action == "fail_stage":
        return manifest.fail_stage(
            "stage",
            StageError(code="INJECTED", message=f"failure-{step}"),
            at=NOW,
        )
    if action == "skip_stage":
        return manifest.mark_stage(
            "stage", StageStatus.SKIPPED_NOT_REQUIRED, f"skip-{step}", at=NOW
        )
    if action == "unsupported_stage":
        return manifest.mark_stage("stage", StageStatus.UNSUPPORTED, f"unsupported-{step}", at=NOW)
    if action == "invalidate_stage":
        return manifest.invalidate({"stage"})
    if action == "succeed_run":
        return manifest.succeed(at=NOW)
    if action == "fail_run":
        return manifest.fail(at=NOW)
    raise AssertionError(f"unknown manifest action: {action}")


@settings(max_examples=80, deadline=None)
@given(
    st.booleans(),
    st.lists(MANIFEST_ACTIONS, min_size=1, max_size=25),
)
def test_manifest_state_machine_accepts_only_declared_transitions(
    required: bool, actions: list[str]
) -> None:
    """Random action sequences preserve immutability, evidence fields, and retry counts."""

    manifest = _new_manifest(required=required)
    expected_attempt = 0
    for step, action in enumerate(actions):
        before = manifest.model_dump()
        allowed = _transition_is_allowed(manifest, action)

        try:
            transitioned = _apply_transition(manifest, action, step)
        except ArtifactIntegrityError:
            assert not allowed
            assert manifest.model_dump() == before
            continue

        assert allowed
        assert manifest.model_dump() == before
        assert transitioned is not manifest
        if action == "start_stage":
            expected_attempt += 1
        stage = transitioned.stages["stage"]
        assert stage.attempt == expected_attempt
        if stage.status is StageStatus.SUCCESS:
            assert stage.fingerprint is not None
            assert stage.started_at is not None
            assert stage.completed_at is not None
        if stage.status is StageStatus.FAILED:
            assert stage.error is not None
        if stage.status is StageStatus.UNSUPPORTED:
            assert stage.reason
        if action == "succeed_run" and required:
            assert stage.status is StageStatus.SUCCESS
        if transitioned.status in {RunStatus.SUCCESS, RunStatus.FAILED}:
            assert transitioned.started_at is not None
            assert transitioned.completed_at is not None
        manifest = transitioned
