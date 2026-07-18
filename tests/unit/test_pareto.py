"""Pareto objective direction and tolerance requirements."""

from __future__ import annotations

import pytest

from inkling_quant_lab.comparison import (
    DatasetIdentity,
    MetricValue,
    NormalizedRunSummary,
    ParetoObjective,
    ParetoPoint,
    dominates,
    pareto_frontier,
    pareto_points_from_summaries,
)

pytestmark = pytest.mark.unit


def _summary(run_id: str, quality: MetricValue | None = None) -> NormalizedRunSummary:
    return NormalizedRunSummary(
        run_id=run_id,
        artifact_path=f"artifacts/{run_id}",
        model_id="fixture://tiny",
        model_revision="fixture-v1",
        datasets=(
            DatasetIdentity(
                dataset_id="fixture://eval",
                dataset_revision="v1",
                split="test",
                dataset_sha256="a" * 64,
            ),
        ),
        seed_set=(17,),
        sample_ids=("s1",),
        prompt_template_hash="c" * 64,
        decode_config={"do_sample": False},
        benchmark_protocol_version="cpu-v1",
        hardware_environment={
            "hardware": {"device": "cpu"},
            "runtime": {"backend": "torch_eager_cpu", "device": "cpu"},
        },
        metrics={"quality": quality or MetricValue(value=0.9, category="quality")},
    )


def test_known_quality_memory_latency_frontier() -> None:
    """TC-PARETO-001: only the expected trade-off points remain optimal."""

    objectives = (
        ParetoObjective(metric="quality", direction="maximize"),
        ParetoObjective(metric="memory", direction="minimize"),
        ParetoObjective(metric="latency", direction="minimize"),
    )
    points = (
        ParetoPoint(point_id="a", values={"quality": 0.90, "memory": 100.0, "latency": 10.0}),
        ParetoPoint(point_id="b", values={"quality": 0.92, "memory": 90.0, "latency": 9.0}),
        ParetoPoint(point_id="c", values={"quality": 0.95, "memory": 120.0, "latency": 8.0}),
        ParetoPoint(point_id="d", values={"quality": 0.91, "memory": 110.0, "latency": 11.0}),
    )

    result = pareto_frontier(points, objectives)

    assert result.optimal_ids == ("b", "c")
    assert result.membership_for("a").dominated_by == ("b",)
    assert result.membership_for("d").dominated_by == ("b",)


def test_values_within_tolerance_do_not_falsely_dominate() -> None:
    """TC-PARETO-002: changes inside tolerance do not count as improvement."""

    objectives = (
        ParetoObjective(metric="quality", direction="maximize", tolerance=0.001),
        ParetoObjective(metric="memory", direction="minimize", tolerance=1.0),
    )
    points = (
        ParetoPoint(point_id="a", values={"quality": 0.9000, "memory": 100.0}),
        ParetoPoint(point_id="b", values={"quality": 0.9005, "memory": 99.5}),
    )

    result = pareto_frontier(points, objectives)

    assert result.optimal_ids == ("a", "b")


def test_objective_direction_is_applied_per_metric() -> None:
    """Maximize and minimize directions must not be globally inferred."""

    result = pareto_frontier(
        (
            ParetoPoint(point_id="worse", values={"quality": 0.8, "latency": 12.0}),
            ParetoPoint(point_id="better", values={"quality": 0.9, "latency": 10.0}),
        ),
        (
            ParetoObjective(metric="quality", direction="maximize"),
            ParetoObjective(metric="latency", direction="minimize"),
        ),
    )

    assert result.optimal_ids == ("better",)


@pytest.mark.parametrize(
    ("points", "objectives", "message"),
    (
        ((ParetoPoint(point_id="a", values={"x": 1.0}),), (), "at least one objective"),
        (
            (ParetoPoint(point_id="a", values={"x": 1.0}),),
            (
                ParetoObjective(metric="x", direction="maximize"),
                ParetoObjective(metric="x", direction="minimize"),
            ),
            "names must be unique",
        ),
        (
            (
                ParetoPoint(point_id="a", values={"x": 1.0}),
                ParetoPoint(point_id="a", values={"x": 2.0}),
            ),
            (ParetoObjective(metric="x", direction="maximize"),),
            "IDs must be unique",
        ),
        (
            (ParetoPoint(point_id="a", values={"x": 1.0}),),
            (ParetoObjective(metric="missing", direction="maximize"),),
            "missing objectives",
        ),
    ),
)
def test_invalid_pareto_inputs_fail_actionably(
    points: tuple[ParetoPoint, ...],
    objectives: tuple[ParetoObjective, ...],
    message: str,
) -> None:
    """Malformed objective matrices fail before producing a frontier."""

    with pytest.raises(ValueError, match=message):
        pareto_frontier(points, objectives)


def test_nonfinite_and_empty_pareto_values_are_rejected() -> None:
    """Pareto coordinates must be complete finite measurements."""

    with pytest.raises(ValueError, match="at least one value"):
        ParetoPoint(point_id="empty", values={})
    with pytest.raises(ValueError, match="finite"):
        ParetoPoint(point_id="infinite", values={"quality": float("inf")})


def test_summary_extraction_skips_unavailable_objectives_without_zero_fill() -> None:
    """Unavailable objectives exclude a point instead of being interpreted as zero."""

    available = _summary("available")
    unavailable = _summary("unavailable").model_copy(
        update={
            "metrics": {
                **_summary("unavailable").metrics,
                "quality": MetricValue(
                    status="unsupported",
                    category="quality",
                    direction="maximize",
                    reason="not supported",
                ),
            }
        }
    )
    objective = ParetoObjective(metric="quality", direction="maximize")

    points = pareto_points_from_summaries((available, unavailable), (objective,))

    assert tuple(point.point_id for point in points) == ("available",)
    with pytest.raises(ValueError, match="missing objective"):
        dominates(
            ParetoPoint(point_id="incomplete", values={"other": 1.0}),
            ParetoPoint(point_id="complete", values={"quality": 1.0}),
            (objective,),
        )
