"""Comparison compatibility and metric delta requirements."""

from __future__ import annotations

import pytest

from inkling_quant_lab.comparison import (
    BenchmarkMemoryIdentity,
    BenchmarkWorkloadIdentity,
    DatasetIdentity,
    EvaluationSuiteIdentity,
    MetricDelta,
    MetricValue,
    NormalizedRunSummary,
    ParetoObjective,
    RoutingCaptureIdentity,
    compare_summaries,
    compute_metric_deltas,
    pareto_points_from_summaries,
)
from inkling_quant_lab.evaluation.base import EvaluationFailure, EvaluationResult
from inkling_quant_lab.exceptions import ComparisonCompatibilityError
from inkling_quant_lab.pipeline.summaries import _dataset_contract, _evaluation_metrics

pytestmark = pytest.mark.unit


def summary(
    run_id: str,
    *,
    sample_ids: tuple[str, ...] = ("sample-1", "sample-2"),
    benchmark_sample_ids: tuple[str, ...] = ("bench-1", "bench-2"),
    quality: float | None = 0.8,
) -> NormalizedRunSummary:
    """Build a minimal normalized summary with a complete comparison contract."""

    quality_metric = (
        MetricValue(value=quality, category="quality", direction="maximize")
        if quality is not None
        else MetricValue(
            status="unsupported",
            category="quality",
            direction="maximize",
            reason="evaluator unavailable",
        )
    )
    return NormalizedRunSummary(
        run_id=run_id,
        artifact_path=f"artifacts/{run_id}",
        model_id="fixture://tiny-moe",
        model_revision="fixture-v1",
        datasets=(
            DatasetIdentity(
                dataset_id="fixture://evaluation",
                dataset_revision="fixture-v1",
                split="test",
                dataset_sha256="b" * 64,
            ),
        ),
        seed_set=(17,),
        sample_ids=sample_ids,
        prompt_template_hash="a" * 64,
        decode_config={"do_sample": False, "max_new_tokens": 4},
        routing_dataset=DatasetIdentity(
            dataset_id="fixture://routing",
            dataset_revision="fixture-v1",
            split="test",
            dataset_sha256="c" * 64,
        ),
        routing_sample_ids=("route-1", "route-2"),
        routing_capture=RoutingCaptureIdentity(
            configured_mode="full_trace",
            captured_mode="full_trace",
            observed_event_count=4,
            recorded_event_count=4,
            alignment_key_count=4,
            alignment_key_sha256="d" * 64,
        ),
        benchmark_protocol_version="cpu-v1",
        benchmark_workload=BenchmarkWorkloadIdentity(
            dataset_id="fixture://benchmark",
            dataset_revision="fixture-v1",
            dataset_sha256="e" * 64,
            split="test",
            sample_ids=benchmark_sample_ids,
            seed=17,
            prompt_template_hash="f" * 64,
            decode_config={"do_sample": False, "max_new_tokens": 4},
            execution_mode="sequential_samples",
        ),
        benchmark_memory=BenchmarkMemoryIdentity(
            host_measurement_kind="process_lifetime_high_water_rss",
            host_scope="Python process lifetime through the post-trial sample",
        ),
        hardware_environment={
            "hardware": {"device": "cpu", "logical_cpu_count": 8},
            "runtime": {"backend": "torch_eager_cpu", "device": "cpu"},
        },
        metrics={
            "quality": quality_metric,
            "latency_ms": MetricValue(
                value=10.0,
                unit="ms",
                category="resource",
                direction="minimize",
            ),
        },
    )


def test_compatible_runs_produce_absolute_and_relative_deltas() -> None:
    """TC-COMPARE-001: compatible runs produce deterministic metric deltas."""

    baseline = summary("baseline", quality=0.8)
    candidate = summary("candidate", quality=0.9)

    result = compare_summaries(baseline, candidate)

    assert result.contract_compatible is True
    assert result.mismatches == ()
    quality = result.delta_for("quality")
    assert quality.status == "available"
    assert quality.absolute_delta == pytest.approx(0.1)
    assert quality.relative_delta == pytest.approx(0.125)


def test_model_load_delta_is_unavailable_when_operations_differ() -> None:
    baseline = summary("baseline").model_copy(
        update={
            "benchmark_model_load_time_kind": "cold_model_load",
            "metrics": {
                **summary("baseline").metrics,
                "load_time_ms": MetricValue(
                    value=400.0,
                    unit="ms",
                    category="resource",
                    direction="minimize",
                ),
            },
        }
    )
    candidate = summary("candidate").model_copy(
        update={
            "benchmark_model_load_time_kind": "candidate_source_weight_free_export_load",
            "metrics": {
                **summary("candidate").metrics,
                "load_time_ms": MetricValue(
                    value=200.0,
                    unit="ms",
                    category="resource",
                    direction="minimize",
                ),
            },
        }
    )

    delta = compare_summaries(baseline, candidate).delta_for("load_time_ms")

    assert delta.status == "unavailable"
    assert delta.baseline_value == 400.0
    assert delta.candidate_value == 200.0
    assert delta.absolute_delta is None
    assert "operations differ" in (delta.reason or "")
    assert "cold_model_load" in (delta.reason or "")
    assert "candidate_source_weight_free_export_load" in (delta.reason or "")


def test_model_load_delta_remains_comparable_for_the_same_operation() -> None:
    load = MetricValue(
        value=400.0,
        unit="ms",
        category="resource",
        direction="minimize",
    )
    baseline = summary("baseline").model_copy(
        update={
            "benchmark_model_load_time_kind": "cold_model_load",
            "metrics": {**summary("baseline").metrics, "load_time_ms": load},
        }
    )
    candidate = summary("candidate").model_copy(
        update={
            "benchmark_model_load_time_kind": "cold_model_load",
            "metrics": {
                **summary("candidate").metrics,
                "load_time_ms": load.model_copy(update={"value": 300.0}),
            },
        }
    )

    delta = compare_summaries(baseline, candidate).delta_for("load_time_ms")

    assert delta.status == "available"
    assert delta.absolute_delta == -100.0


def test_incompatible_samples_fail_with_structured_mismatch_list() -> None:
    """TC-COMPARE-002: different samples fail and identify the dimension."""

    with pytest.raises(ComparisonCompatibilityError) as captured:
        compare_summaries(summary("baseline"), summary("candidate", sample_ids=("other",)))

    assert "sample_ids" in captured.value.message
    mismatches = captured.value.details["mismatches"]
    assert isinstance(mismatches, list)
    assert mismatches[0]["dimension"] == "sample_ids"
    assert mismatches[0]["baseline"] == ["sample-1", "sample-2"]
    assert mismatches[0]["candidate"] == ["other"]


def test_explicit_unsafe_override_is_prominent_and_persisted() -> None:
    """TC-COMPARE-003: an unsafe override permits and clearly marks comparison."""

    result = compare_summaries(
        summary("baseline"),
        summary("candidate", sample_ids=("other",)),
        unsafe_overrides={"sample_ids"},
    )

    assert result.contract_compatible is False
    assert result.unsafe_override is True
    assert result.overridden_dimensions == ("sample_ids",)
    assert any(warning.startswith("UNSAFE COMPARISON:") for warning in result.warnings)


def test_reordered_benchmark_workload_is_incompatible_and_explicitly_overridable() -> None:
    baseline = summary("baseline")
    candidate = summary("candidate", benchmark_sample_ids=("bench-2", "bench-1"))

    with pytest.raises(ComparisonCompatibilityError, match="benchmark_workload"):
        compare_summaries(baseline, candidate)

    result = compare_summaries(
        baseline,
        candidate,
        unsafe_overrides={"benchmark_workload"},
    )
    assert result.overridden_dimensions == ("benchmark_workload",)
    assert result.mismatches[0].dimension == "benchmark_workload"


def _suite_identity(
    evaluator: str,
    dataset: str,
    digest: str,
    samples: tuple[str, ...],
    prompt_digest: str,
    max_new_tokens: int,
) -> EvaluationSuiteIdentity:
    return EvaluationSuiteIdentity(
        evaluator_name=evaluator,
        evaluator_version="fixture-v1",
        dataset_id=dataset,
        dataset_revision="fixture-data-v1",
        split="test",
        dataset_sha256=digest,
        sample_ids=samples,
        seed=17,
        prompt_template_hash=prompt_digest,
        decode_config={"do_sample": False, "max_new_tokens": max_new_tokens},
        status="success",
    )


def _summary_with_suites(
    run_id: str, suites: tuple[EvaluationSuiteIdentity, ...]
) -> NormalizedRunSummary:
    ordered = tuple(sorted(suites, key=lambda suite: suite.canonical_json()))
    base = summary(run_id)
    quality = base.metrics["quality"].model_copy(update={"evaluation_suite": ordered[0]})
    datasets = tuple(
        sorted(
            {
                DatasetIdentity(
                    dataset_id=suite.dataset_id,
                    dataset_revision=suite.dataset_revision,
                    split=suite.split,
                    dataset_sha256=suite.dataset_sha256 or "",
                )
                for suite in ordered
            },
            key=lambda dataset: dataset.dataset_id,
        )
    )
    return base.model_copy(
        update={
            "schema_version": "1.1",
            "datasets": datasets,
            "sample_ids": tuple(
                sorted({sample for suite in ordered for sample in suite.sample_ids})
            ),
            "evaluation_suites": ordered,
            "metrics": {**base.metrics, "quality": quality},
        }
    )


@pytest.mark.parametrize(
    ("dimension", "candidate_suites"),
    (
        (
            "datasets",
            (
                _suite_identity("eval-a", "fixture://data-b", "b" * 64, ("a",), "c" * 64, 2),
                _suite_identity("eval-b", "fixture://data-a", "a" * 64, ("b",), "d" * 64, 4),
            ),
        ),
        (
            "sample_ids",
            (
                _suite_identity("eval-a", "fixture://data-a", "a" * 64, ("b",), "c" * 64, 2),
                _suite_identity("eval-b", "fixture://data-b", "b" * 64, ("a",), "d" * 64, 4),
            ),
        ),
        (
            "prompt_template_hash",
            (
                _suite_identity("eval-a", "fixture://data-a", "a" * 64, ("a",), "d" * 64, 2),
                _suite_identity("eval-b", "fixture://data-b", "b" * 64, ("b",), "c" * 64, 4),
            ),
        ),
        (
            "decode_config",
            (
                _suite_identity("eval-a", "fixture://data-a", "a" * 64, ("a",), "c" * 64, 4),
                _suite_identity("eval-b", "fixture://data-b", "b" * 64, ("b",), "d" * 64, 2),
            ),
        ),
    ),
)
def test_evaluator_scoped_contract_rejects_swapped_suite_inputs(
    dimension: str,
    candidate_suites: tuple[EvaluationSuiteIdentity, ...],
) -> None:
    """Aggregate sets cannot hide provenance moved between evaluator suites."""

    baseline_suites = (
        _suite_identity("eval-a", "fixture://data-a", "a" * 64, ("a",), "c" * 64, 2),
        _suite_identity("eval-b", "fixture://data-b", "b" * 64, ("b",), "d" * 64, 4),
    )
    baseline = _summary_with_suites("baseline", baseline_suites)
    candidate = _summary_with_suites("candidate", candidate_suites)

    assert _summary_with_suites("stable", baseline_suites).canonical_json() == (
        _summary_with_suites("stable", tuple(reversed(baseline_suites))).canonical_json()
    )
    assert baseline.datasets == candidate.datasets
    assert baseline.sample_ids == candidate.sample_ids
    with pytest.raises(ComparisonCompatibilityError, match=dimension):
        compare_summaries(baseline, candidate)


def test_duplicate_evaluator_types_remain_separate_and_failed_setup_sha_is_honest() -> None:
    duplicate_type_suites = (
        _suite_identity("same-evaluator", "fixture://data-a", "a" * 64, ("a",), "c" * 64, 2),
        _suite_identity("same-evaluator", "fixture://data-b", "b" * 64, ("b",), "d" * 64, 4),
    )
    scoped = _summary_with_suites("duplicates", duplicate_type_suites)

    assert len(scoped.evaluation_suites) == 2
    assert {suite.dataset_id for suite in scoped.evaluation_suites} == {
        "fixture://data-a",
        "fixture://data-b",
    }
    missing_metric_provenance = scoped.model_dump(mode="json")
    missing_metric_provenance["metrics"]["quality"]["evaluation_suite"] = None
    with pytest.raises(ValueError, match="requires suite provenance"):
        NormalizedRunSummary.model_validate(missing_metric_provenance)
    absent_metric_provenance = scoped.model_dump(mode="json")
    absent_metric_provenance["metrics"]["quality"]["evaluation_suite"] = _suite_identity(
        "other-evaluator",
        "fixture://other-data",
        "e" * 64,
        ("other",),
        "f" * 64,
        3,
    ).model_dump(mode="json")
    with pytest.raises(ValueError, match="absent from summary"):
        NormalizedRunSummary.model_validate(absent_metric_provenance)
    failed = EvaluationSuiteIdentity(
        evaluator_name="missing-data",
        evaluator_version="fixture-v1",
        dataset_id="file:///missing.jsonl",
        dataset_revision="fixture-data-v1",
        split="test",
        dataset_sha256=None,
        sample_ids=(),
        seed=17,
        prompt_template_hash="e" * 64,
        decode_config={"do_sample": False},
        status="failed",
    )
    assert failed.dataset_sha256 is None
    with pytest.raises(ValueError, match="non-failed suites"):
        EvaluationSuiteIdentity.model_validate(
            failed.model_dump(mode="json") | {"status": "unsupported"}
        )


def _perplexity_result(dataset_id: str, digest: str, mean_nll: float) -> EvaluationResult:
    return EvaluationResult(
        evaluator_name="perplexity",
        evaluator_version="fixture-v1",
        dataset_id=dataset_id,
        dataset_revision="fixture-data-v1",
        split="evaluation",
        dataset_sha256=digest,
        sample_ids=(f"{dataset_id}-sample",),
        sample_count=1,
        seed=17,
        prompt_template_hash="f" * 64,
        decode_config={"do_sample": False},
        metrics={"mean_nll": mean_nll, "perplexity": 2.718281828459045**mean_nll},
    )


def _exact_match_result(dataset_id: str, digest: str, value: float) -> EvaluationResult:
    return EvaluationResult(
        evaluator_name="exact_match",
        evaluator_version="fixture-v1",
        dataset_id=dataset_id,
        dataset_revision="fixture-data-v1",
        split="evaluation",
        dataset_sha256=digest,
        sample_ids=(f"{dataset_id}-sample",),
        sample_count=1,
        seed=17,
        prompt_template_hash="f" * 64,
        decode_config={"do_sample": False},
        metrics={"exact_match": value},
    )


def test_duplicate_evaluator_metric_producers_are_scoped_without_overwrite() -> None:
    first = _perplexity_result("fixture://data-a", "a" * 64, 1.0)
    second = _perplexity_result("fixture://data-b", "b" * 64, 2.0)

    metrics, failures = _evaluation_metrics((second, first))

    assert failures == []
    scoped_nll = {
        name: metric
        for name, metric in metrics.items()
        if name.startswith("evaluation.") and name.endswith(".mean_nll")
    }
    assert len(scoped_nll) == 2
    assert {metric.value for metric in scoped_nll.values()} == {1.0, 2.0}
    assert {
        metric.evaluation_suite.dataset_id
        for metric in scoped_nll.values()
        if metric.evaluation_suite is not None
    } == {"fixture://data-a", "fixture://data-b"}
    assert metrics["mean_nll"] in scoped_nll.values()
    assert metrics["mean_nll"].evaluation_suite is not None
    assert metrics["perplexity"].evaluation_suite is not None
    assert (
        metrics["mean_nll"].evaluation_suite.scientific_json()
        == metrics["perplexity"].evaluation_suite.scientific_json()
    )
    assert metrics["quality"].evaluation_suite is not None
    scoped_quality = [
        metric
        for name, metric in metrics.items()
        if name.startswith("evaluation.") and name.endswith(".quality")
    ]
    assert metrics["quality"] in scoped_quality

    with pytest.raises(ValueError, match="duplicate evaluation suite scientific identities"):
        _evaluation_metrics((first, first))


def test_failed_selection_digest_stays_in_suite_evidence_not_measured_datasets() -> None:
    measured_shape = _perplexity_result("fixture://data-a", "a" * 64, 1.0)
    failed_selection = EvaluationResult.model_validate(
        measured_shape.model_dump(mode="json")
        | {
            "metrics": {},
            "failures": [
                EvaluationFailure(
                    sample_id=None,
                    code="DATASET_SELECTION_FAILED",
                    message="configured sample was absent",
                ).model_dump(mode="json")
            ],
            "status": "failed",
        }
    )

    datasets, _, _, _, _, suites = _dataset_contract((failed_selection,))

    assert datasets == ()
    assert suites[0].dataset_sha256 == "a" * 64
    assert suites[0].status == "failed"


def test_headline_source_is_fixed_by_suite_contract_and_cross_source_math_is_blocked() -> None:
    first = _exact_match_result("fixture://data-a", "a" * 64, 0.8)
    second = _exact_match_result("fixture://data-b", "b" * 64, 0.9)
    candidate_metrics, _ = _evaluation_metrics((first, second))
    preferred = candidate_metrics["quality"].evaluation_suite
    assert preferred is not None
    preferred_result = first if preferred.dataset_id == first.dataset_id else second
    fallback_result = second if preferred_result is first else first
    failed_preferred = EvaluationResult.model_validate(
        preferred_result.model_dump(mode="json")
        | {
            "metrics": {},
            "failures": [
                EvaluationFailure(
                    sample_id=None,
                    code="OPTIONAL_FAILURE",
                    message="preferred optional suite failed",
                ).model_dump(mode="json")
            ],
            "status": "failed",
        }
    )
    baseline_metrics, _ = _evaluation_metrics((failed_preferred, fallback_result))

    baseline_quality = baseline_metrics["quality"]
    candidate_quality = candidate_metrics["quality"]
    assert baseline_quality.status == "failed"
    assert baseline_quality.evaluation_suite is not None
    assert candidate_quality.evaluation_suite is not None
    assert (
        baseline_quality.evaluation_suite.scientific_json()
        == candidate_quality.evaluation_suite.scientific_json()
    )
    assert (
        compute_metric_deltas({"quality": baseline_quality}, {"quality": candidate_quality})[
            0
        ].status
        == "unavailable"
    )

    mismatched_baseline = MetricValue(
        value=0.8,
        unit="fraction",
        category="quality",
        direction="maximize",
        evaluation_suite=_evaluation_suite_identity_for_test(first),
    )
    mismatched_candidate = MetricValue(
        value=0.9,
        unit="fraction",
        category="quality",
        direction="maximize",
        evaluation_suite=_evaluation_suite_identity_for_test(second),
    )
    mismatch = compute_metric_deltas(
        {"quality": mismatched_baseline}, {"quality": mismatched_candidate}
    )[0]
    assert mismatch.status == "unavailable"
    assert mismatch.reason == "evaluation metric suite provenance differs"

    baseline_summary = summary("baseline").model_copy(
        update={"metrics": {"quality": mismatched_baseline}}
    )
    candidate_summary = summary("candidate").model_copy(
        update={"metrics": {"quality": mismatched_candidate}}
    )
    assert (
        pareto_points_from_summaries(
            (baseline_summary, candidate_summary),
            (ParetoObjective(metric="quality", direction="maximize"),),
        )
        == ()
    )


def _evaluation_suite_identity_for_test(result: EvaluationResult) -> EvaluationSuiteIdentity:
    return EvaluationSuiteIdentity(
        evaluator_name=result.evaluator_name,
        evaluator_version=result.evaluator_version,
        dataset_id=result.dataset_id,
        dataset_revision=result.dataset_revision,
        split=result.split,
        dataset_sha256=result.dataset_sha256,
        sample_ids=result.sample_ids,
        seed=result.seed,
        prompt_template_hash=result.prompt_template_hash,
        decode_config=result.decode_config,
        status=result.status,
    )


def test_benchmark_memory_and_evaluation_suite_evidence_reject_ambiguous_inputs() -> None:
    """Collector scope and per-suite samples must remain explicit and unambiguous."""

    with pytest.raises(ValueError, match="host memory collector kind and scope"):
        BenchmarkMemoryIdentity(host_measurement_kind="interval_peak_rss")
    with pytest.raises(ValueError, match="device memory collector kind and scope"):
        BenchmarkMemoryIdentity(device_scope="candidate interval")
    with pytest.raises(ValueError, match="requires an available collector"):
        BenchmarkMemoryIdentity()
    with pytest.raises(ValueError, match="isolated host memory identity requires a host collector"):
        BenchmarkMemoryIdentity(
            host_process_isolated=True,
            device_measurement_kind="cuda_allocator_peak_since_previous_sample",
            device_scope="PyTorch CUDA allocator peak since the previous sample",
        )

    suite = _suite_identity(
        "eval-a",
        "fixture://data-a",
        "a" * 64,
        ("sample-a",),
        "c" * 64,
        2,
    )
    with pytest.raises(ValueError, match="sample IDs must be unique"):
        EvaluationSuiteIdentity.model_validate(
            suite.model_dump(mode="json") | {"sample_ids": ["same", "same"]}
        )
    with pytest.raises(ValueError, match="sample IDs must be non-empty"):
        EvaluationSuiteIdentity.model_validate(suite.model_dump(mode="json") | {"sample_ids": [""]})
    failed_suite = suite.model_copy(update={"status": "failed", "dataset_sha256": None})
    with pytest.raises(ValueError, match="successful or partial suite evidence"):
        MetricValue(value=1.0, category="quality", evaluation_suite=failed_suite)


def test_scientific_contract_rejects_and_can_record_each_new_override_dimension() -> None:
    """Seeds, exact data, routing alignment, and hardware are strict dimensions."""

    baseline = summary("baseline")
    routing_dataset = baseline.routing_dataset
    assert routing_dataset is not None
    variants = {
        "seed_set": {"seed_set": (29,)},
        "datasets": {
            "datasets": (baseline.datasets[0].model_copy(update={"dataset_sha256": "e" * 64}),)
        },
        "routing_dataset": {
            "routing_dataset": routing_dataset.model_copy(update={"dataset_sha256": "f" * 64})
        },
        "routing_sample_ids": {"routing_sample_ids": ("route-other",)},
        "routing_capture": {
            "routing_capture": baseline.routing_capture.model_copy(
                update={"alignment_key_sha256": "1" * 64}
            )
            if baseline.routing_capture is not None
            else None
        },
        "benchmark_memory": {
            "benchmark_memory": baseline.benchmark_memory.model_copy(
                update={"host_scope": "different process-lifetime boundary"}
            )
            if baseline.benchmark_memory is not None
            else None
        },
        "hardware_environment": {
            "hardware_environment": {
                "hardware": {"device": "cpu", "logical_cpu_count": 16},
                "runtime": {"backend": "torch_eager_cpu", "device": "cpu"},
            }
        },
    }

    for dimension, updates in variants.items():
        candidate = NormalizedRunSummary.model_validate(
            {**baseline.model_dump(mode="json"), "run_id": "candidate", **updates}
        )
        with pytest.raises(ComparisonCompatibilityError, match=dimension):
            compare_summaries(baseline, candidate)
        overridden = compare_summaries(
            baseline,
            candidate,
            unsafe_overrides={dimension},
        )
        assert overridden.overridden_dimensions == (dimension,)
        assert overridden.mismatches[0].dimension == dimension


def test_dataset_and_routing_capture_evidence_require_exact_digests() -> None:
    """Content and token-alignment evidence cannot be represented by labels alone."""

    with pytest.raises(ValueError, match="dataset_sha256"):
        DatasetIdentity(
            dataset_id="fixture://evaluation",
            dataset_revision="fixture-v1",
            split="test",
            dataset_sha256="not-a-digest",
        )
    with pytest.raises(ValueError, match="alignment evidence"):
        RoutingCaptureIdentity(
            configured_mode="aggregate",
            captured_mode="aggregate",
            observed_event_count=4,
            recorded_event_count=0,
            alignment_key_count=1,
            alignment_key_sha256=None,
        )

    payload = summary("candidate").model_dump(mode="json")
    payload["routing_capture"] = {
        "configured_mode": "aggregate",
        "captured_mode": "aggregate",
        "observed_event_count": 4,
        "recorded_event_count": 0,
        "alignment_key_count": 0,
        "alignment_key_sha256": None,
    }
    payload["metrics"]["route_agreement"] = {
        "value": 1.0,
        "status": "available",
        "unit": "fraction",
        "category": "routing",
        "direction": "maximize",
        "reason": None,
    }
    with pytest.raises(ValueError, match="retained alignment evidence"):
        NormalizedRunSummary.model_validate(payload)


def test_unsupported_metric_delta_remains_unavailable() -> None:
    """Missing measurements never become a numeric zero during comparison."""

    result = compare_summaries(summary("baseline"), summary("candidate", quality=None))

    quality = result.delta_for("quality")
    assert quality.status == "unavailable"
    assert quality.candidate_value is None
    assert quality.absolute_delta is None
    assert "evaluator unavailable" in (quality.reason or "")


def test_zero_baseline_has_absolute_but_no_relative_delta() -> None:
    """Relative change from an exact zero is explicitly unavailable."""

    result = compare_summaries(summary("baseline", quality=0.0), summary("candidate", quality=1.0))

    quality = result.delta_for("quality")
    assert quality.absolute_delta == 1.0
    assert quality.relative_delta is None
    assert quality.relative_delta_reason == "baseline value is zero"


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"status": "unsupported", "value": 1.0, "reason": "not available"},
        {"status": "unsupported"},
    ),
)
def test_metric_availability_schema_rejects_ambiguous_values(payload: dict[str, object]) -> None:
    """Unavailable and numeric values cannot be conflated in normalized artifacts."""

    with pytest.raises(ValueError):
        MetricValue.model_validate(payload)


def test_normalized_summary_validates_identity_and_serializes_canonically() -> None:
    """Normalized summaries reject ambiguous contracts and have stable JSON."""

    baseline = summary("baseline")
    assert baseline.canonical_json() == baseline.canonical_json()
    payload = baseline.model_dump()
    payload["sample_ids"] = ("same", "same")
    with pytest.raises(ValueError, match="sample IDs must be unique"):
        NormalizedRunSummary.model_validate(payload)
    payload = baseline.model_dump()
    payload["metrics"] = {}
    with pytest.raises(ValueError, match="at least one metric"):
        NormalizedRunSummary.model_validate(payload)


def test_unknown_override_and_unknown_delta_are_rejected() -> None:
    """Unsafe dimensions and metric lookup must be explicit and auditable."""

    with pytest.raises(ValueError, match="unknown unsafe"):
        compare_summaries(summary("baseline"), summary("candidate"), unsafe_overrides={"typo"})
    with pytest.raises(KeyError):
        compare_summaries(summary("baseline"), summary("candidate")).delta_for("missing")


def test_missing_units_and_directions_produce_unavailable_deltas() -> None:
    """Incomparable metric encodings are retained but never numerically combined."""

    baseline = {
        "missing_candidate": MetricValue(value=1.0),
        "units": MetricValue(value=1.0, unit="ms"),
        "direction": MetricValue(value=1.0, direction="minimize"),
    }
    candidate = {
        "missing_baseline": MetricValue(value=2.0),
        "units": MetricValue(value=2.0, unit="seconds"),
        "direction": MetricValue(value=2.0, direction="maximize"),
    }

    deltas = {delta.metric: delta for delta in compute_metric_deltas(baseline, candidate)}

    assert deltas["missing_candidate"].reason == "candidate metric is missing"
    assert deltas["missing_baseline"].reason == "baseline metric is missing"
    assert deltas["units"].reason and "units differ" in deltas["units"].reason
    assert deltas["direction"].reason and "directions differ" in deltas["direction"].reason


def test_metric_delta_schema_rejects_inconsistent_state() -> None:
    """Serialized deltas cannot claim availability without numeric evidence."""

    with pytest.raises(ValueError, match="requires baseline"):
        MetricDelta(
            metric="quality",
            status="available",
            category="quality",
            direction="maximize",
        )
    with pytest.raises(ValueError, match="cannot contain numeric"):
        MetricDelta(
            metric="quality",
            status="unavailable",
            category="quality",
            direction="maximize",
            absolute_delta=1.0,
            reason="not comparable",
        )
