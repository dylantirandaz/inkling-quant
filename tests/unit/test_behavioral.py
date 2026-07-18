"""Behavioral retention math and local rubric evaluator tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.evaluation.behavioral import retention_score
from inkling_quant_lab.evaluation.runner import run_evaluations
from inkling_quant_lab.models.local import LocalFixtureAdapter
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("base", "fine_tuned", "quantized", "expected"),
    (
        (0.4, 0.8, 0.7, 0.75),
        (0.4, 0.8, 0.8, 1.0),
        (0.4, 0.8, 0.4, 0.0),
        (0.5, 0.5, 0.5, 1.0),
        (0.5, 0.5, 0.499999995, 0.5),
    ),
)
def test_behavioral_retention_formula(
    base: float,
    fine_tuned: float,
    quantized: float,
    expected: float,
) -> None:
    """TC-BEHAVIOR-001: normal, perfect, degraded, and near-zero cases."""

    assert retention_score(base, fine_tuned, quantized) == pytest.approx(expected)


def test_retention_formula_rejects_nonfinite_or_nonpositive_epsilon() -> None:
    """Invalid normalized-score inputs fail instead of producing misleading values."""

    with pytest.raises(ValueError, match="finite"):
        retention_score(float("nan"), 0.8, 0.7)
    with pytest.raises(ValueError, match="epsilon"):
        retention_score(0.4, 0.8, 0.7, epsilon=0.0)


def test_behavioral_rubric_result_is_deterministic_and_contains_no_outputs(
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """The local rubric emits raw scores and provenance without generated text."""

    raw = config_factory(
        model_id="local://fixtures/tiny-dense", routing_mode="off"
    ).canonical_dict()
    raw["evaluation"] = {
        "allow_partial": True,
        "suites": [
            {
                "type": "behavioral_retention",
                "dataset": "local://fixtures/behavior-prompts",
                "revision": "fixture-data-v1",
                "split": "evaluation",
                "decode": {"max_new_tokens": 1, "do_sample": False},
            }
        ],
    }
    config = ExperimentConfig.model_validate(raw)
    adapter = LocalFixtureAdapter()
    model = adapter.load(config, TorchEagerCPURuntime())

    first = run_evaluations(adapter, model, config)[0]
    second = run_evaluations(adapter, model, config)[0]

    assert first.metrics["successful_samples"] == 3
    assert first.metrics["base_rubric_score"] == 0.0
    assert first.metrics["fine_tuned_rubric_score"] == 1.0
    assert first.metrics["rubric_score"] == 1.0
    assert first.metrics["reference_evidence_samples"] == 3
    assert first.canonical_json() == second.canonical_json()
    assert first.artifacts == ()
    assert "safe helpful" not in first.canonical_json()
    assert "token_ids" not in first.canonical_json()


def test_behavioral_rubric_without_explicit_base_and_fine_tuned_evidence_is_unavailable(
    tmp_path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """A raw rubric score is useful, but is not itself B/F/Q retention evidence."""

    dataset = tmp_path / "behavior-without-reference-evidence.jsonl"
    dataset.write_text(
        '{"id":"behavior-1","text":"safe helpful","accepted_tokens":[2]}\n',
        encoding="utf-8",
    )
    raw = config_factory(
        model_id="local://fixtures/tiny-dense", routing_mode="off"
    ).canonical_dict()
    raw["evaluation"] = {
        "allow_partial": True,
        "suites": [
            {
                "type": "behavioral_retention",
                "dataset": dataset.as_uri(),
                "revision": "local-v1",
                "split": "evaluation",
                "decode": {"max_new_tokens": 1, "do_sample": False},
            }
        ],
    }
    config = ExperimentConfig.model_validate(raw)
    adapter = LocalFixtureAdapter()
    model = adapter.load(config, TorchEagerCPURuntime())

    result = run_evaluations(adapter, model, config)[0]

    assert result.status == "success"
    assert result.metrics["rubric_score"] == 1.0
    assert result.metrics["base_rubric_score"] is None
    assert result.metrics["fine_tuned_rubric_score"] is None
    assert result.metrics["reference_evidence_samples"] == 0
