"""Evaluation failure retention and behavioral-evidence pipeline integration."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from inkling_quant_lab.config import ExperimentConfig, load_config
from inkling_quant_lab.evaluation.base import (
    EvaluationFailure,
    RequiredEvaluationFailure,
)
from inkling_quant_lab.exceptions import EvaluationError
from inkling_quant_lab.manifests import RunStatus, StageStatus, load_manifest
from inkling_quant_lab.pipeline import runner as pipeline_runner
from inkling_quant_lab.pipeline.runner import resume_experiment, run_experiment

pytestmark = pytest.mark.integration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = PROJECT_ROOT / "configs" / "experiments"


def _evaluation_config(
    config_factory: Callable[..., ExperimentConfig],
    artifact_root: Path,
    *,
    optional_multimodal: bool,
    include_perplexity: bool,
) -> ExperimentConfig:
    raw = config_factory(
        model_id="local://fixtures/tiny-dense", routing_mode="off"
    ).canonical_dict()
    suites: list[dict[str, object]] = []
    if include_perplexity:
        suites.append(
            {
                "type": "perplexity",
                "dataset": "local://fixtures/tiny-corpus",
                "revision": "fixture-data-v1",
                "split": "evaluation",
            }
        )
    suites.append(
        {
            "type": "multimodal_contract",
            "dataset": "local://fixtures/multimodal",
            "revision": "fixture-data-v1",
            "split": "evaluation",
            "optional": optional_multimodal,
        }
    )
    raw["evaluation"] = {"allow_partial": True, "suites": suites}
    raw["benchmark"] = {"enabled": False}
    raw["routing"] = {"mode": "off", "required": False}
    raw["output"]["root"] = str(artifact_root)
    raw["reporting"] = {"markdown": True, "html": False, "plots": False}
    return ExperimentConfig.model_validate(raw)


def test_optional_evaluator_failure_is_reported_and_run_completes(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """FR-007: an optional unavailable evaluator reaches the final report."""

    config = _evaluation_config(
        config_factory,
        tmp_path / "artifacts",
        optional_multimodal=True,
        include_perplexity=True,
    )
    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="optional-evaluator-failure",
    )

    manifest = load_manifest(run_directory)
    results = json.loads(
        (run_directory / "metrics/evaluation_candidate/results.json").read_text(encoding="utf-8")
    )
    report = (run_directory / "reports/report.md").read_text(encoding="utf-8")

    assert manifest.status is RunStatus.SUCCESS
    assert manifest.stages["evaluate_candidate"].status is StageStatus.SUCCESS
    assert results[1]["evaluator_name"] == "multimodal_contract"
    assert results[1]["status"] == "unsupported"
    assert results[1]["failures"][0]["sample_id"] is None
    assert "multimodal_contract/suite" in report
    assert "does not support multimodal inputs" in report


def test_optional_dataset_load_failure_is_reported_and_run_completes(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """An optional pre-execution failure stays in normal result and report artifacts."""

    raw = config_factory(
        model_id="local://fixtures/tiny-dense", routing_mode="off"
    ).canonical_dict()
    raw["evaluation"] = {
        "allow_partial": True,
        "suites": [
            {
                "type": "generation_regression",
                "dataset": (tmp_path / "missing-optional.jsonl").as_uri(),
                "revision": "local-v1",
                "split": "evaluation",
                "optional": True,
            },
        ],
    }
    raw["benchmark"] = {"enabled": False}
    raw["routing"] = {"mode": "off", "required": False}
    raw["output"]["root"] = str(tmp_path / "artifacts")
    raw["reporting"] = {"markdown": True, "html": False, "plots": False}
    config = ExperimentConfig.model_validate(raw)

    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="optional-dataset-load-failure",
    )
    manifest = load_manifest(run_directory)
    results = json.loads(
        (run_directory / "metrics/evaluation_candidate/results.json").read_text(encoding="utf-8")
    )
    report = (run_directory / "reports/report.md").read_text(encoding="utf-8")

    assert manifest.status is RunStatus.SUCCESS
    assert results[0]["evaluator_name"] == "generation_regression"
    assert results[0]["status"] == "failed"
    assert results[0]["dataset_sha256"] is None
    assert "generation_regression/suite" in report
    assert "Unable to read dataset" in report


@pytest.mark.parametrize("failure_kind", ("load", "selection"))
def test_required_dataset_setup_failure_writes_attempt_scoped_evidence(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
    failure_kind: str,
) -> None:
    """Required load/selection failures fail the stage and retain typed evidence."""

    raw = config_factory(
        model_id="local://fixtures/tiny-dense", routing_mode="off"
    ).canonical_dict()
    suite: dict[str, object] = {
        "type": "perplexity",
        "dataset": "local://fixtures/tiny-corpus",
        "revision": "fixture-data-v1",
        "split": "evaluation",
    }
    if failure_kind == "load":
        suite["dataset"] = (tmp_path / "missing-required.jsonl").as_uri()
    else:
        suite["sample_ids"] = ["absent-required-sample"]
    raw["evaluation"] = {"allow_partial": True, "suites": [suite]}
    raw["benchmark"] = {"enabled": False}
    raw["routing"] = {"mode": "off", "required": False}
    raw["output"]["root"] = str(tmp_path / "artifacts")
    raw["reporting"] = {"markdown": True, "html": False, "plots": False}
    config = ExperimentConfig.model_validate(raw)
    run_id = f"required-dataset-{failure_kind}-failure"
    run_directory = tmp_path / "artifacts" / run_id

    with pytest.raises(RequiredEvaluationFailure):
        run_experiment(config, project_root=PROJECT_ROOT, run_id=run_id)

    manifest = load_manifest(run_directory)
    failure_results = json.loads(
        (run_directory / "metrics/evaluation_failures/baseline/attempt-1/results.json").read_text(
            encoding="utf-8"
        )
    )
    failure_report = run_directory / "failure_reports/evaluation/evaluate_baseline-attempt-1.md"

    assert manifest.status is RunStatus.FAILED
    assert manifest.stages["evaluate_baseline"].status is StageStatus.FAILED
    assert failure_results[0]["evaluator_name"] == "perplexity"
    assert failure_results[0]["status"] == "failed"
    assert failure_results[0]["metrics"] == {}
    assert failure_report.is_file()
    assert not (run_directory / "reports/report.md").exists()


def test_checked_in_demo_reports_evidenced_base_fine_tuned_candidate_retention(
    tmp_path: Path,
) -> None:
    """The default CPU demo carries real per-sample B/F rubric evidence."""

    config = load_config(
        EXPERIMENTS / "tiny_moe_int8.yaml",
        (
            f"output.root={json.dumps(str(tmp_path / 'artifacts'))}",
            "benchmark.enabled=false",
            'routing.mode="off"',
            "routing.required=false",
            "reporting.plots=false",
        ),
    )
    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="behavioral-evidence-demo",
    )

    baseline = json.loads(
        (run_directory / "reports/baseline_summary.json").read_text(encoding="utf-8")
    )
    candidate = json.loads(
        (run_directory / "reports/candidate_summary.json").read_text(encoding="utf-8")
    )

    assert baseline["metrics"]["behavioral_base_score"]["value"] == 0.0
    assert baseline["metrics"]["behavioral_fine_tuned_score"]["value"] == 1.0
    assert baseline["metrics"]["behavioral_score"]["value"] == 1.0
    assert baseline["metrics"]["behavioral_retention"]["value"] == 1.0
    assert candidate["metrics"]["behavioral_score"]["value"] == 1.0
    assert candidate["metrics"]["behavioral_retention"]["value"] == 1.0


@pytest.mark.parametrize(
    ("reference_fields", "reason_fragment"),
    (
        ("", "without explicit base/fine-tuned evidence"),
        (
            ',"base_first_token":0,"fine_tuned_first_token":0',
            "does not match the declared fine-tuned reference evidence",
        ),
    ),
)
def test_behavioral_retention_is_unavailable_without_consistent_b_f_evidence(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
    reference_fields: str,
    reason_fragment: str,
) -> None:
    """Raw rubric measurement never licenses an invented normalization anchor."""

    dataset = tmp_path / "behavior-evidence.jsonl"
    dataset.write_text(
        '{"id":"behavior-1","text":"safe helpful","accepted_tokens":[2]' + reference_fields + "}\n",
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
    raw["benchmark"] = {"enabled": False}
    raw["routing"] = {"mode": "off", "required": False}
    raw["output"]["root"] = str(tmp_path / "artifacts")
    raw["reporting"] = {"markdown": True, "html": False, "plots": False}
    config = ExperimentConfig.model_validate(raw)

    run_directory = run_experiment(
        config,
        project_root=PROJECT_ROOT,
        run_id="behavioral-evidence-unavailable",
    )
    candidate = json.loads(
        (run_directory / "reports/candidate_summary.json").read_text(encoding="utf-8")
    )
    retention = candidate["metrics"]["behavioral_retention"]

    assert candidate["metrics"]["behavioral_score"]["value"] == 1.0
    assert retention["status"] == "unavailable"
    assert retention["value"] is None
    assert reason_fragment in retention["reason"]


def test_required_evaluator_failure_retains_typed_evidence_and_fails_run(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """A typed result is evidence, not permission to call a required stage successful."""

    config = _evaluation_config(
        config_factory,
        tmp_path / "artifacts",
        optional_multimodal=False,
        include_perplexity=False,
    )
    run_directory = tmp_path / "artifacts" / "required-evaluator-failure"

    with pytest.raises(EvaluationError, match="Required evaluation suites failed"):
        run_experiment(
            config,
            project_root=PROJECT_ROOT,
            run_id="required-evaluator-failure",
        )

    manifest = load_manifest(run_directory)
    failure = json.loads(
        (run_directory / "failures/evaluate_baseline-attempt-1.json").read_text(encoding="utf-8")
    )

    assert manifest.status is RunStatus.FAILED
    assert manifest.stages["evaluate_baseline"].status is StageStatus.FAILED
    assert failure["code"] == EvaluationError.code
    assert failure["details"]["evaluation_results"][0]["status"] == "unsupported"
    assert failure["details"]["evaluation_results"][0]["failures"][0]["sample_id"] is None
    results_path = run_directory / failure["details"]["evaluation_results_path"]
    report_path = run_directory / failure["details"]["failure_report_markdown"]
    assert json.loads(results_path.read_text(encoding="utf-8"))[0]["status"] == "unsupported"
    report = report_path.read_text(encoding="utf-8")
    assert "run that is **failed**" in report
    assert "multimodal_contract" in report
    assert "does not support multimodal inputs" in report
    assert not (run_directory / "reports/report.md").exists()

    with pytest.raises(EvaluationError, match="Required evaluation suites failed"):
        resume_experiment(run_directory, project_root=PROJECT_ROOT)
    resumed_manifest = load_manifest(run_directory)
    assert resumed_manifest.stages["evaluate_baseline"].attempt == 2
    assert (run_directory / "metrics/evaluation_failures/baseline/attempt-2/results.json").is_file()
    assert (run_directory / "failure_reports/evaluation/evaluate_baseline-attempt-2.md").is_file()


def test_transient_required_evaluator_failure_resumes_without_losing_evidence(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful retry publishes reports and links the preserved failed attempt."""

    config = _evaluation_config(
        config_factory,
        tmp_path / "artifacts",
        optional_multimodal=True,
        include_perplexity=True,
    )
    original_evaluate_model = pipeline_runner.evaluate_model
    failed_once = False

    def fail_first_required_evaluation(config, components, model):
        nonlocal failed_once
        results = original_evaluate_model(config, components, model)
        if failed_once:
            return results
        failed_once = True
        failed_result = results[0].model_copy(
            update={
                "metrics": {},
                "status": "failed",
                "failures": (
                    EvaluationFailure(
                        sample_id=None,
                        code="TRANSIENT_EVALUATOR_FAILURE",
                        message="synthetic fail-once evaluator fault",
                    ),
                ),
            }
        )
        retained = (failed_result, *results[1:])
        raise RequiredEvaluationFailure(retained, (failed_result.evaluator_name,))

    monkeypatch.setattr(pipeline_runner, "evaluate_model", fail_first_required_evaluation)
    run_directory = tmp_path / "artifacts" / "transient-evaluator-failure"

    with pytest.raises(EvaluationError, match="Required evaluation suites failed"):
        run_experiment(
            config,
            project_root=PROJECT_ROOT,
            run_id="transient-evaluator-failure",
        )

    failure_report = run_directory / "failure_reports/evaluation/evaluate_baseline-attempt-1.md"
    failure_results = run_directory / "metrics/evaluation_failures/baseline/attempt-1/results.json"
    assert failure_report.is_file()
    assert failure_results.is_file()
    assert (run_directory / "reports").is_dir()
    assert not any((run_directory / "reports").iterdir())

    resumed_directory = resume_experiment(run_directory, project_root=PROJECT_ROOT)
    manifest = load_manifest(resumed_directory)
    final_report = (resumed_directory / "reports/report.md").read_text(encoding="utf-8")
    report_json = json.loads(
        (resumed_directory / "reports/report_data.json").read_text(encoding="utf-8")
    )

    assert manifest.status is RunStatus.SUCCESS
    assert manifest.stages["evaluate_baseline"].attempt == 2
    assert manifest.stages["generate_reports"].status is StageStatus.SUCCESS
    assert failure_report.is_file()
    assert failure_results.is_file()
    assert "Prior Failed Evaluation Attempts" in final_report
    assert "../failure_reports/evaluation/evaluate_baseline-attempt-1.md" in final_report
    assert "../metrics/evaluation_failures/baseline/attempt-1/results.json" in final_report
    assert any("prior failure evidence" in warning for warning in report_json["warnings"])
    assert report_json["metadata"]["prior_evaluation_failure_artifacts"] == [
        "failure_reports/evaluation/evaluate_baseline-attempt-1.json",
        "failure_reports/evaluation/evaluate_baseline-attempt-1.md",
        "metrics/evaluation_failures/baseline/attempt-1/results.json",
    ]
