"""Evaluation registry and resolved-suite execution."""

from __future__ import annotations

from typing import Literal

from inkling_quant_lab.bootstrap import register_builtins
from inkling_quant_lab.config import EvaluationSuiteConfig, ExperimentConfig
from inkling_quant_lab.data import LocalDataset, load_local_dataset, select_dataset_samples
from inkling_quant_lab.evaluation.base import (
    EvaluationContext,
    EvaluationFailure,
    EvaluationResult,
    Evaluator,
    RequiredEvaluationFailure,
    prompt_template_hash,
)
from inkling_quant_lab.exceptions import CapabilityError, EvaluationError
from inkling_quant_lab.models.base import LoadedModel, ModelAdapter
from inkling_quant_lab.registry import EVALUATORS


def run_evaluations(
    adapter: ModelAdapter, model: LoadedModel, config: ExperimentConfig
) -> tuple[EvaluationResult, ...]:
    """Run configured suites against one model with shared resolved settings."""

    register_builtins()
    results: list[EvaluationResult] = []
    required_failures: list[str] = []
    for suite in config.evaluation.suites:
        evaluator = EVALUATORS.create(suite.type)
        dataset: LocalDataset | None = None
        try:
            dataset = load_local_dataset(suite.dataset, suite.revision, suite.split)
            dataset = select_dataset_samples(dataset, suite.sample_ids)
            context = EvaluationContext(
                adapter=adapter,
                model=model,
                experiment=config,
                suite=suite,
                dataset=dataset,
            )
            result = evaluator.run(context)
        except CapabilityError as error:
            result = _failed_result(
                config,
                suite,
                evaluator,
                error,
                dataset=dataset,
                status="unsupported",
            )
        except EvaluationError as error:
            result = _failed_result(
                config,
                suite,
                evaluator,
                error,
                dataset=dataset,
                status="failed",
            )
        results.append(result)
        if not suite.optional and result.status in {"failed", "unsupported"}:
            required_failures.append(suite.type)
    completed = tuple(results)
    if required_failures:
        raise RequiredEvaluationFailure(
            completed,
            tuple(required_failures),
        )
    return completed


def _failed_result(
    experiment: ExperimentConfig,
    suite: EvaluationSuiteConfig,
    evaluator: Evaluator,
    error: CapabilityError | EvaluationError,
    *,
    dataset: LocalDataset | None,
    status: Literal["unsupported", "failed"],
) -> EvaluationResult:
    """Retain typed suite failure evidence, including any verified provenance."""

    if suite.sample_ids:
        sample_ids = tuple(dict.fromkeys(suite.sample_ids))
    elif dataset is not None:
        sample_ids = dataset.sample_ids
    else:
        sample_ids = ()
    failure_message = error.message
    if error.remediation is not None:
        failure_message += f" Remediation: {error.remediation}"

    return EvaluationResult.model_validate(
        {
            "evaluator_name": evaluator.name,
            "evaluator_version": evaluator.version,
            "dataset_id": suite.dataset,
            "dataset_revision": suite.revision,
            "split": suite.split,
            "dataset_sha256": dataset.sha256 if dataset is not None else None,
            "sample_ids": sample_ids,
            "sample_count": len(sample_ids),
            "seed": experiment.seed,
            "prompt_template_hash": prompt_template_hash(suite.prompt_template),
            "decode_config": suite.decode.model_dump(mode="json"),
            "metrics": {},
            "failures": (
                EvaluationFailure(
                    code=error.code,
                    message=failure_message,
                ),
            ),
            "status": status,
        }
    )
