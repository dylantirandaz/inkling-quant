"""Behavioral rubric scoring and normalized retention math."""

from __future__ import annotations

import math
from typing import cast

from inkling_quant_lab.evaluation.base import (
    EvaluationContext,
    EvaluationFailure,
    EvaluationResult,
    TokenizerLike,
    complete_sample_ids,
    prompt_template_hash,
    render_prompt,
    retain_invalid_sample,
    validate_single_sample_output,
)
from inkling_quant_lab.exceptions import EvaluationError
from inkling_quant_lab.models.base import ModelBatch


def retention_score(
    base_score: float, fine_tuned_score: float, quantized_score: float, *, epsilon: float = 1e-8
) -> float:
    """Compute documented higher-is-better behavioral retention."""

    if not all(math.isfinite(score) for score in (base_score, fine_tuned_score, quantized_score)):
        raise ValueError("behavioral retention scores must be finite")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and greater than zero")
    denominator = max(epsilon, abs(fine_tuned_score - base_score))
    return 1.0 - max(0.0, fine_tuned_score - quantized_score) / denominator


class BehavioralEvaluator:
    """Score whether the first generated token belongs to a local rubric set."""

    name = "behavioral_retention"
    version = "1.0"

    def run(self, context: EvaluationContext) -> EvaluationResult:
        """Evaluate a deterministic local rubric without persisting prompt/output text."""

        if context.suite.decode.do_sample:
            raise EvaluationError(
                "Behavioral evaluation requires deterministic decoding with do_sample=false",
                component=self.name,
            )
        tokenizer = cast(TokenizerLike, context.model.tokenizer)
        successful_ids: list[str] = []
        failures: list[EvaluationFailure] = []
        accepted = 0
        base_evidence: list[bool] = []
        fine_tuned_evidence: list[bool] = []
        evidence_complete = True
        for sample in context.dataset.samples:
            try:
                text = sample.values["text"]
                accepted_tokens = sample.values["accepted_tokens"]
                if not isinstance(accepted_tokens, list) or not accepted_tokens:
                    raise ValueError("text and accepted_tokens are required")
                if any(
                    not isinstance(token, int) or isinstance(token, bool) or token < 0
                    for token in accepted_tokens
                ):
                    raise ValueError("accepted_tokens must contain non-negative integers")
                accepted_set = set(accepted_tokens)
                prompt = render_prompt(context.suite.prompt_template, text)
                input_ids, attention_mask = tokenizer.batch_encode((prompt,))
                output = context.adapter.generate(
                    context.model,
                    ModelBatch(
                        sample_ids=(sample.sample_id,),
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ),
                    context.suite.decode,
                )
                validate_single_sample_output(
                    sample.sample_id,
                    output.sample_ids,
                    len(output.token_ids),
                )
                if not output.token_ids[0]:
                    raise EvaluationError(
                        "Model adapter returned no generated tokens for behavioral evaluation",
                        component=self.name,
                    )
            except (KeyError, TypeError, ValueError):
                retain_invalid_sample(
                    context,
                    failures,
                    sample_id=sample.sample_id,
                    evaluator_name=self.name,
                )
                continue
            first_token = output.token_ids[0][0]
            accepted += int(first_token in accepted_set)
            successful_ids.append(sample.sample_id)
            base_token = sample.values.get("base_first_token")
            fine_tuned_token = sample.values.get("fine_tuned_first_token")
            if (
                isinstance(base_token, int)
                and not isinstance(base_token, bool)
                and base_token >= 0
                and isinstance(fine_tuned_token, int)
                and not isinstance(fine_tuned_token, bool)
                and fine_tuned_token >= 0
            ):
                base_evidence.append(base_token in accepted_set)
                fine_tuned_evidence.append(fine_tuned_token in accepted_set)
            else:
                evidence_complete = False
        if not successful_ids:
            raise EvaluationError(
                "Behavior evaluation produced no valid samples",
                component=self.name,
            )
        has_reference_evidence = evidence_complete and len(base_evidence) == len(successful_ids)
        base_score = sum(base_evidence) / len(base_evidence) if has_reference_evidence else None
        fine_tuned_score = (
            sum(fine_tuned_evidence) / len(fine_tuned_evidence) if has_reference_evidence else None
        )
        return EvaluationResult(
            evaluator_name=self.name,
            evaluator_version=self.version,
            dataset_id=context.dataset.dataset_id,
            dataset_revision=context.dataset.revision,
            split=context.dataset.split,
            dataset_sha256=context.dataset.sha256,
            sample_ids=complete_sample_ids(context),
            sample_count=len(context.dataset.samples),
            seed=context.experiment.seed,
            prompt_template_hash=prompt_template_hash(context.suite.prompt_template),
            decode_config=context.suite.decode.model_dump(mode="json"),
            metrics={
                "rubric_score": accepted / len(successful_ids),
                "base_rubric_score": base_score,
                "fine_tuned_rubric_score": fine_tuned_score,
                "reference_evidence_samples": (
                    len(successful_ids) if has_reference_evidence else 0
                ),
                "accepted": accepted,
                "successful_samples": len(successful_ids),
            },
            failures=tuple(failures),
            status="partial" if failures else "success",
        )
