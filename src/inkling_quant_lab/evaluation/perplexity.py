"""Forward-loss and perplexity evaluators."""

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


def perplexity_from_nll(mean_nll: float) -> float:
    """Compute perplexity from mean NLL with a finite input requirement."""

    if not math.isfinite(mean_nll):
        raise ValueError("mean_nll must be finite")
    return math.exp(mean_nll)


class PerplexityEvaluator:
    """Evaluate samples independently so configured partial failures are retained."""

    name = "perplexity"
    version = "1.0"

    def run(self, context: EvaluationContext) -> EvaluationResult:
        """Return token-weighted loss/perplexity and complete provenance."""

        tokenizer = cast(TokenizerLike, context.model.tokenizer)
        successful_ids: list[str] = []
        losses: list[float] = []
        counts: list[int] = []
        failures: list[EvaluationFailure] = []
        for sample in context.dataset.samples:
            try:
                prompt = render_prompt(context.suite.prompt_template, sample.values["text"])
                input_ids, attention_mask = tokenizer.batch_encode((prompt,))
                output = context.adapter.forward_loss(
                    context.model,
                    ModelBatch(
                        sample_ids=(sample.sample_id,),
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    ),
                )
                validate_single_sample_output(
                    sample.sample_id,
                    output.sample_ids,
                    len(output.negative_log_likelihoods),
                )
                if len(output.negative_log_likelihoods) != len(output.token_counts):
                    raise EvaluationError(
                        "Model adapter returned inconsistent loss and token-count lengths",
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
            successful_ids.append(sample.sample_id)
            losses.extend(output.negative_log_likelihoods)
            counts.extend(output.token_counts)
        token_count = sum(counts)
        if token_count <= 0:
            raise EvaluationError(
                "Perplexity evaluation produced no valid tokens",
                component=self.name,
            )
        mean_nll = (
            sum(loss * count for loss, count in zip(losses, counts, strict=True)) / token_count
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
                "mean_nll": mean_nll,
                "perplexity": perplexity_from_nll(mean_nll),
                "evaluated_tokens": token_count,
                "successful_samples": len(successful_ids),
            },
            failures=tuple(failures),
            status="partial" if failures else "success",
        )


class ForwardLossEvaluator(PerplexityEvaluator):
    """Alias with a narrower metric name for explicit forward-loss suites."""

    name = "forward_loss"
