"""Deterministic generation regression and exact-match evaluation."""

from __future__ import annotations

import hashlib
import json
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


class GenerationRegressionEvaluator:
    """Compare generated token IDs while persisting only hashes by default."""

    name = "generation_regression"
    version = "1.0"

    def run(self, context: EvaluationContext) -> EvaluationResult:
        """Run fixed greedy decoding and compute token-sequence exact match."""

        if context.suite.decode.do_sample:
            raise EvaluationError(
                "Generation regression requires deterministic decoding with do_sample=false",
                component=self.name,
            )
        tokenizer = cast(TokenizerLike, context.model.tokenizer)
        successful_ids: list[str] = []
        failures: list[EvaluationFailure] = []
        matches = 0
        output_hashes: list[str] = []
        for sample in context.dataset.samples:
            try:
                text = sample.values["text"]
                expected = sample.values["expected_tokens"]
                if not isinstance(expected, list) or not expected:
                    raise ValueError("text and expected_tokens are required")
                if any(
                    not isinstance(token, int) or isinstance(token, bool) or token < 0
                    for token in expected
                ):
                    raise ValueError("expected_tokens must contain non-negative integers")
                expected_tokens = tuple(expected)
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
            except (KeyError, TypeError, ValueError):
                retain_invalid_sample(
                    context,
                    failures,
                    sample_id=sample.sample_id,
                    evaluator_name=self.name,
                )
                continue
            actual = output.token_ids[0]
            matches += int(actual == expected_tokens)
            successful_ids.append(sample.sample_id)
            output_hashes.append(
                hashlib.sha256(
                    json.dumps(
                        {"sample_id": sample.sample_id, "tokens": actual},
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest()
            )
        if not successful_ids:
            raise EvaluationError(
                "Generation evaluation produced no valid samples",
                component=self.name,
            )
        digest = hashlib.sha256("\n".join(output_hashes).encode("ascii")).hexdigest()
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
                "exact_match": matches / len(successful_ids),
                "matched_samples": matches,
                "successful_samples": len(successful_ids),
                "output_token_hash": digest,
            },
            failures=tuple(failures),
            status="partial" if failures else "success",
        )


class ExactMatchEvaluator(GenerationRegressionEvaluator):
    """Exact-match task adapter sharing deterministic generation behavior."""

    name = "exact_match"
