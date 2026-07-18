"""Contract evaluator for the tiny multimodal adapter stub."""

from __future__ import annotations

import math
from importlib import import_module
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
from inkling_quant_lab.exceptions import CapabilityError, EvaluationError
from inkling_quant_lab.models.base import ModelBatch


class MultimodalContractEvaluator:
    """Verify multimodal input execution without making quality claims."""

    name = "multimodal_contract"
    version = "1.0"

    def run(self, context: EvaluationContext) -> EvaluationResult:
        """Execute every valid modality vector and report capability success."""

        if not context.model.descriptor.capabilities.supports_images:
            raise CapabilityError(
                "Selected model does not support multimodal inputs",
                component=self.name,
            )
        tokenizer = cast(TokenizerLike, context.model.tokenizer)
        torch_module = import_module("torch")
        tensor = torch_module.tensor
        float32 = torch_module.float32
        successful_ids: list[str] = []
        failures: list[EvaluationFailure] = []
        for sample in context.dataset.samples:
            try:
                values = sample.values["multimodal"]
                if not isinstance(values, list) or len(values) != 4:
                    raise ValueError("multimodal samples require four numeric values")
                if any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    for value in values
                ):
                    raise ValueError("multimodal values must be finite numbers")
                prompt = render_prompt(
                    context.suite.prompt_template,
                    sample.values["text"],
                )
                input_ids, attention_mask = tokenizer.batch_encode((prompt,))
                output = context.adapter.forward_loss(
                    context.model,
                    ModelBatch(
                        sample_ids=(sample.sample_id,),
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        multimodal_inputs=tensor((values,), dtype=float32),
                    ),
                )
                validate_single_sample_output(
                    sample.sample_id,
                    output.sample_ids,
                    len(output.negative_log_likelihoods),
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
        if not successful_ids:
            raise EvaluationError(
                "Multimodal evaluation produced no valid samples",
                component=self.name,
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
                "contract_passed": True,
                "executed_samples": len(successful_ids),
                "successful_samples": len(successful_ids),
            },
            failures=tuple(failures),
            status="partial" if failures else "success",
        )
