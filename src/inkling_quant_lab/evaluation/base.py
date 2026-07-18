"""Evaluation schemas, contexts, and protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from string import Formatter
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from inkling_quant_lab.config import EvaluationSuiteConfig, ExperimentConfig
from inkling_quant_lab.data import LocalDataset
from inkling_quant_lab.exceptions import EvaluationError
from inkling_quant_lab.models.base import LoadedModel, ModelAdapter


class ImmutableEvaluationRecord(BaseModel):
    """Strict immutable evaluator result base."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class EvaluationFailure(ImmutableEvaluationRecord):
    """One retained sample- or suite-level failure."""

    sample_id: str | None = None
    code: str
    message: str


class EvaluationResult(ImmutableEvaluationRecord):
    """Machine-readable metric values with complete dataset provenance."""

    evaluator_name: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    dataset_revision: str = Field(min_length=1)
    split: str = Field(min_length=1)
    dataset_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sample_ids: tuple[str, ...]
    sample_count: int = Field(ge=0)
    seed: int
    prompt_template_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decode_config: dict[str, JsonValue]
    metrics: dict[str, float | int | str | bool | None]
    failures: tuple[EvaluationFailure, ...] = ()
    artifacts: tuple[str, ...] = ()
    status: Literal["success", "partial", "unsupported", "failed"] = "success"

    @model_validator(mode="after")
    def validate_sample_provenance(self) -> EvaluationResult:
        """Keep dataset membership complete even when individual samples fail."""

        if self.sample_count != len(self.sample_ids):
            raise ValueError("sample_count must equal the number of stable sample_ids")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("evaluation result sample_ids must be unique")
        unknown_failures = {
            failure.sample_id
            for failure in self.failures
            if failure.sample_id is not None and failure.sample_id not in self.sample_ids
        }
        if unknown_failures:
            raise ValueError(
                "failure records reference unknown sample_ids: "
                + ", ".join(sorted(unknown_failures))
            )
        if self.failures and self.status == "success":
            raise ValueError("a result with retained failures must have a non-success status")
        if self.status in {"partial", "unsupported", "failed"} and not self.failures:
            raise ValueError(f"a {self.status} result must retain at least one failure")
        if self.dataset_sha256 is None and self.status != "failed":
            raise ValueError("only failed dataset setup may omit the exact dataset SHA-256")
        return self

    def canonical_json(self) -> str:
        """Serialize a result byte-stably for immutable metric artifacts."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """Adapter, model, resolved config, suite, and loaded dataset."""

    adapter: ModelAdapter
    model: LoadedModel
    experiment: ExperimentConfig
    suite: EvaluationSuiteConfig
    dataset: LocalDataset


class Evaluator(Protocol):
    """Common deterministic evaluator contract."""

    name: str
    version: str

    def run(self, context: EvaluationContext) -> EvaluationResult: ...


class RequiredEvaluationFailure(EvaluationError):
    """Aggregate typed suite results while preserving required failure semantics."""

    def __init__(
        self,
        results: tuple[EvaluationResult, ...],
        failed_suites: tuple[str, ...],
    ) -> None:
        self.results = results
        self.failed_suites = failed_suites
        retained_messages = [
            f"{result.evaluator_name}: {failure.message}"
            for result in results
            if result.evaluator_name in failed_suites
            for failure in result.failures
        ]
        super().__init__(
            "Required evaluation suites failed: " + "; ".join(retained_messages),
            component="evaluation",
            details={
                "failed_suites": list(failed_suites),
                "evaluation_results": [result.model_dump(mode="json") for result in results],
            },
        )


class TokenizerLike(Protocol):
    """Architecture-neutral tokenizer surface required by local evaluators."""

    def batch_encode(self, texts: tuple[str, ...]) -> tuple[object, object]: ...


def prompt_template_hash(template: str) -> str:
    """Hash a prompt template without persisting its text in result artifacts."""

    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def render_prompt(template: str, text: str) -> str:
    """Apply the sole supported ``{text}`` field without arbitrary attribute access."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    saw_text = False
    for _, field_name, format_spec, conversion in Formatter().parse(template):
        if field_name is None:
            continue
        if field_name != "text" or format_spec or conversion:
            raise ValueError("prompt template may contain only the plain {text} field")
        saw_text = True
    if not saw_text:
        raise ValueError("prompt template must contain the {text} field")
    rendered = template.format(text=text)
    if not rendered.strip():
        raise ValueError("rendered prompt must not be empty")
    return rendered


def retain_invalid_sample(
    context: EvaluationContext,
    failures: list[EvaluationFailure],
    *,
    sample_id: str,
    evaluator_name: str,
) -> None:
    """Retain privacy-safe failure context or fail the suite under strict policy."""

    message = f"Sample does not satisfy the {evaluator_name} evaluator schema"
    failure = EvaluationFailure(
        sample_id=sample_id,
        code="INVALID_SAMPLE",
        message=message,
    )
    if not context.experiment.evaluation.allow_partial:
        raise EvaluationError(
            message,
            component=evaluator_name,
            details={"sample_id": sample_id, "code": failure.code},
        )
    failures.append(failure)


def complete_sample_ids(context: EvaluationContext) -> tuple[str, ...]:
    """Return all attempted stable IDs, including retained failures."""

    return tuple(sample.sample_id for sample in context.dataset.samples)


def validate_single_sample_output(
    expected_sample_id: str,
    actual_sample_ids: Sequence[str],
    output_count: int,
) -> None:
    """Reject adapter contract drift before metrics are computed."""

    if tuple(actual_sample_ids) != (expected_sample_id,) or output_count != 1:
        raise EvaluationError(
            "Model adapter returned output with mismatched sample identity",
            component="evaluation",
            details={"expected_sample_id": expected_sample_id},
        )
