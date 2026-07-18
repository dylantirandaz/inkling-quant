"""CPU evaluator correctness, provenance, determinism, and privacy tests."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pytest

from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.evaluation.base import (
    EvaluationResult,
    RequiredEvaluationFailure,
)
from inkling_quant_lab.evaluation.multimodal import MultimodalContractEvaluator
from inkling_quant_lab.evaluation.perplexity import perplexity_from_nll
from inkling_quant_lab.evaluation.runner import run_evaluations
from inkling_quant_lab.exceptions import CapabilityError, EvaluationError
from inkling_quant_lab.models.local import LocalFixtureAdapter
from inkling_quant_lab.runtimes.torch_cpu import TorchEagerCPURuntime

pytestmark = pytest.mark.unit


def _config_with_suite(
    config_factory: Callable[..., ExperimentConfig],
    suite: dict[str, object],
    *,
    model_id: str = "local://fixtures/tiny-dense",
    allow_partial: bool = True,
) -> ExperimentConfig:
    raw = config_factory(model_id=model_id, routing_mode="off").canonical_dict()
    raw["evaluation"] = {"allow_partial": allow_partial, "suites": [suite]}
    return ExperimentConfig.model_validate(raw)


def _evaluate(config: ExperimentConfig):
    adapter = LocalFixtureAdapter()
    loaded = adapter.load(config, TorchEagerCPURuntime())
    return run_evaluations(adapter, loaded, config)


def test_perplexity_equals_exp_of_token_weighted_mean_nll(
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """TC-EVAL-001: reported perplexity is exactly derived from mean NLL."""

    config = _config_with_suite(
        config_factory,
        {
            "type": "perplexity",
            "dataset": "local://fixtures/tiny-corpus",
            "revision": "fixture-data-v1",
            "split": "evaluation",
        },
    )

    result = _evaluate(config)[0]

    assert result.metrics["perplexity"] == pytest.approx(
        math.exp(float(result.metrics["mean_nll"]))
    )
    assert result.metrics["evaluated_tokens"] > 0


def test_generation_is_byte_stable_and_does_not_persist_prompts_or_outputs(
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """TC-EVAL-002: fixed-seed greedy generation produces byte-stable results."""

    config = _config_with_suite(
        config_factory,
        {
            "type": "generation_regression",
            "dataset": "local://fixtures/generation-prompts",
            "revision": "fixture-data-v1",
            "split": "evaluation",
            "prompt_template": "tiny {text}",
            "decode": {"max_new_tokens": 4, "do_sample": False},
        },
    )

    first = _evaluate(config)[0]
    second = _evaluate(config)[0]

    assert first.canonical_json().encode("utf-8") == second.canonical_json().encode("utf-8")
    assert first.metrics["output_token_hash"] == second.metrics["output_token_hash"]
    assert first.artifacts == ()
    serialized = first.canonical_json()
    assert "alpha beta" not in serialized
    assert "tiny {text}" not in serialized
    assert "token_ids" not in serialized
    assert "texts" not in serialized


def test_malformed_sample_failure_is_retained_without_leaking_content(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """TC-EVAL-003: one bad sample does not erase valid results under allow_partial."""

    secret = "prompt-secret-must-not-persist"
    dataset = tmp_path / "partial.jsonl"
    dataset.write_text(
        f'{{"id":"good","text":"alpha beta"}}\n{{"id":"bad","text":["{secret}"]}}\n',
        encoding="utf-8",
    )
    config = _config_with_suite(
        config_factory,
        {
            "type": "perplexity",
            "dataset": dataset.as_uri(),
            "revision": "local-v1",
            "split": "test",
        },
    )

    result = _evaluate(config)[0]

    assert result.status == "partial"
    assert result.sample_ids == ("good", "bad")
    assert result.sample_count == 2
    assert result.metrics["successful_samples"] == 1
    assert len(result.failures) == 1
    assert result.failures[0].sample_id == "bad"
    assert secret not in result.canonical_json()


def test_result_has_complete_dataset_metadata_and_stable_selected_ids(
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """TC-EVAL-004: provenance and configured sample identity are complete."""

    config = _config_with_suite(
        config_factory,
        {
            "type": "forward_loss",
            "dataset": "local://fixtures/tiny-corpus",
            "revision": "fixture-data-v1",
            "split": "held-out",
            "sample_ids": ["eval-loss-003", "eval-loss-001"],
        },
    )

    result = _evaluate(config)[0]

    assert result.evaluator_name == "forward_loss"
    assert result.evaluator_version == "1.0"
    assert result.dataset_id == "local://fixtures/tiny-corpus"
    assert result.dataset_revision == "fixture-data-v1"
    assert result.split == "held-out"
    assert len(result.dataset_sha256) == 64
    assert result.sample_ids == ("eval-loss-003", "eval-loss-001")
    assert result.sample_count == 2
    assert result.seed == 17
    assert len(result.prompt_template_hash) == 64
    assert result.decode_config["do_sample"] is False


def test_exact_match_adapter_uses_generation_contract(
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """The exact-match task adapter produces deterministic task metrics and metadata."""

    config = _config_with_suite(
        config_factory,
        {
            "type": "exact_match",
            "dataset": "local://fixtures/generation-prompts",
            "revision": "fixture-data-v1",
            "split": "evaluation",
        },
    )

    first = _evaluate(config)[0]
    second = _evaluate(config)[0]

    assert first.evaluator_name == "exact_match"
    assert 0.0 <= float(first.metrics["exact_match"]) <= 1.0
    assert first.metrics["matched_samples"] <= first.metrics["successful_samples"]
    assert first.canonical_json() == second.canonical_json()


def test_multimodal_stub_contract_and_unsupported_dense_model(
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """The multimodal evaluator executes the stub and rejects unsupported adapters."""

    suite = {
        "type": "multimodal_contract",
        "dataset": "local://fixtures/multimodal",
        "revision": "fixture-data-v1",
        "split": "evaluation",
    }
    multimodal = _config_with_suite(
        config_factory,
        suite,
        model_id="local://fixtures/tiny-multimodal",
    )
    result = _evaluate(multimodal)[0]

    assert result.evaluator_name == MultimodalContractEvaluator.name
    assert result.metrics == {
        "contract_passed": True,
        "executed_samples": 2,
        "successful_samples": 2,
    }
    assert result.sample_count == 2
    dense = _config_with_suite(config_factory, suite)
    with pytest.raises(EvaluationError, match="Required evaluation suites failed") as captured:
        _evaluate(dense)
    retained = captured.value.details["evaluation_results"]
    assert len(retained) == 1
    failed = EvaluationResult.model_validate(retained[0])
    assert failed.status == "unsupported"
    assert failed.failures[0].sample_id is None
    assert "does not support multimodal" in failed.failures[0].message


def test_optional_suite_capability_and_execution_failures_are_typed_results(
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """Optional suite failures remain reportable without hiding their status."""

    unsupported = _config_with_suite(
        config_factory,
        {
            "type": "multimodal_contract",
            "dataset": "local://fixtures/multimodal",
            "revision": "fixture-data-v1",
            "split": "evaluation",
            "optional": True,
        },
    )
    unsupported_result = _evaluate(unsupported)[0]

    assert unsupported_result.status == "unsupported"
    assert unsupported_result.metrics == {}
    assert unsupported_result.failures[0].code == CapabilityError.code
    assert unsupported_result.failures[0].sample_id is None

    failed = _config_with_suite(
        config_factory,
        {
            "type": "generation_regression",
            "dataset": "local://fixtures/generation-prompts",
            "revision": "fixture-data-v1",
            "split": "evaluation",
            "decode": {"do_sample": True},
            "optional": True,
        },
    )
    failed_result = _evaluate(failed)[0]

    assert failed_result.status == "failed"
    assert failed_result.failures[0].code == EvaluationError.code
    assert "do_sample=false" in failed_result.failures[0].message


def test_required_dataset_load_failure_is_a_typed_suite_result(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """Dataset setup failure retains configured identity without inventing a checksum."""

    missing = tmp_path / "missing-evaluation.jsonl"
    config = _config_with_suite(
        config_factory,
        {
            "type": "perplexity",
            "dataset": missing.as_uri(),
            "revision": "local-v1",
            "split": "evaluation",
        },
    )

    with pytest.raises(RequiredEvaluationFailure) as captured:
        _evaluate(config)

    result = captured.value.results[0]
    assert captured.value.failed_suites == ("perplexity",)
    assert result.status == "failed"
    assert result.dataset_id == missing.as_uri()
    assert result.dataset_sha256 is None
    assert result.sample_ids == ()
    assert result.metrics == {}
    assert result.failures[0].code == EvaluationError.code
    assert "Unable to read dataset" in result.failures[0].message
    assert "readable UTF-8 JSONL" in result.failures[0].message


def test_invalid_dataset_encoding_is_a_typed_load_failure(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """A bounded text-decoding failure follows the same retained setup contract."""

    invalid = tmp_path / "invalid-utf8.jsonl"
    invalid.write_bytes(b"\xff\xfe")
    config = _config_with_suite(
        config_factory,
        {
            "type": "perplexity",
            "dataset": invalid.as_uri(),
            "revision": "local-v1",
            "split": "evaluation",
        },
    )

    with pytest.raises(RequiredEvaluationFailure) as captured:
        _evaluate(config)

    result = captured.value.results[0]
    assert result.status == "failed"
    assert result.dataset_sha256 is None
    assert "Unable to read dataset" in result.failures[0].message
    assert "readable UTF-8 JSONL" in result.failures[0].message


def test_required_dataset_selection_failure_retains_loaded_provenance(
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """Configured missing sample IDs fail as evidence with the loaded content digest."""

    config = _config_with_suite(
        config_factory,
        {
            "type": "perplexity",
            "dataset": "local://fixtures/tiny-corpus",
            "revision": "fixture-data-v1",
            "split": "evaluation",
            "sample_ids": ["absent-sample"],
        },
    )

    with pytest.raises(RequiredEvaluationFailure) as captured:
        _evaluate(config)

    result = captured.value.results[0]
    assert result.status == "failed"
    assert result.dataset_sha256 is not None
    assert len(result.dataset_sha256) == 64
    assert result.sample_ids == ("absent-sample",)
    assert result.sample_count == 1
    assert result.failures[0].code == EvaluationError.code
    assert "absent from the dataset" in result.failures[0].message
    assert "exist in the configured dataset revision" in result.failures[0].message


def test_duplicate_dataset_selection_failure_is_typed_without_ambiguous_ids(
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """Duplicate requested IDs retain the selection error and a canonical evidence set."""

    config = _config_with_suite(
        config_factory,
        {
            "type": "perplexity",
            "dataset": "local://fixtures/tiny-corpus",
            "revision": "fixture-data-v1",
            "split": "evaluation",
            "sample_ids": ["eval-loss-001", "eval-loss-001"],
        },
    )

    with pytest.raises(RequiredEvaluationFailure) as captured:
        _evaluate(config)

    result = captured.value.results[0]
    assert result.status == "failed"
    assert result.dataset_sha256 is not None
    assert result.sample_ids == ("eval-loss-001",)
    assert "must be unique" in result.failures[0].message
    assert "Remove duplicate IDs" in result.failures[0].message


@pytest.mark.parametrize("failure_kind", ("load", "selection"))
def test_optional_dataset_setup_failure_remains_a_normal_typed_result(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
    failure_kind: str,
) -> None:
    """Optional setup failures do not escape or erase their suite evidence."""

    suite: dict[str, object] = {
        "type": "perplexity",
        "dataset": "local://fixtures/tiny-corpus",
        "revision": "fixture-data-v1",
        "split": "evaluation",
        "optional": True,
    }
    if failure_kind == "load":
        suite["dataset"] = (tmp_path / "missing-optional.jsonl").as_uri()
    else:
        suite["sample_ids"] = ["absent-optional-sample"]
    config = _config_with_suite(config_factory, suite)

    result = _evaluate(config)[0]

    assert result.status == "failed"
    assert result.metrics == {}
    assert result.failures[0].code == EvaluationError.code
    if failure_kind == "load":
        assert result.dataset_sha256 is None
    else:
        assert result.dataset_sha256 is not None


def test_generation_rejects_sampling_and_retains_schema_failures_privately(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """Regression evaluators require greedy decode and redact malformed expected output."""

    sampling = _config_with_suite(
        config_factory,
        {
            "type": "generation_regression",
            "dataset": "local://fixtures/generation-prompts",
            "revision": "fixture-data-v1",
            "split": "evaluation",
            "decode": {"do_sample": True},
        },
    )
    with pytest.raises(EvaluationError, match="do_sample=false"):
        _evaluate(sampling)

    secret = "expected-output-secret"
    dataset = tmp_path / "generation-partial.jsonl"
    dataset.write_text(
        '{"id":"good","text":"alpha beta","expected_tokens":[2,2,2,2]}\n'
        f'{{"id":"bad","text":"red blue","expected_tokens":["{secret}"]}}\n',
        encoding="utf-8",
    )
    partial = _config_with_suite(
        config_factory,
        {
            "type": "exact_match",
            "dataset": dataset.as_uri(),
            "revision": "local-v1",
            "split": "evaluation",
        },
    )

    result = _evaluate(partial)[0]

    assert result.status == "partial"
    assert result.metrics["successful_samples"] == 1
    assert secret not in result.canonical_json()


def test_multimodal_partial_failure_is_retained_as_unavailable_sample(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """Malformed modality vectors retain IDs and generic failure details."""

    dataset = tmp_path / "multimodal-partial.jsonl"
    dataset.write_text(
        '{"id":"good","text":"image answer","multimodal":[1,0,0.5,-0.5]}\n'
        '{"id":"bad","text":"sound answer","multimodal":[true,0,0,0]}\n',
        encoding="utf-8",
    )
    config = _config_with_suite(
        config_factory,
        {
            "type": "multimodal_contract",
            "dataset": dataset.as_uri(),
            "revision": "local-v1",
            "split": "evaluation",
        },
        model_id="local://fixtures/tiny-multimodal",
    )

    result = _evaluate(config)[0]

    assert result.status == "partial"
    assert result.sample_ids == ("good", "bad")
    assert result.metrics["successful_samples"] == 1
    assert result.failures[0].message == (
        "Sample does not satisfy the multimodal_contract evaluator schema"
    )


def test_strict_partial_policy_and_nonfinite_perplexity_fail_actionably(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    """Strict suites stop at malformed input and perplexity rejects nonfinite NLL."""

    dataset = tmp_path / "invalid.jsonl"
    dataset.write_text('{"id":"bad","text":null}\n', encoding="utf-8")
    config = _config_with_suite(
        config_factory,
        {
            "type": "perplexity",
            "dataset": dataset.as_uri(),
            "revision": "local-v1",
            "split": "evaluation",
        },
        allow_partial=False,
    )

    with pytest.raises(EvaluationError, match="does not satisfy"):
        _evaluate(config)
    with pytest.raises(ValueError, match="finite"):
        perplexity_from_nll(float("nan"))
