"""Deterministic selection contracts for the public-model quality script."""

from __future__ import annotations

from pathlib import Path

import pytest

from inkling_quant_lab.config import ExperimentConfig, load_config
from scripts import evaluate_public_moe_quality as quality_script
from scripts.evaluate_public_moe_quality import (
    OFFICIAL_TINYSTORIES_VALID_SHA256,
    evaluate,
    require_official_dataset_sha256,
    sample_index,
    select_stories,
    split_stories,
    validate_quality_contract,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUALITY_CONFIG = PROJECT_ROOT / "configs/experiments/hf_stories15m_tinystories_quality.yaml"


def test_tinystories_split_and_declared_selection_order() -> None:
    stories = split_stories(" first \n<|endoftext|>\nsecond\n<|endoftext|>\n third ")

    selected = select_stories(stories, ("story-000002", "story-000000"))

    assert stories == ("first", "second", "third")
    assert selected == ("third", "first")


@pytest.mark.parametrize("sample_id", ["story-1", "sample-000001", "story-abcdef"])
def test_sample_index_rejects_unstable_identifiers(sample_id: str) -> None:
    with pytest.raises(ValueError, match="unsupported TinyStories sample ID"):
        sample_index(sample_id)


def test_selection_rejects_duplicates_and_out_of_range() -> None:
    stories = ("first", "second")
    with pytest.raises(ValueError, match="unique"):
        select_stories(stories, ("story-000001", "story-000001"))
    with pytest.raises(ValueError, match="outside"):
        select_stories(stories, ("story-000002",))


def test_checked_baseline_quality_contract_is_exact() -> None:
    config = load_config(QUALITY_CONFIG)

    validate_quality_contract(config)
    assert require_official_dataset_sha256(OFFICIAL_TINYSTORIES_VALID_SHA256) == (
        OFFICIAL_TINYSTORIES_VALID_SHA256
    )


def test_baseline_evaluator_rejects_caller_selected_digest_before_reading_data(
    tmp_path: Path,
) -> None:
    config = load_config(QUALITY_CONFIG)

    with pytest.raises(ValueError, match="official TinyStories validation SHA-256"):
        evaluate(
            config,
            tmp_path / "not-opened.txt",
            expected_dataset_sha256="0" * 64,
            project_root=tmp_path,
        )
    with pytest.raises(ValueError, match="official TinyStories validation SHA-256"):
        require_official_dataset_sha256("0" * 64)
    with pytest.raises(SystemExit):
        quality_script._arguments(
            [
                str(QUALITY_CONFIG),
                "TinyStories-valid.txt",
                "--expected-dataset-sha256",
                "0" * 64,
            ]
        )


def test_baseline_evaluator_rejects_ignored_prompt_template() -> None:
    config = load_config(QUALITY_CONFIG)
    raw = config.canonical_dict()
    raw["evaluation"]["suites"][0]["prompt_template"] = "Story: {text}"
    changed = ExperimentConfig.model_validate(raw)

    with pytest.raises(ValueError, match="exact '\\{text\\}' prompt template"):
        validate_quality_contract(changed)
