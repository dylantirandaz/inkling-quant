"""Deterministic, CPU-only post-training corpus provenance contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from inkling_quant_lab.post_training.corpora import (
    CorpusCollectionContract,
    CorpusContract,
    CorpusSampleContract,
    CorpusSourceContract,
    content_sha256,
    stable_sample_sha256,
)

pytestmark = pytest.mark.unit


def _source(
    *,
    dataset_id: str = "public-domain://alice",
    revision: str = "gutenberg-11-2025-06-01",
    data: bytes = b"Alice was beginning to get very tired.\n",
) -> CorpusSourceContract:
    return CorpusSourceContract(
        dataset_id=dataset_id,
        revision=revision,
        declared_license="Public domain in the United States",
        size_bytes=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        encoding="utf-8",
        parser_version="plain-text-documents-v1",
    )


def _sample(
    sample_id: str,
    text: str,
    split: str,
    *,
    labels: tuple[str, ...] = ("domain:fiction", "source:alice"),
    token_count: int = 8,
) -> CorpusSampleContract:
    return CorpusSampleContract.from_text(
        sample_id=sample_id,
        text=text,
        split=split,  # type: ignore[arg-type]
        labels=labels,
        token_count=token_count,
    )


def _corpus(
    *,
    source: CorpusSourceContract | None = None,
    samples: tuple[CorpusSampleContract, ...] | None = None,
) -> CorpusContract:
    selected = samples or (
        _sample("alice-train-001", "Down the rabbit hole.", "train"),
        _sample("alice-test-001", "A mad tea party.", "test", token_count=7),
    )
    return CorpusContract(
        source=source or _source(),
        samples=selected,
        token_budget=sum(sample.token_count for sample in selected),
    )


def _sample_payload(sample: CorpusSampleContract) -> dict[str, Any]:
    return sample.model_dump(mode="python")


def test_source_contract_verifies_exact_size_digest_and_strict_utf8() -> None:
    data = "A deterministic café corpus.\n".encode()
    source = _source(data=data)

    assert source.verify_bytes(data) == data.decode("utf-8")

    with pytest.raises(ValueError, match="byte size mismatch"):
        source.verify_bytes(data + b"!")
    with pytest.raises(ValueError, match="SHA-256"):
        source.verify_bytes(b"x" * len(data))

    invalid_utf8 = b"\xff"
    invalid_source = _source(data=invalid_utf8)
    with pytest.raises(ValueError, match="strict UTF-8"):
        invalid_source.verify_bytes(invalid_utf8)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("dataset_id", ""),
        ("revision", " revision-with-leading-space"),
        ("declared_license", ""),
        ("parser_version", "parser-v1 "),
        ("encoding", "utf-16"),
        ("sha256", "A" * 64),
        ("size_bytes", 0),
    ),
)
def test_source_contract_rejects_incomplete_or_noncanonical_provenance(
    field: str, value: object
) -> None:
    payload = _source().model_dump(mode="python")
    payload[field] = value

    with pytest.raises(ValidationError):
        CorpusSourceContract.model_validate(payload)


def test_content_and_sample_hashes_are_exact_and_label_order_independent() -> None:
    text = "Exact Unicode bytes: naïve."
    digest = content_sha256(text)
    first = CorpusSampleContract.from_text(
        sample_id="sample-001",
        text=text,
        split="train",
        labels=("source:alice", "domain:fiction"),
        token_count=5,
    )
    second = CorpusSampleContract.from_text(
        sample_id="sample-001",
        text=text,
        split="train",
        labels=("domain:fiction", "source:alice"),
        token_count=5,
    )

    assert digest == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert first.content_sha256 == digest
    assert first.labels == ("domain:fiction", "source:alice")
    assert first.sample_sha256 == second.sample_sha256
    assert first.sample_sha256 == stable_sample_sha256(
        sample_id="sample-001",
        content_digest=digest,
        split="train",
        labels=first.labels,
        token_count=5,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sample_id", "different-id"),
        ("content_sha256", "0" * 64),
        ("split", "test"),
        ("labels", ("domain:other",)),
        ("token_count", 99),
        ("sample_sha256", "0" * 64),
    ),
)
def test_canonical_sample_hash_rejects_metadata_or_digest_tampering(
    field: str, value: object
) -> None:
    payload = _sample_payload(_sample("sample-001", "bound text", "train"))
    payload[field] = value

    with pytest.raises(ValidationError, match="sample SHA-256"):
        CorpusSampleContract.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sample_id", " sample-001"),
        ("labels", ()),
        ("labels", ("domain:fiction", "domain:fiction")),
        ("labels", (" domain:fiction",)),
        ("token_count", 0),
        ("split", "evaluation"),
    ),
)
def test_sample_contract_rejects_invalid_identity_labels_budget_or_split(
    field: str, value: object
) -> None:
    payload = _sample_payload(_sample("sample-001", "bound text", "train"))
    payload[field] = value

    with pytest.raises(ValidationError):
        CorpusSampleContract.model_validate(payload)


def test_corpus_binds_exact_selected_token_budget_and_unique_sample_ids() -> None:
    corpus = _corpus()
    assert corpus.token_budget == 15

    with pytest.raises(ValidationError, match="exact sum"):
        CorpusContract(
            source=corpus.source,
            samples=corpus.samples,
            token_budget=corpus.token_budget + 1,
        )

    duplicate = _sample("alice-train-001", "Different content.", "train")
    with pytest.raises(ValidationError, match="sample IDs must be unique"):
        CorpusContract(
            source=corpus.source,
            samples=(corpus.samples[0], duplicate),
            token_budget=corpus.samples[0].token_count + duplicate.token_count,
        )


@pytest.mark.parametrize("evaluation_split", ("validation", "test"))
def test_corpus_rejects_train_evaluation_content_hash_overlap(
    evaluation_split: str,
) -> None:
    repeated_text = "This exact content must not cross the split boundary."
    train = _sample("train-001", repeated_text, "train")
    evaluation = _sample("evaluation-001", repeated_text, evaluation_split)

    with pytest.raises(ValidationError, match="content-hash overlap is forbidden") as error:
        _corpus(samples=(train, evaluation))

    assert train.content_sha256 in str(error.value)


def test_validation_and_test_may_share_content_without_inventing_train_leakage() -> None:
    repeated_text = "Evaluation-only repeated fixture."
    validation = _sample("validation-001", repeated_text, "validation")
    test = _sample("test-001", repeated_text, "test")

    corpus = _corpus(samples=(validation, test))

    assert corpus.token_budget == validation.token_count + test.token_count


def test_collection_rejects_cross_source_train_evaluation_overlap() -> None:
    repeated_text = "Cross-source leakage is still leakage."
    train = _corpus(
        source=_source(dataset_id="public-domain://train", revision="train-v1"),
        samples=(_sample("train-001", repeated_text, "train"),),
    )
    evaluation = _corpus(
        source=_source(dataset_id="public-domain://evaluation", revision="eval-v1"),
        samples=(_sample("eval-001", repeated_text, "test"),),
    )

    with pytest.raises(ValidationError, match="across corpus sources") as error:
        CorpusCollectionContract(corpora=(train, evaluation))

    assert train.samples[0].content_sha256 in str(error.value)


def test_collection_requires_unique_source_identity_and_sums_token_budgets() -> None:
    first = _corpus()
    second = _corpus(
        source=_source(dataset_id="public-domain://sherlock", revision="gutenberg-1661-v1"),
        samples=(_sample("sherlock-train-001", "A study in scarlet.", "train"),),
    )
    collection = CorpusCollectionContract(corpora=(first, second))

    assert collection.token_budget == first.token_budget + second.token_budget

    with pytest.raises(ValidationError, match="dataset ID/revision pairs must be unique"):
        CorpusCollectionContract(corpora=(first, first))


def test_contracts_are_frozen_extra_forbidden_and_byte_stably_serializable() -> None:
    corpus = _corpus()
    canonical = corpus.canonical_json()
    restored = CorpusContract.model_validate_json(canonical)

    assert restored == corpus
    assert restored.canonical_json() == canonical
    assert json.loads(canonical)["source"]["declared_license"] == (
        "Public domain in the United States"
    )
    assert "Down the rabbit hole" not in canonical
    assert "A mad tea party" not in canonical
    with pytest.raises(ValidationError, match="frozen"):
        corpus.token_budget = 1  # type: ignore[misc]

    source_payload = corpus.source.model_dump(mode="python")
    source_payload["access_token"] = "must-not-enter-contracts"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CorpusSourceContract.model_validate(source_payload)


def test_collection_canonical_json_is_independent_of_mapping_formatting() -> None:
    collection = CorpusCollectionContract(corpora=(_corpus(),))
    canonical = collection.canonical_json()

    assert CorpusCollectionContract.model_validate_json(canonical).canonical_json() == canonical
    assert ": " not in canonical
    assert ", " not in canonical
