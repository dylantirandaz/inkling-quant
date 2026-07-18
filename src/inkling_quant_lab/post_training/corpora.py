"""Immutable, deterministic corpus provenance for post-training experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CorpusSplit: TypeAlias = Literal["train", "validation", "test"]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SAMPLE_HASH_DOMAIN = b"inkling-quant-corpus-sample-v1\0"
_EVALUATION_SPLITS = frozenset({"validation", "test"})


class _ImmutableCorpusRecord(BaseModel):
    """Strict immutable base for corpus evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def content_sha256(text: str) -> str:
    """Hash the exact UTF-8 bytes of one parsed training/evaluation text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_labels(labels: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(labels)
    if not normalized:
        raise ValueError("corpus sample labels must not be empty")
    if any(not label or label != label.strip() for label in normalized):
        raise ValueError("corpus sample labels must be non-empty and whitespace-canonical")
    if len(set(normalized)) != len(normalized):
        raise ValueError("corpus sample labels must be unique")
    return tuple(sorted(normalized))


def stable_sample_sha256(
    *,
    sample_id: str,
    content_digest: str,
    split: CorpusSplit,
    labels: Sequence[str],
    token_count: int,
) -> str:
    """Hash the canonical parsed sample record without retaining its text."""

    canonical = json.dumps(
        {
            "content_sha256": content_digest,
            "labels": list(_canonical_labels(labels)),
            "sample_id": sample_id,
            "split": split,
            "token_count": token_count,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(_SAMPLE_HASH_DOMAIN + canonical).hexdigest()


class CorpusSourceContract(_ImmutableCorpusRecord):
    """Exact source bytes, identity, license declaration, and parser contract."""

    dataset_id: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    declared_license: str = Field(min_length=1)
    size_bytes: int = Field(gt=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    encoding: Literal["utf-8"] = "utf-8"
    parser_version: str = Field(min_length=1)

    @field_validator("dataset_id", "revision", "declared_license", "parser_version")
    @classmethod
    def identity_fields_are_whitespace_canonical(cls, value: str) -> str:
        """Reject visually ambiguous provenance strings."""

        if value != value.strip():
            raise ValueError("corpus provenance fields must not have surrounding whitespace")
        return value

    def verify_bytes(self, data: bytes) -> str:
        """Verify size and digest, then decode the exact source bytes as strict UTF-8."""

        if len(data) != self.size_bytes:
            raise ValueError(
                f"corpus byte size mismatch: expected {self.size_bytes}, observed {len(data)}"
            )
        observed_sha256 = hashlib.sha256(data).hexdigest()
        if observed_sha256 != self.sha256:
            raise ValueError("corpus SHA-256 does not match the declared source digest")
        try:
            return data.decode(self.encoding, errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("corpus source is not valid strict UTF-8") from error


class CorpusSampleContract(_ImmutableCorpusRecord):
    """One parsed sample with stable content, membership, label, and token evidence."""

    sample_id: str = Field(min_length=1)
    sample_sha256: str = Field(pattern=_SHA256_PATTERN)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)
    split: CorpusSplit
    labels: tuple[str, ...]
    token_count: int = Field(gt=0)

    @field_validator("sample_id")
    @classmethod
    def sample_id_is_whitespace_canonical(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError("corpus sample IDs must not have surrounding whitespace")
        return value

    @field_validator("labels")
    @classmethod
    def labels_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _canonical_labels(value)

    @model_validator(mode="after")
    def sample_hash_matches_canonical_record(self) -> Self:
        expected = stable_sample_sha256(
            sample_id=self.sample_id,
            content_digest=self.content_sha256,
            split=self.split,
            labels=self.labels,
            token_count=self.token_count,
        )
        if self.sample_sha256 != expected:
            raise ValueError("sample SHA-256 does not match the canonical parsed sample record")
        return self

    @classmethod
    def from_text(
        cls,
        *,
        sample_id: str,
        text: str,
        split: CorpusSplit,
        labels: Sequence[str],
        token_count: int,
    ) -> CorpusSampleContract:
        """Construct exact content and canonical-record digests without retaining text."""

        digest = content_sha256(text)
        return cls(
            sample_id=sample_id,
            sample_sha256=stable_sample_sha256(
                sample_id=sample_id,
                content_digest=digest,
                split=split,
                labels=labels,
                token_count=token_count,
            ),
            content_sha256=digest,
            split=split,
            labels=tuple(labels),
            token_count=token_count,
        )


def _train_evaluation_overlap(
    corpora: Sequence[tuple[CorpusSourceContract, tuple[CorpusSampleContract, ...]]],
) -> tuple[str, ...]:
    train_hashes: set[str] = set()
    evaluation_hashes: set[str] = set()
    for _, samples in corpora:
        for sample in samples:
            if sample.split == "train":
                train_hashes.add(sample.content_sha256)
            elif sample.split in _EVALUATION_SPLITS:
                evaluation_hashes.add(sample.content_sha256)
    return tuple(sorted(train_hashes & evaluation_hashes))


class CorpusContract(_ImmutableCorpusRecord):
    """One source's exact parsed selection and selected-token budget."""

    schema_version: Literal["corpus-contract-v1"] = "corpus-contract-v1"
    source: CorpusSourceContract
    samples: tuple[CorpusSampleContract, ...] = Field(min_length=1)
    token_budget: int = Field(gt=0)

    @model_validator(mode="after")
    def samples_are_unique_disjoint_and_budgeted(self) -> Self:
        sample_ids = tuple(sample.sample_id for sample in self.samples)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("corpus sample IDs must be unique within one source")
        sample_hashes = tuple(sample.sample_sha256 for sample in self.samples)
        if len(set(sample_hashes)) != len(sample_hashes):
            raise ValueError("corpus canonical sample hashes must be unique within one source")
        selected_tokens = sum(sample.token_count for sample in self.samples)
        if selected_tokens != self.token_budget:
            raise ValueError(
                "corpus token budget must equal the exact sum of selected sample token counts"
            )
        overlap = _train_evaluation_overlap(((self.source, self.samples),))
        if overlap:
            raise ValueError(
                "training/evaluation content-hash overlap is forbidden: " + ", ".join(overlap)
            )
        return self

    def verify_source_bytes(self, data: bytes) -> str:
        """Verify and decode the bound source bytes through its exact source contract."""

        return self.source.verify_bytes(data)

    def canonical_json(self) -> str:
        """Serialize this contract deterministically for hashing and immutable artifacts."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


class CorpusCollectionContract(_ImmutableCorpusRecord):
    """Multiple source contracts with global train/evaluation leakage protection."""

    schema_version: Literal["corpus-collection-contract-v1"] = "corpus-collection-contract-v1"
    corpora: tuple[CorpusContract, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def sources_are_unique_and_globally_disjoint(self) -> Self:
        identities = tuple(
            (corpus.source.dataset_id, corpus.source.revision) for corpus in self.corpora
        )
        if len(set(identities)) != len(identities):
            raise ValueError("corpus collection source dataset ID/revision pairs must be unique")
        overlap = _train_evaluation_overlap(
            tuple((corpus.source, corpus.samples) for corpus in self.corpora)
        )
        if overlap:
            raise ValueError(
                "training/evaluation content-hash overlap across corpus sources is forbidden: "
                + ", ".join(overlap)
            )
        return self

    @property
    def token_budget(self) -> int:
        """Return the exact combined selected-token budget."""

        return sum(corpus.token_budget for corpus in self.corpora)

    def canonical_json(self) -> str:
        """Serialize the collection deterministically for immutable artifacts."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


__all__ = [
    "CorpusCollectionContract",
    "CorpusContract",
    "CorpusSampleContract",
    "CorpusSourceContract",
    "CorpusSplit",
    "content_sha256",
    "stable_sample_sha256",
]
