"""Immutable local JSONL dataset loading and stable sample identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from inkling_quant_lab.exceptions import EvaluationError

_FIXTURE_URIS = {
    "local://fixtures/tiny-corpus": "tiny-corpus.jsonl",
    "local://fixtures/generation-prompts": "generation-prompts.jsonl",
    "local://fixtures/routing-prompts": "routing-prompts.jsonl",
    "local://fixtures/behavior-prompts": "behavior-prompts.jsonl",
    "local://fixtures/calibration": "calibration.jsonl",
    "local://fixtures/gptq-calibration": "gptq-calibration.jsonl",
    "local://fixtures/multimodal": "multimodal.jsonl",
}


@dataclass(frozen=True, slots=True)
class DatasetSample:
    """One raw sample whose fields are validated by the selected evaluator."""

    sample_id: str
    values: dict[str, Any]
    line_number: int


@dataclass(frozen=True, slots=True)
class LocalDataset:
    """Loaded dataset plus exact content checksum."""

    dataset_id: str
    revision: str
    split: str
    sha256: str
    samples: tuple[DatasetSample, ...]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Return stable IDs in evaluation order."""

        return tuple(sample.sample_id for sample in self.samples)


def _dataset_text(dataset_id: str) -> str:
    if dataset_id in _FIXTURE_URIS:
        resource = resources.files("inkling_quant_lab.fixture_data").joinpath(
            _FIXTURE_URIS[dataset_id]
        )
        return resource.read_text(encoding="utf-8")
    if dataset_id.startswith("file://"):
        path = Path(dataset_id.removeprefix("file://")).expanduser().resolve()
        return path.read_text(encoding="utf-8")
    raise EvaluationError(
        f"Unsupported dataset identifier: {dataset_id}",
        component="dataset",
        remediation="Use a checked-in local://fixtures dataset or an explicit file:// path.",
    )


def load_local_dataset(dataset_id: str, revision: str, split: str) -> LocalDataset:
    """Load JSONL without network access and require stable unique sample IDs."""

    try:
        text = _dataset_text(dataset_id)
    except (OSError, UnicodeError) as error:
        raise EvaluationError(
            f"Unable to read dataset {dataset_id}: {error}",
            component="dataset",
            remediation="Verify that the local dataset exists and is readable UTF-8 JSONL.",
        ) from error
    samples: list[DatasetSample] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationError(
                f"Invalid JSON in dataset {dataset_id} at line {line_number}",
                component="dataset",
            ) from error
        if (
            not isinstance(value, dict)
            or not isinstance(value.get("id"), str)
            or not value["id"].strip()
        ):
            raise EvaluationError(
                f"Dataset {dataset_id} line {line_number} requires a string id",
                component="dataset",
            )
        sample_id = value["id"]
        if sample_id in seen:
            raise EvaluationError(
                f"Duplicate sample id in dataset {dataset_id}: {sample_id}", component="dataset"
            )
        seen.add(sample_id)
        samples.append(
            DatasetSample(
                sample_id=sample_id,
                values={str(key): item for key, item in value.items()},
                line_number=line_number,
            )
        )
    if not samples:
        raise EvaluationError(f"Dataset is empty: {dataset_id}", component="dataset")
    return LocalDataset(
        dataset_id=dataset_id,
        revision=revision,
        split=split,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        samples=tuple(samples),
    )


def select_dataset_samples(dataset: LocalDataset, sample_ids: tuple[str, ...]) -> LocalDataset:
    """Select configured samples in declared order and reject missing or duplicate IDs."""

    if not sample_ids:
        return dataset
    if len(set(sample_ids)) != len(sample_ids):
        raise EvaluationError(
            "Configured evaluation sample IDs must be unique",
            component="dataset",
            remediation="Remove duplicate IDs from evaluation.suites[].sample_ids.",
        )
    by_id = {sample.sample_id: sample for sample in dataset.samples}
    missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
    if missing:
        raise EvaluationError(
            "Configured evaluation sample IDs are absent from the dataset: " + ", ".join(missing),
            component="dataset",
            remediation="Choose stable sample IDs that exist in the configured dataset revision.",
        )
    return LocalDataset(
        dataset_id=dataset.dataset_id,
        revision=dataset.revision,
        split=dataset.split,
        sha256=dataset.sha256,
        samples=tuple(by_id[sample_id] for sample_id in sample_ids),
    )
