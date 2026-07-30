"""Host-only checks for the checked Inkling measurement inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.gguf.inkling_measurement import (
    CORPUS_MATERIALIZER_RELATIVE_PATH,
    build_diagnostic_fixture_bytes,
    load_diagnostic_items,
    load_measurement_bundle,
    measurement_protocol_sha256,
    measurement_workload_sha256,
)
from scripts import materialize_inkling_measurement_corpus as corpus_materializer

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_PATH = PROJECT_ROOT / "configs/experiments/inkling_quality_diagnostic_v1.jsonl"
CORPUS_REFERENCE_PATH = (
    PROJECT_ROOT / "configs/experiments/inkling_wikitext2_raw_test_reference.json"
)


def _copy_measurement_inputs(destination: Path) -> None:
    shutil.copytree(PROJECT_ROOT / "configs", destination / "configs")
    shutil.copytree(PROJECT_ROOT / "patches", destination / "patches")
    materializer = destination / CORPUS_MATERIALIZER_RELATIVE_PATH
    materializer.parent.mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / CORPUS_MATERIALIZER_RELATIVE_PATH, materializer)


def test_checked_measurement_bundle_loads_and_binds_all_inputs() -> None:
    bundle = load_measurement_bundle(PROJECT_ROOT)

    assert len(bundle.diagnostic_items) == 64
    assert bundle.corpus.materializer_path == CORPUS_MATERIALIZER_RELATIVE_PATH
    assert bundle.config.model_id == bundle.matched.config.model_id
    assert bundle.config.revision == bundle.matched.config.revision
    assert len(measurement_protocol_sha256(bundle.config)) == 64
    assert len(measurement_workload_sha256(bundle.config)) == 64


def test_checked_multimodal_fixtures_materialize_deterministically() -> None:
    items = load_diagnostic_items(DIAGNOSTIC_PATH)
    fixtures = tuple(item.fixture for item in items if item.fixture is not None)

    assert {fixture.algorithm for fixture in fixtures if fixture.kind == "image"} == {
        "solid",
        "checkerboard_4px",
        "vertical_split",
        "horizontal_split",
        "quadrants",
        "vertical_bands_4",
        "horizontal_bands_4",
        "center_square_16px",
    }
    assert {fixture.algorithm for fixture in fixtures if fixture.kind == "audio"} == {
        "silence",
        "square_tone",
        "pulse_train",
    }
    for fixture in fixtures:
        first = build_diagnostic_fixture_bytes(fixture)
        second = build_diagnostic_fixture_bytes(fixture)
        assert first == second
        assert first is not None
        if fixture.kind == "image":
            assert first.startswith(b"\x89PNG\r\n\x1a\n")
        else:
            assert first.startswith(b"RIFF")
            assert first[8:12] == b"WAVE"

    assert build_diagnostic_fixture_bytes(None) is None


def test_measurement_bundle_rejects_materializer_byte_tamper(tmp_path: Path) -> None:
    _copy_measurement_inputs(tmp_path)
    materializer = tmp_path / CORPUS_MATERIALIZER_RELATIVE_PATH
    materializer.write_bytes(materializer.read_bytes() + b"\n")

    with pytest.raises(
        ConfigurationError,
        match="Corpus materializer byte identity differs",
    ):
        load_measurement_bundle(tmp_path)


def test_diagnostic_loader_accepts_only_the_checked_field_order_jsonl(
    tmp_path: Path,
) -> None:
    items = load_diagnostic_items(DIAGNOSTIC_PATH)
    assert len(items) == 64

    lines = DIAGNOSTIC_PATH.read_text(encoding="utf-8").splitlines(keepends=True)
    first_item = json.loads(lines[0])
    alternate = (
        json.dumps(
            first_item,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    assert alternate != lines[0]
    lines[0] = alternate
    alternate_path = tmp_path / "alternate-diagnostics.jsonl"
    alternate_path.write_text("".join(lines), encoding="utf-8", newline="")

    with pytest.raises(ConfigurationError, match="not canonical diagnostic JSONL"):
        load_diagnostic_items(alternate_path)


def test_materializer_rejects_a_suffix_matching_repository_path(
    tmp_path: Path,
) -> None:
    reference = json.loads(CORPUS_REFERENCE_PATH.read_text(encoding="utf-8"))
    reference["materializer_path"] = "other/scripts/materialize_inkling_measurement_corpus.py"
    payload = dict(reference)
    del payload["reference_sha256"]
    payload_bytes = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    reference["reference_sha256"] = hashlib.sha256(
        corpus_materializer.REFERENCE_HASH_DOMAIN + payload_bytes
    ).hexdigest()
    reference_path = tmp_path / "corpus-reference.json"
    reference_path.write_text(
        json.dumps(
            reference,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
        newline="",
    )

    with pytest.raises(ValueError, match="names a different materializer"):
        corpus_materializer.materialize(
            reference_path,
            Path(reference["materialized_path"]),
        )
