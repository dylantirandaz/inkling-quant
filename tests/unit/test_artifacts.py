from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from inkling_quant_lab.artifacts import LocalArtifactStore, sha256_bytes, sha256_file
from inkling_quant_lab.exceptions import ArtifactIntegrityError

pytestmark = pytest.mark.unit


def test_sha256_helpers_produce_the_expected_digest(tmp_path):
    payload = b"abcdef"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert sha256_bytes(payload) == expected
    assert sha256_file(path, chunk_size=2) == expected


def test_store_writes_reads_and_replaces_bytes(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")

    first = store.write_bytes("metrics/value.bin", b"first")
    assert store.root == (tmp_path / "artifacts").resolve()
    assert store.path("metrics/value.bin") == first
    assert store.read_bytes("metrics/value.bin") == b"first"

    with pytest.raises(ArtifactIntegrityError, match="Refusing to overwrite"):
        store.write_bytes("metrics/value.bin", b"second")

    store.write_bytes("metrics/value.bin", b"second", replace=True)
    assert store.read_bytes("metrics/value.bin") == b"second"


def test_store_appends_text_and_round_trips_json(tmp_path):
    store = LocalArtifactStore(tmp_path)

    store.write_text("events.jsonl", "first\n", append=True)
    store.write_text("events.jsonl", "second\n", append=True)
    store.write_json("metrics/value.json", {"z": 1, "a": [2, 3]})

    assert store.read_bytes("events.jsonl") == b"first\nsecond\n"
    assert store.read_json("metrics/value.json") == {"a": [2, 3], "z": 1}
    assert store.checksum("events.jsonl") == sha256_bytes(b"first\nsecond\n")


def test_ensure_directory_creates_nested_path(tmp_path):
    store = LocalArtifactStore(tmp_path)

    directory = store.ensure_directory("reports/nested")

    assert directory.is_dir()
    assert directory.is_relative_to(store.root)


def test_checksum_rejects_missing_artifact(tmp_path):
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ArtifactIntegrityError, match="missing or not a file"):
        store.checksum("missing.bin")


def test_staged_directory_is_published_atomically(tmp_path):
    store = LocalArtifactStore(tmp_path)

    with store.staged_directory("stages/evaluate") as temporary:
        (temporary / "nested").mkdir()
        (temporary / "nested" / "result.json").write_text("result", encoding="utf-8")

    destination = store.path("stages/evaluate")
    assert (destination / "nested" / "result.json").read_text(encoding="utf-8") == "result"

    with (
        pytest.raises(ArtifactIntegrityError, match="already exists"),
        store.staged_directory("stages/evaluate"),
    ):
        pass


def test_failed_staged_directory_removes_files_directories_and_symlinks(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    with (
        pytest.raises(RuntimeError, match="producer failed"),
        store.staged_directory("stages/failure") as temporary,
    ):
        nested = temporary / "nested"
        nested.mkdir()
        (nested / "partial.json").write_text("partial", encoding="utf-8")
        (temporary / "link").symlink_to(outside)
        raise RuntimeError("producer failed")

    assert not store.path("stages/failure").exists()
    assert not list(store.path("stages").glob(".failure.*.tmp"))
    assert outside.read_text(encoding="utf-8") == "outside"


def test_failed_staged_directory_restores_replaced_empty_placeholder(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    placeholder = store.ensure_directory("reports")

    with (
        pytest.raises(RuntimeError, match="producer failed"),
        store.staged_directory("reports", replace_empty_placeholder=True) as temporary,
    ):
        (temporary / "partial.md").write_text("partial", encoding="utf-8")
        raise RuntimeError("producer failed")

    assert placeholder.is_dir()
    assert not any(placeholder.iterdir())
    assert not list(store.root.glob(".reports.*.tmp"))


def test_staged_directory_detects_concurrent_destination(tmp_path):
    store = LocalArtifactStore(tmp_path)

    with (
        pytest.raises(ArtifactIntegrityError, match="appeared concurrently"),
        store.staged_directory("stages/race") as temporary,
    ):
        (temporary / "partial.json").write_text("partial", encoding="utf-8")
        store.ensure_directory("stages/race")

    assert store.path("stages/race").is_dir()
    assert not list(store.path("stages").glob(".race.*.tmp"))


def test_write_detects_destination_created_after_temporary_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    store = LocalArtifactStore(tmp_path)
    destination = store.path("race.bin")
    real_exists = Path.exists
    destination_checks = 0

    def exists_with_race(path: Path) -> bool:
        nonlocal destination_checks
        if path == destination:
            destination_checks += 1
            return destination_checks > 1
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", exists_with_race)

    with pytest.raises(ArtifactIntegrityError, match="Refusing to overwrite"):
        store.write_bytes("race.bin", b"payload")

    assert destination_checks == 2
