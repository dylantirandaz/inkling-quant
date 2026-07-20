from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.gguf.publication import (
    PublicationReceipt,
    finalize_publication,
    prepare_publication_intent,
    publication_intent_path,
    publication_receipt_path,
)
from inkling_quant_lab.manifests import ArtifactChecksum

CONFIG_HASH = "a" * 64
STAGE = "quantize_text"


def _record(path: Path, root: Path) -> ArtifactChecksum:
    payload = path.read_bytes()
    return ArtifactChecksum(
        path=path.relative_to(root).as_posix(),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _attempt(tmp_path: Path) -> tuple[Path, Path, Path, tuple[ArtifactChecksum, ...]]:
    run_root = tmp_path / "run"
    partial = run_root / ".partial-q3-1"
    canonical = run_root / "q3_k_m"
    nested = partial / "shards"
    nested.mkdir(parents=True)
    first = partial / "inkling-Q3_K_M-00001-of-00002.gguf"
    second = nested / "inkling-Q3_K_M-00002-of-00002.gguf"
    first.write_bytes(b"first shard")
    second.write_bytes(b"second shard")
    return run_root, partial, canonical, (_record(first, partial), _record(second, partial))


def _prepare(
    run_root: Path,
    partial: Path,
    canonical: Path,
    outputs: tuple[ArtifactChecksum, ...],
) -> None:
    prepare_publication_intent(
        run_root,
        config_hash=CONFIG_HASH,
        stage=STAGE,
        partial_directory=partial,
        canonical_directory=canonical,
        outputs=outputs,
    )


def _finalize(run_root: Path, partial: Path, canonical: Path) -> PublicationReceipt:
    return finalize_publication(
        run_root,
        config_hash=CONFIG_HASH,
        stage=STAGE,
        partial_directory=partial,
        canonical_directory=canonical,
    )


def test_finalize_recovers_from_intent_before_rename_and_is_idempotent(tmp_path: Path) -> None:
    run_root, partial, canonical, outputs = _attempt(tmp_path)
    _prepare(run_root, partial, canonical, outputs)

    intent_path = publication_intent_path(run_root, STAGE)
    assert intent_path.is_file()
    assert intent_path.parent == run_root / "control"
    assert partial.is_dir()
    assert not canonical.exists()

    receipt = _finalize(run_root, partial, canonical)

    assert not partial.exists()
    assert canonical.is_dir()
    assert receipt.canonical_directory == "q3_k_m"
    assert tuple(item.path for item in receipt.outputs) == (
        "q3_k_m/inkling-Q3_K_M-00001-of-00002.gguf",
        "q3_k_m/shards/inkling-Q3_K_M-00002-of-00002.gguf",
    )
    receipt_path = publication_receipt_path(run_root, STAGE)
    original_receipt_bytes = receipt_path.read_bytes()

    repeated = _finalize(run_root, partial, canonical)

    assert repeated == receipt
    assert receipt_path.read_bytes() == original_receipt_bytes


def test_publication_uses_only_volume_v1_safe_atomic_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root, partial, canonical, outputs = _attempt(tmp_path)

    def unsupported_hard_link(*args: object, **kwargs: object) -> None:
        raise OSError("hard links unsupported")

    monkeypatch.setattr(os, "link", unsupported_hard_link)
    monkeypatch.setattr(os, "fsync", unsupported_hard_link)
    _prepare(run_root, partial, canonical, outputs)
    receipt = _finalize(run_root, partial, canonical)

    assert receipt.status == "success"
    assert canonical.is_dir()


def test_publication_accepts_a_mount_alias_above_the_run_root(tmp_path: Path) -> None:
    real_mount = tmp_path / "real-mount"
    real_run_root = real_mount / "runs" / "run-id"
    real_partial = real_run_root / ".partial-q3-1"
    real_partial.mkdir(parents=True)
    output = real_partial / "inkling-Q3_K_M.gguf"
    output.write_bytes(b"quantized model")

    logical_mount = tmp_path / "logical-mount"
    logical_mount.symlink_to(real_mount, target_is_directory=True)
    run_root = logical_mount / "runs" / "run-id"
    partial = run_root / ".partial-q3-1"
    canonical = run_root / "q3_k_m"

    _prepare(run_root, partial, canonical, (_record(output, real_partial),))
    receipt = _finalize(run_root, partial, canonical)

    assert receipt.canonical_directory == "q3_k_m"
    assert tuple(item.path for item in receipt.outputs) == ("q3_k_m/inkling-Q3_K_M.gguf",)
    assert publication_intent_path(run_root, STAGE) == run_root / "control" / (
        f"{STAGE}.publication.intent.json"
    )
    assert canonical.joinpath("inkling-Q3_K_M.gguf").read_bytes() == b"quantized model"


def test_finalize_recovers_after_directory_was_already_renamed(tmp_path: Path) -> None:
    run_root, partial, canonical, outputs = _attempt(tmp_path)
    _prepare(run_root, partial, canonical, outputs)
    os.rename(partial, canonical)
    assert not publication_receipt_path(run_root, STAGE).exists()

    receipt = _finalize(run_root, partial, canonical)

    assert receipt.status == "success"
    assert canonical.joinpath(outputs[0].path).read_bytes() == b"first shard"
    assert publication_receipt_path(run_root, STAGE).is_file()


@pytest.mark.parametrize("replacement", [b"other shard", b"longer tampered shard"])
def test_finalize_rejects_output_size_or_hash_drift(tmp_path: Path, replacement: bytes) -> None:
    run_root, partial, canonical, outputs = _attempt(tmp_path)
    _prepare(run_root, partial, canonical, outputs)
    (partial / outputs[0].path).write_bytes(replacement)

    with pytest.raises(ArtifactIntegrityError, match=r"(size|checksum) drifted"):
        _finalize(run_root, partial, canonical)

    assert partial.is_dir()
    assert not canonical.exists()
    assert not publication_receipt_path(run_root, STAGE).exists()


def test_prepare_rejects_unrecorded_file_before_writing_intent(tmp_path: Path) -> None:
    run_root, partial, canonical, outputs = _attempt(tmp_path)
    (partial / "unrecorded.tmp").write_bytes(b"not committed")

    with pytest.raises(ArtifactIntegrityError, match="unrecorded"):
        _prepare(run_root, partial, canonical, outputs)

    assert not (run_root / "control" / f"{STAGE}.publication.intent.json").exists()


def test_finalize_rejects_config_and_path_binding_drift(tmp_path: Path) -> None:
    run_root, partial, canonical, outputs = _attempt(tmp_path)
    _prepare(run_root, partial, canonical, outputs)

    with pytest.raises(ArtifactIntegrityError, match="config hash drifted"):
        finalize_publication(
            run_root,
            config_hash="b" * 64,
            stage=STAGE,
            partial_directory=partial,
            canonical_directory=canonical,
        )
    with pytest.raises(ArtifactIntegrityError, match="canonical path binding drifted"):
        finalize_publication(
            run_root,
            config_hash=CONFIG_HASH,
            stage=STAGE,
            partial_directory=partial,
            canonical_directory=run_root / "different",
        )

    assert partial.is_dir()
    assert not canonical.exists()


def test_finalize_rejects_intent_hash_drift(tmp_path: Path) -> None:
    run_root, partial, canonical, outputs = _attempt(tmp_path)
    _prepare(run_root, partial, canonical, outputs)
    path = publication_intent_path(run_root, STAGE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["partial_directory"] = ".partial-q3-tampered"
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(ArtifactIntegrityError, match="metadata is malformed"):
        _finalize(run_root, partial, canonical)

    assert partial.is_dir()
    assert not canonical.exists()


def test_prepare_rejects_symlink_output_and_path_escape(tmp_path: Path) -> None:
    run_root, partial, canonical, outputs = _attempt(tmp_path)
    outside = tmp_path / "outside.gguf"
    outside.write_bytes(b"outside")
    link = partial / "linked.gguf"
    link.symlink_to(outside)
    linked_record = ArtifactChecksum(
        path="linked.gguf",
        sha256=hashlib.sha256(b"outside").hexdigest(),
        size_bytes=len(b"outside"),
    )

    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        _prepare(run_root, partial, canonical, (*outputs, linked_record))
    with pytest.raises(ArtifactIntegrityError, match="parent traversal"):
        prepare_publication_intent(
            run_root,
            config_hash=CONFIG_HASH,
            stage=STAGE,
            partial_directory=partial,
            canonical_directory=run_root / ".." / "escaped",
            outputs=outputs,
        )

    assert not (run_root / "control").exists()


def test_prepare_rejects_a_symlinked_attempt_directory(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    output = outside / "output.gguf"
    output.write_bytes(b"payload")
    partial = run_root / ".partial"
    partial.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactIntegrityError, match="symlink"):
        prepare_publication_intent(
            run_root,
            config_hash=CONFIG_HASH,
            stage=STAGE,
            partial_directory=partial,
            canonical_directory=run_root / "canonical",
            outputs=(_record(output, outside),),
        )


def test_publication_rejects_a_symlinked_run_root(tmp_path: Path) -> None:
    real_run_root = tmp_path / "real-run"
    real_run_root.mkdir()
    run_root = tmp_path / "run"
    run_root.symlink_to(real_run_root, target_is_directory=True)

    with pytest.raises(ArtifactIntegrityError, match="Run root is not a regular directory"):
        publication_intent_path(run_root, STAGE)


def test_prepare_rejects_an_internal_symlink_path_component(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    actual_parent = run_root / "actual"
    partial = actual_parent / ".partial"
    partial.mkdir(parents=True)
    output = partial / "output.gguf"
    output.write_bytes(b"payload")
    alias = run_root / "alias"
    alias.symlink_to(actual_parent, target_is_directory=True)

    with pytest.raises(ArtifactIntegrityError, match="Publication path contains a symlink"):
        prepare_publication_intent(
            run_root,
            config_hash=CONFIG_HASH,
            stage=STAGE,
            partial_directory=alias / ".partial",
            canonical_directory=alias / "canonical",
            outputs=(_record(output, partial),),
        )


def test_prepare_rejects_an_absolute_path_outside_the_run_root(tmp_path: Path) -> None:
    run_root, partial, _, outputs = _attempt(tmp_path)

    with pytest.raises(ArtifactIntegrityError, match="Publication path escapes run root"):
        prepare_publication_intent(
            run_root,
            config_hash=CONFIG_HASH,
            stage=STAGE,
            partial_directory=partial,
            canonical_directory=tmp_path / "outside",
            outputs=outputs,
        )


def test_finalize_rejects_both_directories_without_deleting_either(tmp_path: Path) -> None:
    run_root, partial, canonical, outputs = _attempt(tmp_path)
    _prepare(run_root, partial, canonical, outputs)
    canonical.mkdir()
    sentinel = canonical / "do-not-delete"
    sentinel.write_bytes(b"canonical evidence")

    with pytest.raises(ArtifactIntegrityError, match="both exist"):
        _finalize(run_root, partial, canonical)

    assert partial.is_dir()
    assert sentinel.read_bytes() == b"canonical evidence"
    assert not publication_receipt_path(run_root, STAGE).exists()


def test_finalize_rejects_neither_directory(tmp_path: Path) -> None:
    run_root, partial, canonical, outputs = _attempt(tmp_path)
    _prepare(run_root, partial, canonical, outputs)
    renamed = run_root / "moved-elsewhere"
    os.rename(partial, renamed)

    with pytest.raises(ArtifactIntegrityError, match="neither exists"):
        _finalize(run_root, partial, canonical)

    assert renamed.is_dir()
    assert not publication_receipt_path(run_root, STAGE).exists()


def test_finalize_never_replaces_a_drifted_success_receipt(tmp_path: Path) -> None:
    run_root, partial, canonical, outputs = _attempt(tmp_path)
    _prepare(run_root, partial, canonical, outputs)
    _finalize(run_root, partial, canonical)
    receipt_path = publication_receipt_path(run_root, STAGE)
    receipt_path.write_bytes(b'{"tampered":true}\n')

    with pytest.raises(ArtifactIntegrityError, match="Refusing to overwrite immutable"):
        _finalize(run_root, partial, canonical)

    assert canonical.is_dir()
    assert receipt_path.read_bytes() == b'{"tampered":true}\n'
