from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inkling_quant_lab.gguf.inkling import (
    INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
    inkling_control_plane_provenance,
    load_inkling_source_adoption_reference,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = PROJECT_ROOT / INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH


def _paid_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("IQL_MODAL_WORKSPACE_BUDGET_CONFIRMED", "800")
    monkeypatch.setenv("IQL_MODAL_BILLING_CYCLE_END_CONFIRMED", "2099-08-01T00:00:00Z")
    return importlib.import_module("scripts.quantize_inkling_modal")


def _artifact(paid: Any, path: Path) -> Any:
    payload = path.read_bytes()
    return paid.SourceAdoptionArtifact(
        path=str(path),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _file_state(root: Path) -> dict[str, tuple[bytes, int, int, int, int, int]]:
    state: dict[str, tuple[bytes, int, int, int, int, int]] = {}
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        metadata = path.lstat()
        state[path.relative_to(root).as_posix()] = (
            path.read_bytes(),
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_mode,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
    return state


def _synthetic_origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Build a tiny but otherwise real direct-materialization receipt tree."""

    paid = _paid_module(monkeypatch)
    config = paid.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    source_mount = tmp_path / "source"
    origin_run_id = paid.inkling_run_id(config, control_plane.tree_sha256)
    origin_root = source_mount / "runs" / origin_run_id
    snapshot = origin_root / "snapshot"
    history_root = origin_root / "control" / "history"
    snapshot.mkdir(parents=True)
    history_root.mkdir(parents=True)

    shard = snapshot / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"synthetic-safetensors-payload")
    source_config = snapshot / "config.json"
    paid._atomic_json(source_config, {"architectures": [config.source.architecture]})
    weight_index = snapshot / "model.safetensors.index.json"
    paid._atomic_json(
        weight_index,
        {
            "metadata": {"total_size": len(shard.read_bytes())},
            "weight_map": {"model.layers.0.weight": shard.name},
        },
    )

    inventory_path = origin_root / "source_inventory.json"
    records = {
        path.name: paid._file_record(path, snapshot)
        for path in (source_config, weight_index, shard)
    }
    paid._atomic_json(
        inventory_path,
        {
            "config_hash": config.config_hash(),
            "model_id": config.source.model_id,
            "revision": config.source.revision,
            "required_file_count": len(records),
            "files": records,
        },
    )
    resolved_config = origin_root / "resolved_config.json"
    origin_control = origin_root / "control_plane.json"
    paid._atomic_json(resolved_config, config.canonical_dict())
    paid._atomic_json(origin_control, control_plane.model_dump(mode="json"))

    history = history_root / "materialize_source.attempt.1.synthetic.json"
    ledger = origin_root / "control" / "materialize_source.attempts.json"
    paid._atomic_json(history, {"status": "started", "sequence": 1})
    paid._atomic_json(ledger, {"attempts": 1, "last_history_sha256": paid._sha256(history)})

    audit = SimpleNamespace(
        weight_index_sha256="c" * 64,
        source_bytes=len(shard.read_bytes()),
        source_tensor_count=1,
        source_shard_count=1,
    )
    receipt_path = origin_root / "source.success.json"
    paid._atomic_json(
        receipt_path,
        {
            "verified": True,
            "config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "model_id": config.source.model_id,
            "revision": config.source.revision,
            "license": config.source.license,
            "source_dir": str(snapshot),
            "weight_index_sha256": audit.weight_index_sha256,
            "source_tensor_bytes": audit.source_bytes,
            "materialized_weight_file_bytes": shard.stat().st_size,
            "source_tensor_count": audit.source_tensor_count,
            "source_shard_count": audit.source_shard_count,
            "file_count": len(records),
            "inventory_sha256": paid._sha256(inventory_path),
            "source_config_sha256": paid._sha256(source_config),
            "warnings": ["synthetic fixture"],
        },
    )

    reference = SimpleNamespace(
        source_run_root=str(origin_root),
        origin_run_id=origin_run_id,
        origin_config_hash=config.config_hash(),
        origin_control_plane_sha256=control_plane.tree_sha256,
        source_success_receipt=_artifact(paid, receipt_path),
        source_inventory=_artifact(paid, inventory_path),
        origin_resolved_config=_artifact(paid, resolved_config),
        origin_control_plane=_artifact(paid, origin_control),
        origin_materialize_attempt_ledger=_artifact(paid, ledger),
        origin_materialize_invocation_history=_artifact(paid, history),
        snapshot_config=_artifact(paid, source_config),
        snapshot_weight_index=_artifact(paid, weight_index),
    )
    monkeypatch.setattr(paid, "SOURCE_MOUNT", source_mount)
    monkeypatch.setattr(paid, "EXPECTED_MODEL_BYTES", shard.stat().st_size)
    monkeypatch.setattr(paid, "audit_inkling_source", lambda *_args, **_kwargs: audit)
    monkeypatch.setattr(
        paid,
        "_load_bound_source_adoption_reference",
        lambda *_args, **_kwargs: reference,
    )
    return SimpleNamespace(
        paid=paid,
        config=config,
        control_plane=control_plane,
        source_mount=source_mount,
        origin_root=origin_root,
        snapshot=snapshot,
        shard=shard,
        receipt_path=receipt_path,
        reference=reference,
    )


def test_real_synthetic_adoption_rehashes_every_file_without_mutating_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _synthetic_origin(tmp_path, monkeypatch)
    before = _file_state(fixture.origin_root)

    reference, receipt = fixture.paid._verify_adopted_source_origin(
        fixture.config,
        "distinct-target-run",
        fixture.control_plane,
    )

    assert reference is fixture.reference
    assert receipt["verified"] is True
    assert receipt["source_dir"] == str(fixture.snapshot)
    assert _file_state(fixture.origin_root) == before


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("missing", r"Inventory tree differs|missing"),
        ("extra", r"Inventory tree differs|unrecorded"),
        ("symlink", "symlink"),
        ("size", "size changed"),
        ("hash", "checksum changed"),
        ("source_directory", "not bound to this run"),
    ),
)
def test_real_synthetic_adoption_rejects_every_inventory_or_source_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    message: str,
) -> None:
    fixture = _synthetic_origin(tmp_path, monkeypatch)
    original = fixture.shard.read_bytes()
    if drift == "missing":
        fixture.shard.unlink()
    elif drift == "extra":
        (fixture.snapshot / "unrecorded.bin").write_bytes(b"extra")
    elif drift == "symlink":
        outside = tmp_path / "outside.safetensors"
        outside.write_bytes(original)
        fixture.shard.unlink()
        fixture.shard.symlink_to(outside)
    elif drift == "size":
        fixture.shard.write_bytes(original + b"!")
    elif drift == "hash":
        fixture.shard.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    elif drift == "source_directory":
        receipt = json.loads(fixture.receipt_path.read_text(encoding="utf-8"))
        receipt["source_dir"] = str(tmp_path / "substituted" / "snapshot")
        fixture.paid._atomic_json(fixture.receipt_path, receipt)
        fixture.reference.source_success_receipt = _artifact(
            fixture.paid,
            fixture.receipt_path,
        )
    else:  # pragma: no cover - the parameter list is closed above
        raise AssertionError(f"Unknown drift case: {drift}")

    with pytest.raises(RuntimeError, match=message):
        fixture.paid._verify_adopted_source_origin(
            fixture.config,
            "distinct-target-run",
            fixture.control_plane,
        )


class _Volume:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def reload(self) -> None:
        self.events.append("reload")

    def commit(self) -> None:
        self.events.append("commit")


def test_materialize_adoption_publishes_success_only_after_toolchain_and_full_rehash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    config = paid.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    run_id = paid.inkling_run_id(config, control_plane.tree_sha256)
    source_mount = tmp_path / "source"
    run_root = source_mount / "runs" / run_id
    success = run_root / "source.success.json"
    reference = load_inkling_source_adoption_reference(REFERENCE_PATH)
    reference_payload = REFERENCE_PATH.read_bytes()
    events: list[str] = []
    volume = _Volume(events)
    acknowledgement = SimpleNamespace(launch_intent_sha256="a" * 64)
    toolchain = {"commit": config.toolchain.commit, "python_inventory_scope": "fixture"}

    monkeypatch.setattr(paid, "SOURCE_MOUNT", source_mount)
    monkeypatch.setattr(paid, "source_volume", volume)
    monkeypatch.setattr(
        paid,
        "_remote_config",
        lambda *_args, **_kwargs: (config, control_plane, acknowledgement),
    )
    monkeypatch.setattr(
        paid,
        "_current_invocation_ids",
        lambda: ("fc-adoption", "in-adoption", "ta-adoption"),
    )

    def verify_toolchain(_config: Any) -> dict[str, str]:
        assert not success.exists()
        events.append("toolchain")
        return toolchain

    monkeypatch.setattr(paid, "_verify_toolchain", verify_toolchain)
    monkeypatch.setattr(
        paid,
        "_checked_source_adoption_payload",
        lambda *_args, **_kwargs: (reference_payload, reference),
    )
    real_immutable_bytes = paid._immutable_bytes

    def immutable_bytes(path: Path, payload: bytes) -> None:
        assert not success.exists()
        events.append("reference")
        real_immutable_bytes(path, payload)

    monkeypatch.setattr(paid, "_immutable_bytes", immutable_bytes)

    def verify_origin(*_args: Any, **_kwargs: Any) -> tuple[Any, dict[str, Any]]:
        assert (run_root / "source.adoption.json").read_bytes() == reference_payload
        assert not success.exists()
        events.append("full_rehash")
        return reference, {"warnings": ["synthetic origin"]}

    monkeypatch.setattr(paid, "_verify_adopted_source_origin", verify_origin)
    real_immutable_json = paid._immutable_json

    def immutable_json(path: Path, value: dict[str, Any]) -> None:
        if path == success:
            assert events.index("toolchain") < events.index("full_rehash")
            events.append("success_marker")
        real_immutable_json(path, value)

    monkeypatch.setattr(paid, "_immutable_json", immutable_json)

    result = paid.materialize_source.get_raw_f()(
        config.canonical_json(),
        run_id,
        "{}",
        control_plane.canonical_json(),
    )

    assert result["status"] == "success"
    assert success.is_file()
    assert events.index("toolchain") < events.index("reference")
    assert events.index("reference") < events.index("full_rehash")
    assert events.index("full_rehash") < events.index("success_marker")


def test_toolchain_failure_hashes_no_origin_snapshot_and_publishes_no_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    config = paid.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    run_id = paid.inkling_run_id(config, control_plane.tree_sha256)
    source_mount = tmp_path / "source"
    origin_snapshot = source_mount / "runs" / "pinned-origin" / "snapshot"
    origin_snapshot.mkdir(parents=True)
    origin_file = origin_snapshot / "model.safetensors"
    origin_file.write_bytes(b"must-not-be-read")
    origin_before = origin_file.read_bytes()
    events: list[str] = []
    volume = _Volume(events)
    acknowledgement = SimpleNamespace(launch_intent_sha256="b" * 64)
    snapshot_hashes: list[Path] = []

    monkeypatch.setattr(paid, "SOURCE_MOUNT", source_mount)
    monkeypatch.setattr(paid, "source_volume", volume)
    monkeypatch.setattr(
        paid,
        "_remote_config",
        lambda *_args, **_kwargs: (config, control_plane, acknowledgement),
    )
    monkeypatch.setattr(
        paid,
        "_current_invocation_ids",
        lambda: ("fc-toolchain-failure", "in-toolchain-failure", "ta-toolchain-failure"),
    )
    monkeypatch.setattr(
        paid,
        "_checked_source_adoption_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reference must not be read before toolchain verification")
        ),
    )
    real_sha256 = paid._sha256

    def observed_sha256(path: Path, **kwargs: Any) -> str:
        if path.resolve(strict=False).is_relative_to(origin_snapshot.resolve()):
            snapshot_hashes.append(path)
        return real_sha256(path, **kwargs)

    monkeypatch.setattr(paid, "_sha256", observed_sha256)

    def fail_toolchain(_config: Any) -> dict[str, str]:
        events.append("toolchain_failure")
        raise RuntimeError("synthetic toolchain drift")

    monkeypatch.setattr(paid, "_verify_toolchain", fail_toolchain)

    with pytest.raises(RuntimeError, match="synthetic toolchain drift"):
        paid.materialize_source.get_raw_f()(
            config.canonical_json(),
            run_id,
            "{}",
            control_plane.canonical_json(),
        )

    run_root = source_mount / "runs" / run_id
    assert snapshot_hashes == []
    assert origin_file.read_bytes() == origin_before
    assert not (run_root / "source.success.json").exists()
    assert not (run_root / "source.adoption.json").exists()
    outcomes = list((run_root / "control" / "outcomes").glob("*.json"))
    assert len(outcomes) == 1
    failure = json.loads(outcomes[0].read_text(encoding="utf-8"))
    assert failure["exception_type"] == "builtins.RuntimeError"


def test_selected_completed_source_records_and_verifies_latest_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    config = paid.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    run_id = paid.inkling_run_id(config, control_plane.tree_sha256)
    source_mount = tmp_path / "source"
    run_root = source_mount / "runs" / run_id
    run_root.mkdir(parents=True)
    success = run_root / "source.success.json"
    volume = _Volume([])
    creation_launch = "c" * 64
    validation_launch = "d" * 64
    toolchain = {"commit": config.toolchain.commit, "python_inventory_scope": "fixture"}

    monkeypatch.setattr(paid, "SOURCE_MOUNT", source_mount)
    monkeypatch.setattr(paid, "source_volume", volume)
    paid._bind_config(run_root, config)
    paid._bind_control_plane(run_root, control_plane)
    monkeypatch.setattr(
        paid,
        "_current_invocation_ids",
        lambda: ("fc-creation", "in-creation", "ta-creation"),
    )
    paid._begin_attempt(
        run_root,
        config,
        "materialize_source",
        control_plane,
        creation_launch,
    )
    creation_binding = paid._latest_attempt_binding(run_root, config, control_plane)
    paid._atomic_json(
        success,
        {
            "run_id": run_id,
            "source_adoption_reference_sha256": "e" * 64,
            **creation_binding,
        },
    )

    monkeypatch.setattr(
        paid,
        "_current_invocation_ids",
        lambda: ("fc-validation", "in-validation", "ta-validation"),
    )
    acknowledgement = SimpleNamespace(launch_intent_sha256=validation_launch)
    monkeypatch.setattr(
        paid,
        "_remote_config",
        lambda *_args, **_kwargs: (config, control_plane, acknowledgement),
    )
    monkeypatch.setattr(paid, "_verify_toolchain", lambda _config: toolchain)
    observed_pending: list[dict[str, Any]] = []

    def verify_materialized_source(
        _config: Any,
        _run_id: str,
        _control_plane: Any,
        *,
        toolchain: dict[str, Any],
        pending_latest_validation: dict[str, Any],
    ) -> dict[str, Any]:
        assert toolchain == {
            "commit": config.toolchain.commit,
            "python_inventory_scope": "fixture",
        }
        assert pending_latest_validation["invocation_sequence"] == 2
        assert pending_latest_validation["call_id"] == "fc-validation"
        observed_pending.append(pending_latest_validation)
        return paid._read_json(success)

    monkeypatch.setattr(paid, "_verify_materialized_source", verify_materialized_source)

    result = paid.materialize_source.get_raw_f()(
        config.canonical_json(),
        run_id,
        "{}",
        control_plane.canonical_json(),
    )

    assert result == {"status": "already_successful", "run_id": run_id}
    assert len(observed_pending) == 1
    validation_paths = list(
        (run_root / "control" / "history").glob("materialize_source.completed_validation.2.*.json")
    )
    assert len(validation_paths) == 1
    validation = paid._read_json(validation_paths[0])
    assert validation["call_id"] == "fc-validation"
    assert validation["input_id"] == "in-validation"
    assert validation["task_id"] == "ta-validation"
    assert validation["launch_intent_sha256"] == validation_launch
    assert validation["source_success_sha256"] == paid._sha256(success)
    assert validation["source_adoption_reference_sha256"] == "e" * 64
    assert validation["toolchain"] == toolchain
