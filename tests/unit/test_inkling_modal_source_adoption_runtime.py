from __future__ import annotations

import ast
import hashlib
import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inkling_quant_lab.gguf.inkling import (
    INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
    ORIGIN_SOURCE_RUN_ID,
    inkling_control_plane_provenance,
    load_inkling_source_adoption_reference,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAID_SCRIPT = PROJECT_ROOT / "scripts/quantize_inkling_modal.py"
REFERENCE_PATH = PROJECT_ROOT / INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH


def _paid_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("IQL_MODAL_WORKSPACE_BUDGET_CONFIRMED", "800")
    monkeypatch.setenv("IQL_MODAL_BILLING_CYCLE_END_CONFIRMED", "2099-08-01T00:00:00Z")
    return importlib.import_module("scripts.quantize_inkling_modal")


def _functions() -> dict[str, ast.FunctionDef]:
    tree = ast.parse(PAID_SCRIPT.read_text(encoding="utf-8"), filename=str(PAID_SCRIPT))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _modal_function_decorator(function: ast.FunctionDef) -> ast.Call:
    matches = [
        decorator
        for decorator in function.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "function"
    ]
    assert len(matches) == 1
    return matches[0]


def test_selected_materialize_function_is_offline_secret_free_adoption() -> None:
    functions = _functions()
    selected = functions["materialize_source"]
    downloader = functions["download_materialize_source"]
    selected_keywords = {
        item.arg: item.value for item in _modal_function_decorator(selected).keywords
    }
    assert ast.literal_eval(selected_keywords["block_network"]) is True
    assert isinstance(selected_keywords["image"], ast.Name)
    assert selected_keywords["image"].id == "toolchain_image"
    assert "secrets" not in selected_keywords
    assert not any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "function"
        for decorator in downloader.decorator_list
    )
    selected_names = {node.id for node in ast.walk(selected) if isinstance(node, ast.Name)}
    assert "audit_pinned_inkling_online" not in selected_names
    assert "hf_secret" not in selected_names


def test_selected_adoption_attempt_precedes_rehash_and_success_is_marker_last() -> None:
    selected = _functions()["materialize_source"]
    calls = [
        node
        for node in ast.walk(selected)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    begin = sorted(node.lineno for node in calls if node.func.id == "_begin_attempt")
    toolchain = sorted(node.lineno for node in calls if node.func.id == "_verify_toolchain")
    verify = sorted(
        node.lineno for node in calls if node.func.id == "_verify_adopted_source_origin"
    )
    publish = sorted(node.lineno for node in calls if node.func.id == "_immutable_json")

    assert len(begin) == 2  # completed revalidation and first execution are both ledgered
    assert len(toolchain) == 2
    assert len(verify) == 1
    assert len(publish) >= 1
    assert begin[-1] < toolchain[-1] < verify[-1] < publish[-1]
    assert any(node.func.id == "_verify_materialized_source" for node in calls)


def test_bound_reference_copy_matches_the_current_control_plane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    source_mount = tmp_path / "source"
    run_id = paid.inkling_run_id(
        paid.InklingGGUFConfig(),
        inkling_control_plane_provenance(PROJECT_ROOT).tree_sha256,
    )
    run_root = source_mount / "runs" / run_id
    run_root.mkdir(parents=True)
    shutil.copyfile(REFERENCE_PATH, run_root / "source.adoption.json")
    monkeypatch.setattr(paid, "SOURCE_MOUNT", source_mount)

    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    reference = paid._load_bound_source_adoption_reference(
        paid.InklingGGUFConfig(),
        run_id,
        control_plane,
    )
    assert reference == load_inkling_source_adoption_reference(REFERENCE_PATH)

    (run_root / "source.adoption.json").write_bytes(REFERENCE_PATH.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match=r"(control-plane|canonical|reference)"):
        paid._load_bound_source_adoption_reference(
            paid.InklingGGUFConfig(),
            run_id,
            control_plane,
        )


def test_execution_paths_resolve_the_adopted_snapshot_without_a_link_or_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    source_mount = tmp_path / "source"
    run_id = "target-run"
    run_root = source_mount / "runs" / run_id
    origin_snapshot = source_mount / "runs" / ORIGIN_SOURCE_RUN_ID / "snapshot"
    run_root.mkdir(parents=True)
    origin_snapshot.mkdir(parents=True)
    (run_root / "source.success.json").write_text(
        json.dumps({"source_dir": str(origin_snapshot)}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(paid, "SOURCE_MOUNT", source_mount)

    paths = paid._execution_paths(
        run_id,
        partial_work=tmp_path / "work",
        partial_final=tmp_path / "final",
    )
    assert paths.source_dir == origin_snapshot
    assert not (run_root / "snapshot").exists()

    (run_root / "source.success.json").write_text(
        json.dumps({"source_dir": str(tmp_path / "outside" / "snapshot")}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="source_dir"):
        paid._execution_paths(
            run_id,
            partial_work=tmp_path / "work",
            partial_final=tmp_path / "final",
        )


def test_final_verifier_uses_the_receipt_resolved_source_directory() -> None:
    verifier = _functions()["_verify_final_dependency_chain"]
    called = {
        node.func.id
        for node in ast.walk(verifier)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_source_dir_for_run" in called
    assert "_verify_materialized_source" in called


def test_origin_verification_rechecks_evidence_around_the_full_snapshot_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    source_mount = tmp_path / "source"
    origin_control = inkling_control_plane_provenance(PROJECT_ROOT)
    config = paid.InklingGGUFConfig()
    legacy_config = config.canonical_dict()
    for stage in legacy_config["modal"]["stages"]:
        stage.pop("ephemeral_disk_mib")
    legacy_config_hash = hashlib.sha256(
        json.dumps(legacy_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    origin_run_id = paid.inkling_run_id_for_config_hash(
        config,
        legacy_config_hash,
        origin_control.tree_sha256,
    )
    origin_root = source_mount / "runs" / origin_run_id
    origin_root.mkdir(parents=True)
    resolved_config = origin_root / "resolved_config.json"
    origin_control_path = origin_root / "control_plane.json"
    resolved_config.write_text(
        json.dumps(legacy_config, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    origin_control_path.write_text(
        json.dumps(origin_control.model_dump(mode="json"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reference = SimpleNamespace(
        source_run_root=str(origin_root),
        origin_run_id=origin_run_id,
        origin_config_hash=legacy_config_hash,
        origin_control_plane_sha256=origin_control.tree_sha256,
        origin_resolved_config=paid.SourceAdoptionArtifact(
            path=str(resolved_config),
            sha256="a" * 64,
            size_bytes=1,
        ),
        origin_control_plane=paid.SourceAdoptionArtifact(
            path=str(origin_control_path),
            sha256="b" * 64,
            size_bytes=1,
        ),
    )
    events: list[str] = []
    monkeypatch.setattr(paid, "SOURCE_MOUNT", source_mount)
    monkeypatch.setattr(
        paid,
        "_load_bound_source_adoption_reference",
        lambda *_: reference,
    )
    monkeypatch.setattr(
        paid,
        "_verify_reference_remote_artifacts",
        lambda _: events.append("artifacts"),
    )
    monkeypatch.setattr(
        paid,
        "_verify_direct_materialized_source",
        lambda *_, **__: events.append("snapshot") or {"verified": True},
    )

    observed_reference, receipt = paid._verify_adopted_source_origin(
        config,
        "target-run",
        origin_control,
    )
    assert observed_reference is reference
    assert receipt == {"verified": True}
    assert events == ["artifacts", "snapshot", "artifacts"]


def test_adopted_success_receipt_is_bound_to_current_control_and_origin(
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
    monkeypatch.setattr(paid, "SOURCE_MOUNT", source_mount)
    paid._bind_control_plane(run_root, control_plane)
    shutil.copyfile(REFERENCE_PATH, run_root / "source.adoption.json")
    reference = load_inkling_source_adoption_reference(REFERENCE_PATH)
    warnings = ["exact origin warning"]
    monkeypatch.setattr(
        paid,
        "_verify_adopted_source_origin",
        lambda *_: (reference, {"warnings": warnings}),
    )
    monkeypatch.setattr(paid, "_verify_success_invocation_binding", lambda *_, **__: None)
    receipt = {
        "schema_version": "inkling-adopted-source-success-v2",
        "status": "success",
        "verified": True,
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "run_id": run_id,
        "materialization_kind": "adopted_verified_source_v1",
        "source_dir": reference.snapshot_path,
        "source_adoption_reference_sha256": reference.reference_sha256,
        "source_adoption_file_sha256": paid._sha256(run_root / "source.adoption.json"),
        "origin": paid._adopted_source_origin_record(reference),
        "model_id": config.source.model_id,
        "revision": config.source.revision,
        "license": config.source.license,
        "weight_index_sha256": reference.weight_index_canonical_sha256,
        "source_tensor_bytes": reference.indexed_tensor_bytes,
        "materialized_weight_file_bytes": reference.materialized_weight_file_bytes,
        "source_tensor_count": reference.source_tensor_count,
        "source_shard_count": reference.source_shard_count,
        "file_count": reference.materialized_file_count,
        "inventory_sha256": reference.source_inventory.sha256,
        "source_config_sha256": reference.snapshot_config.sha256,
        "warnings": warnings,
        "toolchain": {"commit": config.toolchain.commit},
        "invocation_sequence": 1,
        "invocation_history_path": (
            "control/history/materialize_source.attempt.1." + "d" * 64 + ".json"
        ),
        "invocation_history_sha256": "e" * 64,
        "call_id": "fc-adoption",
        "input_id": "in-adoption",
        "task_id": "ta-adoption",
        "launch_intent_sha256": "f" * 64,
    }
    paid._atomic_json(run_root / "source.success.json", receipt)

    assert (
        paid._verify_materialized_source(
            config,
            run_id,
            control_plane,
            toolchain={"commit": config.toolchain.commit},
        )
        == receipt
    )
    (run_root / "snapshot").mkdir()
    with pytest.raises(RuntimeError, match="must not contain a target-run snapshot"):
        paid._verify_materialized_source(
            config,
            run_id,
            control_plane,
            toolchain={"commit": config.toolchain.commit},
        )
    (run_root / "snapshot").rmdir()
    paid._atomic_json(
        run_root / "source.success.json",
        {**receipt, "source_dir": "/source/runs/substituted/snapshot"},
    )
    with pytest.raises(RuntimeError, match="source_dir"):
        paid._verify_materialized_source(
            config,
            run_id,
            control_plane,
            toolchain={"commit": config.toolchain.commit},
        )


def test_adoption_bound_control_plane_rejects_a_direct_current_run_receipt(
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
    monkeypatch.setattr(paid, "SOURCE_MOUNT", source_mount)
    paid._bind_control_plane(run_root, control_plane)
    paid._atomic_json(run_root / "source.success.json", {"verified": True})
    called = False

    def direct(*_: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {"verified": True}

    monkeypatch.setattr(paid, "_verify_direct_materialized_source", direct)
    with pytest.raises(RuntimeError, match="rejects direct current-run source receipts"):
        paid._verify_materialized_source(config, run_id, control_plane)
    assert called is False


def test_adopted_receipt_call_is_present_in_the_immutable_attempt_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    config = paid.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    run_root = tmp_path / "run"
    history_root = run_root / "control" / "history"
    history_root.mkdir(parents=True)
    event = {
        "schema_version": "inkling-modal-stage-invocation-v2",
        "status": "started",
        "kind": "attempt",
        "sequence": 1,
        "limit": 12,
        "stage": "materialize_source",
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "launch_intent_sha256": "a" * 64,
        "call_id": "fc-adoption",
        "input_id": "in-adoption",
        "task_id": "ta-adoption",
        "previous_history_sha256": None,
    }
    event_id = hashlib.sha256(
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    history_path = history_root / f"materialize_source.attempt.1.{event_id}.json"
    paid._atomic_json(history_path, event)
    history_sha256 = paid._sha256(history_path)
    paid._atomic_json(
        run_root / "control" / "materialize_source.attempts.json",
        {
            "attempts": 1,
            "limit": 12,
            "config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "launch_intent_sha256": "a" * 64,
            "last_call_id": "fc-adoption",
            "last_input_id": "in-adoption",
            "last_task_id": "ta-adoption",
            "last_history_path": history_path.relative_to(run_root).as_posix(),
            "last_history_sha256": history_sha256,
        },
    )

    receipt = {
        "run_id": "target-run",
        "source_adoption_reference_sha256": "f" * 64,
        "invocation_sequence": 1,
        "invocation_history_path": history_path.relative_to(run_root).as_posix(),
        "invocation_history_sha256": history_sha256,
        "call_id": "fc-adoption",
        "input_id": "in-adoption",
        "task_id": "ta-adoption",
        "launch_intent_sha256": "a" * 64,
    }
    paid._verify_success_invocation_binding(
        run_root,
        config,
        control_plane,
        receipt,
        toolchain={"commit": config.toolchain.commit},
    )
    with pytest.raises(RuntimeError, match="exact immutable creation invocation"):
        paid._verify_success_invocation_binding(
            run_root,
            config,
            control_plane,
            {**receipt, "input_id": "in-substituted"},
            toolchain={"commit": config.toolchain.commit},
        )

    second_event = {
        **event,
        "sequence": 2,
        "launch_intent_sha256": "b" * 64,
        "call_id": "fc-later",
        "input_id": "in-later",
        "task_id": "ta-later",
        "previous_history_sha256": history_sha256,
    }
    second_event_id = hashlib.sha256(
        json.dumps(second_event, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    second_path = history_root / f"materialize_source.attempt.2.{second_event_id}.json"
    paid._atomic_json(second_path, second_event)
    second_sha256 = paid._sha256(second_path)
    paid._atomic_json(
        run_root / "control" / "materialize_source.attempts.json",
        {
            "attempts": 2,
            "limit": 12,
            "config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "launch_intent_sha256": "b" * 64,
            "last_call_id": "fc-later",
            "last_input_id": "in-later",
            "last_task_id": "ta-later",
            "last_history_path": second_path.relative_to(run_root).as_posix(),
            "last_history_sha256": second_sha256,
        },
    )
    with pytest.raises(RuntimeError, match="completed-validation chain"):
        paid._verify_success_invocation_binding(
            run_root,
            config,
            control_plane,
            receipt,
            toolchain={"commit": config.toolchain.commit},
        )

    paid._atomic_json(run_root / "source.success.json", receipt)
    latest_binding = paid._latest_attempt_binding(run_root, config, control_plane)
    toolchain = {"commit": config.toolchain.commit}
    validation = paid._adoption_validation_record(
        run_id="target-run",
        config=config,
        control_plane=control_plane,
        source_success_sha256=paid._sha256(run_root / "source.success.json"),
        source_adoption_reference_sha256="f" * 64,
        toolchain=toolchain,
        invocation_binding=latest_binding,
    )
    paid._immutable_json(paid._adoption_validation_path(run_root, validation), validation)
    paid._verify_success_invocation_binding(
        run_root,
        config,
        control_plane,
        receipt,
        toolchain=toolchain,
    )
