from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inkling_quant_lab.gguf.inkling import (
    INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
    InklingGGUFConfig,
    load_inkling_source_adoption_reference,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE = load_inkling_source_adoption_reference(
    PROJECT_ROOT / INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH
)


def _artifact(path: str, payload: bytes) -> SimpleNamespace:
    return SimpleNamespace(
        path=path,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def _adopted_source_receipt(
    config: InklingGGUFConfig,
    control_plane: Any,
    run_id: str,
) -> dict[str, object]:
    reference_file = next(
        item
        for item in control_plane.files
        if item.path == INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH
    )
    return {
        "schema_version": "inkling-adopted-source-success-v2",
        "status": "success",
        "verified": True,
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "run_id": run_id,
        "materialization_kind": "adopted_verified_source_v1",
        "source_dir": REFERENCE.snapshot_path,
        "source_adoption_reference_sha256": REFERENCE.reference_sha256,
        "source_adoption_file_sha256": reference_file.sha256,
        "origin": {
            "run_id": REFERENCE.origin_run_id,
            "config_hash": REFERENCE.origin_config_hash,
            "control_plane_sha256": REFERENCE.origin_control_plane_sha256,
            "source_success_sha256": REFERENCE.source_success_receipt.sha256,
            "source_inventory_sha256": REFERENCE.source_inventory.sha256,
            "source_config_sha256": REFERENCE.snapshot_config.sha256,
            "weight_index_file_sha256": REFERENCE.snapshot_weight_index.sha256,
            "materialize_function_call_id": REFERENCE.materialize_function_call_id,
            "materialize_input_id": REFERENCE.materialize_input_id,
            "materialize_task_id": REFERENCE.materialize_task_id,
        },
        "model_id": config.source.model_id,
        "revision": config.source.revision,
        "license": config.source.license,
        "weight_index_sha256": REFERENCE.weight_index_canonical_sha256,
        "source_tensor_bytes": REFERENCE.indexed_tensor_bytes,
        "materialized_weight_file_bytes": REFERENCE.materialized_weight_file_bytes,
        "source_tensor_count": REFERENCE.source_tensor_count,
        "source_shard_count": REFERENCE.source_shard_count,
        "file_count": REFERENCE.materialized_file_count,
        "inventory_sha256": REFERENCE.source_inventory.sha256,
        "source_config_sha256": REFERENCE.snapshot_config.sha256,
        "toolchain": {"commit": config.toolchain.commit},
        "invocation_sequence": 1,
        "invocation_history_path": "control/history/materialize_source.attempt.1.event.json",
        "invocation_history_sha256": "e" * 64,
        "call_id": "fc-current-adoption",
        "input_id": "in-current-adoption",
        "task_id": "ta-current-adoption",
        "launch_intent_sha256": "a" * 64,
    }


def test_exact_local_adoption_json_rejects_byte_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    payload = b'{"status":"accepted"}\n'
    artifact = _artifact("artifacts/origin/call.json", payload)
    path = tmp_path / artifact.path
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    monkeypatch.setattr(manager, "PROJECT_ROOT", tmp_path)

    assert manager._read_exact_local_adoption_json(artifact, label="origin call") == {
        "status": "accepted"
    }

    path.write_bytes(payload + b" ")
    with pytest.raises(RuntimeError, match=r"size|SHA-256"):
        manager._read_exact_local_adoption_json(artifact, label="origin call")


def test_exact_remote_adoption_json_uses_reference_path_size_and_hash() -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    payload = b'{"verified":true}\n'
    artifact = _artifact("/source/runs/origin/source.success.json", payload)

    class Volume:
        def read_file(self, path: str) -> Any:
            assert path == "runs/origin/source.success.json"
            yield payload

    assert manager._read_exact_volume_adoption_json(
        Volume(),
        artifact,
        mount_path="/source",
        label="origin source receipt",
    ) == {"verified": True}

    changed = _artifact(artifact.path, payload)
    changed.sha256 = "0" * 64
    with pytest.raises(RuntimeError, match="SHA-256"):
        manager._read_exact_volume_adoption_json(
            Volume(),
            changed,
            mount_path="/source",
            label="origin source receipt",
        )


def test_completed_adoption_validations_bind_exact_marker_bytes_and_record_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    config, control_plane, run_id = manager._load_checked_context()
    receipt = _adopted_source_receipt(config, control_plane, run_id)
    marker_payload = json.dumps(receipt, sort_keys=True).encode("utf-8") + b"\n"
    record = {
        "schema_version": "inkling-adopted-source-completed-validation-v1",
        "status": "success",
        "verified": True,
        "invocation_sequence": 2,
    }
    record_payload = json.dumps(record, sort_keys=True).encode("utf-8") + b"\n"
    record_id = hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    marker_path = f"runs/{run_id}/source.success.json"
    validation_path = (
        f"runs/{run_id}/control/history/materialize_source.completed_validation.2.{record_id}.json"
    )

    class Volume:
        def __init__(self) -> None:
            self.files = {marker_path: marker_payload, validation_path: record_payload}
            self.paths = [validation_path]

        def hydrate(self) -> None:
            return None

        def listdir(self, _path: str, *, recursive: bool) -> list[SimpleNamespace]:
            assert recursive is False
            return [SimpleNamespace(path=path) for path in self.paths]

        def read_file(self, path: str) -> Any:
            if path not in self.files:
                raise FileNotFoundError(path)
            yield self.files[path]

    volume = Volume()
    monkeypatch.setattr(manager.modal.Volume, "from_name", lambda *_args, **_kwargs: volume)

    validations, marker_sha256 = manager._completed_adoption_validations_from_volume(
        config,
        run_id=run_id,
        expected_source_receipt=receipt,
    )

    assert validations == {2: record}
    assert marker_sha256 == hashlib.sha256(marker_payload).hexdigest()

    malformed_path = validation_path.replace(record_id, "0" * 64)
    volume.paths = [malformed_path]
    volume.files[malformed_path] = record_payload
    with pytest.raises(RuntimeError, match="completed-validation history drifted"):
        manager._completed_adoption_validations_from_volume(
            config,
            run_id=run_id,
            expected_source_receipt=receipt,
        )


def test_origin_call_graph_requires_one_successful_childless_exact_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    root = SimpleNamespace(
        function_call_id=REFERENCE.materialize_function_call_id,
        # Modal's graph normalizes away the immutable history suffix and omits
        # task identity. The exact history/ledger records bind both values.
        input_id=REFERENCE.materialize_input_id.split(":", maxsplit=1)[0],
        task_id="",
        function_name="materialize_source",
        status=manager.InputStatus.SUCCESS,
        children=[],
    )

    class Call:
        def get_call_graph(self) -> list[SimpleNamespace]:
            return [root]

    monkeypatch.setattr(manager.modal.FunctionCall, "from_id", lambda _call_id: Call())
    manager._require_origin_call_graph_success(REFERENCE)

    root.children = [SimpleNamespace()]
    with pytest.raises(RuntimeError, match="zero children"):
        manager._require_origin_call_graph_success(REFERENCE)


def test_origin_app_must_be_exactly_stopped_with_zero_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    row = {
        "app_id": REFERENCE.origin_app_id,
        "description": REFERENCE.origin_app_name,
        "state": "stopped",
        "tasks": "0",
    }
    monkeypatch.setattr(manager, "_modal_app_list", lambda _config: [row])
    manager._require_origin_app_stopped(InklingGGUFConfig(), REFERENCE)

    row["tasks"] = "1"
    with pytest.raises(RuntimeError, match="stopped with zero active tasks"):
        manager._require_origin_app_stopped(InklingGGUFConfig(), REFERENCE)


def test_checked_context_loads_and_validates_exact_adoption_reference() -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    config, control_plane, run_id = manager._load_checked_context()

    reference = manager._load_checked_source_adoption(config, control_plane, run_id)

    assert reference.reference_sha256 == REFERENCE.reference_sha256
    assert reference.origin_run_id == REFERENCE.origin_run_id
    assert reference.origin_app_id == REFERENCE.origin_app_id
    assert run_id != reference.origin_run_id


def test_real_current_source_marker_requires_exact_adoption_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    config, control_plane, run_id = manager._load_checked_context()
    adopted = _adopted_source_receipt(config, control_plane, run_id)
    monkeypatch.setattr(
        manager,
        "_validate_adopted_source_invocation_binding",
        lambda *_args, **_kwargs: None,
    )

    manager._validate_volume_receipt(
        adopted,
        config=config,
        control_plane=control_plane,
        run_id=run_id,
        stage="materialize_source",
    )

    direct = {
        "verified": True,
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "run_id": run_id,
        "model_id": config.source.model_id,
        "revision": config.source.revision,
        "source_dir": f"/source/runs/{run_id}/snapshot",
    }
    with pytest.raises(RuntimeError, match="adopted source binding"):
        manager._validate_volume_receipt(
            direct,
            config=config,
            control_plane=control_plane,
            run_id=run_id,
            stage="materialize_source",
        )

    # Helper-level fixtures can retain a compact direct receipt, but every public
    # manager entry derives the production run ID and therefore takes the strict path.
    manager._validate_volume_receipt(
        direct,
        config=config,
        control_plane=control_plane,
        run_id="synthetic-unit-run",
        stage="materialize_source",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", "inkling-adopted-source-success-v1"),
        ("materialization_kind", "direct_huggingface_snapshot_v1"),
        ("source_adoption_reference_sha256", "0" * 64),
        ("origin", {"run_id": "substituted"}),
    ),
)
def test_real_current_source_marker_rejects_adoption_identity_drift(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    config, control_plane, run_id = manager._load_checked_context()
    receipt = _adopted_source_receipt(config, control_plane, run_id)
    monkeypatch.setattr(
        manager,
        "_validate_adopted_source_invocation_binding",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="exact adopted source binding"):
        manager._validate_volume_receipt(
            {**receipt, field: value},
            config=config,
            control_plane=control_plane,
            run_id=run_id,
            stage="materialize_source",
        )


def test_real_adopted_source_marker_binds_creation_and_later_validation_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    config, control_plane, run_id = manager._load_checked_context()
    receipt = _adopted_source_receipt(config, control_plane, run_id)
    history_path = f"runs/{run_id}/control/history/materialize_source.attempt.1.event.json"
    history_sha256 = "e" * 64
    event = {
        "sequence": 1,
        "call_id": receipt["call_id"],
        "input_id": receipt["input_id"],
        "task_id": receipt["task_id"],
        "launch_intent_sha256": receipt["launch_intent_sha256"],
    }
    history = [(history_path, event, history_sha256)]
    monkeypatch.setattr(
        manager,
        "_authoritative_invocation_history",
        lambda *_args, **_kwargs: history,
    )
    ledger = {
        "attempts": 1,
        "limit": next(
            item.max_attempts for item in config.modal.stages if item.name == "materialize_source"
        ),
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "last_history_path": receipt["invocation_history_path"],
        "last_history_sha256": history_sha256,
        "last_call_id": receipt["call_id"],
        "last_input_id": receipt["input_id"],
        "last_task_id": receipt["task_id"],
        "launch_intent_sha256": receipt["launch_intent_sha256"],
    }
    monkeypatch.setattr(
        manager,
        "_read_volume_json_file",
        lambda *_args, **_kwargs: ledger,
    )
    validations: dict[int, dict[str, object]] = {}
    source_success_sha256 = "f" * 64
    monkeypatch.setattr(
        manager,
        "_completed_adoption_validations_from_volume",
        lambda *_args, **_kwargs: (validations, source_success_sha256),
    )

    manager._validate_adopted_source_invocation_binding(
        receipt,
        config=config,
        control_plane=control_plane,
        run_id=run_id,
    )

    with pytest.raises(RuntimeError, match="exact creation invocation"):
        manager._validate_adopted_source_invocation_binding(
            {**receipt, "task_id": "ta-substituted"},
            config=config,
            control_plane=control_plane,
            run_id=run_id,
        )

    second_path = f"runs/{run_id}/control/history/materialize_source.attempt.2.event.json"
    second_sha256 = "d" * 64
    second_event = {
        "sequence": 2,
        "call_id": "fc-later-validation",
        "input_id": "in-later-validation",
        "task_id": "ta-later-validation",
        "launch_intent_sha256": "b" * 64,
    }
    history.append((second_path, second_event, second_sha256))
    ledger.update(
        {
            "attempts": 2,
            "last_history_path": second_path.removeprefix(f"runs/{run_id}/"),
            "last_history_sha256": second_sha256,
            "last_call_id": second_event["call_id"],
            "last_input_id": second_event["input_id"],
            "last_task_id": second_event["task_id"],
            "launch_intent_sha256": second_event["launch_intent_sha256"],
        }
    )
    with pytest.raises(RuntimeError, match="completed-validation chain"):
        manager._validate_adopted_source_invocation_binding(
            receipt,
            config=config,
            control_plane=control_plane,
            run_id=run_id,
        )

    validations[2] = {
        "schema_version": "inkling-adopted-source-completed-validation-v1",
        "status": "success",
        "verified": True,
        "run_id": run_id,
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "source_success_sha256": source_success_sha256,
        "source_adoption_reference_sha256": receipt["source_adoption_reference_sha256"],
        "toolchain": receipt["toolchain"],
        "invocation_sequence": 2,
        "invocation_history_path": second_path.removeprefix(f"runs/{run_id}/"),
        "invocation_history_sha256": second_sha256,
        "call_id": second_event["call_id"],
        "input_id": second_event["input_id"],
        "task_id": second_event["task_id"],
        "launch_intent_sha256": second_event["launch_intent_sha256"],
    }
    manager._validate_adopted_source_invocation_binding(
        receipt,
        config=config,
        control_plane=control_plane,
        run_id=run_id,
    )

    validations[2] = {**validations[2], "task_id": "ta-substituted"}
    with pytest.raises(RuntimeError, match="completed-validation receipt drifted"):
        manager._validate_adopted_source_invocation_binding(
            receipt,
            config=config,
            control_plane=control_plane,
            run_id=run_id,
        )


@pytest.mark.parametrize(
    "stage",
    (
        "convert_text_bf16",
        "convert_multimodal_projector",
        "quantize_text",
        "verify_export",
    ),
)
def test_every_real_downstream_launch_revalidates_adopted_source_marker(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    config, control_plane, run_id = manager._load_checked_context()
    adopted = _adopted_source_receipt(config, control_plane, run_id)
    monkeypatch.setattr(
        manager,
        "_validate_adopted_source_invocation_binding",
        lambda *_args, **_kwargs: None,
    )
    stage_index = manager.STAGE_ORDER.index(stage)
    predecessor = manager.STAGE_ORDER[stage_index - 1]
    predecessor_receipt = (
        adopted
        if predecessor == "materialize_source"
        else {
            "status": "success",
            "config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "call_id": "fc-complete-predecessor",
        }
    )
    reads: list[str] = []

    def read_receipt(
        _config: object,
        *,
        run_id: str,
        stage: str,
    ) -> dict[str, object] | None:
        del run_id
        reads.append(stage)
        if stage == "materialize_source":
            return adopted
        if stage == predecessor:
            return predecessor_receipt
        return None

    monkeypatch.setattr(manager, "_read_volume_receipt", read_receipt)
    monkeypatch.setattr(manager, "_require_stage_invocation_budget", lambda *_args: 1)

    class CompleteCall:
        def get_call_graph(self) -> list[object]:
            return []

    monkeypatch.setattr(manager.modal.FunctionCall, "from_id", lambda _call_id: CompleteCall())

    assert (
        manager._require_launchable_stage(
            config,
            control_plane,
            run_id,
            stage,
        )
        == 1
    )
    assert "materialize_source" in reads


def test_real_downstream_launch_rejects_direct_current_source_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    config, control_plane, run_id = manager._load_checked_context()
    direct = {
        "verified": True,
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "run_id": run_id,
    }

    def read_receipt(
        _config: object,
        *,
        run_id: str,
        stage: str,
    ) -> dict[str, object] | None:
        del run_id
        return direct if stage == "materialize_source" else None

    monkeypatch.setattr(manager, "_read_volume_receipt", read_receipt)
    monkeypatch.setattr(manager, "_require_stage_invocation_budget", lambda *_args: 1)

    with pytest.raises(RuntimeError, match="adopted source binding"):
        manager._require_launchable_stage(
            config,
            control_plane,
            run_id,
            "quantize_text",
        )


def test_readiness_validates_all_local_and_remote_records_before_live_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    events: list[str] = []
    monkeypatch.setattr(
        manager,
        "_validate_source_adoption_local_artifacts",
        lambda _reference: events.append("local"),
    )
    monkeypatch.setattr(
        manager,
        "_validate_source_adoption_volume_artifacts",
        lambda _config, _reference: events.append("volume"),
    )
    monkeypatch.setattr(
        manager,
        "_require_origin_call_graph_success",
        lambda _reference: events.append("call_graph"),
    )
    monkeypatch.setattr(
        manager,
        "_require_origin_app_stopped",
        lambda _config, _reference: events.append("app"),
    )

    evidence = manager._require_source_adoption_origin_ready(InklingGGUFConfig(), REFERENCE)

    assert events == ["local", "volume", "call_graph", "app"]
    assert evidence["source_adoption_reference_sha256"] == REFERENCE.reference_sha256
    assert evidence["source_adoption_origin_run_id"] == REFERENCE.origin_run_id
    assert evidence["source_adoption_origin_app_id"] == REFERENCE.origin_app_id


def test_materialize_launch_rechecks_origin_immediately_before_intent_and_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    config = InklingGGUFConfig()
    control_plane = SimpleNamespace(tree_sha256="d" * 64, canonical_json=lambda: "{}")
    run_id = manager.inkling_run_id(config, control_plane.tree_sha256)
    events: list[str] = []
    written: list[dict[str, object]] = []
    deployment = {
        "app_name": "inkling-target-app",
        "version": 1,
        "tag": "iql-target",
        "billing_cycle_end_utc": "2099-08-01T00:00:00Z",
        "initial_billing_window_policy": "full_workflow_plus_storage_lag_v1",
        "billing_cycle_end_source": "dashboard_exact_utc",
        "stage_function_bindings": {
            "materialize_source": {
                "function_id": "fu-materialize",
                "function_name": "materialize_source",
            }
        },
    }

    class Call:
        object_id = "fc-target"

        def cancel(self, *, terminate_containers: bool) -> None:
            raise AssertionError("no cancellation expected")

    class Function:
        def hydrate(self) -> None:
            events.append("hydrate")

        def spawn(self, *_args: object) -> Call:
            events.append("spawn")
            return Call()

    monkeypatch.setattr(manager, "_run_root", lambda _run_id: tmp_path)
    monkeypatch.setattr(
        manager, "_require_paid_gate", lambda *_args: deployment["billing_cycle_end_utc"]
    )
    monkeypatch.setattr(manager, "_require_no_unresolved_launch_intent", lambda *_args: None)
    monkeypatch.setattr(manager, "_require_no_active_stage_call", lambda *_args: None)
    monkeypatch.setattr(manager, "_require_launchable_stage", lambda *_args: 1)
    monkeypatch.setattr(manager, "require_stage_billing_window", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(manager, "_audit_plan", lambda *_args: {})
    monkeypatch.setattr(manager, "_validated_deployment", lambda *_args: deployment)
    monkeypatch.setattr(manager, "_revalidate_deployment_history", lambda *_args: None)
    monkeypatch.setattr(
        manager,
        "_function_binding",
        lambda *_args, **_kwargs: deployment["stage_function_bindings"]["materialize_source"],
    )
    monkeypatch.setattr(manager.modal.Function, "from_name", lambda *_args, **_kwargs: Function())
    monkeypatch.setattr(manager, "_load_checked_source_adoption", lambda *_args: REFERENCE)
    monkeypatch.setattr(manager, "_load_source_adoption_for_target", lambda *_args: REFERENCE)

    readiness_count = 0

    def readiness(_config: object, _reference: object) -> dict[str, str]:
        nonlocal readiness_count
        readiness_count += 1
        events.append(f"ready-{readiness_count}")
        return manager._source_adoption_binding(REFERENCE)

    monkeypatch.setattr(manager, "_require_source_adoption_origin_ready", readiness)

    def create(*_args: object, **_kwargs: object) -> tuple[str, str]:
        events.append("intent")
        return "launch-intents/test.json", "a" * 64

    monkeypatch.setattr(manager, "_create_launch_intent", create)
    monkeypatch.setattr(
        manager,
        "_write_immutable_json",
        lambda _path, value: written.append(dict(value)),
    )

    manager._launch_locked(
        config,
        control_plane,
        run_id,
        "materialize_source",
        config.budget.workspace_hard_budget_usd,
    )

    assert readiness_count == 2
    assert events.index("ready-1") < events.index("hydrate")
    assert events[-3:] == ["ready-2", "intent", "spawn"]
    assert written[0]["source_adoption_reference_sha256"] == REFERENCE.reference_sha256
    assert written[0]["source_adoption_origin_run_id"] == REFERENCE.origin_run_id


def test_deploy_rechecks_origin_immediately_before_modal_deploy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    config = InklingGGUFConfig()
    control_plane = SimpleNamespace(tree_sha256="d" * 64, file_count=1)
    run_id = manager.inkling_run_id(config, control_plane.tree_sha256)
    events: list[str] = []
    history = [{"tag": "iql-target", "version": "v1", "time_deployed": "now"}]
    bindings = {
        stage: {"function_id": f"fu-{stage}", "function_name": stage}
        for stage in manager.STAGE_ORDER
    }
    written: list[dict[str, object]] = []

    monkeypatch.setattr(manager, "_require_paid_gate", lambda *_args: "2099-08-01T00:00:00Z")
    monkeypatch.setattr(
        manager,
        "_initial_billing_window_evidence",
        lambda *_args, **_kwargs: (
            "full_workflow_plus_storage_lag_v1",
            "dashboard_exact_utc",
        ),
    )
    monkeypatch.setattr(manager, "_load_checked_source_adoption", lambda *_args: REFERENCE)

    readiness_count = 0

    def readiness(_config: object, _reference: object) -> dict[str, str]:
        nonlocal readiness_count
        readiness_count += 1
        events.append(f"ready-{readiness_count}")
        return manager._source_adoption_binding(REFERENCE)

    monkeypatch.setattr(manager, "_require_source_adoption_origin_ready", readiness)
    monkeypatch.setattr(
        manager,
        "_audit_plan",
        lambda *_args: events.append("audit") or {"status": "passed"},
    )
    monkeypatch.setattr(manager, "_deployment_path", lambda _run_id: tmp_path / "deployment.json")
    monkeypatch.setattr(manager, "_deployment_tag", lambda _control: "iql-target")
    monkeypatch.setattr(manager, "_deployment_name", lambda *_args: "inkling-target-app")
    monkeypatch.setattr(manager, "_modal_history_or_empty", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(manager, "_modal_history", lambda *_args, **_kwargs: history)
    monkeypatch.setattr(manager, "_deployed_function_bindings", lambda *_args, **_kwargs: bindings)
    monkeypatch.setattr(
        manager.subprocess,
        "run",
        lambda *_args, **_kwargs: events.append("deploy") or SimpleNamespace(),
    )
    monkeypatch.setattr(
        manager,
        "_write_immutable_json",
        lambda _path, value: written.append(dict(value)),
    )

    manager._deploy_locked(
        config,
        control_plane,
        run_id,
        config.budget.workspace_hard_budget_usd,
    )

    assert events == ["ready-1", "audit", "ready-2", "deploy"]
    assert written[0]["source_adoption_reference_sha256"] == REFERENCE.reference_sha256
    assert written[0]["source_adoption_origin_app_id"] == REFERENCE.origin_app_id


def test_audit_and_new_receipts_bind_adoption_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = pytest.importorskip("scripts.manage_inkling_modal")
    config = InklingGGUFConfig()
    control_plane = SimpleNamespace(tree_sha256="d" * 64, file_count=1)
    audit = SimpleNamespace(
        model_id=config.source.model_id,
        revision=config.source.revision,
        source_bytes=REFERENCE.indexed_tensor_bytes,
        source_tensor_count=REFERENCE.source_tensor_count,
        source_shard_count=REFERENCE.source_shard_count,
        warnings=(),
    )
    monkeypatch.setattr(manager, "audit_pinned_inkling_online", lambda *_args, **_kwargs: audit)
    monkeypatch.setattr(manager, "_load_checked_source_adoption", lambda *_args: REFERENCE)
    monkeypatch.setattr(manager, "_load_source_adoption_for_target", lambda *_args: REFERENCE)
    monkeypatch.setattr(manager, "_run_root", lambda _run_id: tmp_path)

    plan = manager._audit_plan(config, control_plane, "inkling-target-run")
    deployment = {
        "app_name": "inkling-target-app",
        "version": 1,
        "tag": "iql-target",
        "billing_cycle_end_utc": "2099-08-01T00:00:00Z",
        "initial_billing_window_policy": "full_workflow_plus_storage_lag_v1",
        "billing_cycle_end_source": "dashboard_exact_utc",
    }
    binding = {"function_id": "fu-materialize", "function_name": "materialize_source"}
    relative, _sha256 = manager._create_launch_intent(
        config,
        control_plane,
        "inkling-target-run",
        "materialize_source",
        deployment,
        binding,
    )
    intent = json.loads((tmp_path / relative).read_text(encoding="utf-8"))

    for value in (plan, intent):
        assert value["source_adoption_reference_sha256"] == REFERENCE.reference_sha256
        assert value["source_adoption_origin_run_id"] == REFERENCE.origin_run_id
        assert value["source_adoption_origin_app_name"] == REFERENCE.origin_app_name
        assert value["source_adoption_origin_app_id"] == REFERENCE.origin_app_id

    assert manager.DEPLOYMENT_SCHEMA == "inkling-modal-deployment-v7"
    assert manager.LAUNCH_INTENT_SCHEMA == "inkling-modal-launch-intent-v5"
    assert manager.CALL_SCHEMA == "inkling-modal-call-v6"
