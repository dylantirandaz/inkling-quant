from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from modal.exception import FunctionTimeoutError

from inkling_quant_lab.gguf.inkling import inkling_control_plane_provenance

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAID_SCRIPT = PROJECT_ROOT / "scripts/quantize_inkling_modal.py"
MANAGER_SCRIPT = PROJECT_ROOT / "scripts/manage_inkling_modal.py"
PAID_FUNCTIONS = {
    "materialize_source",
    "convert_text_bf16",
    "convert_multimodal_projector",
    "quantize_text",
    "verify_export",
}
PAID_RESOURCE_NAMES = {
    "materialize_source": "MATERIALIZE_RESOURCES",
    "convert_text_bf16": "TEXT_CONVERT_RESOURCES",
    "convert_multimodal_projector": "MMPROJ_RESOURCES",
    "quantize_text": "QUANTIZE_RESOURCES",
    "verify_export": "VERIFY_RESOURCES",
}


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def _function_decorator(node: ast.FunctionDef) -> ast.Call:
    matches = [
        decorator
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "function"
    ]
    assert len(matches) == 1
    return matches[0]


def _keyword_literals(call: ast.Call) -> dict[str, object]:
    return {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in call.keywords
        if keyword.arg is not None and isinstance(keyword.value, ast.Constant)
    }


def test_paid_functions_share_the_four_part_remote_gate_and_stable_concurrency() -> None:
    functions = _functions(PAID_SCRIPT)
    assert set(functions) >= PAID_FUNCTIONS

    for name in PAID_FUNCTIONS:
        function = functions[name]
        assert [argument.arg for argument in function.args.args] == [
            "config_json",
            "run_id",
            "budget_acknowledgement_json",
            "control_plane_json",
        ]
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_remote_config"
        ]
        assert len(calls) == 1
        assert len(calls[0].args) == 5
        assert ast.literal_eval(calls[0].args[4]) == name

        decorator = _function_decorator(function)
        literals = _keyword_literals(decorator)
        assert literals["retries"] == 0
        assert literals["max_containers"] == 1
        assert literals["single_use_containers"] is True
        startup_keywords = [
            keyword for keyword in decorator.keywords if keyword.arg == "startup_timeout"
        ]
        assert len(startup_keywords) == 1
        startup_value = startup_keywords[0].value
        assert isinstance(startup_value, ast.Attribute)
        assert startup_value.attr == "startup_timeout_seconds"
        assert isinstance(startup_value.value, ast.Name)
        assert startup_value.value.id == PAID_RESOURCE_NAMES[name]
        disk_keywords = [
            keyword for keyword in decorator.keywords if keyword.arg == "ephemeral_disk"
        ]
        assert len(disk_keywords) == 1
        disk_value = disk_keywords[0].value
        assert isinstance(disk_value, ast.Attribute)
        assert disk_value.attr == "ephemeral_disk_mib"
        assert isinstance(disk_value.value, ast.Name)
        assert disk_value.value.id == PAID_RESOURCE_NAMES[name]
        assert literals["block_network"] is True

    remote_gate = functions["_remote_config"]
    cycle_checks = [
        node
        for node in ast.walk(remote_gate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "require_stage_billing_window"
    ]
    assert len(cycle_checks) == 1

    materialize = functions["download_materialize_source"]
    continuation_lines = sorted(
        node.lineno
        for node in ast.walk(materialize)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "spawn"
    )
    continuation_cycle_lines = sorted(
        node.lineno
        for node in ast.walk(materialize)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "require_stage_billing_window"
    )
    assert len(continuation_lines) == len(continuation_cycle_lines) == 2
    assert all(
        check < spawn
        for check, spawn in zip(continuation_cycle_lines, continuation_lines, strict=True)
    )


def test_every_paid_stage_arms_immutable_failure_recording() -> None:
    functions = _functions(PAID_SCRIPT)
    for name in PAID_FUNCTIONS:
        capture_decorators = [
            decorator
            for decorator in functions[name].decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Name)
            and decorator.func.id == "_capture_stage_failures"
        ]
        assert len(capture_decorators) == 1
        assert [ast.literal_eval(argument) for argument in capture_decorators[0].args] == [name]
        arm_calls = [
            node
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_arm_failure_recording"
        ]
        assert len(arm_calls) == 1


def test_expensive_completed_stage_revalidation_spends_a_reserved_ledger_slot() -> None:
    functions = _functions(PAID_SCRIPT)
    for name in PAID_FUNCTIONS - {"materialize_source"}:
        recovery_calls = [
            node
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_begin_recovery"
        ]
        # One path adopts an interrupted publication and the other revalidates
        # an already committed success. The configured reserve permits either,
        # but never an unledgered multi-TB validation.
        assert len(recovery_calls) == 2

    materialize_attempt_calls = [
        node
        for node in ast.walk(functions["materialize_source"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_begin_attempt"
    ]
    assert len(materialize_attempt_calls) == 2


def test_paid_script_has_no_ephemeral_modal_run_entrypoint() -> None:
    tree = ast.parse(PAID_SCRIPT.read_text(encoding="utf-8"), filename=str(PAID_SCRIPT))
    local_entrypoints = [
        decorator
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "local_entrypoint"
    ]
    assert local_entrypoints == []


def test_recovery_app_does_not_export_a_networked_materializer_image() -> None:
    tree = ast.parse(PAID_SCRIPT.read_text(encoding="utf-8"), filename=str(PAID_SCRIPT))
    assert not any(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "python_image" for target in node.targets
        )
    )
    downloader = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "download_materialize_source"
    )
    assert not any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "function"
        for decorator in downloader.decorator_list
    )


def test_manager_uses_sealed_app_name_and_passes_all_remote_bindings() -> None:
    functions = _functions(MANAGER_SCRIPT)
    launch = functions["_launch_locked"]
    from_name_calls = [
        node
        for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_name"
    ]
    assert len(from_name_calls) == 1
    keywords = {keyword.arg for keyword in from_name_calls[0].keywords}
    assert "environment_name" in keywords
    assert "version" not in keywords
    first_argument = from_name_calls[0].args[0]
    assert isinstance(first_argument, ast.Call)
    assert isinstance(first_argument.func, ast.Name)
    assert first_argument.func.id == "str"

    spawn_calls = [
        node
        for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "spawn"
    ]
    assert len(spawn_calls) == 1
    assert len(spawn_calls[0].args) == 4
    hydrate_calls = [
        node
        for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "hydrate"
    ]
    assert len(hydrate_calls) == 1
    binding_checks = [
        node
        for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_function_binding"
    ]
    assert len(binding_checks) == 1
    history_checks = [
        node
        for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_revalidate_deployment_history"
    ]
    assert len(history_checks) == 2
    cancel_calls = [
        node
        for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "cancel"
    ]
    assert len(cancel_calls) == 1

    active_checks = [
        node.lineno
        for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_require_no_active_stage_call"
    ]
    launchable_checks = [
        node.lineno
        for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_require_launchable_stage"
    ]
    intent_line = next(
        node.lineno
        for node in ast.walk(launch)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_create_launch_intent"
    )
    spawn_line = spawn_calls[0].lineno
    assert len(active_checks) == 2
    assert len(launchable_checks) == 2
    assert max(active_checks) < max(launchable_checks) < intent_line < spawn_line


def test_manager_rejects_a_tagged_deployment_that_is_not_newest() -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    history = [
        {"version": "v3", "tag": "foreign"},
        {"version": "v2", "tag": "expected"},
    ]
    version, _ = manager._deployment_row(history, tag="expected")

    with pytest.raises(RuntimeError, match="newer deployment"):
        manager._require_newest_deployment(history, version=version)


def test_short_initial_window_requires_flag_exact_cycle_token_and_materialize_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    config = manager.InklingGGUFConfig()
    cycle_end = "2026-08-01T00:00:00Z"
    checked: list[str] = []

    def reject_full_window(config: Any, observed_cycle_end: str) -> None:
        assert observed_cycle_end == cycle_end
        raise manager.ConfigurationError("short window", component="inkling_paid_gate")

    def require_materialize(config: Any, observed_cycle_end: str) -> None:
        assert observed_cycle_end == cycle_end
        checked.append(observed_cycle_end)

    monkeypatch.setattr(manager, "require_initial_billing_window", reject_full_window)
    monkeypatch.setattr(
        manager,
        "require_materialize_initial_billing_window",
        require_materialize,
    )

    with pytest.raises(manager.ConfigurationError, match="short window"):
        manager._initial_billing_window_evidence(
            config,
            cycle_end,
            accept_short_initial_window_risk=False,
        )
    monkeypatch.setenv("IQL_MODAL_SHORT_CYCLE_CONFIRMED", "2026-09-01T00:00:00Z")
    with pytest.raises(RuntimeError, match="exactly equal"):
        manager._initial_billing_window_evidence(
            config,
            cycle_end,
            accept_short_initial_window_risk=True,
        )

    monkeypatch.setenv("IQL_MODAL_SHORT_CYCLE_CONFIRMED", cycle_end)
    evidence = manager._initial_billing_window_evidence(
        config,
        cycle_end,
        accept_short_initial_window_risk=True,
    )
    assert evidence == (
        "operator_accepted_short_initial_window_v1",
        "user_confirmed_date_assumed_utc_midnight",
    )
    assert checked == [cycle_end]


def test_manager_hydrates_and_seals_unique_function_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    hydrated: list[str] = []

    class FakeFunction:
        def __init__(self, stage: str) -> None:
            self.stage = stage
            self.object_id = f"fu-{stage}"

        def hydrate(self) -> FakeFunction:
            hydrated.append(self.stage)
            return self

        def _get_metadata(self) -> SimpleNamespace:
            return SimpleNamespace(function_name=self.stage, definition_id="")

    def fake_from_name(
        app_name: str,
        stage: str,
        *,
        environment_name: str,
    ) -> FakeFunction:
        assert app_name == "implementation-addressed"
        assert environment_name == "inkling-quant"
        return FakeFunction(stage)

    monkeypatch.setattr(manager.modal.Function, "from_name", fake_from_name)
    observed = manager._deployed_function_bindings(
        manager.InklingGGUFConfig(),
        app_name="implementation-addressed",
    )

    assert list(observed) == list(manager.STAGE_ORDER)
    assert hydrated == list(manager.STAGE_ORDER)
    assert len({binding["function_id"] for binding in observed.values()}) == len(
        manager.STAGE_ORDER
    )
    assert {binding["function_name"] for binding in observed.values()} == set(manager.STAGE_ORDER)


def test_manager_rejects_a_function_name_that_does_not_match_its_stage() -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    function = SimpleNamespace(
        object_id="fu-materialize-source",
        _get_metadata=lambda: SimpleNamespace(function_name="quantize_text", definition_id=""),
    )

    with pytest.raises(RuntimeError, match="wrong Function name"):
        manager._function_binding(function, stage="materialize_source")


def _patch_manager_launch_context(
    manager: Any,
    monkeypatch: pytest.MonkeyPatch,
    function: Any,
) -> None:
    config = manager.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    run_id = "inkling-launch-test"
    deployment = {
        "app_name": "implementation-addressed",
        "environment_name": config.modal.environment_name,
        "version": 1,
        "tag": "sealed-tag",
        "billing_cycle_end_utc": "2099-08-01T00:00:00Z",
        "initial_billing_window_policy": "full_workflow_plus_storage_lag_v1",
        "billing_cycle_end_source": "dashboard_exact_utc",
        "stage_function_bindings": {
            stage: {
                "function_id": f"fu-{stage}",
                "function_name": stage,
            }
            for stage in manager.STAGE_ORDER
        },
    }
    monkeypatch.setattr(
        manager,
        "_load_checked_context",
        lambda: (config, control_plane, run_id),
    )
    monkeypatch.setattr(
        manager,
        "_require_paid_gate",
        lambda config, budget: "2099-08-01T00:00:00Z",
    )
    monkeypatch.setattr(
        manager,
        "_require_no_unresolved_launch_intent",
        lambda config, control, observed_run_id: None,
    )
    monkeypatch.setattr(
        manager,
        "_require_launchable_stage",
        lambda config, control, observed_run_id, stage: 12,
    )
    monkeypatch.setattr(
        manager,
        "_require_no_active_stage_call",
        lambda config, control, observed_run_id, stage: None,
    )
    monkeypatch.setattr(
        manager,
        "_create_launch_intent",
        lambda config, control, observed_run_id, stage, deployment, binding: (
            "launch-intents/unit-test.json",
            "a" * 64,
        ),
    )
    monkeypatch.setattr(manager, "_audit_plan", lambda config, control, observed_run_id: {})
    monkeypatch.setattr(
        manager,
        "_validated_deployment",
        lambda config, control, observed_run_id: deployment,
    )
    monkeypatch.setattr(manager.modal.Function, "from_name", lambda *args, **kwargs: function)


def test_manager_refuses_a_changed_function_binding_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")

    class ChangedFunction:
        object_id = "fu-redeployed-materialize-source"
        spawned = False

        def hydrate(self) -> ChangedFunction:
            return self

        def _get_metadata(self) -> SimpleNamespace:
            return SimpleNamespace(function_name="materialize_source")

        def spawn(self, *args: object) -> None:
            self.spawned = True

    function = ChangedFunction()
    _patch_manager_launch_context(manager, monkeypatch, function)

    with pytest.raises(RuntimeError, match="binding changed after sealing"):
        manager.launch("materialize_source", manager.Decimal("800"))
    assert function.spawned is False


def test_manager_rejects_a_pending_predecessor_even_after_its_marker_is_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")

    class NeverSpawnedFunction:
        object_id = "fu-quantize_text"
        spawned = False

        def hydrate(self) -> NeverSpawnedFunction:
            return self

        def _get_metadata(self) -> SimpleNamespace:
            return SimpleNamespace(function_name="quantize_text")

        def spawn(self, *args: object) -> None:
            self.spawned = True

    function = NeverSpawnedFunction()
    _patch_manager_launch_context(manager, monkeypatch, function)
    checked: list[str] = []

    def reject_pending_predecessor(
        config: Any,
        control_plane: Any,
        run_id: str,
        checked_stage: str,
    ) -> None:
        checked.append(checked_stage)
        if checked_stage == "convert_multimodal_projector":
            raise RuntimeError("predecessor still has a pending call")

    monkeypatch.setattr(manager, "_require_no_active_stage_call", reject_pending_predecessor)

    with pytest.raises(RuntimeError, match="predecessor still has a pending call"):
        manager.launch("quantize_text", manager.Decimal("800"))

    assert checked == ["convert_multimodal_projector"]
    assert function.spawned is False


def test_manager_cancels_and_records_a_call_if_history_changes_after_spawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    recorded: list[dict[str, Any]] = []

    class Call:
        object_id = "fc-accepted"
        cancelled = False

        def cancel(self, *, terminate_containers: bool) -> None:
            assert terminate_containers is True
            self.cancelled = True

    call = Call()

    class StableFunction:
        object_id = "fu-materialize_source"

        def hydrate(self) -> StableFunction:
            return self

        def _get_metadata(self) -> SimpleNamespace:
            return SimpleNamespace(function_name="materialize_source")

        def spawn(self, *args: object) -> Call:
            assert len(args) == 4
            return call

    _patch_manager_launch_context(manager, monkeypatch, StableFunction())
    checks = 0

    def revalidate(config: Any, deployment: Any) -> None:
        nonlocal checks
        checks += 1
        if checks == 2:
            raise RuntimeError("new deployment observed")

    monkeypatch.setattr(manager, "_revalidate_deployment_history", revalidate)
    monkeypatch.setattr(
        manager,
        "_write_immutable_json",
        lambda path, receipt: recorded.append(dict(receipt)),
    )

    with pytest.raises(RuntimeError, match="cancellation was requested"):
        manager.launch("materialize_source", manager.Decimal("800"))

    assert call.cancelled is True
    assert checks == 2
    assert recorded[-1]["call_id"] == "fc-accepted"
    assert recorded[-1]["status"] == "cancellation_requested_after_deployment_change"
    assert recorded[-1]["launch_intent_sha256"]


def test_manager_rejects_missing_predecessor_without_spawning_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    config = manager.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)

    class EmptyVolume:
        def hydrate(self) -> EmptyVolume:
            return self

        def listdir(self, path: str, *, recursive: bool = False) -> list[Any]:
            return []

        def read_file(self, path: str) -> Any:
            raise FileNotFoundError(path)

    monkeypatch.setattr(manager.modal.Volume, "from_name", lambda *args, **kwargs: EmptyVolume())

    with pytest.raises(RuntimeError, match="requires successful predecessor"):
        manager._require_launchable_stage(
            config,
            control_plane,
            "inkling-order-test",
            "quantize_text",
        )


def test_manager_accepts_only_a_matching_committed_predecessor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    config = manager.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    predecessor = json.dumps(
        {
            "status": "success",
            "config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "call_id": "fc-predecessor-complete",
        }
    ).encode("utf-8")

    class ReceiptVolume:
        def hydrate(self) -> ReceiptVolume:
            return self

        def listdir(self, path: str, *, recursive: bool = False) -> list[Any]:
            return []

        def read_file(self, path: str) -> Any:
            if path.endswith("quantize_text.success.json") or "/control/" in path:
                raise FileNotFoundError(path)
            assert path.endswith("convert_multimodal_projector.success.json")
            yield predecessor

    monkeypatch.setattr(
        manager.modal.Volume,
        "from_name",
        lambda *args, **kwargs: ReceiptVolume(),
    )

    class CompletedPredecessorCall:
        def get_call_graph(self) -> list[SimpleNamespace]:
            return []

    monkeypatch.setattr(
        manager.modal.FunctionCall,
        "from_id",
        lambda call_id: CompletedPredecessorCall(),
    )

    manager._require_launchable_stage(
        config,
        control_plane,
        "inkling-order-test",
        "quantize_text",
    )

    class PendingPredecessorCall:
        def get_call_graph(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(status=manager.InputStatus.PENDING, children=[])]

    monkeypatch.setattr(
        manager.modal.FunctionCall,
        "from_id",
        lambda call_id: PendingPredecessorCall(),
    )
    with pytest.raises(RuntimeError, match="still has pending call"):
        manager._require_launchable_stage(
            config,
            control_plane,
            "inkling-order-test",
            "quantize_text",
        )


def test_manager_does_not_treat_a_missing_named_volume_as_a_missing_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")

    class MissingNamedVolume:
        def read_file(self, path: str) -> Any:
            raise manager.modal.exception.NotFoundError(path)

    monkeypatch.setattr(
        manager.modal.Volume,
        "from_name",
        lambda *args, **kwargs: MissingNamedVolume(),
    )

    with pytest.raises(manager.modal.exception.NotFoundError):
        manager._read_volume_receipt(
            manager.InklingGGUFConfig(),
            run_id="inkling-missing-volume-test",
            stage="materialize_source",
        )


def test_manager_treats_a_missing_history_path_as_empty_only_after_hydration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    config = manager.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    hydrated = False

    class MissingHistoryVolume:
        def hydrate(self) -> MissingHistoryVolume:
            nonlocal hydrated
            hydrated = True
            return self

        def listdir(self, path: str, *, recursive: bool = False) -> list[Any]:
            assert hydrated is True
            assert path.endswith("/control/history")
            assert recursive is False
            raise manager.modal.exception.NotFoundError("No such file")

    monkeypatch.setattr(
        manager.modal.Volume,
        "from_name",
        lambda *args, **kwargs: MissingHistoryVolume(),
    )

    assert (
        manager._authoritative_invocation_history(
            config,
            control_plane,
            volume_attribute="source_volume",
            run_id="inkling-missing-history-test",
            stage="materialize_source",
            kind="attempt",
            limit=12,
        )
        == []
    )


def test_manager_does_not_treat_a_missing_named_volume_as_empty_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    config = manager.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)

    class MissingNamedVolume:
        def hydrate(self) -> MissingNamedVolume:
            raise manager.modal.exception.NotFoundError("Volume not found")

        def listdir(self, path: str, *, recursive: bool = False) -> list[Any]:
            raise AssertionError("listdir must not run when Volume hydration fails")

    monkeypatch.setattr(
        manager.modal.Volume,
        "from_name",
        lambda *args, **kwargs: MissingNamedVolume(),
    )

    with pytest.raises(manager.modal.exception.NotFoundError, match="Volume not found"):
        manager._authoritative_invocation_history(
            config,
            control_plane,
            volume_attribute="source_volume",
            run_id="inkling-missing-volume-history-test",
            stage="materialize_source",
            kind="attempt",
            limit=12,
        )


def test_manager_enforces_immutable_history_when_latest_ledger_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    config = manager.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    run_id = "inkling-history-ahead-test"
    history_root = f"runs/{run_id}/control/history"
    previous_sha256: str | None = None
    payloads: dict[str, bytes] = {}
    for sequence in (1, 2):
        event = {
            "schema_version": "inkling-modal-stage-invocation-v2",
            "status": "started",
            "kind": "attempt",
            "sequence": sequence,
            "limit": 2,
            "stage": "quantize_text",
            "config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "launch_intent_sha256": f"{sequence}" * 64,
            "call_id": f"fc-{sequence}",
            "input_id": f"in-{sequence}",
            "task_id": f"ta-{sequence}",
            "previous_history_sha256": previous_sha256,
        }
        event_id = hashlib.sha256(
            json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        path = f"{history_root}/quantize_text.attempt.{sequence}.{event_id}.json"
        payload = (json.dumps(event, sort_keys=True, indent=2) + "\n").encode("utf-8")
        payloads[path] = payload
        previous_sha256 = hashlib.sha256(payload).hexdigest()

    class HistoryOnlyVolume:
        def hydrate(self) -> HistoryOnlyVolume:
            return self

        def listdir(self, path: str, *, recursive: bool = False) -> list[Any]:
            assert path == history_root
            assert recursive is False
            return [SimpleNamespace(path=item) for item in payloads]

        def read_file(self, path: str) -> Any:
            if path not in payloads:
                raise FileNotFoundError(path)
            yield payloads[path]

    monkeypatch.setattr(
        manager.modal.Volume,
        "from_name",
        lambda *args, **kwargs: HistoryOnlyVolume(),
    )

    with pytest.raises(RuntimeError, match="spent its 2-attempts launch cap"):
        manager._require_stage_invocation_budget(
            config,
            control_plane,
            run_id,
            "quantize_text",
        )


def test_manager_rejects_a_locally_recorded_pending_stage_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    config = manager.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    run_id = "inkling-active-test"
    monkeypatch.setattr(manager, "_run_root", lambda observed_run_id: tmp_path / observed_run_id)
    deployment = {
        "app_name": "implementation-addressed",
        "version": 1,
        "tag": "sealed-tag",
        "billing_cycle_end_utc": "2099-08-01T00:00:00Z",
        "initial_billing_window_policy": "full_workflow_plus_storage_lag_v1",
        "billing_cycle_end_source": "dashboard_exact_utc",
    }
    launch_intent, launch_intent_sha256 = manager._create_launch_intent(
        config,
        control_plane,
        run_id,
        "materialize_source",
        deployment,
        {
            "function_id": "fu-materialize_source",
            "function_name": "materialize_source",
        },
    )
    receipt_path = tmp_path / run_id / "calls" / "accepted.json"
    manager._write_immutable_json(
        receipt_path,
        {
            "schema_version": manager.CALL_SCHEMA,
            "run_id": run_id,
            "stage": "materialize_source",
            "call_id": "fc-pending",
            "launch_intent": launch_intent,
            "launch_intent_sha256": launch_intent_sha256,
            "function_id": "fu-materialize_source",
            "function_name": "materialize_source",
            "config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "billing_cycle_end_utc": "2099-08-01T00:00:00Z",
            "initial_billing_window_policy": "full_workflow_plus_storage_lag_v1",
            "billing_cycle_end_source": "dashboard_exact_utc",
        },
    )

    class PendingCall:
        def get_call_graph(self) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    status=manager.InputStatus.PENDING,
                    children=[],
                )
            ]

    monkeypatch.setattr(
        manager.modal.FunctionCall,
        "from_id",
        lambda call_id: PendingCall(),
    )

    with pytest.raises(RuntimeError, match="already has pending call"):
        manager._require_no_active_stage_call(
            config,
            control_plane,
            run_id,
            "materialize_source",
        )


def test_manager_lock_is_exclusive_and_released_after_a_clean_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    run_id = "inkling-lock-test"
    monkeypatch.setattr(manager, "_run_root", lambda observed_run_id: tmp_path / observed_run_id)
    with (
        manager._exclusive_manager_operation(run_id, "first"),
        pytest.raises(RuntimeError, match="Another manager operation"),
        manager._exclusive_manager_operation(run_id, "second"),
    ):
        raise AssertionError("the second manager must not enter")
    with manager._exclusive_manager_operation(run_id, "after-release"):
        pass


def test_manager_reconciles_only_the_exact_confirmed_stale_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    run_id = "inkling-stale-lock-test"
    monkeypatch.setattr(manager, "_run_root", lambda observed_run_id: tmp_path / observed_run_id)
    lock_path = tmp_path / run_id / "control" / "manager.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(
        manager._canonical_json(
            {
                "schema_version": "inkling-modal-manager-lock-v1",
                "run_id": run_id,
                "operation": "launch:quantize_text",
                "pid": 999_999,
                "created_at": "2099-07-17T00:00:00+00:00",
                "nonce": "a" * 32,
            }
        ),
        encoding="utf-8",
    )
    lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    monkeypatch.setattr(
        manager.os,
        "kill",
        lambda pid, signal: (_ for _ in ()).throw(ProcessLookupError()),
    )

    manager._reconcile_stale_manager_lock(
        run_id,
        expected_sha256=lock_sha256,
        confirm_owner_process_stopped=True,
    )

    assert not lock_path.exists()
    receipt = tmp_path / run_id / "reconciliations" / f"manager-lock-{lock_sha256}.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["manager_lock_sha256"] == lock_sha256


def test_local_immutable_publication_never_exposes_a_partial_final_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    destination = tmp_path / "receipt.json"

    def fail_link(*args: object, **kwargs: object) -> None:
        raise OSError("simulated publication interruption")

    monkeypatch.setattr(manager.os, "link", fail_link)
    with pytest.raises(OSError, match="simulated publication interruption"):
        manager._write_immutable_json(destination, {"status": "complete"})

    assert not destination.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_manager_unresolved_pre_spawn_intent_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    config = manager.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    run_id = "inkling-unresolved-intent-test"
    monkeypatch.setattr(manager, "_run_root", lambda observed_run_id: tmp_path / observed_run_id)
    manager._create_launch_intent(
        config,
        control_plane,
        run_id,
        "materialize_source",
        {
            "app_name": "implementation-addressed",
            "version": 1,
            "tag": "sealed-tag",
            "billing_cycle_end_utc": "2099-08-01T00:00:00Z",
            "initial_billing_window_policy": "full_workflow_plus_storage_lag_v1",
            "billing_cycle_end_source": "dashboard_exact_utc",
        },
        {
            "function_id": "fu-materialize_source",
            "function_name": "materialize_source",
        },
    )

    with pytest.raises(RuntimeError, match="Unresolved pre-spawn intent"):
        manager._require_no_unresolved_launch_intent(config, control_plane, run_id)


def test_manager_reconciles_an_intent_only_from_exact_remote_call_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    config = manager.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    run_id = "inkling-reconcile-intent-test"
    monkeypatch.setattr(manager, "_run_root", lambda observed_run_id: tmp_path / observed_run_id)
    monkeypatch.setattr(
        manager,
        "_load_checked_context",
        lambda: (config, control_plane, run_id),
    )
    relative, intent_sha256 = manager._create_launch_intent(
        config,
        control_plane,
        run_id,
        "materialize_source",
        {
            "app_name": "implementation-addressed",
            "version": 1,
            "tag": "sealed-tag",
            "billing_cycle_end_utc": "2099-08-01T00:00:00Z",
            "initial_billing_window_policy": "full_workflow_plus_storage_lag_v1",
            "billing_cycle_end_source": "dashboard_exact_utc",
        },
        {
            "function_id": "fu-materialize_source",
            "function_name": "materialize_source",
        },
    )
    assert relative.startswith("launch-intents/")
    event = {
        "schema_version": "inkling-modal-stage-invocation-v2",
        "status": "started",
        "kind": "attempt",
        "sequence": 1,
        "limit": 12,
        "stage": "materialize_source",
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "launch_intent_sha256": intent_sha256,
        "call_id": "fc-intent-bound",
        "input_id": "in-intent-bound",
        "task_id": "ta-intent-bound",
        "previous_history_sha256": None,
    }
    event_id = hashlib.sha256(
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    history_path = f"runs/{run_id}/control/history/materialize_source.attempt.1.{event_id}.json"
    event_payload = (json.dumps(event, sort_keys=True, indent=2) + "\n").encode("utf-8")

    class EvidenceVolume:
        def hydrate(self) -> EvidenceVolume:
            return self

        def listdir(self, path: str, *, recursive: bool = False) -> list[Any]:
            if path.endswith("/control/history"):
                return [SimpleNamespace(path=history_path)]
            return []

        def read_file(self, path: str) -> Any:
            if path == history_path:
                return iter([event_payload])
            raise FileNotFoundError(path)

    monkeypatch.setattr(
        manager.modal.Volume,
        "from_name",
        lambda *args, **kwargs: EvidenceVolume(),
    )

    class BoundCall:
        def get_call_graph(self) -> list[SimpleNamespace]:
            return []

    monkeypatch.setattr(
        manager.modal.FunctionCall,
        "from_id",
        lambda call_id: BoundCall(),
    )
    manager.reconcile_launch_intent(intent_sha256, call_id="fc-intent-bound")
    manager._require_no_unresolved_launch_intent(config, control_plane, run_id)


def test_manager_keeps_ambiguous_intent_fail_closed_without_remote_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    config = manager.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    run_id = "inkling-no-evidence-intent-test"
    monkeypatch.setattr(manager, "_run_root", lambda observed_run_id: tmp_path / observed_run_id)
    monkeypatch.setattr(
        manager,
        "_load_checked_context",
        lambda: (config, control_plane, run_id),
    )
    _, intent_sha256 = manager._create_launch_intent(
        config,
        control_plane,
        run_id,
        "materialize_source",
        {
            "app_name": "implementation-addressed",
            "version": 1,
            "tag": "sealed-tag",
            "billing_cycle_end_utc": "2099-08-01T00:00:00Z",
            "initial_billing_window_policy": "full_workflow_plus_storage_lag_v1",
            "billing_cycle_end_source": "dashboard_exact_utc",
        },
        {
            "function_id": "fu-materialize_source",
            "function_name": "materialize_source",
        },
    )

    class EmptyVolume:
        def hydrate(self) -> EmptyVolume:
            return self

        def listdir(self, path: str, *, recursive: bool = False) -> list[Any]:
            return []

        def read_file(self, path: str) -> Any:
            # A mutable latest ledger alone is deliberately insufficient to
            # reconcile an ambiguous spawn.
            if path.endswith("materialize_source.attempts.json"):
                return iter(
                    [
                        json.dumps(
                            {
                                "launch_intent_sha256": intent_sha256,
                                "last_call_id": "fc-unproven",
                            }
                        ).encode("utf-8")
                    ]
                )
            raise FileNotFoundError(path)

    monkeypatch.setattr(
        manager.modal.Volume,
        "from_name",
        lambda *args, **kwargs: EmptyVolume(),
    )
    with pytest.raises(RuntimeError, match="must remain fail-closed"):
        manager.reconcile_launch_intent(intent_sha256, call_id="fc-unproven")


def test_manager_authoritative_history_requires_v2_task_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")
    config = manager.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    run_id = "inkling-task-history-test"
    stage = "quantize_text"
    event: dict[str, Any] = {
        "schema_version": "inkling-modal-stage-invocation-v2",
        "status": "started",
        "kind": "attempt",
        "sequence": 1,
        "limit": 2,
        "stage": stage,
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "launch_intent_sha256": "a" * 64,
        "call_id": "fc-task-bound",
        "input_id": "in-task-bound",
        "task_id": "ta-task-bound",
        "previous_history_sha256": None,
    }

    class HistoryVolume:
        def hydrate(self) -> HistoryVolume:
            return self

        def _path(self) -> str:
            event_id = hashlib.sha256(
                json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            return f"runs/{run_id}/control/history/{stage}.attempt.1.{event_id}.json"

        def listdir(self, path: str, *, recursive: bool = False) -> list[Any]:
            return [SimpleNamespace(path=self._path())]

        def read_file(self, path: str) -> Any:
            if path != self._path():
                raise FileNotFoundError(path)
            return iter([json.dumps(event).encode("utf-8")])

    monkeypatch.setattr(
        manager.modal.Volume,
        "from_name",
        lambda *args, **kwargs: HistoryVolume(),
    )
    history = manager._authoritative_invocation_history(
        config,
        control_plane,
        volume_attribute="final_volume",
        run_id=run_id,
        stage=stage,
        kind="attempt",
        limit=2,
    )
    assert history[0][1]["task_id"] == "ta-task-bound"

    event.pop("task_id")
    with pytest.raises(RuntimeError, match="history drifted"):
        manager._authoritative_invocation_history(
            config,
            control_plane,
            volume_attribute="final_volume",
            run_id=run_id,
            stage=stage,
            kind="attempt",
            limit=2,
        )


def test_manager_status_distinguishes_polling_from_terminal_timeouts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = importlib.import_module("scripts.manage_inkling_modal")

    class PollingCall:
        def get(self, *, timeout: int) -> None:
            assert timeout == 0
            raise manager.modal.exception.TimeoutError()

    monkeypatch.setattr(
        manager.modal.FunctionCall,
        "from_id",
        lambda call_id: PollingCall(),
    )
    manager.status("fc-running")
    assert '"status": "running_or_queued"' in capsys.readouterr().out

    class TerminalCall:
        def get(self, *, timeout: int) -> None:
            raise FunctionTimeoutError()

    monkeypatch.setattr(
        manager.modal.FunctionCall,
        "from_id",
        lambda call_id: TerminalCall(),
    )
    with pytest.raises(FunctionTimeoutError):
        manager.status("fc-terminal")


def _paid_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("IQL_MODAL_WORKSPACE_BUDGET_CONFIRMED", "800")
    monkeypatch.setenv(
        "IQL_MODAL_BILLING_CYCLE_END_CONFIRMED",
        "2099-08-01T00:00:00Z",
    )
    return importlib.import_module("scripts.quantize_inkling_modal")


def _install_toolchain_verification_fixture(
    paid: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    live_python_freeze: str = "alpha==1\nbeta==2\n",
) -> tuple[Path, list[list[str]]]:
    root = tmp_path / "llama.cpp"
    binary_directory = root / "build" / "bin"
    binary_directory.mkdir(parents=True)
    config = paid.InklingGGUFConfig()
    (root / ".iql-build-commit").write_text(
        f"{config.toolchain.commit}\n",
        encoding="utf-8",
    )

    for binary_name, manifest_name in (
        ("llama-quantize", ".iql-quantize.sha256"),
        ("llama-gguf-split", ".iql-gguf-split.sha256"),
    ):
        binary = binary_directory / binary_name
        binary.write_bytes(f"{binary_name}\n".encode())
        (root / manifest_name).write_text(
            f"{paid._sha256(binary)}  {binary}\n",
            encoding="utf-8",
        )

    freeze_path = root / ".iql-python-freeze.txt"
    freeze_path.write_text("alpha==1\nbeta==2\n", encoding="utf-8")
    (root / ".iql-python-freeze.sha256").write_text(
        f"{paid._sha256(freeze_path)}  {freeze_path}\n",
        encoding="utf-8",
    )
    purelib_path = root / ".iql-python-purelib.txt"
    purelib_path.write_text("/image/python/purelib\n", encoding="utf-8")
    (root / ".iql-python-purelib.sha256").write_text(
        f"{paid._sha256(purelib_path)}  {purelib_path}\n",
        encoding="utf-8",
    )
    dpkg_path = root / ".iql-dpkg-inventory.txt"
    dpkg_path.write_text("base-files=1\nca-certificates=2\n", encoding="utf-8")
    (root / ".iql-dpkg-inventory.sha256").write_text(
        f"{paid._sha256(dpkg_path)}  {dpkg_path}\n",
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def run(argv: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[:4] == ["git", "-C", str(root), "rev-parse"]:
            return subprocess.CompletedProcess(argv, 0, f"{config.toolchain.commit}\n", "")
        if argv[:4] == ["git", "-C", str(root), "diff"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[:4] == ["git", "-C", str(root), "remote"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                f"{config.toolchain.repository}\n",
                "",
            )
        if argv[:4] == [paid.sys.executable, "-m", "pip", "freeze"]:
            return subprocess.CompletedProcess(argv, 0, live_python_freeze, "")
        if argv[0] == "dpkg-query":
            return subprocess.CompletedProcess(
                argv,
                0,
                "ca-certificates=2\nbase-files=1\n",
                "",
            )
        raise AssertionError(f"Unexpected subprocess argv: {argv!r}")

    monkeypatch.setattr(paid, "LLAMA_CPP_DIR", root)
    monkeypatch.setattr(paid.sysconfig, "get_path", lambda name: "/image/python/purelib")
    monkeypatch.setattr(paid.subprocess, "run", run)
    return root, calls


def test_paid_python_inventory_build_and_runtime_use_the_same_purelib_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    requested_paths: list[str] = []

    def get_path(name: str) -> str:
        requested_paths.append(name)
        return "/image/python/purelib"

    monkeypatch.setattr(paid.sysconfig, "get_path", get_path)
    output_path = tmp_path / "python-freeze.txt"
    build_command = paid._python_inventory_build_command(output_path)

    assert paid.PYTHON_INVENTORY_SCOPE == "image_sysconfig_purelib_v1"
    assert f'sysconfig.get_path("{paid.PYTHON_INVENTORY_PATH_KEY}")' in build_command
    assert "pip freeze --all --path" in build_command
    assert str(output_path) in build_command
    assert paid._python_inventory_argv() == [
        paid.sys.executable,
        "-m",
        "pip",
        "freeze",
        "--all",
        "--path",
        "/image/python/purelib",
    ]
    assert requested_paths == [paid.PYTHON_INVENTORY_PATH_KEY]


def test_paid_toolchain_accepts_only_the_exact_scoped_python_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    _, calls = _install_toolchain_verification_fixture(
        paid,
        tmp_path,
        monkeypatch,
        live_python_freeze="beta==2\nalpha==1\n",
    )

    evidence = paid._verify_toolchain(paid.InklingGGUFConfig())

    assert evidence["python_inventory_scope"] == "image_sysconfig_purelib_v1"
    assert evidence["python_inventory_path"] == "/image/python/purelib"
    assert evidence["python_inventory_size_bytes"] == len(b"alpha==1\nbeta==2\n")
    assert evidence["python_inventory_sha256"] == hashlib.sha256(b"alpha==1\nbeta==2\n").hexdigest()
    assert [
        paid.sys.executable,
        "-m",
        "pip",
        "freeze",
        "--all",
        "--path",
        "/image/python/purelib",
    ] in calls


@pytest.mark.parametrize(
    "live_python_freeze",
    [
        "alpha==1\nbeta==2\ngamma==3\n",
        "alpha==1\n",
        "alpha==1\nbeta==3\n",
    ],
    ids=["added", "removed", "version-drift"],
)
def test_paid_toolchain_rejects_any_scoped_python_inventory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_python_freeze: str,
) -> None:
    paid = _paid_module(monkeypatch)
    _install_toolchain_verification_fixture(
        paid,
        tmp_path,
        monkeypatch,
        live_python_freeze=live_python_freeze,
    )

    with pytest.raises(RuntimeError, match="Runtime Python distributions differ"):
        paid._verify_toolchain(paid.InklingGGUFConfig())


@pytest.mark.parametrize("drift", ["path", "hash"])
def test_paid_toolchain_rejects_python_inventory_receipt_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    paid = _paid_module(monkeypatch)
    root, _ = _install_toolchain_verification_fixture(paid, tmp_path, monkeypatch)
    freeze_path = root / ".iql-python-freeze.txt"
    manifest = root / ".iql-python-freeze.sha256"
    if drift == "path":
        manifest.write_text(
            f"{paid._sha256(freeze_path)}  {root / 'other-freeze.txt'}\n",
            encoding="utf-8",
        )
        expected_error = "unexpected path"
    else:
        freeze_path.write_text("alpha==9\nbeta==2\n", encoding="utf-8")
        expected_error = "differs from its image build receipt"

    with pytest.raises(RuntimeError, match=expected_error):
        paid._verify_toolchain(paid.InklingGGUFConfig())


@pytest.mark.parametrize("drift", ["recorded-value", "manifest-path", "manifest-hash"])
def test_paid_toolchain_rejects_python_purelib_build_path_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    paid = _paid_module(monkeypatch)
    root, _ = _install_toolchain_verification_fixture(paid, tmp_path, monkeypatch)
    purelib_path = root / ".iql-python-purelib.txt"
    manifest = root / ".iql-python-purelib.sha256"
    if drift == "recorded-value":
        purelib_path.write_text("/other/python/purelib\n", encoding="utf-8")
        manifest.write_text(
            f"{paid._sha256(purelib_path)}  {purelib_path}\n",
            encoding="utf-8",
        )
        expected_error = "Runtime Python purelib path differs"
    elif drift == "manifest-path":
        manifest.write_text(
            f"{paid._sha256(purelib_path)}  {root / 'other-purelib.txt'}\n",
            encoding="utf-8",
        )
        expected_error = "unexpected path"
    else:
        manifest.write_text(
            f"{'0' * 64}  {purelib_path}\n",
            encoding="utf-8",
        )
        expected_error = "differs from its image build receipt"

    with pytest.raises(RuntimeError, match=expected_error):
        paid._verify_toolchain(paid.InklingGGUFConfig())


def test_paid_source_binding_verifies_toolchain_before_the_full_source_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    source_mount = tmp_path / "source"
    run_id = "run-order"
    source_root = source_mount / "runs" / run_id
    source_root.mkdir(parents=True)
    (source_root / "source.success.json").write_text("{}\n", encoding="utf-8")
    calls: list[str] = []
    toolchain = {"commit": paid.InklingGGUFConfig().toolchain.commit}

    def verify_toolchain(_: Any) -> dict[str, str]:
        calls.append("toolchain")
        return toolchain

    def verify_source(*_: Any, **__: Any) -> dict[str, Any]:
        calls.append("source")
        return {"verified": True}

    monkeypatch.setattr(paid, "SOURCE_MOUNT", source_mount)
    monkeypatch.setattr(paid, "_verify_toolchain", verify_toolchain)
    monkeypatch.setattr(paid, "_verify_materialized_source", verify_source)
    monkeypatch.setattr(paid, "verify_execution_bindings", lambda *args, **kwargs: None)

    result = paid._verify_source_binding(
        paid.InklingGGUFConfig(),
        run_id,
        object(),
        object(),
    )

    assert result == toolchain
    assert calls == ["toolchain", "source"]


def test_paid_attempt_ledger_enforces_the_frozen_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    config = paid.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    root = tmp_path / "run"
    root.mkdir()
    call_ids = iter(
        ("fc-attempt-1", "fc-attempt-2", "fc-attempt-3", "fc-recovery-1", "fc-recovery-2")
    )
    input_ids = iter(
        ("in-attempt-1", "in-attempt-2", "in-attempt-3", "in-recovery-1", "in-recovery-2")
    )
    monkeypatch.setattr(paid.modal, "current_function_call_id", lambda: next(call_ids))
    monkeypatch.setattr(paid.modal, "current_input_id", lambda: next(input_ids))

    intent_sha256 = "b" * 64
    monkeypatch.setenv("MODAL_TASK_ID", "ta-attempt-1")
    assert paid._begin_attempt(root, config, "quantize_text", control_plane, intent_sha256) == 1
    monkeypatch.setenv("MODAL_TASK_ID", "ta-attempt-2")
    assert paid._begin_attempt(root, config, "quantize_text", control_plane, intent_sha256) == 2
    monkeypatch.setenv("MODAL_TASK_ID", "ta-attempt-3")
    with pytest.raises(RuntimeError, match="2-attempt cost cap"):
        paid._begin_attempt(root, config, "quantize_text", control_plane, intent_sha256)

    monkeypatch.setenv("MODAL_TASK_ID", "ta-recovery-1")
    assert paid._begin_recovery(root, config, "quantize_text", control_plane, intent_sha256) == 1
    monkeypatch.setenv("MODAL_TASK_ID", "ta-recovery-2")
    with pytest.raises(RuntimeError, match="1-recovery cost cap"):
        paid._begin_recovery(root, config, "quantize_text", control_plane, intent_sha256)

    history = sorted((root / "control" / "history").glob("*.json"))
    assert len(history) == 3
    assert [json.loads(path.read_text(encoding="utf-8"))["kind"] for path in history] == [
        "attempt",
        "attempt",
        "recovery",
    ]
    latest = json.loads((root / "control" / "quantize_text.attempts.json").read_text())
    assert latest["last_history_sha256"] == paid._sha256(root / latest["last_history_path"])
    assert latest["last_task_id"] == "ta-attempt-2"


def test_paid_stage_failure_wrapper_commits_a_redacted_immutable_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    config = paid.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    acknowledgement = paid.PaidLaunchAcknowledgement(
        config_hash=config.config_hash(),
        control_plane_sha256=control_plane.tree_sha256,
        launch_intent_sha256="c" * 64,
        workspace_budget_usd=config.budget.workspace_hard_budget_usd,
        billing_cycle_end_utc="2099-08-01T00:00:00Z",
    )
    commits = 0

    class Volume:
        def commit(self) -> None:
            nonlocal commits
            commits += 1

    monkeypatch.setattr(paid.modal, "current_function_call_id", lambda: "fc-failure")
    monkeypatch.setattr(paid.modal, "current_input_id", lambda: "in-failure")
    monkeypatch.setenv("MODAL_TASK_ID", "ta-failure")

    @paid._capture_stage_failures("quantize_text")
    def failing_stage(
        config_json: str,
        run_id: str,
        budget_acknowledgement_json: str,
        control_plane_json: str,
    ) -> dict[str, Any]:
        paid._arm_failure_recording(
            tmp_path,
            config,
            "quantize_text",
            control_plane,
            acknowledgement,
            Volume(),
        )
        raise ValueError("secret-bearing message is deliberately not persisted")

    with pytest.raises(ValueError, match="secret-bearing"):
        failing_stage("config", "run", "budget", "control")

    outcomes = list((tmp_path / "control" / "outcomes").glob("*.json"))
    assert len(outcomes) == 1
    outcome = json.loads(outcomes[0].read_text(encoding="utf-8"))
    assert outcome["schema_version"] == "inkling-modal-stage-outcome-v2"
    assert outcome["status"] == "failed"
    assert outcome["exception_type"] == "builtins.ValueError"
    assert outcome["task_id"] == "ta-failure"
    assert "secret-bearing" not in outcomes[0].read_text(encoding="utf-8")
    assert commits == 1


def test_paid_immutable_history_recovers_when_latest_ledger_was_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    config = paid.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    root = tmp_path / "run"
    root.mkdir()
    invocation = {"call": "fc-first", "input": "in-first", "task": "ta-first"}
    monkeypatch.setattr(
        paid.modal,
        "current_function_call_id",
        lambda: invocation["call"],
    )
    monkeypatch.setattr(
        paid.modal,
        "current_input_id",
        lambda: invocation["input"],
    )
    monkeypatch.setenv("MODAL_TASK_ID", invocation["task"])

    assert (
        paid._begin_attempt(
            root,
            config,
            "quantize_text",
            control_plane,
            "d" * 64,
        )
        == 1
    )
    ledger = root / "control" / "quantize_text.attempts.json"
    ledger.unlink()

    # An exact replay of the same intent, call, input, and task adopts the event.
    assert (
        paid._begin_attempt(
            root,
            config,
            "quantize_text",
            control_plane,
            "d" * 64,
        )
        == 1
    )
    ledger.unlink()

    # A rescheduled container keeps the call/input but receives a new task ID and
    # therefore consumes the next slot even if the latest ledger lagged.
    invocation["task"] = "ta-second"
    monkeypatch.setenv("MODAL_TASK_ID", invocation["task"])
    assert (
        paid._begin_attempt(
            root,
            config,
            "quantize_text",
            control_plane,
            "d" * 64,
        )
        == 2
    )
    assert len(list((root / "control" / "history").glob("*.json"))) == 2


def test_paid_invocation_boundary_fails_closed_without_modal_task_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    config = paid.InklingGGUFConfig()
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    root = tmp_path / "run"
    root.mkdir()
    monkeypatch.setattr(paid.modal, "current_function_call_id", lambda: "fc-no-task")
    monkeypatch.setattr(paid.modal, "current_input_id", lambda: "in-no-task")
    monkeypatch.delenv("MODAL_TASK_ID", raising=False)

    with pytest.raises(RuntimeError, match="task ID is unavailable"):
        paid._begin_attempt(root, config, "quantize_text", control_plane, "e" * 64)

    assert not (root / "control" / "history").exists()


def test_paid_publication_evidence_rejects_metadata_and_inventory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    config = paid.InklingGGUFConfig()
    run_root = tmp_path / "run"
    partial = run_root / ".partial-q3-1"
    canonical = run_root / "q3_k_m"
    partial.mkdir(parents=True)
    output = partial / "inkling-Q3_K_M-00001-of-00001.gguf"
    output.write_bytes(b"tiny test shard")
    publication = paid.prepare_publication_intent(
        run_root,
        config_hash=config.config_hash(),
        stage="quantize_text",
        partial_directory=partial,
        canonical_directory=canonical,
        outputs=[paid._file_record(output, partial)],
    )
    assert publication.stage == "quantize_text"
    published = paid.finalize_publication(
        run_root,
        config_hash=config.config_hash(),
        stage="quantize_text",
        partial_directory=partial,
        canonical_directory=canonical,
    )
    receipt = paid._publication_evidence(run_root, "quantize_text", published)

    paid._verify_publication_evidence(
        run_root,
        config=config,
        stage="quantize_text",
        receipt=receipt,
    )
    (canonical / "unrecorded.tmp").write_bytes(b"drift")
    with pytest.raises(RuntimeError, match="inventory changed"):
        paid._verify_publication_evidence(
            run_root,
            config=config,
            stage="quantize_text",
            receipt=receipt,
        )


def test_paid_json_reader_and_safe_child_reject_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    link = root / "receipt.json"
    link.symlink_to(outside)

    with pytest.raises(RuntimeError, match="symlink"):
        paid._safe_child(root, "receipt.json")
    with pytest.raises(RuntimeError, match="regular file"):
        paid._read_json(link)


def test_paid_inventory_membership_rejects_unrecorded_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paid = _paid_module(monkeypatch)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    required = snapshot / "config.json"
    required.write_text("{}\n", encoding="utf-8")
    records = [paid._file_record(required, snapshot)]

    paid._verify_inventory_membership(snapshot, records)
    (snapshot / "unrecorded.py").write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="unrecorded"):
        paid._verify_inventory_membership(snapshot, records)


def test_final_verifier_rehashes_the_materialized_source() -> None:
    functions = _functions(PAID_SCRIPT)
    verifier = functions["_verify_final_dependency_chain"]
    calls = [
        node
        for node in ast.walk(verifier)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_verify_materialized_source"
    ]
    assert len(calls) == 1


def test_completed_stage_fast_paths_revalidate_their_exact_dependency_chains() -> None:
    functions = _functions(PAID_SCRIPT)
    expected_calls = {
        "materialize_source": {"_verify_materialized_source"},
        "convert_text_bf16": {"_verify_source_binding", "_verify_bf16_dependency"},
        "convert_multimodal_projector": {
            "_verify_source_binding",
            "_verify_bf16_dependency",
            "_verify_mmproj_dependency",
        },
        "quantize_text": {"_verify_toolchain", "_verify_final_dependency_chain"},
    }
    for stage, required in expected_calls.items():
        fast_paths = [
            node
            for node in ast.walk(functions[stage])
            if isinstance(node, ast.If)
            and any(
                isinstance(descendant, ast.Constant) and descendant.value == "already_successful"
                for descendant in ast.walk(node)
            )
        ]
        assert len(fast_paths) == 1
        called = {
            node.func.id
            for node in ast.walk(fast_paths[0])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert required <= called


def test_flattened_remote_file_import_needs_no_repository_yaml(tmp_path: Path) -> None:
    pytest.importorskip("modal")
    remote_root = tmp_path / "remote"
    remote_root.mkdir()
    flattened = remote_root / "quantize_inkling_modal.py"
    shutil.copyfile(PAID_SCRIPT, flattened)
    remote_package = remote_root / "inkling_quant_lab"
    shutil.copytree(PROJECT_ROOT / "src/inkling_quant_lab", remote_package)
    control_plane_json = inkling_control_plane_provenance(PROJECT_ROOT).canonical_json()
    code = f"""
import importlib.util
import modal
modal.is_local = lambda: False
spec = importlib.util.spec_from_file_location('remote_paid_smoke', {str(flattened)!r})
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.DEFAULT_CONFIG.source.model_id == 'thinkingmachines/Inkling'
assert not (module.LOCAL_PROJECT_ROOT / 'configs').exists()
validated = module.validate_deployed_control_plane(
    {control_plane_json!r},
    deployment_script={str(flattened)!r},
    deployed_package_root={str(remote_package)!r},
)
assert validated.tree_sha256 == {inkling_control_plane_provenance(PROJECT_ROOT).tree_sha256!r}
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(remote_root)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    assert result.returncode == 0, result.stderr
