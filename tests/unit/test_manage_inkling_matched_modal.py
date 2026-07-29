"""CPU-only control-plane contracts for the matched Modal manager.

These tests exercise only parsing and local append-only file helpers.  They
never invoke ``deploy()``, ``launch()``, a Modal Function, or a Modal Volume.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from inkling_quant_lab.gguf.inkling_matched_execution import (
    MatchedFailureReceipt,
    MatchedPublicationState,
    MatchedRollupReceipt,
    MatchedSanitizedFailureDiagnostic,
    MatchedSubject,
    MatchedSubjectReceiptReference,
    matched_failure_receipt_sha256,
    matched_rollup_receipt_sha256,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANAGER_PATH = PROJECT_ROOT / "scripts/manage_inkling_matched_modal.py"
EXPECTED_COMMANDS = {
    "inspect",
    "prepare-deploy",
    "deploy",
    "prepare-launch",
    "launch",
    "status",
}
PREPARE_OR_INSPECT_FUNCTIONS = {
    "inspect",
    "prepare_deploy",
    "prepare_launch",
}
REMOTE_EFFECT_CALLS = {
    "cancel",
    "deploy",
    "from_name",
    "read_file",
    "remote",
    "spawn",
}


def _module_ast() -> ast.Module:
    assert MANAGER_PATH.is_file(), (
        "the matched control-plane entrypoint must be scripts/manage_inkling_matched_modal.py"
    )
    return ast.parse(MANAGER_PATH.read_text(encoding="utf-8"), filename=str(MANAGER_PATH))


def _import_manager() -> ModuleType:
    assert MANAGER_PATH.is_file()
    module_name = "_inkling_matched_modal_manager_contract"
    spec = importlib.util.spec_from_file_location(module_name, MANAGER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


def _function_map(module: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        statement.name: statement
        for statement in module.body
        if isinstance(statement, ast.FunctionDef)
    }


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _calls(nodes: ast.AST | Iterable[ast.AST]) -> list[ast.Call]:
    roots = (nodes,) if isinstance(nodes, ast.AST) else tuple(nodes)
    return [
        candidate
        for root in roots
        for candidate in ast.walk(root)
        if isinstance(candidate, ast.Call)
    ]


def _modal_imports(nodes: ast.AST | Iterable[ast.AST]) -> list[str]:
    roots = (nodes,) if isinstance(nodes, ast.AST) else tuple(nodes)
    found: list[str] = []
    for root in roots:
        for node in ast.walk(root):
            if isinstance(node, ast.Import):
                found.extend(
                    alias.name
                    for alias in node.names
                    if alias.name == "modal" or alias.name.startswith("modal.")
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "modal" or module.startswith("modal."):
                    found.append(module)
    return found


def _reachable_functions(
    functions: dict[str, ast.FunctionDef],
    roots: Iterable[str],
) -> dict[str, ast.FunctionDef]:
    pending = list(roots)
    reached: dict[str, ast.FunctionDef] = {}
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        function = functions.get(name)
        assert function is not None, f"expected top-level {name}()"
        reached[name] = function
        for call in _calls(function):
            called = _call_name(call)
            if called in functions and called not in reached:
                pending.append(called)
    return reached


def _subcommand_names(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("manager parser does not define subcommands")


def _call_line(function: ast.FunctionDef, name: str) -> int:
    lines = [call.lineno for call in _calls(function) if _call_name(call) == name]
    assert len(lines) == 1, f"{function.name}() must call {name}() exactly once"
    return lines[0]


def test_import_and_prepare_paths_have_no_modal_import_or_remote_effects() -> None:
    module = _module_ast()
    functions = _function_map(module)

    assert not _modal_imports(
        statement
        for statement in module.body
        if isinstance(statement, (ast.Import, ast.ImportFrom))
    )

    reached = _reachable_functions(functions, PREPARE_OR_INSPECT_FUNCTIONS)
    for name, function in reached.items():
        modal_imports = _modal_imports(function)
        assert not modal_imports, f"{name}() must remain a local-only operation"
        effects = {_call_name(call) for call in _calls(function)} & REMOTE_EFFECT_CALLS
        assert not effects, f"{name}() unexpectedly exposes remote effects: {sorted(effects)}"
        assert not [
            call
            for call in _calls(function)
            if _call_name(call).startswith("_modal")
            or _call_name(call) in {"_load_modal", "_import_modal"}
        ]

    # Loading the module itself is part of inspect/prepare.  It must stay safe
    # even on a CPU-only machine where Modal is not installed or authenticated.
    imported = _import_manager()
    assert callable(imported._build_parser)


def test_commands_are_separate_and_no_command_ever_defaults_to_launch() -> None:
    manager = _import_manager()
    parser = manager._build_parser()

    assert _subcommand_names(parser) == EXPECTED_COMMANDS
    with pytest.raises(SystemExit):
        parser.parse_args([])

    module = _module_ast()
    functions = _function_map(module)
    deploy_reachable = _reachable_functions(functions, {"deploy"})
    launch_reachable = _reachable_functions(functions, {"launch"})

    deploy_calls = {
        _call_name(call) for function in deploy_reachable.values() for call in _calls(function)
    }
    assert (
        not {
            "spawn",
            "_launch_once",
            "_publish_remote_launch_intent",
            "_publish_post_spawn_acceptance",
        }
        & deploy_calls
    )

    launch_calls = {
        _call_name(call) for function in launch_reachable.values() for call in _calls(function)
    }
    assert not {"deploy", "_deploy_control_plane"} & launch_calls


def test_prepare_commands_issue_exact_content_addressed_challenges() -> None:
    module = _module_ast()
    functions = _function_map(module)
    expectations = {
        "prepare_deploy": "MatchedDeployConfirmationChallenge",
        "prepare_launch": "MatchedLaunchConfirmationChallenge",
    }
    for root, challenge_type in expectations.items():
        reached = _reachable_functions(functions, {root})
        names = {
            node.id
            for function in reached.values()
            for node in ast.walk(function)
            if isinstance(node, ast.Name)
        }
        calls = {_call_name(call) for function in reached.values() for call in _calls(function)}
        assert challenge_type in names
        assert "confirmation_text" in calls
        assert "_write_immutable_json" in calls

    for root, challenge_type in (
        ("deploy", "MatchedDeployConfirmationChallenge"),
        ("launch", "MatchedLaunchConfirmationChallenge"),
    ):
        reached = _reachable_functions(functions, {root})
        names = {
            node.id
            for function in reached.values()
            for node in ast.walk(function)
            if isinstance(node, ast.Name)
        }
        calls = {_call_name(call) for function in reached.values() for call in _calls(function)}
        assert challenge_type in names
        assert "confirm" in calls


def test_local_receipts_are_canonical_idempotent_and_never_overwritten(
    tmp_path: Path,
) -> None:
    manager = _import_manager()
    path = tmp_path / "receipt.json"
    first: dict[str, Any] = {"z": 1, "a": ["stable"]}

    manager._write_immutable_json(path, first)
    expected = (
        json.dumps(
            first,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    assert path.read_text(encoding="utf-8") == expected

    manager._write_immutable_json(path, first)
    with pytest.raises(
        (FileExistsError, RuntimeError, ValueError),
        match=r"immutable|different|exists|replace",
    ):
        manager._write_immutable_json(path, {"a": ["changed"], "z": 1})
    assert path.read_text(encoding="utf-8") == expected

    run_root = manager._run_root("inkling-matched-contract-test")
    assert run_root == PROJECT_ROOT or PROJECT_ROOT in run_root.parents
    assert "artifacts" in run_root.parts


def test_every_control_transition_persists_through_the_immutable_writer() -> None:
    module = _module_ast()
    functions = _function_map(module)

    for root in ("prepare_deploy", "deploy", "prepare_launch", "launch"):
        reached = _reachable_functions(functions, {root})
        assert any(
            _call_name(call) == "_write_immutable_json"
            for function in reached.values()
            for call in _calls(function)
        ), f"{root}() must persist an immutable local control receipt"


@pytest.mark.parametrize(
    "object_id",
    (
        None,
        "",
        "fu-FunctionNotCall",
        "fc-hyphen-not-allowed",
        "fc-unicode-\N{SNOWMAN}",
    ),
)
def test_call_id_validation_fails_closed_without_publishing(
    object_id: object,
) -> None:
    manager = _import_manager()

    with pytest.raises((RuntimeError, TypeError, ValueError), match=r"call|ID|object"):
        manager._validated_call_id(SimpleNamespace(object_id=object_id))
    assert (
        manager._validated_call_id(SimpleNamespace(object_id="fc-MatchedCall123"))
        == "fc-MatchedCall123"
    )


def test_launch_uploads_intent_before_spawn_and_accepts_only_a_valid_call_id() -> None:
    module = _module_ast()
    functions = _function_map(module)
    launch_once = functions.get("_launch_once")
    assert launch_once is not None, (
        "post-spawn authority must be isolated in _launch_once() so its ordering "
        "can be reviewed without executing paid work"
    )

    publish_intent_line = _call_line(launch_once, "_publish_remote_launch_intent")
    spawn_line = _call_line(launch_once, "spawn")
    validate_call_id_line = _call_line(launch_once, "_validated_call_id")
    publish_acceptance_line = _call_line(
        launch_once,
        "_publish_post_spawn_acceptance",
    )
    assert publish_intent_line < spawn_line < validate_call_id_line < publish_acceptance_line

    guarded_call_id = [
        node
        for node in ast.walk(launch_once)
        if isinstance(node, ast.Try)
        and any(_call_name(call) == "_validated_call_id" for call in _calls(node.body))
        and any(_call_name(call) == "cancel" for call in _calls(node.handlers))
    ]
    assert len(guarded_call_id) == 1, (
        "an invalid provider call ID must request cancellation before acceptance publication"
    )

    cancel_calls = [
        call for call in _calls(guarded_call_id[0].handlers) if _call_name(call) == "cancel"
    ]
    assert len(cancel_calls) == 1
    cancel_keywords = {
        keyword.arg: keyword.value
        for keyword in cancel_calls[0].keywords
        if keyword.arg is not None
    }
    assert ast.literal_eval(cancel_keywords["terminate_containers"]) is True


def _deployment(manager: ModuleType) -> Any:
    control_sha256 = "a" * 64
    return manager.MatchedDeploymentIdentity(
        control_plane_sha256=control_sha256,
        app_name=manager.matched_app_name(control_sha256),
        deployment_version=1,
        deployment_tag=manager.matched_deployment_tag(control_sha256),
        function_id="fu-MatchedFunction123",
        attempt_registry_id="di-MatchedAttempt123",
        attempt_registry_created_at_utc="2026-07-28T12:00:00.000000Z",
        evidence_volume_id="vo-MatchedEvidence123",
    )


def _acceptance(manager: ModuleType) -> Any:
    deployment = _deployment(manager)
    return manager.MatchedPostSpawnAcceptance(
        accepted_at_utc="2026-07-28T12:01:00.000000Z",
        run_id="inkling-matched-contract-test",
        launch_intent_sha256="b" * 64,
        call_id="fc-MatchedCall123",
        deployment=deployment,
        matched_config_sha256="c" * 64,
        control_plane_sha256=deployment.control_plane_sha256,
    )


def _attempt_claim(
    manager: ModuleType,
    *,
    call_id: str = "fc-MatchedCall123",
    input_id: str = "in-MatchedInput123:0-0",
    task_id: str = "ta-MatchedTask123",
) -> Any:
    acceptance = _acceptance(manager)
    deployment = acceptance.deployment
    return manager.MatchedAttemptClaim(
        registry_id=deployment.attempt_registry_id,
        registry_created_at_utc=deployment.attempt_registry_created_at_utc,
        registry_key=manager.matched_attempt_registry_key(acceptance.run_id),
        run_id=acceptance.run_id,
        call_id=call_id,
        input_id=input_id,
        task_id=task_id,
        launch_intent_sha256=acceptance.launch_intent_sha256,
        post_spawn_acceptance_path=manager.matched_post_spawn_acceptance_path(
            acceptance.run_id,
            acceptance.launch_intent_sha256,
        ),
        post_spawn_acceptance_sha256=acceptance.acceptance_sha256(),
        matched_config_sha256=acceptance.matched_config_sha256,
        control_plane_sha256=acceptance.control_plane_sha256,
    )


class _FakeAttemptRegistry:
    def __init__(
        self,
        payload: object | None,
        *,
        object_id: str = "di-MatchedAttempt123",
        name: str = "inkling-matched-attempt-registry-v1",
        created_at: float = 1_753_704_000.0,
    ) -> None:
        self.payload = payload
        self.object_id = object_id
        self.name = name
        self.created_at = created_at
        self.hydrated = False
        self.keys: list[str] = []

    def hydrate(self) -> None:
        self.hydrated = True

    def info(self) -> SimpleNamespace:
        return SimpleNamespace(name=self.name)

    def _get_metadata(self) -> SimpleNamespace:
        return SimpleNamespace(creation_info=SimpleNamespace(created_at=self.created_at))

    def contains(self, key: str) -> bool:
        self.keys.append(key)
        return self.payload is not None

    def get(self, key: str) -> object:
        self.keys.append(key)
        return self.payload


def test_fresh_attempt_registry_is_hydrated_and_seal_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _import_manager()
    deployment = _deployment(manager)
    created_at = manager.datetime.fromisoformat(
        deployment.attempt_registry_created_at_utc.replace("Z", "+00:00")
    ).timestamp()
    registry = _FakeAttemptRegistry(None, created_at=created_at)

    def from_name(
        name: str,
        *,
        environment_name: str,
        create_if_missing: bool,
    ) -> _FakeAttemptRegistry:
        assert name == manager.MATCHED_ATTEMPT_REGISTRY_NAME
        assert environment_name == manager.MATCHED_ENVIRONMENT_NAME
        assert create_if_missing is False
        return registry

    monkeypatch.setattr(
        manager,
        "_load_modal",
        lambda: SimpleNamespace(Dict=SimpleNamespace(from_name=from_name)),
    )

    assert manager._fresh_attempt_registry(deployment) is registry
    assert registry.hydrated is True

    registry.object_id = "di-DifferentRegistry123"
    with pytest.raises(RuntimeError, match="differs from the local seal"):
        manager._fresh_attempt_registry(deployment)


class _FakeUpload:
    def __init__(self, volume: _FakeVolume) -> None:
        self._volume = volume

    def __enter__(self) -> _FakeUpload:
        return self

    def put_file(self, source: Any, path: str, *, mode: int) -> None:
        assert path.startswith("/")
        assert mode == 0o400
        self._volume.payload = source.read()

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        if exc_type is None and self._volume.raise_after_upload:
            raise OSError("upload response was lost")


class _FakeVolume:
    def __init__(
        self,
        payload: bytes | None = None,
        *,
        raise_after_upload: bool = False,
    ) -> None:
        self.payload = payload
        self.raise_after_upload = raise_after_upload

    def read_file(self, path: str) -> tuple[bytes]:
        assert path.startswith("/")
        if self.payload is None:
            raise FileNotFoundError(path)
        return (self.payload,)

    def batch_upload(self, *, force: bool) -> _FakeUpload:
        assert force is False
        return _FakeUpload(self)


def test_post_spawn_acceptance_uses_a_fresh_exact_volume_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _import_manager()
    acceptance = _acceptance(manager)
    payload = acceptance.canonical_bytes()
    initial = _FakeVolume(raise_after_upload=True)
    fresh = _FakeVolume(payload)

    monkeypatch.setattr(manager, "_fresh_evidence_volume", lambda deployment: fresh)

    manager._publish_post_spawn_acceptance(initial, acceptance)


def test_post_spawn_acceptance_distinguishes_mismatch_from_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _import_manager()
    acceptance = _acceptance(manager)

    monkeypatch.setattr(
        manager,
        "_fresh_evidence_volume",
        lambda deployment: _FakeVolume(b"different"),
    )
    with pytest.raises(manager._PostSpawnAcceptanceMismatchError, match="absent"):
        manager._publish_post_spawn_acceptance(_FakeVolume(), acceptance)

    def unavailable(deployment: object) -> object:
        raise OSError("fresh read unavailable")

    monkeypatch.setattr(manager, "_fresh_evidence_volume", unavailable)
    with pytest.raises(manager._PostSpawnAcceptanceStateUnknownError, match="unknown"):
        manager._publish_post_spawn_acceptance(_FakeVolume(), acceptance)


class _FakeCall:
    def __init__(self, object_id: str) -> None:
        self.object_id = object_id
        self.cancelled = False

    def cancel(self, *, terminate_containers: bool) -> None:
        assert terminate_containers is True
        self.cancelled = True


class _FakeFunction:
    def __init__(self, call: _FakeCall) -> None:
        self.call = call

    def spawn(self, run_id: str, intent_sha256: str) -> _FakeCall:
        assert run_id == "inkling-matched-contract-test"
        assert intent_sha256 == "b" * 64
        return self.call


def _launch_intent(manager: ModuleType) -> Any:
    deployment = _deployment(manager)
    return SimpleNamespace(
        run_id="inkling-matched-contract-test",
        deployment=deployment,
        reviewed_inputs=SimpleNamespace(
            matched_config_sha256="c" * 64,
            control_plane_sha256=deployment.control_plane_sha256,
        ),
        intent_sha256=lambda: "b" * 64,
    )


def test_launch_cancels_proven_mismatch_but_retains_unknown_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _import_manager()
    monkeypatch.setattr(manager, "_publish_remote_launch_intent", lambda *args: None)

    def publish_mismatch(*args: object) -> None:
        raise manager._PostSpawnAcceptanceMismatchError("absent")

    mismatch_call = _FakeCall("fc-MatchedCall123")
    monkeypatch.setattr(
        manager,
        "_publish_post_spawn_acceptance",
        publish_mismatch,
    )
    with pytest.raises(RuntimeError, match="cancellation was requested"):
        manager._launch_once(
            _FakeFunction(mismatch_call),
            object(),
            _launch_intent(manager),
        )
    assert mismatch_call.cancelled is True

    def publish_unknown(*args: object) -> None:
        raise manager._PostSpawnAcceptanceStateUnknownError("unknown")

    unknown_call = _FakeCall("fc-MatchedCall123")
    monkeypatch.setattr(
        manager,
        "_publish_post_spawn_acceptance",
        publish_unknown,
    )
    with pytest.raises(RuntimeError, match="was not cancelled"):
        manager._launch_once(
            _FakeFunction(unknown_call),
            object(),
            _launch_intent(manager),
        )
    assert unknown_call.cancelled is False


def _execution_bytes(value: Any) -> bytes:
    return json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _failure_receipt(run_id: str) -> MatchedFailureReceipt:
    publication = MatchedPublicationState(
        state="not_started",
        success_files_installed=False,
        commit_requested=False,
        volume_reloaded=False,
        mounted_readback_verified=False,
        terminal_success_proven=False,
        failure_receipt_allowed=True,
    )
    diagnostic = MatchedSanitizedFailureDiagnostic(
        schema_version="inkling-matched-sanitized-failure-v1",
        category="server_start",
        failure_type="RuntimeError",
        subject=MatchedSubject.BF16,
        message_sha256=hashlib.sha256(b"private provider detail").hexdigest(),
        raw_message_recorded=False,
        traceback_recorded=False,
        raw_server_log_recorded=False,
    )
    payload: dict[str, object] = {
        "schema_version": "inkling-matched-failure-v1",
        "status": "failed",
        "stage": "matched_smoke",
        "run_id": run_id,
        "allocation_identity_sha256": "1" * 64,
        "runtime_identity_sha256": "2" * 64,
        "probe_control_sha256": "3" * 64,
        "subject_at_failure": MatchedSubject.BF16,
        "completed_subject_receipts": (),
        "diagnostic": diagnostic,
        "publication": publication,
        "prompt_text_recorded": False,
        "output_text_recorded": False,
        "completed_at_utc": "2026-07-28T12:15:00+00:00",
    }
    payload["receipt_sha256"] = matched_failure_receipt_sha256(payload)
    return MatchedFailureReceipt.model_validate(payload)


def _success_receipt(run_id: str) -> MatchedRollupReceipt:
    common = {
        "allocation_identity_sha256": "1" * 64,
        "runtime_identity_sha256": "2" * 64,
        "probe_control_sha256": "3" * 64,
        "projector_path": "/baseline/mmproj/mmproj-BF16.gguf",
        "projector_sha256": "4" * 64,
    }
    subjects = (
        MatchedSubjectReceiptReference(
            subject=MatchedSubject.BF16,
            subject_ordinal=0,
            receipt_sha256="5" * 64,
            server_process_id=101,
            **common,
        ),
        MatchedSubjectReceiptReference(
            subject=MatchedSubject.Q3,
            subject_ordinal=1,
            receipt_sha256="6" * 64,
            server_process_id=202,
            **common,
        ),
    )
    payload: dict[str, object] = {
        "schema_version": "inkling-matched-rollup-v1",
        "status": "passed",
        "stage": "matched_smoke",
        "run_id": run_id,
        "subjects": subjects,
        "allocation_identity_sha256": common["allocation_identity_sha256"],
        "runtime_identity_sha256": common["runtime_identity_sha256"],
        "probe_control_sha256": common["probe_control_sha256"],
        "both_subjects_passed": True,
        "same_allocation": True,
        "same_runtime": True,
        "same_probe_control": True,
        "fresh_server_processes": True,
        "prompt_text_recorded": False,
        "output_text_recorded": False,
        "quality_measured": False,
        "benchmark_measured": False,
        "completed_at_utc": "2026-07-28T12:15:00+00:00",
    }
    payload["receipt_sha256"] = matched_rollup_receipt_sha256(payload)
    return MatchedRollupReceipt.model_validate(payload)


class _TerminalVolume:
    def __init__(self, records: dict[str, bytes]) -> None:
        self.records = records

    def read_file(self, path: str) -> tuple[bytes]:
        relative = path.removeprefix("/")
        try:
            return (self.records[relative],)
        except KeyError:
            raise FileNotFoundError(path) from None

    def listdir(self, path: str, *, recursive: bool) -> list[SimpleNamespace]:
        assert recursive is False
        relative = path.removeprefix("/")
        prefix = f"{relative}/"
        return [
            SimpleNamespace(
                path=record_path,
                type=SimpleNamespace(name="FILE"),
                size=len(payload),
            )
            for record_path, payload in sorted(self.records.items())
            if record_path.startswith(prefix) and "/" not in record_path.removeprefix(prefix)
        ]


def _attempt_claim_records(manager: ModuleType, claim: Any) -> dict[str, bytes]:
    return {
        manager.matched_attempt_claim_path(
            claim.run_id,
            claim.claim_sha256(),
        ): claim.canonical_bytes()
    }


def _inspect_attempt_claims(
    manager: ModuleType,
    *,
    live_payload: object | None,
    records: dict[str, bytes],
) -> Any:
    acceptance = _acceptance(manager)
    return manager._validated_attempt_claim_inspection(
        _FakeAttemptRegistry(live_payload),
        _TerminalVolume(records),
        run_id=acceptance.run_id,
        deployment=acceptance.deployment,
        binding=manager._StatusLaunchBinding(
            launch_intent=_launch_intent(manager),
            acceptance=acceptance,
        ),
    )


def test_attempt_claim_inspection_matches_exact_live_and_durable_bytes() -> None:
    manager = _import_manager()
    claim = _attempt_claim(manager)
    registry = _FakeAttemptRegistry(claim.canonical_bytes())
    inspection = _inspect_attempt_claims(
        manager,
        live_payload=registry.payload,
        records=_attempt_claim_records(manager, claim),
    )

    assert inspection.registry_key == claim.registry_key
    assert inspection.live_claim == claim
    assert inspection.durable_claim == claim
    assert inspection.live_payload == claim.canonical_bytes()
    assert inspection.durable_payload == claim.canonical_bytes()
    assert inspection.claim_sha256 == claim.claim_sha256()


def test_attempt_claim_inspection_fails_closed_when_live_registry_is_unreadable() -> None:
    manager = _import_manager()
    acceptance = _acceptance(manager)

    class UnreadableRegistry(_FakeAttemptRegistry):
        def contains(self, key: str) -> bool:
            raise OSError(f"unreadable {key}")

    with pytest.raises(RuntimeError, match="live matched attempt registry is unreadable"):
        manager._validated_attempt_claim_inspection(
            UnreadableRegistry(None),
            _TerminalVolume({}),
            run_id=acceptance.run_id,
            deployment=acceptance.deployment,
            binding=manager._StatusLaunchBinding(
                launch_intent=_launch_intent(manager),
                acceptance=acceptance,
            ),
        )


def test_attempt_claim_inspection_rejects_present_key_with_missing_value() -> None:
    manager = _import_manager()
    acceptance = _acceptance(manager)

    class MissingValueRegistry(_FakeAttemptRegistry):
        def contains(self, key: str) -> bool:
            self.keys.append(key)
            return True

    with pytest.raises(
        RuntimeError,
        match="live matched attempt claim is invalid",
    ):
        manager._validated_attempt_claim_inspection(
            MissingValueRegistry(None),
            _TerminalVolume({}),
            run_id=acceptance.run_id,
            deployment=acceptance.deployment,
            binding=manager._StatusLaunchBinding(
                launch_intent=_launch_intent(manager),
                acceptance=acceptance,
            ),
        )


def test_attempt_claim_inspection_uses_durable_claim_after_live_key_expiry() -> None:
    manager = _import_manager()
    claim = _attempt_claim(manager)
    inspection = _inspect_attempt_claims(
        manager,
        live_payload=None,
        records=_attempt_claim_records(manager, claim),
    )

    assert inspection.live_claim is None
    assert inspection.durable_claim == claim
    assert inspection.claim_sha256 == claim.claim_sha256()


def test_attempt_claim_inspection_retains_live_only_hard_stop_state() -> None:
    manager = _import_manager()
    claim = _attempt_claim(manager)
    inspection = _inspect_attempt_claims(
        manager,
        live_payload=claim.canonical_bytes(),
        records={},
    )

    assert inspection.live_claim == claim
    assert inspection.durable_claim is None
    assert inspection.attempt_consumed_before_volume_bookkeeping is True


def test_attempt_claim_inspection_rejects_live_and_durable_disagreement() -> None:
    manager = _import_manager()
    live = _attempt_claim(manager)
    durable = _attempt_claim(manager, input_id="in-DifferentInput123:0-0")

    with pytest.raises(RuntimeError, match="differs from its durable Volume claim"):
        _inspect_attempt_claims(
            manager,
            live_payload=live.canonical_bytes(),
            records=_attempt_claim_records(manager, durable),
        )


@pytest.mark.parametrize(
    "live_payload",
    (
        "not-bytes",
        b"{}",
        b'{"schema_version":"inkling-matched-attempt-claim-v1"}\n',
    ),
)
def test_attempt_claim_inspection_rejects_malformed_live_values(
    live_payload: object,
) -> None:
    manager = _import_manager()

    with pytest.raises(RuntimeError, match="live matched attempt claim is invalid"):
        _inspect_attempt_claims(
            manager,
            live_payload=live_payload,
            records={},
        )


def test_attempt_claim_inspection_rejects_more_than_one_durable_claim() -> None:
    manager = _import_manager()
    first = _attempt_claim(manager)
    second = _attempt_claim(manager, input_id="in-DifferentInput123:0-0")
    records = {
        **_attempt_claim_records(manager, first),
        **_attempt_claim_records(manager, second),
    }

    with pytest.raises(RuntimeError, match="more than one durable attempt claim"):
        _inspect_attempt_claims(
            manager,
            live_payload=None,
            records=records,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("call_id", "fc-DifferentCall123"),
        ("launch_intent_sha256", "d" * 64),
        ("post_spawn_acceptance_sha256", "e" * 64),
        ("matched_config_sha256", "f" * 64),
        ("control_plane_sha256", "0" * 64),
        ("registry_id", "di-DifferentRegistry123"),
        ("registry_created_at_utc", "2026-07-28T12:00:01.000000Z"),
    ),
)
def test_attempt_claim_inspection_rejects_changed_launch_bindings(
    field: str,
    value: str,
) -> None:
    manager = _import_manager()
    claim = _attempt_claim(manager).model_copy(update={field: value})

    with pytest.raises(RuntimeError, match="live matched attempt claim is invalid"):
        _inspect_attempt_claims(
            manager,
            live_payload=claim.canonical_bytes(),
            records={},
        )


def _publication_records(
    manager: ModuleType,
    reference: Any,
    *,
    attempt_claim_sha256: str | None = None,
    final_status: str = "confirmed",
) -> dict[str, bytes]:
    if attempt_claim_sha256 is None:
        attempt_claim_sha256 = _attempt_claim(manager).claim_sha256()
    initial = manager.MatchedPublicationSnapshot(
        publication_id="8" * 64,
        run_id=reference.run_id,
        attempt_claim_sha256=attempt_claim_sha256,
        status="not_started",
        cycle=0,
    )
    installing = manager.MatchedPublicationSnapshot(
        **{
            **initial.model_dump(mode="json"),
            "status": "installing",
            "cycle": 1,
            "terminal_receipt": reference.model_dump(mode="json"),
        }
    )
    final = manager.MatchedPublicationSnapshot(
        **{
            **installing.model_dump(mode="json"),
            "status": final_status,
            "mounted_reload_completed": final_status == "confirmed",
        }
    )
    return {
        manager.matched_publication_state_path(
            snapshot.run_id,
            snapshot.state_sha256(),
        ): snapshot.canonical_bytes()
        for snapshot in (initial, installing, final)
    }


@pytest.mark.parametrize(
    ("outcome", "receipt_factory", "receipt_type"),
    (
        ("success", _success_receipt, MatchedRollupReceipt),
        ("failure", _failure_receipt, MatchedFailureReceipt),
    ),
)
def test_terminal_status_validates_exact_reference_and_receipt_semantics(
    outcome: str,
    receipt_factory: Any,
    receipt_type: type[Any],
) -> None:
    manager = _import_manager()
    run_id = "inkling-matched-contract-test"
    receipt = receipt_factory(run_id)
    payload = _execution_bytes(receipt)
    reference = manager.build_matched_terminal_receipt_reference(
        payload,
        run_id=run_id,
        outcome=outcome,
    )
    records = {
        reference.path: payload,
        **_publication_records(manager, reference),
    }

    evidence = manager._validated_terminal_evidence(
        _TerminalVolume(records),
        run_id=run_id,
        durable_attempt_claim=_attempt_claim(manager),
    )

    assert evidence is not None
    assert evidence.reference == reference
    assert isinstance(evidence.receipt, receipt_type)


def test_terminal_status_rejects_wrong_paths_conflicts_and_weak_envelopes() -> None:
    manager = _import_manager()
    run_id = "inkling-matched-contract-test"
    failure = _failure_receipt(run_id)
    failure_payload = _execution_bytes(failure)
    failure_reference = manager.build_matched_terminal_receipt_reference(
        failure_payload,
        run_id=run_id,
        outcome="failure",
    )
    success = _success_receipt(run_id)
    success_payload = _execution_bytes(success)
    success_reference = manager.build_matched_terminal_receipt_reference(
        success_payload,
        run_id=run_id,
        outcome="success",
    )

    wrong_path = failure_reference.path.replace(
        failure_reference.content_sha256,
        "f" * 64,
    )
    with pytest.raises(RuntimeError, match="differs"):
        manager._validated_terminal_evidence(
            _TerminalVolume(
                {
                    wrong_path: failure_payload,
                    **_publication_records(manager, failure_reference),
                }
            ),
            run_id=run_id,
            durable_attempt_claim=_attempt_claim(manager),
        )

    with pytest.raises(RuntimeError, match="conflicting"):
        manager._validated_terminal_evidence(
            _TerminalVolume(
                {
                    failure_reference.path: failure_payload,
                    success_reference.path: success_payload,
                    **_publication_records(manager, failure_reference),
                }
            ),
            run_id=run_id,
            durable_attempt_claim=_attempt_claim(manager),
        )

    weak_payload = json.dumps(
        {
            "schema_version": "inkling-matched-failure-v1",
            "status": "failed",
            "stage": "matched_smoke",
            "run_id": run_id,
            "prompt_text_recorded": False,
            "output_text_recorded": False,
            "receipt_sha256": "7" * 64,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    weak_reference = manager.build_matched_terminal_receipt_reference(
        weak_payload,
        run_id=run_id,
        outcome="failure",
    )
    with pytest.raises(RuntimeError, match="semantics"):
        manager._validated_terminal_evidence(
            _TerminalVolume(
                {
                    weak_reference.path: weak_payload,
                    **_publication_records(manager, weak_reference),
                }
            ),
            run_id=run_id,
            durable_attempt_claim=_attempt_claim(manager),
        )


@pytest.mark.parametrize("final_status", ("installing", "unknown"))
def test_terminal_status_requires_confirmed_mounted_publication(
    final_status: str,
) -> None:
    manager = _import_manager()
    run_id = "inkling-matched-contract-test"
    receipt = _failure_receipt(run_id)
    payload = _execution_bytes(receipt)
    reference = manager.build_matched_terminal_receipt_reference(
        payload,
        run_id=run_id,
        outcome="failure",
    )
    records = {
        reference.path: payload,
        **_publication_records(
            manager,
            reference,
            final_status=final_status,
        ),
    }

    with pytest.raises(RuntimeError, match=r"unknown|not yet confirmed"):
        manager._validated_terminal_evidence(
            _TerminalVolume(records),
            run_id=run_id,
            durable_attempt_claim=_attempt_claim(manager),
        )


def test_terminal_status_binds_publication_to_durable_attempt_claim() -> None:
    manager = _import_manager()
    claim = _attempt_claim(manager)
    receipt = _success_receipt(claim.run_id)
    payload = _execution_bytes(receipt)
    reference = manager.build_matched_terminal_receipt_reference(
        payload,
        run_id=claim.run_id,
        outcome="success",
    )

    with pytest.raises(RuntimeError, match="durable attempt claim"):
        manager._validated_terminal_evidence(
            _TerminalVolume(
                {
                    reference.path: payload,
                    **_publication_records(
                        manager,
                        reference,
                        attempt_claim_sha256=claim.claim_sha256(),
                    ),
                }
            ),
            run_id=claim.run_id,
            durable_attempt_claim=None,
        )

    with pytest.raises(RuntimeError, match="attempt claim"):
        manager._validated_terminal_evidence(
            _TerminalVolume(
                {
                    reference.path: payload,
                    **_publication_records(
                        manager,
                        reference,
                        attempt_claim_sha256="0" * 64,
                    ),
                }
            ),
            run_id=claim.run_id,
            durable_attempt_claim=claim,
        )


def _patch_status_context(
    manager: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    volume: _TerminalVolume,
    registry: _FakeAttemptRegistry,
) -> dict[str, int]:
    deployment = _deployment(manager)
    acceptance = _acceptance(manager)
    context = SimpleNamespace(
        run_id=acceptance.run_id,
        provenance=SimpleNamespace(control_plane_sha256=deployment.control_plane_sha256),
    )
    calls = {"fresh_volume": 0, "fresh_registry": 0}
    monkeypatch.setattr(manager, "_build_reviewed_context", lambda: context)
    monkeypatch.setattr(manager, "_read_deployment", lambda run_id: deployment)
    monkeypatch.setattr(
        manager,
        "_read_status_launch_binding",
        lambda *args: manager._StatusLaunchBinding(
            launch_intent=_launch_intent(manager),
            acceptance=acceptance,
        ),
    )

    def fresh(observed_deployment: object) -> _TerminalVolume:
        assert observed_deployment == deployment
        calls["fresh_volume"] += 1
        return volume

    monkeypatch.setattr(manager, "_fresh_evidence_volume", fresh)

    def fresh_registry(observed_deployment: object) -> _FakeAttemptRegistry:
        assert observed_deployment == deployment
        calls["fresh_registry"] += 1
        return registry

    monkeypatch.setattr(manager, "_fresh_attempt_registry", fresh_registry)
    return calls


def test_status_reads_fresh_volume_and_returns_nonzero_for_validated_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _import_manager()
    acceptance = _acceptance(manager)
    receipt = _failure_receipt(acceptance.run_id)
    terminal_payload = _execution_bytes(receipt)
    reference = manager.build_matched_terminal_receipt_reference(
        terminal_payload,
        run_id=acceptance.run_id,
        outcome="failure",
    )
    remote_acceptance = manager.matched_post_spawn_acceptance_path(
        acceptance.run_id,
        acceptance.launch_intent_sha256,
    )
    claim = _attempt_claim(manager)
    volume = _TerminalVolume(
        {
            remote_acceptance: acceptance.canonical_bytes(),
            **_attempt_claim_records(manager, claim),
            reference.path: terminal_payload,
            **_publication_records(
                manager,
                reference,
                attempt_claim_sha256=claim.claim_sha256(),
            ),
        }
    )
    calls = _patch_status_context(
        manager,
        monkeypatch,
        volume,
        _FakeAttemptRegistry(None),
    )

    def provider_must_not_be_polled(call_id: str) -> tuple[str, int]:
        raise AssertionError(f"provider was polled after terminal evidence: {call_id}")

    monkeypatch.setattr(manager, "_provider_call_state", provider_must_not_be_polled)

    assert manager.status(call_id=acceptance.call_id) == 1
    output = json.loads(capsys.readouterr().out)
    assert calls == {"fresh_volume": 1, "fresh_registry": 1}
    assert output["status"] == "failed"
    assert output["evidence_status"] == "validated_terminal_failure"
    assert output["terminal_receipt"] == reference.model_dump(mode="json")
    assert output["attempt_claim_sha256"] == claim.claim_sha256()
    assert output["attempt_consumed"] is True
    assert output["retry_allowed"] is False
    assert output["failure_message_sha256"] == receipt.diagnostic.message_sha256
    assert "private provider detail" not in json.dumps(output)
    assert "result" not in output


def test_status_returns_zero_only_for_confirmed_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _import_manager()
    acceptance = _acceptance(manager)
    receipt = _success_receipt(acceptance.run_id)
    terminal_payload = _execution_bytes(receipt)
    reference = manager.build_matched_terminal_receipt_reference(
        terminal_payload,
        run_id=acceptance.run_id,
        outcome="success",
    )
    remote_acceptance = manager.matched_post_spawn_acceptance_path(
        acceptance.run_id,
        acceptance.launch_intent_sha256,
    )
    claim = _attempt_claim(manager)
    volume = _TerminalVolume(
        {
            remote_acceptance: acceptance.canonical_bytes(),
            **_attempt_claim_records(manager, claim),
            reference.path: terminal_payload,
            **_publication_records(
                manager,
                reference,
                attempt_claim_sha256=claim.claim_sha256(),
            ),
        }
    )
    calls = _patch_status_context(
        manager,
        monkeypatch,
        volume,
        _FakeAttemptRegistry(None),
    )

    def provider_must_not_be_polled(call_id: str) -> tuple[str, int]:
        raise AssertionError(f"provider was polled after success: {call_id}")

    monkeypatch.setattr(manager, "_provider_call_state", provider_must_not_be_polled)

    assert manager.status(call_id=acceptance.call_id) == 0
    output = json.loads(capsys.readouterr().out)
    assert calls == {"fresh_volume": 1, "fresh_registry": 1}
    assert output["status"] == "passed"
    assert output["evidence_status"] == "validated_terminal_success"
    assert output["both_subjects_passed"] is True
    assert output["quality_measured"] is False
    assert output["benchmark_measured"] is False
    assert output["attempt_claim_sha256"] == claim.claim_sha256()
    assert output["attempt_consumed"] is True
    assert output["retry_allowed"] is False
    assert "result" not in output


def test_status_reports_live_only_attempt_as_consumed_without_provider_poll(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = _import_manager()
    acceptance = _acceptance(manager)
    claim = _attempt_claim(manager)
    remote_acceptance = manager.matched_post_spawn_acceptance_path(
        acceptance.run_id,
        acceptance.launch_intent_sha256,
    )
    volume = _TerminalVolume({remote_acceptance: acceptance.canonical_bytes()})
    registry = _FakeAttemptRegistry(claim.canonical_bytes())
    calls = _patch_status_context(manager, monkeypatch, volume, registry)

    def provider_must_not_be_polled(call_id: str) -> tuple[str, int]:
        raise AssertionError(f"provider was polled after the Dict claim: {call_id}")

    monkeypatch.setattr(manager, "_provider_call_state", provider_must_not_be_polled)

    assert manager.status(call_id=acceptance.call_id) == 1
    output = json.loads(capsys.readouterr().out)
    assert calls == {"fresh_volume": 1, "fresh_registry": 1}
    assert registry.keys == [claim.registry_key, claim.registry_key]
    assert output["status"] == "attempt_consumed"
    assert output["evidence_status"] == "attempt_consumed_before_volume_bookkeeping"
    assert output["attempt_registry_key"] == claim.registry_key
    assert output["attempt_registry_key_present"] is True
    assert output["attempt_consumed"] is True
    assert output["retry_allowed"] is False
    assert output["volume_attempt_claim_path"] is None
    assert "result" not in output


def test_provider_status_never_downloads_or_prints_function_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _import_manager()

    class FakeCall:
        def get_call_graph(self) -> list[SimpleNamespace]:
            return [
                SimpleNamespace(
                    status=SimpleNamespace(name="SUCCESS"),
                    children=[],
                )
            ]

        def get(self, *, timeout: int) -> object:
            raise AssertionError(f"Function return was downloaded with timeout {timeout}")

    fake_call = FakeCall()
    fake_modal = SimpleNamespace(FunctionCall=SimpleNamespace(from_id=lambda call_id: fake_call))
    monkeypatch.setattr(manager, "_load_modal", lambda: fake_modal)

    assert manager._provider_call_state("fc-MatchedCall123") == (
        "returned_without_terminal_evidence",
        1,
    )


def test_main_propagates_terminal_status_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _import_manager()
    monkeypatch.setattr(manager, "status", lambda *, call_id: 1)

    assert manager.main(["status", "--call-id", "fc-MatchedCall123"]) == 1
