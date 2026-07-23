from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inkling_quant_lab.gguf.inkling_smoke import LLAMA_SERVER_AUDIT_LOG_VERBOSITY
from inkling_quant_lab.gguf.inkling_smoke_acceptance import (
    SmokePostSpawnAcceptance,
    smoke_post_spawn_acceptance_path,
    validate_smoke_post_spawn_acceptance,
)
from inkling_quant_lab.gguf.inkling_smoke_attempt import (
    SmokeAttemptRegistryClaim,
    claim_smoke_attempt,
    smoke_attempt_registry_key,
    validate_smoke_attempt_registry_claim,
)
from inkling_quant_lab.gguf.inkling_smoke_authorization import (
    smoke_launch_intent_remote_path,
)
from inkling_quant_lab.gguf.inkling_smoke_execution import (
    SmokeGpuTopologyEvidence,
    SmokeInvocationEvidence,
    SmokeNvidiaSmiTopologyDiagnostic,
    SmokeSubprocessFailureEvidence,
    parse_cgroup_cpu_quota_millicores,
    parse_cgroup_memory_limit_bytes,
    smoke_deployment_tag,
    strict_json_object,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = PROJECT_ROOT / "scripts/smoke_inkling_modal.py"
MANAGER_PATH = PROJECT_ROOT / "scripts/manage_inkling_smoke_modal.py"
_DEFAULT_CHILDREN = object()


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _app_functions(module: ast.Module) -> list[ast.FunctionDef]:
    functions: list[ast.FunctionDef] = []
    for node in module.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "app"
                and decorator.func.attr == "function"
            ):
                functions.append(node)
    return functions


def _assignment_literal(module: ast.Module, name: str) -> object:
    for node in module.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment {name}")


def _isolated_runner_functions(*names: str) -> dict[str, object]:
    module = _module(RUNNER_PATH)
    functions = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {function.name for function in functions} == set(names)
    namespace: dict[str, object] = {
        "EVIDENCE_MOUNT": Path("/evidence"),
        "Any": Any,
        "Mapping": Mapping,
        "Path": Path,
        "Sequence": Sequence,
        "os": os,
        "parse_cgroup_cpu_quota_millicores": parse_cgroup_cpu_quota_millicores,
        "parse_cgroup_memory_limit_bytes": parse_cgroup_memory_limit_bytes,
        "stat": stat,
        "suppress": suppress,
    }
    isolated = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(RUNNER_PATH), "exec"), namespace)
    return namespace


def _isolated_manager_functions(*names: str) -> dict[str, object]:
    module = _module(MANAGER_PATH)
    functions = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {function.name for function in functions} == set(names)
    namespace: dict[str, object] = {
        "Any": Any,
        "Decimal": Decimal,
        "InklingGGUFConfig": Any,
        "InklingSmokeConfig": Any,
        "InklingVerifiedExportReference": Any,
        "Mapping": Mapping,
        "Path": Path,
        "Sequence": Sequence,
        "SmokeControlPlaneProvenance": Any,
        "SmokeLaunchDeploymentIdentity": Any,
        "json": json,
        "smoke_deployment_tag": smoke_deployment_tag,
    }
    isolated = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(MANAGER_PATH), "exec"), namespace)
    return namespace


def _isolated_smoke_exception_handler() -> dict[str, object]:
    module = _module(RUNNER_PATH)
    smoke = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "smoke_test"
    )
    smoke_try = next(node for node in smoke.body if isinstance(node, ast.Try))
    assert len(smoke_try.handlers) == 1

    template = ast.parse(
        """
def _execute(
    error,
    run_root,
    invocation,
    success_publication_started,
    success_publication_confirmed,
):
    process = None
    monitor = None
    log_handle = None
    try:
        raise error
    except BaseException:
        pass
"""
    )
    function = template.body[0]
    assert isinstance(function, ast.FunctionDef)
    wrapper_try = next(node for node in function.body if isinstance(node, ast.Try))
    wrapper_try.handlers = smoke_try.handlers
    namespace: dict[str, object] = {
        "_DurablePublicationStateUnknownError": _TestDurablePublicationStateUnknownError,
        "suppress": suppress,
    }
    ast.fix_missing_locations(template)
    exec(compile(template, str(RUNNER_PATH), "exec"), namespace)
    return namespace


class _TestDurablePublicationStateUnknownError(RuntimeError):
    pass


class _TestVolumeUploadStateUnknownError(RuntimeError):
    pass


class _CommitSequence:
    def __init__(
        self,
        outcomes: Sequence[object],
        reload_outcomes: Sequence[object] | None = None,
    ) -> None:
        self._outcomes = tuple(outcomes)
        self._reload_outcomes = tuple(
            reload_outcomes if reload_outcomes is not None else (None,) * len(self._outcomes)
        )
        self.commit_calls = 0
        self.reload_calls = 0

    def commit(self) -> None:
        if self.commit_calls >= len(self._outcomes):
            raise AssertionError("unexpected terminal evidence commit")
        outcome = self._outcomes[self.commit_calls]
        self.commit_calls += 1
        if isinstance(outcome, BaseException):
            raise outcome

    def reload(self) -> None:
        if self.reload_calls >= len(self._reload_outcomes):
            raise AssertionError("unexpected terminal evidence reload")
        outcome = self._reload_outcomes[self.reload_calls]
        self.reload_calls += 1
        if isinstance(outcome, BaseException):
            raise outcome


def _isolated_atomic_bytes(
    *,
    readback_error: OSError | None = None,
    fail_directory_fsync: bool = False,
) -> tuple[Any, SimpleNamespace]:
    namespace = _isolated_runner_functions("_atomic_bytes")
    state = SimpleNamespace(fsync_calls=0, rename_calls=0)

    def create_safe_evidence_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def rename_noreplace(source: Path, destination: Path) -> None:
        state.rename_calls += 1
        assert not destination.exists()
        os.rename(source, destination)

    def read_existing_regular_bytes(path: Path) -> bytes:
        if readback_error is not None:
            raise readback_error
        return path.read_bytes()

    def fsync(descriptor: int) -> None:
        state.fsync_calls += 1
        if fail_directory_fsync and state.fsync_calls == 2:
            raise OSError("directory fsync failed after rename")
        os.fsync(descriptor)

    isolated_os = SimpleNamespace(
        O_CREAT=os.O_CREAT,
        O_EXCL=os.O_EXCL,
        O_NOFOLLOW=getattr(os, "O_NOFOLLOW", 0),
        O_RDONLY=os.O_RDONLY,
        O_WRONLY=os.O_WRONLY,
        close=os.close,
        fdopen=os.fdopen,
        fsync=fsync,
        getpid=os.getpid,
        open=os.open,
    )
    namespace.update(
        {
            "_DurablePublicationStateUnknownError": (_TestDurablePublicationStateUnknownError),
            "_create_safe_evidence_parent": create_safe_evidence_parent,
            "_read_existing_regular_bytes": read_existing_regular_bytes,
            "_rename_noreplace": rename_noreplace,
            "os": isolated_os,
            "secrets": SimpleNamespace(token_hex=lambda _length: "0" * 32),
            "suppress": suppress,
        }
    )
    return namespace["_atomic_bytes"], state


def _isolated_receipt_publisher(
    tmp_path: Path,
    *,
    publisher_name: str,
    remote_path: str,
    expected_allow_identical: bool,
    commit_outcomes: Sequence[object],
    reload_outcomes: Sequence[object] | None,
    read_outcomes: Sequence[object],
) -> tuple[Any, _CommitSequence, list[bytes]]:
    namespace = _isolated_runner_functions(
        "_commit_and_reconcile_volume_files",
        publisher_name,
    )
    volume = _CommitSequence(commit_outcomes, reload_outcomes)
    installed: list[bytes] = []
    read_offset = 0

    def canonical_json(value: Mapping[str, Any]) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def atomic_bytes(
        path: Path,
        payload: bytes,
        *,
        allow_identical: bool,
    ) -> None:
        if not installed:
            assert allow_identical is expected_allow_identical
        else:
            assert allow_identical is True
        if path.exists():
            if allow_identical and path.read_bytes() == payload:
                return
            raise RuntimeError("test publisher refused to replace evidence")
        installed.append(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def read_reloaded_volume_bytes(observed_remote_path: str) -> bytes | None:
        nonlocal read_offset
        assert observed_remote_path == remote_path
        if read_offset >= len(read_outcomes):
            raise AssertionError("unexpected reloaded terminal evidence read")
        outcome = read_outcomes[read_offset]
        read_offset += 1
        if isinstance(outcome, BaseException):
            raise outcome
        assert outcome is None or isinstance(outcome, bytes)
        mounted_path = tmp_path / observed_remote_path
        if outcome is None:
            with suppress(FileNotFoundError):
                mounted_path.unlink()
        else:
            mounted_path.parent.mkdir(parents=True, exist_ok=True)
            mounted_path.write_bytes(outcome)
        return outcome

    namespace.update(
        {
            "EVIDENCE_MOUNT": tmp_path,
            "MAX_TERMINAL_RECEIPT_BYTES": 64 * 1024,
            "SMOKE_STAGE": "smoke_test",
            "_DurablePublicationStateUnknownError": (_TestDurablePublicationStateUnknownError),
            "_atomic_bytes": atomic_bytes,
            "_canonical_json": canonical_json,
            "_read_reloaded_volume_bytes": read_reloaded_volume_bytes,
            "_safe_child": lambda root, *parts: root.joinpath(*parts),
            "evidence_volume": volume,
            "sys": sys,
        }
    )
    return namespace[publisher_name], volume, installed


def _isolated_success_publisher(
    tmp_path: Path,
    *,
    commit_outcomes: Sequence[object],
    reload_outcomes: Sequence[object] | None = None,
    read_outcomes: Sequence[object],
) -> tuple[Any, _CommitSequence, list[bytes]]:
    return _isolated_receipt_publisher(
        tmp_path,
        publisher_name="_publish_success_receipt",
        remote_path="run/smoke_test.success.json",
        expected_allow_identical=False,
        commit_outcomes=commit_outcomes,
        reload_outcomes=reload_outcomes,
        read_outcomes=read_outcomes,
    )


def _isolated_failure_publisher(
    tmp_path: Path,
    *,
    commit_outcomes: Sequence[object],
    reload_outcomes: Sequence[object] | None = None,
    read_outcomes: Sequence[object],
) -> tuple[Any, _CommitSequence, list[bytes]]:
    return _isolated_receipt_publisher(
        tmp_path,
        publisher_name="_publish_failure_receipt",
        remote_path="run/control/outcomes/smoke_test.failed.test.json",
        expected_allow_identical=True,
        commit_outcomes=commit_outcomes,
        reload_outcomes=reload_outcomes,
        read_outcomes=read_outcomes,
    )


class _AcceptanceUploadScenario:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.initial_volume = self._InitialVolume(self)
        self.fresh_volume = object()
        self.persisted: dict[str, bytes] = {}
        self.volume_requests: list[bool] = []
        self.batch_force_values: list[bool] = []

    class _InitialVolume:
        def __init__(self, scenario: _AcceptanceUploadScenario) -> None:
            self.scenario = scenario

        def batch_upload(self, *, force: bool) -> _AcceptanceUploadScenario._Batch:
            self.scenario.batch_force_values.append(force)
            return _AcceptanceUploadScenario._Batch(self.scenario)

    class _Batch:
        def __init__(self, scenario: _AcceptanceUploadScenario) -> None:
            self.scenario = scenario

        def __enter__(self) -> _AcceptanceUploadScenario._Batch:
            return self

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            traceback: object,
        ) -> None:
            return None

        def put_file(self, local_path: Path, remote_path: str) -> None:
            payload = local_path.read_bytes()
            if self.scenario.mode == "absent":
                raise OSError("upload failed before apply")
            if self.scenario.mode == "different":
                self.scenario.persisted[remote_path] = b"different\n"
                raise OSError("upload response failed after a different apply")
            self.scenario.persisted[remote_path] = payload
            raise OSError("upload response failed after apply")

    def evidence_volume(self, _config: object, *, create: bool) -> object:
        self.volume_requests.append(create)
        return self.initial_volume if create else self.fresh_volume

    def read_volume_bytes(self, volume: object, remote_path: str) -> bytes | None:
        if volume is self.initial_volume:
            return None
        assert volume is self.fresh_volume
        if self.mode == "unreadable":
            raise OSError("fresh Volume read unavailable")
        return self.persisted.get(remote_path)


def _isolated_post_spawn_acceptance_publisher(
    tmp_path: Path,
    scenario: _AcceptanceUploadScenario,
) -> Any:
    namespace = _isolated_manager_functions(
        "_reconcile_uploaded_volume_bytes",
        "_publish_post_spawn_acceptance",
    )

    def write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
        payload = (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
        ).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    namespace.update(
        {
            "SMOKE_STAGE": "smoke_test",
            "SmokePostSpawnAcceptance": SmokePostSpawnAcceptance,
            "_VolumeUploadStateUnknownError": _TestVolumeUploadStateUnknownError,
            "_evidence_volume": scenario.evidence_volume,
            "_read_volume_bytes": scenario.read_volume_bytes,
            "_run_root": lambda run_id: tmp_path / run_id,
            "_sha256": lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
            "_write_immutable_json": write_immutable_json,
            "json": json,
            "smoke_post_spawn_acceptance_path": smoke_post_spawn_acceptance_path,
            "sys": sys,
            "validate_smoke_post_spawn_acceptance": validate_smoke_post_spawn_acceptance,
        }
    )
    return namespace["_publish_post_spawn_acceptance"]


class _FakeSmokeCall:
    object_id = "fc-TestCall123"

    def __init__(self) -> None:
        self.cancel_requests: list[bool] = []

    def cancel(self, *, terminate_containers: bool) -> None:
        self.cancel_requests.append(terminate_containers)


class _FakeSmokeFunction:
    def __init__(self, call: _FakeSmokeCall) -> None:
        self.call = call
        self.spawn_calls: list[tuple[object, ...]] = []

    def spawn(self, *args: object) -> _FakeSmokeCall:
        self.spawn_calls.append(args)
        return self.call


class _FakeLaunchAcknowledgement:
    def __init__(self, **values: object) -> None:
        self.values = values

    def canonical_json(self) -> str:
        return "{}"


def _run_isolated_launch(
    tmp_path: Path,
    publisher: Any,
) -> tuple[_FakeSmokeCall, list[dict[str, Any]], RuntimeError | None]:
    namespace = _isolated_manager_functions("_launch_locked")
    config_hash = "c" * 64
    reference_sha256 = "d" * 64
    control_sha256 = "e" * 64
    run_id = "inkling-smoke-test"
    binding = {"function_id": "fu-TestFunction123", "function_name": "smoke_test"}
    deployment = {
        "app_name": f"inkling-q3-smoke-{control_sha256[:12]}",
        "version": 1,
        "tag": smoke_deployment_tag(control_sha256),
        "billing_cycle_end_utc": "2026-08-01T00:00:00Z",
        "function_binding": binding,
        "attempt_registry_name": "inkling-smoke-attempt-registry-v1",
        "attempt_registry_id": "di-TestRegistry123",
        "attempt_registry_created_at_utc": "2026-07-22T12:00:00.000000Z",
    }
    deployment_identity = SimpleNamespace(
        app_name=deployment["app_name"],
        environment_name="inkling-quant",
        deployment_version=deployment["version"],
        deployment_tag=deployment["tag"],
        function_id=binding["function_id"],
        function_name=binding["function_name"],
        attempt_registry_name=deployment["attempt_registry_name"],
        attempt_registry_id=deployment["attempt_registry_id"],
        attempt_registry_created_at_utc=deployment["attempt_registry_created_at_utc"],
    )
    config = SimpleNamespace(
        canonical_json=lambda: "{}",
        config_hash=lambda: config_hash,
    )
    reference = SimpleNamespace(
        canonical_json=lambda: "{}",
        reference_sha256=reference_sha256,
    )
    control_plane = SimpleNamespace(
        canonical_json=lambda: "{}",
        tree_sha256=control_sha256,
    )
    call = _FakeSmokeCall()
    function = _FakeSmokeFunction(call)
    receipts: list[dict[str, Any]] = []

    namespace.update(
        {
            "CALL_SCHEMA": "inkling-smoke-modal-call-v4",
            "SMOKE_ENVIRONMENT_NAME": "inkling-quant",
            "SMOKE_STAGE": "smoke_test",
            "SMOKE_WORKSPACE_BUDGET_USD": Decimal("800"),
            "SmokeLaunchAcknowledgement": _FakeLaunchAcknowledgement,
            "UTC": UTC,
            "_VolumeUploadStateUnknownError": _TestVolumeUploadStateUnknownError,
            "_create_launch_intent": lambda *args: ("launch-intents/intent.json", "a" * 64),
            "_deployment_identity": lambda value: deployment_identity,
            "_function_binding": lambda value: binding,
            "_hydrate_binding": lambda **kwargs: (function, binding),
            "_publish_post_spawn_acceptance": publisher,
            "_publish_remote_launch_intent": lambda *args: (
                f"runs/{run_id}/control/launch-authorizations/{'a' * 64}.json"
            ),
            "_require_fresh_attempt": lambda *args: None,
            "_require_no_unresolved_intent": lambda *args: None,
            "_require_paid_gate": lambda *args: deployment["billing_cycle_end_utc"],
            "_revalidate_history": lambda value: None,
            "_run_root": lambda value: tmp_path / value,
            "_validated_deployment": lambda *args: deployment,
            "_validated_local_calls": lambda *args: [],
            "_write_immutable_json": lambda path, value: receipts.append(dict(value)),
            "datetime": datetime,
            "json": json,
            "print": lambda *args, **kwargs: None,
            "re": re,
        }
    )
    launch = namespace["_launch_locked"]
    assert callable(launch)
    launch_error: RuntimeError | None = None
    try:
        launch(
            config,
            reference,
            object(),
            control_plane,
            run_id,
            Decimal("800"),
        )
    except RuntimeError as error:
        launch_error = error
    return call, receipts, launch_error


def test_post_spawn_acceptance_accepts_exact_fresh_state_after_response_error(
    tmp_path: Path,
) -> None:
    scenario = _AcceptanceUploadScenario("exact-after-apply")
    publisher = _isolated_post_spawn_acceptance_publisher(tmp_path, scenario)

    call, receipts, launch_error = _run_isolated_launch(tmp_path, publisher)

    assert launch_error is None
    assert call.cancel_requests == []
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["status"] == "accepted"
    assert "post_spawn_error" not in receipt
    assert receipt["remote_post_spawn_acceptance"] in scenario.persisted
    assert (
        scenario.persisted[receipt["remote_post_spawn_acceptance"]]
        == (tmp_path / "inkling-smoke-test" / receipt["post_spawn_acceptance"]).read_bytes()
    )
    assert scenario.volume_requests == [True, False]
    assert scenario.initial_volume is not scenario.fresh_volume
    assert scenario.batch_force_values == [False]


@pytest.mark.parametrize(
    "mode",
    ("absent", "different"),
    ids=("error-before-apply", "different-state-after-error"),
)
def test_post_spawn_acceptance_cancels_when_fresh_state_is_not_exact(
    tmp_path: Path,
    mode: str,
) -> None:
    scenario = _AcceptanceUploadScenario(mode)
    publisher = _isolated_post_spawn_acceptance_publisher(tmp_path, scenario)

    call, receipts, launch_error = _run_isolated_launch(tmp_path, publisher)

    assert launch_error is not None
    assert str(launch_error) == (
        "Post-spawn acceptance failed; retained call "
        "fc-TestCall123 with status cancellation_requested"
    )
    assert call.cancel_requests == [True]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["status"] == "cancellation_requested"
    assert receipt["post_spawn_error"] == (
        "Remote post-spawn acceptance differs after upload reconciliation"
    )
    assert receipt["cancel_error"] is None
    assert "post_spawn_acceptance" not in receipt
    assert scenario.volume_requests == [True, False]
    assert scenario.initial_volume is not scenario.fresh_volume
    assert scenario.batch_force_values == [False]


def test_post_spawn_acceptance_unknown_fresh_state_does_not_cancel(
    tmp_path: Path,
) -> None:
    scenario = _AcceptanceUploadScenario("unreadable")
    publisher = _isolated_post_spawn_acceptance_publisher(tmp_path, scenario)

    call, receipts, launch_error = _run_isolated_launch(tmp_path, publisher)

    assert launch_error is not None
    assert str(launch_error) == (
        "Post-spawn acceptance failed; retained call "
        "fc-TestCall123 with status acceptance_state_unknown"
    )
    assert call.cancel_requests == []
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["status"] == "acceptance_state_unknown"
    assert receipt["post_spawn_error"] == (
        "Could not reconcile the remote post-spawn acceptance after its upload"
    )
    assert receipt["cancel_error"] is None
    assert "post_spawn_acceptance" not in receipt
    assert scenario.volume_requests == [True, False]
    assert scenario.initial_volume is not scenario.fresh_volume
    assert scenario.batch_force_values == [False]


def _call_graph_node(
    *,
    status_name: object = "PENDING",
    task_id: object = "ta-task",
    children: object = _DEFAULT_CHILDREN,
    **overrides: object,
) -> SimpleNamespace:
    values: dict[str, object] = {
        "input_id": "in-input",
        "function_call_id": "fc-call",
        "task_id": task_id,
        "status": SimpleNamespace(name=status_name),
        "function_name": "smoke_test",
        "module_name": "smoke_inkling_modal",
        "children": [] if children is _DEFAULT_CHILDREN else children,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_smoke_runner_exposes_one_exact_paid_function() -> None:
    module = _module(RUNNER_PATH)
    functions = _app_functions(module)

    assert [function.name for function in functions] == ["smoke_test"]
    function = functions[0]
    assert [argument.arg for argument in function.args.args] == [
        "config_json",
        "reference_json",
        "run_id",
        "launch_intent_sha256",
        "acknowledgement_json",
        "control_plane_json",
    ]
    decorator = function.decorator_list[0]
    assert isinstance(decorator, ast.Call)
    keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
    assert ast.literal_eval(keywords["gpu"]) == "B300:2"
    assert ast.literal_eval(keywords["cpu"]) == (16, 16)
    assert ast.literal_eval(keywords["memory"]) == (65_536, 65_536)
    assert ast.literal_eval(keywords["ephemeral_disk"]) == 524_288
    assert ast.literal_eval(keywords["retries"]) == 0
    assert ast.literal_eval(keywords["timeout"]) == 7_200
    assert ast.literal_eval(keywords["startup_timeout"]) == 900
    assert ast.literal_eval(keywords["max_containers"]) == 1
    assert ast.literal_eval(keywords["single_use_containers"]) is True
    assert ast.literal_eval(keywords["block_network"]) is True

    volumes = keywords["volumes"]
    assert isinstance(volumes, ast.Dict)
    assert all(
        isinstance(key, ast.Call) and isinstance(key.func, ast.Name) and key.func.id == "str"
        for key in volumes.keys
    )


def test_smoke_runner_build_and_mount_contract_is_closed() -> None:
    module = _module(RUNNER_PATH)
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert _assignment_literal(module, "BUILD_TARGETS") == (
        "llama-cli",
        "llama-server",
        "llama-bench",
        "llama-perplexity",
    )
    assert ".with_mount_options(\n    read_only=True," in source
    assert '"PYTHONDONTWRITEBYTECODE": "1"' in source
    assert 'ignore=["**/__pycache__/**", "**/*.pyc", "**/*.pyo", "**/.DS_Store"]' in source
    assert "local_entrypoint" not in source
    assert "SERVER_REQUIRED_FLAGS" in source
    assert "SOURCE_BLOB_PINS" in source
    assert "SOURCE_CONTRACT_ASSERTIONS" in source
    assert LLAMA_SERVER_AUDIT_LOG_VERBOSITY == 4
    assert '("common/log.h", "#define LOG_LEVEL_TRACE  4")' in source
    assert '("common/log.cpp", "case GGML_LOG_LEVEL_INFO:  return LOG_LEVEL_TRACE;")' in source
    assert '"CMakeLists.txt", "option(LLAMA_BUILD_UI"' in source
    assert '"CMakeLists.txt", "option(LLAMA_USE_PREBUILT_UI"' in source
    assert "-DLLAMA_BUILD_UI=OFF" in source
    assert "-DLLAMA_USE_PREBUILT_UI=OFF" in source
    assert "LLAMA_BUILD_UI:BOOL=OFF" in source
    assert "LLAMA_USE_PREBUILT_UI:BOOL=OFF" in source
    assert "build/tools/ui/dist/index.html" in source
    assert "build/tools/ui/dist.tar.gz" in source
    assert ".add_local_file(PATCH_PATH, str(REMOTE_PATCH), copy=True)" in source
    assert "git -C {LLAMA_CPP_DIR} apply --check {REMOTE_PATCH}" in source
    assert ".iql-smoke-patch.sha256" in source
    assert ".iql-patched-diff.sha256" in source
    assert ".iql-llama-server-help.txt" in source
    assert "CUDA_DRIVER_STUB" in source
    assert 'f"-D{CUDA_DRIVER_STUB_RPATH_LINK_DEFINITION}"' in source
    assert "LD_LIBRARY_PATH={CUDA_DRIVER_LINK_DIR}" in source
    assert '"LD_LIBRARY_PATH":' not in source
    assert "libggml-cuda.so must require libcuda.so.1" in source
    assert "must not retain the CUDA driver stub path" in source
    assert "unlink {CUDA_DRIVER_LINK_SONAME}" in source
    assert "rmdir {CUDA_DRIVER_LINK_DIR}" in source
    assert "test ! -e {CUDA_DRIVER_LINK_DIR}" in source
    assert "parse_cuda_driver_linkage" in source
    assert "Runtime CUDA driver resolved to the build stub" in source


def test_smoke_runner_binds_trace_verbosity_for_required_info_evidence() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    command_start = source.index("def _server_command(")
    command_end = source.index("\n\ndef _terminate_process(", command_start)
    command_source = source[command_start:command_end]

    assert '"--log-verbosity"' in command_source
    assert "str(DEFAULT_CONFIG.runtime.log_verbosity)" in command_source
    assert '"3"' not in command_source
    assert "_rename_noreplace(temporary, path)" in source
    assert "os.lstat(destination)" in source
    assert "os.rename(source, destination)" in source
    assert "renameat2" not in source
    assert "os.link(" not in source
    assert "os.replace(temporary, path)" not in source
    assert "Published immutable evidence failed read-back" in source
    assert '"post_sampling_probs": probe.post_sampling_probs' in source
    assert "_full_vocab_summary" not in source
    assert "full_vocabulary_check" not in source
    assert '"IQL_SMOKE_BACKEND_AUDIT": "1"' in source
    assert '"IQL_SMOKE_RAW_LOGIT_AUDIT": "1"' in source
    assert '"post_rehash_load_to_health_seconds": load_seconds' in source
    assert "cold_load_to_health_seconds" not in source
    assert "importlib.metadata.distributions(path=[purelib])" in source
    assert "parse_dpkg_inventory(runtime_dpkg)" in source
    assert "${binary:Package}" in source
    assert "${Package}" not in source
    assert "parse_nvcc_version(nvcc)" in source
    assert "platform.python_implementation()" in source
    assert "platform.python_version()" in source
    assert "immutable_source_tree_identity(" in source
    assert 'getattr(modal, "__file__", None)' in source
    assert '["nvidia-smi", "topo", "-m"]' in source
    assert 'fields = "uuid,name,memory.total,driver_version,compute_cap"' in source
    assert "pci.bus_id" not in source
    assert "enumerate_cuda_driver_gpus(cuda_driver_library_path)" in source
    assert "combine_gpu_identity(" in source
    assert '"--query-gpu=uuid,memory.used,utilization.gpu"' in source
    assert "parse_nvidia_smi_monitor_csv(output, expected_uuids=self._gpu_uuids)" in source
    assert "_require_gpu_inventory_unchanged(" in source
    assert 'Path("/proc/cpuinfo")' in source
    assert 'Path("/proc/meminfo")' in source
    assert 'getattr(os, "sched_getaffinity", None)' in source
    assert "resolve_current_process_cgroup_hierarchy_paths(" in source
    assert 'Path("/proc/self/cgroup")' in source
    assert 'Path("/proc/self/mountinfo")' in source
    assert 'Path("/sys/fs/cgroup/cpu.max")' not in source
    assert 'Path("/sys/fs/cgroup/memory.max")' not in source
    assert 'Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")' not in source


def test_atomic_bytes_types_post_rename_readback_error_as_unknown(
    tmp_path: Path,
) -> None:
    readback_error = OSError("readback failed after rename")
    publish, state = _isolated_atomic_bytes(readback_error=readback_error)
    assert callable(publish)
    destination = tmp_path / "evidence" / "receipt.json"
    payload = b'{"status":"passed"}\n'

    with pytest.raises(
        _TestDurablePublicationStateUnknownError,
        match="unknown durable state",
    ) as unknown:
        publish(destination, payload, allow_identical=False)

    assert unknown.value.__cause__ is readback_error
    assert destination.read_bytes() == payload
    assert state.rename_calls == 1
    assert state.fsync_calls == 1


def test_atomic_bytes_types_post_rename_directory_fsync_error_as_unknown(
    tmp_path: Path,
) -> None:
    publish, state = _isolated_atomic_bytes(fail_directory_fsync=True)
    assert callable(publish)
    destination = tmp_path / "evidence" / "receipt.json"
    payload = b'{"status":"passed"}\n'

    with pytest.raises(
        _TestDurablePublicationStateUnknownError,
        match="unknown durable state",
    ) as unknown:
        publish(destination, payload, allow_identical=False)

    assert isinstance(unknown.value.__cause__, OSError)
    assert destination.read_bytes() == payload
    assert state.rename_calls == 1
    assert state.fsync_calls == 2


def test_smoke_host_and_gpu_preflight_bracket_subject_rehash_before_server_start() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    smoke_start = source.index("def smoke_test(")
    smoke_source = source[smoke_start:]

    runtime_offset = smoke_source.index("_runtime_toolchain_evidence()")
    driver_path_offset = smoke_source.index(
        'cuda_driver_library_path = str(toolchain["cuda_driver_library_path"])'
    )
    gpu_offset = smoke_source.index("_nvidia_identity(cuda_driver_library_path)")
    peer_topology_offset = smoke_source.index("_cuda_peer_topology(cuda_driver_library_path)")
    peer_binding_offset = smoke_source.index(
        "_require_peer_topology_matches_hardware(hardware, peer_topology)"
    )
    topology_diagnostic_offset = smoke_source.index("_nvidia_topology_diagnostic()")
    topology_identity_offset = smoke_source.index("_gpu_topology_identity(")
    host_offset = smoke_source.index("_host_identity(")
    rehash_offset = smoke_source.index("_verify_complete_subject(reference)")
    cgroup_recheck_offset = smoke_source.index("_require_cgroup_inventory_unchanged(host)")
    gpu_recheck_offset = smoke_source.index("_require_gpu_inventory_unchanged(")
    second_gpu_identity_offset = smoke_source.index(
        "_nvidia_identity(cuda_driver_library_path)",
        gpu_recheck_offset,
    )
    second_peer_topology_offset = smoke_source.index(
        "_cuda_peer_topology(cuda_driver_library_path)",
        rehash_offset,
    )
    second_peer_binding_offset = smoke_source.index(
        "_require_peer_topology_matches_hardware(hardware, observed_peer_topology)",
        second_peer_topology_offset,
    )
    peer_unchanged_offset = smoke_source.index(
        "_require_peer_topology_unchanged(peer_topology, observed_peer_topology)",
        second_peer_binding_offset,
    )
    server_offset = smoke_source.index("subprocess.Popen(")

    assert (
        runtime_offset
        < driver_path_offset
        < gpu_offset
        < peer_topology_offset
        < peer_binding_offset
        < topology_identity_offset
        < topology_diagnostic_offset
        < host_offset
        < rehash_offset
    )
    assert (
        rehash_offset
        < cgroup_recheck_offset
        < gpu_recheck_offset
        < second_gpu_identity_offset
        < second_peer_topology_offset
        < second_peer_binding_offset
        < peer_unchanged_offset
        < server_offset
    )
    assert '"gpu_topology": gpu_topology,' in smoke_source
    host_start = source.index("def _host_identity(")
    host_end = source.index("\ndef _png_chunk(", host_start)
    assert "SmokeHostEvidence.model_validate(host)" in source[host_start:host_end]


def test_post_rehash_gpu_recheck_requires_stable_cuda_uuid_identity() -> None:
    namespace = _isolated_runner_functions("_require_gpu_inventory_unchanged")
    recheck = namespace["_require_gpu_inventory_unchanged"]
    assert callable(recheck)
    base = {
        "identity_protocol": "cuda-driver-uuid+nvidia-smi-uuid-v1",
        "cuda_driver_api_version": 13_010,
        "cuda_ordinal": 0,
        "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "cuda_driver_name": "NVIDIA B300",
        "nvidia_smi_name": "NVIDIA B300",
        "cuda_compute_capability": "10.3",
        "nvidia_smi_compute_capability": "10.3",
        "cuda_total_memory_bytes": 200 * 1024**3,
        "nvidia_smi_memory_total_mib": 191_456,
        "driver_version": "590.44.01",
        "pci_bus_id": None,
        "pci_bus_id_status": "unavailable",
        "pci_bus_id_error_code": 999,
    }
    second = {
        **base,
        "cuda_ordinal": 1,
        "uuid": "GPU-ffffffff-1111-2222-3333-444444444444",
    }
    expected = (base, second)
    pci_now_available = {
        **base,
        "pci_bus_id": "00000000:45:00.0",
        "pci_bus_id_status": "available",
        "pci_bus_id_error_code": None,
    }

    recheck(expected, (pci_now_available, second))

    with pytest.raises(RuntimeError, match="GPU UUID inventory changed before server start"):
        recheck(
            expected,
            (
                {
                    **pci_now_available,
                    "uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
                },
                second,
            ),
        )


def _gpu_hardware_fixture() -> tuple[dict[str, object], dict[str, object]]:
    first = {
        "cuda_driver_api_version": 13_010,
        "cuda_ordinal": 0,
        "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    }
    second = {
        "cuda_driver_api_version": 13_010,
        "cuda_ordinal": 1,
        "uuid": "GPU-ffffffff-1111-2222-3333-444444444444",
    }
    return first, second


def _cuda_peer_topology_fixture() -> dict[str, object]:
    first, second = _gpu_hardware_fixture()
    return {
        "topology_protocol": "cuda-driver-p2p-attributes-v1",
        "cuda_driver_api_version": 13_010,
        "links": [
            {
                "source_cuda_ordinal": first["cuda_ordinal"],
                "source_uuid": first["uuid"],
                "destination_cuda_ordinal": second["cuda_ordinal"],
                "destination_uuid": second["uuid"],
                "can_access_peer": True,
                "performance_rank": 0,
                "access_supported": True,
                "native_atomic_supported": True,
                "cuda_array_access_supported": True,
                "only_partial_native_atomic_supported": False,
            },
            {
                "source_cuda_ordinal": second["cuda_ordinal"],
                "source_uuid": second["uuid"],
                "destination_cuda_ordinal": first["cuda_ordinal"],
                "destination_uuid": first["uuid"],
                "can_access_peer": True,
                "performance_rank": 0,
                "access_supported": True,
                "native_atomic_supported": True,
                "cuda_array_access_supported": True,
                "only_partial_native_atomic_supported": False,
            },
        ],
    }


def _nvidia_topology_diagnostic_fixture() -> dict[str, object]:
    empty_sha256 = hashlib.sha256(b"").hexdigest()
    return {
        "schema_version": "inkling-smoke-nvidia-smi-topology-diagnostic-v1",
        "argv": ["nvidia-smi", "topo", "-m"],
        "status": "command_failed",
        "return_code": 255,
        "stdout_size_bytes": 0,
        "stdout_sha256": empty_sha256,
        "stderr_size_bytes": 0,
        "stderr_sha256": empty_sha256,
        "stdout_recorded": False,
        "stderr_recorded": False,
    }


def _nvidia_topology_namespace(run: Any) -> dict[str, object]:
    namespace = _isolated_runner_functions(
        "_captured_subprocess_bytes",
        "_nvidia_topology_diagnostic",
    )
    namespace.update(
        {
            "SmokeNvidiaSmiTopologyDiagnostic": SmokeNvidiaSmiTopologyDiagnostic,
            "hashlib": hashlib,
            "subprocess": SimpleNamespace(
                TimeoutExpired=subprocess.TimeoutExpired,
                run=run,
            ),
        }
    )
    return namespace


@pytest.mark.parametrize(
    ("return_code", "stdout", "stderr", "expected_status"),
    (
        (0, b"GPU0 GPU1\\n", b"", "available"),
        (255, b"", b"topology unavailable\\n", "command_failed"),
    ),
)
def test_nvidia_topology_diagnostic_hashes_exact_bytes_without_recording_them(
    return_code: int,
    stdout: bytes,
    stderr: bytes,
    expected_status: str,
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def run(argv: object, **kwargs: object) -> SimpleNamespace:
        calls.append((argv, kwargs))
        return SimpleNamespace(
            returncode=return_code,
            stdout=stdout,
            stderr=stderr,
        )

    namespace = _nvidia_topology_namespace(run)
    diagnostic = namespace["_nvidia_topology_diagnostic"]
    assert callable(diagnostic)

    observed = diagnostic()

    assert calls == [
        (
            ["nvidia-smi", "topo", "-m"],
            {
                "check": False,
                "capture_output": True,
                "timeout": 60,
                "shell": False,
            },
        )
    ]
    assert observed == {
        "schema_version": "inkling-smoke-nvidia-smi-topology-diagnostic-v1",
        "argv": ["nvidia-smi", "topo", "-m"],
        "status": expected_status,
        "return_code": return_code,
        "stdout_size_bytes": len(stdout),
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_size_bytes": len(stderr),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stdout_recorded": False,
        "stderr_recorded": False,
    }
    assert "stdout" not in observed
    assert "stderr" not in observed


def test_nvidia_topology_diagnostic_records_blank_success_as_nonfatal_evidence() -> None:
    stdout = b" \n\t"
    namespace = _nvidia_topology_namespace(
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=stdout,
            stderr=b"",
        )
    )
    diagnostic = namespace["_nvidia_topology_diagnostic"]
    assert callable(diagnostic)

    observed = diagnostic()

    assert observed["status"] == "empty_output"
    assert observed["return_code"] == 0
    assert observed["stdout_size_bytes"] == len(stdout)
    assert observed["stdout_sha256"] == hashlib.sha256(stdout).hexdigest()


def test_nvidia_topology_diagnostic_records_timeout_as_nonfatal_evidence() -> None:
    stdout = b"partial topology"
    stderr = b"deadline exceeded"

    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(
            cmd=["nvidia-smi", "topo", "-m"],
            timeout=60,
            output=stdout,
            stderr=stderr,
        )

    namespace = _nvidia_topology_namespace(timeout)
    diagnostic = namespace["_nvidia_topology_diagnostic"]
    assert callable(diagnostic)

    observed = diagnostic()

    assert observed["status"] == "timed_out"
    assert observed["return_code"] is None
    assert observed["stdout_size_bytes"] == len(stdout)
    assert observed["stdout_sha256"] == hashlib.sha256(stdout).hexdigest()
    assert observed["stderr_size_bytes"] == len(stderr)
    assert observed["stderr_sha256"] == hashlib.sha256(stderr).hexdigest()


def test_nvidia_topology_diagnostic_records_os_error_as_nonfatal_evidence() -> None:
    def unavailable(*_args: object, **_kwargs: object) -> None:
        raise OSError("nvidia-smi is unavailable")

    namespace = _nvidia_topology_namespace(unavailable)
    diagnostic = namespace["_nvidia_topology_diagnostic"]
    assert callable(diagnostic)

    observed = diagnostic()

    assert observed["status"] == "unavailable"
    assert observed["return_code"] is None
    assert observed["stdout_size_bytes"] == 0
    assert observed["stdout_sha256"] == hashlib.sha256(b"").hexdigest()
    assert observed["stderr_size_bytes"] == 0
    assert observed["stderr_sha256"] == hashlib.sha256(b"").hexdigest()


def test_nvidia_topology_diagnostic_records_invalid_timeout_streams() -> None:
    def invalid_stream(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(
            cmd=["nvidia-smi", "topo", "-m"],
            timeout=60,
            output=object(),
            stderr=b"",
        )

    namespace = _nvidia_topology_namespace(invalid_stream)
    diagnostic = namespace["_nvidia_topology_diagnostic"]
    assert callable(diagnostic)

    observed = diagnostic()

    assert observed["status"] == "invalid_result"
    assert observed["return_code"] is None
    assert observed["stdout_size_bytes"] == 0
    assert observed["stderr_size_bytes"] == 0


def test_failure_server_log_hashes_oversized_content_and_cross_chunk_signals(
    tmp_path: Path,
) -> None:
    chunk_size = 1024 * 1024
    signal = b"out of memory"
    payload = b"x" * (chunk_size - 4) + signal + b"y" * (chunk_size + 17)
    server_log = tmp_path / "llama-server.log"
    server_log.write_bytes(payload)
    namespace = _isolated_runner_functions("_failure_server_log_evidence")
    namespace.update(
        {
            "MAX_SERVER_LOG_BYTES": 128,
            "MAX_TERMINAL_RECEIPT_BYTES": 128,
            "SERVER_LOG": server_log,
            "hashlib": hashlib,
        }
    )
    evidence = namespace["_failure_server_log_evidence"]
    assert callable(evidence)
    assert len(payload) > 128

    digest, signals = evidence()

    assert digest == hashlib.sha256(payload).hexdigest()
    assert signals == {
        "out_of_memory_observed": True,
        "no_usable_gpu_observed": False,
        "model_load_failure_observed": False,
        "projector_load_failure_observed": False,
        "unsupported_architecture_observed": False,
    }


def test_gpu_topology_identity_preserves_the_exact_ordered_peer_links() -> None:
    namespace = _isolated_runner_functions("_gpu_topology_identity")
    namespace["SmokeGpuTopologyEvidence"] = SmokeGpuTopologyEvidence
    identity = namespace["_gpu_topology_identity"]
    assert callable(identity)
    peer_topology = _cuda_peer_topology_fixture()
    diagnostic = _nvidia_topology_diagnostic_fixture()

    observed = identity(peer_topology, diagnostic)

    assert observed["schema_version"] == "inkling-smoke-gpu-topology-v1"
    assert observed["protocol"] == ("cuda-driver-p2p-v1+nvidia-smi-topo-diagnostic-v1")
    assert observed["cuda_driver_api_version"] == 13_010
    assert observed["edges"] == peer_topology["links"]
    assert observed["nvidia_smi_topology"] == diagnostic


def test_peer_topology_must_match_the_joined_hardware_identity() -> None:
    namespace = _isolated_runner_functions("_require_peer_topology_matches_hardware")
    require_match = namespace["_require_peer_topology_matches_hardware"]
    assert callable(require_match)
    hardware = _gpu_hardware_fixture()
    topology = _cuda_peer_topology_fixture()

    require_match(hardware, topology)

    links = topology["links"]
    assert isinstance(links, list)
    links[0] = {
        **links[0],
        "destination_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
    }
    with pytest.raises(
        RuntimeError,
        match="differs from the joined GPU identity",
    ):
        require_match(hardware, topology)


def test_peer_topology_recheck_rejects_any_post_rehash_drift() -> None:
    namespace = _isolated_runner_functions("_require_peer_topology_unchanged")
    require_unchanged = namespace["_require_peer_topology_unchanged"]
    assert callable(require_unchanged)
    expected = _cuda_peer_topology_fixture()
    observed = json.loads(json.dumps(expected))
    links = observed["links"]
    assert isinstance(links, list)
    links[0] = {
        **links[0],
        "performance_rank": 1,
    }

    with pytest.raises(RuntimeError, match="changed before server start"):
        require_unchanged(expected, observed)


def _safe_subprocess_failure_namespace() -> dict[str, object]:
    namespace = _isolated_runner_functions(
        "_failed_subprocess_command_id",
        "_captured_subprocess_bytes",
        "_safe_subprocess_failure",
    )
    namespace.update(
        {
            "LLAMA_CPP_DIR": Path("/opt/llama.cpp"),
            "SOURCE_BLOB_PINS": (
                ("src/first.cpp", "a" * 40),
                ("src/second.cpp", "b" * 40),
            ),
            "SmokeSubprocessFailureEvidence": SmokeSubprocessFailureEvidence,
            "hashlib": hashlib,
            "subprocess": subprocess,
            "sys": sys,
            "sysconfig": sysconfig,
        }
    )
    return namespace


def test_safe_subprocess_failure_maps_every_allowlisted_preflight_command() -> None:
    namespace = _safe_subprocess_failure_namespace()
    command_id = namespace["_failed_subprocess_command_id"]
    assert callable(command_id)
    purelib = sysconfig.get_path("purelib") or ""
    llama_cpp_dir = Path("/opt/llama.cpp")
    patched_paths = ("src/first.cpp", "src/second.cpp")
    commands = {
        (
            sys.executable,
            "-m",
            "pip",
            "freeze",
            "--all",
            "--path",
            purelib,
        ): "python_package_inventory_v1",
        (
            "git",
            "-C",
            str(llama_cpp_dir),
            "diff",
            "--name-only",
        ): "llama_cpp_git_changed_paths_v1",
        (
            "git",
            "-C",
            str(llama_cpp_dir),
            "diff",
            "--binary",
            "--",
            *patched_paths,
        ): "llama_cpp_git_patched_diff_v1",
        (
            "dpkg-query",
            "-W",
            "-f=${binary:Package}=${Version}\\n",
        ): "dpkg_inventory_v1",
        (
            "ldd",
            str(llama_cpp_dir / "build/bin/libggml-cuda.so"),
        ): "cuda_driver_linkage_v1",
        ("nvcc", "--version"): "cuda_compiler_version_v1",
        (
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ): "nvidia_smi_identity_v1",
    }

    assert {command: command_id(command) for command in commands} == commands


@pytest.mark.parametrize(
    ("stdout", "stderr"),
    (
        ("private stdout", b"private stderr"),
        (b"private stdout", "private stderr"),
        (None, None),
    ),
)
def test_safe_subprocess_failure_hashes_supported_captures_without_raw_data(
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> None:
    namespace = _safe_subprocess_failure_namespace()
    safe_failure = namespace["_safe_subprocess_failure"]
    assert callable(safe_failure)
    command = ("nvcc", "--version")
    error = subprocess.CalledProcessError(
        -9,
        command,
        output=stdout,
        stderr=stderr,
    )

    observed = safe_failure(error)

    stdout_bytes = (
        b"" if stdout is None else (stdout if isinstance(stdout, bytes) else stdout.encode("utf-8"))
    )
    stderr_bytes = (
        b"" if stderr is None else (stderr if isinstance(stderr, bytes) else stderr.encode("utf-8"))
    )
    assert observed == {
        "schema_version": "inkling-smoke-subprocess-failure-v1",
        "command_id": "cuda_compiler_version_v1",
        "return_code": -9,
        "stdout_size_bytes": len(stdout_bytes),
        "stdout_sha256": hashlib.sha256(stdout_bytes).hexdigest(),
        "stderr_size_bytes": len(stderr_bytes),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
        "stdout_recorded": False,
        "stderr_recorded": False,
    }
    serialized = json.dumps(observed, sort_keys=True)
    assert "private stdout" not in serialized
    assert "private stderr" not in serialized
    assert "argv" not in serialized
    assert "environment" not in serialized


@pytest.mark.parametrize(
    ("error", "message"),
    (
        (
            subprocess.CalledProcessError(1, ("unknown", "command")),
            "not allowlisted",
        ),
        (
            subprocess.CalledProcessError(0, ("nvcc", "--version")),
            "invalid return code",
        ),
        (
            subprocess.CalledProcessError(True, ("nvcc", "--version")),
            "invalid return code",
        ),
        (
            subprocess.CalledProcessError(1, "nvcc --version"),
            "not a string argument sequence",
        ),
        (
            subprocess.CalledProcessError(1, ("nvcc", 1)),
            "not a string argument sequence",
        ),
        (
            subprocess.CalledProcessError(
                1,
                ("nvcc", "--version"),
                output=object(),
            ),
            "stdout has an unsupported type",
        ),
        (
            subprocess.CalledProcessError(
                1,
                ("nvcc", "--version"),
                stderr=object(),
            ),
            "stderr has an unsupported type",
        ),
    ),
)
def test_safe_subprocess_failure_rejects_untrusted_failure_shapes(
    error: subprocess.CalledProcessError,
    message: str,
) -> None:
    namespace = _safe_subprocess_failure_namespace()
    safe_failure = namespace["_safe_subprocess_failure"]
    assert callable(safe_failure)

    with pytest.raises(RuntimeError, match=message):
        safe_failure(error)


def test_record_failure_uses_terminal_v4_safe_subprocess_evidence() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    start = source.index("def _record_failure(")
    end = source.index("\n\n@app.function(", start)
    function_source = source[start:end]

    assert '"schema_version": "inkling-smoke-terminal-v4"' in function_source
    assert '"safe_subprocess_failure": _safe_subprocess_failure(error)' in function_source
    assert '"subprocess_failure":' not in function_source


def test_cgroup_inventory_recheck_rejects_post_rehash_drift() -> None:
    namespace = _isolated_runner_functions("_require_cgroup_inventory_unchanged")
    recheck = namespace["_require_cgroup_inventory_unchanged"]
    assert callable(recheck)
    namespace["_cgroup_limit_inventory"] = lambda: {
        "cgroup_cpu_quota_millicores": 8_000,
        "cgroup_memory_limit_bytes": 64 * 1024**3,
    }

    with pytest.raises(RuntimeError, match="changed before server start"):
        recheck(
            {
                "cgroup_cpu_quota_millicores": 16_000,
                "cgroup_memory_limit_bytes": 64 * 1024**3,
            }
        )


def test_cgroup_hierarchy_uses_the_strictest_visible_ancestor(tmp_path: Path) -> None:
    namespace = _isolated_runner_functions(
        "_cgroup_control_file_exists",
        "_read_cgroup_control",
        "_cgroup_cpu_quota",
        "_cgroup_memory_limit",
    )
    cpu_quota = namespace["_cgroup_cpu_quota"]
    memory_limit = namespace["_cgroup_memory_limit"]
    assert callable(cpu_quota)
    assert callable(memory_limit)

    root = tmp_path / "cgroup"
    parent = root / "parent"
    leaf = parent / "leaf"
    leaf.mkdir(parents=True)
    (root / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    (parent / "cpu.max").write_text("800000 100000\n", encoding="utf-8")
    (leaf / "cpu.max").write_text("1600000 100000\n", encoding="utf-8")
    (root / "memory.max").write_text(str(128 * 1024**3), encoding="utf-8")
    (parent / "memory.max").write_text(str(64 * 1024**3), encoding="utf-8")
    (leaf / "memory.max").write_text("max\n", encoding="utf-8")

    cpu = cpu_quota(leaf, hierarchy_root=root)
    memory = memory_limit(leaf, hierarchy_root=root)

    assert cpu[0] == 8_000
    assert cpu[2] == tuple(str(path / "cpu.max") for path in (leaf, parent, root))
    assert cpu[3] == (16_000, 8_000, None)
    assert memory[0] == 64 * 1024**3
    assert memory[2] == tuple(str(path / "memory.max") for path in (leaf, parent, root))
    assert memory[3] == (None, 64 * 1024**3, 128 * 1024**3)


def test_cgroup_hierarchy_stops_at_the_explicit_mount_boundary(tmp_path: Path) -> None:
    namespace = _isolated_runner_functions(
        "_cgroup_control_file_exists",
        "_read_cgroup_control",
        "_cgroup_cpu_quota",
        "_cgroup_memory_limit",
    )
    cpu_quota = namespace["_cgroup_cpu_quota"]
    memory_limit = namespace["_cgroup_memory_limit"]
    assert callable(cpu_quota)
    assert callable(memory_limit)

    hierarchy_root = tmp_path / "mounted-cgroup"
    parent = hierarchy_root / "parent"
    leaf = parent / "leaf"
    leaf.mkdir(parents=True)

    # These files are outside the mounted hierarchy. The walkers must not use them.
    (tmp_path / "cpu.max").write_text("400000 100000\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text(str(32 * 1024**3), encoding="utf-8")
    (hierarchy_root / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    (parent / "cpu.max").write_text("800000 100000\n", encoding="utf-8")
    (leaf / "cpu.max").write_text("1600000 100000\n", encoding="utf-8")
    (hierarchy_root / "memory.max").write_text("max\n", encoding="utf-8")
    (parent / "memory.max").write_text(str(64 * 1024**3), encoding="utf-8")
    (leaf / "memory.max").write_text(str(128 * 1024**3), encoding="utf-8")

    cpu = cpu_quota(leaf, hierarchy_root=hierarchy_root)
    memory = memory_limit(leaf, hierarchy_root=hierarchy_root)

    assert cpu[0] == 8_000
    assert cpu[2] == tuple(str(path / "cpu.max") for path in (leaf, parent, hierarchy_root))
    assert memory[0] == 64 * 1024**3
    assert memory[2] == tuple(str(path / "memory.max") for path in (leaf, parent, hierarchy_root))


def test_v2_root_without_resource_files_is_an_unlimited_boundary(tmp_path: Path) -> None:
    namespace = _isolated_runner_functions(
        "_cgroup_control_file_exists",
        "_read_cgroup_control",
        "_cgroup_cpu_quota",
        "_cgroup_memory_limit",
    )
    cpu_quota = namespace["_cgroup_cpu_quota"]
    memory_limit = namespace["_cgroup_memory_limit"]
    assert callable(cpu_quota)
    assert callable(memory_limit)

    hierarchy_root = tmp_path / "root-cgroup"
    parent = hierarchy_root / "parent"
    leaf = parent / "leaf"
    leaf.mkdir(parents=True)
    (parent / "cpu.max").write_text("1600000 100000\n", encoding="utf-8")
    (leaf / "cpu.max").write_text("max 100000\n", encoding="utf-8")
    (parent / "memory.max").write_text(str(64 * 1024**3), encoding="utf-8")
    (leaf / "memory.max").write_text("max\n", encoding="utf-8")

    cpu = cpu_quota(leaf, hierarchy_root=hierarchy_root)
    memory = memory_limit(leaf, hierarchy_root=hierarchy_root)

    assert cpu[0] == 16_000
    assert cpu[2] == tuple(str(path / "cpu.max") for path in (leaf, parent))
    assert cpu[3] == (None, 16_000)
    assert memory[0] == 64 * 1024**3
    assert memory[2] == tuple(str(path / "memory.max") for path in (leaf, parent))
    assert memory[3] == (None, 64 * 1024**3)


def test_v1_cpu_hierarchy_uses_strictest_ancestor_and_two_paths_per_level(
    tmp_path: Path,
) -> None:
    namespace = _isolated_runner_functions(
        "_cgroup_control_file_exists",
        "_read_cgroup_control",
        "_cgroup_cpu_quota",
    )
    cpu_quota = namespace["_cgroup_cpu_quota"]
    assert callable(cpu_quota)

    hierarchy_root = tmp_path / "cpu-controller"
    parent = hierarchy_root / "parent"
    leaf = parent / "leaf"
    leaf.mkdir(parents=True)
    levels = (
        (leaf, "1600000", "100000"),
        (parent, "800000", "100000"),
        (hierarchy_root, "-1", "100000"),
    )
    for level, quota, period in levels:
        (level / "cpu.cfs_quota_us").write_text(quota, encoding="utf-8")
        (level / "cpu.cfs_period_us").write_text(period, encoding="utf-8")

    observed = cpu_quota(leaf, hierarchy_root=hierarchy_root)

    assert observed[0] == 8_000
    assert observed[1] == "cgroup_v1_visible_hierarchy_cpu.cfs_quota_us"
    assert observed[2] == tuple(
        str(level / filename)
        for level, _quota, _period in levels
        for filename in ("cpu.cfs_quota_us", "cpu.cfs_period_us")
    )
    assert observed[3] == (16_000, 8_000, None)


def test_v1_cpu_hierarchy_rejects_an_incomplete_ancestor_pair(tmp_path: Path) -> None:
    namespace = _isolated_runner_functions(
        "_cgroup_control_file_exists",
        "_read_cgroup_control",
        "_cgroup_cpu_quota",
    )
    cpu_quota = namespace["_cgroup_cpu_quota"]
    assert callable(cpu_quota)

    hierarchy_root = tmp_path / "cpu-controller"
    parent = hierarchy_root / "parent"
    leaf = parent / "leaf"
    leaf.mkdir(parents=True)
    (leaf / "cpu.cfs_quota_us").write_text("1600000", encoding="utf-8")
    (leaf / "cpu.cfs_period_us").write_text("100000", encoding="utf-8")
    (parent / "cpu.cfs_quota_us").write_text("800000", encoding="utf-8")

    with pytest.raises(RuntimeError, match="v1 CPU quota inventory is incomplete"):
        cpu_quota(leaf, hierarchy_root=hierarchy_root)


@pytest.mark.parametrize("version", (1, 2), ids=("v1", "v2"))
def test_cgroup_helpers_reject_all_unlimited_hierarchies(
    tmp_path: Path,
    version: int,
) -> None:
    namespace = _isolated_runner_functions(
        "_cgroup_control_file_exists",
        "_read_cgroup_control",
        "_cgroup_cpu_quota",
        "_cgroup_memory_limit",
    )
    cpu_quota = namespace["_cgroup_cpu_quota"]
    memory_limit = namespace["_cgroup_memory_limit"]
    assert callable(cpu_quota)
    assert callable(memory_limit)

    hierarchy_root = tmp_path / f"cgroup-v{version}"
    leaf = hierarchy_root / "leaf"
    leaf.mkdir(parents=True)
    for level in (leaf, hierarchy_root):
        if version == 2:
            (level / "cpu.max").write_text("max 100000\n", encoding="utf-8")
            (level / "memory.max").write_text("max\n", encoding="utf-8")
        else:
            (level / "cpu.cfs_quota_us").write_text("-1\n", encoding="utf-8")
            (level / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")
            (level / "memory.limit_in_bytes").write_text(
                "9223372036854771712\n",
                encoding="utf-8",
            )

    with pytest.raises(RuntimeError, match="CPU quota must be finite"):
        cpu_quota(leaf, hierarchy_root=hierarchy_root)
    with pytest.raises(RuntimeError, match="memory limit must be finite"):
        memory_limit(leaf, hierarchy_root=hierarchy_root)


class _SequencedProcPath:
    def __init__(
        self,
        value: str | Path,
        *,
        payloads: dict[str, list[bytes]],
        offsets: dict[str, int],
    ) -> None:
        self._value = str(value)
        self._payloads = payloads
        self._offsets = offsets

    def read_bytes(self) -> bytes:
        values = self._payloads[self._value]
        offset = self._offsets.get(self._value, 0)
        self._offsets[self._value] = offset + 1
        return values[min(offset, len(values) - 1)]


def _isolated_cgroup_inventory(
    *,
    membership_payloads: list[bytes] | None = None,
    mountinfo_payloads: list[bytes] | None = None,
) -> tuple[dict[str, object], Path, Path]:
    namespace = _isolated_runner_functions("_cgroup_limit_inventory")
    payloads = {
        "/proc/self/cgroup": membership_payloads or [b"0::/tenant/leaf\n"],
        "/proc/self/mountinfo": mountinfo_payloads
        or [b"42 29 0:31 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n"],
    }
    offsets: dict[str, int] = {}
    namespace["Path"] = lambda value: _SequencedProcPath(
        value,
        payloads=payloads,
        offsets=offsets,
    )
    cpu_leaf = Path("/test/cpu/leaf")
    memory_leaf = Path("/test/memory/leaf")
    namespace["resolve_current_process_cgroup_hierarchy_paths"] = lambda **_kwargs: {
        "cpu": (cpu_leaf, cpu_leaf.parent),
        "memory": (memory_leaf, memory_leaf.parent),
    }
    namespace["os"] = SimpleNamespace(getpid=lambda: 4321)
    namespace["_cgroup_leaf_pid_identity"] = lambda _leaf, *, pid: "a" * 64
    namespace["_cgroup_path_sha256"] = lambda _path: "b" * 64
    return namespace, cpu_leaf, memory_leaf


@pytest.mark.parametrize("changed_inventory", ("membership", "mountinfo"))
def test_cgroup_inventory_rejects_proc_inventory_changes(changed_inventory: str) -> None:
    membership = [b"0::/tenant/leaf\n"]
    mountinfo = [b"42 29 0:31 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n"]
    if changed_inventory == "membership":
        membership.append(b"0::/tenant/different-leaf\n")
    else:
        mountinfo.append(b"43 29 0:32 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n")
    namespace, _cpu_leaf, _memory_leaf = _isolated_cgroup_inventory(
        membership_payloads=membership,
        mountinfo_payloads=mountinfo,
    )
    namespace["_cgroup_cpu_quota"] = lambda *_args, **_kwargs: (
        16_000,
        "cgroup_v2_visible_hierarchy_cpu.max",
        ("/test/cpu/leaf/cpu.max",),
        (16_000,),
    )
    namespace["_cgroup_memory_limit"] = lambda *_args, **_kwargs: (
        64 * 1024**3,
        "cgroup_v2_visible_hierarchy_memory.max",
        ("/test/memory/leaf/memory.max",),
        (64 * 1024**3,),
    )
    inventory = namespace["_cgroup_limit_inventory"]
    assert callable(inventory)

    with pytest.raises(RuntimeError, match="membership changed during preflight"):
        inventory()


@pytest.mark.parametrize("changed_control", ("cpu", "memory"))
def test_cgroup_inventory_rejects_control_file_changes(changed_control: str) -> None:
    namespace, _cpu_leaf, _memory_leaf = _isolated_cgroup_inventory()
    stable_cpu = (
        16_000,
        "cgroup_v2_visible_hierarchy_cpu.max",
        ("/test/cpu/leaf/cpu.max",),
        (16_000,),
    )
    stable_memory = (
        64 * 1024**3,
        "cgroup_v2_visible_hierarchy_memory.max",
        ("/test/memory/leaf/memory.max",),
        (64 * 1024**3,),
    )
    cpu = [stable_cpu, stable_cpu]
    memory = [stable_memory, stable_memory]
    if changed_control == "cpu":
        cpu[1] = (
            8_000,
            "cgroup_v2_visible_hierarchy_cpu.max",
            ("/test/cpu/leaf/cpu.max",),
            (8_000,),
        )
    else:
        memory[1] = (
            32 * 1024**3,
            "cgroup_v2_visible_hierarchy_memory.max",
            ("/test/memory/leaf/memory.max",),
            (32 * 1024**3,),
        )
    cpu_offsets = iter(cpu)
    memory_offsets = iter(memory)
    namespace["_cgroup_cpu_quota"] = lambda *_args, **_kwargs: next(cpu_offsets)
    namespace["_cgroup_memory_limit"] = lambda *_args, **_kwargs: next(memory_offsets)
    inventory = namespace["_cgroup_limit_inventory"]
    assert callable(inventory)

    expected_label = "CPU quota" if changed_control == "cpu" else "memory limit"
    with pytest.raises(RuntimeError, match=rf"cgroup {expected_label} changed"):
        inventory()


def test_cgroup_inventory_rejects_pid_disappearance_before_final_proof() -> None:
    namespace, _cpu_leaf, _memory_leaf = _isolated_cgroup_inventory()
    namespace["_cgroup_cpu_quota"] = lambda *_args, **_kwargs: (
        16_000,
        "cgroup_v2_visible_hierarchy_cpu.max",
        ("/test/cpu/leaf/cpu.max",),
        (16_000,),
    )
    namespace["_cgroup_memory_limit"] = lambda *_args, **_kwargs: (
        64 * 1024**3,
        "cgroup_v2_visible_hierarchy_memory.max",
        ("/test/memory/leaf/memory.max",),
        (64 * 1024**3,),
    )
    calls = 0

    def pid_identity(_leaf: Path, *, pid: int) -> str:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("resolved cgroup leaf does not contain the current process")
        assert pid == 4321
        return "a" * 64

    namespace["_cgroup_leaf_pid_identity"] = pid_identity
    inventory = namespace["_cgroup_limit_inventory"]
    assert callable(inventory)

    with pytest.raises(RuntimeError, match="does not contain the current process"):
        inventory()


def test_evidence_parent_accepts_modal_style_mount_symlink(tmp_path: Path) -> None:
    namespace = _isolated_runner_functions(
        "_resolved_evidence_mount",
        "_create_safe_evidence_parent",
    )
    backing = tmp_path / "backing"
    backing.mkdir()
    mount = tmp_path / "evidence"
    mount.symlink_to(backing, target_is_directory=True)
    namespace["EVIDENCE_MOUNT"] = mount
    create_parent = namespace["_create_safe_evidence_parent"]
    assert callable(create_parent)

    destination = mount / "runs" / "run-id" / "receipt.json"
    create_parent(destination)

    assert destination.parent.resolve() == (backing / "runs" / "run-id").resolve()
    assert destination.parent.is_dir()


def test_evidence_parent_rejects_symlink_below_mount(tmp_path: Path) -> None:
    namespace = _isolated_runner_functions(
        "_resolved_evidence_mount",
        "_create_safe_evidence_parent",
    )
    backing = tmp_path / "backing"
    backing.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (backing / "runs").symlink_to(outside, target_is_directory=True)
    mount = tmp_path / "evidence"
    mount.symlink_to(backing, target_is_directory=True)
    namespace["EVIDENCE_MOUNT"] = mount
    create_parent = namespace["_create_safe_evidence_parent"]
    assert callable(create_parent)

    with pytest.raises(RuntimeError, match="symlink or non-directory ancestor"):
        create_parent(mount / "runs" / "run-id" / "receipt.json")


def _reloaded_volume_reader(tmp_path: Path, *, max_bytes: int = 64) -> Any:
    namespace = _isolated_runner_functions(
        "_safe_child",
        "_read_existing_regular_bytes",
        "_read_reloaded_volume_bytes",
    )
    namespace.update(
        {
            "EVIDENCE_MOUNT": tmp_path,
            "MAX_TERMINAL_RECEIPT_BYTES": max_bytes,
        }
    )
    return namespace["_read_reloaded_volume_bytes"]


def test_reloaded_volume_reader_accepts_exact_regular_bytes(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "run-id" / "receipt.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"exact\n")
    reader = _reloaded_volume_reader(tmp_path)

    assert reader("runs/run-id/receipt.json") == b"exact\n"
    assert reader("runs/run-id/missing.json") is None


@pytest.mark.parametrize(
    "remote_path",
    (
        "/absolute.json",
        "../outside.json",
        "runs//receipt.json",
        "runs/./receipt.json",
        "runs\\receipt.json",
        "runs/receipt.json/",
    ),
)
def test_reloaded_volume_reader_rejects_noncanonical_paths(
    tmp_path: Path,
    remote_path: str,
) -> None:
    reader = _reloaded_volume_reader(tmp_path)

    with pytest.raises(RuntimeError):
        reader(remote_path)


def test_reloaded_volume_reader_rejects_symlink_and_non_regular_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"exact\n")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    directory = tmp_path / "directory"
    directory.mkdir()
    reader = _reloaded_volume_reader(tmp_path)

    with pytest.raises(OSError):
        reader("link.json")
    with pytest.raises(RuntimeError, match="not a regular file"):
        reader("directory")


def test_reloaded_volume_reader_rejects_oversized_file(tmp_path: Path) -> None:
    (tmp_path / "receipt.json").write_bytes(b"12345")
    reader = _reloaded_volume_reader(tmp_path, max_bytes=4)

    with pytest.raises(RuntimeError, match="exceeds its size limit"):
        reader("receipt.json")


class _AttemptHardStop(BaseException):
    """Uncatchable process-stop stand-in used at attempt persistence boundaries."""


class _FakeSealedAttemptRegistry:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.put_calls: list[tuple[str, bytes, bool]] = []

    def put(
        self,
        key: Any,
        value: Any,
        *,
        skip_if_exists: bool = False,
    ) -> bool:
        assert isinstance(key, str)
        assert isinstance(value, bytes)
        self.put_calls.append((key, value, skip_if_exists))
        if skip_if_exists and key in self.values:
            return False
        self.values[key] = value
        return True

    def contains(self, key: str) -> bool:
        return key in self.values

    def get(self, key: str) -> bytes:
        return self.values[key]


class _AttemptEvidenceVolume:
    def __init__(self) -> None:
        self.commit_calls = 0
        self.reload_calls = 0

    def commit(self) -> None:
        self.commit_calls += 1

    def reload(self) -> None:
        self.reload_calls += 1


class _DurableAttemptEvidenceVolume:
    def __init__(self, files: Mapping[str, bytes]) -> None:
        self.files = dict(files)


def _canonical_json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _attempt_contract_fixture() -> SimpleNamespace:
    run_id = "inkling-smoke-attempt-contract"
    config_hash = "c" * 64
    control_plane_sha256 = "d" * 64
    reference_sha256 = "e" * 64
    launch_intent_sha256 = "a" * 64
    deployment = SimpleNamespace(
        app_name=f"inkling-q3-smoke-{control_plane_sha256[:12]}",
        environment_name="inkling-quant",
        deployment_version=1,
        deployment_tag=smoke_deployment_tag(control_plane_sha256),
        function_id="fu-TestFunction123",
        function_name="smoke_test",
        attempt_registry_name="inkling-smoke-attempt-registry-v1",
        attempt_registry_id="di-TestRegistry123",
        attempt_registry_created_at_utc="2026-07-22T12:00:00.000000Z",
    )
    return SimpleNamespace(
        run_id=run_id,
        config_hash=config_hash,
        control_plane_sha256=control_plane_sha256,
        reference_sha256=reference_sha256,
        launch_intent_sha256=launch_intent_sha256,
        call_id="fc-TestCall123",
        input_id="in-TestInput123:0-0",
        task_id="ta-TestTask123",
        config=SimpleNamespace(
            resources=SimpleNamespace(max_attempts=1),
            config_hash=lambda: config_hash,
        ),
        reference=SimpleNamespace(reference_sha256=reference_sha256),
        control_plane=SimpleNamespace(tree_sha256=control_plane_sha256),
        deployment=deployment,
        acknowledgement=SimpleNamespace(deployment=deployment),
    )


def _attempt_contract_acceptance(fixture: SimpleNamespace) -> SmokePostSpawnAcceptance:
    return SmokePostSpawnAcceptance(
        accepted_at="2026-07-22T12:01:00.000000Z",
        run_id=fixture.run_id,
        launch_intent_sha256=fixture.launch_intent_sha256,
        call_id=fixture.call_id,
        app_name=fixture.deployment.app_name,
        environment_name=fixture.deployment.environment_name,
        deployment_version=fixture.deployment.deployment_version,
        deployment_tag=fixture.deployment.deployment_tag,
        function_id=fixture.deployment.function_id,
        function_name=fixture.deployment.function_name,
        attempt_registry_name=fixture.deployment.attempt_registry_name,
        attempt_registry_id=fixture.deployment.attempt_registry_id,
        attempt_registry_created_at_utc=(fixture.deployment.attempt_registry_created_at_utc),
        smoke_config_hash=fixture.config_hash,
        verified_export_reference_sha256=fixture.reference_sha256,
        control_plane_sha256=fixture.control_plane_sha256,
    )


class _AttemptExecutionHarness:
    def __init__(
        self,
        tmp_path: Path,
        *,
        stop_after_claim: bool = False,
        stop_before_volume_write: int | None = None,
    ) -> None:
        self.fixture = _attempt_contract_fixture()
        self.acceptance = _attempt_contract_acceptance(self.fixture)
        self.run_root = tmp_path / "runs" / self.fixture.run_id
        self.registry = _FakeSealedAttemptRegistry()
        self.volume = _AttemptEvidenceVolume()
        self.invocation_state: dict[str, Any] = {}
        self.attempted_writes: list[str] = []
        self.stop_before_volume_write = stop_before_volume_write

        namespace = _isolated_runner_functions(
            "_begin_only_attempt",
            "_commit_and_reconcile_volume_files",
            "_complete_owned_attempt_records",
        )

        def safe_child(root: Path, *parts: str) -> Path:
            path = root.joinpath(*parts)
            assert path.resolve().is_relative_to(root.resolve())
            return path

        def atomic_bytes(
            path: Path,
            payload: bytes,
            *,
            allow_identical: bool,
        ) -> None:
            self.attempted_writes.append(path.relative_to(self.run_root).as_posix())
            if self.stop_before_volume_write == len(self.attempted_writes):
                raise _AttemptHardStop
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if allow_identical and path.read_bytes() == payload:
                    return
                raise RuntimeError("test fixture refused to replace immutable evidence")
            path.write_bytes(payload)

        def atomic_json(
            path: Path,
            value: Mapping[str, Any],
            *,
            allow_identical: bool,
        ) -> None:
            atomic_bytes(
                path,
                (_canonical_json_text(value) + "\n").encode("utf-8"),
                allow_identical=allow_identical,
            )

        claim = claim_smoke_attempt
        if stop_after_claim:

            def claim_then_stop(
                registry: _FakeSealedAttemptRegistry,
                registry_claim: SmokeAttemptRegistryClaim,
            ) -> str:
                claim(registry, registry_claim)
                raise _AttemptHardStop

            attempt_claim = claim_then_stop
        else:
            attempt_claim = claim

        namespace.update(
            {
                "EVIDENCE_MOUNT": tmp_path,
                "SMOKE_STAGE": "smoke_test",
                "SmokeAttemptRegistryClaim": SmokeAttemptRegistryClaim,
                "_DurablePublicationStateUnknownError": (_TestDurablePublicationStateUnknownError),
                "_atomic_bytes": atomic_bytes,
                "_atomic_json": atomic_json,
                "_canonical_json": _canonical_json_text,
                "_read_reloaded_volume_bytes": (
                    lambda remote_path: (
                        (tmp_path / remote_path).read_bytes()
                        if (tmp_path / remote_path).is_file()
                        else None
                    )
                ),
                "_read_existing_regular_bytes": lambda path: path.read_bytes(),
                "_safe_child": safe_child,
                "_sha256": lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
                "claim_smoke_attempt": attempt_claim,
                "evidence_volume": self.volume,
                "hashlib": hashlib,
                "smoke_attempt_registry_key": smoke_attempt_registry_key,
                "sys": sys,
            }
        )
        self._begin: Any = namespace["_begin_only_attempt"]

    def execute(self) -> dict[str, Any]:
        relative_acceptance_path = smoke_post_spawn_acceptance_path(
            self.fixture.run_id,
            self.fixture.launch_intent_sha256,
        ).removeprefix(f"runs/{self.fixture.run_id}/")
        return self._begin(
            self.run_root,
            config=self.fixture.config,
            control_plane=self.fixture.control_plane,
            acknowledgement=self.fixture.acknowledgement,
            attempt_registry=self.registry,
            run_id=self.fixture.run_id,
            launch_intent_sha256=self.fixture.launch_intent_sha256,
            invocation_ids=(
                self.fixture.call_id,
                self.fixture.input_id,
                self.fixture.task_id,
            ),
            post_spawn_acceptance_path=relative_acceptance_path,
            post_spawn_acceptance_sha256=self.acceptance.acceptance_sha256(),
            invocation_state=self.invocation_state,
        )


def _manager_attempt_namespace(
    registry: _FakeSealedAttemptRegistry,
) -> tuple[dict[str, object], list[object]]:
    namespace = _isolated_manager_functions(
        "_read_attempt_registry_claim",
        "_require_fresh_attempt",
    )
    volume_requests: list[object] = []

    def sealed_registry(
        _config: object,
        _deployment: object,
    ) -> _FakeSealedAttemptRegistry:
        return registry

    def evidence_volume(*args: object, **kwargs: object) -> object:
        volume_requests.append((args, kwargs))
        raise AssertionError("live attempt claim must block before a Volume lookup")

    namespace.update(
        {
            "SMOKE_STAGE": "smoke_test",
            "SmokeAttemptRegistryClaim": SmokeAttemptRegistryClaim,
            "_evidence_volume": evidence_volume,
            "_sealed_attempt_registry": sealed_registry,
            "hashlib": hashlib,
            "smoke_attempt_registry_key": smoke_attempt_registry_key,
            "strict_json_object": strict_json_object,
            "validate_smoke_attempt_registry_claim": (validate_smoke_attempt_registry_claim),
        }
    )
    return namespace, volume_requests


@pytest.mark.parametrize(
    ("boundary", "stop_after_claim", "stop_before_volume_write", "persisted_paths"),
    (
        ("immediately-after-dict-claim", True, None, ()),
        ("before-volume-claim", False, 1, ()),
        (
            "before-volume-history",
            False,
            2,
            ("control/smoke_test.attempt.claim.json",),
        ),
        (
            "before-volume-ledger",
            False,
            3,
            (
                "control/smoke_test.attempt.claim.json",
                "control/history",
            ),
        ),
    ),
)
def test_hard_stop_after_dict_claim_still_blocks_every_retry(
    tmp_path: Path,
    boundary: str,
    stop_after_claim: bool,
    stop_before_volume_write: int | None,
    persisted_paths: tuple[str, ...],
) -> None:
    harness = _AttemptExecutionHarness(
        tmp_path / boundary,
        stop_after_claim=stop_after_claim,
        stop_before_volume_write=stop_before_volume_write,
    )

    with pytest.raises(_AttemptHardStop):
        harness.execute()

    key = smoke_attempt_registry_key(harness.fixture.run_id)
    assert harness.registry.contains(key) is True
    live_payload = harness.registry.get(key)
    live_claim = SmokeAttemptRegistryClaim.model_validate(strict_json_object(live_payload))
    assert live_claim.registry_key == key
    assert live_claim.claim_sha256() == hashlib.sha256(live_payload).hexdigest()
    assert bool(harness.invocation_state) is not stop_after_claim

    observed_paths = {
        path.relative_to(harness.run_root).as_posix()
        for path in harness.run_root.rglob("*")
        if path.is_file()
    }
    for persisted_path in persisted_paths:
        if persisted_path == "control/history":
            assert any(path.startswith("control/history/") for path in observed_paths)
        else:
            assert persisted_path in observed_paths

    manager, volume_requests = _manager_attempt_namespace(harness.registry)
    read_claim: Any = manager["_read_attempt_registry_claim"]
    observed = read_claim(
        harness.fixture.config,
        harness.fixture.control_plane,
        harness.fixture.deployment,
        harness.fixture.run_id,
    )
    assert observed is not None
    assert observed[1] == live_payload

    require_fresh: Any = manager["_require_fresh_attempt"]
    with pytest.raises(RuntimeError, match="atomic smoke attempt claim already exists"):
        require_fresh(
            harness.fixture.config,
            harness.fixture.reference,
            harness.fixture.control_plane,
            harness.fixture.deployment,
            harness.fixture.run_id,
        )
    assert volume_requests == []


def _durable_attempt_terminal_harness(tmp_path: Path) -> SimpleNamespace:
    runner = _AttemptExecutionHarness(tmp_path)
    invocation_record = runner.execute()
    invocation = SmokeInvocationEvidence.model_validate(invocation_record)
    assert runner.volume.commit_calls == 2
    assert runner.volume.reload_calls == 2

    owner_key = smoke_attempt_registry_key(runner.fixture.run_id)
    owner_claim_payload = runner.registry.get(owner_key)
    runner.registry.values.clear()

    remote_files = {
        f"runs/{runner.fixture.run_id}/{path.relative_to(runner.run_root).as_posix()}": (
            path.read_bytes()
        )
        for path in runner.run_root.rglob("*")
        if path.is_file()
    }
    acceptance_path = smoke_post_spawn_acceptance_path(
        runner.fixture.run_id,
        runner.fixture.launch_intent_sha256,
    )
    remote_files[acceptance_path] = runner.acceptance.canonical_bytes()
    authorization_path = smoke_launch_intent_remote_path(
        runner.fixture.run_id,
        runner.fixture.launch_intent_sha256,
    )
    authorization_payload = b"exact launch authorization fixture\n"
    remote_files[authorization_path] = authorization_payload
    volume = _DurableAttemptEvidenceVolume(remote_files)

    namespace = _isolated_manager_functions(
        "_canonical_json",
        "_read_attempt_registry_claim",
        "_validate_attempt_registry_claim_for_invocation",
        "_validate_remote_post_spawn_acceptance",
        "_validate_persisted_invocation_records",
        "_validate_terminal_receipt",
    )

    def sealed_registry(
        _config: object,
        _deployment: object,
    ) -> _FakeSealedAttemptRegistry:
        return runner.registry

    def read_volume_bytes(
        observed_volume: _DurableAttemptEvidenceVolume,
        remote_path: str,
    ) -> bytes | None:
        assert observed_volume is volume
        return observed_volume.files.get(remote_path)

    def list_volume(
        observed_volume: _DurableAttemptEvidenceVolume,
        remote_root: str,
    ) -> list[SimpleNamespace]:
        assert observed_volume is volume
        prefix = f"{remote_root.rstrip('/')}/"
        return [
            SimpleNamespace(path=path)
            for path in sorted(observed_volume.files)
            if path.startswith(prefix) and "/" not in path.removeprefix(prefix)
        ]

    terminal_value = {"schema_version": "exact-terminal-fixture-v1"}
    terminal_validations: list[Mapping[str, Any]] = []
    authorization_validations: list[bytes] = []

    def validate_terminal_receipt(
        value: Mapping[str, Any],
        **_kwargs: object,
    ) -> SimpleNamespace:
        assert value == terminal_value
        terminal_validations.append(value)
        return SimpleNamespace(
            launch_intent_sha256=runner.fixture.launch_intent_sha256,
            invocation=invocation,
        )

    def validate_launch_intent(
        payload: bytes,
        **_kwargs: object,
    ) -> object:
        assert payload == authorization_payload
        authorization_validations.append(payload)
        return object()

    namespace.update(
        {
            "SMOKE_STAGE": "smoke_test",
            "SmokeAttemptRegistryClaim": SmokeAttemptRegistryClaim,
            "_list_volume": list_volume,
            "_read_volume_bytes": read_volume_bytes,
            "_sealed_attempt_registry": sealed_registry,
            "hashlib": hashlib,
            "smoke_attempt_registry_key": smoke_attempt_registry_key,
            "smoke_launch_intent_remote_path": smoke_launch_intent_remote_path,
            "smoke_post_spawn_acceptance_path": smoke_post_spawn_acceptance_path,
            "strict_json_object": strict_json_object,
            "validate_smoke_attempt_registry_claim": (validate_smoke_attempt_registry_claim),
            "validate_smoke_launch_intent": validate_launch_intent,
            "validate_smoke_post_spawn_acceptance": (validate_smoke_post_spawn_acceptance),
            "validate_smoke_terminal_receipt": validate_terminal_receipt,
        }
    )
    return SimpleNamespace(
        authorization_validations=authorization_validations,
        fixture=runner.fixture,
        invocation=invocation,
        namespace=namespace,
        owner_claim_payload=owner_claim_payload,
        owner_key=owner_key,
        registry=runner.registry,
        terminal_validations=terminal_validations,
        terminal_value=terminal_value,
        volume=volume,
    )


def _validate_durable_attempt_terminal(harness: SimpleNamespace) -> None:
    validate_terminal: Any = harness.namespace["_validate_terminal_receipt"]
    validate_terminal(
        harness.terminal_value,
        volume=harness.volume,
        config=harness.fixture.config,
        reference=harness.fixture.reference,
        control_plane=harness.fixture.control_plane,
        deployment=harness.fixture.deployment,
        run_id=harness.fixture.run_id,
    )


def test_terminal_validation_uses_exact_durable_records_after_live_dict_expiry(
    tmp_path: Path,
) -> None:
    harness = _durable_attempt_terminal_harness(tmp_path)
    canonical_json: Any = harness.namespace["_canonical_json"]
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}\n'
    history_path = f"runs/{harness.fixture.run_id}/{harness.invocation.invocation_history_path}"
    ledger_path = f"runs/{harness.fixture.run_id}/control/smoke_test.attempts.json"
    for remote_path in (history_path, ledger_path):
        payload = harness.volume.files[remote_path]
        assert payload.endswith(b"\n")
        assert not payload.endswith(b"\n\n")
        assert b"\r" not in payload
    read_claim: Any = harness.namespace["_read_attempt_registry_claim"]
    assert (
        read_claim(
            harness.fixture.config,
            harness.fixture.control_plane,
            harness.fixture.deployment,
            harness.fixture.run_id,
        )
        is None
    )

    _validate_durable_attempt_terminal(harness)

    assert harness.terminal_validations == [harness.terminal_value]
    assert harness.authorization_validations == [b"exact launch authorization fixture\n"]


def test_terminal_validation_rejects_live_dict_bytes_that_differ_from_volume(
    tmp_path: Path,
) -> None:
    harness = _durable_attempt_terminal_harness(tmp_path)
    owner = SmokeAttemptRegistryClaim.model_validate(
        strict_json_object(harness.owner_claim_payload)
    )
    competing = SmokeAttemptRegistryClaim(
        **{
            **owner.model_dump(mode="json"),
            "task_id": "ta-CompetingTask123",
        }
    )
    harness.registry.values[harness.owner_key] = competing.canonical_bytes()

    with pytest.raises(
        RuntimeError,
        match="attempt claim differs from its atomic Dict claim",
    ):
        _validate_durable_attempt_terminal(harness)


@pytest.mark.parametrize(
    ("record_name", "mutator", "message"),
    (
        ("history", lambda value: value.removesuffix(b"\n"), "invocation history differs"),
        ("history", lambda value: value + b"\n", "invocation history differs"),
        (
            "history",
            lambda value: value.removesuffix(b"\n") + b"\r\n",
            "invocation history differs",
        ),
        ("ledger", lambda value: value.removesuffix(b"\n"), "attempt ledger differs"),
        ("ledger", lambda value: value + b"\n", "attempt ledger differs"),
        (
            "ledger",
            lambda value: value.removesuffix(b"\n") + b"\r\n",
            "attempt ledger differs",
        ),
    ),
)
def test_terminal_validation_rejects_noncanonical_record_terminators(
    tmp_path: Path,
    record_name: str,
    mutator: Any,
    message: str,
) -> None:
    harness = _durable_attempt_terminal_harness(tmp_path)
    if record_name == "history":
        remote_path = f"runs/{harness.fixture.run_id}/{harness.invocation.invocation_history_path}"
    else:
        remote_path = f"runs/{harness.fixture.run_id}/control/smoke_test.attempts.json"
    harness.volume.files[remote_path] = mutator(harness.volume.files[remote_path])

    with pytest.raises(RuntimeError, match=message):
        _validate_durable_attempt_terminal(harness)


@pytest.mark.parametrize(
    ("schema_version", "expects_invocation_validation", "expects_safe_diagnostic"),
    (
        ("inkling-smoke-terminal-v2", False, False),
        ("inkling-smoke-terminal-v3", True, False),
        ("inkling-smoke-terminal-v4", True, True),
    ),
)
def test_manager_validates_failure_invocations_and_projects_only_safe_diagnostics(
    schema_version: str,
    expects_invocation_validation: bool,
    expects_safe_diagnostic: bool,
) -> None:
    namespace = _isolated_manager_functions(
        "_safe_subprocess_failure_record",
        "_validated_remote_failure_receipts",
    )
    launch_intent_sha256 = "a" * 64
    receipt_sha256 = "b" * 64
    run_id = "run-id"
    outcome_path = f"runs/{run_id}/control/outcomes/smoke_test.failed.{receipt_sha256}.json"
    authorization_path = f"runs/{run_id}/control/launch-intents/{launch_intent_sha256}.json"
    outcome_payload = json.dumps(
        {"launch_intent_sha256": launch_intent_sha256},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    authorization_payload = b"sealed launch authorization\n"
    files = {
        outcome_path: outcome_payload,
        authorization_path: authorization_payload,
    }
    invocation = object()
    safe_diagnostic = SimpleNamespace(
        schema_version="inkling-smoke-subprocess-failure-v1",
        command_id="nvidia_smi_identity_v1",
        return_code=255,
        stdout_size_bytes=0,
        stdout_sha256=hashlib.sha256(b"").hexdigest(),
        stderr_size_bytes=24,
        stderr_sha256=hashlib.sha256(b"safe diagnostic payload").hexdigest(),
        stdout_recorded=False,
        stderr_recorded=False,
        argv=["nvidia-smi", "--query-gpu"],
        stdout="secret raw stdout",
        stderr="secret raw stderr",
        environment={"TOKEN": "secret"},
    )
    receipt_fields: dict[str, object] = {
        "schema_version": schema_version,
        "receipt_sha256": receipt_sha256,
        "launch_intent_sha256": launch_intent_sha256,
    }
    if schema_version in {
        "inkling-smoke-terminal-v3",
        "inkling-smoke-terminal-v4",
    }:
        receipt_fields["invocation"] = invocation
    if schema_version == "inkling-smoke-terminal-v4":
        receipt_fields["safe_subprocess_failure"] = safe_diagnostic
    receipt = SimpleNamespace(**receipt_fields)
    acceptance_validations: list[object] = []
    invocation_validations: list[object] = []

    def read_volume_bytes(_volume: object, path: str) -> bytes | None:
        return files.get(path)

    def validate_launch_intent(payload: bytes, **_kwargs: object) -> object:
        assert payload == authorization_payload
        return object()

    def validate_failure_receipt(payload: bytes, **_kwargs: object) -> object:
        assert payload == outcome_payload
        return receipt

    def validate_acceptance(**kwargs: object) -> None:
        acceptance_validations.append(kwargs["invocation"])

    def validate_invocation(**kwargs: object) -> None:
        invocation_validations.append(kwargs["invocation"])

    namespace.update(
        {
            "_read_volume_bytes": read_volume_bytes,
            "_validate_persisted_invocation_records": validate_invocation,
            "_validate_remote_post_spawn_acceptance": validate_acceptance,
            "re": re,
            "smoke_launch_intent_remote_path": (lambda _run_id, _sha256: authorization_path),
            "strict_json_object": strict_json_object,
            "validate_smoke_failure_receipt": validate_failure_receipt,
            "validate_smoke_launch_intent": validate_launch_intent,
        }
    )
    validate = namespace["_validated_remote_failure_receipts"]
    assert callable(validate)
    records = validate(
        object(),
        [SimpleNamespace(path=outcome_path)],
        config=object(),
        reference=object(),
        control_plane=object(),
        deployment=object(),
        run_id=run_id,
    )

    expected_validations = [invocation] if expects_invocation_validation else []
    assert acceptance_validations == expected_validations
    assert invocation_validations == expected_validations
    expected_record: dict[str, object] = {
        "path": outcome_path,
        "receipt_sha256": receipt_sha256,
        "launch_intent_sha256": launch_intent_sha256,
        "authorization_path": authorization_path,
    }
    if expects_safe_diagnostic:
        expected_record["safe_subprocess_failure"] = {
            "schema_version": "inkling-smoke-subprocess-failure-v1",
            "command_id": "nvidia_smi_identity_v1",
            "return_code": 255,
            "stdout_size_bytes": 0,
            "stdout_sha256": hashlib.sha256(b"").hexdigest(),
            "stderr_size_bytes": 24,
            "stderr_sha256": hashlib.sha256(b"safe diagnostic payload").hexdigest(),
            "stdout_recorded": False,
            "stderr_recorded": False,
        }
    assert records == [expected_record]
    serialized = json.dumps(records, sort_keys=True)
    assert "secret" not in serialized
    assert "argv" not in serialized


def test_smoke_runner_claims_the_only_attempt_atomically_before_volume_writes() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    begin_start = source.index("def _begin_only_attempt(")
    begin_end = source.index("\ndef _regular_file_identity(", begin_start)
    begin_source = source[begin_start:begin_end]

    assert '"control", "smoke_test.attempt.claim.json"' in begin_source
    assert "registry_key=smoke_attempt_registry_key(run_id)" in begin_source
    claim_offset = begin_source.index("claim_smoke_attempt(")
    state_offset = begin_source.index("invocation_state.update(")
    complete_offset = begin_source.index("_complete_owned_attempt_records(")
    assert claim_offset < state_offset < complete_offset
    assert "_bind_run_inputs(" not in begin_source
    assert "evidence_volume.commit()" not in begin_source[:claim_offset]


def test_attempt_bookkeeping_uses_common_commit_reconciliation() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    complete_start = source.index("def _complete_owned_attempt_records(")
    complete_end = source.index("\ndef _regular_file_identity(", complete_start)
    complete_source = source[complete_start:complete_end]

    assert complete_source.count("_commit_and_reconcile_volume_files(") == 2
    assert 'event="attempt_claim_commit_reconciled"' in complete_source
    assert 'event="attempt_bookkeeping_commit_reconciled"' in complete_source
    assert "evidence_volume.commit()" not in complete_source


def test_smoke_runner_uses_the_sealed_dict_as_the_attempt_authority() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    registry_start = source.index("def _sealed_attempt_registry(")
    registry_end = source.index(
        "\ndef _wait_for_post_spawn_acceptance(",
        registry_start,
    )
    registry_source = source[registry_start:registry_end]
    begin_start = source.index("def _begin_only_attempt(")
    begin_end = source.index("\ndef _regular_file_identity(", begin_start)
    begin_source = source[begin_start:begin_end]

    assert "modal.Dict.from_id(deployment.attempt_registry_id)" in registry_source
    assert "registry.object_id" in registry_source
    assert "info.name" in registry_source
    assert "info.created_at" in registry_source
    assert "claim_smoke_attempt(" in begin_source
    assert "attempt_registry," in begin_source


def test_local_paid_gate_runs_before_modal_objects_are_created() -> None:
    module = _module(RUNNER_PATH)
    source = RUNNER_PATH.read_text(encoding="utf-8")
    app_offset = source.index("app = modal.App")
    volume_offset = source.index("subject_volume = modal.Volume")
    gate_offset = source.index("control_plane_sha256 = os.environ.get(")

    assert gate_offset < app_offset < volume_offset
    assert 'IQL_MODAL_WORKSPACE_BUDGET_CONFIRMED") != "800"' in source
    assert 'IQL_MODAL_BILLING_CYCLE_END_CONFIRMED")' in source
    assert 'IQL_MODAL_SMOKE_CONTROL_PLANE_SHA256")' in source
    assert "require_stage_billing_window(" in source
    assert len(_app_functions(module)) == 1


def test_manager_seals_deployment_and_writes_intent_before_spawn() -> None:
    source = MANAGER_PATH.read_text(encoding="utf-8")
    launch_start = source.index("def _launch_locked(")
    launch_end = source.index("\ndef launch(", launch_start)
    launch_source = source[launch_start:launch_end]
    deploy_start = source.index("def _deploy_locked(")
    deploy_end = source.index("\ndef deploy(", deploy_start)
    deploy_source = source[deploy_start:deploy_end]

    assert "IQL_MODAL_SMOKE_CONTROL_PLANE_SHA256" in deploy_source
    assert "env=deployment_environment" in deploy_source
    create_offset = launch_source.index("_create_launch_intent(")
    publish_offset = launch_source.index("_publish_remote_launch_intent(", create_offset)
    spawn_offset = launch_source.index("function.spawn(")
    assert create_offset < publish_offset < spawn_offset
    assert launch_source.index("_revalidate_history(deployment)", publish_offset) < spawn_offset
    assert launch_source.index("_require_fresh_attempt(", publish_offset) < spawn_offset
    assert launch_source.count("function.spawn(") == 1
    assert launch_source.index("function.spawn(") < launch_source.index(
        "_revalidate_history(deployment)", launch_source.index("function.spawn(")
    )
    post_spawn_history_offset = launch_source.index(
        "_revalidate_history(deployment)",
        spawn_offset,
    )
    fresh_lookup_offset = launch_source.index("_hydrate_binding(", spawn_offset)
    fresh_binding_check_offset = launch_source.index(
        'raise RuntimeError("Smoke Function binding changed after spawn")',
        fresh_lookup_offset,
    )
    acceptance_offset = launch_source.index("_publish_post_spawn_acceptance(")
    call_receipt_offset = launch_source.index("_write_immutable_json(call_path")
    assert (
        spawn_offset
        < post_spawn_history_offset
        < fresh_lookup_offset
        < fresh_binding_check_offset
        < acceptance_offset
        < call_receipt_offset
    )
    assert "call.cancel(terminate_containers=True)" in launch_source
    assert "_require_fresh_attempt" in launch_source


def test_manager_publishes_launch_authorization_without_overwrite() -> None:
    source = MANAGER_PATH.read_text(encoding="utf-8")
    publish_start = source.index("def _publish_remote_launch_intent(")
    publish_end = source.index("\ndef _launch_locked(", publish_start)
    publish_source = source[publish_start:publish_end]

    assert "smoke_launch_intent_remote_path(" in publish_source
    assert "validate_smoke_launch_intent(" in publish_source
    assert "_read_volume_bytes(volume, remote_path) is not None" in publish_source
    assert "batch_upload(force=False)" in publish_source
    assert "batch.put_file(local_path, remote_path)" in publish_source
    assert publish_source.count("validate_smoke_launch_intent(") == 2


def test_manager_deployment_tag_fits_modal_limit_and_binds_hash_prefix() -> None:
    deployment_tag = _isolated_manager_functions("_deployment_tag")["_deployment_tag"]
    assert callable(deployment_tag)
    control_sha256 = "a" * 64

    observed = deployment_tag(SimpleNamespace(tree_sha256=control_sha256))

    assert observed == smoke_deployment_tag(control_sha256)
    assert observed == f"iql-smoke-{control_sha256[:40]}"
    assert len(observed) == 50


@pytest.mark.parametrize(
    ("status", "extra"),
    (
        ("accepted", {}),
        (
            "cancellation_requested",
            {
                "post_spawn_error": "deployment moved",
                "cancel_error": None,
            },
        ),
        (
            "acceptance_state_unknown",
            {
                "post_spawn_error": "upload response was ambiguous",
                "cancel_error": None,
            },
        ),
        (
            "cancellation_failed",
            {
                "post_spawn_error": "deployment moved",
                "cancel_error": "cancel failed",
            },
        ),
    ),
)
def test_manager_accepts_every_emitted_call_status(
    status: str,
    extra: dict[str, object],
) -> None:
    validate = _isolated_manager_functions("_validated_call_status")["_validated_call_status"]
    assert callable(validate)

    assert validate({"status": status, **extra}, Path("call.json")) == status


@pytest.mark.parametrize(
    "receipt",
    (
        {"status": "unknown"},
        {"status": "accepted", "cancel_error": None},
        {"status": "cancellation_requested"},
        {
            "status": "cancellation_requested",
            "post_spawn_error": "deployment moved",
        },
        {
            "status": "cancellation_failed",
            "post_spawn_error": "deployment moved",
            "cancel_error": None,
        },
    ),
)
def test_manager_rejects_impossible_call_status_evidence(
    receipt: dict[str, object],
) -> None:
    validate = _isolated_manager_functions("_validated_call_status")["_validated_call_status"]
    assert callable(validate)

    with pytest.raises(RuntimeError):
        validate(receipt, Path("call.json"))


def test_runner_validates_remote_authorization_before_attempt_commit() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    smoke_start = source.index("def smoke_test(")
    smoke_source = source[smoke_start:]
    wait_start = source.index("def _wait_for_post_spawn_acceptance(")
    wait_end = source.index("\ndef _bind_run_inputs(", wait_start)
    wait_source = source[wait_start:wait_end]

    authorization_offset = smoke_source.index("validate_smoke_launch_intent(")
    acceptance_offset = smoke_source.index("_wait_for_post_spawn_acceptance(")
    bind_offset = smoke_source.index("_bind_run_inputs(")
    attempt_offset = smoke_source.index("_begin_only_attempt(")
    server_offset = smoke_source.index("subprocess.Popen(")
    assert authorization_offset < acceptance_offset < attempt_offset < bind_offset < server_offset
    assert 'phase = "validate_remote_launch_authorization"' in smoke_source
    assert 'phase = "wait_for_post_spawn_acceptance"' in smoke_source
    assert "_read_existing_regular_bytes(authorization_path)" in smoke_source
    assert "MAX_LAUNCH_INTENT_BYTES" in smoke_source
    reload_offset = wait_source.index("evidence_volume.reload()")
    read_offset = wait_source.index("_read_existing_regular_bytes(acceptance_path)")
    validate_offset = wait_source.index("validate_smoke_post_spawn_acceptance(")
    assert reload_offset < read_offset < validate_offset


def test_manager_fails_closed_on_attempt_claim_and_terminal_receipts() -> None:
    source = MANAGER_PATH.read_text(encoding="utf-8")
    fresh_start = source.index("def _require_fresh_attempt(")
    fresh_end = source.index("\ndef _validated_local_calls(", fresh_start)
    fresh_source = source[fresh_start:fresh_end]
    invocation_start = source.index("def _validate_persisted_invocation_records(")
    invocation_end = source.index("\ndef _validate_terminal_receipt(", invocation_start)
    invocation_source = source[invocation_start:invocation_end]
    terminal_start = source.index("def _validate_terminal_receipt(")
    terminal_end = source.index("\ndef _require_fresh_attempt(", terminal_start)
    terminal_source = source[terminal_start:terminal_end]
    remote_start = source.index("def _remote_status(")
    remote_end = source.index("\ndef inspect(", remote_start)
    remote_source = source[remote_start:remote_end]
    failure_start = source.index("def _validated_remote_failure_receipts(")
    failure_end = source.index("\ndef _validated_local_calls(", failure_start)
    failure_source = source[failure_start:failure_end]

    assert "control/smoke_test.attempt.claim.json" in fresh_source
    assert "claim is not None" in fresh_source
    assert "validate_smoke_terminal_receipt(" in terminal_source
    assert "smoke_launch_intent_remote_path(" in terminal_source
    assert "validate_smoke_launch_intent(" in terminal_source
    assert "Remote smoke success has no launch authorization" in terminal_source
    assert "_validate_remote_post_spawn_acceptance(" in terminal_source
    assert "_validate_persisted_invocation_records(" in terminal_source
    assert "_read_volume_bytes(volume, claim_path)" in invocation_source
    assert "hashlib.sha256(persisted_claim).hexdigest()" in invocation_source
    assert "_validate_attempt_registry_claim_for_invocation(" in invocation_source
    assert "_read_attempt_registry_claim(" in invocation_source
    assert "registry_claim[1] != persisted_claim" in invocation_source
    assert "invocation.attempt_claim_path" in invocation_source
    assert "invocation.invocation_history_path" in invocation_source
    assert "history_payload != expected_history_payload" in invocation_source
    assert "_validated_remote_failure_receipts(" in fresh_source
    assert "validate_smoke_failure_receipt(" in failure_source
    assert '"inkling-smoke-terminal-v3"' in failure_source
    assert '"inkling-smoke-terminal-v4"' in failure_source
    assert "_safe_subprocess_failure_record(" in failure_source
    assert '"failure_records": failures' in remote_source
    assert "_validate_remote_post_spawn_acceptance(" in failure_source
    assert "_validate_persisted_invocation_records(" in failure_source
    assert "validate_smoke_launch_intent(" in failure_source
    assert "smoke_launch_intent_remote_path(" in failure_source
    assert "_validated_remote_failure_receipts(" in remote_source
    assert "conflicting success and failure outcomes" in fresh_source
    assert "conflicting success and failure outcomes" in remote_source


def test_runner_reconciles_terminal_success_without_writing_a_failure() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert ".read_file(" not in source
    reconcile_start = source.index("def _commit_and_reconcile_volume_files(")
    publish_start = source.index("def _publish_success_receipt(")
    reconcile_source = source[reconcile_start:publish_start]
    publish_end = source.index("\ndef _publish_failure_receipt(", publish_start)
    publish_source = source[publish_start:publish_end]
    smoke_start = source.index("def smoke_test(")
    smoke_source = source[smoke_start:]

    install_offset = publish_source.index("_atomic_bytes(")
    reconcile_call_offset = publish_source.index("_commit_and_reconcile_volume_files(")
    assert install_offset < reconcile_call_offset
    retry_offset = reconcile_source.index("for commit_sequence in (1, 2):")
    commit_offset = reconcile_source.index("evidence_volume.commit()")
    reload_offset = reconcile_source.index("evidence_volume.reload()")
    mounted_read_offset = reconcile_source.index("_read_reloaded_volume_bytes(")
    assert retry_offset < commit_offset < reload_offset < mounted_read_offset
    assert "modal.Volume.from_id(" not in reconcile_source
    assert ".read_file(" not in reconcile_source
    assert "_DurablePublicationStateUnknownError" in reconcile_source
    assert "success_publication_started = False" in smoke_source
    assert "success_publication_confirmed = False" in smoke_source
    started_offset = smoke_source.index("success_publication_started = True")
    publish_offset = smoke_source.index("_publish_success_receipt(")
    confirmation_offset = smoke_source.index("success_publication_confirmed = True")
    assert started_offset < publish_offset < confirmation_offset
    typed_guard_offset = smoke_source.index(
        "isinstance(error, _DurablePublicationStateUnknownError)"
    )
    failure_offset = smoke_source.index(
        "_record_failure(",
        typed_guard_offset,
    )
    assert typed_guard_offset < failure_offset
    assert "Terminal success publication was confirmed." in smoke_source
    assert "Durable publication has an unknown committed state." in smoke_source
    assert "conflicting result is unsafe." in smoke_source


def test_success_publication_retries_a_commit_that_raised_before_persisting(
    tmp_path: Path,
) -> None:
    receipt = {"receipt_sha256": "a" * 64, "schema_version": "terminal-test-v1"}
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    publisher, volume, installed = _isolated_success_publisher(
        tmp_path,
        commit_outcomes=(OSError("commit failed before persistence"), None),
        read_outcomes=(None, payload),
    )
    assert callable(publisher)

    publisher(
        tmp_path / "run" / "smoke_test.success.json",
        receipt,
        run_id="inkling-smoke-test",
    )

    assert volume.commit_calls == 2
    assert volume.reload_calls == 2
    assert installed == [payload, payload]


def test_success_publication_accepts_commit_reload_and_exact_mounted_bytes(
    tmp_path: Path,
) -> None:
    receipt = {"receipt_sha256": "9" * 64, "schema_version": "terminal-test-v1"}
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    publisher, volume, installed = _isolated_success_publisher(
        tmp_path,
        commit_outcomes=(None,),
        read_outcomes=(payload,),
    )
    assert callable(publisher)

    publisher(
        tmp_path / "run" / "smoke_test.success.json",
        receipt,
        run_id="inkling-smoke-test",
    )

    assert volume.commit_calls == 1
    assert volume.reload_calls == 1
    assert installed == [payload]


def test_success_publication_accepts_exact_mounted_read_after_commit_response_error(
    tmp_path: Path,
) -> None:
    receipt = {"receipt_sha256": "b" * 64, "schema_version": "terminal-test-v1"}
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    publisher, volume, installed = _isolated_success_publisher(
        tmp_path,
        commit_outcomes=(OSError("commit persisted but response was lost"),),
        read_outcomes=(payload,),
    )
    assert callable(publisher)

    publisher(
        tmp_path / "run" / "smoke_test.success.json",
        receipt,
        run_id="inkling-smoke-test",
    )

    assert volume.commit_calls == 1
    assert volume.reload_calls == 1
    assert installed == [payload]


def test_success_publication_reconciles_commit_interruption_after_apply(
    tmp_path: Path,
) -> None:
    receipt = {"receipt_sha256": "7" * 64, "schema_version": "terminal-test-v1"}
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    publisher, volume, installed = _isolated_success_publisher(
        tmp_path,
        commit_outcomes=(KeyboardInterrupt("commit applied before interruption"),),
        read_outcomes=(payload,),
    )
    assert callable(publisher)

    publisher(
        tmp_path / "run" / "smoke_test.success.json",
        receipt,
        run_id="inkling-smoke-test",
    )

    assert volume.commit_calls == 1
    assert volume.reload_calls == 1
    assert installed == [payload]


def test_failure_publication_retries_and_reconciles_exact_mounted_bytes(
    tmp_path: Path,
) -> None:
    receipt = {"receipt_sha256": "e" * 64, "schema_version": "terminal-test-v1"}
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    publisher, volume, installed = _isolated_failure_publisher(
        tmp_path,
        commit_outcomes=(OSError("commit failed before persistence"), None),
        read_outcomes=(None, payload),
    )
    assert callable(publisher)

    publisher(
        tmp_path / "run" / "control" / "outcomes" / "smoke_test.failed.test.json",
        receipt,
        run_id="inkling-smoke-test",
    )

    assert volume.commit_calls == 2
    assert volume.reload_calls == 2
    assert installed == [payload, payload]


def test_failure_publication_accepts_mounted_bytes_after_lost_commit_response(
    tmp_path: Path,
) -> None:
    receipt = {"receipt_sha256": "f" * 64, "schema_version": "terminal-test-v1"}
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    publisher, volume, installed = _isolated_failure_publisher(
        tmp_path,
        commit_outcomes=(OSError("commit persisted but response was lost"),),
        read_outcomes=(payload,),
    )
    assert callable(publisher)

    publisher(
        tmp_path / "run" / "control" / "outcomes" / "smoke_test.failed.test.json",
        receipt,
        run_id="inkling-smoke-test",
    )

    assert volume.commit_calls == 1
    assert volume.reload_calls == 1
    assert installed == [payload]


def test_multi_file_bookkeeping_commit_retries_and_reconciles_every_file() -> None:
    namespace = _isolated_runner_functions("_commit_and_reconcile_volume_files")
    expected = {
        "run/control/history/invocation.json": b'{"history":"exact"}\n',
        "run/control/smoke_test.attempts.json": b'{"ledger":"exact"}\n',
    }
    volume = _CommitSequence((OSError("first commit response failed"), None))
    reads: list[tuple[int, int, str]] = []
    reinstalls: list[tuple[str, bytes, bool]] = []

    def read_reloaded_volume_bytes(remote_path: str) -> bytes | None:
        reads.append((volume.commit_calls, volume.reload_calls, remote_path))
        if volume.commit_calls == 1:
            return None
        return expected[remote_path]

    namespace.update(
        {
            "SMOKE_STAGE": "smoke_test",
            "_DurablePublicationStateUnknownError": (_TestDurablePublicationStateUnknownError),
            "_atomic_bytes": lambda path, payload, *, allow_identical: reinstalls.append(
                (path.as_posix(), payload, allow_identical)
            ),
            "_canonical_json": _canonical_json_text,
            "_read_reloaded_volume_bytes": read_reloaded_volume_bytes,
            "_safe_child": lambda root, *parts: root.joinpath(*parts),
            "evidence_volume": volume,
            "sys": sys,
        }
    )
    reconcile = namespace["_commit_and_reconcile_volume_files"]
    assert callable(reconcile)

    reconcile(
        expected,
        event="attempt_bookkeeping_commit_reconciled",
        run_id="inkling-smoke-test",
    )

    ordered_paths = sorted(expected)
    assert volume.commit_calls == 2
    assert volume.reload_calls == 2
    assert reads == [
        (1, 1, ordered_paths[0]),
        (1, 1, ordered_paths[1]),
        (2, 2, ordered_paths[0]),
        (2, 2, ordered_paths[1]),
    ]
    assert reinstalls == [
        (f"/evidence/{ordered_paths[0]}", expected[ordered_paths[0]], True),
        (f"/evidence/{ordered_paths[1]}", expected[ordered_paths[1]], True),
    ]


@pytest.mark.parametrize(
    "remote_path",
    (
        "/absolute.json",
        "../outside.json",
        "runs//receipt.json",
        "runs/./receipt.json",
        "runs\\receipt.json",
    ),
)
def test_publication_rejects_invalid_path_before_commit(remote_path: str) -> None:
    namespace = _isolated_runner_functions(
        "_safe_child",
        "_commit_and_reconcile_volume_files",
    )
    volume = _CommitSequence(())
    namespace.update(
        {
            "_DurablePublicationStateUnknownError": (_TestDurablePublicationStateUnknownError),
            "evidence_volume": volume,
        }
    )
    reconcile = namespace["_commit_and_reconcile_volume_files"]
    assert callable(reconcile)

    with pytest.raises((ValueError, RuntimeError)):
        reconcile(
            {remote_path: b"exact\n"},
            event="invalid_path_must_not_commit",
            run_id="inkling-smoke-test",
        )

    assert volume.commit_calls == 0
    assert volume.reload_calls == 0


def test_success_publication_treats_two_unreadable_mounted_reads_as_unknown(
    tmp_path: Path,
) -> None:
    receipt = {"receipt_sha256": "c" * 64, "schema_version": "terminal-test-v1"}
    publisher, volume, _installed = _isolated_success_publisher(
        tmp_path,
        commit_outcomes=(None, None),
        read_outcomes=(
            OSError("first mounted read failed"),
            OSError("second mounted read failed"),
        ),
    )
    assert callable(publisher)

    with pytest.raises(
        _TestDurablePublicationStateUnknownError,
        match="unknown committed state",
    ):
        publisher(
            tmp_path / "run" / "smoke_test.success.json",
            receipt,
            run_id="inkling-smoke-test",
        )

    assert volume.commit_calls == 2
    assert volume.reload_calls == 2


def test_success_publication_treats_two_reload_failures_as_unknown(
    tmp_path: Path,
) -> None:
    receipt = {"receipt_sha256": "1" * 64, "schema_version": "terminal-test-v1"}
    publisher, volume, _installed = _isolated_success_publisher(
        tmp_path,
        commit_outcomes=(None, None),
        reload_outcomes=(
            OSError("first mounted reload failed"),
            OSError("second mounted reload failed"),
        ),
        read_outcomes=(),
    )
    assert callable(publisher)

    with pytest.raises(
        _TestDurablePublicationStateUnknownError,
        match="unknown committed state",
    ):
        publisher(
            tmp_path / "run" / "smoke_test.success.json",
            receipt,
            run_id="inkling-smoke-test",
        )

    assert volume.commit_calls == 2
    assert volume.reload_calls == 2


@pytest.mark.parametrize("interrupted_operation", ("reload", "read"))
def test_success_publication_treats_repeated_base_interruptions_as_unknown(
    tmp_path: Path,
    interrupted_operation: str,
) -> None:
    receipt = {"receipt_sha256": "2" * 64, "schema_version": "terminal-test-v1"}
    reload_outcomes: Sequence[object] | None = None
    read_outcomes: Sequence[object] = (
        KeyboardInterrupt("first mounted read interruption"),
        KeyboardInterrupt("second mounted read interruption"),
    )
    if interrupted_operation == "reload":
        reload_outcomes = (
            KeyboardInterrupt("first mounted reload interruption"),
            KeyboardInterrupt("second mounted reload interruption"),
        )
        read_outcomes = ()
    publisher, volume, _installed = _isolated_success_publisher(
        tmp_path,
        commit_outcomes=(None, None),
        reload_outcomes=reload_outcomes,
        read_outcomes=read_outcomes,
    )
    assert callable(publisher)

    with pytest.raises(
        _TestDurablePublicationStateUnknownError,
        match="unknown committed state",
    ):
        publisher(
            tmp_path / "run" / "smoke_test.success.json",
            receipt,
            run_id="inkling-smoke-test",
        )

    assert volume.commit_calls == 2
    assert volume.reload_calls == 2


def test_reconciliation_diagnostic_failure_is_best_effort() -> None:
    namespace = _isolated_runner_functions("_commit_and_reconcile_volume_files")
    expected = {"run/smoke_test.success.json": b'{"exact":true}\n'}
    volume = _CommitSequence((OSError("commit response was lost"),))

    class _FailingDiagnosticStream:
        def write(self, _value: str) -> int:
            raise OSError("diagnostic stream failed")

        def flush(self) -> None:
            raise OSError("diagnostic stream failed")

    namespace.update(
        {
            "SMOKE_STAGE": "smoke_test",
            "_DurablePublicationStateUnknownError": (_TestDurablePublicationStateUnknownError),
            "_atomic_bytes": lambda *_args, **_kwargs: None,
            "_canonical_json": _canonical_json_text,
            "_read_reloaded_volume_bytes": lambda remote_path: expected[remote_path],
            "_safe_child": lambda root, *parts: root.joinpath(*parts),
            "evidence_volume": volume,
            "suppress": suppress,
            "sys": SimpleNamespace(stderr=_FailingDiagnosticStream()),
        }
    )
    reconcile = namespace["_commit_and_reconcile_volume_files"]
    assert callable(reconcile)

    reconcile(
        expected,
        event="success_commit_reconciled",
        run_id="inkling-smoke-test",
    )

    assert volume.commit_calls == 1
    assert volume.reload_calls == 1


def test_interruption_after_success_installation_becomes_durability_unknown(
    tmp_path: Path,
) -> None:
    namespace = _isolated_runner_functions("_publish_success_receipt")
    success_path = tmp_path / "run" / "smoke_test.success.json"
    receipt = {"receipt_sha256": "8" * 64, "schema_version": "terminal-test-v1"}
    failure_calls: list[object] = []

    def install_then_return(
        path: Path,
        payload: bytes,
        *,
        allow_identical: bool,
    ) -> None:
        assert allow_identical is False
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)

    def interrupt_before_reconciliation(
        _expected_files: Mapping[str, bytes],
        *,
        event: str,
        run_id: str,
    ) -> None:
        assert event == "success_commit_reconciled"
        assert run_id == "inkling-smoke-test"
        raise KeyboardInterrupt("interrupted after success installation")

    namespace.update(
        {
            "EVIDENCE_MOUNT": tmp_path,
            "MAX_TERMINAL_RECEIPT_BYTES": 64 * 1024,
            "_DurablePublicationStateUnknownError": (_TestDurablePublicationStateUnknownError),
            "_atomic_bytes": install_then_return,
            "_canonical_json": _canonical_json_text,
            "_commit_and_reconcile_volume_files": interrupt_before_reconciliation,
        }
    )
    publisher = namespace["_publish_success_receipt"]
    assert callable(publisher)

    with pytest.raises(_TestDurablePublicationStateUnknownError) as publication:
        publisher(success_path, receipt, run_id="inkling-smoke-test")

    assert success_path.is_file()
    handler_namespace = _isolated_smoke_exception_handler()
    handler_namespace["_record_failure"] = lambda *_args, **_kwargs: failure_calls.append(object())
    handler = handler_namespace["_execute"]
    assert callable(handler)

    with pytest.raises(_TestDurablePublicationStateUnknownError):
        handler(publication.value, tmp_path / "run", object(), True, False)

    assert failure_calls == []


def test_differing_committed_success_is_unknown_and_cannot_publish_failure(
    tmp_path: Path,
) -> None:
    receipt = {"receipt_sha256": "d" * 64, "schema_version": "terminal-test-v1"}
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")
    publisher, volume, installed = _isolated_success_publisher(
        tmp_path,
        commit_outcomes=(None,),
        read_outcomes=(b'{"different":"committed bytes"}\n',),
    )
    assert callable(publisher)

    with pytest.raises(
        _TestDurablePublicationStateUnknownError,
        match="differs from the installed bytes",
    ) as publication:
        publisher(
            tmp_path / "run" / "smoke_test.success.json",
            receipt,
            run_id="inkling-smoke-test",
        )
    assert volume.commit_calls == 1
    assert volume.reload_calls == 1
    assert installed == [payload]

    failure_calls: list[object] = []
    handler_namespace = _isolated_smoke_exception_handler()
    handler_namespace["_record_failure"] = lambda *_args, **_kwargs: failure_calls.append(object())
    handler = handler_namespace["_execute"]
    assert callable(handler)

    with pytest.raises(_TestDurablePublicationStateUnknownError) as handled:
        handler(publication.value, tmp_path / "run", object(), True, False)

    assert handled.value is publication.value
    assert failure_calls == []
    assert any(
        "No failure receipt was written because a conflicting terminal result is unsafe." in note
        for note in handled.value.__notes__
    )


def test_generic_failure_before_success_confirmation_is_recorded(
    tmp_path: Path,
) -> None:
    error = RuntimeError("ordinary smoke failure")
    failure_calls: list[tuple[object, dict[str, object]]] = []
    handler_namespace = _isolated_smoke_exception_handler()

    def record_failure(run_root: object, **kwargs: object) -> None:
        failure_calls.append((run_root, kwargs))

    handler_namespace.update(
        {
            "_record_failure": record_failure,
            "config": object(),
            "control_plane": object(),
            "launch_intent_sha256": "a" * 64,
            "phase": "server_probe",
        }
    )
    handler = handler_namespace["_execute"]
    assert callable(handler)
    invocation = object()
    run_root = tmp_path / "run"

    with pytest.raises(RuntimeError) as handled:
        handler(error, run_root, invocation, False, False)

    assert handled.value is error
    assert len(failure_calls) == 1
    assert failure_calls[0][0] == run_root
    assert failure_calls[0][1]["invocation"] is invocation
    assert failure_calls[0][1]["error"] is error


def test_exception_after_confirmed_success_cannot_publish_failure(
    tmp_path: Path,
) -> None:
    error = RuntimeError("return construction failed after durable success")
    failure_calls: list[object] = []
    handler_namespace = _isolated_smoke_exception_handler()
    handler_namespace["_record_failure"] = lambda *_args, **_kwargs: failure_calls.append(object())
    handler = handler_namespace["_execute"]
    assert callable(handler)

    with pytest.raises(RuntimeError) as handled:
        handler(error, tmp_path / "run", object(), True, True)

    assert handled.value is error
    assert failure_calls == []
    assert any(
        "Terminal success publication was confirmed. No failure receipt was written "
        "because a conflicting terminal result is unsafe." in note
        for note in handled.value.__notes__
    )


def test_exception_after_success_publication_started_cannot_publish_failure(
    tmp_path: Path,
) -> None:
    error = KeyboardInterrupt("interrupted after success publisher returned")
    failure_calls: list[object] = []
    handler_namespace = _isolated_smoke_exception_handler()
    handler_namespace["_record_failure"] = lambda *_args, **_kwargs: failure_calls.append(object())
    handler = handler_namespace["_execute"]
    assert callable(handler)

    with pytest.raises(KeyboardInterrupt) as handled:
        handler(error, tmp_path / "run", object(), True, False)

    assert handled.value is error
    assert failure_calls == []
    assert any(
        "Terminal success publication started. No failure receipt was written "
        "because a conflicting terminal result is unsafe." in note
        for note in handled.value.__notes__
    )


def test_manager_cross_reads_local_acceptance_for_each_accepted_call() -> None:
    source = MANAGER_PATH.read_text(encoding="utf-8")
    helper_start = source.index("def _validate_local_post_spawn_acceptance(")
    helper_end = source.index("\ndef _validated_local_calls(", helper_start)
    helper_source = source[helper_start:helper_end]
    calls_start = source.index("def _validated_local_calls(")
    calls_end = source.index("\ndef _require_no_unresolved_intent(", calls_start)
    calls_source = source[calls_start:calls_end]

    assert 'if status == "accepted":' in calls_source
    assert "_validate_local_post_spawn_acceptance(" in calls_source
    assert "payload = local_path.read_bytes()" in helper_source
    assert "hashlib.sha256(payload).hexdigest()" in helper_source
    assert "validate_smoke_post_spawn_acceptance(" in helper_source


def test_manager_treats_builtin_timeout_as_a_nonterminal_poll() -> None:
    source = MANAGER_PATH.read_text(encoding="utf-8")
    status_start = source.index("def status(")
    status_source = source[status_start : source.index("\ndef _parser(", status_start)]

    assert "except (TimeoutError, modal.exception.TimeoutError) as error:" in status_source
    assert "type(error) not in {TimeoutError, modal.exception.TimeoutError}" in status_source
    assert status_source.index("call.get_call_graph()") < status_source.index("call.get(timeout=0)")
    assert 'function_state = "failed_before_terminal_evidence"' in status_source


@pytest.mark.parametrize("status_name", ("FAILURE", "PENDING"))
def test_manager_call_graph_accepts_transient_empty_task_id(status_name: str) -> None:
    function = _isolated_manager_functions("_call_graph_records")["_call_graph_records"]
    assert callable(function)

    assert function([_call_graph_node(status_name=status_name, task_id="")]) == [
        {
            "input_id": "in-input",
            "function_call_id": "fc-call",
            "task_id": "",
            "status": status_name.casefold(),
            "function_name": "smoke_test",
            "module_name": "smoke_inkling_modal",
        }
    ]


@pytest.mark.parametrize("task_id", (None, 7))
def test_manager_call_graph_rejects_non_string_task_id(task_id: object) -> None:
    function = _isolated_manager_functions("_call_graph_records")["_call_graph_records"]
    assert callable(function)

    with pytest.raises(RuntimeError, match="incomplete input identity"):
        function([_call_graph_node(task_id=task_id)])


@pytest.mark.parametrize(
    "field_name",
    ("input_id", "function_call_id", "status", "function_name", "module_name"),
)
def test_manager_call_graph_rejects_empty_stable_identity(field_name: str) -> None:
    function = _isolated_manager_functions("_call_graph_records")["_call_graph_records"]
    assert callable(function)
    node = (
        _call_graph_node(status_name="")
        if field_name == "status"
        else _call_graph_node(**{field_name: ""})
    )

    with pytest.raises(RuntimeError, match="incomplete input identity"):
        function([node])


@pytest.mark.parametrize("children", (None, (), "not-a-list"))
def test_manager_call_graph_rejects_invalid_child_collection(children: object) -> None:
    function = _isolated_manager_functions("_call_graph_records")["_call_graph_records"]
    assert callable(function)

    with pytest.raises(RuntimeError, match="invalid child collection"):
        function([_call_graph_node(children=children)])
