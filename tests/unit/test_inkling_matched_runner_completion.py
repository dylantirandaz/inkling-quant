"""CPU-only completion contracts for the matched Modal runner.

The paid runner is parsed or sliced instead of imported. Importing it would
construct Modal resources and a CUDA image during an ordinary unit test.
"""

from __future__ import annotations

import ast
import hashlib
import math
import os
import re
import stat
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inkling_quant_lab.gguf.inkling_matched_execution import MatchedFailureCauseCode

pytestmark = pytest.mark.unit

RUNNER_PATH = Path(__file__).resolve().parents[2] / "scripts/run_inkling_matched_modal.py"

_Definition = ast.FunctionDef | ast.ClassDef


def _module() -> ast.Module:
    return ast.parse(
        RUNNER_PATH.read_text(encoding="utf-8"),
        filename=str(RUNNER_PATH),
    )


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, f"expected exactly one top-level {name}()"
    return matches[0]


def _class(module: ast.Module, name: str) -> ast.ClassDef:
    matches = [node for node in module.body if isinstance(node, ast.ClassDef) and node.name == name]
    assert len(matches) == 1, f"expected exactly one top-level {name}"
    return matches[0]


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call) and _call_name(candidate) == name
    ]


def _call_keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg == name),
        None,
    )


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(
        (isinstance(candidate, ast.Name) and candidate.id == name)
        or (isinstance(candidate, ast.Attribute) and candidate.attr == name)
        for candidate in ast.walk(node)
    )


def _local_assignments(node: ast.AST) -> dict[str, ast.expr]:
    assignments: dict[str, ast.expr] = {}
    for candidate in ast.walk(node):
        if isinstance(candidate, ast.Assign):
            for target in candidate.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = candidate.value
        elif (
            isinstance(candidate, ast.AnnAssign)
            and isinstance(candidate.target, ast.Name)
            and candidate.value is not None
        ):
            assignments[candidate.target.id] = candidate.value
    return assignments


def _resolved_expression(
    node: ast.expr,
    assignments: dict[str, ast.expr],
    *,
    seen: frozenset[str] = frozenset(),
) -> ast.expr:
    if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        return _resolved_expression(
            assignments[node.id],
            assignments,
            seen=seen | {node.id},
        )
    return node


def _runtime_slice(
    *roots: str,
    extra_namespace: dict[str, object] | None = None,
) -> dict[str, object]:
    """Execute dependency-closed runner definitions without importing Modal."""

    tree = _module()
    definitions: dict[str, _Definition] = {
        node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    missing = set(roots).difference(definitions)
    assert not missing, f"missing runner definitions: {sorted(missing)}"

    selected = set(roots)
    pending = list(roots)
    while pending:
        definition = definitions[pending.pop()]
        for node in ast.walk(definition):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in definitions
                and node.id not in selected
            ):
                selected.add(node.id)
                pending.append(node.id)

    body: list[ast.stmt] = [
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    body.extend(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in selected
    )
    isolated = ast.Module(body=body, type_ignores=[])
    assert not any(isinstance(node, ast.Name) and node.id == "modal" for node in ast.walk(isolated))

    namespace: dict[str, object] = {
        "Any": Any,
        "dataclass": dataclass,
        "hashlib": hashlib,
        "math": math,
        "MatchedFailureCauseCode": MatchedFailureCauseCode,
        "os": os,
        "Path": Path,
        "re": re,
        "stat": stat,
        "subprocess": subprocess,
        "time": time,
    }
    if extra_namespace is not None:
        namespace.update(extra_namespace)
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(RUNNER_PATH), "exec"), namespace)
    return namespace


def test_atomic_claim_and_all_later_work_share_one_failure_boundary() -> None:
    module = _module()
    entrypoint = _function(module, "matched_smoke_test")
    claim_calls = _calls(entrypoint, "_claim_attempt")
    assert len(claim_calls) == 1
    claim_call = claim_calls[0]

    boundaries = [
        statement
        for statement in entrypoint.body
        if isinstance(statement, ast.Try)
        and claim_call in tuple(ast.walk(ast.Module(body=statement.body, type_ignores=[])))
        and any(_calls(handler, "_build_failure_receipt") for handler in statement.handlers)
        and any(_calls(handler, "_publish_terminal") for handler in statement.handlers)
    ]
    assert len(boundaries) == 1, (
        "the irreversible Dict claim must be inside the same outer try that "
        "records and publishes a sanitized terminal failure"
    )
    boundary = boundaries[0]
    assert entrypoint.body[-1] is boundary, (
        "no operation may follow the outer failure-recording boundary after "
        "the atomic attempt has been consumed"
    )
    assert _calls(ast.Module(body=boundary.body, type_ignores=[]), "_publish_attempt_records")
    assert any(isinstance(node, ast.Return) for node in ast.walk(boundary)), (
        "success-result validation and construction must remain inside the "
        "failure-recording boundary"
    )
    assert any(
        isinstance(handler.type, ast.Name)
        and handler.type.id == "BaseException"
        and _contains_name(handler, "claim_won")
        for handler in boundary.handlers
    )


def test_success_and_failure_reconcile_exact_claim_and_ack_bytes() -> None:
    module = _module()
    entrypoint = _function(module, "matched_smoke_test")
    prepared = _class(module, "_PreparedAttempt")
    publish = _function(module, "_publish_terminal")

    records = next(
        node
        for node in prepared.body
        if isinstance(node, ast.FunctionDef) and node.name == "records"
    )
    payloads = next(
        node
        for node in prepared.body
        if isinstance(node, ast.FunctionDef) and node.name == "payloads"
    )
    assert _contains_name(records, "claim_path")
    assert _contains_name(records, "claim")
    assert _contains_name(records, "acknowledgement_path")
    assert _contains_name(records, "acknowledgement")
    assert _calls(payloads, "records")
    assert _calls(payloads, "canonical_bytes")

    assignments = _local_assignments(entrypoint)
    attempt_files = assignments.get("attempt_files")
    assert isinstance(attempt_files, ast.Call)
    assert _call_name(attempt_files) == "payloads"
    terminal_calls = _calls(entrypoint, "_publish_terminal")
    assert len(terminal_calls) == 2
    assert {getattr(_call_keyword(call, "outcome"), "value", None) for call in terminal_calls} == {
        "success",
        "failure",
    }
    for call in terminal_calls:
        required = _call_keyword(call, "required_files")
        assert isinstance(required, ast.Name) and required.id == "attempt_files"

    publish_assignments = _local_assignments(publish)
    expected_files = publish_assignments.get("expected_files")
    assert isinstance(expected_files, ast.Dict)
    assert any(
        key is None and isinstance(value, ast.Name) and value.id == "required"
        for key, value in zip(expected_files.keys, expected_files.values, strict=True)
    ), "claim and acknowledgement bytes must join the terminal reconciled file set"
    reconciliation_loops = [
        node
        for node in ast.walk(publish)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Attribute)
        and isinstance(node.iter.func.value, ast.Name)
        and node.iter.func.value.id == "expected_files"
        and node.iter.func.attr == "items"
        and _calls(node, "_read_regular_bytes")
    ]
    assert reconciliation_loops, (
        "the same mounted reload view must validate attempt-control and terminal bytes"
    )


def test_broken_exception_string_still_produces_a_stable_sanitized_digest() -> None:
    namespace = _runtime_slice("_failure_message_sha256")
    digest = namespace["_failure_message_sha256"]
    assert callable(digest)

    class BrokenStringError(RuntimeError):
        def __str__(self) -> str:
            raise KeyboardInterrupt("string conversion failed")

    first = digest(BrokenStringError())
    second = digest(BrokenStringError())
    expected = hashlib.sha256(
        (f"<unprintable:{BrokenStringError.__module__}.{BrokenStringError.__qualname__}>").encode()
    ).hexdigest()
    assert first == second == expected


def test_failure_helpers_retain_one_safe_bounded_cause() -> None:
    namespace = _runtime_slice(
        "_failure_artifact_path",
        "_failure_cause_code",
        "_failure_message_sha256",
        "_failure_type",
    )
    stage_error = namespace["_MatchedStageError"]
    underlying = OSError("short write")
    try:
        raise stage_error(
            "artifact_rehash",
            cause_code=MatchedFailureCauseCode.ARTIFACT_SIZE_MISMATCH,
            artifact_path="q3_k_m/inkling-Q3_K_M-00015-of-00049.gguf",
        ) from underlying
    except stage_error as error:
        assert namespace["_failure_cause_code"](error) is (
            MatchedFailureCauseCode.ARTIFACT_SIZE_MISMATCH
        )
        assert namespace["_failure_artifact_path"](error) == (
            "q3_k_m/inkling-Q3_K_M-00015-of-00049.gguf"
        )
        assert namespace["_failure_type"](error) == "OSError"
        assert (
            namespace["_failure_message_sha256"](error)
            == hashlib.sha256(b"short write").hexdigest()
        )


def test_artifact_mismatches_fail_with_path_before_success_model_validation(
    tmp_path: Path,
) -> None:
    namespace = _runtime_slice("_observe_artifact")
    observe = namespace["_observe_artifact"]
    sha256_file = namespace["_sha256_file"]
    stage_error = namespace["_MatchedStageError"]
    size_error = namespace["_FileSizeMismatchError"]
    artifact = SimpleNamespace(
        path="q3_k_m/inkling-Q3_K_M-00015-of-00049.gguf",
        sha256="a" * 64,
        size_bytes=10,
    )

    undersized = tmp_path / "undersized.gguf"
    undersized.write_bytes(b"")
    with pytest.raises(size_error, match="size drifted"):
        sha256_file(
            undersized,
            expected_size=artifact.size_bytes,
            failure_category="artifact_rehash",
            work_deadline_monotonic=time.monotonic() + 5.0,
        )

    def raise_size_mismatch(*_args: object, **_kwargs: object) -> None:
        raise size_error("subject artifact size drifted")

    namespace["_sha256_file"] = raise_size_mismatch
    with pytest.raises(stage_error) as size_mismatch:
        observe(
            subject=object(),
            kind="text_shard",
            artifact=artifact,
            absolute_path=f"/final/{artifact.path}",
            work_deadline_monotonic=100.0,
            shard_ordinal=15,
        )
    assert size_mismatch.value.cause_code is MatchedFailureCauseCode.ARTIFACT_SIZE_MISMATCH
    assert size_mismatch.value.artifact_path == artifact.path

    namespace["_sha256_file"] = lambda *_args, **_kwargs: ("b" * 64, 10)
    with pytest.raises(stage_error) as wrong_hash:
        observe(
            subject=object(),
            kind="text_shard",
            artifact=artifact,
            absolute_path=f"/final/{artifact.path}",
            work_deadline_monotonic=100.0,
            shard_ordinal=15,
        )
    assert wrong_hash.value.cause_code is MatchedFailureCauseCode.ARTIFACT_HASH_MISMATCH
    assert wrong_hash.value.artifact_path == artifact.path


def test_work_timeout_is_monotonic_bounded_and_fail_closed() -> None:
    clock = SimpleNamespace(monotonic=lambda: 100.0)
    namespace = _runtime_slice(
        "_remaining_work_timeout",
        extra_namespace={"time": clock},
    )
    remaining = namespace["_remaining_work_timeout"]
    stage_error = namespace["_MatchedStageError"]
    assert callable(remaining)
    assert isinstance(stage_error, type)

    assert remaining(130.0, 900.0, "probe") == pytest.approx(30.0)
    assert remaining(130.0, 5.0, "server_health") == pytest.approx(5.0)
    with pytest.raises(stage_error) as captured:
        remaining(100.0, 5.0, "server_health")
    assert captured.value.category == "server_health"
    assert captured.value.cause_code is MatchedFailureCauseCode.DEADLINE_EXHAUSTED
    assert captured.value.artifact_path is None


def test_whole_run_deadline_reserves_terminal_time_and_clamps_remote_waits() -> None:
    module = _module()
    entrypoint = _function(module, "matched_smoke_test")
    run_subject = _function(module, "_run_subject")
    rehash_subject = _function(module, "_rehash_subject")
    sha256_file = _function(module, "_sha256_file")

    constants = _local_assignments(module)
    function_timeout = constants.get("FUNCTION_TIMEOUT_SECONDS")
    terminal_reserve = constants.get("TERMINAL_PUBLICATION_RESERVE_SECONDS")
    monitor_timeout = constants.get("MONITOR_COMMAND_TIMEOUT_SECONDS")
    assert isinstance(function_timeout, ast.Constant)
    assert isinstance(terminal_reserve, ast.Constant)
    assert isinstance(monitor_timeout, ast.Constant)
    assert isinstance(function_timeout.value, (int, float))
    assert isinstance(terminal_reserve.value, (int, float))
    assert isinstance(monitor_timeout.value, (int, float))
    assert 0 < terminal_reserve.value < function_timeout.value
    assert terminal_reserve.value > 60.0 + monitor_timeout.value + 5.0, (
        "the reserve must cover both bounded process waits and the monitor join, "
        "then leave positive time for terminal publication"
    )

    modal_decorators = [
        decorator
        for decorator in entrypoint.decorator_list
        if isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "function"
    ]
    assert len(modal_decorators) == 1
    modal_timeout = _call_keyword(modal_decorators[0], "timeout")
    assert modal_timeout is not None
    modal_timeout = _resolved_expression(modal_timeout, constants)
    assert isinstance(modal_timeout, ast.Constant)
    assert modal_timeout.value == function_timeout.value

    entry_assignments = _local_assignments(entrypoint)
    work_deadline = entry_assignments.get("work_deadline_monotonic")
    assert work_deadline is not None
    assert _calls(work_deadline, "monotonic")
    assert _contains_name(work_deadline, "FUNCTION_TIMEOUT_SECONDS")
    assert _contains_name(work_deadline, "TERMINAL_PUBLICATION_RESERVE_SECONDS")

    for function in (run_subject, rehash_subject, sha256_file):
        assert any(
            argument.arg == "work_deadline_monotonic"
            for argument in (*function.args.args, *function.args.kwonlyargs)
        ), f"{function.name} must receive the shared whole-run work deadline"
    hash_loops = [
        node
        for node in ast.walk(sha256_file)
        if isinstance(node, ast.While)
        and _calls(node, "read")
        and _calls(node, "_remaining_work_timeout")
    ]
    assert len(hash_loops) == 1, (
        "large shard hashing must re-check the whole-run deadline between chunks"
    )

    for call_name in ("_rehash_subject", "_run_subject", "_observe_runtime_identity"):
        for call in _calls(entrypoint, call_name):
            deadline = _call_keyword(call, "work_deadline_monotonic")
            assert isinstance(deadline, ast.Name)
            assert deadline.id == "work_deadline_monotonic"

    for helper_name in ("_wait_ready", "_model_properties", "_run_probe"):
        helper = _function(module, helper_name)
        assert any(
            argument.arg == "work_deadline_monotonic"
            for argument in (*helper.args.args, *helper.args.kwonlyargs)
        )
        assignments = _local_assignments(helper)
        http_calls = _calls(helper, "_http_json")
        assert http_calls, f"{helper_name} must make a bounded local server request"
        for call in http_calls:
            timeout = _call_keyword(call, "timeout")
            assert timeout is not None
            timeout = _resolved_expression(timeout, assignments)
            assert _calls(timeout, "_remaining_work_timeout"), (
                f"{helper_name} has an HTTP timeout that is not clamped to "
                "the whole-run work deadline"
            )
        run_subject_calls = _calls(run_subject, helper_name)
        assert run_subject_calls
        for call in run_subject_calls:
            deadline = _call_keyword(call, "work_deadline_monotonic")
            assert isinstance(deadline, ast.Name)
            assert deadline.id == "work_deadline_monotonic"


class _TerminateRaisesProcess:
    def __init__(self) -> None:
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        self.terminate_calls += 1
        raise OSError("terminate failed")

    def kill(self) -> None:
        self.kill_calls += 1

    def wait(self, *, timeout: float) -> int:
        del timeout
        self.wait_calls += 1
        return 0


def test_terminate_failure_still_attempts_kill() -> None:
    namespace = _runtime_slice("_terminate_process")
    terminate = namespace["_terminate_process"]
    assert callable(terminate)
    process = _TerminateRaisesProcess()

    terminate(process)
    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.wait_calls >= 1


def test_process_and_log_cleanup_faults_retain_the_primary_failure() -> None:
    namespace = _runtime_slice("_merge_cleanup_error", "_close_log_handle")
    merge = namespace["_merge_cleanup_error"]
    close = namespace["_close_log_handle"]
    stage_error = namespace["_MatchedStageError"]
    assert callable(merge)
    assert callable(close)
    assert isinstance(stage_error, type)

    class BrokenLog:
        def close(self) -> None:
            raise OSError("private log path must not leak")

    primary = RuntimeError("primary probe failure")
    process_cleanup = merge(primary, RuntimeError("process cleanup failed"))
    assert process_cleanup is primary
    log_cleanup = close(BrokenLog(), process_cleanup)
    assert log_cleanup is primary
    notes = tuple(getattr(primary, "__notes__", ()))
    assert any("builtins.RuntimeError" in note for note in notes)
    assert any("builtins.OSError" in note for note in notes)
    assert all("private log path" not in note for note in notes)

    cleanup_only = close(BrokenLog(), None)
    assert isinstance(cleanup_only, stage_error)
    assert cleanup_only.category == "cleanup"
    assert cleanup_only.cause_code is MatchedFailureCauseCode.CLEANUP_FAILED
    assert isinstance(cleanup_only.__cause__, OSError)

    module = _module()
    run_subject = _function(module, "_run_subject")
    finalizers = [
        node
        for node in ast.walk(run_subject)
        if isinstance(node, ast.Try)
        and _calls(ast.Module(body=node.finalbody, type_ignores=[]), "_close_log_handle")
        and _calls(
            ast.Module(body=node.finalbody, type_ignores=[]),
            "_stop_subject_runtime",
        )
    ]
    assert len(finalizers) == 1
    assert _calls(
        ast.Module(body=finalizers[0].finalbody, type_ignores=[]),
        "_merge_cleanup_error",
    )
