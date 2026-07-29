"""Static, CPU-only contracts for the matched Modal data-plane boundary.

The runner is intentionally parsed instead of imported.  Importing a Modal
entrypoint can build images, resolve Volumes, or otherwise cross a remote
boundary during an ordinary unit-test run.
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = PROJECT_ROOT / "scripts/run_inkling_matched_modal.py"

PINNED_LLAMA_CPP_COMMIT = "a015409e6c27b84f60d688823d4c0126a11571fd"
BUILD_TARGETS = {
    "llama-cli",
    "llama-server",
    "llama-bench",
    "llama-perplexity",
}
READ_ONLY_MOUNTS = {"/baseline", "/final", "/source"}
WRITABLE_MOUNTS = {"/evidence"}
EXPECTED_VOLUME_NAMES = {
    "inkling-work-v1",
    "inkling-final-v1",
    "inkling-source-v1",
    "inkling-matched-evidence-v1",
}
HISTORICAL_TWO_GPU_NAMES = {
    "SmokeGpuTopologyEvidence",
    "combine_gpu_identity",
    "enumerate_cuda_driver_gpus",
    "enumerate_cuda_driver_peer_topology",
    "parse_nvidia_smi_csv",
    "parse_nvidia_smi_monitor_csv",
}

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
}


def _module() -> ast.Module:
    assert RUNNER_PATH.is_file(), (
        "the reviewed matched data-plane entrypoint must be scripts/run_inkling_matched_modal.py"
    )
    return ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))


def _assignments(module: ast.Module) -> dict[str, ast.expr]:
    values: dict[str, ast.expr] = {}
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = statement.value
        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
        ):
            values[statement.target.id] = statement.value
    return values


def _literal(
    node: ast.expr,
    assignments: dict[str, ast.expr],
    *,
    seen: frozenset[str] = frozenset(),
) -> Any:
    if isinstance(node, ast.Name):
        if node.id in seen or node.id not in assignments:
            raise ValueError(f"{node.id} is not a local literal")
        return _literal(
            assignments[node.id],
            assignments,
            seen=seen | {node.id},
        )
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
        operation = _BINARY_OPERATORS[type(node.op)]
        return operation(
            _literal(node.left, assignments, seen=seen),
            _literal(node.right, assignments, seen=seen),
        )
    if isinstance(node, ast.Tuple):
        return tuple(_literal(item, assignments, seen=seen) for item in node.elts)
    if isinstance(node, ast.List):
        return [_literal(item, assignments, seen=seen) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            _literal(key, assignments, seen=seen): _literal(value, assignments, seen=seen)
            for key, value in zip(node.keys, node.values, strict=True)
            if key is not None
        }
    return ast.literal_eval(node)


def _function(module: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        statement
        for statement in module.body
        if isinstance(statement, ast.FunctionDef) and statement.name == name
    ]
    assert len(matches) == 1, f"expected exactly one top-level {name}()"
    return matches[0]


def _call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return ""


def _modal_functions(module: ast.Module) -> list[tuple[ast.FunctionDef, ast.Call]]:
    found: list[tuple[ast.FunctionDef, ast.Call]] = []
    for statement in module.body:
        if not isinstance(statement, ast.FunctionDef):
            continue
        for decorator in statement.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "function"
            ):
                found.append((statement, decorator))
    return found


def _path_text(node: ast.expr, assignments: dict[str, ast.expr]) -> str:
    if isinstance(node, ast.Name):
        return _path_text(assignments[node.id], assignments)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Path"
        and len(node.args) == 1
    ):
        return str(_literal(node.args[0], assignments))
    return str(_literal(node, assignments))


def _is_read_only_volume(node: ast.expr, assignments: dict[str, ast.expr]) -> bool:
    if isinstance(node, ast.Name):
        return _is_read_only_volume(assignments[node.id], assignments)
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_only"
    )


def _calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call) and _call_name(candidate) == name
    ]


def _top_level_functions(module: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        statement.name: statement
        for statement in module.body
        if isinstance(statement, ast.FunctionDef)
    }


def _reachable_function_nodes(
    module: ast.Module,
    root: ast.AST,
) -> tuple[ast.AST, ...]:
    functions = _top_level_functions(module)
    result: list[ast.AST] = [root]
    pending = [
        _call_name(call)
        for call in ast.walk(root)
        if isinstance(call, ast.Call) and _call_name(call) in functions
    ]
    visited: set[str] = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        function = functions[name]
        result.append(function)
        pending.extend(
            _call_name(call)
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and _call_name(call) in functions
            and _call_name(call) not in visited
        )
    return tuple(result)


def _reachable_call_names(module: ast.Module, root: ast.AST) -> set[str]:
    return {
        _call_name(call)
        for node in _reachable_function_nodes(module, root)
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    }


def _local_assignments(function: ast.FunctionDef) -> dict[str, ast.expr]:
    values: dict[str, ast.expr] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            values[node.target.id] = node.value
    return values


def _expression_fingerprint(
    node: ast.AST,
    assignments: dict[str, ast.expr],
    *,
    seen: frozenset[str] = frozenset(),
) -> object:
    if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        return _expression_fingerprint(
            assignments[node.id],
            assignments,
            seen=seen | {node.id},
        )
    fields: list[tuple[str, object]] = []
    for name, value in ast.iter_fields(node):
        if name == "ctx":
            continue
        if isinstance(value, ast.AST):
            fields.append(
                (
                    name,
                    _expression_fingerprint(value, assignments, seen=seen),
                )
            )
        elif isinstance(value, list):
            fields.append(
                (
                    name,
                    tuple(
                        _expression_fingerprint(item, assignments, seen=seen)
                        if isinstance(item, ast.AST)
                        else item
                        for item in value
                    ),
                )
            )
        else:
            fields.append((name, value))
    return (type(node).__name__, tuple(fields))


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(candidate, ast.Name) and candidate.id == name for candidate in ast.walk(node)
    )


def _contains_raise(node: ast.AST) -> bool:
    return any(isinstance(candidate, ast.Raise) for candidate in ast.walk(node))


def _call_keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next(
        (keyword.value for keyword in call.keywords if keyword.arg == name),
        None,
    )


def test_modal_function_has_the_exact_reviewed_resources_and_four_mounts() -> None:
    module = _module()
    assignments = _assignments(module)
    functions = _modal_functions(module)

    assert [function.name for function, _decorator in functions] == ["matched_smoke_test"]
    _function_node, decorator = functions[0]
    keywords = {
        keyword.arg: keyword.value for keyword in decorator.keywords if keyword.arg is not None
    }
    assert _literal(keywords["gpu"], assignments) == "B300:8"
    assert _literal(keywords["cpu"], assignments) == 16
    assert _literal(keywords["memory"], assignments) == 64 * 1024
    assert _literal(keywords["ephemeral_disk"], assignments) == 512 * 1024
    assert _literal(keywords["retries"], assignments) == 0
    assert _literal(keywords["timeout"], assignments) == 14_400
    assert _literal(keywords["startup_timeout"], assignments) == 1_800
    assert _literal(keywords["max_containers"], assignments) == 1
    assert _literal(keywords["single_use_containers"], assignments) is True
    assert _literal(keywords["block_network"], assignments) is True

    volumes_node = keywords["volumes"]
    if isinstance(volumes_node, ast.Name):
        volumes_node = assignments[volumes_node.id]
    assert isinstance(volumes_node, ast.Dict)
    mounts = {
        _path_text(key, assignments): value
        for key, value in zip(volumes_node.keys, volumes_node.values, strict=True)
        if key is not None
    }
    assert set(mounts) == READ_ONLY_MOUNTS | WRITABLE_MOUNTS
    assert {
        path for path, value in mounts.items() if _is_read_only_volume(value, assignments)
    } == READ_ONLY_MOUNTS

    string_literals = {
        node.value
        for node in ast.walk(module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert string_literals >= EXPECTED_VOLUME_NAMES


def test_runtime_is_pinned_and_builds_the_four_reviewed_binaries() -> None:
    module = _module()
    assignments = _assignments(module)
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert PINNED_LLAMA_CPP_COMMIT in source
    assert _literal(assignments["BUILD_TARGETS"], assignments) == (
        "llama-cli",
        "llama-server",
        "llama-bench",
        "llama-perplexity",
    )
    assert all(target in source for target in BUILD_TARGETS)


def test_runner_uses_only_the_new_eight_gpu_parsers_and_never_volume_read_file() -> None:
    module = _module()
    observed_names = {node.id for node in ast.walk(module) if isinstance(node, ast.Name)} | {
        node.attr for node in ast.walk(module) if isinstance(node, ast.Attribute)
    }

    assert not HISTORICAL_TWO_GPU_NAMES & observed_names
    assert "parse_matched_nvidia_smi_identity_csv" in observed_names
    assert "parse_matched_nvidia_smi_monitor_csv" in observed_names
    assert "parse_exact_cuda_backend_audit" in observed_names
    assert not [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Attribute) and node.attr == "read_file"
    ], "a network-blocked data-plane runner must use mounted files, not Volume.read_file()"


def test_subjects_run_bf16_then_q3_as_fresh_processes_with_distinct_commands() -> None:
    module = _module()
    entrypoint = _function(module, "matched_smoke_test")
    run_subject = _function(module, "_run_subject")

    subject_loops = [
        node
        for node in ast.walk(entrypoint)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "MATCHED_SUBJECT_ORDER"
        and _calls(node, "_run_subject")
    ]
    assert len(subject_loops) == 1
    assert len(_calls(run_subject, "Popen")) == 1
    assert len(_calls(run_subject, "build_matched_server_command")) == 1

    imported_or_used_names = {node.id for node in ast.walk(module) if isinstance(node, ast.Name)}
    assert "MATCHED_SUBJECT_ORDER" in imported_or_used_names


def test_monitor_stop_and_join_are_attempted_before_server_termination_on_all_paths() -> None:
    module = _module()
    cleanup = _function(module, "_stop_subject_runtime")
    run_subject = _function(module, "_run_subject")

    ordered_cleanup: list[ast.Try] = []
    for outer_try in (node for node in ast.walk(cleanup) if isinstance(node, ast.Try)):
        nested_finalizers = [
            node
            for statement in outer_try.finalbody
            for node in ast.walk(statement)
            if isinstance(node, ast.Try)
            and _calls(ast.Module(body=node.body, type_ignores=[]), "join")
            and _calls(
                ast.Module(body=node.finalbody, type_ignores=[]),
                "_terminate_process",
            )
        ]
        if (
            _calls(ast.Module(body=outer_try.body, type_ignores=[]), "stop")
            and len(nested_finalizers) == 1
        ):
            ordered_cleanup.append(outer_try)
    assert len(ordered_cleanup) == 1, (
        "_stop_subject_runtime() must nest stop -> join -> terminate through "
        "successive finally clauses, so an exception from either monitor action "
        "still cannot terminate the server first"
    )

    finalizers = [
        node
        for node in ast.walk(run_subject)
        if isinstance(node, ast.Try)
        and _calls(
            ast.Module(body=node.finalbody, type_ignores=[]),
            "_stop_subject_runtime",
        )
    ]
    assert len(finalizers) == 1, (
        "_run_subject() must call _stop_subject_runtime() from its outer finally "
        "clause so success, probe failure, and receipt failure use the same ordering"
    )


def test_terminal_and_control_records_use_their_distinct_canonical_encodings() -> None:
    module = _module()
    functions = _top_level_functions(module)
    publication_functions = [
        function
        for function in functions.values()
        if _calls(function, "build_matched_terminal_receipt_reference")
    ]
    assert len(publication_functions) == 1
    publication = publication_functions[0]
    terminal_reference_call = _calls(
        publication,
        "build_matched_terminal_receipt_reference",
    )[0]
    assert terminal_reference_call.args
    terminal_payload_argument = terminal_reference_call.args[0]
    assert isinstance(terminal_payload_argument, ast.Name)

    publication_assignments = _local_assignments(publication)
    terminal_payload_expression = publication_assignments[terminal_payload_argument.id]
    assert isinstance(terminal_payload_expression, ast.Call)
    serializer_name = _call_name(terminal_payload_expression)
    assert serializer_name in functions
    serializer = functions[serializer_name]
    dumps_calls = _calls(serializer, "dumps")
    assert len(dumps_calls) == 1
    dumps_call = dumps_calls[0]
    ensure_ascii = _call_keyword(dumps_call, "ensure_ascii")
    allow_nan = _call_keyword(dumps_call, "allow_nan")
    sort_keys = _call_keyword(dumps_call, "sort_keys")
    assert isinstance(ensure_ascii, ast.Constant) and ensure_ascii.value is True
    assert isinstance(allow_nan, ast.Constant) and allow_nan.value is False
    assert isinstance(sort_keys, ast.Constant) and sort_keys.value is True
    assert not [
        constant
        for return_node in ast.walk(serializer)
        if isinstance(return_node, ast.Return) and return_node.value is not None
        for constant in ast.walk(return_node.value)
        if isinstance(constant, ast.Constant) and constant.value in {"\n", b"\n"}
    ], "execution receipts must use compact JSON without a trailing newline"

    snapshot_variables: list[tuple[ast.FunctionDef, str]] = []
    for function in functions.values():
        for name, value in _local_assignments(function).items():
            if isinstance(value, ast.Call) and _call_name(value) in {
                "_publication_snapshot",
                "MatchedPublicationSnapshot",
            }:
                snapshot_variables.append((function, name))
    assert len(snapshot_variables) >= 3

    for function, variable in snapshot_variables:
        directly_canonicalized = any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "canonical_bytes"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == variable
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
        )
        helper_canonicalizes = any(
            any(
                isinstance(candidate.func, ast.Attribute)
                and candidate.func.attr == "canonical_bytes"
                for node in _reachable_function_nodes(module, functions[_call_name(call)])
                for candidate in ast.walk(node)
                if isinstance(candidate, ast.Call)
            )
            for call in ast.walk(function)
            if isinstance(call, ast.Call)
            and _call_name(call) in functions
            and _contains_name(call, variable)
        )
        assert directly_canonicalized or helper_canonicalizes, (
            "each control-plane publication snapshot must use its canonical_bytes() "
            "encoding instead of the execution-receipt serializer"
        )


def test_cuda_build_uses_the_reviewed_link_time_driver_search_path_only() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")

    assert "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/opt/iql-cuda-driver-link" in source
    assert "CMAKE_BUILD_RPATH" not in source


def test_runtime_cuda_linkage_uses_the_strict_shared_parser() -> None:
    module = _module()

    assert _calls(module, "parse_cuda_driver_linkage")
    assert not [
        constant
        for constant in ast.walk(module)
        if isinstance(constant, ast.Constant)
        and isinstance(constant.value, str)
        and "libcuda" in constant.value
        and "\\s" in constant.value
    ], "the runner must not replace the shared linkage parser with a local regex"


def test_subject_hashing_uses_no_follow_descriptors_and_stable_fstat_metadata() -> None:
    module = _module()
    hasher = _function(module, "_sha256_file")

    os_open_calls = [
        call
        for call in ast.walk(hasher)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "os"
        and call.func.attr == "open"
    ]
    assert len(os_open_calls) == 1
    assert any(
        (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr == "O_NOFOLLOW"
        )
        or (isinstance(node, ast.Constant) and node.value == "O_NOFOLLOW")
        for node in ast.walk(os_open_calls[0])
    )
    assert len(_calls(hasher, "fstat")) >= 2, (
        "the hasher must compare descriptor metadata before and after reading"
    )
    assert not [
        node
        for node in ast.walk(hasher)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "open"
        and not (isinstance(node.func.value, ast.Name) and node.func.value.id == "os")
    ], "Path.open() would reintroduce a symlink race after validation"


def test_allocation_topology_is_reobserved_fail_closed_inside_each_subject_iteration() -> None:
    module = _module()
    entrypoint = _function(module, "matched_smoke_test")
    subject_loops = [
        node
        for node in ast.walk(entrypoint)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "MATCHED_SUBJECT_ORDER"
        and _calls(node, "_run_subject")
    ]
    assert len(subject_loops) == 1
    subject_loop = subject_loops[0]
    run_subject_call = _calls(subject_loop, "_run_subject")[0]

    observations: list[ast.Call] = []
    for call in ast.walk(subject_loop):
        if not isinstance(call, ast.Call) or call is run_subject_call:
            continue
        if getattr(call, "lineno", 0) >= run_subject_call.lineno:
            continue
        if "enumerate_matched_cuda_peer_topology" in _reachable_call_names(
            module,
            call,
        ):
            observations.append(call)
    assert observations, (
        "the allocation must be observed inside the BF16-to-Q3 loop so both "
        "subjects receive a just-in-time topology check"
    )

    check_scope = ast.Module(
        body=[
            statement
            for statement in subject_loop.body
            if getattr(statement, "lineno", 0) < run_subject_call.lineno
        ],
        type_ignores=[],
    )
    scopes = (
        check_scope,
        *(
            node
            for observation in observations
            for node in _reachable_function_nodes(module, observation)
        ),
    )
    assert any(
        isinstance(candidate, ast.If)
        and isinstance(candidate.test, ast.Compare)
        and _contains_raise(candidate)
        for scope in scopes
        for candidate in ast.walk(scope)
    ), "topology drift must stop the run before either subject server starts"


def test_probe_records_truthful_timings_and_checks_greedy_repeatability() -> None:
    module = _module()
    run_probe = _function(module, "_run_probe")
    assignments = _local_assignments(run_probe)
    trial_calls = _calls(run_probe, "MatchedProbeTrialEvidence")
    assert len(trial_calls) == 1
    trial_call = trial_calls[0]
    prompt_processing = _call_keyword(trial_call, "prompt_processing_ms")
    decode = _call_keyword(trial_call, "decode_ms")
    assert prompt_processing is not None and decode is not None
    assert _call_keyword(trial_call, "time_to_first_token_ms") is None
    assert "time_to_first_token_ms" not in ast.unparse(run_probe)

    def depends_on_token_ids(
        node: ast.AST,
        *,
        seen: frozenset[str] = frozenset(),
    ) -> bool:
        if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
            return depends_on_token_ids(
                assignments[node.id],
                seen=seen | {node.id},
            )
        if isinstance(node, ast.Attribute) and node.attr == "token_ids":
            return True
        return any(depends_on_token_ids(child, seen=seen) for child in ast.iter_child_nodes(node))

    assert any(
        isinstance(candidate, ast.If)
        and depends_on_token_ids(candidate.test)
        and _contains_raise(candidate)
        for candidate in ast.walk(run_probe)
    ), "two greedy trials must be compared instead of declaring repeatability"


def test_runner_binds_vocabulary_checks_to_the_reviewed_config() -> None:
    module = _module()
    scope = ast.Module(
        body=[
            _function(module, "_run_probe"),
            _function(module, "_run_subject"),
        ],
        type_ignores=[],
    )
    source = ast.unparse(scope)
    assert "200058" not in source
    assert "201024" not in source
    assert "bundle.config.output_vocabulary" in source
    assert "output_vocabulary.vocab_size" in source
    assert "output_vocabulary.unpadded_vocab_size" in source


def test_subject_receipts_are_committed_and_read_back_before_terminal_publication() -> None:
    module = _module()
    entrypoint = _function(module, "matched_smoke_test")
    run_subject_calls = _calls(entrypoint, "_run_subject")
    assert len(run_subject_calls) == 1
    run_subject_call = run_subject_calls[0]
    success_publications = [
        call
        for call in _calls(entrypoint, "_publish_terminal")
        if isinstance(_call_keyword(call, "outcome"), ast.Constant)
        and _call_keyword(call, "outcome").value == "success"  # type: ignore[union-attr]
    ]
    assert len(success_publications) == 1
    success_publication = success_publications[0]

    persistence_calls = [
        call
        for call in ast.walk(entrypoint)
        if isinstance(call, ast.Call)
        and run_subject_call.lineno < getattr(call, "lineno", 0) < success_publication.lineno
        and "_write_once" in _reachable_call_names(module, call)
        and (
            "_commit_and_verify" in _reachable_call_names(module, call)
            or {"commit", "reload"} <= _reachable_call_names(module, call)
        )
    ]
    assert persistence_calls, (
        "subject receipts must be immutable, committed, reloaded, and verified "
        "before terminal publication can begin"
    )


def test_outer_failure_receipt_is_guarded_by_publication_not_started_state() -> None:
    module = _module()
    entrypoint = _function(module, "matched_smoke_test")
    protected_regions = [
        candidate
        for candidate in ast.walk(entrypoint)
        if isinstance(candidate, ast.Try)
        and _calls(ast.Module(body=candidate.body, type_ignores=[]), "_run_subject")
        and _calls(ast.Module(body=candidate.body, type_ignores=[]), "_publish_terminal")
    ]
    assert protected_regions, (
        "the claimed attempt must wrap execution and terminal publication in one "
        "failure-recording boundary"
    )
    protected = min(
        protected_regions,
        key=lambda candidate: getattr(candidate, "end_lineno", candidate.lineno) - candidate.lineno,
    )
    assert protected.handlers
    handler_scope = ast.Module(
        body=list(protected.handlers),
        type_ignores=[],
    )
    reachable_nodes = _reachable_function_nodes(module, handler_scope)
    reachable_calls = {
        _call_name(call)
        for node in reachable_nodes
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
    }
    assert "MatchedFailureReceipt" in reachable_calls
    assert (
        "_publish_terminal" in reachable_calls
        or "build_matched_terminal_receipt_reference" in reachable_calls
    )
    assert any(
        isinstance(constant, ast.Constant) and constant.value == "failure"
        for node in reachable_nodes
        for constant in ast.walk(node)
    )

    guarded = False
    for node in reachable_nodes:
        for candidate in ast.walk(node):
            if not isinstance(candidate, ast.If):
                continue
            condition_markers = {
                marker
                for marker in (
                    *(name.id for name in ast.walk(candidate.test) if isinstance(name, ast.Name)),
                    *(
                        attribute.attr
                        for attribute in ast.walk(candidate.test)
                        if isinstance(attribute, ast.Attribute)
                    ),
                    *(
                        constant.value
                        for constant in ast.walk(candidate.test)
                        if isinstance(constant, ast.Constant) and isinstance(constant.value, str)
                    ),
                )
                if "publication" in marker or marker == "not_started"
            }
            if not condition_markers:
                continue
            body_calls = _reachable_call_names(
                module,
                ast.Module(body=candidate.body, type_ignores=[]),
            )
            else_calls = _reachable_call_names(
                module,
                ast.Module(body=candidate.orelse, type_ignores=[]),
            )
            failure_names = {"MatchedFailureReceipt", "matched_failure_receipt_sha256"}
            if bool(body_calls & failure_names) != bool(else_calls & failure_names):
                guarded = True
                break
        if guarded:
            break
    assert guarded, (
        "a failure receipt may be installed only while terminal publication is "
        "still not_started; installing or ambiguous publication must suppress it"
    )
