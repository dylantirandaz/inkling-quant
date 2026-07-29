"""CPU-only runtime tests for the matched Modal publication boundary.

The paid runner cannot be imported in a unit test because its module-level
code creates Modal objects and defines a CUDA image.  These tests parse the
runner and execute only selected top-level function and class definitions.
The executed function bodies are therefore the production bodies, while all
Volume operations use a small scripted test seam.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import secrets
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Literal, Protocol

import pytest
from pydantic import BaseModel

from inkling_quant_lab.gguf.inkling_matched_control import (
    MatchedPublicationSnapshot,
    MatchedTerminalReceiptReference,
    build_matched_terminal_receipt_reference,
    matched_publication_state_path,
    validate_matched_publication_state,
    validate_matched_publication_transition,
)

RUNNER_PATH = Path(__file__).resolve().parents[2] / "scripts/run_inkling_matched_modal.py"
RUN_ID = "inkling-matched-publication-test"
CLAIM_SHA256 = "c" * 64

_MISSING = object()
_ALL_INSTALLED = object()
_Definition = ast.FunctionDef | ast.ClassDef
_ReloadPolicy = Callable[[str, Mapping[str, bytes]], object]


def _runner_slice(*roots: str) -> dict[str, object]:
    """Execute a dependency-closed runner slice without importing the module."""

    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
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
        "BaseModel": BaseModel,
        "Literal": Literal,
        "Mapping": Mapping,
        "MatchedPublicationSnapshot": MatchedPublicationSnapshot,
        "MatchedTerminalReceiptReference": MatchedTerminalReceiptReference,
        "Path": Path,
        "Protocol": Protocol,
        "PurePosixPath": PurePosixPath,
        "Sequence": Sequence,
        "build_matched_terminal_receipt_reference": (build_matched_terminal_receipt_reference),
        "dataclass": dataclass,
        "hashlib": hashlib,
        "json": json,
        "matched_publication_state_path": matched_publication_state_path,
        "os": os,
        "secrets": secrets,
        "stat": stat,
        "suppress": suppress,
        "sys": sys,
        "validate_matched_publication_state": validate_matched_publication_state,
        "validate_matched_publication_transition": (validate_matched_publication_transition),
    }
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(RUNNER_PATH), "exec"), namespace)
    return namespace


class _ScriptedVolume:
    """Expose one scripted mounted view after each successful reload."""

    def __init__(
        self,
        *,
        commit_outcomes: Sequence[BaseException | None] = (),
        reload_outcomes: Sequence[
            BaseException | Mapping[str, object] | _ReloadPolicy | object
        ] = (),
    ) -> None:
        self._commit_outcomes = tuple(commit_outcomes)
        self._reload_outcomes = tuple(reload_outcomes)
        self.commit_calls = 0
        self.reload_calls = 0
        self.current_view: Mapping[str, object] | _ReloadPolicy | object | None = None
        self.events: list[tuple[str, int]] = []

    def commit(self) -> None:
        self.commit_calls += 1
        self.events.append(("commit", self.commit_calls))
        outcome = (
            self._commit_outcomes[self.commit_calls - 1]
            if self.commit_calls <= len(self._commit_outcomes)
            else None
        )
        if isinstance(outcome, BaseException):
            raise outcome

    def reload(self) -> None:
        self.reload_calls += 1
        self.events.append(("reload", self.reload_calls))
        outcome = (
            self._reload_outcomes[self.reload_calls - 1]
            if self.reload_calls <= len(self._reload_outcomes)
            else _ALL_INSTALLED
        )
        if isinstance(outcome, BaseException):
            raise outcome
        self.current_view = outcome


@dataclass
class _ScriptedIo:
    installed: dict[str, bytes]
    installs: list[tuple[str, bytes]]
    reads: list[tuple[int, str]]


def _install_scripted_io(
    namespace: dict[str, object],
    *,
    mount: Path,
    volume: _ScriptedVolume,
    initial: Mapping[str, bytes] | None = None,
) -> _ScriptedIo:
    """Install deterministic mounted-path I/O into one isolated runner slice."""

    installed = dict(initial or {})
    installs: list[tuple[str, bytes]] = []
    reads: list[tuple[int, str]] = []
    collision = namespace["_PublicationCollisionError"]
    assert isinstance(collision, type)

    def relative(path: Path) -> str:
        return path.relative_to(mount).as_posix()

    def write_once(path: Path, payload: bytes) -> None:
        path_text = relative(path)
        existing = installed.get(path_text)
        if existing is not None and existing != payload:
            raise collision("test seam refused to replace immutable bytes")
        installed[path_text] = payload
        installs.append((path_text, payload))

    def read_regular_bytes(path: Path, *, maximum_bytes: int) -> bytes:
        path_text = relative(path)
        reads.append((volume.reload_calls, path_text))
        view = volume.current_view
        if view is None:
            raise AssertionError("mounted evidence was read before a successful reload")
        if view is _ALL_INSTALLED:
            observed: object = installed.get(path_text, _MISSING)
        elif callable(view):
            observed = view(path_text, installed)
        else:
            assert isinstance(view, Mapping)
            observed = view.get(path_text, _MISSING)
        if observed is _MISSING:
            raise FileNotFoundError(path)
        if isinstance(observed, BaseException):
            raise observed
        assert isinstance(observed, bytes)
        if len(observed) > maximum_bytes:
            raise RuntimeError("mounted evidence exceeds its byte limit")
        return observed

    namespace.update(
        {
            "EVIDENCE_MOUNT": mount,
            "_read_regular_bytes": read_regular_bytes,
            "_write_once": write_once,
            "evidence_volume": volume,
        }
    )
    return _ScriptedIo(installed=installed, installs=installs, reads=reads)


def _claim() -> SimpleNamespace:
    return SimpleNamespace(
        run_id=RUN_ID,
        claim_sha256=lambda: CLAIM_SHA256,
    )


def _terminal_receipt(outcome: Literal["success", "failure"]) -> dict[str, object]:
    schema_version, status = {
        "success": ("inkling-matched-rollup-v1", "passed"),
        "failure": ("inkling-matched-failure-v1", "failed"),
    }[outcome]
    return {
        "schema_version": schema_version,
        "status": status,
        "stage": "matched_smoke",
        "run_id": RUN_ID,
        "prompt_text_recorded": False,
        "output_text_recorded": False,
        "receipt_sha256": ("a" if outcome == "success" else "b") * 64,
    }


def _required_files() -> dict[str, bytes]:
    return {
        f"runs/{RUN_ID}/control/attempt-claims/{CLAIM_SHA256}.json": b"claim",
        f"runs/{RUN_ID}/control/attempt-acknowledgements/{'d' * 64}.json": b"ack",
    }


def test_atomic_install_is_exact_idempotent_and_collision_safe(
    tmp_path: Path,
) -> None:
    namespace = _runner_slice("_atomic_install_exact")
    namespace["EVIDENCE_MOUNT"] = tmp_path
    install = namespace["_atomic_install_exact"]
    collision = namespace["_PublicationCollisionError"]
    assert callable(install)
    assert isinstance(collision, type)

    destination = tmp_path / "runs" / RUN_ID / "receipt.json"
    payload = b'{"exact":true}'
    install(destination, payload)
    inode = destination.stat().st_ino

    install(destination, payload)
    assert destination.read_bytes() == payload
    assert destination.stat().st_ino == inode

    with pytest.raises(collision):
        install(destination, b'{"different":true}')
    assert destination.read_bytes() == payload
    assert not tuple(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_atomic_install_rejects_symlink_paths(tmp_path: Path) -> None:
    namespace = _runner_slice("_atomic_install_exact")
    namespace["EVIDENCE_MOUNT"] = tmp_path
    install = namespace["_atomic_install_exact"]
    unknown = namespace["_EvidenceStateUnknownError"]
    assert callable(install)
    assert isinstance(unknown, type)

    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "runs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink or non-directory ancestor"):
        install(tmp_path / "runs" / RUN_ID / "receipt.json", b"exact")
    assert not (outside / RUN_ID / "receipt.json").exists()

    (tmp_path / "runs").unlink()
    target = tmp_path / "target.json"
    target.write_bytes(b"target")
    destination = tmp_path / "terminal.json"
    destination.symlink_to(target)
    with pytest.raises(unknown, match="cannot be validated"):
        install(destination, b"exact")
    assert target.read_bytes() == b"target"


@pytest.mark.parametrize(
    "fault",
    ("rename_after_apply", "post_rename_readback", "directory_fsync"),
)
def test_atomic_install_post_apply_faults_are_unknown(
    tmp_path: Path,
    fault: str,
) -> None:
    namespace = _runner_slice("_atomic_install_exact")
    namespace["EVIDENCE_MOUNT"] = tmp_path
    install = namespace["_atomic_install_exact"]
    unknown = namespace["_EvidenceStateUnknownError"]
    assert callable(install)
    assert isinstance(unknown, type)

    destination = tmp_path / "runs" / RUN_ID / f"{fault}.json"
    payload = b'{"installed":"possibly"}'
    if fault == "rename_after_apply":
        rename = namespace["_rename_noreplace"]
        assert callable(rename)

        def rename_then_interrupt(source: Path, target: Path) -> None:
            rename(source, target)
            raise KeyboardInterrupt("interrupted after rename")

        namespace["_rename_noreplace"] = rename_then_interrupt
    elif fault == "post_rename_readback":

        def fail_readback(_path: Path, *, maximum_bytes: int) -> bytes:
            del maximum_bytes
            raise OSError("post-rename readback failed")

        namespace["_read_regular_bytes"] = fail_readback
    else:

        def fail_directory_fsync(_path: Path) -> None:
            raise OSError("directory fsync failed")

        namespace["_fsync_directory"] = fail_directory_fsync

    with pytest.raises(unknown):
        install(destination, payload)
    assert destination.read_bytes() == payload
    assert not tuple(destination.parent.glob(f".{destination.name}.*.tmp"))


@pytest.mark.parametrize(
    "relative",
    (
        "/absolute.json",
        "../outside.json",
        "runs//receipt.json",
        "runs/./receipt.json",
        "runs\\receipt.json",
    ),
)
def test_commit_and_verify_rejects_invalid_paths_before_volume_use(
    tmp_path: Path,
    relative: str,
) -> None:
    namespace = _runner_slice("_commit_and_verify")
    volume = _ScriptedVolume()
    _install_scripted_io(namespace, mount=tmp_path, volume=volume)
    commit_and_verify = namespace["_commit_and_verify"]
    assert callable(commit_and_verify)

    with pytest.raises((ValueError, RuntimeError)):
        commit_and_verify({relative: b"exact"})
    assert volume.events == []


@pytest.mark.parametrize(
    "commit_outcome",
    (
        None,
        OSError("commit response was lost"),
        KeyboardInterrupt("commit was interrupted after apply"),
    ),
)
def test_commit_and_verify_accepts_one_exact_reloaded_view(
    tmp_path: Path,
    commit_outcome: BaseException | None,
) -> None:
    namespace = _runner_slice("_commit_and_verify")
    relative = f"runs/{RUN_ID}/receipt.json"
    payload = b'{"exact":true}'
    volume = _ScriptedVolume(
        commit_outcomes=(commit_outcome,),
        reload_outcomes=(_ALL_INSTALLED,),
    )
    io_state = _install_scripted_io(
        namespace,
        mount=tmp_path,
        volume=volume,
        initial={relative: payload},
    )
    commit_and_verify = namespace["_commit_and_verify"]
    assert callable(commit_and_verify)

    commit_and_verify({relative: payload})

    assert volume.events == [("commit", 1), ("reload", 1)]
    assert io_state.reads == [(1, relative)]
    assert io_state.installs == []


def test_commit_and_verify_reinstalls_only_missing_files_once(
    tmp_path: Path,
) -> None:
    namespace = _runner_slice("_commit_and_verify")
    first = f"runs/{RUN_ID}/a.json"
    second = f"runs/{RUN_ID}/b.json"
    expected = {first: b"a", second: b"b"}
    volume = _ScriptedVolume(
        commit_outcomes=(OSError("first commit did not apply"), None),
        reload_outcomes=(
            {first: _MISSING, second: _ALL_INSTALLED},
            _ALL_INSTALLED,
        ),
    )

    def first_view(path: str, installed: Mapping[str, bytes]) -> object:
        if path == first:
            return _MISSING
        return installed.get(path, _MISSING)

    volume._reload_outcomes = (first_view, _ALL_INSTALLED)
    io_state = _install_scripted_io(
        namespace,
        mount=tmp_path,
        volume=volume,
        initial=expected,
    )
    commit_and_verify = namespace["_commit_and_verify"]
    assert callable(commit_and_verify)

    commit_and_verify(expected)

    assert volume.events == [
        ("commit", 1),
        ("reload", 1),
        ("commit", 2),
        ("reload", 2),
    ]
    assert io_state.installs == [(first, expected[first])]
    assert io_state.reads == [
        (1, first),
        (1, second),
        (2, first),
        (2, second),
    ]


@pytest.mark.parametrize(
    "views",
    (
        (
            OSError("first reload failed"),
            KeyboardInterrupt("second reload failed"),
        ),
        (
            {"runs/inkling-matched-publication-test/receipt.json": _MISSING},
            {"runs/inkling-matched-publication-test/receipt.json": _MISSING},
        ),
    ),
)
def test_commit_and_verify_two_unproved_cycles_are_unknown(
    tmp_path: Path,
    views: Sequence[object],
) -> None:
    namespace = _runner_slice("_commit_and_verify")
    relative = f"runs/{RUN_ID}/receipt.json"
    payload = b"exact"
    volume = _ScriptedVolume(reload_outcomes=views)
    _install_scripted_io(
        namespace,
        mount=tmp_path,
        volume=volume,
        initial={relative: payload},
    )
    commit_and_verify = namespace["_commit_and_verify"]
    unknown = namespace["_EvidenceStateUnknownError"]
    assert callable(commit_and_verify)
    assert isinstance(unknown, type)

    with pytest.raises(unknown):
        commit_and_verify({relative: payload})
    assert volume.commit_calls == 2
    assert volume.reload_calls == 2


@pytest.mark.parametrize(
    "observed",
    (
        b"other",
        OSError("mounted read failed"),
        KeyboardInterrupt("mounted read was interrupted"),
    ),
)
def test_commit_and_verify_never_accepts_an_unsafe_reloaded_file(
    tmp_path: Path,
    observed: bytes | BaseException,
) -> None:
    namespace = _runner_slice("_commit_and_verify")
    relative = f"runs/{RUN_ID}/receipt.json"
    payload = b"exact"
    volume = _ScriptedVolume(reload_outcomes=({relative: observed},))
    io_state = _install_scripted_io(
        namespace,
        mount=tmp_path,
        volume=volume,
        initial={relative: payload},
    )
    commit_and_verify = namespace["_commit_and_verify"]
    collision = namespace["_PublicationCollisionError"]
    unknown = namespace["_EvidenceStateUnknownError"]
    assert callable(commit_and_verify)
    assert isinstance(collision, type)
    assert isinstance(unknown, type)

    expected_error = collision if isinstance(observed, bytes) else unknown
    with pytest.raises(expected_error):
        commit_and_verify({relative: payload})
    assert io_state.installs == []
    assert volume.events == [("commit", 1), ("reload", 1)]


@pytest.mark.parametrize("outcome", ("success", "failure"))
@pytest.mark.parametrize(
    "first_commit",
    (
        None,
        OSError("terminal commit response was lost"),
        KeyboardInterrupt("terminal commit was interrupted after apply"),
    ),
)
def test_terminal_publication_confirms_exact_success_and_failure_bytes(
    tmp_path: Path,
    outcome: Literal["success", "failure"],
    first_commit: BaseException | None,
) -> None:
    namespace = _runner_slice("_new_publication_tracker", "_publish_terminal")
    volume = _ScriptedVolume(
        commit_outcomes=(first_commit, None),
        reload_outcomes=(_ALL_INSTALLED, _ALL_INSTALLED),
    )
    io_state = _install_scripted_io(namespace, mount=tmp_path, volume=volume)
    new_tracker = namespace["_new_publication_tracker"]
    publish = namespace["_publish_terminal"]
    execution_bytes = namespace["_execution_json_bytes"]
    assert callable(new_tracker)
    assert callable(publish)
    assert callable(execution_bytes)

    claim = _claim()
    tracker = new_tracker(claim)
    receipt = _terminal_receipt(outcome)
    publication = publish(
        claim=claim,
        publication_tracker=tracker,
        terminal_receipt=receipt,
        outcome=outcome,
        required_files=_required_files(),
    )

    assert publication.status == "confirmed"
    assert publication.cycle == 1
    assert tracker.snapshot == publication
    assert publication.terminal_receipt is not None
    assert publication.terminal_receipt.outcome == outcome
    terminal_payload = io_state.installed[publication.terminal_receipt.path]
    assert terminal_payload == execution_bytes(receipt)
    assert not terminal_payload.endswith(b"\n")
    control_payloads = [
        payload
        for path, payload in io_state.installed.items()
        if "/control/publication-states/" in path
    ]
    assert control_payloads
    assert all(
        payload.endswith(b"\n") and not payload.endswith(b"\n\n") for payload in control_payloads
    )
    assert volume.commit_calls == 2
    assert volume.reload_calls == 2


def test_terminal_publication_recovers_one_missing_terminal_view(
    tmp_path: Path,
) -> None:
    namespace = _runner_slice("_new_publication_tracker", "_publish_terminal")

    def terminal_missing(path: str, installed: Mapping[str, bytes]) -> object:
        if "/terminal/" in path:
            return _MISSING
        return installed.get(path, _MISSING)

    volume = _ScriptedVolume(
        reload_outcomes=(
            terminal_missing,
            _ALL_INSTALLED,
            _ALL_INSTALLED,
        )
    )
    _install_scripted_io(namespace, mount=tmp_path, volume=volume)
    new_tracker = namespace["_new_publication_tracker"]
    publish = namespace["_publish_terminal"]
    assert callable(new_tracker)
    assert callable(publish)

    claim = _claim()
    tracker = new_tracker(claim)
    publication = publish(
        claim=claim,
        publication_tracker=tracker,
        terminal_receipt=_terminal_receipt("success"),
        outcome="success",
        required_files=_required_files(),
    )

    assert publication.status == "confirmed"
    assert publication.cycle == 2
    assert volume.commit_calls == 3
    assert volume.reload_calls == 3


def test_terminal_publication_unknown_state_rejects_a_conflicting_terminal(
    tmp_path: Path,
) -> None:
    namespace = _runner_slice("_new_publication_tracker", "_publish_terminal")
    volume = _ScriptedVolume(
        reload_outcomes=(
            OSError("first terminal reload failed"),
            KeyboardInterrupt("second terminal reload failed"),
            _ALL_INSTALLED,
        )
    )
    io_state = _install_scripted_io(namespace, mount=tmp_path, volume=volume)
    new_tracker = namespace["_new_publication_tracker"]
    publish = namespace["_publish_terminal"]
    unknown = namespace["_PublicationUnknownError"]
    assert callable(new_tracker)
    assert callable(publish)
    assert isinstance(unknown, type)

    claim = _claim()
    tracker = new_tracker(claim)
    with pytest.raises(unknown):
        publish(
            claim=claim,
            publication_tracker=tracker,
            terminal_receipt=_terminal_receipt("success"),
            outcome="success",
            required_files=_required_files(),
        )

    assert tracker.snapshot.status == "unknown"
    assert tracker.snapshot.failure_receipt_publication_allowed is False
    writes_before_retry = tuple(io_state.installs)
    events_before_retry = tuple(volume.events)
    with pytest.raises(RuntimeError, match="already started"):
        publish(
            claim=claim,
            publication_tracker=tracker,
            terminal_receipt=_terminal_receipt("failure"),
            outcome="failure",
            required_files=_required_files(),
        )
    assert tuple(io_state.installs) == writes_before_retry
    assert tuple(volume.events) == events_before_retry


def test_preinstall_failure_keeps_failure_terminal_available(tmp_path: Path) -> None:
    namespace = _runner_slice("_new_publication_tracker", "_publish_terminal")
    volume = _ScriptedVolume(
        reload_outcomes=(_ALL_INSTALLED, _ALL_INSTALLED),
    )
    _install_scripted_io(namespace, mount=tmp_path, volume=volume)
    new_tracker = namespace["_new_publication_tracker"]
    publish = namespace["_publish_terminal"]
    write_once = namespace["_write_once"]
    assert callable(new_tracker)
    assert callable(publish)
    assert callable(write_once)

    claim = _claim()
    tracker = new_tracker(claim)

    def fail_before_install(_path: Path, _payload: bytes) -> None:
        raise OSError("state-zero installation failed before apply")

    namespace["_write_once"] = fail_before_install
    with pytest.raises(OSError, match="before apply"):
        publish(
            claim=claim,
            publication_tracker=tracker,
            terminal_receipt=_terminal_receipt("success"),
            outcome="success",
            required_files=_required_files(),
        )
    assert tracker.snapshot.status == "not_started"
    assert volume.events == []

    namespace["_write_once"] = write_once
    publication = publish(
        claim=claim,
        publication_tracker=tracker,
        terminal_receipt=_terminal_receipt("failure"),
        outcome="failure",
        required_files=_required_files(),
    )
    assert publication.status == "confirmed"
    assert publication.terminal_receipt is not None
    assert publication.terminal_receipt.outcome == "failure"


def test_confirmed_terminal_requires_durable_confirmation_snapshot(
    tmp_path: Path,
) -> None:
    namespace = _runner_slice("_new_publication_tracker", "_publish_terminal")
    volume = _ScriptedVolume(reload_outcomes=(_ALL_INSTALLED,))
    _install_scripted_io(namespace, mount=tmp_path, volume=volume)
    new_tracker = namespace["_new_publication_tracker"]
    publish = namespace["_publish_terminal"]
    commit_and_verify = namespace["_commit_and_verify"]
    assert callable(new_tracker)
    assert callable(publish)
    assert callable(commit_and_verify)

    def fail_confirmed_snapshot(_files: Mapping[str, bytes]) -> None:
        raise KeyboardInterrupt("confirmed snapshot diagnostic failed")

    namespace["_commit_and_verify"] = fail_confirmed_snapshot
    claim = _claim()
    tracker = new_tracker(claim)
    unknown = namespace["_PublicationUnknownError"]

    with pytest.raises(unknown):
        publish(
            claim=claim,
            publication_tracker=tracker,
            terminal_receipt=_terminal_receipt("success"),
            outcome="success",
            required_files=_required_files(),
        )

    assert tracker.snapshot.status == "unknown"
    assert volume.events == [("commit", 1), ("reload", 1)]
