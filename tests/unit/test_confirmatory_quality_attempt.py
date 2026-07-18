"""Append-only execution contracts for confirmatory quality attempts."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

import pytest

from scripts import run_confirmatory_quality_attempt as attempt_runner

pytestmark = pytest.mark.unit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment() -> dict[str, Any]:
    return {
        "hardware_label": attempt_runner.HARDWARE_LABEL,
        "python": {"implementation": "CPython", "version": "3.12.13"},
        "platform": {"system": "Darwin", "release": "24.3.0", "machine": "arm64"},
        "software": {
            "inkling-quant-lab": "0.1.0",
            "torch": "2.13.0",
            "transformers": "5.12.1",
            "tokenizers": "0.22.2",
            "numpy": "2.2.6",
            "safetensors": "0.8.0",
            "huggingface-hub": "1.23.0",
        },
        "torch_cpu": {
            "intraop_threads": 4,
            "interop_threads": 8,
            "required_supported_engine": "qnnpack",
        },
        "offline_environment": dict(attempt_runner.OFFLINE_ENVIRONMENT),
        "python_executable": {
            "invocation_path": "/fixture/.venv/bin/python",
            "resolved_path": "/fixture/base-python",
            "sha256": "7" * 64,
            "size_bytes": 123,
        },
        "virtual_environment": {
            "pyvenv_cfg_path": ".venv/pyvenv.cfg",
            "sha256": "8" * 64,
            "size_bytes": 45,
        },
    }


def _thaw(directory: Path) -> None:
    if not directory.exists() or directory.is_symlink():
        return
    directory.chmod(0o755)
    for path in directory.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o755)
        elif path.is_file():
            path.chmod(0o644)


def _valid_record(ordinal: Literal[1, 2]) -> dict[str, Any]:
    return {
        "schema_version": "public-moe-native-linear-quality-v2",
        "protocol": {
            "confirmatory": {
                "definition": {"protocol_id": attempt_runner.PROTOCOL_ID},
            }
        },
        "confirmatory_status": {
            "status": "provisional_until_2_clean_executions",
            "confirmatory_claim_ready": False,
            "required_clean_process_executions": 2,
            "execution_ordinal": ordinal,
            "within_execution_noninferiority_passed": True,
            "within_execution_decision_is_overall_confirmatory_claim": False,
            "pair_verification_required": True,
        },
        "dataset": {"perplexity_sample_ids": ["story-000147"]},
        "candidates": [
            {"generation_retention": {"samples": [{"candidate_output_sha256": "a" * 64}]}}
        ],
    }


def _sealed_attempt_one(root: Path) -> Path:
    attempt = root / attempt_runner.ATTEMPT_ROOT / "attempt-1"
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "start.json").write_text("{}\n", encoding="utf-8")
    (attempt / "stdout.log").write_bytes(b"")
    (attempt / "stderr.log").write_bytes(b"")
    (attempt / "record.json").write_text(json.dumps(_valid_record(1)), encoding="utf-8")
    (attempt / "completion.json").write_text(
        json.dumps(
            {
                "schema_version": "confirmatory-quality-attempt-completion-v1",
                "status": "attempt_execution_complete",
                "ordinal": 1,
                "runner_exit_code": 0,
                "confirmatory_claim_ready": False,
                "record": {"valid": True},
            }
        ),
        encoding="utf-8",
    )
    for path in attempt.iterdir():
        path.chmod(0o444)
    attempt.chmod(0o555)
    return attempt


def _project_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, dict[str, Any]]:
    monkeypatch.setattr(
        attempt_runner,
        "_interpreter_isolation_facts",
        lambda: {
            "no_user_site": True,
            "safe_path": True,
            "dont_write_bytecode": True,
            "pycache_prefix": "/dev/null",
        },
    )
    for name in attempt_runner.FORBIDDEN_AMBIENT_IMPORT_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    root = tmp_path.absolute()
    for relative in attempt_runner.REQUIRED_BOUND_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"bound fixture: {relative.as_posix()}\n", encoding="utf-8")
    extra_source = root / "src/inkling_quant_lab/nested/extra_fixture.py"
    extra_source.parent.mkdir(parents=True, exist_ok=True)
    extra_source.write_text("BOUND = True\n", encoding="utf-8")
    dataset = root / "TinyStories-valid.txt"
    dataset.write_text("first<|endoftext|>second<|endoftext|>third", encoding="utf-8")
    dataset_sha256 = _sha256(dataset)
    monkeypatch.setattr(attempt_runner, "OFFICIAL_DATASET_SHA256", dataset_sha256)
    monkeypatch.setattr(attempt_runner, "OFFICIAL_DATASET_SIZE_BYTES", dataset.stat().st_size)
    monkeypatch.setattr(attempt_runner, "OFFICIAL_DATASET_STORY_COUNT", 3)
    scientific = {
        "baseline": attempt_runner.BASELINE_CONFIG_SHA256,
        "candidate": attempt_runner.CANDIDATE_CONFIG_SHA256,
        "protocol_definition": "9" * 64,
    }
    environment = _environment()
    monkeypatch.setattr(
        attempt_runner,
        "_resolved_scientific_bindings",
        lambda project_root: dict(scientific),
    )
    local_imports = {
        "inkling_quant_lab": {
            "source_path": "src/inkling_quant_lab/__init__.py",
            "source_sha256": "5" * 64,
            "cached_path": "/dev/null/inkling_quant_lab/__init__.pyc",
            "cached_exists": False,
        },
        "scripts": {
            "source_path": "scripts/__init__.py",
            "source_sha256": "6" * 64,
            "cached_path": "/dev/null/scripts/__init__.pyc",
            "cached_exists": False,
        },
    }
    monkeypatch.setattr(
        attempt_runner,
        "_local_import_provenance",
        lambda project_root, current_files: dict(local_imports),
    )
    monkeypatch.setattr(
        attempt_runner, "_environment_contract", lambda project_root: dict(environment)
    )
    files = attempt_runner._required_file_bindings(root)
    preregistration: dict[str, Any] = {
        "schema_version": "confirmatory-quality-preregistration-v1",
        "status": "locked_before_execution",
        "protocol_id": attempt_runner.PROTOCOL_ID,
        "created_at_utc": "2026-07-16T22:30:00+00:00",
        "outcomes": {
            "holdout_model_forward_executed": False,
            "holdout_outcomes_inspected": False,
        },
        "bindings": {
            "files": files,
            "resolved_config_sha256": {
                "baseline": scientific["baseline"],
                "candidate": scientific["candidate"],
            },
            "protocol_definition_sha256": scientific["protocol_definition"],
            "dataset": {
                "file_sha256": dataset_sha256,
                "size_bytes": dataset.stat().st_size,
                "story_count": 3,
            },
        },
        "environment_contract": environment,
        "planned_attempts": [
            {
                "ordinal": ordinal,
                "directory": (attempt_runner.ATTEMPT_ROOT / f"attempt-{ordinal}").as_posix(),
                "record_path": (
                    attempt_runner.ATTEMPT_ROOT / f"attempt-{ordinal}" / "record.json"
                ).as_posix(),
            }
            for ordinal in (1, 2)
        ],
        "primary_analysis": {"decision": "prospective fixture"},
        "claim_boundary": ["unit fixture"],
    }
    preregistration_path = root / attempt_runner.PROJECT_PREREGISTRATION
    preregistration_path.parent.mkdir(parents=True, exist_ok=True)
    preregistration_path.write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root, dataset, preregistration_path, preregistration


def _output_path(command: list[str], cwd: Path) -> Path:
    return cwd / Path(command[command.index("--output") + 1])


def test_entry_guard_precedes_all_local_package_imports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "load_config" not in vars(attempt_runner)
    monkeypatch.setattr(attempt_runner.sys, "pycache_prefix", None)
    monkeypatch.setattr(attempt_runner.sys, "dont_write_bytecode", False)

    with pytest.raises(RuntimeError, match="PYTHONPYCACHEPREFIX=/dev/null"):
        attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=tmp_path / "not-read.txt",
            preregistration_path=attempt_runner.PROJECT_PREREGISTRATION,
            project_root=tmp_path,
        )

    assert not (tmp_path / "artifacts").exists()


def test_entry_guard_rejects_ambient_import_path_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        attempt_runner,
        "_interpreter_isolation_facts",
        lambda: {
            "no_user_site": True,
            "safe_path": True,
            "dont_write_bytecode": True,
            "pycache_prefix": "/dev/null",
        },
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    with pytest.raises(RuntimeError, match="forbid ambient local import path"):
        attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=tmp_path / "not-read.txt",
            preregistration_path=attempt_runner.PROJECT_PREREGISTRATION,
            project_root=tmp_path,
        )

    assert not (tmp_path / "artifacts").exists()


def test_symlinked_venv_invocation_is_launched_while_resolved_target_is_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.absolute()
    venv_bin = root / ".venv/bin"
    venv_bin.mkdir(parents=True)
    resolved = root / "base-python"
    resolved.write_bytes(b"synthetic resolved Python executable\n")
    invocation = venv_bin / "python"
    invocation.symlink_to(resolved)
    monkeypatch.setattr(attempt_runner.sys, "executable", str(invocation))

    fact = attempt_runner._python_executable_fact(root)
    command = attempt_runner._command(
        ordinal=1,
        dataset=root / "TinyStories-valid.txt",
        relative_record=attempt_runner.ATTEMPT_ROOT / "attempt-1/record.json",
    )

    assert fact == {
        "invocation_path": str(invocation),
        "resolved_path": str(resolved),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    assert command[0] == fact["invocation_path"] == str(invocation)
    assert command[0] != fact["resolved_path"]


def test_success_writes_start_before_no_shell_offline_execution_and_freezes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dataset, preregistration, _ = _project_fixture(tmp_path, monkeypatch)
    observed: dict[str, Any] = {}

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        observed["command"] = command
        observed.update(kwargs)
        cwd = cast(Path, kwargs["cwd"])
        attempt = cwd / attempt_runner.ATTEMPT_ROOT / "attempt-1"
        assert (attempt / "start.json").is_file()
        assert (attempt / "stdout.log").is_file()
        assert (attempt / "stderr.log").is_file()
        cast(Any, kwargs["stdout"]).write(b"evaluator stdout\n")
        cast(Any, kwargs["stderr"]).write(b"evaluator stderr\n")
        _output_path(command, cwd).write_text(json.dumps(_valid_record(1)), encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(attempt_runner.subprocess, "run", fake_run)
    attempt = root / attempt_runner.ATTEMPT_ROOT / "attempt-1"
    try:
        result = attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=dataset,
            preregistration_path=preregistration,
            project_root=root,
        )

        assert result.status == "attempt_execution_complete"
        assert result.exit_code == 0
        command = observed["command"]
        assert command[:5] == [
            sys.executable,
            "-s",
            "-P",
            "-B",
            attempt_runner.EVALUATOR.as_posix(),
        ]
        assert command[5:7] == [
            attempt_runner.BASELINE_CONFIG.as_posix(),
            attempt_runner.CANDIDATE_CONFIG.as_posix(),
        ]
        assert command[command.index("--execution-ordinal") + 1] == "1"
        assert (
            command[command.index("--output") + 1]
            == (attempt_runner.ATTEMPT_ROOT / "attempt-1" / "record.json").as_posix()
        )
        assert observed["shell"] is False
        assert observed["check"] is False
        assert observed["stdin"] is subprocess.DEVNULL
        for name, value in attempt_runner.OFFLINE_ENVIRONMENT.items():
            assert observed["env"][name] == value
        for name in attempt_runner.FORBIDDEN_AMBIENT_IMPORT_ENVIRONMENT:
            assert name not in observed["env"]
        completion = json.loads((attempt / "completion.json").read_text(encoding="utf-8"))
        assert completion["status"] == "attempt_execution_complete"
        assert completion["record"]["valid"] is True
        assert completion["record"]["sha256"] == _sha256(attempt / "record.json")
        assert completion["stdout"]["sha256"] == _sha256(attempt / "stdout.log")
        assert completion["stderr"]["size_bytes"] == (attempt / "stderr.log").stat().st_size
        assert completion["confirmatory_claim_ready"] is False
        start = json.loads((attempt / "start.json").read_text(encoding="utf-8"))
        assert start["preregistration"]["sha256"] == _sha256(preregistration)
        assert start["dataset"]["sha256"] == _sha256(dataset)
        assert start["selected_child_environment"] == attempt_runner.OFFLINE_ENVIRONMENT
        assert start["source_import_contract"] == {
            "python_flags": ["-s", "-P", "-B"],
            "sys_flags": {
                "no_user_site": True,
                "safe_path": True,
                "dont_write_bytecode": True,
            },
            "sys_pycache_prefix": "/dev/null",
            "forbidden_ambient_import_environment_absent": [
                "PYTHONPATH",
                "PYTHONHOME",
                "PYTHONUSERBASE",
            ],
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": "/dev/null",
                "PYTHONNOUSERSITE": "1",
            },
            "local_project_imports_deferred_until_after_entry_guard": True,
            "verified_local_modules": {
                "inkling_quant_lab": {
                    "source_path": "src/inkling_quant_lab/__init__.py",
                    "source_sha256": "5" * 64,
                    "cached_path": "/dev/null/inkling_quant_lab/__init__.pyc",
                    "cached_exists": False,
                },
                "scripts": {
                    "source_path": "scripts/__init__.py",
                    "source_sha256": "6" * 64,
                    "cached_path": "/dev/null/scripts/__init__.pyc",
                    "cached_exists": False,
                },
            },
        }
        assert observed["env"]["PYTHONPYCACHEPREFIX"] == "/dev/null"
        assert observed["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert "src/inkling_quant_lab/nested/extra_fixture.py" in start["file_sha256"]
        assert "configs/models/hf_stories15m_moe.yaml" in start["file_sha256"]
        assert "configs/quantization/native_dynamic_int8.yaml" in start["file_sha256"]
        assert {"AGENTS.md", "SPEC.md", "SDD.md", "TDD.md"}.issubset(start["file_sha256"])
        assert list(start["file_sha256"]) == sorted(start["file_sha256"])
        assert stat.S_IMODE(attempt.stat().st_mode) == 0o555
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o444
            for path in attempt.iterdir()
            if path.is_file()
        )
    finally:
        _thaw(attempt)


def test_evaluator_failure_is_retained_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dataset, preregistration, _ = _project_fixture(tmp_path, monkeypatch)

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        del command
        cast(Any, kwargs["stderr"]).write(b"evaluator failed\n")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(attempt_runner.subprocess, "run", fake_run)
    attempt = root / attempt_runner.ATTEMPT_ROOT / "attempt-1"
    try:
        result = attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=dataset,
            preregistration_path=preregistration,
            project_root=root,
        )
        completion = json.loads((attempt / "completion.json").read_text(encoding="utf-8"))
        assert result.exit_code == 1
        assert result.status == "evaluator_failed"
        assert completion["subprocess_launched"] is True
        assert completion["subprocess_returncode"] == 7
        assert completion["record"]["exists"] is False
        assert (attempt / "start.json").is_file()
        assert (attempt / "stderr.log").read_bytes() == b"evaluator failed\n"
    finally:
        _thaw(attempt)


def test_stale_preregistration_binding_is_frozen_and_never_launched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dataset, preregistration_path, preregistration = _project_fixture(tmp_path, monkeypatch)
    preregistration["bindings"]["files"][attempt_runner.EVALUATOR.as_posix()] = "0" * 64
    preregistration_path.write_text(json.dumps(preregistration), encoding="utf-8")
    calls: list[bool] = []
    monkeypatch.setattr(
        attempt_runner.subprocess,
        "run",
        lambda *args, **kwargs: calls.append(True),
    )
    attempt = root / attempt_runner.ATTEMPT_ROOT / "attempt-1"
    try:
        result = attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=dataset,
            preregistration_path=preregistration_path,
            project_root=root,
        )
        completion = json.loads((attempt / "completion.json").read_text(encoding="utf-8"))
        assert result.status == "failed_preflight"
        assert result.exit_code == 1
        assert calls == []
        assert completion["subprocess_launched"] is False
        assert completion["preflight_complete"] is False
        assert completion["failure"]["type"] == "PreflightError"
        assert "stale preregistered file binding" in (attempt / "stderr.log").read_text()
        assert stat.S_IMODE(attempt.stat().st_mode) == 0o555
    finally:
        _thaw(attempt)


def test_python_source_added_after_preregistration_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dataset, preregistration, _ = _project_fixture(tmp_path, monkeypatch)
    added = root / "src/inkling_quant_lab/new_after_lock.py"
    added.write_text("NEW_AFTER_LOCK = True\n", encoding="utf-8")
    calls: list[bool] = []
    monkeypatch.setattr(
        attempt_runner.subprocess,
        "run",
        lambda *args, **kwargs: calls.append(True),
    )
    attempt = root / attempt_runner.ATTEMPT_ROOT / "attempt-1"
    try:
        result = attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=dataset,
            preregistration_path=preregistration,
            project_root=root,
        )
        start = json.loads((attempt / "start.json").read_text(encoding="utf-8"))
        completion = json.loads((attempt / "completion.json").read_text(encoding="utf-8"))
        assert result.status == "failed_preflight"
        assert calls == []
        assert "src/inkling_quant_lab/new_after_lock.py" in start["file_sha256"]
        assert completion["subprocess_launched"] is False
        assert "omits required file bindings" in completion["failure"]["message"]
    finally:
        _thaw(attempt)


def test_fact_collection_exception_still_closes_failed_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dataset, preregistration, _ = _project_fixture(tmp_path, monkeypatch)
    calls: list[bool] = []

    def fail_environment(project_root: Path) -> dict[str, Any]:
        del project_root
        raise RuntimeError("environment probe fixture failed")

    monkeypatch.setattr(attempt_runner, "_environment_contract", fail_environment)
    monkeypatch.setattr(
        attempt_runner.subprocess,
        "run",
        lambda *args, **kwargs: calls.append(True),
    )
    attempt = root / attempt_runner.ATTEMPT_ROOT / "attempt-1"
    try:
        result = attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=dataset,
            preregistration_path=preregistration,
            project_root=root,
        )
        start = json.loads((attempt / "start.json").read_text(encoding="utf-8"))
        completion = json.loads((attempt / "completion.json").read_text(encoding="utf-8"))
        assert result.status == "failed_preflight"
        assert calls == []
        assert start["preflight_materialization_failure"]["type"] == "RuntimeError"
        assert completion["status"] == "failed_preflight"
        assert completion["subprocess_launched"] is False
    finally:
        _thaw(attempt)


def test_invalid_record_after_zero_exit_is_retained_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dataset, preregistration, _ = _project_fixture(tmp_path, monkeypatch)
    attempt_one = _sealed_attempt_one(root)

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        _output_path(command, cast(Path, kwargs["cwd"])).write_text(
            json.dumps({"schema_version": "wrong"}), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(attempt_runner.subprocess, "run", fake_run)
    attempt = root / attempt_runner.ATTEMPT_ROOT / "attempt-2"
    try:
        result = attempt_runner.run_attempt(
            ordinal=2,
            dataset_path=dataset,
            preregistration_path=preregistration,
            project_root=root,
        )
        completion = json.loads((attempt / "completion.json").read_text(encoding="utf-8"))
        assert result.status == "invalid_record"
        assert result.exit_code == 1
        assert completion["subprocess_returncode"] == 0
        assert completion["record"]["exists"] is True
        assert completion["record"]["valid"] is False
        assert completion["record"]["validation_error"] == ("record is missing confirmatory_status")
    finally:
        _thaw(attempt)
        _thaw(attempt_one)


def test_subprocess_exception_is_closed_as_runner_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dataset, preregistration, _ = _project_fixture(tmp_path, monkeypatch)

    def fail_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        del command, kwargs
        raise OSError("exec fixture failed")

    monkeypatch.setattr(attempt_runner.subprocess, "run", fail_run)
    attempt = root / attempt_runner.ATTEMPT_ROOT / "attempt-1"
    try:
        result = attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=dataset,
            preregistration_path=preregistration,
            project_root=root,
        )
        completion = json.loads((attempt / "completion.json").read_text(encoding="utf-8"))
        assert result.status == "runner_exception"
        assert completion["failure"] == {
            "type": "OSError",
            "message": "exec fixture failed",
        }
        assert completion["subprocess_launched"] is True
    finally:
        _thaw(attempt)


def test_duplicate_attempt_directory_is_never_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dataset, preregistration, _ = _project_fixture(tmp_path, monkeypatch)
    attempt = root / attempt_runner.ATTEMPT_ROOT / "attempt-1"
    attempt.mkdir(parents=True)
    marker = attempt / "owned.txt"
    marker.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=dataset,
            preregistration_path=preregistration,
            project_root=root,
        )

    assert marker.read_text(encoding="utf-8") == "existing"
    assert not (attempt / "start.json").exists()


def test_attempt_root_extras_fail_before_creating_or_launching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dataset, preregistration, _ = _project_fixture(tmp_path, monkeypatch)
    attempt_root = root / attempt_runner.ATTEMPT_ROOT
    attempt_root.mkdir(parents=True)
    (attempt_root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    calls: list[bool] = []
    monkeypatch.setattr(
        attempt_runner.subprocess,
        "run",
        lambda *args, **kwargs: calls.append(True),
    )

    with pytest.raises(attempt_runner.PreflightError, match="empty before ordinal 1"):
        attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=dataset,
            preregistration_path=preregistration,
            project_root=root,
        )
    with pytest.raises(attempt_runner.PreflightError, match="exactly sealed attempt-1"):
        attempt_runner.run_attempt(
            ordinal=2,
            dataset_path=dataset,
            preregistration_path=preregistration,
            project_root=root,
        )

    assert calls == []
    assert not (attempt_root / "attempt-1").exists()
    assert not (attempt_root / "attempt-2").exists()


def test_ordinal_two_rejects_unsealed_or_extra_attempt_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dataset, preregistration, _ = _project_fixture(tmp_path, monkeypatch)
    attempt_one = _sealed_attempt_one(root)
    attempt_one.chmod(0o755)
    (attempt_one / "extra.txt").write_text("extra", encoding="utf-8")
    calls: list[bool] = []
    monkeypatch.setattr(
        attempt_runner.subprocess,
        "run",
        lambda *args, **kwargs: calls.append(True),
    )
    try:
        with pytest.raises(attempt_runner.PreflightError, match="mode must be exactly 0555"):
            attempt_runner.run_attempt(
                ordinal=2,
                dataset_path=dataset,
                preregistration_path=preregistration,
                project_root=root,
            )
        assert calls == []
        assert not (root / attempt_runner.ATTEMPT_ROOT / "attempt-2").exists()
    finally:
        _thaw(attempt_one)


def test_ordinal_and_preregistration_path_are_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dataset, preregistration, _ = _project_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="ordinal"):
        attempt_runner.run_attempt(
            ordinal=cast(Literal[1, 2], 3),
            dataset_path=dataset,
            preregistration_path=preregistration,
            project_root=root,
        )
    wrong = root / "wrong-preregistration.json"
    wrong.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="only the checked preregistration path"):
        attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=dataset,
            preregistration_path=wrong,
            project_root=root,
        )
    with pytest.raises(SystemExit):
        attempt_runner._arguments(
            ["--ordinal", "3", "--dataset", str(dataset), "--preregistration", str(preregistration)]
        )


def test_symlink_dataset_and_attempt_parent_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dataset, preregistration, _ = _project_fixture(tmp_path, monkeypatch)
    linked_dataset = root / "linked-valid.txt"
    linked_dataset.symlink_to(dataset)
    with pytest.raises(ValueError, match="non-symlink"):
        attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=linked_dataset,
            preregistration_path=preregistration,
            project_root=root,
        )

    research_slices = root / "artifacts" / "research-slices"
    research_slices.mkdir(parents=True)
    outside = root / "outside"
    outside.mkdir()
    (research_slices / "stories15m-native-int8-confirmatory-256").symlink_to(
        outside, target_is_directory=True
    )
    with pytest.raises(ValueError, match="not a real directory"):
        attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=dataset,
            preregistration_path=preregistration,
            project_root=root,
        )
    assert not (outside / "attempt-1").exists()


def test_source_closure_symlink_is_retained_as_failed_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, dataset, preregistration, _ = _project_fixture(tmp_path, monkeypatch)
    target = root / "src/inkling_quant_lab/nested/extra_fixture.py"
    (root / "src/inkling_quant_lab/linked.py").symlink_to(target)
    calls: list[bool] = []
    monkeypatch.setattr(
        attempt_runner.subprocess,
        "run",
        lambda *args, **kwargs: calls.append(True),
    )
    attempt = root / attempt_runner.ATTEMPT_ROOT / "attempt-1"
    try:
        result = attempt_runner.run_attempt(
            ordinal=1,
            dataset_path=dataset,
            preregistration_path=preregistration,
            project_root=root,
        )
        completion = json.loads((attempt / "completion.json").read_text(encoding="utf-8"))
        assert result.status == "failed_preflight"
        assert calls == []
        assert completion["failure"]["type"] == "PreflightError"
        assert "source closure contains a symlink" in completion["failure"]["message"]
    finally:
        _thaw(attempt)


@pytest.mark.parametrize("bad_token", ("NaN", "Infinity"))
def test_preregistration_json_rejects_nonfinite_numbers_and_duplicate_keys(
    tmp_path: Path, bad_token: str
) -> None:
    path = tmp_path / "prereg.json"
    path.write_text(f'{{"value":{bad_token}}}', encoding="utf-8")
    with pytest.raises(attempt_runner.PreflightError, match="non-finite"):
        attempt_runner._load_preregistration(path)

    path.write_text('{"status":"one","status":"two"}', encoding="utf-8")
    with pytest.raises(attempt_runner.PreflightError, match="duplicate"):
        attempt_runner._load_preregistration(path)
