"""Fail-closed tests for the matched Inkling offline preflight."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import inkling_quant_lab.gguf.inkling_matched_preflight as matched_preflight_module
from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.gguf.inkling import (
    INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
)
from inkling_quant_lab.gguf.inkling_matched import (
    BF16_SUBJECT_REFERENCE_RELATIVE_PATH,
    MATCHED_CELL_CONFIG_RELATIVE_PATH,
)
from inkling_quant_lab.gguf.inkling_matched_preflight import (
    MATCHED_PREFLIGHT_PLAN_HASH_DOMAIN,
    InklingMatchedPreflightReport,
    build_matched_preflight_report,
)
from inkling_quant_lab.gguf.inkling_smoke import (
    VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH,
)
from scripts import manage_inkling_matched

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATCH_RELATIVE_PATH = "patches/inkling-smoke-a015409.patch"
CONTROL_RELATIVE_PATHS = (
    MATCHED_CELL_CONFIG_RELATIVE_PATH,
    BF16_SUBJECT_REFERENCE_RELATIVE_PATH,
    VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH,
    INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
    PATCH_RELATIVE_PATH,
)

PayloadMutation = Callable[[dict[str, Any]], None]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _copy_control_bundle(project_root: Path) -> None:
    for relative_path in CONTROL_RELATIVE_PATHS:
        source = PROJECT_ROOT / relative_path
        destination = project_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def _reseal(payload: dict[str, Any]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "plan_sha256"}
    payload["plan_sha256"] = hashlib.sha256(
        MATCHED_PREFLIGHT_PLAN_HASH_DOMAIN + _canonical_json(unsigned).encode("utf-8")
    ).hexdigest()


def _delete_first_control(payload: dict[str, Any]) -> None:
    del payload["control_files"][0]


def _forge_first_control_hash(payload: dict[str, Any]) -> None:
    payload["control_files"][0]["sha256"] = "0" * 64


def _forge_mount(payload: dict[str, Any]) -> None:
    payload["mounts"][0]["volume"] = "forged-volume"


def _forge_subject_summary(payload: dict[str, Any]) -> None:
    payload["subjects"][0]["reference_sha256"] = "0" * 64


def _forge_embedded_reference_identity(payload: dict[str, Any]) -> None:
    payload["bf16_reference"]["subject_run_id"] = "forged-run"


def _forge_inventory_path(payload: dict[str, Any]) -> None:
    payload["inventory_assignments"][0]["path"] = "/outside/forged.gguf"


def _forge_inventory_hash(payload: dict[str, Any]) -> None:
    payload["inventory_assignments"][0]["sha256"] = "0" * 64


def _forge_inventory_size(payload: dict[str, Any]) -> None:
    payload["inventory_assignments"][0]["size_bytes"] += 1


def _forge_inventory_stage(payload: dict[str, Any]) -> None:
    payload["inventory_assignments"][0]["stage"] = "rehash_q3_subject"


def _make_capacity_negative(payload: dict[str, Any]) -> None:
    payload["declared_capacity"]["declared_remaining_bytes"] = -1


def _forge_capacity_arithmetic(payload: dict[str, Any]) -> None:
    payload["declared_capacity"]["bf16_subject_bytes"] += 1


def _forge_declared_resource_cell(payload: dict[str, Any]) -> None:
    payload["declared_resource_cell"]["cpu_cores"] = 32


def _forge_negative_fact(payload: dict[str, Any]) -> None:
    payload["facts"]["provider_contacted"] = True


@pytest.mark.parametrize(
    "mutation",
    (
        _delete_first_control,
        _forge_first_control_hash,
        _forge_mount,
        _forge_subject_summary,
        _forge_embedded_reference_identity,
        _forge_inventory_path,
        _forge_inventory_hash,
        _forge_inventory_size,
        _forge_inventory_stage,
        _make_capacity_negative,
        _forge_capacity_arithmetic,
        _forge_declared_resource_cell,
        _forge_negative_fact,
    ),
    ids=(
        "control-count",
        "control-hash",
        "mount",
        "subject-summary",
        "reference-identity",
        "inventory-path",
        "inventory-hash",
        "inventory-size",
        "inventory-stage",
        "negative-capacity",
        "capacity-arithmetic",
        "resource-cell",
        "negative-fact",
    ),
)
def test_resealed_semantic_mutations_are_rejected(mutation: PayloadMutation) -> None:
    payload = build_matched_preflight_report(PROJECT_ROOT).model_dump(mode="json")
    mutation(payload)
    _reseal(payload)

    with pytest.raises(ValidationError):
        InklingMatchedPreflightReport.model_validate(payload)


def test_stale_plan_hash_is_rejected() -> None:
    payload = build_matched_preflight_report(PROJECT_ROOT).model_dump(mode="json")
    payload["plan_sha256"] = "0" * 64

    with pytest.raises(ValidationError, match="plan SHA-256"):
        InklingMatchedPreflightReport.model_validate(payload)


def test_canonical_json_round_trip_preserves_report_and_hash() -> None:
    report = build_matched_preflight_report(PROJECT_ROOT)

    restored = InklingMatchedPreflightReport.model_validate_json(report.canonical_json())

    assert restored == report
    assert restored.canonical_json() == report.canonical_json()
    assert restored.computed_plan_sha256() == restored.plan_sha256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "bf16_subject_reference_path",
            "configs/experiments/forged-bf16-reference.json",
            "control paths",
        ),
        (
            "instrumentation_patch_sha256",
            "0" * 64,
            "patch control SHA-256",
        ),
    ),
)
def test_build_rejects_control_and_config_binding_drift_before_reference_parsing(
    field: str,
    value: str,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_config = build_matched_preflight_report(PROJECT_ROOT).config
    if field == "instrumentation_patch_sha256":
        forged_runtime = checked_config.runtime.model_copy(update={field: value})
        forged_config = checked_config.model_copy(update={"runtime": forged_runtime})
    else:
        forged_config = checked_config.model_copy(update={field: value})

    monkeypatch.setattr(
        matched_preflight_module,
        "parse_matched_cell_config_bytes",
        lambda *_args, **_kwargs: forged_config,
    )
    monkeypatch.setattr(
        matched_preflight_module,
        "_parse_canonical_json_reference",
        lambda *_args, **_kwargs: pytest.fail("reference parser ran before control binding"),
    )

    with pytest.raises(ConfigurationError, match=message):
        build_matched_preflight_report(PROJECT_ROOT)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        (
            "q3_verified_export_reference_path",
            "configs/experiments/forged-q3-reference.json",
        ),
        ("instrumentation_patch_sha256", "0" * 64),
    ),
)
def test_report_validator_rechecks_control_and_config_binding(
    field: str,
    value: str,
) -> None:
    report = build_matched_preflight_report(PROJECT_ROOT)
    if field == "instrumentation_patch_sha256":
        forged_runtime = report.config.runtime.model_copy(update={field: value})
        forged_config = report.config.model_copy(update={"runtime": forged_runtime})
    else:
        forged_config = report.config.model_copy(update={field: value})
    forged_report = report.model_copy(update={"config": forged_config})

    with pytest.raises(ValueError, match="control files differ from the embedded config"):
        forged_report.fail_closed_truth()


@pytest.mark.parametrize(
    "capability",
    ("_NATIVE_OPEN_SUPPORTS_DIR_FD", "_NATIVE_STAT_SUPPORTS_FOLLOW_SYMLINKS"),
)
def test_secure_reader_fails_closed_without_required_posix_capability(
    capability: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matched_preflight_module, capability, False)

    with pytest.raises(ConfigurationError, match=r"(?i)POSIX"):
        build_matched_preflight_report(PROJECT_ROOT)


def test_secure_reader_rejects_a_symlinked_ancestor(tmp_path: Path) -> None:
    project_root = tmp_path / "control"
    _copy_control_bundle(project_root)
    configs = project_root / "configs"
    configs_target = project_root / "configs-target"
    configs.rename(configs_target)
    configs.symlink_to(configs_target, target_is_directory=True)

    with pytest.raises(ConfigurationError, match=r"(?i)(symlink|directory|open)"):
        build_matched_preflight_report(project_root)


def test_secure_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    project_root = tmp_path / "control"
    _copy_control_bundle(project_root)
    target = project_root / MATCHED_CELL_CONFIG_RELATIVE_PATH
    target.unlink()
    os.mkfifo(target)

    program = """
import sys

from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.gguf.inkling_matched_preflight import (
    build_matched_preflight_report,
)

try:
    build_matched_preflight_report(sys.argv[1])
except ConfigurationError as error:
    if "regular file" in str(error).lower():
        raise SystemExit(0)
    raise SystemExit(2) from error
raise SystemExit(3)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-B", "-c", program, str(project_root)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr


def test_secure_reader_rejects_directory_leaf(tmp_path: Path) -> None:
    project_root = tmp_path / "control"
    _copy_control_bundle(project_root)
    target = project_root / MATCHED_CELL_CONFIG_RELATIVE_PATH
    target.unlink()
    target.mkdir()

    with pytest.raises(ConfigurationError, match=r"(?i)regular file"):
        build_matched_preflight_report(project_root)


def test_secure_reader_rejects_device_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = os.open
    original_fstat = os.fstat
    config_name = Path(MATCHED_CELL_CONFIG_RELATIVE_PATH).name
    device_descriptor: int | None = None

    def track_leaf_open(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal device_descriptor
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == config_name and kwargs.get("dir_fd") is not None:
            device_descriptor = descriptor
        return descriptor

    def report_device_metadata(descriptor: int) -> os.stat_result:
        metadata = original_fstat(descriptor)
        if descriptor != device_descriptor:
            return metadata
        fields = list(metadata)
        fields[0] = stat.S_IFCHR | stat.S_IRUSR
        return os.stat_result(fields)

    monkeypatch.setattr(os, "open", track_leaf_open)
    monkeypatch.setattr(os, "fstat", report_device_metadata)

    with pytest.raises(ConfigurationError, match=r"(?i)regular file"):
        build_matched_preflight_report(PROJECT_ROOT)


def test_close_failure_is_normalized_and_every_descriptor_is_closed_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_close = os.close
    close_attempts: list[int] = []
    failure_injected = False

    def close_then_fail_once(descriptor: int) -> None:
        nonlocal failure_injected
        close_attempts.append(descriptor)
        original_close(descriptor)
        if not failure_injected:
            failure_injected = True
            raise OSError(errno.EIO, "simulated close failure")

    monkeypatch.setattr(os, "close", close_then_fail_once)

    with pytest.raises(ConfigurationError, match=r"(?i)close"):
        build_matched_preflight_report(PROJECT_ROOT)

    assert failure_injected
    assert len(close_attempts) >= 4
    assert len(close_attempts) == len(set(close_attempts))


def test_leaf_close_failure_still_closes_parent_and_root_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_open = os.open
    original_close = os.close
    config_name = Path(MATCHED_CELL_CONFIG_RELATIVE_PATH).name
    close_attempts: list[int] = []
    leaf_descriptor: int | None = None
    leaf_parent_descriptor: int | None = None
    failure_injected = False

    def track_leaf_open(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal leaf_descriptor, leaf_parent_descriptor
        descriptor = original_open(path, flags, *args, **kwargs)
        if path == config_name and kwargs.get("dir_fd") is not None:
            leaf_descriptor = descriptor
            leaf_parent_descriptor = int(kwargs["dir_fd"])
            close_attempts.clear()
        return descriptor

    def close_leaf_then_fail(descriptor: int) -> None:
        nonlocal failure_injected
        close_attempts.append(descriptor)
        original_close(descriptor)
        if descriptor == leaf_descriptor and not failure_injected:
            failure_injected = True
            raise OSError(errno.EIO, "simulated leaf close failure")

    monkeypatch.setattr(os, "open", track_leaf_open)
    monkeypatch.setattr(os, "close", close_leaf_then_fail)

    with pytest.raises(ConfigurationError, match=r"(?i)close"):
        build_matched_preflight_report(PROJECT_ROOT)

    assert failure_injected
    assert leaf_descriptor is not None
    assert leaf_parent_descriptor is not None
    assert close_attempts[0] == leaf_descriptor
    assert leaf_parent_descriptor in close_attempts
    assert len(close_attempts) >= 3
    assert len(close_attempts) == len(set(close_attempts))


def test_path_swap_after_open_keeps_descriptor_bound_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "control"
    _copy_control_bundle(project_root)
    target = project_root / MATCHED_CELL_CONFIG_RELATIVE_PATH
    original_bytes = target.read_bytes()
    replacement_bytes = b"schema_version: forged\n"
    backup = target.with_name(f"{target.name}.opened")
    original_open = os.open
    swapped = False

    def swap_path_after_leaf_open(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        nonlocal swapped
        descriptor = original_open(path, flags, *args, **kwargs)
        if not swapped and path == target.name and kwargs.get("dir_fd") is not None:
            target.rename(backup)
            target.write_bytes(replacement_bytes)
            swapped = True
        return descriptor

    monkeypatch.setattr(os, "open", swap_path_after_leaf_open)

    report = build_matched_preflight_report(project_root)
    config_control = next(
        control for control in report.control_files if control.role == "matched_config"
    )

    assert swapped
    assert target.read_bytes() == replacement_bytes
    assert backup.read_bytes() == original_bytes
    assert config_control.sha256 == hashlib.sha256(original_bytes).hexdigest()


def test_parser_uses_captured_bytes_after_named_file_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "control"
    _copy_control_bundle(project_root)
    target = project_root / MATCHED_CELL_CONFIG_RELATIVE_PATH
    replacement_bytes = b"schema_version: forged\n"
    original_parser = matched_preflight_module.parse_matched_cell_config_bytes
    parser_called = False

    def mutate_then_parse(raw_bytes: bytes, *, source: str | Path) -> object:
        nonlocal parser_called
        parser_called = True
        target.write_bytes(replacement_bytes)
        return original_parser(raw_bytes, source=source)

    monkeypatch.setattr(
        matched_preflight_module,
        "parse_matched_cell_config_bytes",
        mutate_then_parse,
    )

    report = build_matched_preflight_report(project_root)

    assert parser_called
    assert target.read_bytes() == replacement_bytes
    assert report.status == "ready_for_operator_review"


def test_raw_identity_mismatch_fails_before_any_parser_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "control"
    _copy_control_bundle(project_root)
    target = project_root / VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH
    target.write_bytes(b"[" * 10_000 + b"]" * 10_000)

    def unexpected_parser(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("untrusted control bytes reached a parser")

    monkeypatch.setattr(
        matched_preflight_module,
        "parse_matched_cell_config_bytes",
        unexpected_parser,
    )

    with pytest.raises(ConfigurationError, match=r"(?i)q3 reference control file"):
        build_matched_preflight_report(project_root)


def test_json_recursion_error_becomes_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_loads(*_args: object, **_kwargs: object) -> object:
        raise RecursionError("simulated JSON nesting failure")

    monkeypatch.setattr(matched_preflight_module.json, "loads", fail_loads)

    with pytest.raises(ConfigurationError, match=r"(?i)nesting failure"):
        build_matched_preflight_report(PROJECT_ROOT)


def test_os_read_error_becomes_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_read(_descriptor: int, _size: int) -> bytes:
        raise OSError(errno.EIO, "simulated read failure")

    monkeypatch.setattr(os, "read", fail_read)

    with pytest.raises(ConfigurationError, match=r"(?i)read"):
        build_matched_preflight_report(PROJECT_ROOT)


def test_os_fstat_error_becomes_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_fstat(_descriptor: int) -> os.stat_result:
        raise OSError(errno.EIO, "simulated fstat failure")

    monkeypatch.setattr(os, "fstat", fail_fstat)

    with pytest.raises(ConfigurationError, match=r"(?i)(inspect|fstat)"):
        build_matched_preflight_report(PROJECT_ROOT)


def test_premature_eof_becomes_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "read", lambda _descriptor, _size: b"")

    with pytest.raises(ConfigurationError, match=r"(?i)(ended|size)"):
        build_matched_preflight_report(PROJECT_ROOT)


def test_root_resolve_error_becomes_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_resolve(_path: Path, *, strict: bool = False) -> Path:
        del strict
        raise OSError(errno.EIO, "simulated resolve failure")

    monkeypatch.setattr(Path, "resolve", fail_resolve)

    with pytest.raises(ConfigurationError, match=r"(?i)resolve"):
        build_matched_preflight_report(PROJECT_ROOT)


def test_cli_json_normalizes_read_error_to_one_canonical_record(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_read(_descriptor: int, _size: int) -> bytes:
        raise OSError(errno.EIO, "simulated read failure")

    monkeypatch.setattr(os, "read", fail_read)

    assert manage_inkling_matched.main(["preflight", "--json"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "invalid"
    assert payload["error"]["code"] == "CONFIGURATION_ERROR"
    assert payload["error"]["component"] == "inkling_matched_preflight"
    assert "read" in payload["error"]["message"].lower()
    assert captured.err == _canonical_json(payload) + "\n"


def test_cli_process_normalizes_deep_malformed_reference_before_parsing(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "control"
    _copy_control_bundle(project_root)
    target = project_root / VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH
    target.write_bytes(b"[" * 10_000 + b"]" * 10_000)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)))

    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/manage_inkling_matched.py"),
            "preflight",
            "--project-root",
            str(project_root),
            "--json",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        timeout=10,
    )

    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "Traceback" not in completed.stderr
    lines = completed.stderr.splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["status"] == "invalid"
    assert payload["error"]["code"] == "CONFIGURATION_ERROR"
    assert payload["error"]["component"] == "inkling_matched_preflight"
    assert "q3 reference control file" in payload["error"]["message"].lower()
    assert completed.stderr == _canonical_json(payload) + "\n"
