"""Failure-first tests for the launch-free matched Inkling preflight."""

from __future__ import annotations

import ast
import builtins
import io
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.gguf.inkling import (
    INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
)
from inkling_quant_lab.gguf.inkling_matched import (
    BF16_SUBJECT_REFERENCE_RELATIVE_PATH,
    CAPACITY_SCREEN_LIMITATION,
    EXPECTED_BF16_TOTAL_BYTES,
    MATCHED_CELL_CONFIG_RELATIVE_PATH,
)
from inkling_quant_lab.gguf.inkling_matched_preflight import (
    build_matched_preflight_report,
)
from inkling_quant_lab.gguf.inkling_smoke import (
    EXPECTED_PROJECTOR_BYTES,
    EXPECTED_Q3_TOTAL_BYTES,
    VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH,
)
from scripts import manage_inkling_matched

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANAGER_PATH = PROJECT_ROOT / "scripts/manage_inkling_matched.py"
PREFLIGHT_PATH = PROJECT_ROOT / "src/inkling_quant_lab/gguf/inkling_matched_preflight.py"
PATCH_RELATIVE_PATH = "patches/inkling-smoke-a015409.patch"
BF16_RUN_SUBPATH = "runs/inkling-q3km-86b4d430-a015409e-ffd466dd93-8083cf41e1"
SOURCE_RUN_SUBPATH = "runs/inkling-q3km-86b4d430-a015409e-551ab8f240-bcc168525e"
EXPECTED_STAGES = (
    ("verify_subject_references", "passed"),
    ("rehash_bf16_subject", "not_executed"),
    ("rehash_q3_subject", "not_executed"),
    ("screen_aggregate_capacity", "not_executed"),
    ("smoke_bf16_subject", "not_executed"),
    ("smoke_q3_subject", "not_executed"),
    ("verify_matched_smoke_evidence", "not_executed"),
)
CONTROL_RELATIVE_PATHS = (
    MATCHED_CELL_CONFIG_RELATIVE_PATH,
    BF16_SUBJECT_REFERENCE_RELATIVE_PATH,
    VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH,
    INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
    PATCH_RELATIVE_PATH,
)


def _mapping(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        raise AssertionError(f"Expected a mapping-like report record, got {type(value)!r}")
    result = model_dump(mode="json")
    if not isinstance(result, dict):
        raise AssertionError("model_dump(mode='json') must return a mapping")
    return result


def _values_for_key(value: object, expected_key: str) -> tuple[object, ...]:
    values: list[object] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == expected_key:
                values.append(child)
            values.extend(_values_for_key(child, expected_key))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            values.extend(_values_for_key(child, expected_key))
    return tuple(values)


def _copy_control_bundle(project_root: Path) -> None:
    for relative_path in CONTROL_RELATIVE_PATHS:
        source = PROJECT_ROOT / relative_path
        destination = project_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def _copy_clean_preflight_tree(project_root: Path) -> None:
    shutil.copytree(
        PROJECT_ROOT / "src",
        project_root / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    scripts = project_root / "scripts"
    scripts.mkdir()
    for name in ("__init__.py", "manage_inkling_matched.py"):
        shutil.copy2(PROJECT_ROOT / "scripts" / name, scripts / name)
    _copy_control_bundle(project_root)


def _tree_snapshot(
    root: Path,
) -> tuple[tuple[str, ...], dict[str, tuple[int, int, bytes]]]:
    directories = tuple(
        path.relative_to(root).as_posix() for path in sorted(root.rglob("*")) if path.is_dir()
    )
    files = {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    return directories, files


def _expected_inventory_paths() -> tuple[str, ...]:
    return (
        *(f"/baseline/bf16/inkling-BF16-{index:05d}-of-00049.gguf" for index in range(1, 50)),
        "/baseline/convert_text_bf16.success.json",
        *(f"/final/q3_k_m/inkling-Q3_K_M-{index:05d}-of-00049.gguf" for index in range(1, 50)),
        "/final/mmproj/mmproj-BF16.gguf",
        "/final/verification/export_manifest.json",
        "/final/verify_export.success.json",
        "/final/quantize_text.success.json",
        "/final/convert_multimodal_projector.success.json",
        "/source/snapshot/chat_template.jinja",
        "/source/snapshot/processor_config.json",
        "/source/snapshot/special_tokens_map.json",
        "/source/snapshot/tiktoken/tokenizer.model",
        "/source/snapshot/tokenizer.json",
        "/source/snapshot/tokenizer_config.json",
    )


def _dotted_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent is not None else None
    return None


def test_report_is_canonical_deterministic_and_self_identifying() -> None:
    first = build_matched_preflight_report(PROJECT_ROOT)
    second = build_matched_preflight_report(PROJECT_ROOT)

    assert first == second
    assert first.canonical_json() == second.canonical_json()
    assert re.fullmatch(r"[0-9a-f]{64}", first.plan_sha256)
    payload = json.loads(first.canonical_json())
    assert first.canonical_json() == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    assert payload["plan_sha256"] == first.plan_sha256
    assert payload["stages"] == [_mapping(item) for item in first.stages]
    assert payload["mounts"] == [_mapping(item) for item in first.mounts]
    assert payload["inventory_assignments"] == [
        _mapping(item) for item in first.inventory_assignments
    ]
    assert payload["declared_capacity"] == _mapping(first.declared_capacity)


def test_report_marks_only_static_reference_validation_as_passed() -> None:
    report = build_matched_preflight_report(PROJECT_ROOT)
    stages = tuple((record["name"], record["status"]) for record in map(_mapping, report.stages))

    assert stages == EXPECTED_STAGES
    for record in map(_mapping, report.stages):
        assert record["execution_performed"] is False
        assert record["measurement_observed"] is False


def test_report_keeps_every_execution_and_claim_fact_false() -> None:
    payload = json.loads(build_matched_preflight_report(PROJECT_ROOT).canonical_json())
    required_false_facts = (
        "remote_execution_default_enabled",
        "remote_execution_performed",
        "measurement_execution_allowed",
        "measurement_execution_performed",
        "artifact_rehash_performed",
        "hardware_probed",
        "capacity_screen_executed",
        "bf16_smoke_executed",
        "q3_smoke_executed",
        "matched_smoke_verified",
        "runtime_fit_proven",
        "quality_measured",
        "performance_measured",
        "quality_retention_claim_allowed",
        "speedup_claim_allowed",
        "mtp_included",
        "mtp_supported",
    )

    for key in required_false_facts:
        values = _values_for_key(payload, key)
        assert values, f"preflight report omitted required negative fact {key!r}"
        assert all(value is False for value in values), (
            f"preflight report made a positive {key!r} claim: {values!r}"
        )

    assert payload["execution"]["record_status"] == "execution_ready"
    assert payload["execution"]["runner_implemented"] is True
    assert payload["facts"]["remote_execution_performed"] is False
    assert payload["facts"]["paid_compute_started"] is False


def test_report_records_explicit_negative_side_effect_facts() -> None:
    facts = json.loads(build_matched_preflight_report(PROJECT_ROOT).canonical_json())["facts"]
    expected = {
        "provider_contacted": False,
        "network_access_performed": False,
        "remote_volume_inspected": False,
        "subprocess_execution_performed": False,
        "local_write_performed": False,
    }

    assert {key: facts.get(key) for key in expected} == expected


def test_capacity_is_declared_arithmetic_not_observed_hardware_evidence() -> None:
    capacity = _mapping(build_matched_preflight_report(PROJECT_ROOT).declared_capacity)

    assert capacity == {
        "gpu_count": 8,
        "configured_minimum_gpu_memory_bytes": 287_000_000_000,
        "configured_minimum_total_gpu_memory_bytes": 2_296_000_000_000,
        "capacity_reserve_basis_points": 1_000,
        "declared_headroom_bytes": 229_600_000_000,
        "declared_usable_bytes": 2_066_400_000_000,
        "bf16_subject_bytes": EXPECTED_BF16_TOTAL_BYTES + EXPECTED_PROJECTOR_BYTES,
        "q3_subject_bytes": EXPECTED_Q3_TOTAL_BYTES + EXPECTED_PROJECTOR_BYTES,
        "sequential_peak_subject_bytes": EXPECTED_BF16_TOTAL_BYTES + EXPECTED_PROJECTOR_BYTES,
        "declared_remaining_bytes": 171_938_188_160,
        "hardware_probed": False,
        "allocation_observed": False,
        "capacity_screen_executed": False,
        "runtime_fit_proven": False,
        "limitation": CAPACITY_SCREEN_LIMITATION,
    }
    assert not any(key.startswith("observed_") for key in capacity)


def test_report_records_the_exact_declared_resource_cell() -> None:
    payload = json.loads(build_matched_preflight_report(PROJECT_ROOT).canonical_json())

    assert payload["declared_resource_cell"] == {
        "provider": "modal",
        "gpu_type": "B300",
        "gpu_count": 8,
        "compute_capability": "10.3",
        "minimum_gpu_memory_bytes": 287_000_000_000,
        "capacity_reserve_basis_points": 1_000,
        "capacity_strategy": "sequential_peak_plus_reserve",
        "cpu_cores": 16,
        "memory_gib": 64,
        "ephemeral_disk_mib": 524_288,
        "startup_timeout_seconds": 1_800,
        "function_timeout_seconds": 14_400,
        "max_attempts": 1,
        "max_recovery_attempts": 0,
        "declared_only": True,
    }


def test_mount_plan_uses_exact_version_one_subpaths_and_access_modes() -> None:
    mounts = tuple(_mapping(item) for item in build_matched_preflight_report(PROJECT_ROOT).mounts)

    assert mounts == (
        {
            "name": "bf16",
            "volume": "inkling-work-v1",
            "volume_version": 1,
            "mount_path": "/baseline",
            "sub_path": BF16_RUN_SUBPATH,
            "read_only": True,
            "create_if_missing": False,
        },
        {
            "name": "q3",
            "volume": "inkling-final-v1",
            "volume_version": 1,
            "mount_path": "/final",
            "sub_path": BF16_RUN_SUBPATH,
            "read_only": True,
            "create_if_missing": False,
        },
        {
            "name": "source",
            "volume": "inkling-source-v1",
            "volume_version": 1,
            "mount_path": "/source",
            "sub_path": SOURCE_RUN_SUBPATH,
            "read_only": True,
            "create_if_missing": False,
        },
        {
            "name": "evidence",
            "volume": "inkling-matched-evidence-v1",
            "volume_version": 1,
            "mount_path": "/evidence",
            "sub_path": None,
            "read_only": False,
            "create_if_missing": True,
        },
    )


def test_inventory_assignments_are_complete_and_do_not_repeat_volume_subpaths() -> None:
    assignments = tuple(
        _mapping(item)
        for item in build_matched_preflight_report(PROJECT_ROOT).inventory_assignments
    )
    expected_paths = _expected_inventory_paths()

    assert tuple(item["path"] for item in assignments) == expected_paths
    assert len(assignments) == 110
    assert len({item["path"] for item in assignments}) == len(assignments)
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in assignments)
    assert all(type(item["size_bytes"]) is int and item["size_bytes"] > 0 for item in assignments)
    assert all(item["declared_only"] is True for item in assignments)
    assert all(item["rehash_performed"] is False for item in assignments)
    assert Counter(item["stage"] for item in assignments) == {
        "rehash_bf16_subject": 50,
        "rehash_q3_subject": 60,
    }
    assert all(
        fragment not in item["path"]
        for item in assignments
        for fragment in (
            f"/baseline/{BF16_RUN_SUBPATH}/",
            f"/final/{BF16_RUN_SUBPATH}/",
            f"/source/{SOURCE_RUN_SUBPATH}/",
        )
    )


def test_preflight_rejects_missing_runtime_patch(tmp_path: Path) -> None:
    project_root = tmp_path / "control"
    _copy_control_bundle(project_root)
    (project_root / PATCH_RELATIVE_PATH).unlink()

    with pytest.raises(ConfigurationError) as captured:
        build_matched_preflight_report(project_root)

    assert captured.value.component == "inkling_matched_preflight"
    assert "patch" in str(captured.value).lower()


def test_preflight_rejects_runtime_patch_hash_drift(tmp_path: Path) -> None:
    project_root = tmp_path / "control"
    _copy_control_bundle(project_root)
    patch_path = project_root / PATCH_RELATIVE_PATH
    patch_path.write_bytes(patch_path.read_bytes() + b"\n")

    with pytest.raises(ConfigurationError, match=r"(?i)patch.*(sha-?256|hash|differ)"):
        build_matched_preflight_report(project_root)


def test_preflight_rejects_reference_hash_drift(tmp_path: Path) -> None:
    project_root = tmp_path / "control"
    _copy_control_bundle(project_root)
    reference_path = project_root / BF16_SUBJECT_REFERENCE_RELATIVE_PATH
    value = json.loads(reference_path.read_text(encoding="utf-8"))
    value["reference_sha256"] = "0" * 64
    reference_path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match=r"(?i)(self-hash|sha-?256|hash)"):
        build_matched_preflight_report(project_root)


@pytest.mark.parametrize("relative_path", CONTROL_RELATIVE_PATHS)
def test_preflight_rejects_symlinked_control_files(
    tmp_path: Path,
    relative_path: str,
) -> None:
    project_root = tmp_path / "control"
    _copy_control_bundle(project_root)
    controlled_path = project_root / relative_path
    target = project_root / "symlink-targets" / controlled_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(controlled_path.read_bytes())
    controlled_path.unlink()
    controlled_path.symlink_to(target)

    with pytest.raises(ConfigurationError, match=r"(?i)(symlink|regular file|control file)"):
        build_matched_preflight_report(project_root)


def test_environment_cannot_enable_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = build_matched_preflight_report(PROJECT_ROOT).canonical_json()
    for name in (
        "IQL_MATCHED_REMOTE_EXECUTION_ALLOWED",
        "IQL_MATCHED_MEASUREMENT_EXECUTION_ALLOWED",
        "IQL_MODAL_WORKSPACE_BUDGET_CONFIRMED",
        "IQL_MODAL_BILLING_CYCLE_END_CONFIRMED",
        "CONFIRM_LAUNCH",
    ):
        monkeypatch.setenv(name, "1")

    actual = build_matched_preflight_report(PROJECT_ROOT)

    assert actual.canonical_json() == expected
    assert (
        tuple((item["name"], item["status"]) for item in map(_mapping, actual.stages))
        == EXPECTED_STAGES
    )


def test_cli_preflight_json_is_the_exact_canonical_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = build_matched_preflight_report(PROJECT_ROOT)

    assert manage_inkling_matched.main(["preflight", "--json"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == expected.canonical_json() + "\n"


def test_cli_inspect_reports_the_static_plan_without_execution(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = build_matched_preflight_report(PROJECT_ROOT)

    assert manage_inkling_matched.main(["inspect"]) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert report.plan_sha256 in captured.out
    assert "execution_ready" in captured.out
    assert "not_executed" in captured.out


@pytest.mark.parametrize("command", ("launch", "deploy"))
def test_cli_rejects_remote_commands_before_building_a_plan(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    def unexpected_build(_project_root: Path) -> object:
        raise AssertionError("unknown commands must fail before preflight work")

    monkeypatch.setattr(
        manage_inkling_matched,
        "build_matched_preflight_report",
        unexpected_build,
    )

    with pytest.raises(SystemExit) as captured:
        manage_inkling_matched.main([command])

    assert captured.value.code == 2


@pytest.mark.parametrize("path", (PREFLIGHT_PATH, MANAGER_PATH))
def test_preflight_surfaces_have_no_remote_execution_dependencies(path: Path) -> None:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_import_roots = {
        "aiohttp",
        "boto3",
        "google",
        "httpx",
        "huggingface_hub",
        "modal",
        "requests",
        "runpod",
        "socket",
        "subprocess",
        "urllib",
    }
    imports: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.split(".", maxsplit=1)[0])
    assert imports.isdisjoint(forbidden_import_roots)

    forbidden_calls = {
        "eval",
        "exec",
        "os.popen",
        "os.spawnl",
        "os.spawnle",
        "os.spawnlp",
        "os.spawnlpe",
        "os.spawnv",
        "os.spawnve",
        "os.spawnvp",
        "os.spawnvpe",
        "os.system",
    }
    forbidden_attributes = {
        "deploy",
        "for_each",
        "from_name",
        "remote",
        "remote_gen",
        "spawn",
    }
    calls = tuple(node for node in ast.walk(module) if isinstance(node, ast.Call))
    assert not {
        dotted
        for call in calls
        if (dotted := _dotted_name(call.func)) is not None and dotted in forbidden_calls
    }
    assert not {
        call.func.attr
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr in forbidden_attributes
    }


def test_fresh_process_full_closure_performs_no_remote_effects() -> None:
    program = r"""
import sys

blocked_events = {
    "http.client.connect",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.system",
    "subprocess.Popen",
    "urllib.Request",
}
provider_modules = (
    "boto3",
    "botocore",
    "google.cloud",
    "huggingface_hub",
    "modal",
    "runpod",
)


def reject_remote_effect(event, arguments):
    if event in blocked_events or event.startswith("socket."):
        raise RuntimeError(f"offline preflight attempted forbidden audit event: {event}")
    if event == "import" and arguments and isinstance(arguments[0], str):
        module = arguments[0]
        if any(module == name or module.startswith(f"{name}.") for name in provider_modules):
            raise RuntimeError(f"offline preflight imported provider module: {module}")


sys.addaudithook(reject_remote_effect)

from scripts import manage_inkling_matched

raise SystemExit(
    manage_inkling_matched.main(
        ["preflight", "--project-root", sys.argv[1], "--json"]
    )
)
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)))

    completed = subprocess.run(
        [sys.executable, "-B", "-c", program, str(PROJECT_ROOT)],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "ready_for_operator_review"


def test_build_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "control"
    _copy_control_bundle(project_root)

    def reject_path_mutation(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("matched preflight must not mutate the local filesystem")

    for method_name in (
        "chmod",
        "mkdir",
        "rename",
        "replace",
        "rmdir",
        "symlink_to",
        "touch",
        "unlink",
        "write_bytes",
        "write_text",
    ):
        monkeypatch.setattr(Path, method_name, reject_path_mutation)

    original_builtin_open = builtins.open
    original_io_open = io.open
    original_os_open = os.open

    def read_only_open(
        opener: Callable[..., Any],
        file: object,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ) -> Any:
        if any(character in mode for character in "wax+"):
            raise AssertionError("matched preflight attempted to open a local file for writing")
        return opener(file, mode, *args, **kwargs)

    monkeypatch.setattr(
        builtins,
        "open",
        lambda file, mode="r", *args, **kwargs: read_only_open(
            original_builtin_open, file, mode, *args, **kwargs
        ),
    )
    monkeypatch.setattr(
        io,
        "open",
        lambda file, mode="r", *args, **kwargs: read_only_open(
            original_io_open, file, mode, *args, **kwargs
        ),
    )

    def read_only_os_open(
        path: object,
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
        if flags & write_flags:
            raise AssertionError("matched preflight attempted a write-capable os.open")
        return original_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", read_only_os_open)

    report = build_matched_preflight_report(project_root)

    assert re.fullmatch(r"[0-9a-f]{64}", report.plan_sha256)


def test_normal_cli_process_creates_no_bytecode_or_application_files(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "clean-project"
    _copy_clean_preflight_tree(project_root)
    before = _tree_snapshot(project_root)
    environment = dict(os.environ)
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    environment.pop("PYTHONPYCACHEPREFIX", None)
    environment["PYTHONPATH"] = str(project_root / "src")

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts/manage_inkling_matched.py"),
            "preflight",
            "--project-root",
            str(project_root),
            "--json",
        ],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    json.loads(completed.stdout)
    after = _tree_snapshot(project_root)
    before_directories, before_files = before
    after_directories, after_files = after
    assert set(after_directories) == set(before_directories), {
        "added": sorted(set(after_directories) - set(before_directories)),
        "removed": sorted(set(before_directories) - set(after_directories)),
    }
    assert set(after_files) == set(before_files), {
        "added": sorted(set(after_files) - set(before_files)),
        "removed": sorted(set(before_files) - set(after_files)),
    }
    assert after_files == before_files
