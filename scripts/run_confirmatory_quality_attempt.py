#!/usr/bin/env python3
"""Run one append-only, preregistration-bound confirmatory quality attempt."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import stat
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

PROJECT_PREREGISTRATION = Path(
    "docs/experiments/stories15m-native-int8-confirmatory-256-preregistration.json"
)
BASELINE_CONFIG = Path("configs/experiments/hf_stories15m_tinystories_native_confirmatory_256.yaml")
CANDIDATE_CONFIG = Path(
    "configs/experiments/hf_stories15m_tinystories_native_int8_confirmatory_256.yaml"
)
PROTOCOL_CONFIG = Path("configs/evaluations/stories15m_native_int8_confirmatory_256.yaml")
EVALUATOR = Path("scripts/evaluate_public_moe_native_quality.py")
STATISTICAL_MODULE = Path("src/inkling_quant_lab/evaluation/noninferiority.py")
REPEAT_VERIFIER = Path("scripts/verify_confirmatory_quality_repeats.py")
RUNNER = Path("scripts/run_confirmatory_quality_attempt.py")
ATTEMPT_ROOT = Path("artifacts/research-slices/stories15m-native-int8-confirmatory-256")
HARDWARE_LABEL = "Apple M3 MacBook Air (8 cores, 16 GB)"
PROTOCOL_ID = "stories15m-native-int8-confirmatory-quality-v1"
OFFICIAL_DATASET_SHA256 = "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
OFFICIAL_DATASET_SIZE_BYTES = 19_447_282
OFFICIAL_DATASET_STORY_COUNT = 21_990
BASELINE_CONFIG_SHA256 = "e6db2959221babf9aaba1f529a20349fe3462d4309b6929caf3a7d3331d668f2"
CANDIDATE_CONFIG_SHA256 = "d6ee2054d6db801c56596042df7e9eca130109ab922ee46735b61953da8d7912"
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONHASHSEED": "0",
    "PYTHONPYCACHEPREFIX": "/dev/null",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
}
FORBIDDEN_AMBIENT_IMPORT_ENVIRONMENT = ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE")
REQUIRED_BOUND_FILES = (
    Path("scripts/__init__.py"),
    RUNNER,
    EVALUATOR,
    STATISTICAL_MODULE,
    REPEAT_VERIFIER,
    BASELINE_CONFIG,
    CANDIDATE_CONFIG,
    PROTOCOL_CONFIG,
    Path("pyproject.toml"),
    Path("uv.lock"),
    Path("configs/models/hf_stories15m_moe.yaml"),
    Path("configs/quantization/native_dynamic_int8.yaml"),
)
SOURCE_CLOSURE_ROOT = Path("src/inkling_quant_lab")
SEALED_ATTEMPT_FILES = frozenset(
    {"start.json", "stdout.log", "stderr.log", "record.json", "completion.json"}
)


@dataclass(frozen=True, slots=True)
class AttemptResult:
    """One terminal runner result after completion evidence has been sealed."""

    attempt_directory: Path
    status: str
    exit_code: int


class PreflightError(ValueError):
    """A preregistration or current-binding mismatch detected before launch."""


def _interpreter_isolation_facts() -> dict[str, Any]:
    return {
        "no_user_site": bool(sys.flags.no_user_site),
        "safe_path": bool(sys.flags.safe_path),
        "dont_write_bytecode": sys.dont_write_bytecode,
        "pycache_prefix": sys.pycache_prefix,
    }


def _require_source_import_contract() -> None:
    """Fail before importing project packages if bytecode could be read or written."""

    facts = _interpreter_isolation_facts()
    if facts != {
        "no_user_site": True,
        "safe_path": True,
        "dont_write_bytecode": True,
        "pycache_prefix": "/dev/null",
    }:
        raise RuntimeError(
            "confirmatory attempts require python -s -P -B with "
            "PYTHONPYCACHEPREFIX=/dev/null and PYTHONDONTWRITEBYTECODE=1"
        )
    present = tuple(name for name in FORBIDDEN_AMBIENT_IMPORT_ENVIRONMENT if name in os.environ)
    if present:
        raise RuntimeError(
            "confirmatory attempts forbid ambient local import path variables: "
            + ", ".join(present)
        )


def _source_import_contract(
    verified_local_modules: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    _require_source_import_contract()
    return {
        "python_flags": ["-s", "-P", "-B"],
        "sys_flags": {
            "no_user_site": True,
            "safe_path": True,
            "dont_write_bytecode": True,
        },
        "sys_pycache_prefix": "/dev/null",
        "forbidden_ambient_import_environment_absent": list(FORBIDDEN_AMBIENT_IMPORT_ENVIRONMENT),
        "environment": {
            "PYTHONDONTWRITEBYTECODE": OFFLINE_ENVIRONMENT["PYTHONDONTWRITEBYTECODE"],
            "PYTHONPYCACHEPREFIX": OFFLINE_ENVIRONMENT["PYTHONPYCACHEPREFIX"],
            "PYTHONNOUSERSITE": OFFLINE_ENVIRONMENT["PYTHONNOUSERSITE"],
        },
        "local_project_imports_deferred_until_after_entry_guard": True,
        "verified_local_modules": verified_local_modules,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_fact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required input must be a regular non-symlink file: {path}")
    return {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.part")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"refusing to overwrite append-only attempt file: {path}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite append-only attempt file: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _reject_traversal(path: Path, *, field: str) -> None:
    if ".." in path.parts or any(part in {"", "."} for part in path.parts[1:]):
        raise ValueError(f"{field} cannot contain traversal or ambiguous path components: {path}")


def _regular_input_path(path: Path, *, field: str) -> Path:
    _reject_traversal(path, field=field)
    candidate = path.expanduser().absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"{field} must be a regular non-symlink file: {path}")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate:
        raise ValueError(f"{field} must not traverse symlinked path components: {path}")
    return resolved


def _project_file(project_root: Path, relative: Path) -> Path:
    _reject_traversal(relative, field="project binding")
    if relative.is_absolute():
        raise ValueError(f"project binding must be relative: {relative}")
    candidate = (project_root / relative).absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"project binding must be a regular non-symlink file: {relative}")
    resolved = candidate.resolve(strict=True)
    if resolved != candidate or not resolved.is_relative_to(project_root):
        raise ValueError(f"project binding escapes through a symlink: {relative}")
    return resolved


def _ensure_attempt_parent(project_root: Path) -> Path:
    current = project_root
    for component in ATTEMPT_ROOT.parts:
        current = current / component
        if current.exists() or current.is_symlink():
            if current.is_symlink() or not current.is_dir():
                raise ValueError(f"attempt path component is not a real directory: {current}")
        else:
            current.mkdir(mode=0o700)
    return current


def _attempt_paths(project_root: Path, ordinal: Literal[1, 2]) -> tuple[Path, Path, Path]:
    relative_directory = ATTEMPT_ROOT / f"attempt-{ordinal}"
    directory = project_root / relative_directory
    return directory, relative_directory, relative_directory / "record.json"


def _validate_attempt_sequence(attempt_root: Path, ordinal: Literal[1, 2]) -> None:
    """Enforce exactly empty-before-1 and exactly one sealed successful attempt-before-2."""

    entries = tuple(sorted(attempt_root.iterdir(), key=lambda path: path.name))
    if ordinal == 1:
        if entries:
            attempt_one = attempt_root / "attempt-1"
            if attempt_one in entries:
                raise FileExistsError(
                    f"confirmatory attempt directory already exists: {attempt_one}"
                )
            raise PreflightError("attempt root must be empty before ordinal 1")
        return
    attempt_one = attempt_root / "attempt-1"
    if entries != (attempt_one,):
        raise PreflightError("ordinal 2 requires attempt root to contain exactly sealed attempt-1")
    if attempt_one.is_symlink() or not attempt_one.is_dir():
        raise PreflightError("attempt-1 must be a real sealed directory")
    if stat.S_IMODE(attempt_one.stat().st_mode) != 0o555:
        raise PreflightError("attempt-1 directory mode must be exactly 0555")
    children = tuple(sorted(attempt_one.iterdir(), key=lambda path: path.name))
    if {path.name for path in children} != SEALED_ATTEMPT_FILES:
        raise PreflightError("attempt-1 must contain exactly the five sealed attempt files")
    for path in children:
        if path.is_symlink() or not path.is_file():
            raise PreflightError(f"attempt-1 contains a non-regular file: {path.name}")
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise PreflightError(f"attempt-1 file mode must be 0444: {path.name}")
    try:
        completion = json.loads((attempt_one / "completion.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PreflightError(f"attempt-1 completion record is unreadable: {error}") from error
    if not isinstance(completion, dict) or any(
        completion.get(key) != value
        for key, value in {
            "schema_version": "confirmatory-quality-attempt-completion-v1",
            "status": "attempt_execution_complete",
            "ordinal": 1,
            "runner_exit_code": 0,
            "confirmatory_claim_ready": False,
        }.items()
    ):
        raise PreflightError("attempt-1 completion does not declare a clean ordinal-1 execution")
    record = completion.get("record")
    if not isinstance(record, dict) or record.get("valid") is not True:
        raise PreflightError("attempt-1 completion does not bind a valid raw record")
    validation = _record_validation(attempt_one / "record.json", ordinal=1)
    if not validation["valid"]:
        raise PreflightError(
            f"attempt-1 raw record no longer validates: {validation['validation_error']}"
        )


def _required_bound_paths(project_root: Path) -> tuple[Path, ...]:
    """Return the deterministic static-plus-complete local Python source closure."""

    source_root = (project_root / SOURCE_CLOSURE_ROOT).absolute()
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError(
            f"source closure root must be a real non-symlink directory: {SOURCE_CLOSURE_ROOT}"
        )
    resolved_source = source_root.resolve(strict=True)
    if resolved_source != source_root or not resolved_source.is_relative_to(project_root):
        raise ValueError("source closure root escapes through a symlink")
    discovered: list[Path] = []
    for path in sorted(source_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(
                f"source closure contains a symlink and cannot be preregistered: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"source closure contains a non-regular entry: {path}")
        if path.suffix == ".py":
            relative = path.relative_to(project_root)
            _reject_traversal(relative, field="source closure")
            discovered.append(relative)
    if not discovered:
        raise ValueError("source closure contains no Python files")
    return tuple(
        sorted(set(REQUIRED_BOUND_FILES).union(discovered), key=lambda path: path.as_posix())
    )


def _required_file_bindings(project_root: Path) -> dict[str, str]:
    return {
        relative.as_posix(): _sha256_file(_project_file(project_root, relative))
        for relative in _required_bound_paths(project_root)
    }


def _load_bound_scripts_package(project_root: Path) -> None:
    """Load ``scripts`` from its bound initializer without relying on ``sys.path``."""

    _require_source_import_contract()
    expected = _project_file(project_root, Path("scripts/__init__.py"))
    existing = sys.modules.get("scripts")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if (
            not isinstance(existing_file, str)
            or Path(existing_file).resolve(strict=True) != expected
        ):
            raise PreflightError("an unbound scripts package was imported before preflight")
        return
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "scripts",
        expected,
        submodule_search_locations=[str(expected.parent)],
    )
    if spec is None or spec.loader is None:
        raise PreflightError("unable to construct the bound scripts package import")
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("scripts", None)
        raise


def _resolved_scientific_bindings(project_root: Path) -> dict[str, str]:
    _require_source_import_contract()
    from inkling_quant_lab.config import load_config

    baseline = load_config(_project_file(project_root, BASELINE_CONFIG))
    candidate = load_config(_project_file(project_root, CANDIDATE_CONFIG))
    _load_bound_scripts_package(project_root)
    from scripts.evaluate_public_moe_native_quality import (
        load_confirmatory_protocol,
    )

    protocol = load_confirmatory_protocol(_project_file(project_root, PROTOCOL_CONFIG))
    return {
        "baseline": baseline.config_hash(),
        "candidate": candidate.config_hash(),
        "protocol_definition": _canonical_sha256(protocol.model_dump(mode="json")),
    }


def _local_import_provenance(
    project_root: Path, current_files: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Prove every imported local package module came from bound source, never bytecode."""

    _require_source_import_contract()
    observed: dict[str, dict[str, Any]] = {}
    for name, module in sorted(sys.modules.items()):
        if name not in {"inkling_quant_lab", "scripts"} and not name.startswith(
            ("inkling_quant_lab.", "scripts.")
        ):
            continue
        source_value = getattr(module, "__file__", None)
        if not isinstance(source_value, str):
            raise PreflightError(f"local module {name} has no concrete source file")
        source = Path(source_value).absolute().resolve(strict=True)
        if source.suffix != ".py" or not source.is_relative_to(project_root):
            raise PreflightError(f"local module {name} did not import from project .py source")
        relative = source.relative_to(project_root).as_posix()
        source_sha256 = _sha256_file(source)
        if current_files.get(relative) != source_sha256:
            raise PreflightError(f"local module {name} source is absent or stale in the closure")
        cached_value = getattr(module, "__cached__", None)
        if not isinstance(cached_value, str):
            raise PreflightError(f"local module {name} has no redirected __cached__ path")
        cached = Path(cached_value).absolute()
        if not cached.is_relative_to(Path("/dev/null")) or cached.exists():
            raise PreflightError(f"local module {name} could resolve unbound bytecode: {cached}")
        observed[name] = {
            "source_path": relative,
            "source_sha256": source_sha256,
            "cached_path": str(cached),
            "cached_exists": False,
        }
    if "inkling_quant_lab" not in observed or "scripts" not in observed:
        raise PreflightError("deferred project package imports were not both observed")
    return observed


def _python_executable_fact(project_root: Path) -> dict[str, Any]:
    """Bind the venv invocation symlink separately from its executable target."""

    raw_invocation = Path(sys.executable)
    _reject_traversal(raw_invocation, field="Python invocation path")
    invocation = raw_invocation.absolute()
    expected = (project_root / ".venv/bin/python").absolute()
    if not raw_invocation.is_absolute() or sys.executable != str(invocation):
        raise PreflightError("Python invocation path must be absolute and traversal-free")
    if invocation != expected:
        raise PreflightError("Python invocation must be the project .venv/bin/python")
    for directory in (project_root / ".venv", project_root / ".venv/bin"):
        absolute = directory.absolute()
        if absolute.is_symlink() or not absolute.is_dir():
            raise PreflightError("virtual-environment path components must be real directories")
        if absolute.resolve(strict=True) != absolute:
            raise PreflightError("virtual-environment path components must not traverse symlinks")
    if not invocation.is_symlink():
        raise PreflightError("project .venv/bin/python must be a real symlink")
    resolved = invocation.resolve(strict=True)
    return {
        "invocation_path": str(invocation),
        "resolved_path": str(resolved),
        **_file_fact(resolved),
    }


def _environment_contract(project_root: Path) -> dict[str, Any]:
    import torch

    packages = (
        "inkling-quant-lab",
        "torch",
        "transformers",
        "tokenizers",
        "numpy",
        "safetensors",
        "huggingface-hub",
    )
    pyvenv = _project_file(project_root, Path(".venv/pyvenv.cfg"))
    return {
        "hardware_label": HARDWARE_LABEL,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "software": {name: importlib.metadata.version(name) for name in packages},
        "torch_cpu": {
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "required_supported_engine": (
                "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else None
            ),
        },
        "offline_environment": dict(OFFLINE_ENVIRONMENT),
        "python_executable": _python_executable_fact(project_root),
        "virtual_environment": {
            "pyvenv_cfg_path": ".venv/pyvenv.cfg",
            **_file_fact(pyvenv),
        },
    }


def _dataset_fact(path: Path) -> dict[str, Any]:
    fact = _file_fact(path)
    text = path.read_text(encoding="utf-8")
    stories = tuple(story.strip() for story in text.split("<|endoftext|>") if story.strip())
    return {**fact, "story_count": len(stories)}


def _load_preregistration(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise ValueError(f"non-finite JSON number is forbidden: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise PreflightError(f"unable to parse preregistration JSON: {error}") from error
    if not isinstance(value, dict):
        raise PreflightError("preregistration must be one JSON object")
    return cast(dict[str, Any], value)


def _exact_keys(value: Any, expected: set[str], *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise PreflightError(f"{field} has an unexpected schema")
    return cast(dict[str, Any], value)


def _validate_preregistration(
    preregistration: dict[str, Any],
    *,
    ordinal: Literal[1, 2],
    relative_attempt: Path,
    relative_record: Path,
    current_files: dict[str, str],
    scientific: dict[str, str],
    dataset: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    top = _exact_keys(
        preregistration,
        {
            "schema_version",
            "status",
            "protocol_id",
            "created_at_utc",
            "outcomes",
            "bindings",
            "environment_contract",
            "planned_attempts",
            "primary_analysis",
            "claim_boundary",
        },
        field="preregistration",
    )
    if (
        top["schema_version"] != "confirmatory-quality-preregistration-v1"
        or top["status"] != "locked_before_execution"
        or top["protocol_id"] != PROTOCOL_ID
    ):
        raise PreflightError("preregistration lock identity is invalid")
    if not isinstance(top["created_at_utc"], str):
        raise PreflightError("preregistration creation timestamp must be a string")
    try:
        created = datetime.fromisoformat(top["created_at_utc"])
    except ValueError as error:
        raise PreflightError("preregistration creation timestamp is invalid") from error
    if created.tzinfo is None:
        raise PreflightError("preregistration creation timestamp must be timezone-aware")
    outcomes = _exact_keys(
        top["outcomes"],
        {"holdout_model_forward_executed", "holdout_outcomes_inspected"},
        field="preregistration.outcomes",
    )
    if outcomes != {
        "holdout_model_forward_executed": False,
        "holdout_outcomes_inspected": False,
    }:
        raise PreflightError("preregistration is not outcome-blind")

    bindings = _exact_keys(
        top["bindings"],
        {
            "files",
            "resolved_config_sha256",
            "protocol_definition_sha256",
            "dataset",
        },
        field="preregistration.bindings",
    )
    declared_files = bindings["files"]
    if not isinstance(declared_files, dict) or any(
        not isinstance(path, str) or not isinstance(digest, str)
        for path, digest in declared_files.items()
    ):
        raise PreflightError("preregistration file bindings must map paths to SHA-256 strings")
    missing = sorted(set(current_files) - set(declared_files))
    if missing:
        raise PreflightError(f"preregistration omits required file bindings: {missing}")
    unexpected = sorted(set(declared_files) - set(current_files))
    if unexpected:
        raise PreflightError(f"preregistration contains non-runtime file bindings: {unexpected}")
    for relative_text, actual_sha256 in current_files.items():
        if declared_files[relative_text] != actual_sha256:
            raise PreflightError(f"stale preregistered file binding: {relative_text}")

    resolved = _exact_keys(
        bindings["resolved_config_sha256"],
        {"baseline", "candidate"},
        field="preregistration.bindings.resolved_config_sha256",
    )
    if resolved != {"baseline": scientific["baseline"], "candidate": scientific["candidate"]}:
        raise PreflightError("resolved config hashes differ from preregistration")
    if (
        scientific["baseline"] != BASELINE_CONFIG_SHA256
        or scientific["candidate"] != CANDIDATE_CONFIG_SHA256
    ):
        raise PreflightError("resolved configs differ from the checked confirmatory identities")
    if bindings["protocol_definition_sha256"] != scientific["protocol_definition"]:
        raise PreflightError("protocol definition hash differs from preregistration")

    declared_dataset = _exact_keys(
        bindings["dataset"],
        {"file_sha256", "size_bytes", "story_count"},
        field="preregistration.bindings.dataset",
    )
    expected_dataset = {
        "file_sha256": OFFICIAL_DATASET_SHA256,
        "size_bytes": OFFICIAL_DATASET_SIZE_BYTES,
        "story_count": OFFICIAL_DATASET_STORY_COUNT,
    }
    if declared_dataset != expected_dataset or dataset != {
        "sha256": OFFICIAL_DATASET_SHA256,
        "size_bytes": OFFICIAL_DATASET_SIZE_BYTES,
        "story_count": OFFICIAL_DATASET_STORY_COUNT,
    }:
        raise PreflightError("dataset identity differs from the official preregistered file")
    if top["environment_contract"] != environment:
        raise PreflightError("current execution environment differs from preregistration")

    planned = top["planned_attempts"]
    if not isinstance(planned, list) or len(planned) != 2:
        raise PreflightError("preregistration must declare exactly two planned attempts")
    expected_plans = [
        {
            "ordinal": planned_ordinal,
            "directory": (ATTEMPT_ROOT / f"attempt-{planned_ordinal}").as_posix(),
            "record_path": (ATTEMPT_ROOT / f"attempt-{planned_ordinal}" / "record.json").as_posix(),
        }
        for planned_ordinal in (1, 2)
    ]
    if planned != expected_plans:
        raise PreflightError("planned attempt paths differ from the fixed append-only paths")
    selected = planned[ordinal - 1]
    if (
        selected["directory"] != relative_attempt.as_posix()
        or selected["record_path"] != relative_record.as_posix()
    ):
        raise PreflightError("selected attempt path differs from preregistration")
    if not isinstance(top["primary_analysis"], dict) or not isinstance(top["claim_boundary"], list):
        raise PreflightError("primary analysis and claim boundary must retain their JSON schemas")


def _command(*, ordinal: Literal[1, 2], dataset: Path, relative_record: Path) -> list[str]:
    return [
        sys.executable,
        "-s",
        "-P",
        "-B",
        EVALUATOR.as_posix(),
        BASELINE_CONFIG.as_posix(),
        CANDIDATE_CONFIG.as_posix(),
        "--dataset",
        str(dataset),
        "--expected-dataset-sha256",
        OFFICIAL_DATASET_SHA256,
        "--hardware-label",
        HARDWARE_LABEL,
        "--confirmatory-protocol",
        PROTOCOL_CONFIG.as_posix(),
        "--execution-ordinal",
        str(ordinal),
        "--output",
        relative_record.as_posix(),
    ]


def _record_validation(path: Path, *, ordinal: Literal[1, 2]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.exists() or path.is_symlink(),
        "is_regular_non_symlink": False,
        "size_bytes": None,
        "sha256": None,
        "schema_version": None,
        "execution_ordinal": None,
        "provisional_status": None,
        "confirmatory_claim_ready": None,
        "valid": False,
        "validation_error": None,
    }
    if path.is_symlink() or not path.is_file():
        result["validation_error"] = "record.json is missing or is not a regular file"
        return result
    result["is_regular_non_symlink"] = True
    result["size_bytes"] = path.stat().st_size
    result["sha256"] = _sha256_file(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError("record must be a JSON object")
        result["schema_version"] = record.get("schema_version")
        status = record.get("confirmatory_status")
        if not isinstance(status, dict):
            raise ValueError("record is missing confirmatory_status")
        result["execution_ordinal"] = status.get("execution_ordinal")
        result["provisional_status"] = status.get("status")
        result["confirmatory_claim_ready"] = status.get("confirmatory_claim_ready")
        protocol = record.get("protocol")
        confirmatory = protocol.get("confirmatory") if isinstance(protocol, dict) else None
        definition = confirmatory.get("definition") if isinstance(confirmatory, dict) else None
        if record.get("schema_version") != "public-moe-native-linear-quality-v2":
            raise ValueError("record schema is not confirmatory v2")
        if not isinstance(definition, dict) or definition.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("record protocol identity is missing or incorrect")
        expected_status = {
            "status": "provisional_until_2_clean_executions",
            "confirmatory_claim_ready": False,
            "required_clean_process_executions": 2,
            "execution_ordinal": ordinal,
            "within_execution_decision_is_overall_confirmatory_claim": False,
            "pair_verification_required": True,
        }
        if any(status.get(key) != value for key, value in expected_status.items()):
            raise ValueError("record ordinal or provisional claim boundary is invalid")
        if not isinstance(status.get("within_execution_noninferiority_passed"), bool):
            raise ValueError("record is missing its within-execution statistical decision")
        result["valid"] = True
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        result["validation_error"] = str(error)
    return result


def _log_fact(path: Path) -> dict[str, Any]:
    fact = _file_fact(path)
    return {"path": path.name, **fact}


def _freeze_attempt(directory: Path) -> None:
    directories = [directory]
    files: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            files.append(path)
        else:
            raise ValueError(f"attempt contains a non-regular entry: {path}")
    for path in files:
        path.chmod(0o444)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555)
    for path in files:
        if stat.S_IMODE(path.stat().st_mode) != 0o444:
            raise RuntimeError(f"failed to freeze attempt file: {path}")
    for path in directories:
        if stat.S_IMODE(path.stat().st_mode) != 0o555:
            raise RuntimeError(f"failed to freeze attempt directory: {path}")


def run_attempt(
    *,
    ordinal: Literal[1, 2],
    dataset_path: Path,
    preregistration_path: Path,
    project_root: Path | None = None,
) -> AttemptResult:
    """Execute one fixed attempt and return only after completion evidence is sealed."""

    _require_source_import_contract()
    if ordinal not in (1, 2):
        raise ValueError("confirmatory attempt ordinal must be 1 or 2")
    root = (Path.cwd() if project_root is None else project_root).absolute()
    if root.is_symlink() or not root.is_dir() or root.resolve(strict=True) != root:
        raise ValueError("project root must be a real non-symlink directory")
    _reject_traversal(preregistration_path, field="preregistration path")
    expected_preregistration = (root / PROJECT_PREREGISTRATION).absolute()
    requested_preregistration = (
        preregistration_path.expanduser().absolute()
        if preregistration_path.is_absolute()
        else (root / preregistration_path).absolute()
    )
    if requested_preregistration != expected_preregistration:
        raise ValueError(
            f"only the checked preregistration path is accepted: {PROJECT_PREREGISTRATION}"
        )
    preregistration_file = _regular_input_path(
        requested_preregistration, field="preregistration path"
    )
    dataset_file = _regular_input_path(dataset_path, field="dataset path")
    attempt_parent = _ensure_attempt_parent(root)
    _validate_attempt_sequence(attempt_parent, ordinal)
    attempt, relative_attempt, relative_record = _attempt_paths(root, ordinal)
    if attempt.parent != attempt_parent:
        raise RuntimeError("derived attempt parent is inconsistent")
    try:
        attempt.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"confirmatory attempt directory already exists: {attempt}"
        ) from error

    started_at = _utc_now()
    started_clock = time.perf_counter()
    command = _command(ordinal=ordinal, dataset=dataset_file, relative_record=relative_record)
    preregistration_fact: dict[str, Any] = {"sha256": None, "size_bytes": None}
    dataset_fact: dict[str, Any] = {
        "sha256": None,
        "size_bytes": None,
        "story_count": None,
    }
    current_files: dict[str, str] = {}
    scientific: dict[str, str] = {
        "baseline": "unavailable",
        "candidate": "unavailable",
        "protocol_definition": "unavailable",
    }
    environment: dict[str, Any] = {}
    local_imports: dict[str, dict[str, Any]] = {}
    materialization_failure: BaseException | None = None
    try:
        preregistration_fact = _file_fact(preregistration_file)
        dataset_fact = _dataset_fact(dataset_file)
        current_files = _required_file_bindings(root)
        scientific = _resolved_scientific_bindings(root)
        local_imports = _local_import_provenance(root, current_files)
        environment = _environment_contract(root)
    except BaseException as error:
        materialization_failure = error
    start_record = {
        "schema_version": "confirmatory-quality-attempt-start-v1",
        "status": "started_before_preflight_and_model_execution",
        "created_at_utc": started_at,
        "ordinal": ordinal,
        "attempt_directory": relative_attempt.as_posix(),
        "planned_record_path": relative_record.as_posix(),
        "command_argv": command,
        "shell": False,
        "working_directory": str(root),
        "preregistration": {
            "path": PROJECT_PREREGISTRATION.as_posix(),
            **preregistration_fact,
        },
        "dataset": {"path": str(dataset_file), **dataset_fact},
        "file_sha256": current_files,
        "resolved_config_sha256": {
            "baseline": scientific["baseline"],
            "candidate": scientific["candidate"],
        },
        "protocol_definition_sha256": scientific["protocol_definition"],
        "environment_contract": environment,
        "selected_child_environment": dict(OFFLINE_ENVIRONMENT),
        "source_import_contract": _source_import_contract(local_imports),
        "stdout_path": "stdout.log",
        "stderr_path": "stderr.log",
        "outcome_fields_present": False,
        "preflight_materialization_failure": (
            None
            if materialization_failure is None
            else {
                "type": type(materialization_failure).__name__,
                "message": str(materialization_failure),
            }
        ),
    }
    _atomic_exclusive_json(attempt / "start.json", start_record)

    stdout_path = attempt / "stdout.log"
    stderr_path = attempt / "stderr.log"
    subprocess_launched = False
    subprocess_returncode: int | None = None
    failure: BaseException | None = None
    preflight_complete = False
    with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
        try:
            if materialization_failure is not None:
                raise PreflightError(
                    "unable to materialize preregistration preflight facts: "
                    f"{type(materialization_failure).__name__}: {materialization_failure}"
                ) from materialization_failure
            preregistration = _load_preregistration(preregistration_file)
            _validate_preregistration(
                preregistration,
                ordinal=ordinal,
                relative_attempt=relative_attempt,
                relative_record=relative_record,
                current_files=current_files,
                scientific=scientific,
                dataset=dataset_fact,
                environment=environment,
            )
            preflight_complete = True
            child_environment = dict(os.environ)
            for name in FORBIDDEN_AMBIENT_IMPORT_ENVIRONMENT:
                child_environment.pop(name, None)
            child_environment.update(OFFLINE_ENVIRONMENT)
            subprocess_launched = True
            completed = subprocess.run(
                command,
                cwd=root,
                env=child_environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
                shell=False,
            )
            subprocess_returncode = int(completed.returncode)
        except BaseException as error:
            failure = error
            message = f"{type(error).__name__}: {error}\n".encode("utf-8", errors="replace")
            stderr_handle.write(message)
        finally:
            stdout_handle.flush()
            stderr_handle.flush()
            os.fsync(stdout_handle.fileno())
            os.fsync(stderr_handle.fileno())

    record_path = attempt / "record.json"
    record_validation = _record_validation(record_path, ordinal=ordinal)
    if failure is not None and not preflight_complete:
        status = "failed_preflight"
    elif failure is not None:
        status = "runner_exception"
    elif subprocess_returncode != 0:
        status = "evaluator_failed"
    elif not record_validation["valid"]:
        status = "invalid_record"
    else:
        status = "attempt_execution_complete"
    exit_code = 0 if status == "attempt_execution_complete" else 1
    elapsed = time.perf_counter() - started_clock
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError("attempt elapsed time is invalid")
    completion_record = {
        "schema_version": "confirmatory-quality-attempt-completion-v1",
        "status": status,
        "created_at_utc": _utc_now(),
        "ordinal": ordinal,
        "attempt_directory": relative_attempt.as_posix(),
        "subprocess_launched": subprocess_launched,
        "preflight_complete": preflight_complete,
        "subprocess_returncode": subprocess_returncode,
        "runner_exit_code": exit_code,
        "elapsed_seconds": elapsed,
        "failure": (
            None if failure is None else {"type": type(failure).__name__, "message": str(failure)}
        ),
        "stdout": _log_fact(stdout_path),
        "stderr": _log_fact(stderr_path),
        "start": _log_fact(attempt / "start.json"),
        "record": record_validation,
        "confirmatory_claim_ready": False,
        "claim_boundary": (
            "attempt execution status only; the within-execution statistical decision is not "
            "an overall confirmatory claim"
        ),
    }
    _atomic_exclusive_json(attempt / "completion.json", completion_record)
    _freeze_attempt(attempt)
    return AttemptResult(attempt_directory=attempt, status=status, exit_code=exit_code)


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinal", type=int, choices=(1, 2), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    """Run one attempt and return a nonzero process status for every failure."""

    try:
        _require_source_import_contract()
    except RuntimeError as error:
        print(f"RuntimeError: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    arguments = _arguments()
    try:
        result = run_attempt(
            ordinal=cast(Literal[1, 2], arguments.ordinal),
            dataset_path=arguments.dataset,
            preregistration_path=arguments.preregistration,
        )
    except (OSError, ValueError, RuntimeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(result.attempt_directory)
    raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()
