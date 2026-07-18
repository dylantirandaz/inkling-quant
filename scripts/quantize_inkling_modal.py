"""Implementation-addressed deployed Modal stages for the exact Inkling Q3_K_M export.

This file is deployed once with ``modal deploy`` after the dashboard hard budget is
active. Stages are then spawned by ``manage_inkling_modal.py`` from sealed concrete
Function IDs; do not use ``modal run`` for paid work. Deployed calls survive local
Wi-Fi loss, while Modal Volumes and marker-last receipts remain the durable ledger.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

ENTRYPOINT_PATH = Path(__file__).resolve()
LOCAL_PROJECT_ROOT = ENTRYPOINT_PATH.parents[1]
LOCAL_SRC_ROOT = LOCAL_PROJECT_ROOT / "src"
if LOCAL_SRC_ROOT.is_dir() and str(LOCAL_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC_ROOT))

import modal  # noqa: E402

EXPECTED_MODAL_VERSION = "1.5.0"
if modal.__version__ != EXPECTED_MODAL_VERSION:
    raise RuntimeError(
        f"This paid boundary requires Modal {EXPECTED_MODAL_VERSION}, got {modal.__version__}"
    )

from inkling_quant_lab.exceptions import ConfigurationError  # noqa: E402
from inkling_quant_lab.gguf.inkling import (  # noqa: E402
    EXPECTED_MODEL_BYTES,
    ControlPlaneProvenance,
    InklingGGUFConfig,
    PaidLaunchAcknowledgement,
    WorkflowPaths,
    audit_inkling_source,
    audit_pinned_inkling_online,
    build_conversion_plan,
    build_quantize_command,
    build_verification_plan,
    inkling_run_id,
    load_inkling_gguf_config,
    modal_stage_resources,
    require_initial_billing_window,
    require_materialize_initial_billing_window,
    require_stage_billing_window,
    validate_deployed_control_plane,
    validate_paid_launch_acknowledgement,
    verify_execution_bindings,
)
from inkling_quant_lab.gguf.publication import (  # noqa: E402
    PublicationIntent,
    PublicationReceipt,
    finalize_publication,
    prepare_publication_intent,
    publication_intent_path,
    publication_receipt_path,
)

CONFIG_PATH = LOCAL_PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_modal.yaml"
DEFAULT_CONFIG = InklingGGUFConfig()
if modal.is_local() and CONFIG_PATH.is_file():
    checked_config = load_inkling_gguf_config(CONFIG_PATH)
    if checked_config.config_hash() != DEFAULT_CONFIG.config_hash():
        raise RuntimeError("Checked Inkling YAML differs from the frozen deployment schema")
if modal.is_local():
    billing_cycle_end_utc = os.environ.get("IQL_MODAL_BILLING_CYCLE_END_CONFIRMED")
    if os.environ.get("IQL_MODAL_WORKSPACE_BUDGET_CONFIRMED") != "800" or not billing_cycle_end_utc:
        raise RuntimeError(
            "Do not run or deploy this paid App directly. First activate the $800 Modal "
            "workspace hard budget, export IQL_MODAL_WORKSPACE_BUDGET_CONFIRMED=800 and "
            "IQL_MODAL_BILLING_CYCLE_END_CONFIRMED=YYYY-MM-DDTHH:MM:SSZ, then use "
            "scripts/manage_inkling_modal.py deploy."
        )
    try:
        require_initial_billing_window(DEFAULT_CONFIG, billing_cycle_end_utc)
    except ConfigurationError:
        if os.environ.get("IQL_MODAL_SHORT_CYCLE_CONFIRMED") != billing_cycle_end_utc:
            raise RuntimeError(
                "A short-cycle deployment requires IQL_MODAL_SHORT_CYCLE_CONFIRMED to "
                "exactly equal IQL_MODAL_BILLING_CYCLE_END_CONFIRMED"
            ) from None
        require_materialize_initial_billing_window(DEFAULT_CONFIG, billing_cycle_end_utc)

DEBIAN_IMAGE = (
    "debian:bookworm-slim@sha256:7b140f374b289a7c2befc338f42ebe6441b7ea838a042bbd5acbfca6ec875818"
)
REMOTE_PACKAGE = "/root/inkling_quant_lab"
LLAMA_CPP_DIR = Path("/opt/llama.cpp")
SOURCE_MOUNT = Path("/source")
WORK_MOUNT = Path("/work")
FINAL_MOUNT = Path("/final")
RUN_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
SAFE_METADATA_SUFFIXES = {
    ".json",
    ".jinja",
    ".model",
    ".template",
    ".tiktoken",
    ".txt",
}

app = modal.App(DEFAULT_CONFIG.modal.app_name)
source_volume = modal.Volume.from_name(
    DEFAULT_CONFIG.modal.source_volume,
    environment_name=DEFAULT_CONFIG.modal.environment_name,
    create_if_missing=True,
    version=DEFAULT_CONFIG.modal.volume_version,
)
work_volume = modal.Volume.from_name(
    DEFAULT_CONFIG.modal.work_volume,
    environment_name=DEFAULT_CONFIG.modal.environment_name,
    create_if_missing=True,
    version=DEFAULT_CONFIG.modal.volume_version,
)
final_volume = modal.Volume.from_name(
    DEFAULT_CONFIG.modal.final_volume,
    environment_name=DEFAULT_CONFIG.modal.environment_name,
    create_if_missing=True,
    version=DEFAULT_CONFIG.modal.volume_version,
)
hf_secret = modal.Secret.from_name(
    DEFAULT_CONFIG.modal.hf_secret,
    environment_name=DEFAULT_CONFIG.modal.hf_secret_environment,
    required_keys=["HF_TOKEN"],
)

python_image = (
    modal.Image.from_registry(DEBIAN_IMAGE, add_python="3.12")
    .apt_install("ca-certificates")
    .pip_install(
        "pydantic==2.13.4",
        "PyYAML==6.0.3",
        "huggingface-hub==0.36.0",
        "hf-xet==1.4.3",
    )
    .env({"PYTHONPATH": "/root", "HF_HUB_DISABLE_TELEMETRY": "1"})
)
if modal.is_local():
    python_image = python_image.add_local_dir(
        LOCAL_PROJECT_ROOT / "src/inkling_quant_lab", REMOTE_PACKAGE, copy=True
    )

toolchain_image = (
    modal.Image.from_registry(DEBIAN_IMAGE, add_python="3.12")
    .apt_install("build-essential", "ca-certificates", "cmake", "git", "ninja-build")
    .run_commands(
        f"git init {LLAMA_CPP_DIR}",
        f"git -C {LLAMA_CPP_DIR} remote add origin {DEFAULT_CONFIG.toolchain.repository}",
        (f"git -C {LLAMA_CPP_DIR} fetch --depth 1 origin {DEFAULT_CONFIG.toolchain.commit}"),
        f"git -C {LLAMA_CPP_DIR} checkout --detach FETCH_HEAD",
        (
            f"python -m pip install --no-cache-dir -r "
            f"{LLAMA_CPP_DIR}/requirements/requirements-convert_hf_to_gguf.txt"
        ),
        "python -m pip install --no-cache-dir pydantic==2.13.4 PyYAML==6.0.3",
        (
            f"cmake -S {LLAMA_CPP_DIR} -B {LLAMA_CPP_DIR}/build -G Ninja "
            "-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF"
        ),
        (
            f"cmake --build {LLAMA_CPP_DIR}/build --parallel 8 --target "
            "llama-quantize llama-gguf-split"
        ),
        f"git -C {LLAMA_CPP_DIR} rev-parse HEAD > {LLAMA_CPP_DIR}/.iql-build-commit",
        (
            f"sha256sum {LLAMA_CPP_DIR}/build/bin/llama-quantize > "
            f"{LLAMA_CPP_DIR}/.iql-quantize.sha256"
        ),
        (
            f"sha256sum {LLAMA_CPP_DIR}/build/bin/llama-gguf-split > "
            f"{LLAMA_CPP_DIR}/.iql-gguf-split.sha256"
        ),
        (f"python -m pip freeze --all | LC_ALL=C sort > {LLAMA_CPP_DIR}/.iql-python-freeze.txt"),
        (
            f"sha256sum {LLAMA_CPP_DIR}/.iql-python-freeze.txt > "
            f"{LLAMA_CPP_DIR}/.iql-python-freeze.sha256"
        ),
        (
            "dpkg-query -W -f='${Package}=${Version}\\n' | LC_ALL=C sort > "
            f"{LLAMA_CPP_DIR}/.iql-dpkg-inventory.txt"
        ),
        (
            f"sha256sum {LLAMA_CPP_DIR}/.iql-dpkg-inventory.txt > "
            f"{LLAMA_CPP_DIR}/.iql-dpkg-inventory.sha256"
        ),
    )
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
)
if modal.is_local():
    toolchain_image = toolchain_image.add_local_dir(
        LOCAL_PROJECT_ROOT / "src/inkling_quant_lab", REMOTE_PACKAGE, copy=True
    )

MATERIALIZE_RESOURCES = modal_stage_resources(DEFAULT_CONFIG, "materialize_source")
TEXT_CONVERT_RESOURCES = modal_stage_resources(DEFAULT_CONFIG, "convert_text_bf16")
MMPROJ_RESOURCES = modal_stage_resources(DEFAULT_CONFIG, "convert_multimodal_projector")
QUANTIZE_RESOURCES = modal_stage_resources(DEFAULT_CONFIG, "quantize_text")
VERIFY_RESOURCES = modal_stage_resources(DEFAULT_CONFIG, "verify_export")


def _config(config_json: str) -> InklingGGUFConfig:
    config = InklingGGUFConfig.model_validate_json(config_json)
    if config.config_hash() != DEFAULT_CONFIG.config_hash():
        raise RuntimeError("Remote config differs from the checked deployment config")
    return config


def _run_id(config: InklingGGUFConfig, control_plane: ControlPlaneProvenance) -> str:
    return inkling_run_id(config, control_plane.tree_sha256)


def _validate_run_id(run_id: str) -> None:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id contains unsafe characters")


def _remote_config(
    config_json: str,
    run_id: str,
    budget_acknowledgement_json: str,
    control_plane_json: str,
    stage: str,
) -> tuple[InklingGGUFConfig, ControlPlaneProvenance, PaidLaunchAcknowledgement]:
    """Enforce config, orchestration, run namespace, and operator budget acknowledgement."""

    config = _config(config_json)
    control_plane = validate_deployed_control_plane(
        control_plane_json,
        deployment_script=ENTRYPOINT_PATH,
        deployed_package_root=Path(REMOTE_PACKAGE),
    )
    _validate_run_id(run_id)
    expected_run_id = _run_id(config, control_plane)
    if run_id != expected_run_id:
        raise RuntimeError(f"Remote run_id must equal the deterministic run id {expected_run_id}")
    acknowledgement = validate_paid_launch_acknowledgement(
        config,
        budget_acknowledgement_json,
        control_plane_sha256=control_plane.tree_sha256,
    )
    require_stage_billing_window(
        config,
        acknowledgement.billing_cycle_end_utc,
        stage,
        include_startup=False,
    )
    return config, control_plane, acknowledgement


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish deterministic marker metadata once without replacing prior evidence."""

    expected = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Immutable metadata path is unsafe: {path}")
        if path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Refusing to replace immutable metadata: {path}")
        return
    _atomic_json(path, value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError(f"Required JSON receipt is missing: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise RuntimeError(f"JSON receipt is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object at {path}")
    return value


def _sha256(path: Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    try:
        before = path.lstat()
    except FileNotFoundError as error:
        raise RuntimeError(f"Cannot hash missing file: {path}") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"Cannot hash unsafe non-regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    after = path.lstat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def _safe_child(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts)
    try:
        lexical_relative = candidate.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Unsafe path below {root}: {candidate}") from error
    if not lexical_relative.parts or ".." in lexical_relative.parts:
        raise RuntimeError(f"Unsafe path below {root}: {candidate}")
    current = root
    for part in lexical_relative.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"Path below {root} contains a symlink: {current}")
        if not current.exists():
            break
    resolved_root = root.resolve()
    resolved = candidate.resolve(strict=False)
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise RuntimeError(f"Unsafe path below {root}: {candidate}")
    return candidate


def _bind_config(root: Path, config: InklingGGUFConfig) -> None:
    path = _safe_child(root, "resolved_config.json")
    payload = config.canonical_dict()
    if path.exists():
        if _read_json(path) != payload:
            raise RuntimeError("Run directory is already bound to a different config")
        return
    _atomic_json(path, payload)


def _bind_control_plane(root: Path, control_plane: ControlPlaneProvenance) -> None:
    path = _safe_child(root, "control_plane.json")
    payload = control_plane.model_dump(mode="json")
    if path.exists():
        if _read_json(path) != payload:
            raise RuntimeError("Run directory is bound to different orchestration source bytes")
        return
    _atomic_json(path, payload)


def _verify_bound_control_plane(root: Path, control_plane: ControlPlaneProvenance) -> None:
    path = _safe_child(root, "control_plane.json")
    if not path.is_file() or _read_json(path) != control_plane.model_dump(mode="json"):
        raise RuntimeError("Run directory control-plane source binding is missing or changed")


def _verify_receipt_control_plane(
    receipt: Mapping[str, Any],
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
) -> None:
    if (
        receipt.get("config_hash") != config.config_hash()
        or receipt.get("control_plane_sha256") != control_plane.tree_sha256
    ):
        raise RuntimeError("Stage receipt is not bound to this config and control plane")


def _stage_limit(config: InklingGGUFConfig, name: str) -> int:
    matches = [stage.max_attempts for stage in config.modal.stages if stage.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"Missing unique budget entry for stage {name}")
    return matches[0]


def _stage_recovery_limit(config: InklingGGUFConfig, name: str) -> int:
    matches = [stage.max_recovery_attempts for stage in config.modal.stages if stage.name == name]
    if len(matches) != 1:
        raise RuntimeError(f"Missing unique recovery budget entry for stage {name}")
    return matches[0]


InvocationHistoryEntry = tuple[Path, dict[str, Any], str]


def _invocation_event_id(event: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _current_invocation_ids() -> tuple[str, str, str]:
    call_id = modal.current_function_call_id()
    input_id = modal.current_input_id()
    task_id = os.environ.get("MODAL_TASK_ID")
    if (
        not isinstance(call_id, str)
        or not call_id
        or not isinstance(input_id, str)
        or not input_id
        or not task_id
    ):
        raise RuntimeError(
            "Modal invocation call, input, or task ID is unavailable at the paid evidence boundary"
        )
    return call_id, input_id, task_id


def _invocation_history(
    root: Path,
    config: InklingGGUFConfig,
    stage: str,
    control_plane: ControlPlaneProvenance,
    *,
    kind: str,
    limit: int,
) -> list[InvocationHistoryEntry]:
    """Validate the immutable history that authoritatively consumes the cost cap."""

    directory = _safe_child(root, "control", "history")
    if not directory.exists():
        return []
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"Stage {stage} invocation history is unsafe")
    entries: list[InvocationHistoryEntry] = []
    for path in sorted(directory.glob(f"{stage}.{kind}.*.json")):
        event = _read_json(path)
        sequence = event.get("sequence")
        if (
            event.get("schema_version") != "inkling-modal-stage-invocation-v2"
            or event.get("status") != "started"
            or event.get("kind") != kind
            or event.get("limit") != limit
            or event.get("stage") != stage
            or event.get("config_hash") != config.config_hash()
            or event.get("control_plane_sha256") != control_plane.tree_sha256
            or not isinstance(sequence, int)
            or sequence < 1
            or sequence > limit
            or not isinstance(event.get("launch_intent_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(event.get("launch_intent_sha256"))) is None
            or not isinstance(event.get("call_id"), str)
            or not event.get("call_id")
            or not isinstance(event.get("input_id"), str)
            or not event.get("input_id")
            or not isinstance(event.get("task_id"), str)
            or not event.get("task_id")
        ):
            raise RuntimeError(f"Stage {stage} immutable {kind} history drifted")
        event_id = _invocation_event_id(event)
        if path.name != f"{stage}.{kind}.{sequence}.{event_id}.json":
            raise RuntimeError(f"Stage {stage} immutable {kind} filename drifted")
        entries.append((path, event, _sha256(path)))
    entries.sort(key=lambda item: int(item[1]["sequence"]))
    if [entry[1]["sequence"] for entry in entries] != list(range(1, len(entries) + 1)):
        raise RuntimeError(f"Stage {stage} immutable {kind} sequence is not contiguous")
    previous_sha256: str | None = None
    for _, event, sha256 in entries:
        if event.get("previous_history_sha256") != previous_sha256:
            raise RuntimeError(f"Stage {stage} immutable {kind} chain drifted")
        previous_sha256 = sha256
    return entries


def _record_invocation_start(
    root: Path,
    config: InklingGGUFConfig,
    stage: str,
    control_plane: ControlPlaneProvenance,
    launch_intent_sha256: str,
    *,
    kind: str,
    sequence: int,
    limit: int,
    previous_history_sha256: str | None,
    call_id: str,
    input_id: str,
    task_id: str,
) -> InvocationHistoryEntry:
    """Append one immutable authoritative start event."""

    event = {
        "schema_version": "inkling-modal-stage-invocation-v2",
        "status": "started",
        "kind": kind,
        "sequence": sequence,
        "limit": limit,
        "stage": stage,
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "launch_intent_sha256": launch_intent_sha256,
        "call_id": call_id,
        "input_id": input_id,
        "task_id": task_id,
        "previous_history_sha256": previous_history_sha256,
    }
    event_id = _invocation_event_id(event)
    history = _safe_child(root, "control", "history")
    history.mkdir(parents=True, exist_ok=True)
    path = _safe_child(history, f"{stage}.{kind}.{sequence}.{event_id}.json")
    _immutable_json(path, event)
    return path, event, _sha256(path)


def _begin_stage_invocation(
    root: Path,
    config: InklingGGUFConfig,
    stage: str,
    control_plane: ControlPlaneProvenance,
    launch_intent_sha256: str,
    *,
    kind: str,
    counter_name: str,
    limit: int,
) -> int:
    ledger_path = _safe_child(root, "control", f"{stage}.{counter_name}.json")
    latest = _read_json(ledger_path) if ledger_path.exists() else {counter_name: 0}
    history = _invocation_history(
        root,
        config,
        stage,
        control_plane,
        kind=kind,
        limit=limit,
    )
    latest_count = latest.get(counter_name)
    if (
        not isinstance(latest_count, int)
        or latest_count < 0
        or latest_count > len(history)
        or (
            latest_count
            and (
                latest.get("limit") != limit
                or latest.get("config_hash") != config.config_hash()
                or latest.get("control_plane_sha256") != control_plane.tree_sha256
                or latest.get("last_history_path")
                != history[latest_count - 1][0].relative_to(root).as_posix()
                or latest.get("last_history_sha256") != history[latest_count - 1][2]
                or latest.get("launch_intent_sha256")
                != history[latest_count - 1][1]["launch_intent_sha256"]
                or latest.get("last_call_id") != history[latest_count - 1][1]["call_id"]
                or latest.get("last_input_id") != history[latest_count - 1][1]["input_id"]
                or latest.get("last_task_id") != history[latest_count - 1][1]["task_id"]
            )
        )
    ):
        raise RuntimeError(f"Stage {stage} {kind} ledger is malformed or drifted")

    call_id, input_id, task_id = _current_invocation_ids()
    if history and (
        history[-1][1].get("call_id") == call_id
        and history[-1][1].get("input_id") == input_id
        and history[-1][1].get("task_id") == task_id
        and history[-1][1].get("launch_intent_sha256") == launch_intent_sha256
    ):
        entry = history[-1]
    else:
        if len(history) >= limit:
            raise RuntimeError(f"Stage {stage} exceeded its configured {limit}-{kind} cost cap")
        entry = _record_invocation_start(
            root,
            config,
            stage,
            control_plane,
            launch_intent_sha256,
            kind=kind,
            sequence=len(history) + 1,
            limit=limit,
            previous_history_sha256=None if not history else history[-1][2],
            call_id=call_id,
            input_id=input_id,
            task_id=task_id,
        )
        history.append(entry)

    sequence = int(entry[1]["sequence"])
    _atomic_json(
        ledger_path,
        {
            counter_name: sequence,
            "limit": limit,
            "config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "launch_intent_sha256": entry[1]["launch_intent_sha256"],
            "last_call_id": entry[1]["call_id"],
            "last_input_id": entry[1]["input_id"],
            "last_task_id": entry[1]["task_id"],
            "last_history_path": entry[0].relative_to(root).as_posix(),
            "last_history_sha256": entry[2],
        },
    )
    return sequence


def _begin_attempt(
    root: Path,
    config: InklingGGUFConfig,
    stage: str,
    control_plane: ControlPlaneProvenance,
    launch_intent_sha256: str,
) -> int:
    return _begin_stage_invocation(
        root,
        config,
        stage,
        control_plane,
        launch_intent_sha256,
        kind="attempt",
        counter_name="attempts",
        limit=_stage_limit(config, stage),
    )


def _begin_recovery(
    root: Path,
    config: InklingGGUFConfig,
    stage: str,
    control_plane: ControlPlaneProvenance,
    launch_intent_sha256: str,
) -> int:
    """Commit one separately budgeted publication-recovery invocation."""
    return _begin_stage_invocation(
        root,
        config,
        stage,
        control_plane,
        launch_intent_sha256,
        kind="recovery",
        counter_name="recoveries",
        limit=_stage_recovery_limit(config, stage),
    )


StageFunction = Callable[[str, str, str, str], dict[str, Any]]
FailureContext = tuple[
    Path,
    InklingGGUFConfig,
    str,
    ControlPlaneProvenance,
    PaidLaunchAcknowledgement,
    modal.Volume,
]
_FAILURE_CONTEXT: ContextVar[FailureContext | None] = ContextVar(
    "inkling_stage_failure_context",
    default=None,
)


def _arm_failure_recording(
    root: Path,
    config: InklingGGUFConfig,
    stage: str,
    control_plane: ControlPlaneProvenance,
    acknowledgement: PaidLaunchAcknowledgement,
    volume: modal.Volume,
) -> None:
    _FAILURE_CONTEXT.set((root, config, stage, control_plane, acknowledgement, volume))


def _record_terminal_failure(context: FailureContext, error: BaseException) -> None:
    root, config, stage, control_plane, acknowledgement, volume = context
    call_id, input_id, task_id = _current_invocation_ids()
    receipt: dict[str, Any] = {
        "schema_version": "inkling-modal-stage-outcome-v2",
        "status": "failed",
        "stage": stage,
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "launch_intent_sha256": acknowledgement.launch_intent_sha256,
        "call_id": call_id,
        "input_id": input_id,
        "task_id": task_id,
        "exception_type": f"{type(error).__module__}.{type(error).__qualname__}",
    }
    for kind in ("attempts", "recoveries"):
        ledger = _safe_child(root, "control", f"{stage}.{kind}.json")
        if ledger.is_file() and not ledger.is_symlink():
            receipt[f"{kind}_ledger_sha256"] = _sha256(ledger)
    outcome_id = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    path = _safe_child(root, "control", "outcomes", f"{stage}.failed.{outcome_id}.json")
    _immutable_json(path, receipt)
    volume.commit()


def _capture_stage_failures(stage: str) -> Callable[[StageFunction], StageFunction]:
    """Record a redacted immutable terminal outcome after the stage arms its context."""

    def decorate(function: StageFunction) -> StageFunction:
        @wraps(function)
        def wrapped(
            config_json: str,
            run_id: str,
            budget_acknowledgement_json: str,
            control_plane_json: str,
        ) -> dict[str, Any]:
            token = _FAILURE_CONTEXT.set(None)
            try:
                return function(
                    config_json,
                    run_id,
                    budget_acknowledgement_json,
                    control_plane_json,
                )
            except BaseException as error:
                context = _FAILURE_CONTEXT.get()
                if context is not None:
                    if context[2] != stage:
                        raise RuntimeError("Failure-recording stage context drifted") from error
                    try:
                        _record_terminal_failure(context, error)
                    except BaseException as recording_error:
                        error.add_note(
                            "Could not commit the immutable terminal failure receipt: "
                            f"{type(recording_error).__module__}."
                            f"{type(recording_error).__qualname__}"
                        )
                raise
            finally:
                _FAILURE_CONTEXT.reset(token)

        return wrapped

    return decorate


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if path.is_symlink() or not path.is_file() or not resolved.is_relative_to(root.resolve()):
        raise RuntimeError(f"Refusing unsafe output file {path}")
    return {
        "path": str(resolved.relative_to(root.resolve())),
        "sha256": _sha256(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _verify_records(root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise RuntimeError("A successful stage must record at least one output")
    for record in records:
        relative = record.get("path")
        if not isinstance(relative, str):
            raise RuntimeError("Output receipt contains a non-string path")
        path = _safe_child(root, relative)
        try:
            before = path.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(f"Recorded output is missing or unsafe: {relative}") from error
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"Recorded output is missing or unsafe: {relative}")
        if before.st_size != record.get("size_bytes"):
            raise RuntimeError(f"Recorded output size changed: {relative}")
        if _sha256(path) != record.get("sha256"):
            raise RuntimeError(f"Recorded output checksum changed: {relative}")
        after = path.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"Recorded output changed while hashing: {relative}")


def _verify_inventory_membership(root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    expected: set[str] = set()
    for record in records:
        relative = record.get("path")
        if not isinstance(relative, str):
            raise RuntimeError("Inventory contains a non-string path")
        _safe_child(root, relative)
        if relative in expected:
            raise RuntimeError(f"Inventory contains a duplicate path: {relative}")
        expected.add(relative)

    observed: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root).as_posix()
        mode = candidate.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"Inventory tree contains a symlink: {relative}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"Inventory tree contains a special file: {relative}")
        observed.add(relative)
    if observed != expected:
        raise RuntimeError(
            "Inventory tree differs from its receipt "
            f"(missing={sorted(expected - observed)}, unrecorded={sorted(observed - expected)})"
        )


def _repairable_source_drift(
    snapshot: Path,
    records: Mapping[str, Any],
) -> list[str]:
    """Hash a complete source inventory and identify regular files safe to redownload."""

    drifted: list[str] = []
    for filename, raw_record in sorted(records.items()):
        if not isinstance(filename, str) or not isinstance(raw_record, Mapping):
            raise RuntimeError("Source inventory contains malformed records")
        if raw_record.get("path") != filename:
            raise RuntimeError("Source inventory key/path binding changed")
        path = _safe_child(snapshot, filename)
        if path.is_symlink():
            raise RuntimeError(f"Source inventory contains an unsafe symlink: {filename}")
        if not path.exists():
            drifted.append(filename)
            continue
        if not path.is_file():
            raise RuntimeError(f"Source inventory contains a non-file: {filename}")
        if path.stat().st_size != raw_record.get("size_bytes"):
            drifted.append(filename)
            continue
        if _sha256(path) != raw_record.get("sha256"):
            drifted.append(filename)
    return drifted


def _clean_old_partials(root: Path, prefix: str) -> None:
    for candidate in root.glob(f"{prefix}*"):
        if candidate.is_symlink() or not candidate.is_dir() or candidate.parent != root:
            raise RuntimeError(f"Unsafe partial stage path: {candidate}")
        shutil.rmtree(candidate)


def _cheap_success_receipt(
    path: Path,
    *,
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
) -> dict[str, Any]:
    """Reject an unavailable predecessor without doing expensive multi-TB hashing."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Required predecessor success receipt is missing: {path.name}")
    receipt = _read_json(path)
    if receipt.get("status") != "success" and receipt.get("verified") is not True:
        raise RuntimeError(f"Required predecessor receipt is not successful: {path.name}")
    _verify_receipt_control_plane(receipt, config, control_plane)
    return receipt


def _pending_publication(
    run_root: Path,
    *,
    config: InklingGGUFConfig,
    stage: str,
    canonical_name: str,
) -> tuple[Path, Path] | None:
    intent_path = publication_intent_path(run_root, stage)
    if not intent_path.exists():
        return None
    if intent_path.is_symlink() or not intent_path.is_file():
        raise RuntimeError(f"Unsafe publication intent for {stage}")
    intent = PublicationIntent.model_validate(_read_json(intent_path))
    if intent.config_hash != config.config_hash() or intent.stage != stage:
        raise RuntimeError(f"Publication intent is not bound to {stage}")
    if intent.canonical_directory != canonical_name:
        raise RuntimeError(f"Publication intent has the wrong canonical directory for {stage}")
    partial = _safe_child(run_root, intent.partial_directory)
    canonical = _safe_child(run_root, intent.canonical_directory)
    return partial, canonical


def _prepare_and_commit_publication(
    run_root: Path,
    *,
    config: InklingGGUFConfig,
    stage: str,
    partial: Path,
    canonical: Path,
    outputs: Sequence[Mapping[str, Any]],
    volume: modal.Volume,
) -> PublicationReceipt:
    prepare_publication_intent(
        run_root,
        config_hash=config.config_hash(),
        stage=stage,
        partial_directory=partial,
        canonical_directory=canonical,
        outputs=outputs,
    )
    volume.commit()
    publication = finalize_publication(
        run_root,
        config_hash=config.config_hash(),
        stage=stage,
        partial_directory=partial,
        canonical_directory=canonical,
    )
    volume.commit()
    return publication


def _recover_and_commit_publication(
    run_root: Path,
    *,
    config: InklingGGUFConfig,
    stage: str,
    canonical_name: str,
    volume: modal.Volume,
) -> PublicationReceipt | None:
    pending = _pending_publication(
        run_root,
        config=config,
        stage=stage,
        canonical_name=canonical_name,
    )
    if pending is None:
        return None
    partial, canonical = pending
    publication = finalize_publication(
        run_root,
        config_hash=config.config_hash(),
        stage=stage,
        partial_directory=partial,
        canonical_directory=canonical,
    )
    volume.commit()
    return publication


def _publication_evidence(
    run_root: Path,
    stage: str,
    publication: PublicationReceipt,
) -> dict[str, Any]:
    return {
        "publication_intent_sha256": _sha256(publication_intent_path(run_root, stage)),
        "publication_receipt_sha256": _sha256(publication_receipt_path(run_root, stage)),
        "outputs": [record.model_dump(mode="json") for record in publication.outputs],
    }


def _verify_publication_evidence(
    run_root: Path,
    *,
    config: InklingGGUFConfig,
    stage: str,
    receipt: Mapping[str, Any],
    verify_outputs: bool = True,
) -> None:
    intent_path = publication_intent_path(run_root, stage)
    publication_path = publication_receipt_path(run_root, stage)
    if receipt.get("publication_intent_sha256") != _sha256(intent_path) or receipt.get(
        "publication_receipt_sha256"
    ) != _sha256(publication_path):
        raise RuntimeError(f"Publication metadata hashes changed for {stage}")
    intent = PublicationIntent.model_validate(_read_json(intent_path))
    publication = PublicationReceipt.model_validate(_read_json(publication_path))
    expected_outputs = [record.model_dump(mode="json") for record in publication.outputs]
    canonical_from_intent = [
        {
            **record.model_dump(mode="json"),
            "path": (Path(intent.canonical_directory) / record.path).as_posix(),
        }
        for record in intent.outputs
    ]
    if (
        intent.config_hash != config.config_hash()
        or publication.config_hash != config.config_hash()
        or intent.stage != stage
        or publication.stage != stage
        or publication.canonical_directory != intent.canonical_directory
        or publication.intent_sha256 != _sha256(intent_path)
        or expected_outputs != canonical_from_intent
        or receipt.get("outputs") != expected_outputs
    ):
        raise RuntimeError(f"Publication metadata is not bound to {stage}")
    if not verify_outputs:
        return
    canonical = _safe_child(run_root, publication.canonical_directory)
    if canonical.is_symlink() or not canonical.is_dir():
        raise RuntimeError(f"Published directory is missing or unsafe for {stage}")
    observed: set[str] = set()
    for candidate in canonical.rglob("*"):
        relative = candidate.relative_to(run_root).as_posix()
        mode = candidate.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise RuntimeError(f"Published directory contains a symlink for {stage}: {relative}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise RuntimeError(
                f"Published directory contains a special file for {stage}: {relative}"
            )
        observed.add(relative)
    expected_paths = {str(record["path"]) for record in expected_outputs}
    if observed != expected_paths:
        raise RuntimeError(f"Published directory inventory changed for {stage}")
    _verify_records(run_root, expected_outputs)


def _run_streaming(command: Sequence[str], *, log_path: Path, cwd: Path | None = None) -> None:
    if not command or any("\x00" in argument for argument in command):
        raise RuntimeError("Invalid subprocess argv")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=False,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command {command[0]} failed with exit code {return_code}")


def _verify_materialized_source(
    config: InklingGGUFConfig,
    run_id: str,
    control_plane: ControlPlaneProvenance,
) -> dict[str, Any]:
    """Revalidate the complete local snapshot behind an immutable success receipt."""

    run_root = _safe_child(SOURCE_MOUNT, "runs", run_id)
    _verify_bound_control_plane(run_root, control_plane)
    snapshot = _safe_child(run_root, "snapshot")
    receipt_path = _safe_child(run_root, "source.success.json")
    inventory_path = _safe_child(run_root, "source_inventory.json")
    source_config_path = _safe_child(snapshot, "config.json")
    weight_index_path = _safe_child(snapshot, "model.safetensors.index.json")
    receipt = _read_json(receipt_path)
    if (
        receipt.get("verified") is not True
        or receipt.get("config_hash") != config.config_hash()
        or receipt.get("control_plane_sha256") != control_plane.tree_sha256
        or receipt.get("model_id") != config.source.model_id
        or receipt.get("revision") != config.source.revision
        or receipt.get("license") != config.source.license
        or receipt.get("source_dir") != str(snapshot)
        or receipt.get("inventory_sha256") != _sha256(inventory_path)
        or receipt.get("source_config_sha256") != _sha256(source_config_path)
    ):
        raise RuntimeError("Materialized source success receipt is not bound to this run")
    inventory = _read_json(inventory_path)
    records = inventory.get("files")
    if (
        inventory.get("config_hash") != config.config_hash()
        or inventory.get("model_id") != config.source.model_id
        or inventory.get("revision") != config.source.revision
        or not isinstance(records, dict)
        or len(records) != inventory.get("required_file_count")
        or len(records) != receipt.get("file_count")
    ):
        raise RuntimeError("Materialized source inventory is malformed or incomplete")
    source_records = list(records.values())
    _verify_inventory_membership(snapshot, source_records)
    _verify_records(snapshot, source_records)
    weight_index = _read_json(weight_index_path)
    audit = audit_inkling_source(
        config,
        model_info={
            "id": config.source.model_id,
            "sha": config.source.revision,
            "cardData": {"license": config.source.license},
        },
        model_config=_read_json(source_config_path),
        weight_index=weight_index,
    )
    raw_weight_map = weight_index.get("weight_map")
    if not isinstance(raw_weight_map, dict):
        raise RuntimeError("Materialized source weight map is malformed")
    shard_files = {str(filename) for filename in raw_weight_map.values()}
    if not shard_files.issubset(records):
        raise RuntimeError("Materialized source inventory omits an indexed safetensors shard")
    materialized_weight_file_bytes = sum(
        int(records[filename]["size_bytes"])
        for filename in shard_files
        if isinstance(records.get(filename), dict)
    )
    if (
        receipt.get("weight_index_sha256") != audit.weight_index_sha256
        or receipt.get("source_tensor_bytes") != audit.source_bytes
        or receipt.get("source_tensor_count") != audit.source_tensor_count
        or receipt.get("source_shard_count") != audit.source_shard_count
        or receipt.get("materialized_weight_file_bytes") != materialized_weight_file_bytes
        or not EXPECTED_MODEL_BYTES
        <= materialized_weight_file_bytes
        <= EXPECTED_MODEL_BYTES + 1_000_000_000
    ):
        raise RuntimeError("Materialized source receipt differs from the pinned Inkling audit")
    return receipt


@app.function(
    image=python_image,
    cpu=(MATERIALIZE_RESOURCES.cpu_cores, MATERIALIZE_RESOURCES.cpu_cores),
    memory=(
        MATERIALIZE_RESOURCES.memory_gib * 1024,
        MATERIALIZE_RESOURCES.memory_gib * 1024,
    ),
    timeout=int(MATERIALIZE_RESOURCES.max_hours * 3600),
    startup_timeout=MATERIALIZE_RESOURCES.startup_timeout_seconds,
    retries=0,
    max_containers=1,
    single_use_containers=True,
    secrets=[hf_secret],
    volumes={str(SOURCE_MOUNT): source_volume},
)
@_capture_stage_failures("materialize_source")
def materialize_source(
    config_json: str,
    run_id: str,
    budget_acknowledgement_json: str,
    control_plane_json: str,
) -> dict[str, Any]:
    """Download exact-revision files one at a time, committing after every file."""

    from huggingface_hub import HfApi, hf_hub_download

    config, control_plane, acknowledgement = _remote_config(
        config_json,
        run_id,
        budget_acknowledgement_json,
        control_plane_json,
        "materialize_source",
    )
    source_volume.reload()
    run_root = _safe_child(SOURCE_MOUNT, "runs", run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    _bind_config(run_root, config)
    _bind_control_plane(run_root, control_plane)
    _arm_failure_recording(
        run_root,
        config,
        "materialize_source",
        control_plane,
        acknowledgement,
        source_volume,
    )
    success = _safe_child(run_root, "source.success.json")
    if success.exists():
        _begin_attempt(
            run_root,
            config,
            "materialize_source",
            control_plane,
            acknowledgement.launch_intent_sha256,
        )
        source_volume.commit()
        _verify_materialized_source(config, run_id, control_plane)
        return {"status": "already_successful", "run_id": run_id}
    attempt = _begin_attempt(
        run_root,
        config,
        "materialize_source",
        control_plane,
        acknowledgement.launch_intent_sha256,
    )
    source_volume.commit()
    token = os.environ["HF_TOKEN"]
    audit = audit_pinned_inkling_online(config, token=token)
    snapshot = _safe_child(run_root, "snapshot")
    snapshot.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=token)
    repo_files = api.list_repo_files(
        repo_id=config.source.model_id,
        revision=config.source.revision,
        repo_type="model",
    )
    metadata_files = {
        name
        for name in repo_files
        if Path(name).suffix.lower() in SAFE_METADATA_SUFFIXES
        and not Path(name).is_absolute()
        and ".." not in Path(name).parts
        and ".cache" not in Path(name).parts
    }
    required_metadata = {"config.json", "model.safetensors.index.json"}
    if not required_metadata.issubset(metadata_files):
        raise RuntimeError("Pinned repository is missing config or weight index")

    for filename in sorted(required_metadata):
        hf_hub_download(
            repo_id=config.source.model_id,
            filename=filename,
            revision=config.source.revision,
            repo_type="model",
            local_dir=snapshot,
            token=token,
        )
    weight_index = _read_json(snapshot / "model.safetensors.index.json")
    mounted_audit = audit_inkling_source(
        config,
        model_info={
            "id": config.source.model_id,
            "sha": config.source.revision,
            "cardData": {"license": config.source.license},
        },
        model_config=_read_json(snapshot / "config.json"),
        weight_index=weight_index,
    )
    if mounted_audit != audit:
        raise RuntimeError("Mounted config/index audit differs from the online pinned audit")
    raw_weight_map = weight_index.get("weight_map")
    if not isinstance(raw_weight_map, dict):
        raise RuntimeError("Materialized weight index has no weight_map")
    shard_files = {str(value) for value in raw_weight_map.values()}
    if any(
        not name.endswith(".safetensors") or Path(name).is_absolute() or ".." in Path(name).parts
        for name in shard_files
    ):
        raise RuntimeError("Weight index references unsafe or non-safetensors files")
    required_files = sorted(metadata_files | shard_files)

    inventory_path = _safe_child(run_root, "source_inventory.json")
    inventory = _read_json(inventory_path) if inventory_path.exists() else {"files": {}}
    records = inventory.get("files")
    if not isinstance(records, dict):
        raise RuntimeError("Source inventory is malformed")
    deadline = time.monotonic() + 3.5 * 3600
    for filename in required_files:
        destination = _safe_child(snapshot, filename)
        existing = records.get(filename)
        if (
            isinstance(existing, dict)
            and destination.is_file()
            and not destination.is_symlink()
            and destination.stat().st_size == existing.get("size_bytes")
        ):
            continue
        downloaded = Path(
            hf_hub_download(
                repo_id=config.source.model_id,
                filename=filename,
                revision=config.source.revision,
                repo_type="model",
                local_dir=snapshot,
                token=token,
            )
        )
        if downloaded.resolve() != destination.resolve() or downloaded.is_symlink():
            raise RuntimeError(f"Hub download did not materialize a regular local file: {filename}")
        records[filename] = {
            "path": filename,
            "size_bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
        }
        inventory = {
            "config_hash": config.config_hash(),
            "model_id": config.source.model_id,
            "revision": config.source.revision,
            "required_file_count": len(required_files),
            "files": records,
        }
        _atomic_json(inventory_path, inventory)
        source_volume.commit()
        if time.monotonic() >= deadline and len(records) < len(required_files):
            if attempt >= _stage_limit(config, "materialize_source"):
                raise RuntimeError(
                    "Source materialization is incomplete and its configured attempt cap is spent"
                )
            require_stage_billing_window(
                config,
                acknowledgement.billing_cycle_end_utc,
                "materialize_source",
                include_startup=True,
            )
            continuation = materialize_source.spawn(
                config_json,
                run_id,
                budget_acknowledgement_json,
                control_plane_json,
            )
            _atomic_json(
                _safe_child(run_root, "control", "materialize_source.continuation.json"),
                {
                    "from_attempt": attempt,
                    "call_id": continuation.object_id,
                    "completed_files": len(records),
                    "required_files": len(required_files),
                },
            )
            source_volume.commit()
            return {
                "status": "continued",
                "call_id": continuation.object_id,
                "completed_files": len(records),
                "required_files": len(required_files),
            }

    if set(records) != set(required_files):
        raise RuntimeError("Source inventory is incomplete after materialization")
    source_weight_bytes = sum(
        int(records[name]["size_bytes"]) for name in shard_files if isinstance(records[name], dict)
    )
    if not EXPECTED_MODEL_BYTES <= source_weight_bytes <= EXPECTED_MODEL_BYTES + 1_000_000_000:
        raise RuntimeError(f"Unexpected materialized safetensors byte count: {source_weight_bytes}")
    drifted = _repairable_source_drift(snapshot, records)
    if drifted:
        for filename in drifted:
            destination = _safe_child(snapshot, filename)
            if destination.exists():
                if destination.is_symlink() or not destination.is_file():
                    raise RuntimeError(f"Refusing unsafe source repair for {filename}")
                destination.unlink()
            records.pop(filename)
        repaired_inventory = {
            "config_hash": config.config_hash(),
            "model_id": config.source.model_id,
            "revision": config.source.revision,
            "required_file_count": len(required_files),
            "files": records,
        }
        _atomic_json(inventory_path, repaired_inventory)
        source_volume.commit()
        if attempt >= _stage_limit(config, "materialize_source"):
            raise RuntimeError(
                "Source checksum repair is required but the materialization attempt cap is spent"
            )
        require_stage_billing_window(
            config,
            acknowledgement.billing_cycle_end_utc,
            "materialize_source",
            include_startup=True,
        )
        continuation = materialize_source.spawn(
            config_json,
            run_id,
            budget_acknowledgement_json,
            control_plane_json,
        )
        _atomic_json(
            _safe_child(run_root, "control", "materialize_source.repair.json"),
            {
                "from_attempt": attempt,
                "call_id": continuation.object_id,
                "drifted_files": drifted,
            },
        )
        source_volume.commit()
        return {
            "status": "repair_continued",
            "call_id": continuation.object_id,
            "drifted_files": drifted,
        }
    local_cache = _safe_child(snapshot, ".cache")
    if local_cache.exists() or local_cache.is_symlink():
        if local_cache.is_symlink() or not local_cache.is_dir():
            raise RuntimeError("Hugging Face local metadata cache is unsafe")
        shutil.rmtree(local_cache)
    _verify_inventory_membership(snapshot, list(records.values()))
    receipt = {
        "verified": True,
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "model_id": mounted_audit.model_id,
        "revision": mounted_audit.revision,
        "license": mounted_audit.license,
        "source_dir": str(snapshot),
        "weight_index_sha256": mounted_audit.weight_index_sha256,
        "source_tensor_bytes": EXPECTED_MODEL_BYTES,
        "materialized_weight_file_bytes": source_weight_bytes,
        "source_tensor_count": mounted_audit.source_tensor_count,
        "source_shard_count": mounted_audit.source_shard_count,
        "file_count": len(records),
        "inventory_sha256": _sha256(inventory_path),
        "source_config_sha256": _sha256(snapshot / "config.json"),
        "warnings": list(mounted_audit.warnings),
        "call_id": _current_invocation_ids()[0],
        "launch_intent_sha256": acknowledgement.launch_intent_sha256,
    }
    _immutable_json(_safe_child(run_root, "source.success.json"), receipt)
    source_volume.commit()
    return {"status": "success", "run_id": run_id, **receipt}


def _execution_paths(run_id: str, *, partial_work: Path, partial_final: Path) -> WorkflowPaths:
    return WorkflowPaths(
        source_dir=_safe_child(SOURCE_MOUNT, "runs", run_id, "snapshot"),
        work_dir=partial_work,
        final_dir=partial_final,
        llama_cpp_dir=LLAMA_CPP_DIR,
    )


def _verify_toolchain(config: InklingGGUFConfig) -> dict[str, str]:
    commit = subprocess.run(
        ["git", "-C", str(LLAMA_CPP_DIR), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    ).stdout.strip()
    build_commit = (LLAMA_CPP_DIR / ".iql-build-commit").read_text(encoding="utf-8").strip()
    if commit != build_commit:
        raise RuntimeError("llama.cpp checkout differs from its image build receipt")
    clean = subprocess.run(
        ["git", "-C", str(LLAMA_CPP_DIR), "diff", "--quiet", "HEAD", "--"],
        check=False,
        shell=False,
    )
    if clean.returncode != 0:
        raise RuntimeError("llama.cpp tracked source differs from the pinned checkout")
    remote_url = subprocess.run(
        ["git", "-C", str(LLAMA_CPP_DIR), "remote", "get-url", "origin"],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    ).stdout.strip()
    if remote_url != config.toolchain.repository:
        raise RuntimeError("llama.cpp origin differs from the configured Inkling fork")
    binary_hashes: dict[str, str] = {}
    binary_manifests = {
        "quantize_sha256": (".iql-quantize.sha256", LLAMA_CPP_DIR / "build/bin/llama-quantize"),
        "gguf_split_sha256": (
            ".iql-gguf-split.sha256",
            LLAMA_CPP_DIR / "build/bin/llama-gguf-split",
        ),
    }
    for key, (manifest_name, expected_binary) in binary_manifests.items():
        manifest = LLAMA_CPP_DIR / manifest_name
        expected, raw_path = manifest.read_text(encoding="utf-8").split(maxsplit=1)
        binary = Path(raw_path.strip())
        if binary.resolve() != expected_binary.resolve():
            raise RuntimeError(f"Built tool receipt has an unexpected path in {manifest_name}")
        if not binary.is_file() or _sha256(binary) != expected:
            raise RuntimeError(f"Built tool does not match {manifest_name}")
        binary_hashes[key] = expected
    freeze_path = LLAMA_CPP_DIR / ".iql-python-freeze.txt"
    freeze_manifest = LLAMA_CPP_DIR / ".iql-python-freeze.sha256"
    freeze_expected, freeze_raw_path = freeze_manifest.read_text(encoding="utf-8").split(maxsplit=1)
    if Path(freeze_raw_path.strip()).resolve() != freeze_path.resolve():
        raise RuntimeError("Python dependency receipt has an unexpected path")
    if _sha256(freeze_path) != freeze_expected:
        raise RuntimeError("Python dependency inventory differs from its image build receipt")
    live_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    ).stdout.splitlines()
    canonical_live_freeze = "\n".join(sorted(line for line in live_freeze if line)) + "\n"
    if freeze_path.read_text(encoding="utf-8") != canonical_live_freeze:
        raise RuntimeError("Runtime Python distributions differ from the image build inventory")
    dpkg_path = LLAMA_CPP_DIR / ".iql-dpkg-inventory.txt"
    dpkg_manifest = LLAMA_CPP_DIR / ".iql-dpkg-inventory.sha256"
    dpkg_expected, dpkg_raw_path = dpkg_manifest.read_text(encoding="utf-8").split(maxsplit=1)
    if Path(dpkg_raw_path.strip()).resolve() != dpkg_path.resolve():
        raise RuntimeError("Debian dependency receipt has an unexpected path")
    if _sha256(dpkg_path) != dpkg_expected:
        raise RuntimeError("Debian package inventory differs from its image build receipt")
    live_dpkg = subprocess.run(
        ["dpkg-query", "-W", "-f=${Package}=${Version}\n"],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    ).stdout.splitlines()
    canonical_live_dpkg = "\n".join(sorted(line for line in live_dpkg if line)) + "\n"
    if dpkg_path.read_text(encoding="utf-8") != canonical_live_dpkg:
        raise RuntimeError("Runtime Debian packages differ from the image build inventory")
    if commit != config.toolchain.commit:
        raise RuntimeError("Built toolchain is not the configured Inkling commit")
    return {
        "base_image": DEBIAN_IMAGE,
        "repository": config.toolchain.repository,
        "commit": commit,
        "dpkg_inventory_sha256": dpkg_expected,
        "python_freeze_sha256": freeze_expected,
        **binary_hashes,
    }


def _published_toolchain_inventory_records(final_root: Path) -> list[dict[str, Any]]:
    return [
        _file_record(
            _safe_child(final_root, "verification", "toolchain", filename),
            final_root,
        )
        for filename in ("python-freeze.txt", "debian-packages.txt")
    ]


def _verify_source_binding(
    config: InklingGGUFConfig,
    run_id: str,
    paths: WorkflowPaths,
    control_plane: ControlPlaneProvenance,
) -> dict[str, str]:
    receipt_path = _safe_child(SOURCE_MOUNT, "runs", run_id, "source.success.json")
    if not receipt_path.is_file():
        raise RuntimeError("Exact source materialization has not completed")
    receipt = _verify_materialized_source(config, run_id, control_plane)
    toolchain = _verify_toolchain(config)
    verify_execution_bindings(
        config,
        paths,
        source_receipt=receipt,
        actual_llama_cpp_commit=toolchain["commit"],
    )
    return toolchain


def _verify_bf16_dependency(
    config: InklingGGUFConfig,
    run_id: str,
    *,
    control_plane: ControlPlaneProvenance,
    toolchain: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    source_receipt_path = _safe_child(SOURCE_MOUNT, "runs", run_id, "source.success.json")
    work_root = _safe_child(WORK_MOUNT, "runs", run_id)
    receipt_path = _safe_child(work_root, "convert_text_bf16.success.json")
    receipt = _read_json(receipt_path)
    _verify_receipt_control_plane(receipt, config, control_plane)
    outputs = receipt.get("outputs")
    if (
        receipt.get("status") != "success"
        or not isinstance(outputs, list)
        or len(outputs) < 2
        or receipt.get("llama_cpp_commit") != config.toolchain.commit
        or receipt.get("toolchain") != toolchain
        or receipt.get("source_receipt_sha256") != _sha256(source_receipt_path)
    ):
        raise RuntimeError("BF16 receipt is not bound to this source/toolchain/config")
    _verify_publication_evidence(
        work_root,
        config=config,
        stage="convert_text_bf16",
        receipt=receipt,
    )
    return receipt, _sha256(receipt_path)


def _verify_mmproj_dependency(
    config: InklingGGUFConfig,
    run_id: str,
    *,
    control_plane: ControlPlaneProvenance,
    toolchain: Mapping[str, str],
    source_receipt_sha256: str,
    bf16_receipt_sha256: str,
) -> tuple[dict[str, Any], str]:
    final_root = _safe_child(FINAL_MOUNT, "runs", run_id)
    receipt_path = _safe_child(final_root, "convert_multimodal_projector.success.json")
    receipt = _read_json(receipt_path)
    _verify_receipt_control_plane(receipt, config, control_plane)
    outputs = receipt.get("outputs")
    if (
        receipt.get("status") != "success"
        or not isinstance(outputs, list)
        or len(outputs) != 1
        or receipt.get("llama_cpp_commit") != config.toolchain.commit
        or receipt.get("toolchain") != toolchain
        or receipt.get("source_receipt_sha256") != source_receipt_sha256
        or receipt.get("bf16_receipt_sha256") != bf16_receipt_sha256
        or receipt.get("coverage") != {"vision_tensors": 8, "audio_tensors": 2}
    ):
        raise RuntimeError("mmproj receipt is not bound to this source/BF16/toolchain")
    _verify_publication_evidence(
        final_root,
        config=config,
        stage="convert_multimodal_projector",
        receipt=receipt,
    )
    return receipt, _sha256(receipt_path)


def _verify_final_dependency_chain(
    config: InklingGGUFConfig,
    run_id: str,
    *,
    control_plane: ControlPlaneProvenance,
    actual_toolchain: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], list[Any], list[Any], str, str]:
    """Rehash the actual source, receipt chain, BF16 input, and every final output."""

    source_root = _safe_child(SOURCE_MOUNT, "runs", run_id)
    work_root = _safe_child(WORK_MOUNT, "runs", run_id)
    final_root = _safe_child(FINAL_MOUNT, "runs", run_id)
    for root in (source_root, work_root, final_root):
        _verify_bound_control_plane(root, control_plane)

    source_receipt_path = _safe_child(source_root, "source.success.json")
    source_receipt = _verify_materialized_source(config, run_id, control_plane)
    source_receipt_sha = _sha256(source_receipt_path)
    source_dir = _safe_child(source_root, "snapshot")
    verify_execution_bindings(
        config,
        WorkflowPaths(
            source_dir=source_dir,
            work_dir=work_root,
            final_dir=final_root,
            llama_cpp_dir=LLAMA_CPP_DIR,
        ),
        source_receipt=source_receipt,
        actual_llama_cpp_commit=actual_toolchain["commit"],
    )

    bf16_receipt_path = _safe_child(work_root, "convert_text_bf16.success.json")
    bf16_receipt = _read_json(bf16_receipt_path)
    _verify_receipt_control_plane(bf16_receipt, config, control_plane)
    bf16_outputs = bf16_receipt.get("outputs")
    if (
        not isinstance(bf16_outputs, list)
        or len(bf16_outputs) < 2
        or bf16_receipt.get("llama_cpp_commit") != config.toolchain.commit
        or bf16_receipt.get("toolchain") != actual_toolchain
        or bf16_receipt.get("source_receipt_sha256") != source_receipt_sha
    ):
        raise RuntimeError("Actual BF16 receipt is not bound to the source/toolchain")
    _verify_publication_evidence(
        work_root,
        config=config,
        stage="convert_text_bf16",
        receipt=bf16_receipt,
    )
    bf16_receipt_sha = _sha256(bf16_receipt_path)

    quant_receipt_path = _safe_child(final_root, "quantize_text.success.json")
    mmproj_receipt_path = _safe_child(
        final_root,
        "convert_multimodal_projector.success.json",
    )
    quant_receipt = _read_json(quant_receipt_path)
    mmproj_receipt = _read_json(mmproj_receipt_path)
    mmproj_receipt_sha = _sha256(mmproj_receipt_path)
    _verify_receipt_control_plane(quant_receipt, config, control_plane)
    _verify_receipt_control_plane(mmproj_receipt, config, control_plane)
    quant_outputs = quant_receipt.get("outputs")
    mmproj_outputs = mmproj_receipt.get("outputs")
    if (
        not isinstance(quant_outputs, list)
        or not isinstance(mmproj_outputs, list)
        or len(quant_outputs) < 2
        or len(mmproj_outputs) != 1
        or quant_receipt.get("status") != "success"
        or quant_receipt.get("model_id") != config.source.model_id
        or quant_receipt.get("revision") != config.source.revision
        or quant_receipt.get("license") != config.source.license
        or quant_receipt.get("llama_cpp_commit") != config.toolchain.commit
        or quant_receipt.get("toolchain") != actual_toolchain
        or quant_receipt.get("quant_type") != config.quantization.quant_type
        or quant_receipt.get("output_label") != config.quantization.output_label
        or quant_receipt.get("importance_matrix") is not None
        or quant_receipt.get("mtp") != config.coverage.mtp
        or quant_receipt.get("source_receipt_sha256") != source_receipt_sha
        or quant_receipt.get("bf16_receipt_sha256") != bf16_receipt_sha
        or quant_receipt.get("mmproj_receipt_sha256") != mmproj_receipt_sha
        or mmproj_receipt.get("status") != "success"
        or mmproj_receipt.get("llama_cpp_commit") != config.toolchain.commit
        or mmproj_receipt.get("toolchain") != actual_toolchain
        or mmproj_receipt.get("source_receipt_sha256") != source_receipt_sha
        or mmproj_receipt.get("bf16_receipt_sha256") != bf16_receipt_sha
        or mmproj_receipt.get("coverage") != {"vision_tensors": 8, "audio_tensors": 2}
    ):
        raise RuntimeError("Actual final receipts do not form one exact Inkling execution chain")
    _verify_publication_evidence(
        final_root,
        config=config,
        stage="convert_multimodal_projector",
        receipt=mmproj_receipt,
    )
    _verify_publication_evidence(
        final_root,
        config=config,
        stage="quantize_text",
        receipt=quant_receipt,
    )
    return (
        quant_receipt,
        mmproj_receipt,
        quant_outputs,
        mmproj_outputs,
        source_receipt_sha,
        bf16_receipt_sha,
    )


@app.function(
    image=toolchain_image,
    cpu=(TEXT_CONVERT_RESOURCES.cpu_cores, TEXT_CONVERT_RESOURCES.cpu_cores),
    memory=(
        TEXT_CONVERT_RESOURCES.memory_gib * 1024,
        TEXT_CONVERT_RESOURCES.memory_gib * 1024,
    ),
    timeout=int(TEXT_CONVERT_RESOURCES.max_hours * 3600),
    startup_timeout=TEXT_CONVERT_RESOURCES.startup_timeout_seconds,
    retries=0,
    max_containers=1,
    single_use_containers=True,
    block_network=True,
    volumes={
        str(SOURCE_MOUNT): source_volume,
        str(WORK_MOUNT): work_volume,
        str(FINAL_MOUNT): final_volume,
    },
)
@_capture_stage_failures("convert_text_bf16")
def convert_text_bf16(
    config_json: str,
    run_id: str,
    budget_acknowledgement_json: str,
    control_plane_json: str,
) -> dict[str, Any]:
    """Convert the exact local snapshot to split BF16 GGUF and hash every split."""

    config, control_plane, acknowledgement = _remote_config(
        config_json,
        run_id,
        budget_acknowledgement_json,
        control_plane_json,
        "convert_text_bf16",
    )
    source_volume.reload()
    work_volume.reload()
    work_root = _safe_child(WORK_MOUNT, "runs", run_id)
    work_root.mkdir(parents=True, exist_ok=True)
    _bind_config(work_root, config)
    _bind_control_plane(work_root, control_plane)
    _arm_failure_recording(
        work_root,
        config,
        "convert_text_bf16",
        control_plane,
        acknowledgement,
        work_volume,
    )
    success = _safe_child(work_root, "convert_text_bf16.success.json")
    if success.exists():
        _begin_recovery(
            work_root,
            config,
            "convert_text_bf16",
            control_plane,
            acknowledgement.launch_intent_sha256,
        )
        work_volume.commit()
        paths = _execution_paths(
            run_id,
            partial_work=_safe_child(work_root, "bf16"),
            partial_final=FINAL_MOUNT / "unused",
        )
        toolchain = _verify_source_binding(config, run_id, paths, control_plane)
        _verify_bf16_dependency(
            config,
            run_id,
            control_plane=control_plane,
            toolchain=toolchain,
        )
        return {"status": "already_successful", "run_id": run_id}

    source_receipt_path = _safe_child(SOURCE_MOUNT, "runs", run_id, "source.success.json")
    _cheap_success_receipt(
        source_receipt_path,
        config=config,
        control_plane=control_plane,
    )
    canonical = _safe_child(work_root, "bf16")
    pending = _pending_publication(
        work_root,
        config=config,
        stage="convert_text_bf16",
        canonical_name="bf16",
    )
    if pending is not None:
        _begin_recovery(
            work_root,
            config,
            "convert_text_bf16",
            control_plane,
            acknowledgement.launch_intent_sha256,
        )
        work_volume.commit()
        partial, _ = pending
        paths = _execution_paths(
            run_id,
            partial_work=partial,
            partial_final=FINAL_MOUNT / "unused",
        )
        toolchain = _verify_source_binding(config, run_id, paths, control_plane)
        publication = _recover_and_commit_publication(
            work_root,
            config=config,
            stage="convert_text_bf16",
            canonical_name="bf16",
            volume=work_volume,
        )
        assert publication is not None
    else:
        attempt = _begin_attempt(
            work_root,
            config,
            "convert_text_bf16",
            control_plane,
            acknowledgement.launch_intent_sha256,
        )
        work_volume.commit()
        _clean_old_partials(work_root, ".partial-convert-text-")
        partial = _safe_child(work_root, f".partial-convert-text-{attempt}")
        partial.mkdir()
        paths = _execution_paths(
            run_id,
            partial_work=partial,
            partial_final=FINAL_MOUNT / "unused",
        )
        toolchain = _verify_source_binding(config, run_id, paths, control_plane)
        command = build_conversion_plan(config, paths).text_conversion
        _run_streaming(
            command.argv,
            log_path=_safe_child(work_root, "logs", f"convert_text_bf16.{attempt}.log"),
        )
        splits = sorted(partial.glob("inkling-BF16*.gguf"))
        if len(splits) < 2 or not splits[0].name.startswith("inkling-BF16-00001-of-"):
            raise RuntimeError("BF16 converter did not produce a valid split set")
        total_bytes = sum(path.stat().st_size for path in splits)
        if not 1_700_000_000_000 <= total_bytes <= 2_000_000_000_000:
            raise RuntimeError(f"Unexpected Inkling BF16 GGUF size: {total_bytes}")
        publication = _prepare_and_commit_publication(
            work_root,
            config=config,
            stage="convert_text_bf16",
            partial=partial,
            canonical=canonical,
            outputs=[_file_record(path, partial) for path in splits],
            volume=work_volume,
        )
    publication_evidence = _publication_evidence(
        work_root,
        "convert_text_bf16",
        publication,
    )
    total_bytes = sum(record.size_bytes for record in publication.outputs)
    receipt = {
        "status": "success",
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "call_id": _current_invocation_ids()[0],
        "launch_intent_sha256": acknowledgement.launch_intent_sha256,
        "llama_cpp_commit": config.toolchain.commit,
        "toolchain": toolchain,
        "quantization_input": "official Inkling BF16",
        "source_receipt_sha256": _sha256(source_receipt_path),
        "split_count": len(publication.outputs),
        "total_bytes": total_bytes,
        **publication_evidence,
    }
    _immutable_json(success, receipt)
    work_volume.commit()
    return receipt


@app.function(
    image=toolchain_image,
    cpu=(MMPROJ_RESOURCES.cpu_cores, MMPROJ_RESOURCES.cpu_cores),
    memory=(MMPROJ_RESOURCES.memory_gib * 1024, MMPROJ_RESOURCES.memory_gib * 1024),
    timeout=int(MMPROJ_RESOURCES.max_hours * 3600),
    startup_timeout=MMPROJ_RESOURCES.startup_timeout_seconds,
    retries=0,
    max_containers=1,
    single_use_containers=True,
    block_network=True,
    volumes={
        str(SOURCE_MOUNT): source_volume,
        str(WORK_MOUNT): work_volume,
        str(FINAL_MOUNT): final_volume,
    },
)
@_capture_stage_failures("convert_multimodal_projector")
def convert_multimodal_projector(
    config_json: str,
    run_id: str,
    budget_acknowledgement_json: str,
    control_plane_json: str,
) -> dict[str, Any]:
    """Export all eight vision and two audio tensors as a separate BF16 mmproj."""

    config, control_plane, acknowledgement = _remote_config(
        config_json,
        run_id,
        budget_acknowledgement_json,
        control_plane_json,
        "convert_multimodal_projector",
    )
    source_volume.reload()
    work_volume.reload()
    final_volume.reload()
    work_root = _safe_child(WORK_MOUNT, "runs", run_id)
    final_root = _safe_child(FINAL_MOUNT, "runs", run_id)
    final_root.mkdir(parents=True, exist_ok=True)
    _bind_config(final_root, config)
    _bind_control_plane(final_root, control_plane)
    _arm_failure_recording(
        final_root,
        config,
        "convert_multimodal_projector",
        control_plane,
        acknowledgement,
        final_volume,
    )
    success = _safe_child(final_root, "convert_multimodal_projector.success.json")
    if success.exists():
        _begin_recovery(
            final_root,
            config,
            "convert_multimodal_projector",
            control_plane,
            acknowledgement.launch_intent_sha256,
        )
        final_volume.commit()
        paths = _execution_paths(
            run_id,
            partial_work=_safe_child(work_root, "bf16"),
            partial_final=_safe_child(final_root, "mmproj"),
        )
        toolchain = _verify_source_binding(config, run_id, paths, control_plane)
        bf16_receipt, bf16_receipt_sha = _verify_bf16_dependency(
            config,
            run_id,
            control_plane=control_plane,
            toolchain=toolchain,
        )
        _verify_mmproj_dependency(
            config=config,
            run_id=run_id,
            control_plane=control_plane,
            toolchain=toolchain,
            source_receipt_sha256=str(bf16_receipt["source_receipt_sha256"]),
            bf16_receipt_sha256=bf16_receipt_sha,
        )
        return {"status": "already_successful", "run_id": run_id}

    source_receipt_path = _safe_child(SOURCE_MOUNT, "runs", run_id, "source.success.json")
    bf16_receipt_path = _safe_child(
        _safe_child(WORK_MOUNT, "runs", run_id),
        "convert_text_bf16.success.json",
    )
    _cheap_success_receipt(
        source_receipt_path,
        config=config,
        control_plane=control_plane,
    )
    bf16_predecessor = _cheap_success_receipt(
        bf16_receipt_path,
        config=config,
        control_plane=control_plane,
    )
    _verify_publication_evidence(
        work_root,
        config=config,
        stage="convert_text_bf16",
        receipt=bf16_predecessor,
        verify_outputs=False,
    )
    canonical = _safe_child(final_root, "mmproj")
    pending = _pending_publication(
        final_root,
        config=config,
        stage="convert_multimodal_projector",
        canonical_name="mmproj",
    )
    if pending is not None:
        _begin_recovery(
            final_root,
            config,
            "convert_multimodal_projector",
            control_plane,
            acknowledgement.launch_intent_sha256,
        )
        final_volume.commit()
        partial, _ = pending
        paths = _execution_paths(
            run_id,
            partial_work=WORK_MOUNT / "unused",
            partial_final=partial,
        )
        toolchain = _verify_source_binding(config, run_id, paths, control_plane)
        _, bf16_receipt_sha = _verify_bf16_dependency(
            config,
            run_id,
            control_plane=control_plane,
            toolchain=toolchain,
        )
        publication = _recover_and_commit_publication(
            final_root,
            config=config,
            stage="convert_multimodal_projector",
            canonical_name="mmproj",
            volume=final_volume,
        )
        assert publication is not None
    else:
        attempt = _begin_attempt(
            final_root,
            config,
            "convert_multimodal_projector",
            control_plane,
            acknowledgement.launch_intent_sha256,
        )
        final_volume.commit()
        _clean_old_partials(final_root, ".partial-mmproj-")
        partial = _safe_child(final_root, f".partial-mmproj-{attempt}")
        partial.mkdir()
        paths = _execution_paths(
            run_id,
            partial_work=WORK_MOUNT / "unused",
            partial_final=partial,
        )
        toolchain = _verify_source_binding(config, run_id, paths, control_plane)
        _, bf16_receipt_sha = _verify_bf16_dependency(
            config,
            run_id,
            control_plane=control_plane,
            toolchain=toolchain,
        )
        command = build_conversion_plan(config, paths).mmproj_conversion
        _run_streaming(
            command.argv,
            log_path=_safe_child(final_root, "logs", f"convert_mmproj.{attempt}.log"),
        )
        output = partial / "mmproj-BF16.gguf"
        if not output.is_file() or not 100_000_000 <= output.stat().st_size <= 500_000_000:
            raise RuntimeError("Unexpected or missing Inkling multimodal projector")
        publication = _prepare_and_commit_publication(
            final_root,
            config=config,
            stage="convert_multimodal_projector",
            partial=partial,
            canonical=canonical,
            outputs=[_file_record(output, partial)],
            volume=final_volume,
        )
    receipt = {
        "status": "success",
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "call_id": _current_invocation_ids()[0],
        "launch_intent_sha256": acknowledgement.launch_intent_sha256,
        "llama_cpp_commit": config.toolchain.commit,
        "toolchain": toolchain,
        "coverage": {"vision_tensors": 8, "audio_tensors": 2},
        "source_receipt_sha256": _sha256(source_receipt_path),
        "bf16_receipt_sha256": bf16_receipt_sha,
        **_publication_evidence(
            final_root,
            "convert_multimodal_projector",
            publication,
        ),
    }
    _immutable_json(success, receipt)
    final_volume.commit()
    return receipt


@app.function(
    image=toolchain_image,
    cpu=(QUANTIZE_RESOURCES.cpu_cores, QUANTIZE_RESOURCES.cpu_cores),
    memory=(
        QUANTIZE_RESOURCES.memory_gib * 1024,
        QUANTIZE_RESOURCES.memory_gib * 1024,
    ),
    timeout=int(QUANTIZE_RESOURCES.max_hours * 3600),
    startup_timeout=QUANTIZE_RESOURCES.startup_timeout_seconds,
    retries=0,
    max_containers=1,
    single_use_containers=True,
    block_network=True,
    volumes={
        str(SOURCE_MOUNT): source_volume,
        str(WORK_MOUNT): work_volume,
        str(FINAL_MOUNT): final_volume,
    },
)
@_capture_stage_failures("quantize_text")
def quantize_text(
    config_json: str,
    run_id: str,
    budget_acknowledgement_json: str,
    control_plane_json: str,
) -> dict[str, Any]:
    """Quantize our BF16 conversion to honestly labelled stock Q3_K_M."""

    config, control_plane, acknowledgement = _remote_config(
        config_json,
        run_id,
        budget_acknowledgement_json,
        control_plane_json,
        "quantize_text",
    )
    source_volume.reload()
    work_volume.reload()
    final_volume.reload()
    work_root = _safe_child(WORK_MOUNT, "runs", run_id)
    final_root = _safe_child(FINAL_MOUNT, "runs", run_id)
    final_root.mkdir(parents=True, exist_ok=True)
    _bind_config(final_root, config)
    _bind_control_plane(final_root, control_plane)
    _arm_failure_recording(
        final_root,
        config,
        "quantize_text",
        control_plane,
        acknowledgement,
        final_volume,
    )
    success = _safe_child(final_root, "quantize_text.success.json")
    if success.exists():
        _begin_recovery(
            final_root,
            config,
            "quantize_text",
            control_plane,
            acknowledgement.launch_intent_sha256,
        )
        final_volume.commit()
        toolchain = _verify_toolchain(config)
        _verify_final_dependency_chain(
            config,
            run_id,
            control_plane=control_plane,
            actual_toolchain=toolchain,
        )
        return {"status": "already_successful", "run_id": run_id}

    bf16_receipt_path = _safe_child(work_root, "convert_text_bf16.success.json")
    mmproj_receipt_path = _safe_child(
        final_root,
        "convert_multimodal_projector.success.json",
    )
    bf16_predecessor = _cheap_success_receipt(
        bf16_receipt_path,
        config=config,
        control_plane=control_plane,
    )
    mmproj_predecessor = _cheap_success_receipt(
        mmproj_receipt_path,
        config=config,
        control_plane=control_plane,
    )
    _verify_publication_evidence(
        work_root,
        config=config,
        stage="convert_text_bf16",
        receipt=bf16_predecessor,
        verify_outputs=False,
    )
    _verify_publication_evidence(
        final_root,
        config=config,
        stage="convert_multimodal_projector",
        receipt=mmproj_predecessor,
        verify_outputs=False,
    )
    canonical = _safe_child(final_root, "q3_k_m")
    pending = _pending_publication(
        final_root,
        config=config,
        stage="quantize_text",
        canonical_name="q3_k_m",
    )
    if pending is not None:
        _begin_recovery(
            final_root,
            config,
            "quantize_text",
            control_plane,
            acknowledgement.launch_intent_sha256,
        )
        final_volume.commit()
        partial, _ = pending
        toolchain = _verify_toolchain(config)
        bf16_receipt, bf16_receipt_sha = _verify_bf16_dependency(
            config,
            run_id,
            control_plane=control_plane,
            toolchain=toolchain,
        )
        _, mmproj_receipt_sha = _verify_mmproj_dependency(
            config,
            run_id,
            control_plane=control_plane,
            toolchain=toolchain,
            source_receipt_sha256=str(bf16_receipt["source_receipt_sha256"]),
            bf16_receipt_sha256=bf16_receipt_sha,
        )
        publication = _recover_and_commit_publication(
            final_root,
            config=config,
            stage="quantize_text",
            canonical_name="q3_k_m",
            volume=final_volume,
        )
        assert publication is not None
    else:
        attempt = _begin_attempt(
            final_root,
            config,
            "quantize_text",
            control_plane,
            acknowledgement.launch_intent_sha256,
        )
        final_volume.commit()
        _clean_old_partials(final_root, ".partial-q3km-")
        partial = _safe_child(final_root, f".partial-q3km-{attempt}")
        partial.mkdir()
        paths = _execution_paths(
            run_id,
            partial_work=_safe_child(work_root, "bf16"),
            partial_final=partial,
        )
        toolchain = _verify_toolchain(config)
        bf16_receipt, bf16_receipt_sha = _verify_bf16_dependency(
            config,
            run_id,
            control_plane=control_plane,
            toolchain=toolchain,
        )
        _, mmproj_receipt_sha = _verify_mmproj_dependency(
            config,
            run_id,
            control_plane=control_plane,
            toolchain=toolchain,
            source_receipt_sha256=str(bf16_receipt["source_receipt_sha256"]),
            bf16_receipt_sha256=bf16_receipt_sha,
        )
        first_candidates = sorted((work_root / "bf16").glob("inkling-BF16-00001-of-*.gguf"))
        if len(first_candidates) != 1:
            raise RuntimeError("Could not identify exactly one first BF16 split")
        command = build_quantize_command(
            config,
            paths,
            first_bf16_split=first_candidates[0],
        )
        _run_streaming(
            command.argv,
            log_path=_safe_child(final_root, "logs", f"quantize_q3_k_m.{attempt}.log"),
        )
        splits = sorted(partial.glob("inkling-Q3_K_M*.gguf"))
        if len(splits) < 2 or not splits[0].name.startswith("inkling-Q3_K_M-00001-of-"):
            raise RuntimeError("Quantizer did not produce a valid split Q3_K_M set")
        total_bytes = sum(path.stat().st_size for path in splits)
        if not 350_000_000_000 <= total_bytes <= 550_000_000_000:
            raise RuntimeError(f"Unexpected Inkling Q3_K_M size: {total_bytes}")
        publication = _prepare_and_commit_publication(
            final_root,
            config=config,
            stage="quantize_text",
            partial=partial,
            canonical=canonical,
            outputs=[_file_record(path, partial) for path in splits],
            volume=final_volume,
        )
    total_bytes = sum(record.size_bytes for record in publication.outputs)
    receipt = {
        "status": "success",
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "call_id": _current_invocation_ids()[0],
        "launch_intent_sha256": acknowledgement.launch_intent_sha256,
        "model_id": config.source.model_id,
        "revision": config.source.revision,
        "license": config.source.license,
        "source": "our exact official Inkling BF16 conversion",
        "llama_cpp_commit": config.toolchain.commit,
        "toolchain": toolchain,
        "quant_type": "Q3_K_M",
        "importance_matrix": None,
        "output_label": config.quantization.output_label,
        "split_count": len(publication.outputs),
        "total_bytes": total_bytes,
        "mtp": "omitted_unsupported",
        "source_receipt_sha256": bf16_receipt["source_receipt_sha256"],
        "bf16_receipt_sha256": bf16_receipt_sha,
        "mmproj_receipt_sha256": mmproj_receipt_sha,
        **_publication_evidence(final_root, "quantize_text", publication),
    }
    _immutable_json(success, receipt)
    final_volume.commit()
    return receipt


@app.function(
    image=toolchain_image,
    cpu=(VERIFY_RESOURCES.cpu_cores, VERIFY_RESOURCES.cpu_cores),
    memory=(VERIFY_RESOURCES.memory_gib * 1024, VERIFY_RESOURCES.memory_gib * 1024),
    timeout=int(VERIFY_RESOURCES.max_hours * 3600),
    startup_timeout=VERIFY_RESOURCES.startup_timeout_seconds,
    retries=0,
    max_containers=1,
    single_use_containers=True,
    block_network=True,
    volumes={
        str(SOURCE_MOUNT): source_volume,
        str(WORK_MOUNT): work_volume,
        str(FINAL_MOUNT): final_volume,
    },
)
@_capture_stage_failures("verify_export")
def verify_export(
    config_json: str,
    run_id: str,
    budget_acknowledgement_json: str,
    control_plane_json: str,
) -> dict[str, Any]:
    """Rehash source/final artifacts and parse them with the pinned llama.cpp tools."""

    config, control_plane, acknowledgement = _remote_config(
        config_json,
        run_id,
        budget_acknowledgement_json,
        control_plane_json,
        "verify_export",
    )
    source_volume.reload()
    work_volume.reload()
    final_volume.reload()
    final_root = _safe_child(FINAL_MOUNT, "runs", run_id)
    _bind_config(final_root, config)
    _bind_control_plane(final_root, control_plane)
    _arm_failure_recording(
        final_root,
        config,
        "verify_export",
        control_plane,
        acknowledgement,
        final_volume,
    )
    success_path = _safe_child(final_root, "verify_export.success.json")
    quant_receipt_path = _safe_child(final_root, "quantize_text.success.json")
    mmproj_receipt_path = _safe_child(
        final_root,
        "convert_multimodal_projector.success.json",
    )

    if not success_path.exists():
        quant_predecessor = _cheap_success_receipt(
            quant_receipt_path,
            config=config,
            control_plane=control_plane,
        )
        mmproj_predecessor = _cheap_success_receipt(
            mmproj_receipt_path,
            config=config,
            control_plane=control_plane,
        )
        _verify_publication_evidence(
            final_root,
            config=config,
            stage="quantize_text",
            receipt=quant_predecessor,
            verify_outputs=False,
        )
        _verify_publication_evidence(
            final_root,
            config=config,
            stage="convert_multimodal_projector",
            receipt=mmproj_predecessor,
            verify_outputs=False,
        )

    pending = _pending_publication(
        final_root,
        config=config,
        stage="verify_export",
        canonical_name="verification",
    )
    if not success_path.exists() and pending is None:
        attempt = _begin_attempt(
            final_root,
            config,
            "verify_export",
            control_plane,
            acknowledgement.launch_intent_sha256,
        )
        final_volume.commit()
    elif not success_path.exists():
        _begin_recovery(
            final_root,
            config,
            "verify_export",
            control_plane,
            acknowledgement.launch_intent_sha256,
        )
        final_volume.commit()
        attempt = None
    else:
        _begin_recovery(
            final_root,
            config,
            "verify_export",
            control_plane,
            acknowledgement.launch_intent_sha256,
        )
        final_volume.commit()
        attempt = None

    toolchain = _verify_toolchain(config)
    (
        _quant_receipt,
        _mmproj_receipt,
        quant_outputs,
        mmproj_outputs,
        source_receipt_sha,
        bf16_receipt_sha,
    ) = _verify_final_dependency_chain(
        config,
        run_id,
        control_plane=control_plane,
        actual_toolchain=toolchain,
    )

    if success_path.exists():
        success_receipt = _read_json(success_path)
        success_outputs = success_receipt.get("outputs")
        if (
            success_receipt.get("status") != "success"
            or success_receipt.get("config_hash") != config.config_hash()
            or not isinstance(success_outputs, list)
        ):
            raise RuntimeError("Export verification success receipt is malformed")
        _verify_receipt_control_plane(success_receipt, config, control_plane)
        _verify_publication_evidence(
            final_root,
            config=config,
            stage="verify_export",
            receipt=success_receipt,
        )
        manifest_path = _safe_child(final_root, "verification", "export_manifest.json")
        if success_receipt.get("manifest_sha256") != _sha256(manifest_path):
            raise RuntimeError("Export success marker no longer matches its manifest")
        manifest = _read_json(manifest_path)
        if (
            manifest.get("status") != "verified"
            or manifest.get("config_hash") != config.config_hash()
            or manifest.get("control_plane_sha256") != control_plane.tree_sha256
            or manifest.get("toolchain") != toolchain
            or manifest.get("quantized_outputs") != quant_outputs
            or manifest.get("mmproj_outputs") != mmproj_outputs
            or manifest.get("source_receipt_sha256") != source_receipt_sha
            or manifest.get("bf16_receipt_sha256") != bf16_receipt_sha
            or manifest.get("quant_receipt_sha256") != _sha256(quant_receipt_path)
            or manifest.get("mmproj_receipt_sha256") != _sha256(mmproj_receipt_path)
            or manifest.get("source_verification") != "complete_inventory_rehashed"
            or manifest.get("toolchain_inventories")
            != _published_toolchain_inventory_records(final_root)
        ):
            raise RuntimeError("Export success manifest is malformed")
        return {
            "status": "already_successful",
            "run_id": run_id,
            "manifest_sha256": _sha256(manifest_path),
        }

    if pending is not None:
        publication = _recover_and_commit_publication(
            final_root,
            config=config,
            stage="verify_export",
            canonical_name="verification",
            volume=final_volume,
        )
        assert publication is not None
    else:
        assert attempt is not None
        first = sorted((final_root / "q3_k_m").glob("inkling-Q3_K_M-00001-of-*.gguf"))
        if len(first) != 1:
            raise RuntimeError("Final Q3 split set has no unique first shard")
        verification_paths = WorkflowPaths(
            source_dir=SOURCE_MOUNT,
            work_dir=WORK_MOUNT,
            final_dir=final_root,
            llama_cpp_dir=LLAMA_CPP_DIR,
        )
        verification = build_verification_plan(
            config,
            verification_paths,
            first_q3_split=first[0],
            mmproj_file=_safe_child(final_root, str(mmproj_outputs[0]["path"])),
        )
        _run_streaming(
            verification.q3_split_set.argv,
            log_path=_safe_child(final_root, "logs", f"verify_q3_split_set.{attempt}.log"),
        )
        _run_streaming(
            verification.mmproj.argv,
            log_path=_safe_child(final_root, "logs", f"verify_mmproj.{attempt}.log"),
        )
        partial = _safe_child(final_root, f".partial-verify-export-{attempt}")
        partial.mkdir()
        inventory_directory = _safe_child(partial, "toolchain")
        inventory_directory.mkdir()
        inventory_sources = {
            "python-freeze.txt": LLAMA_CPP_DIR / ".iql-python-freeze.txt",
            "debian-packages.txt": LLAMA_CPP_DIR / ".iql-dpkg-inventory.txt",
        }
        partial_inventory_records: list[dict[str, Any]] = []
        for filename, source in inventory_sources.items():
            destination = _safe_child(inventory_directory, filename)
            shutil.copyfile(source, destination)
            partial_inventory_records.append(_file_record(destination, partial))
        toolchain_inventory_records = [
            {**record, "path": f"verification/{record['path']}"}
            for record in partial_inventory_records
        ]
        manifest_path = _safe_child(partial, "export_manifest.json")
        manifest = {
            "schema_version": "1.0",
            "status": "verified",
            "config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "model_id": config.source.model_id,
            "revision": config.source.revision,
            "license": config.source.license,
            "architecture": config.source.architecture,
            "llama_cpp_commit": config.toolchain.commit,
            "toolchain": toolchain,
            "toolchain_inventories": toolchain_inventory_records,
            "quant_type": config.quantization.quant_type,
            "text": "converted_and_quantized",
            "vision_audio": "separate_bf16_mmproj",
            "mtp": "omitted_unsupported",
            "structural_verification": {
                "q3_split_set": "merge_dry_run_success",
                "mmproj": "split_dry_run_success",
            },
            "source_verification": "complete_inventory_rehashed",
            "source_receipt_sha256": source_receipt_sha,
            "bf16_receipt_sha256": bf16_receipt_sha,
            "quant_receipt_sha256": _sha256(quant_receipt_path),
            "mmproj_receipt_sha256": _sha256(mmproj_receipt_path),
            "quantized_outputs": quant_outputs,
            "mmproj_outputs": mmproj_outputs,
        }
        _immutable_json(manifest_path, manifest)
        publication = _prepare_and_commit_publication(
            final_root,
            config=config,
            stage="verify_export",
            partial=partial,
            canonical=_safe_child(final_root, "verification"),
            outputs=[*partial_inventory_records, _file_record(manifest_path, partial)],
            volume=final_volume,
        )

    manifest_path = _safe_child(final_root, "verification", "export_manifest.json")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("status") != "verified"
        or manifest.get("config_hash") != config.config_hash()
        or manifest.get("control_plane_sha256") != control_plane.tree_sha256
        or manifest.get("toolchain") != toolchain
        or manifest.get("quantized_outputs") != quant_outputs
        or manifest.get("mmproj_outputs") != mmproj_outputs
        or manifest.get("source_receipt_sha256") != source_receipt_sha
        or manifest.get("bf16_receipt_sha256") != bf16_receipt_sha
        or manifest.get("quant_receipt_sha256") != _sha256(quant_receipt_path)
        or manifest.get("mmproj_receipt_sha256") != _sha256(mmproj_receipt_path)
        or manifest.get("source_verification") != "complete_inventory_rehashed"
        or manifest.get("toolchain_inventories")
        != _published_toolchain_inventory_records(final_root)
    ):
        raise RuntimeError(
            "Recovered export manifest is not bound to the verified dependency chain"
        )
    manifest_sha256 = _sha256(manifest_path)
    success_receipt = {
        "status": "success",
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "call_id": _current_invocation_ids()[0],
        "launch_intent_sha256": acknowledgement.launch_intent_sha256,
        "manifest_sha256": manifest_sha256,
        **_publication_evidence(final_root, "verify_export", publication),
    }
    _immutable_json(success_path, success_receipt)
    final_volume.commit()
    return {**manifest, "manifest_sha256": manifest_sha256}
