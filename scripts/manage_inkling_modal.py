"""Deploy and invoke the exact Inkling Modal workflow without ephemeral Apps.

This local control program is part of the hashed execution control plane.  It is
the only supported paid launcher: deployment is captured once under a control-hash-
specific App name, its newest version and concrete Function ID/name bindings
are sealed and rechecked, and accepted Function call IDs are retained locally. The
Modal dashboard hard budget and exclusive-deployer discipline remain external.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Final

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import modal  # noqa: E402
from modal.call_graph import InputInfo, InputStatus  # noqa: E402

EXPECTED_MODAL_VERSION: Final = "1.5.0"
if modal.__version__ != EXPECTED_MODAL_VERSION:
    raise RuntimeError(
        f"This launcher requires Modal {EXPECTED_MODAL_VERSION}, got {modal.__version__}"
    )

from inkling_quant_lab.exceptions import ConfigurationError  # noqa: E402
from inkling_quant_lab.gguf.inkling import (  # noqa: E402
    DASHBOARD_EXACT_UTC_SOURCE,
    FULL_INITIAL_BILLING_WINDOW_POLICY,
    INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
    SHORT_INITIAL_BILLING_WINDOW_POLICY,
    USER_CONFIRMED_ASSUMED_UTC_SOURCE,
    ControlPlaneProvenance,
    InklingGGUFConfig,
    InklingSourceAdoptionReference,
    PaidLaunchAcknowledgement,
    SourceAdoptionArtifact,
    audit_inkling_source,
    audit_pinned_inkling_online,
    compute_cost_ceiling_usd,
    inkling_control_plane_provenance,
    inkling_run_id,
    load_inkling_gguf_config,
    load_inkling_source_adoption_reference,
    require_initial_billing_window,
    require_materialize_initial_billing_window,
    require_stage_billing_window,
    validate_inkling_source_adoption_reference,
)

CONFIG_PATH: Final = PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_modal.yaml"
PAID_APP_PATH: Final = PROJECT_ROOT / "scripts/quantize_inkling_modal.py"
STAGE_ORDER: Final = (
    "materialize_source",
    "convert_text_bf16",
    "convert_multimodal_projector",
    "quantize_text",
    "verify_export",
)
DEPLOYMENT_SCHEMA: Final = "inkling-modal-deployment-v7"
CALL_SCHEMA: Final = "inkling-modal-call-v6"
LAUNCH_INTENT_SCHEMA: Final = "inkling-modal-launch-intent-v5"
LOCK_RECONCILIATION_SCHEMA: Final = "inkling-modal-lock-reconciliation-v1"
MAX_CONTROL_RECEIPT_BYTES: Final = 1024 * 1024
STAGE_RECEIPTS: Final = {
    "materialize_source": ("source_volume", "source.success.json"),
    "convert_text_bf16": ("work_volume", "convert_text_bf16.success.json"),
    "convert_multimodal_projector": (
        "final_volume",
        "convert_multimodal_projector.success.json",
    ),
    "quantize_text": ("final_volume", "quantize_text.success.json"),
    "verify_export": ("final_volume", "verify_export.success.json"),
}
SOURCE_ADOPTION_LOCAL_ARTIFACTS: Final = (
    "local_deployment_receipt",
    "local_materialize_call_receipt",
    "local_materialize_launch_intent",
)
SOURCE_ADOPTION_VOLUME_ARTIFACTS: Final = (
    "source_success_receipt",
    "source_inventory",
    "origin_resolved_config",
    "origin_control_plane",
    "origin_materialize_attempt_ledger",
    "origin_materialize_invocation_history",
    "snapshot_config",
    "snapshot_weight_index",
)


def _load_checked_context() -> tuple[InklingGGUFConfig, ControlPlaneProvenance, str]:
    config = load_inkling_gguf_config(CONFIG_PATH)
    if config.config_hash() != InklingGGUFConfig().config_hash():
        raise RuntimeError("Checked Inkling YAML differs from the frozen deployment schema")
    control_plane = inkling_control_plane_provenance(PROJECT_ROOT)
    return config, control_plane, inkling_run_id(config, control_plane.tree_sha256)


def _load_source_adoption_for_target(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
) -> InklingSourceAdoptionReference:
    """Load the one checked reference and bind it to this corrected control plane."""

    relative_path = INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH
    reference_path = PROJECT_ROOT / relative_path
    reference = load_inkling_source_adoption_reference(reference_path)
    validate_inkling_source_adoption_reference(
        reference,
        target_config=config,
        target_control_plane_sha256=control_plane.tree_sha256,
    )
    matching_files = [item for item in control_plane.files if item.path == relative_path]
    if len(matching_files) != 1:
        raise RuntimeError("Control plane does not bind the exact source-adoption reference path")
    raw = reference_path.read_bytes()
    if (
        matching_files[0].size_bytes != len(raw)
        or matching_files[0].sha256 != hashlib.sha256(raw).hexdigest()
    ):
        raise RuntimeError("Control-plane source-adoption reference bytes drifted after hashing")
    return reference


def _load_checked_source_adoption(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
) -> InklingSourceAdoptionReference:
    """Load the reference supplied by a checked manager context."""

    del run_id
    return _load_source_adoption_for_target(config, control_plane)


def _is_checked_target_run(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
) -> bool:
    """Distinguish the real deterministic run from synthetic helper-level fixtures."""

    return run_id == inkling_run_id(config, control_plane.tree_sha256)


def _source_adoption_binding(
    reference: InklingSourceAdoptionReference,
) -> dict[str, str]:
    """Return the compact lineage fields repeated in every local control receipt."""

    return {
        "source_adoption_reference_sha256": reference.reference_sha256,
        "source_adoption_origin_run_id": reference.origin_run_id,
        "source_adoption_origin_app_name": reference.origin_app_name,
        "source_adoption_origin_app_id": reference.origin_app_id,
    }


def _has_source_adoption_binding(
    value: Mapping[str, Any],
    reference: InklingSourceAdoptionReference,
) -> bool:
    return all(
        value.get(name) == expected
        for name, expected in _source_adoption_binding(reference).items()
    )


def _require_paid_gate(config: InklingGGUFConfig, workspace_budget_usd: Decimal) -> str:
    required = config.budget.workspace_hard_budget_usd
    confirmed = os.environ.get("IQL_MODAL_WORKSPACE_BUDGET_CONFIRMED")
    cycle_end = os.environ.get("IQL_MODAL_BILLING_CYCLE_END_CONFIRMED")
    if workspace_budget_usd != required or confirmed != str(int(required)) or not cycle_end:
        raise RuntimeError(
            "Before deployment or launch, visibly activate the Modal workspace hard budget at "
            "$800, export IQL_MODAL_WORKSPACE_BUDGET_CONFIRMED=800, and pass "
            "--workspace-budget-usd 800. Also export the dashboard's current cycle end as "
            "IQL_MODAL_BILLING_CYCLE_END_CONFIRMED=YYYY-MM-DDTHH:MM:SSZ"
        )
    return cycle_end


def _initial_billing_window_evidence(
    config: InklingGGUFConfig,
    cycle_end_utc: str,
    *,
    accept_short_initial_window_risk: bool,
) -> tuple[str, str]:
    """Apply the strict gate or one explicitly cycle-bound deploy-only waiver."""

    try:
        require_initial_billing_window(config, cycle_end_utc)
    except ConfigurationError:
        if not accept_short_initial_window_risk:
            raise
        if os.environ.get("IQL_MODAL_SHORT_CYCLE_CONFIRMED") != cycle_end_utc:
            raise RuntimeError(
                "The short-window override requires IQL_MODAL_SHORT_CYCLE_CONFIRMED to "
                "exactly equal IQL_MODAL_BILLING_CYCLE_END_CONFIRMED"
            ) from None
        require_materialize_initial_billing_window(config, cycle_end_utc)
        return SHORT_INITIAL_BILLING_WINDOW_POLICY, USER_CONFIRMED_ASSUMED_UTC_SOURCE
    return FULL_INITIAL_BILLING_WINDOW_POLICY, DASHBOARD_EXACT_UTC_SOURCE


def _deployment_tag(control_plane: ControlPlaneProvenance) -> str:
    return f"iql-{control_plane.tree_sha256[:32]}"


def _deployment_name(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
) -> str:
    """Use an implementation-addressed App name without a paid plan feature."""

    return f"{config.modal.app_name}-{control_plane.tree_sha256[:12]}"


def _run_root(run_id: str) -> Path:
    return PROJECT_ROOT / "artifacts" / "inkling-modal" / run_id


def _deployment_path(run_id: str) -> Path:
    return _run_root(run_id) / "deployment.json"


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def _valid_initial_billing_evidence(value: Mapping[str, Any]) -> bool:
    return (
        value.get("initial_billing_window_policy"),
        value.get("billing_cycle_end_source"),
    ) in {
        (FULL_INITIAL_BILLING_WINDOW_POLICY, DASHBOARD_EXACT_UTC_SOURCE),
        (SHORT_INITIAL_BILLING_WINDOW_POLICY, USER_CONFIRMED_ASSUMED_UTC_SOURCE),
    }


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    """Publish complete local JSON once; a hard death can strand only a hidden temp."""

    expected = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Unsafe immutable control receipt: {path}")
        if path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Refusing to replace immutable control receipt: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(16)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise RuntimeError(f"Unsafe immutable control receipt: {path}") from None
            if path.read_text(encoding="utf-8") != expected:
                raise RuntimeError(
                    f"Refusing to replace immutable control receipt: {path}"
                ) from None
        else:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def _local_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _exclusive_manager_operation(run_id: str, operation: str) -> Iterator[None]:
    """Serialize local deploy/launch operations; retain the lock after hard process death."""

    lock_path = _run_root(run_id) / "control" / "manager.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = {
        "schema_version": "inkling-modal-manager-lock-v1",
        "run_id": run_id,
        "operation": operation,
        "pid": os.getpid(),
        "created_at": datetime.now(UTC).isoformat(),
        "nonce": secrets.token_hex(16),
    }
    expected = _canonical_json(lock)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"Another manager operation or unresolved local crash holds {lock_path}"
        ) from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        if lock_path.is_symlink() or not lock_path.is_file():
            raise RuntimeError(f"Manager operation lock disappeared or became unsafe: {lock_path}")
        if lock_path.read_text(encoding="utf-8") != expected:
            raise RuntimeError(f"Manager operation lock changed unexpectedly: {lock_path}")
        lock_path.unlink()


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Required control receipt is missing or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object at {path}")
    return value


def _json_object_from_exact_bytes(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Exact adoption artifact is not valid UTF-8 JSON for {label}"
        ) from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Exact adoption artifact is not a JSON object for {label}")
    return value


def _read_exact_local_adoption_json(
    artifact: SourceAdoptionArtifact,
    *,
    label: str,
) -> dict[str, Any]:
    """Read one project-relative origin artifact only when its exact bytes still match."""

    relative = PurePosixPath(artifact.path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"Unsafe local adoption artifact path for {label}")
    path = PROJECT_ROOT.joinpath(*relative.parts)
    root = PROJECT_ROOT.resolve()
    if not path.resolve(strict=False).is_relative_to(root):
        raise RuntimeError(f"Local adoption artifact escapes the project root for {label}")
    current = PROJECT_ROOT
    if current.is_symlink():
        raise RuntimeError(f"Local adoption artifact has a symlinked root for {label}")
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise RuntimeError(f"Local adoption artifact contains a symlink for {label}")
    if not path.is_file():
        raise RuntimeError(f"Required local adoption artifact is missing for {label}")
    before = path.stat()
    if before.st_size != artifact.size_bytes or before.st_size > MAX_CONTROL_RECEIPT_BYTES:
        raise RuntimeError(f"Local adoption artifact size drifted for {label}")
    payload = path.read_bytes()
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError(f"Local adoption artifact changed while reading for {label}")
    if len(payload) != artifact.size_bytes:
        raise RuntimeError(f"Local adoption artifact size drifted for {label}")
    if hashlib.sha256(payload).hexdigest() != artifact.sha256:
        raise RuntimeError(f"Local adoption artifact SHA-256 drifted for {label}")
    return _json_object_from_exact_bytes(payload, label=label)


def _read_exact_volume_adoption_json(
    volume: Any,
    artifact: SourceAdoptionArtifact,
    *,
    mount_path: str,
    label: str,
) -> dict[str, Any]:
    """Read one exact small origin file through the read-only Volume API."""

    absolute = PurePosixPath(artifact.path)
    mount = PurePosixPath(mount_path)
    try:
        relative = absolute.relative_to(mount)
    except ValueError as error:
        raise RuntimeError(f"Volume adoption artifact is outside its mount for {label}") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"Unsafe Volume adoption artifact path for {label}")
    remote_path = relative.as_posix()
    payload = bytearray()
    try:
        for chunk in volume.read_file(remote_path):
            if not isinstance(chunk, bytes):
                raise RuntimeError(f"Modal returned non-byte adoption content for {label}")
            payload.extend(chunk)
            if len(payload) > MAX_CONTROL_RECEIPT_BYTES:
                raise RuntimeError(f"Modal adoption artifact is unexpectedly large for {label}")
    except FileNotFoundError as error:
        raise RuntimeError(f"Required Volume adoption artifact is missing for {label}") from error
    raw = bytes(payload)
    if len(raw) != artifact.size_bytes:
        raise RuntimeError(f"Volume adoption artifact size drifted for {label}")
    if hashlib.sha256(raw).hexdigest() != artifact.sha256:
        raise RuntimeError(f"Volume adoption artifact SHA-256 drifted for {label}")
    return _json_object_from_exact_bytes(raw, label=label)


def _require_mapping_fields(
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    mismatches = [
        name for name, expected_value in expected.items() if value.get(name) != expected_value
    ]
    if mismatches:
        raise RuntimeError(
            f"{label} has drifted critical identities: {', '.join(sorted(mismatches))}"
        )


def _validate_source_adoption_local_records(
    reference: InklingSourceAdoptionReference,
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(records) != set(SOURCE_ADOPTION_LOCAL_ARTIFACTS):
        raise RuntimeError("Origin local adoption artifact set is incomplete")
    deployment = records["local_deployment_receipt"]
    call = records["local_materialize_call_receipt"]
    intent = records["local_materialize_launch_intent"]
    _require_mapping_fields(
        deployment,
        {
            "schema_version": "inkling-modal-deployment-v6",
            "app_name": reference.origin_app_name,
            "environment_name": "inkling-quant",
            "run_id": reference.origin_run_id,
            "config_hash": reference.origin_config_hash,
            "control_plane_sha256": reference.origin_control_plane_sha256,
        },
        label="Origin local deployment receipt",
    )
    bindings = deployment.get("stage_function_bindings")
    if not isinstance(bindings, dict):
        raise RuntimeError("Origin local deployment receipt has no Function bindings")
    materialize_binding = bindings.get("materialize_source")
    if not isinstance(materialize_binding, dict):
        raise RuntimeError("Origin local deployment receipt has no materialize_source binding")
    function_id = materialize_binding.get("function_id")
    if (
        materialize_binding.get("function_name") != "materialize_source"
        or not isinstance(function_id, str)
        or not function_id.startswith("fu-")
    ):
        raise RuntimeError("Origin materialize_source Function binding drifted")
    _require_mapping_fields(
        intent,
        {
            "schema_version": "inkling-modal-launch-intent-v4",
            "status": "prepared_before_spawn",
            "run_id": reference.origin_run_id,
            "stage": "materialize_source",
            "app_name": reference.origin_app_name,
            "environment_name": "inkling-quant",
            "function_id": function_id,
            "function_name": "materialize_source",
            "config_hash": reference.origin_config_hash,
            "control_plane_sha256": reference.origin_control_plane_sha256,
        },
        label="Origin local materialize launch intent",
    )
    if (
        reference.local_materialize_launch_intent.sha256
        != reference.materialize_launch_intent_sha256
    ):
        raise RuntimeError("Origin launch-intent artifact hash is not its launch identity")
    origin_local_root = PurePosixPath("artifacts", "inkling-modal", reference.origin_run_id)
    intent_relative = PurePosixPath(reference.local_materialize_launch_intent.path).relative_to(
        origin_local_root
    )
    _require_mapping_fields(
        call,
        {
            "schema_version": "inkling-modal-call-v5",
            "status": "accepted",
            "run_id": reference.origin_run_id,
            "stage": "materialize_source",
            "call_id": reference.materialize_function_call_id,
            "app_name": reference.origin_app_name,
            "environment_name": "inkling-quant",
            "function_id": function_id,
            "function_name": "materialize_source",
            "launch_intent": intent_relative.as_posix(),
            "launch_intent_sha256": reference.materialize_launch_intent_sha256,
            "config_hash": reference.origin_config_hash,
            "control_plane_sha256": reference.origin_control_plane_sha256,
        },
        label="Origin local materialize call receipt",
    )
    for field in ("deployment_version", "deployment_tag", "billing_cycle_end_utc"):
        if call.get(field) != intent.get(field):
            raise RuntimeError(f"Origin local call/intent {field} binding drifted")


def _validate_source_adoption_local_artifacts(
    reference: InklingSourceAdoptionReference,
) -> None:
    records = {
        name: _read_exact_local_adoption_json(
            getattr(reference, name),
            label=name,
        )
        for name in SOURCE_ADOPTION_LOCAL_ARTIFACTS
    }
    _validate_source_adoption_local_records(reference, records)


def _validate_source_inventory_records(
    records: Mapping[str, Any],
    *,
    expected_count: int,
) -> None:
    if len(records) != expected_count:
        raise RuntimeError("Origin source inventory file count drifted")
    for name, raw_record in records.items():
        if not isinstance(name, str) or not isinstance(raw_record, dict):
            raise RuntimeError("Origin source inventory contains a malformed record")
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or path.as_posix() != name
            or any(part in {"", ".", ".."} for part in path.parts)
            or raw_record.get("path") != name
            or not isinstance(raw_record.get("size_bytes"), int)
            or raw_record["size_bytes"] <= 0
            or re.fullmatch(r"[0-9a-f]{64}", str(raw_record.get("sha256"))) is None
        ):
            raise RuntimeError("Origin source inventory contains a malformed file identity")


def _validate_source_adoption_volume_records(
    config: InklingGGUFConfig,
    reference: InklingSourceAdoptionReference,
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(records) != set(SOURCE_ADOPTION_VOLUME_ARTIFACTS):
        raise RuntimeError("Origin Volume adoption artifact set is incomplete")
    receipt = records["source_success_receipt"]
    inventory = records["source_inventory"]
    resolved_config = records["origin_resolved_config"]
    control_plane_record = records["origin_control_plane"]
    ledger = records["origin_materialize_attempt_ledger"]
    history = records["origin_materialize_invocation_history"]
    snapshot_config = records["snapshot_config"]
    weight_index = records["snapshot_weight_index"]

    if resolved_config != config.canonical_dict():
        raise RuntimeError("Origin resolved configuration differs from the checked target config")
    try:
        origin_control = ControlPlaneProvenance.model_validate(control_plane_record)
    except ValueError as error:
        raise RuntimeError("Origin control-plane record is malformed") from error
    if (
        origin_control.tree_sha256 != reference.origin_control_plane_sha256
        or origin_control.file_count != reference.origin_control_plane_file_count
    ):
        raise RuntimeError("Origin control-plane identity drifted")

    _require_mapping_fields(
        receipt,
        {
            "verified": True,
            "config_hash": reference.origin_config_hash,
            "control_plane_sha256": reference.origin_control_plane_sha256,
            "model_id": reference.model_id,
            "revision": reference.revision,
            "license": reference.license,
            "source_dir": reference.snapshot_path,
            "weight_index_sha256": reference.weight_index_canonical_sha256,
            "source_tensor_bytes": reference.indexed_tensor_bytes,
            "materialized_weight_file_bytes": reference.materialized_weight_file_bytes,
            "source_tensor_count": reference.source_tensor_count,
            "source_shard_count": reference.source_shard_count,
            "file_count": reference.materialized_file_count,
            "inventory_sha256": reference.source_inventory.sha256,
            "source_config_sha256": reference.snapshot_config.sha256,
            "call_id": reference.materialize_function_call_id,
            "launch_intent_sha256": reference.materialize_launch_intent_sha256,
        },
        label="Origin source success receipt",
    )
    raw_files = inventory.get("files")
    if not isinstance(raw_files, dict):
        raise RuntimeError("Origin source inventory has no file mapping")
    _require_mapping_fields(
        inventory,
        {
            "config_hash": reference.origin_config_hash,
            "model_id": reference.model_id,
            "revision": reference.revision,
            "required_file_count": reference.materialized_file_count,
        },
        label="Origin source inventory",
    )
    _validate_source_inventory_records(raw_files, expected_count=reference.materialized_file_count)

    source_audit = audit_inkling_source(
        config,
        model_info={
            "id": reference.model_id,
            "sha": reference.revision,
            "cardData": {"license": reference.license},
        },
        model_config=snapshot_config,
        weight_index=weight_index,
    )
    if (
        source_audit.weight_index_sha256 != reference.weight_index_canonical_sha256
        or source_audit.source_bytes != reference.indexed_tensor_bytes
        or source_audit.source_tensor_count != reference.source_tensor_count
        or source_audit.source_shard_count != reference.source_shard_count
    ):
        raise RuntimeError("Origin snapshot config/index audit drifted")
    raw_weight_map = weight_index.get("weight_map")
    if not isinstance(raw_weight_map, dict):
        raise RuntimeError("Origin snapshot weight index has no weight_map")
    shard_names = {str(name) for name in raw_weight_map.values()}
    if len(shard_names) != reference.source_shard_count or not shard_names.issubset(raw_files):
        raise RuntimeError("Origin source inventory omits indexed shard identities")
    materialized_weight_bytes = sum(int(raw_files[name]["size_bytes"]) for name in shard_names)
    if materialized_weight_bytes != reference.materialized_weight_file_bytes:
        raise RuntimeError("Origin materialized safetensors byte total drifted")

    materialize_limit = next(
        stage.max_attempts for stage in config.modal.stages if stage.name == "materialize_source"
    )
    history_relative = PurePosixPath(
        reference.origin_materialize_invocation_history.path
    ).relative_to(PurePosixPath(reference.source_run_root))
    _require_mapping_fields(
        history,
        {
            "schema_version": "inkling-modal-stage-invocation-v2",
            "status": "started",
            "kind": "attempt",
            "sequence": 1,
            "limit": materialize_limit,
            "stage": "materialize_source",
            "config_hash": reference.origin_config_hash,
            "control_plane_sha256": reference.origin_control_plane_sha256,
            "launch_intent_sha256": reference.materialize_launch_intent_sha256,
            "call_id": reference.materialize_function_call_id,
            "input_id": reference.materialize_input_id,
            "task_id": reference.materialize_task_id,
            "previous_history_sha256": None,
        },
        label="Origin materialize invocation history",
    )
    event_id = hashlib.sha256(
        json.dumps(history, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if history_relative.name != f"materialize_source.attempt.1.{event_id}.json":
        raise RuntimeError("Origin materialize invocation-history filename drifted")
    _require_mapping_fields(
        ledger,
        {
            "attempts": 1,
            "limit": materialize_limit,
            "config_hash": reference.origin_config_hash,
            "control_plane_sha256": reference.origin_control_plane_sha256,
            "launch_intent_sha256": reference.materialize_launch_intent_sha256,
            "last_call_id": reference.materialize_function_call_id,
            "last_input_id": reference.materialize_input_id,
            "last_task_id": reference.materialize_task_id,
            "last_history_path": history_relative.as_posix(),
            "last_history_sha256": reference.origin_materialize_invocation_history.sha256,
        },
        label="Origin materialize attempt ledger",
    )


def _validate_source_adoption_volume_artifacts(
    config: InklingGGUFConfig,
    reference: InklingSourceAdoptionReference,
) -> None:
    if config.modal.source_volume != reference.source_volume:
        raise RuntimeError("Checked target config does not mount the pinned origin source Volume")
    volume = modal.Volume.from_name(
        reference.source_volume,
        environment_name=config.modal.environment_name,
        create_if_missing=False,
        version=config.modal.volume_version,
    )
    volume.hydrate()
    records = {
        name: _read_exact_volume_adoption_json(
            volume,
            getattr(reference, name),
            mount_path=reference.source_mount_path,
            label=name,
        )
        for name in SOURCE_ADOPTION_VOLUME_ARTIFACTS
    }
    _validate_source_adoption_volume_records(config, reference, records)


def _require_origin_call_graph_success(reference: InklingSourceAdoptionReference) -> None:
    call = modal.FunctionCall.from_id(reference.materialize_function_call_id)
    graph = call.get_call_graph()
    if not isinstance(graph, list) or len(graph) != 1:
        raise RuntimeError("Origin materialize call graph must contain exactly one root")
    root = graph[0]
    children = getattr(root, "children", None)
    if not isinstance(children, list) or children:
        raise RuntimeError("Origin materialize call graph root must have zero children")
    # Modal's graph omits task_id and strips the immutable invocation suffix from
    # input_id. The exact input/task pair is instead required above in both the
    # hash-pinned invocation-history event and its attempt-ledger binding.
    expected = {
        "function_call_id": reference.materialize_function_call_id,
        "status": InputStatus.SUCCESS,
    }
    mismatches = [name for name, value in expected.items() if getattr(root, name, None) != value]
    if mismatches:
        raise RuntimeError(
            "Origin materialize root call graph identity drifted: " + ", ".join(sorted(mismatches))
        )
    if reference.materialize_call_child_count != 0 or reference.materialize_continuation_call_ids:
        raise RuntimeError("Checked origin reference permits unexpected materialize continuations")


def _modal_app_list(config: InklingGGUFConfig) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "app",
            "list",
            "-e",
            config.modal.environment_name,
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise RuntimeError("Modal App list returned an unexpected JSON shape")
    return value


def _require_origin_app_stopped(
    config: InklingGGUFConfig,
    reference: InklingSourceAdoptionReference,
) -> None:
    candidates = [
        row
        for row in _modal_app_list(config)
        if row.get("app_id") == reference.origin_app_id
        or row.get("description") == reference.origin_app_name
    ]
    if len(candidates) != 1:
        raise RuntimeError("Expected exactly one Modal App matching the pinned origin ID/name")
    row = candidates[0]
    if (
        row.get("app_id") != reference.origin_app_id
        or row.get("description") != reference.origin_app_name
        or row.get("state") != reference.origin_app_required_state
        or row.get("tasks") not in (reference.origin_app_required_active_tasks, "0")
    ):
        raise RuntimeError("Pinned origin App must remain stopped with zero active tasks")


def _require_source_adoption_origin_ready(
    config: InklingGGUFConfig,
    reference: InklingSourceAdoptionReference,
) -> dict[str, str]:
    """Reprove all cheap origin evidence without starting a Modal container."""

    _validate_source_adoption_local_artifacts(reference)
    _validate_source_adoption_volume_artifacts(config, reference)
    _require_origin_call_graph_success(reference)
    _require_origin_app_stopped(config, reference)
    return _source_adoption_binding(reference)


def _reconcile_stale_manager_lock(
    run_id: str,
    *,
    expected_sha256: str,
    confirm_owner_process_stopped: bool,
) -> None:
    """Remove one exact stale lock only after an explicit operator assertion."""

    lock_path = _run_root(run_id) / "control" / "manager.lock"
    if not confirm_owner_process_stopped:
        raise RuntimeError("Lock reconciliation requires --confirm-owner-process-stopped")
    if lock_path.is_symlink() or not lock_path.is_file():
        raise RuntimeError(f"No safe manager lock exists at {lock_path}")
    actual_sha256 = _local_sha256(lock_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError("Manager lock SHA-256 differs from the operator-confirmed bytes")
    try:
        lock = _read_object(lock_path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        lock = None
    if lock is not None:
        if (
            lock.get("schema_version") != "inkling-modal-manager-lock-v1"
            or lock.get("run_id") != run_id
        ):
            raise RuntimeError("Manager lock does not belong to this run")
        pid = lock.get("pid")
        if not isinstance(pid, int) or pid < 1:
            raise RuntimeError("Manager lock has an invalid owner PID")
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        except PermissionError as error:
            raise RuntimeError("Manager lock owner PID may still be alive") from error
        else:
            raise RuntimeError("Manager lock owner PID is still alive")
    reconciliation = {
        "schema_version": LOCK_RECONCILIATION_SCHEMA,
        "status": "operator_confirmed_owner_stopped",
        "run_id": run_id,
        "manager_lock_sha256": actual_sha256,
    }
    receipt_path = _run_root(run_id) / "reconciliations" / f"manager-lock-{actual_sha256}.json"
    _write_immutable_json(receipt_path, reconciliation)
    if _local_sha256(lock_path) != actual_sha256:
        raise RuntimeError("Manager lock changed during reconciliation")
    lock_path.unlink()


def _read_volume_receipt(
    config: InklingGGUFConfig,
    *,
    run_id: str,
    stage: str,
) -> dict[str, Any] | None:
    """Read one small committed stage marker without starting Modal compute."""

    volume_attribute, filename = STAGE_RECEIPTS[stage]
    return _read_volume_json_file(
        config,
        volume_attribute=volume_attribute,
        remote_path=f"runs/{run_id}/{filename}",
        label=f"stage receipt for {stage}",
    )


def _read_volume_json_file(
    config: InklingGGUFConfig,
    *,
    volume_attribute: str,
    remote_path: str,
    label: str,
) -> dict[str, Any] | None:
    result = _read_volume_json_file_with_sha256(
        config,
        volume_attribute=volume_attribute,
        remote_path=remote_path,
        label=label,
    )
    return None if result is None else result[0]


def _read_volume_json_file_with_sha256(
    config: InklingGGUFConfig,
    *,
    volume_attribute: str,
    remote_path: str,
    label: str,
) -> tuple[dict[str, Any], str] | None:
    volume_name = getattr(config.modal, volume_attribute)
    volume = modal.Volume.from_name(
        volume_name,
        environment_name=config.modal.environment_name,
        create_if_missing=False,
        version=config.modal.volume_version,
    )
    return _read_json_from_volume(volume, remote_path=remote_path, label=label)


def _read_json_from_volume(
    volume: modal.Volume,
    *,
    remote_path: str,
    label: str,
) -> tuple[dict[str, Any], str] | None:
    payload = bytearray()
    try:
        for chunk in volume.read_file(remote_path):
            if not isinstance(chunk, bytes):
                raise RuntimeError(f"Modal returned non-byte content for {label}")
            payload.extend(chunk)
            if len(payload) > MAX_CONTROL_RECEIPT_BYTES:
                raise RuntimeError(f"Modal control file is unexpectedly large for {label}")
    except FileNotFoundError:
        return None
    try:
        value = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Modal control file is not valid UTF-8 JSON for {label}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Modal control file is not a JSON object for {label}")
    return value, hashlib.sha256(payload).hexdigest()


def _authoritative_invocation_history(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    *,
    volume_attribute: str,
    run_id: str,
    stage: str,
    kind: str,
    limit: int,
) -> list[tuple[str, dict[str, Any], str]]:
    """Enumerate the immutable remote history that authoritatively spends slots."""

    volume_name = getattr(config.modal, volume_attribute)
    volume = modal.Volume.from_name(
        volume_name,
        environment_name=config.modal.environment_name,
        create_if_missing=False,
        version=config.modal.volume_version,
    )
    # Hydrate before handling a missing history path so a missing named Volume
    # remains a hard error rather than looking like a run with no invocations.
    volume.hydrate()
    history_root = f"runs/{run_id}/control/history"
    try:
        listed = volume.listdir(history_root, recursive=False)
    except (FileNotFoundError, modal.exception.NotFoundError):
        return []
    prefix = f"{history_root}/{stage}.{kind}."
    pattern = re.compile(rf"^{re.escape(prefix)}([1-9][0-9]*)\.([0-9a-f]{{64}})\.json$")
    entries: list[tuple[str, dict[str, Any], str]] = []
    for item in listed:
        if not item.path.startswith(prefix):
            continue
        match = pattern.fullmatch(item.path)
        if match is None:
            raise RuntimeError(f"Malformed immutable invocation-history path: {item.path}")
        result = _read_json_from_volume(
            volume,
            remote_path=item.path,
            label=f"immutable invocation history at {item.path}",
        )
        if result is None:
            raise RuntimeError(f"Listed invocation history disappeared: {item.path}")
        event, sha256 = result
        sequence = int(match.group(1))
        event_id = hashlib.sha256(
            json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if (
            match.group(2) != event_id
            or event.get("schema_version") != "inkling-modal-stage-invocation-v2"
            or event.get("status") != "started"
            or event.get("kind") != kind
            or event.get("sequence") != sequence
            or event.get("limit") != limit
            or event.get("stage") != stage
            or event.get("config_hash") != config.config_hash()
            or event.get("control_plane_sha256") != control_plane.tree_sha256
            or re.fullmatch(r"[0-9a-f]{64}", str(event.get("launch_intent_sha256"))) is None
            or not isinstance(event.get("call_id"), str)
            or not event.get("call_id")
            or not isinstance(event.get("input_id"), str)
            or not event.get("input_id")
            or not isinstance(event.get("task_id"), str)
            or not event.get("task_id")
        ):
            raise RuntimeError(f"Immutable invocation history drifted: {item.path}")
        entries.append((item.path, event, sha256))
    entries.sort(key=lambda item: int(item[1]["sequence"]))
    if [item[1]["sequence"] for item in entries] != list(range(1, len(entries) + 1)):
        raise RuntimeError(f"Immutable {stage} {kind} history is not contiguous")
    previous_sha256: str | None = None
    for _, event, sha256 in entries:
        if event.get("previous_history_sha256") != previous_sha256:
            raise RuntimeError(f"Immutable {stage} {kind} history chain drifted")
        previous_sha256 = sha256
    return entries


def _validate_volume_receipt(
    receipt: Mapping[str, Any],
    *,
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
    stage: str,
) -> None:
    expected_success = (
        receipt.get("verified") is True
        if stage == "materialize_source"
        else receipt.get("status") == "success"
    )
    if (
        not expected_success
        or receipt.get("config_hash") != config.config_hash()
        or receipt.get("control_plane_sha256") != control_plane.tree_sha256
    ):
        raise RuntimeError(f"Committed Modal receipt drifted for {stage}")
    if stage == "materialize_source" and _is_checked_target_run(config, control_plane, run_id):
        _validate_current_adopted_source_receipt(
            receipt,
            config=config,
            control_plane=control_plane,
            run_id=run_id,
        )


def _expected_adopted_source_origin(
    reference: InklingSourceAdoptionReference,
) -> dict[str, Any]:
    return {
        "run_id": reference.origin_run_id,
        "config_hash": reference.origin_config_hash,
        "control_plane_sha256": reference.origin_control_plane_sha256,
        "source_success_sha256": reference.source_success_receipt.sha256,
        "source_inventory_sha256": reference.source_inventory.sha256,
        "source_config_sha256": reference.snapshot_config.sha256,
        "weight_index_file_sha256": reference.snapshot_weight_index.sha256,
        "materialize_function_call_id": reference.materialize_function_call_id,
        "materialize_input_id": reference.materialize_input_id,
        "materialize_task_id": reference.materialize_task_id,
    }


def _validate_current_adopted_source_receipt(
    receipt: Mapping[str, Any],
    *,
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
) -> None:
    """Require the real current source marker to be the checked adoption, never a download."""

    reference = _load_checked_source_adoption(config, control_plane, run_id)
    reference_files = [
        item
        for item in control_plane.files
        if item.path == INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH
    ]
    if len(reference_files) != 1:
        raise RuntimeError("Current control plane has no unique adoption reference file")
    expected = {
        "schema_version": "inkling-adopted-source-success-v2",
        "status": "success",
        "verified": True,
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "run_id": run_id,
        "materialization_kind": "adopted_verified_source_v1",
        "source_dir": reference.snapshot_path,
        "source_adoption_reference_sha256": reference.reference_sha256,
        "source_adoption_file_sha256": reference_files[0].sha256,
        "origin": _expected_adopted_source_origin(reference),
        "model_id": config.source.model_id,
        "revision": config.source.revision,
        "license": config.source.license,
        "weight_index_sha256": reference.weight_index_canonical_sha256,
        "source_tensor_bytes": reference.indexed_tensor_bytes,
        "materialized_weight_file_bytes": reference.materialized_weight_file_bytes,
        "source_tensor_count": reference.source_tensor_count,
        "source_shard_count": reference.source_shard_count,
        "file_count": reference.materialized_file_count,
        "inventory_sha256": reference.source_inventory.sha256,
        "source_config_sha256": reference.snapshot_config.sha256,
    }
    mismatches = [name for name, value in expected.items() if receipt.get(name) != value]
    toolchain = receipt.get("toolchain")
    if not isinstance(toolchain, dict) or toolchain.get("commit") != config.toolchain.commit:
        mismatches.append("toolchain")
    if mismatches:
        raise RuntimeError(
            "Current materialize_source receipt lacks the exact adopted source binding: "
            + ", ".join(sorted(set(mismatches)))
        )
    _validate_adopted_source_invocation_binding(
        receipt,
        config=config,
        control_plane=control_plane,
        run_id=run_id,
    )


def _validate_adopted_source_invocation_binding(
    receipt: Mapping[str, Any],
    *,
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
) -> None:
    """Bind the marker to its creation attempt and every later validation attempt."""

    resources = next(item for item in config.modal.stages if item.name == "materialize_source")
    history = _authoritative_invocation_history(
        config,
        control_plane,
        volume_attribute="source_volume",
        run_id=run_id,
        stage="materialize_source",
        kind="attempt",
        limit=resources.max_attempts,
    )
    if not history:
        raise RuntimeError("Adopted current source marker has no immutable attempt history")
    latest_path, latest_event, latest_sha256 = history[-1]
    run_prefix = f"runs/{run_id}/"
    if not latest_path.startswith(run_prefix):
        raise RuntimeError("Latest adopted-source invocation history path escapes its run")
    relative_history_path = latest_path.removeprefix(run_prefix)
    ledger_result = _read_volume_json_file(
        config,
        volume_attribute="source_volume",
        remote_path=f"runs/{run_id}/control/materialize_source.attempts.json",
        label="current adopted source attempt ledger",
    )
    ledger_expected = {
        "attempts": len(history),
        "limit": resources.max_attempts,
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "last_history_path": relative_history_path,
        "last_history_sha256": latest_sha256,
        "last_call_id": latest_event["call_id"],
        "last_input_id": latest_event["input_id"],
        "last_task_id": latest_event["task_id"],
        "launch_intent_sha256": latest_event["launch_intent_sha256"],
    }
    if ledger_result is None or any(
        ledger_result.get(name) != value for name, value in ledger_expected.items()
    ):
        raise RuntimeError("Current adopted source materialization attempt ledger drifted")

    creation_sequence = receipt.get("invocation_sequence")
    if (
        not isinstance(creation_sequence, int)
        or creation_sequence < 1
        or creation_sequence > len(history)
    ):
        raise RuntimeError("Current adopted source marker has no valid creation invocation")
    creation_path, creation_event, creation_sha256 = history[creation_sequence - 1]
    if not creation_path.startswith(run_prefix):
        raise RuntimeError("Adopted-source creation history path escapes its run")
    creation_expected = {
        "invocation_sequence": creation_event["sequence"],
        "invocation_history_path": creation_path.removeprefix(run_prefix),
        "invocation_history_sha256": creation_sha256,
        "call_id": creation_event["call_id"],
        "input_id": creation_event["input_id"],
        "task_id": creation_event["task_id"],
        "launch_intent_sha256": creation_event["launch_intent_sha256"],
    }
    if any(receipt.get(name) != value for name, value in creation_expected.items()):
        raise RuntimeError(
            "Current adopted source marker is not bound to its exact creation invocation"
        )

    validations, source_success_sha256 = _completed_adoption_validations_from_volume(
        config,
        run_id=run_id,
        expected_source_receipt=receipt,
    )
    expected_sequences = set(range(creation_sequence + 1, len(history) + 1))
    if set(validations) != expected_sequences:
        raise RuntimeError(
            "Current adopted source completed-validation chain is incomplete or unexpected"
        )
    toolchain = receipt.get("toolchain")
    reference_sha256 = receipt.get("source_adoption_reference_sha256")
    if not isinstance(toolchain, dict) or not isinstance(reference_sha256, str):
        raise RuntimeError("Current adopted source marker has invalid validation identities")
    for sequence in sorted(validations):
        path, event, sha256 = history[sequence - 1]
        if not path.startswith(run_prefix):
            raise RuntimeError("Adopted-source validation history path escapes its run")
        expected_validation = {
            "schema_version": "inkling-adopted-source-completed-validation-v1",
            "status": "success",
            "verified": True,
            "run_id": run_id,
            "config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "source_success_sha256": source_success_sha256,
            "source_adoption_reference_sha256": reference_sha256,
            "toolchain": dict(toolchain),
            "invocation_sequence": event["sequence"],
            "invocation_history_path": path.removeprefix(run_prefix),
            "invocation_history_sha256": sha256,
            "call_id": event["call_id"],
            "input_id": event["input_id"],
            "task_id": event["task_id"],
            "launch_intent_sha256": event["launch_intent_sha256"],
        }
        if validations[sequence] != expected_validation:
            raise RuntimeError("Current adopted source completed-validation receipt drifted")


def _completed_adoption_validations_from_volume(
    config: InklingGGUFConfig,
    *,
    run_id: str,
    expected_source_receipt: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], str]:
    """Read the exact marker bytes and immutable later-validation records."""

    volume = modal.Volume.from_name(
        config.modal.source_volume,
        environment_name=config.modal.environment_name,
        create_if_missing=False,
        version=config.modal.volume_version,
    )
    volume.hydrate()
    marker_path = f"runs/{run_id}/source.success.json"
    marker_result = _read_json_from_volume(
        volume,
        remote_path=marker_path,
        label="current adopted source marker",
    )
    if marker_result is None or marker_result[0] != dict(expected_source_receipt):
        raise RuntimeError("Current adopted source marker changed during validation")
    _, source_success_sha256 = marker_result

    history_root = f"runs/{run_id}/control/history"
    try:
        listed = volume.listdir(history_root, recursive=False)
    except (FileNotFoundError, modal.exception.NotFoundError):
        listed = []
    prefix = f"{history_root}/materialize_source.completed_validation."
    pattern = re.compile(rf"^{re.escape(prefix)}([1-9][0-9]*)\.([0-9a-f]{{64}})\.json$")
    validations: dict[int, dict[str, Any]] = {}
    for item in listed:
        if not item.path.startswith(prefix):
            continue
        match = pattern.fullmatch(item.path)
        if match is None:
            raise RuntimeError(f"Malformed adopted-source completed-validation path: {item.path}")
        result = _read_json_from_volume(
            volume,
            remote_path=item.path,
            label=f"adopted-source completed validation at {item.path}",
        )
        if result is None:
            raise RuntimeError(f"Listed adopted-source validation disappeared: {item.path}")
        record, _ = result
        sequence = int(match.group(1))
        record_id = hashlib.sha256(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if (
            match.group(2) != record_id
            or record.get("invocation_sequence") != sequence
            or sequence in validations
        ):
            raise RuntimeError("Adopted source completed-validation history drifted")
        validations[sequence] = record
    return validations, source_success_sha256


def _require_stage_invocation_budget(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
    stage: str,
) -> int:
    """Reject spent ordinary/recovery ledgers locally before another startup."""

    volume_attribute, _ = STAGE_RECEIPTS[stage]
    control_root = f"runs/{run_id}/control"
    publication_intent = _read_volume_json_file(
        config,
        volume_attribute=volume_attribute,
        remote_path=f"{control_root}/{stage}.publication.intent.json",
        label=f"publication intent for {stage}",
    )
    resources = next(item for item in config.modal.stages if item.name == stage)
    if publication_intent is None:
        ledger_name = "attempts"
        limit = resources.max_attempts
    else:
        if (
            publication_intent.get("stage") != stage
            or publication_intent.get("config_hash") != config.config_hash()
        ):
            raise RuntimeError(f"Publication intent drifted for {stage}")
        ledger_name = "recoveries"
        limit = resources.max_recovery_attempts
    ledger = _read_volume_json_file(
        config,
        volume_attribute=volume_attribute,
        remote_path=f"{control_root}/{stage}.{ledger_name}.json",
        label=f"{ledger_name} ledger for {stage}",
    )
    history = _authoritative_invocation_history(
        config,
        control_plane,
        volume_attribute=volume_attribute,
        run_id=run_id,
        stage=stage,
        kind="attempt" if ledger_name == "attempts" else "recovery",
        limit=limit,
    )
    latest_count = 0 if ledger is None else ledger.get(ledger_name)
    if (
        not isinstance(latest_count, int)
        or latest_count < 0
        or latest_count > len(history)
        or (
            ledger is not None
            and (
                ledger.get("limit") != limit
                or ledger.get("config_hash") != config.config_hash()
                or ledger.get("control_plane_sha256") != control_plane.tree_sha256
                or (
                    latest_count > 0
                    and (
                        ledger.get("last_history_path")
                        != history[latest_count - 1][0].removeprefix(f"runs/{run_id}/")
                        or ledger.get("last_history_sha256") != history[latest_count - 1][2]
                        or ledger.get("launch_intent_sha256")
                        != history[latest_count - 1][1]["launch_intent_sha256"]
                        or ledger.get("last_call_id") != history[latest_count - 1][1]["call_id"]
                        or ledger.get("last_input_id") != history[latest_count - 1][1]["input_id"]
                        or ledger.get("last_task_id") != history[latest_count - 1][1]["task_id"]
                    )
                )
            )
        )
    ):
        raise RuntimeError(f"Committed {ledger_name} ledger drifted for {stage}")
    consumed = len(history)
    if consumed >= limit:
        raise RuntimeError(
            f"Stage {stage} has spent its {limit}-{ledger_name} launch cap; refusing startup"
        )
    return limit - consumed


def _launch_intent_file(run_root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise RuntimeError("Local call receipt has no launch-intent path")
    candidate = Path(relative)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or candidate.parts[:1] != ("launch-intents",)
    ):
        raise RuntimeError("Local call receipt has an unsafe launch-intent path")
    path = run_root.joinpath(*candidate.parts)
    if not path.resolve().is_relative_to(run_root.resolve()):
        raise RuntimeError("Local call receipt launch intent escapes the run root")
    return path


def _validated_local_call_receipts(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
) -> list[dict[str, Any]]:
    reference = _load_source_adoption_for_target(config, control_plane)
    require_adoption_binding = _is_checked_target_run(config, control_plane, run_id)
    run_root = _run_root(run_id)
    calls_directory = run_root / "calls"
    if not calls_directory.exists():
        return []
    if calls_directory.is_symlink() or not calls_directory.is_dir():
        raise RuntimeError(f"Unsafe local call receipt directory: {calls_directory}")
    receipts: list[dict[str, Any]] = []
    observed_intents: set[str] = set()
    for path in sorted(calls_directory.glob("*.json")):
        receipt = _read_object(path)
        intent_path = _launch_intent_file(run_root, receipt.get("launch_intent"))
        intent_sha256 = receipt.get("launch_intent_sha256")
        if (
            receipt.get("schema_version") != CALL_SCHEMA
            or receipt.get("run_id") != run_id
            or receipt.get("config_hash") != config.config_hash()
            or receipt.get("control_plane_sha256") != control_plane.tree_sha256
            or (require_adoption_binding and not _has_source_adoption_binding(receipt, reference))
            or receipt.get("stage") not in STAGE_ORDER
            or not isinstance(intent_sha256, str)
            or len(intent_sha256) != 64
            or any(character not in "0123456789abcdef" for character in intent_sha256)
            or intent_sha256 in observed_intents
        ):
            raise RuntimeError(f"Local call receipt drifted: {path}")
        if _local_sha256(intent_path) != intent_sha256:
            raise RuntimeError(f"Local launch intent drifted: {intent_path}")
        intent = _read_object(intent_path)
        if (
            intent.get("schema_version") != LAUNCH_INTENT_SCHEMA
            or intent.get("run_id") != run_id
            or intent.get("stage") != receipt["stage"]
            or intent.get("function_id") != receipt.get("function_id")
            or intent.get("function_name") != receipt.get("function_name")
            or intent.get("function_name") != intent.get("stage")
            or not isinstance(intent.get("function_id"), str)
            or not str(intent["function_id"]).startswith("fu-")
            or intent.get("config_hash") != config.config_hash()
            or intent.get("control_plane_sha256") != control_plane.tree_sha256
            or (require_adoption_binding and not _has_source_adoption_binding(intent, reference))
            or intent.get("billing_cycle_end_utc") != receipt.get("billing_cycle_end_utc")
            or intent.get("initial_billing_window_policy")
            != receipt.get("initial_billing_window_policy")
            or intent.get("billing_cycle_end_source") != receipt.get("billing_cycle_end_source")
            or not _valid_initial_billing_evidence(intent)
        ):
            raise RuntimeError(f"Local launch intent is not bound to its call: {intent_path}")
        observed_intents.add(intent_sha256)
        receipts.append(receipt)
    return receipts


def _require_no_unresolved_launch_intent(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
) -> None:
    """Fail closed after a crash or ambiguous spawn that left no call receipt."""

    reference = _load_source_adoption_for_target(config, control_plane)
    require_adoption_binding = _is_checked_target_run(config, control_plane, run_id)
    run_root = _run_root(run_id)
    resolved = {
        str(receipt["launch_intent_sha256"])
        for receipt in _validated_local_call_receipts(config, control_plane, run_id)
    }
    intents_directory = run_root / "launch-intents"
    if not intents_directory.exists():
        return
    if intents_directory.is_symlink() or not intents_directory.is_dir():
        raise RuntimeError(f"Unsafe launch-intent directory: {intents_directory}")
    for path in sorted(intents_directory.glob("*.json")):
        intent = _read_object(path)
        if (
            intent.get("schema_version") != LAUNCH_INTENT_SCHEMA
            or intent.get("run_id") != run_id
            or intent.get("stage") not in STAGE_ORDER
            or intent.get("config_hash") != config.config_hash()
            or intent.get("control_plane_sha256") != control_plane.tree_sha256
            or (require_adoption_binding and not _has_source_adoption_binding(intent, reference))
            or not _valid_initial_billing_evidence(intent)
        ):
            raise RuntimeError(f"Local launch intent drifted: {path}")
        intent_sha256 = _local_sha256(path)
        if intent_sha256 not in resolved:
            raise RuntimeError(
                f"Unresolved pre-spawn intent {path}; inspect Modal before any retry"
            )


def _create_launch_intent(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
    stage: str,
    deployment: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> tuple[str, str]:
    reference = _load_source_adoption_for_target(config, control_plane)
    created_at = datetime.now(UTC).isoformat()
    filename = (
        f"{created_at.replace(':', '').replace('+', '')}-{stage}-{secrets.token_hex(16)}.json"
    )
    relative = (Path("launch-intents") / filename).as_posix()
    path = _run_root(run_id) / relative
    _write_immutable_json(
        path,
        {
            "schema_version": LAUNCH_INTENT_SCHEMA,
            "status": "prepared_before_spawn",
            "created_at": created_at,
            "run_id": run_id,
            "stage": stage,
            "app_name": deployment["app_name"],
            "environment_name": config.modal.environment_name,
            "deployment_version": deployment["version"],
            "deployment_tag": deployment["tag"],
            "function_id": binding["function_id"],
            "function_name": binding["function_name"],
            "config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "workspace_budget_usd": str(config.budget.workspace_hard_budget_usd),
            "billing_cycle_end_utc": deployment["billing_cycle_end_utc"],
            "initial_billing_window_policy": deployment["initial_billing_window_policy"],
            "billing_cycle_end_source": deployment["billing_cycle_end_source"],
            **_source_adoption_binding(reference),
        },
    )
    return relative, _local_sha256(path)


def _intent_for_reconciliation(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
    launch_intent_sha256: str,
) -> tuple[str, dict[str, Any]]:
    reference = _load_source_adoption_for_target(config, control_plane)
    require_adoption_binding = _is_checked_target_run(config, control_plane, run_id)
    directory = _run_root(run_id) / "launch-intents"
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"Safe launch-intent directory is missing: {directory}")
    matches: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        if _local_sha256(path) != launch_intent_sha256:
            continue
        intent = _read_object(path)
        if (
            intent.get("schema_version") != LAUNCH_INTENT_SCHEMA
            or intent.get("run_id") != run_id
            or intent.get("stage") not in STAGE_ORDER
            or intent.get("function_name") != intent.get("stage")
            or not isinstance(intent.get("function_id"), str)
            or not str(intent["function_id"]).startswith("fu-")
            or intent.get("config_hash") != config.config_hash()
            or intent.get("control_plane_sha256") != control_plane.tree_sha256
            or (require_adoption_binding and not _has_source_adoption_binding(intent, reference))
            or not _valid_initial_billing_evidence(intent)
        ):
            raise RuntimeError(f"Launch intent drifted: {path}")
        matches.append((path.relative_to(_run_root(run_id)).as_posix(), intent))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one matching unresolved launch intent")
    resolved = {
        str(receipt["launch_intent_sha256"])
        for receipt in _validated_local_call_receipts(config, control_plane, run_id)
    }
    if launch_intent_sha256 in resolved:
        raise RuntimeError("Launch intent is already reconciled")
    return matches[0]


def _remote_intent_evidence(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
    stage: str,
    launch_intent_sha256: str,
) -> list[tuple[str, dict[str, Any]]]:
    """Read only immutable remote evidence that can bind an ambiguous call."""

    volume_attribute, success_filename = STAGE_RECEIPTS[stage]
    remote_paths = (f"runs/{run_id}/{success_filename}",)
    evidence: list[tuple[str, dict[str, Any]]] = []
    for remote_path in remote_paths:
        value = _read_volume_json_file(
            config,
            volume_attribute=volume_attribute,
            remote_path=remote_path,
            label=f"ambiguous launch evidence at {remote_path}",
        )
        if value is not None and value.get("launch_intent_sha256") == launch_intent_sha256:
            _validate_volume_receipt(
                value,
                config=config,
                control_plane=control_plane,
                run_id=run_id,
                stage=stage,
            )
            evidence.append((remote_path, value))
    resources = next(item for item in config.modal.stages if item.name == stage)
    for kind, limit in (
        ("attempt", resources.max_attempts),
        ("recovery", resources.max_recovery_attempts),
    ):
        for path, event, _sha256 in _authoritative_invocation_history(
            config,
            control_plane,
            volume_attribute=volume_attribute,
            run_id=run_id,
            stage=stage,
            kind=kind,
            limit=limit,
        ):
            if event.get("launch_intent_sha256") == launch_intent_sha256:
                evidence.append((path, event))
    return evidence


def reconcile_launch_intent(
    launch_intent_sha256: str,
    *,
    call_id: str,
) -> None:
    """Adopt one intent only after immutable remote evidence proves the exact call ID."""
    config, control_plane, run_id = _load_checked_context()
    reference = _load_checked_source_adoption(config, control_plane, run_id)
    with _exclusive_manager_operation(run_id, "reconcile-intent"):
        relative, intent = _intent_for_reconciliation(
            config,
            control_plane,
            run_id,
            launch_intent_sha256,
        )
        stage = str(intent["stage"])
        evidence = _remote_intent_evidence(
            config,
            control_plane,
            run_id,
            stage,
            launch_intent_sha256,
        )
        if not call_id.startswith("fc-"):
            raise RuntimeError("Reconciled Modal call ID must start with fc-")
        if not evidence:
            raise RuntimeError(
                "No immutable remote intent-bound history/receipt exists; "
                "reconciliation must remain fail-closed"
            )
        evidence_call_ids = {
            candidate
            for _, record in evidence
            for candidate in (record.get("call_id"), record.get("last_call_id"))
            if isinstance(candidate, str)
        }
        if call_id not in evidence_call_ids:
            raise RuntimeError("Call ID does not match the immutable remote history/receipt")
        graph = modal.FunctionCall.from_id(call_id).get_call_graph()
        if not isinstance(graph, list):
            raise RuntimeError("Modal returned an invalid call graph during reconciliation")
        receipt = {
            "schema_version": CALL_SCHEMA,
            "status": "reconciled_after_ambiguous_spawn",
            "run_id": run_id,
            "stage": stage,
            "call_id": call_id,
            "accepted_at": intent["created_at"],
            "reconciled_at": datetime.now(UTC).isoformat(),
            "app_name": intent["app_name"],
            "environment_name": intent["environment_name"],
            "deployment_version": intent["deployment_version"],
            "deployment_tag": intent["deployment_tag"],
            "launch_intent": relative,
            "launch_intent_sha256": launch_intent_sha256,
            "function_id": intent["function_id"],
            "function_name": intent["function_name"],
            "config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "billing_cycle_end_utc": intent["billing_cycle_end_utc"],
            "initial_billing_window_policy": intent["initial_billing_window_policy"],
            "billing_cycle_end_source": intent["billing_cycle_end_source"],
            "remote_evidence_paths": [path for path, _ in evidence],
            **_source_adoption_binding(reference),
        }
        path = _run_root(run_id) / "calls" / f"reconciled-{launch_intent_sha256}.json"
        _write_immutable_json(path, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))


def _require_launchable_stage(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
    stage: str,
) -> int:
    """Reject a completed or out-of-order stage before paid container startup."""

    existing = _read_volume_receipt(config, run_id=run_id, stage=stage)
    if existing is not None:
        _validate_volume_receipt(
            existing,
            config=config,
            control_plane=control_plane,
            run_id=run_id,
            stage=stage,
        )
        raise RuntimeError(f"Stage {stage} is already successful; refusing another paid launch")
    remaining_invocations = _require_stage_invocation_budget(
        config,
        control_plane,
        run_id,
        stage,
    )
    stage_index = STAGE_ORDER.index(stage)
    if stage_index == 0:
        return remaining_invocations
    source_receipt: dict[str, Any] | None = None
    if _is_checked_target_run(config, control_plane, run_id):
        source_receipt = _read_volume_receipt(
            config,
            run_id=run_id,
            stage="materialize_source",
        )
        if source_receipt is None:
            raise RuntimeError(
                f"Stage {stage} requires the successful adopted source marker; refusing paid launch"
            )
        _validate_volume_receipt(
            source_receipt,
            config=config,
            control_plane=control_plane,
            run_id=run_id,
            stage="materialize_source",
        )
    predecessor = STAGE_ORDER[stage_index - 1]
    predecessor_receipt = (
        source_receipt
        if predecessor == "materialize_source" and source_receipt is not None
        else _read_volume_receipt(config, run_id=run_id, stage=predecessor)
    )
    if predecessor_receipt is None:
        raise RuntimeError(
            f"Stage {stage} requires successful predecessor {predecessor}; refusing paid launch"
        )
    _validate_volume_receipt(
        predecessor_receipt,
        config=config,
        control_plane=control_plane,
        run_id=run_id,
        stage=predecessor,
    )
    predecessor_call_id = predecessor_receipt.get("call_id")
    if not isinstance(predecessor_call_id, str) or not predecessor_call_id.startswith("fc-"):
        raise RuntimeError(f"Successful predecessor {predecessor} has no valid call ID")
    predecessor_graph = modal.FunctionCall.from_id(predecessor_call_id).get_call_graph()
    if not isinstance(predecessor_graph, list):
        raise RuntimeError(f"Modal returned an invalid call graph for predecessor {predecessor}")
    if _call_graph_has_pending(predecessor_graph):
        raise RuntimeError(
            f"Successful predecessor {predecessor} still has pending call "
            f"{predecessor_call_id}; refusing concurrent stage {stage}"
        )
    return remaining_invocations


def _call_graph_has_pending(nodes: Sequence[InputInfo]) -> bool:
    return any(
        node.status == InputStatus.PENDING or _call_graph_has_pending(node.children)
        for node in nodes
    )


def _require_no_active_stage_call(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
    stage: str,
) -> None:
    """Reject a duplicate when a locally recorded call or continuation is pending."""

    for receipt in _validated_local_call_receipts(config, control_plane, run_id):
        if receipt["stage"] != stage:
            continue
        call_id = receipt.get("call_id")
        if not isinstance(call_id, str) or not call_id.startswith("fc-"):
            raise RuntimeError("Local call receipt has an invalid call ID")
        graph = modal.FunctionCall.from_id(call_id).get_call_graph()
        if _call_graph_has_pending(graph):
            raise RuntimeError(
                f"Stage {stage} already has pending call {call_id}; refusing a duplicate"
            )


def _modal_history(
    config: InklingGGUFConfig,
    *,
    app_name: str,
) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modal",
            "app",
            "history",
            app_name,
            "-e",
            config.modal.environment_name,
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
        shell=False,
    )
    value = json.loads(result.stdout)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise RuntimeError("Modal App history returned an unexpected JSON shape")
    return value


def _modal_history_or_empty(
    config: InklingGGUFConfig,
    *,
    app_name: str,
) -> list[dict[str, Any]]:
    try:
        return _modal_history(config, app_name=app_name)
    except subprocess.CalledProcessError as error:
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        if "not found" in stderr.lower() or "no app" in stderr.lower():
            return []
        raise


def _deployment_row(
    history: Sequence[Mapping[str, Any]],
    *,
    tag: str,
) -> tuple[int, Mapping[str, Any]]:
    matches = [row for row in history if row.get("tag") == tag]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one Modal deployment tagged {tag!r}")
    version_text = matches[0].get("version")
    if not isinstance(version_text, str) or not version_text.startswith("v"):
        raise RuntimeError("Modal deployment history has an invalid version")
    try:
        version = int(version_text.removeprefix("v"))
    except ValueError as error:
        raise RuntimeError("Modal deployment history has an invalid version") from error
    if version < 1:
        raise RuntimeError("Modal deployment version must be positive")
    return version, matches[0]


def _require_newest_deployment(
    history: Sequence[Mapping[str, Any]],
    *,
    version: int,
) -> None:
    if not history or history[0].get("version") != f"v{version}":
        raise RuntimeError("Implementation-addressed Modal App has a newer deployment")


def _function_binding(function: Any, *, stage: str) -> dict[str, str]:
    object_id = function.object_id
    metadata = function._get_metadata()
    function_name = getattr(metadata, "function_name", None)
    if not isinstance(object_id, str) or not object_id.startswith("fu-"):
        raise RuntimeError(f"Modal returned an invalid Function ID for {stage}")
    if function_name != stage:
        raise RuntimeError(f"Modal returned the wrong Function name for {stage}")
    return {"function_id": object_id, "function_name": function_name}


def _deployed_function_bindings(
    config: InklingGGUFConfig,
    *,
    app_name: str,
) -> dict[str, dict[str, str]]:
    """Hydrate every deployed stage and retain its concrete ID/name binding."""

    bindings: dict[str, dict[str, str]] = {}
    for stage in STAGE_ORDER:
        function = modal.Function.from_name(
            app_name,
            stage,
            environment_name=config.modal.environment_name,
        )
        function.hydrate()
        bindings[stage] = _function_binding(function, stage=stage)
    function_ids = {binding["function_id"] for binding in bindings.values()}
    if len(function_ids) != len(STAGE_ORDER):
        raise RuntimeError("Modal returned duplicate Function IDs for deployed stages")
    return bindings


def _audit_plan(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
) -> dict[str, Any]:
    reference = _load_source_adoption_for_target(config, control_plane)
    audit = audit_pinned_inkling_online(config, token=os.environ.get("HF_TOKEN"))
    return {
        "status": "exact_source_preflight_passed",
        "run_id": run_id,
        "control_plane_sha256": control_plane.tree_sha256,
        "control_plane_file_count": control_plane.file_count,
        "model_id": audit.model_id,
        "revision": audit.revision,
        "source_bytes": audit.source_bytes,
        "source_tensors": audit.source_tensor_count,
        "source_shards": audit.source_shard_count,
        "quant_type": config.quantization.quant_type,
        "output_label": config.quantization.output_label,
        "llama_cpp_commit": config.toolchain.commit,
        "configured_startup_body_window_usd": str(compute_cost_ceiling_usd(config)),
        "planned_compute_usd": str(config.budget.planned_compute_usd),
        "planned_storage_usd": str(config.budget.planned_storage_usd),
        "workspace_budget_required_usd": str(config.budget.workspace_hard_budget_usd),
        "max_total_envelope_usd": str(config.budget.max_total_usd),
        "warnings": list(audit.warnings),
        **_source_adoption_binding(reference),
    }


def deploy(
    workspace_budget_usd: Decimal,
    *,
    accept_short_initial_window_risk: bool = False,
) -> None:
    """Deploy once, then seal the history row and Function ID/name bindings."""

    config, control_plane, run_id = _load_checked_context()
    with _exclusive_manager_operation(run_id, "deploy"):
        _deploy_locked(
            config,
            control_plane,
            run_id,
            workspace_budget_usd,
            accept_short_initial_window_risk=accept_short_initial_window_risk,
        )


def reconcile_manager_lock(
    lock_sha256: str,
    *,
    confirm_owner_process_stopped: bool,
) -> None:
    """Reconcile one exact stale local lock without starting Modal compute."""

    _config, _control_plane, run_id = _load_checked_context()
    _reconcile_stale_manager_lock(
        run_id,
        expected_sha256=lock_sha256,
        confirm_owner_process_stopped=confirm_owner_process_stopped,
    )
    print(
        json.dumps(
            {
                "status": "stale_manager_lock_reconciled",
                "run_id": run_id,
                "manager_lock_sha256": lock_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


def _deploy_locked(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
    workspace_budget_usd: Decimal,
    *,
    accept_short_initial_window_risk: bool = False,
) -> None:
    billing_cycle_end_utc = _require_paid_gate(config, workspace_budget_usd)
    initial_window_policy, billing_cycle_end_source = _initial_billing_window_evidence(
        config,
        billing_cycle_end_utc,
        accept_short_initial_window_risk=accept_short_initial_window_risk,
    )
    reference = _load_checked_source_adoption(config, control_plane, run_id)
    source_adoption = _require_source_adoption_origin_ready(config, reference)
    plan = _audit_plan(config, control_plane, run_id)
    print(json.dumps(plan, indent=2, sort_keys=True))
    receipt_path = _deployment_path(run_id)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise RuntimeError(
            f"Deployment is already sealed at {receipt_path}; do not redeploy during this run"
        )
    tag = _deployment_tag(control_plane)
    app_name = _deployment_name(config, control_plane)
    history = _modal_history_or_empty(config, app_name=app_name)
    existing = [row for row in history if row.get("tag") == tag]
    if len(existing) > 1:
        raise RuntimeError(f"Multiple existing deployments use implementation tag {tag!r}")
    if not existing:
        source_adoption = _require_source_adoption_origin_ready(config, reference)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "modal",
                "deploy",
                "-e",
                config.modal.environment_name,
                "--name",
                app_name,
                "--tag",
                tag,
                str(PAID_APP_PATH),
            ],
            check=True,
            shell=False,
        )
        history = _modal_history(config, app_name=app_name)
    version, row = _deployment_row(history, tag=tag)
    _require_newest_deployment(history, version=version)
    stage_function_bindings = _deployed_function_bindings(config, app_name=app_name)
    # Bracketing hydration with history reads detects ordinary concurrent
    # redeployment. Modal Starter cannot atomically invoke a numeric App version,
    # so operators must also keep this implementation-addressed name exclusive.
    history_after_hydration = _modal_history(config, app_name=app_name)
    observed_version, observed_row = _deployment_row(history_after_hydration, tag=tag)
    _require_newest_deployment(history_after_hydration, version=observed_version)
    if observed_version != version or observed_row != row:
        raise RuntimeError("Modal App changed while its Function bindings were being sealed")
    receipt = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "app_name": app_name,
        "source_app_name": config.modal.app_name,
        "environment_name": config.modal.environment_name,
        "version": version,
        "tag": tag,
        "time_deployed": row.get("time_deployed"),
        "modal_client": row.get("client"),
        "stage_function_bindings": stage_function_bindings,
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "run_id": run_id,
        "billing_cycle_end_utc": billing_cycle_end_utc,
        "initial_billing_window_policy": initial_window_policy,
        "billing_cycle_end_source": billing_cycle_end_source,
        **source_adoption,
    }
    _write_immutable_json(receipt_path, receipt)
    print(json.dumps({"status": "deployed_and_sealed", **receipt}, indent=2, sort_keys=True))


def _validated_deployment(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
) -> dict[str, Any]:
    reference = _load_checked_source_adoption(config, control_plane, run_id)
    receipt = _read_object(_deployment_path(run_id))
    expected = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "app_name": _deployment_name(config, control_plane),
        "source_app_name": config.modal.app_name,
        "environment_name": config.modal.environment_name,
        "tag": _deployment_tag(control_plane),
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "run_id": run_id,
        **_source_adoption_binding(reference),
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    version = receipt.get("version")
    if not isinstance(version, int) or version < 1:
        mismatches.append("version")
    stage_function_bindings = receipt.get("stage_function_bindings")
    if (
        not isinstance(stage_function_bindings, dict)
        or set(stage_function_bindings) != set(STAGE_ORDER)
        or any(
            not isinstance(binding, dict)
            or set(binding) != {"function_id", "function_name"}
            or not isinstance(binding.get("function_id"), str)
            or not str(binding["function_id"]).startswith("fu-")
            or binding.get("function_name") != stage
            for stage, binding in stage_function_bindings.items()
        )
        or len(
            {
                binding["function_id"]
                for binding in stage_function_bindings.values()
                if isinstance(binding, dict) and "function_id" in binding
            }
        )
        != len(STAGE_ORDER)
    ):
        mismatches.append("stage_function_bindings")
    if mismatches:
        raise RuntimeError("Deployment receipt drifted: " + ", ".join(sorted(set(mismatches))))
    if not _valid_initial_billing_evidence(receipt):
        raise RuntimeError("Deployment receipt has invalid initial billing-window evidence")
    history = _modal_history(config, app_name=str(receipt["app_name"]))
    history_version, _ = _deployment_row(history, tag=str(receipt["tag"]))
    if history_version != version:
        raise RuntimeError("Recorded Modal App version no longer matches its unique deployment tag")
    _require_newest_deployment(history, version=version)
    return receipt


def _revalidate_deployment_history(
    config: InklingGGUFConfig,
    deployment: Mapping[str, Any],
) -> None:
    history = _modal_history(config, app_name=str(deployment["app_name"]))
    history_version, _ = _deployment_row(history, tag=str(deployment["tag"]))
    if history_version != deployment["version"]:
        raise RuntimeError("Modal deployment tag moved after the local seal")
    _require_newest_deployment(history, version=history_version)


def launch(stage: str, workspace_budget_usd: Decimal) -> None:
    """Check one sealed Function ID/name binding, spawn it, and retain its call ID."""

    if stage not in STAGE_ORDER:
        raise RuntimeError(f"Unknown stage {stage!r}; choose one of {STAGE_ORDER}")
    config, control_plane, run_id = _load_checked_context()
    with _exclusive_manager_operation(run_id, f"launch:{stage}"):
        _launch_locked(config, control_plane, run_id, stage, workspace_budget_usd)


def _launch_locked(
    config: InklingGGUFConfig,
    control_plane: ControlPlaneProvenance,
    run_id: str,
    stage: str,
    workspace_budget_usd: Decimal,
) -> None:
    billing_cycle_end_utc = _require_paid_gate(config, workspace_budget_usd)
    reference = _load_checked_source_adoption(config, control_plane, run_id)
    checked_target_run = _is_checked_target_run(config, control_plane, run_id)
    if stage == "materialize_source" and checked_target_run:
        _require_source_adoption_origin_ready(config, reference)
    _require_no_unresolved_launch_intent(config, control_plane, run_id)
    stage_index = STAGE_ORDER.index(stage)
    predecessor_and_current = STAGE_ORDER[max(0, stage_index - 1) : stage_index + 1]
    for guarded_stage in predecessor_and_current:
        _require_no_active_stage_call(config, control_plane, run_id, guarded_stage)
    remaining_invocations = _require_launchable_stage(config, control_plane, run_id, stage)
    require_stage_billing_window(
        config,
        billing_cycle_end_utc,
        stage,
        include_startup=True,
        invocations=remaining_invocations if stage == "materialize_source" else 1,
    )
    plan = _audit_plan(config, control_plane, run_id)
    deployment = _validated_deployment(config, control_plane, run_id)
    if deployment.get("billing_cycle_end_utc") != billing_cycle_end_utc:
        raise RuntimeError(
            "The sealed deployment belongs to a different monthly billing-cycle window"
        )
    function = modal.Function.from_name(
        str(deployment["app_name"]),
        stage,
        environment_name=config.modal.environment_name,
    )
    function.hydrate()
    expected_binding = deployment["stage_function_bindings"][stage]
    actual_binding = _function_binding(function, stage=stage)
    if actual_binding != expected_binding:
        raise RuntimeError("Deployed Function binding changed after sealing; refusing the lookup")
    _revalidate_deployment_history(config, deployment)
    # A predecessor may have committed its durable marker before returning or
    # being crash-rescheduled while the checks above were running. Recheck the
    # immediate predecessor call graph plus the target as the final read-only gate.
    for guarded_stage in predecessor_and_current:
        _require_no_active_stage_call(config, control_plane, run_id, guarded_stage)
    remaining_invocations = _require_launchable_stage(config, control_plane, run_id, stage)
    require_stage_billing_window(
        config,
        billing_cycle_end_utc,
        stage,
        include_startup=True,
        invocations=remaining_invocations if stage == "materialize_source" else 1,
    )
    if stage == "materialize_source" and checked_target_run:
        _require_source_adoption_origin_ready(config, reference)
    launch_intent, launch_intent_sha256 = _create_launch_intent(
        config,
        control_plane,
        run_id,
        stage,
        deployment,
        expected_binding,
    )
    acknowledgement = PaidLaunchAcknowledgement(
        config_hash=config.config_hash(),
        control_plane_sha256=control_plane.tree_sha256,
        launch_intent_sha256=launch_intent_sha256,
        workspace_budget_usd=config.budget.workspace_hard_budget_usd,
        billing_cycle_end_utc=billing_cycle_end_utc,
        initial_billing_window_policy=deployment["initial_billing_window_policy"],
        billing_cycle_end_source=deployment["billing_cycle_end_source"],
    )
    call = function.spawn(
        config.canonical_json(),
        run_id,
        acknowledgement.canonical_json(),
        control_plane.canonical_json(),
    )
    accepted_at = datetime.now(UTC).isoformat()
    post_acceptance_error: Exception | None = None
    cancel_error: Exception | None = None
    try:
        _revalidate_deployment_history(config, deployment)
    except Exception as error:
        post_acceptance_error = error
        try:
            call.cancel(terminate_containers=True)
        except Exception as error_during_cancel:
            cancel_error = error_during_cancel
    cancellation_status = None
    if post_acceptance_error is not None:
        cancellation_status = (
            "cancellation_requested_after_deployment_change"
            if cancel_error is None
            else "cancellation_failed_after_deployment_change"
        )
    call_receipt = {
        "schema_version": CALL_SCHEMA,
        "status": ("accepted" if post_acceptance_error is None else cancellation_status),
        "run_id": run_id,
        "stage": stage,
        "call_id": call.object_id,
        "accepted_at": accepted_at,
        "app_name": deployment["app_name"],
        "environment_name": config.modal.environment_name,
        "deployment_version": deployment["version"],
        "deployment_tag": deployment["tag"],
        "launch_intent": launch_intent,
        "launch_intent_sha256": launch_intent_sha256,
        **expected_binding,
        "config_hash": config.config_hash(),
        "control_plane_sha256": control_plane.tree_sha256,
        "billing_cycle_end_utc": billing_cycle_end_utc,
        "initial_billing_window_policy": deployment["initial_billing_window_policy"],
        "billing_cycle_end_source": deployment["billing_cycle_end_source"],
        **_source_adoption_binding(reference),
    }
    if post_acceptance_error is not None:
        call_receipt["deployment_revalidation_error"] = str(post_acceptance_error)
        call_receipt["cancel_error"] = None if cancel_error is None else str(cancel_error)
    call_path = _run_root(run_id) / "calls" / f"{accepted_at.replace(':', '')}-{stage}.json"
    _write_immutable_json(call_path, call_receipt)
    if post_acceptance_error is not None:
        outcome = "cancellation was requested" if cancel_error is None else "cancellation failed"
        raise RuntimeError(
            f"Deployment changed around call acceptance; retained {call.object_id} and {outcome}"
        ) from post_acceptance_error
    print(json.dumps({**plan, **call_receipt}, indent=2, sort_keys=True))


def status(call_id: str) -> None:
    """Read a Function call without changing its state or treating return as success evidence."""

    call = modal.FunctionCall.from_id(call_id)
    try:
        result = call.get(timeout=0)
    except modal.exception.TimeoutError as error:
        # Modal's terminal FunctionTimeoutError and OutputExpiredError are
        # subclasses of this polling timeout.  Only the exact base exception
        # means the call is still queued or running.
        if type(error) is not modal.exception.TimeoutError:
            raise
        print(json.dumps({"status": "running_or_queued", "call_id": call_id}, indent=2))
        return
    print(
        json.dumps(
            {
                "status": "function_returned",
                "call_id": call_id,
                "result": result,
                "warning": (
                    "A Function return is not completion; verify_export.success.json is the gate."
                ),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("deploy", "launch"):
        child = subparsers.add_parser(command)
        child.add_argument(
            "--workspace-budget-usd",
            type=Decimal,
            required=True,
            help="Must equal the externally activated Modal workspace hard budget ($800).",
        )
        if command == "deploy":
            child.add_argument(
                "--accept-short-initial-window-risk",
                action="store_true",
                help=(
                    "Waive only the full-workflow initial window after separately setting "
                    "IQL_MODAL_SHORT_CYCLE_CONFIRMED to the exact confirmed cycle end."
                ),
            )
        if command == "launch":
            child.add_argument("--stage", required=True, choices=STAGE_ORDER)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--call-id", required=True)
    lock_parser = subparsers.add_parser("reconcile-lock")
    lock_parser.add_argument("--lock-sha256", required=True)
    lock_parser.add_argument("--confirm-owner-process-stopped", action="store_true")
    intent_parser = subparsers.add_parser("reconcile-intent")
    intent_parser.add_argument("--launch-intent-sha256", required=True)
    intent_parser.add_argument("--call-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "deploy":
        deploy(
            args.workspace_budget_usd,
            accept_short_initial_window_risk=args.accept_short_initial_window_risk,
        )
    elif args.command == "launch":
        launch(args.stage, args.workspace_budget_usd)
    elif args.command == "status":
        status(args.call_id)
    elif args.command == "reconcile-lock":
        reconcile_manager_lock(
            args.lock_sha256,
            confirm_owner_process_stopped=args.confirm_owner_process_stopped,
        )
    elif args.command == "reconcile-intent":
        reconcile_launch_intent(
            args.launch_intent_sha256,
            call_id=args.call_id,
        )
    else:  # pragma: no cover - argparse enforces this branch away.
        raise RuntimeError(f"Unsupported command {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
