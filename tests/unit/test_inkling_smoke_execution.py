from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.gguf.inkling_smoke import (
    HISTORICAL_INSTRUMENTATION_PATCH_SHA256,
    LEGACY_CURRENT_INSTRUMENTATION_PATCH_SHA256,
    MAX_BACKEND_FAILURE_RECORDS,
    load_inkling_smoke_config,
    load_verified_export_reference,
)
from inkling_quant_lab.gguf.inkling_smoke_execution import (
    SMOKE_CONTROL_PLANE_REQUIRED_FILES,
    SMOKE_WORKSPACE_BUDGET_USD,
    SmokeControlPlaneProvenance,
    SmokeFailureReceiptV2,
    SmokeFailureReceiptV3,
    SmokeFailureReceiptV4,
    SmokeFailureReceiptV5,
    SmokeFailureReceiptV6,
    SmokeHostEvidence,
    SmokeLaunchAcknowledgement,
    SmokeLaunchDeploymentIdentity,
    SmokeServerLogFailureEvidence,
    canonical_python_package_inventory,
    immutable_source_tree_identity,
    parse_cgroup_cpu_quota_millicores,
    parse_cgroup_memory_limit_bytes,
    parse_dpkg_inventory,
    parse_nvcc_version,
    parse_proc_cpu_model,
    parse_proc_mem_total_bytes,
    resolve_current_process_cgroup_hierarchy_paths,
    resolve_current_process_cgroup_leaf_paths,
    smoke_control_plane_provenance,
    smoke_control_plane_tree_sha256,
    smoke_deployment_tag,
    smoke_hardware_topology_sha256,
    smoke_package_manifest_sha256,
    smoke_run_id,
    smoke_terminal_receipt_sha256,
    validate_deployed_smoke_control_plane,
    validate_smoke_failure_receipt,
    validate_smoke_launch_acknowledgement,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_smoke_modal.yaml"
REFERENCE_PATH = PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_verified_export.json"
HISTORICAL_V2_FAILURE_RECEIPT_JSON = (
    '{"call_id":"fc-01KY5YWSR5PNKPJVVWVSR9D9YJ",'
    '"control_plane_sha256":"0b1041ee22f1b76c9c81a0ec5fd3f045042194b16e7b43326b2e94d8dcf3a572",'
    '"exception_type":"pydantic_core._pydantic_core.ValidationError",'
    '"failure_phase":"verify_runtime_and_allocation",'
    '"input_id":"in-01KY5YWSRQG1AR8BEWRJ2A797D:1784759084850-0",'
    '"launch_intent_sha256":"5bd87c9ec6ae18c47ed4e3d9a525d7a62961327134f28243c29f0ad4af414d0f",'
    '"output_text_recorded":false,"prompt_text_recorded":false,'
    '"receipt_sha256":"9cdffb4a962626d9c60335d682276d9a9e55fed1bbbc36e48b495f6d7399f289",'
    '"run_id":"inkling-smoke-86b4d430-a015409e-9b3214d131-0b1041ee22",'
    '"safe_failure_signals":{"model_load_failure_observed":false,'
    '"no_usable_gpu_observed":false,"out_of_memory_observed":false,'
    '"projector_load_failure_observed":false,'
    '"unsupported_architecture_observed":false},'
    '"schema_version":"inkling-smoke-terminal-v2",'
    '"server_log_sha256":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",'
    '"smoke_config_hash":"9b3214d131ba8e65bc5388cc4119e06731a3f8df422c784b5e1934fb77381ddd",'
    '"stage":"smoke_test","status":"failed",'
    '"subject_run_id":"inkling-q3km-86b4d430-a015409e-ffd466dd93-8083cf41e1",'
    '"task_id":"ta-01KY5YWT70ET3K3312V9JJF8NR",'
    '"verified_export_reference_sha256":'
    '"9f0fae0a48058e73aab38c2b4f6c86916b69fd32343e0f7b821c7faac5b33198"}'
)
EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
BACKEND_CPU_PLACEMENT_EXCEPTION_TYPE = (
    "inkling_quant_lab.gguf.inkling_smoke.BackendCpuPlacementError"
)


def _backend_failure_diagnostic(*, cpu_fallback: bool) -> dict[str, Any]:
    if not cpu_fallback:
        return {
            "schema_version": "inkling-smoke-backend-failure-v1",
            "cpu_model_graph_fallback_observed": False,
            "graph_marker_count": 0,
            "affected_graph_marker_count": 0,
            "cpu_node_marker_count": 0,
            "affected_graphs": [],
            "cpu_node_samples": [],
            "records_truncated": False,
            "raw_marker_lines_recorded": False,
            "raw_node_names_recorded": False,
        }
    return {
        "schema_version": "inkling-smoke-backend-failure-v1",
        "cpu_model_graph_fallback_observed": True,
        "graph_marker_count": 1,
        "affected_graph_marker_count": 1,
        "cpu_node_marker_count": 1,
        "affected_graphs": [
            {
                "graph_uid": 7,
                "phase": "post_assignment_pre_split",
                "scope": "non_view_compute",
                "compute": 2,
                "gpu": 1,
                "cpu": 1,
                "accel": 0,
                "other": 0,
                "unassigned": 0,
            }
        ],
        "cpu_node_samples": [
            {
                "graph_uid": 7,
                "ordinal": 11,
                "op": "MUL_MAT",
                "node_name_size_bytes": 18,
                "node_name_sha256": "a" * 64,
                "node_name_recorded": False,
            }
        ],
        "records_truncated": False,
        "raw_marker_lines_recorded": False,
        "raw_node_names_recorded": False,
    }


def _backend_diagnostic_with_graph_without_cpu_marker() -> dict[str, Any]:
    diagnostic = _backend_failure_diagnostic(cpu_fallback=True)
    diagnostic["cpu_model_graph_fallback_observed"] = False
    diagnostic["cpu_node_marker_count"] = 0
    diagnostic["cpu_node_samples"] = []
    return diagnostic


def _backend_diagnostic_with_partial_graph_coverage() -> dict[str, Any]:
    diagnostic = _backend_failure_diagnostic(cpu_fallback=True)
    second_graph = dict(diagnostic["affected_graphs"][0])
    second_graph["graph_uid"] = 8
    diagnostic["graph_marker_count"] = 2
    diagnostic["affected_graph_marker_count"] = 2
    diagnostic["affected_graphs"].append(second_graph)
    return diagnostic


def _failure_receipt(
    schema_version: str = "inkling-smoke-terminal-v3",
    *,
    called_process_error: bool = False,
    backend_cpu_placement_error: bool = False,
    historical_context: bool | None = None,
) -> tuple[dict[str, Any], Any, Any, Any, str, str]:
    if called_process_error and backend_cpu_placement_error:
        raise ValueError("test receipt cannot represent two exception types")
    config = load_inkling_smoke_config(CONFIG_PATH)
    reference = load_verified_export_reference(REFERENCE_PATH)
    control_plane = smoke_control_plane_provenance(PROJECT_ROOT)
    version_one_patch: tuple[str, int] | None = None
    if historical_context is True or (
        historical_context is None
        and schema_version
        in {
            "inkling-smoke-terminal-v2",
            "inkling-smoke-terminal-v3",
        }
    ):
        version_one_patch = (HISTORICAL_INSTRUMENTATION_PATCH_SHA256, 14_870)
    elif historical_context is None and schema_version == "inkling-smoke-terminal-v4":
        version_one_patch = (LEGACY_CURRENT_INSTRUMENTATION_PATCH_SHA256, 15_179)
    if version_one_patch is not None:
        patch_sha256, patch_size_bytes = version_one_patch
        config_payload = config.model_dump(mode="json")
        config_payload["schema_version"] = "inkling-smoke-config-v1"
        config_payload.pop("output_vocabulary")
        config_payload["runtime"]["instrumentation_schema_version"] = (
            "inkling-llama-smoke-instrumentation-v1"
        )
        config_payload["runtime"]["instrumentation_patch_sha256"] = patch_sha256
        config = type(config).model_validate(config_payload)
        files = tuple(
            item.model_copy(
                update={
                    "sha256": patch_sha256,
                    "size_bytes": patch_size_bytes,
                }
            )
            if item.path == "patches/inkling-smoke-a015409.patch"
            else item
            for item in control_plane.files
        )
        control_plane = SmokeControlPlaneProvenance(
            file_count=len(files),
            files=files,
            tree_sha256=smoke_control_plane_tree_sha256(files),
        )
    run_id = smoke_run_id(config, control_plane.tree_sha256)
    launch_intent_sha256 = "1" * 64
    receipt: dict[str, Any] = {
        "schema_version": schema_version,
        "status": "failed",
        "stage": "smoke_test",
        "run_id": run_id,
        "subject_run_id": reference.subject_run_id,
        "smoke_config_hash": config.config_hash(),
        "verified_export_reference_sha256": reference.reference_sha256,
        "control_plane_sha256": control_plane.tree_sha256,
        "launch_intent_sha256": launch_intent_sha256,
        "call_id": "fc-01KY5YWSR5PNKPJVVWVSR9D9YJ",
        "input_id": "in-01KY5YWT70ET3K3312V9JJF8NR",
        "task_id": "ta-01KY5YWT70ET3K3312V9JJF8NR",
        "failure_phase": "verify_runtime_and_allocation",
        "exception_type": (
            "subprocess.CalledProcessError"
            if called_process_error
            else (
                BACKEND_CPU_PLACEMENT_EXCEPTION_TYPE
                if backend_cpu_placement_error
                else "pydantic_core._pydantic_core.ValidationError"
            )
        ),
        "server_log_sha256": "0" * 64,
        "safe_failure_signals": {
            "out_of_memory_observed": False,
            "no_usable_gpu_observed": False,
            "model_load_failure_observed": False,
            "projector_load_failure_observed": False,
            "unsupported_architecture_observed": False,
        },
        "prompt_text_recorded": False,
        "output_text_recorded": False,
    }
    if schema_version in {
        "inkling-smoke-terminal-v3",
        "inkling-smoke-terminal-v4",
        "inkling-smoke-terminal-v5",
        "inkling-smoke-terminal-v6",
    }:
        receipt["invocation"] = {
            "schema_version": "inkling-smoke-invocation-v3",
            "run_id": run_id,
            "stage": "smoke_test",
            "sequence": 1,
            "limit": 1,
            "call_id": receipt["call_id"],
            "input_id": receipt["input_id"],
            "task_id": receipt["task_id"],
            "launch_intent_sha256": launch_intent_sha256,
            "smoke_config_hash": config.config_hash(),
            "control_plane_sha256": control_plane.tree_sha256,
            "post_spawn_acceptance_path": (
                "control/post-spawn-acceptances/" + launch_intent_sha256 + ".json"
            ),
            "post_spawn_acceptance_sha256": "5" * 64,
            "attempt_registry_name": "inkling-smoke-attempt-registry-v1",
            "attempt_registry_id": "di-Attempt123",
            "attempt_registry_created_at_utc": ("2026-07-22T12:00:00.000000Z"),
            "attempt_registry_key": f"{run_id}:smoke_test",
            "attempt_registry_claim_sha256": "6" * 64,
            "attempt_claim_path": "control/smoke_test.attempt.claim.json",
            "attempt_claim_sha256": "6" * 64,
            "invocation_history_path": (
                "control/history/smoke_test.attempt.1." + "7" * 64 + ".json"
            ),
            "invocation_history_sha256": "8" * 64,
        }
    if schema_version in {
        "inkling-smoke-terminal-v4",
        "inkling-smoke-terminal-v5",
        "inkling-smoke-terminal-v6",
    }:
        receipt["safe_subprocess_failure"] = (
            {
                "schema_version": "inkling-smoke-subprocess-failure-v1",
                "command_id": "nvidia_smi_identity_v1",
                "return_code": 255,
                "stdout_size_bytes": 0,
                "stdout_sha256": (
                    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                ),
                "stderr_size_bytes": 32,
                "stderr_sha256": "9" * 64,
                "stdout_recorded": False,
                "stderr_recorded": False,
            }
            if called_process_error
            else None
        )
    if schema_version == "inkling-smoke-terminal-v6":
        receipt["server_log_evidence"] = {
            "schema_version": "inkling-smoke-server-log-failure-v1",
            "present": True,
            "size_bytes": 123,
            "sha256": receipt["server_log_sha256"],
            "raw_log_recorded": False,
            "scan_integrity": "complete",
            "safe_failure_signals": copy.deepcopy(receipt["safe_failure_signals"]),
            "backend_diagnostic": _backend_failure_diagnostic(
                cpu_fallback=backend_cpu_placement_error
            ),
        }
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)
    return (
        receipt,
        config,
        reference,
        control_plane,
        run_id,
        launch_intent_sha256,
    )


def _temporary_control_tree(root: Path) -> None:
    for relative in SMOKE_CONTROL_PLANE_REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"bound bytes for {relative}\n", encoding="utf-8")
    package = root / "src/inkling_quant_lab"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "module.py").write_text("def value() -> int:\n    return 1\n", encoding="utf-8")


def _deployed_paths(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    package = root / "deployed/inkling_quant_lab"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "module.py").write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
    script = root / "deployed/smoke_inkling_modal.py"
    config = root / "deployed/smoke.yaml"
    reference = root / "deployed/reference.json"
    patch = root / "deployed/inkling-smoke-a015409.patch"
    script.write_text("bound bytes for scripts/smoke_inkling_modal.py\n", encoding="utf-8")
    config.write_text(
        "bound bytes for configs/experiments/inkling_q3_k_m_smoke_modal.yaml\n",
        encoding="utf-8",
    )
    reference.write_text(
        "bound bytes for configs/experiments/inkling_q3_k_m_verified_export.json\n",
        encoding="utf-8",
    )
    patch.write_text(
        "bound bytes for patches/inkling-smoke-a015409.patch\n",
        encoding="utf-8",
    )
    return script, package, config, reference, patch


def test_smoke_control_plane_is_deterministic_and_covers_runnable_files(
    tmp_path: Path,
) -> None:
    _temporary_control_tree(tmp_path)

    first = smoke_control_plane_provenance(tmp_path)
    second = smoke_control_plane_provenance(tmp_path)

    assert first == second
    assert "configs/experiments/inkling_q3_k_m_modal.yaml" in (SMOKE_CONTROL_PLANE_REQUIRED_FILES)
    assert first.file_count == len(first.files)
    assert tuple(item.path for item in first.files) == tuple(
        sorted(item.path for item in first.files)
    )
    assert {item.path for item in first.files}.issuperset(
        {
            *SMOKE_CONTROL_PLANE_REQUIRED_FILES,
            "src/inkling_quant_lab/__init__.py",
            "src/inkling_quant_lab/module.py",
        }
    )

    (tmp_path / "src/inkling_quant_lab/module.py").write_text(
        "def value() -> int:\n    return 2\n", encoding="utf-8"
    )
    assert smoke_control_plane_provenance(tmp_path).tree_sha256 != first.tree_sha256


def test_deployed_control_plane_requires_exact_source_inventory(tmp_path: Path) -> None:
    _temporary_control_tree(tmp_path)
    provenance = smoke_control_plane_provenance(tmp_path)
    script, package, config, reference, patch = _deployed_paths(tmp_path)

    observed = validate_deployed_smoke_control_plane(
        provenance.canonical_json(),
        deployment_script=script,
        deployed_package_root=package,
        deployed_config=config,
        deployed_reference=reference,
        deployed_patch=patch,
    )

    assert observed == provenance

    (package / "extra.py").write_text("UNBOUND = True\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="inventory differs"):
        validate_deployed_smoke_control_plane(
            provenance.canonical_json(),
            deployment_script=script,
            deployed_package_root=package,
            deployed_config=config,
            deployed_reference=reference,
            deployed_patch=patch,
        )


@pytest.mark.parametrize(
    "relative",
    ("__pycache__/module.cpython-312.pyc", "module.pyc", ".DS_Store"),
)
def test_deployed_control_plane_rejects_generated_or_unmanifested_code(
    tmp_path: Path,
    relative: str,
) -> None:
    _temporary_control_tree(tmp_path)
    provenance = smoke_control_plane_provenance(tmp_path)
    script, package, config, reference, patch = _deployed_paths(tmp_path)
    generated = package / relative
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"not executable evidence")

    with pytest.raises(ConfigurationError, match="unmanifested generated code"):
        validate_deployed_smoke_control_plane(
            provenance.canonical_json(),
            deployment_script=script,
            deployed_package_root=package,
            deployed_config=config,
            deployed_reference=reference,
            deployed_patch=patch,
        )


def test_smoke_deployment_tag_is_modal_safe_and_deterministic() -> None:
    control_plane_sha256 = "0123456789abcdef" * 4

    observed = smoke_deployment_tag(control_plane_sha256)

    assert observed == "iql-smoke-0123456789abcdef0123456789abcdef01234567"
    assert len(observed) == 50


@pytest.mark.parametrize(
    "control_plane_sha256",
    (
        "",
        "0" * 63,
        "0" * 65,
        "A" * 64,
        "g" * 64,
    ),
)
def test_smoke_deployment_tag_rejects_invalid_control_hash(
    control_plane_sha256: str,
) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        smoke_deployment_tag(control_plane_sha256)


def test_launch_acknowledgement_binds_exact_run_and_external_gate(tmp_path: Path) -> None:
    _temporary_control_tree(tmp_path)
    control_plane = smoke_control_plane_provenance(tmp_path)
    config = load_inkling_smoke_config(CONFIG_PATH)
    run_id = smoke_run_id(config, control_plane.tree_sha256)
    launch_intent_sha256 = "1" * 64
    acknowledgement = SmokeLaunchAcknowledgement(
        smoke_config_hash=config.config_hash(),
        verified_export_reference_sha256=config.verified_export_reference_sha256,
        control_plane_sha256=control_plane.tree_sha256,
        launch_intent_sha256=launch_intent_sha256,
        run_id=run_id,
        deployment=SmokeLaunchDeploymentIdentity(
            app_name=f"inkling-q3-smoke-{control_plane.tree_sha256[:12]}",
            deployment_version=7,
            deployment_tag=smoke_deployment_tag(control_plane.tree_sha256),
            function_id="fu-Abc123",
            attempt_registry_name="inkling-smoke-attempt-registry-v1",
            attempt_registry_id="di-Attempt123",
            attempt_registry_created_at_utc="2026-07-22T12:00:00.000000Z",
        ),
        workspace_budget_usd=SMOKE_WORKSPACE_BUDGET_USD,
        billing_cycle_end_utc="2026-08-01T00:00:00Z",
    )

    assert (
        validate_smoke_launch_acknowledgement(
            acknowledgement.canonical_json(),
            config=config,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )
        == acknowledgement
    )

    tampered = json.loads(acknowledgement.canonical_json())
    tampered["launch_intent_sha256"] = "2" * 64
    with pytest.raises(ConfigurationError, match="does not bind"):
        validate_smoke_launch_acknowledgement(
            json.dumps(tampered),
            config=config,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )

    tampered["workspace_budget_usd"] = "801"
    with pytest.raises(ValidationError, match="wrong workspace cap"):
        SmokeLaunchAcknowledgement.model_validate(tampered)


def test_terminal_receipt_self_hash_excludes_only_the_hash_field() -> None:
    receipt: dict[str, object] = {
        "schema_version": "inkling-smoke-terminal-v3",
        "status": "passed",
        "run_id": "run-one",
        "token_ids": [1, 2, 3],
    }
    digest = smoke_terminal_receipt_sha256(receipt)
    receipt["receipt_sha256"] = digest

    assert digest == "38ade5351814c2cb255bcafb9ed16392aeae7383ed9c1f71d625a94055fc23ce"
    assert smoke_terminal_receipt_sha256(receipt) == digest
    receipt["token_ids"] = [1, 2, 4]
    assert smoke_terminal_receipt_sha256(receipt) != digest


def test_terminal_receipt_versions_use_distinct_stable_hash_domains() -> None:
    payload: dict[str, object] = {
        "schema_version": "inkling-smoke-terminal-v3",
        "status": "passed",
        "run_id": "run-one",
        "token_ids": [1, 2, 3],
    }
    v3_digest = smoke_terminal_receipt_sha256(payload)
    payload["schema_version"] = "inkling-smoke-terminal-v4"
    v4_digest = smoke_terminal_receipt_sha256(payload)
    payload["schema_version"] = "inkling-smoke-terminal-v5"
    v5_digest = smoke_terminal_receipt_sha256(payload)

    assert v3_digest == "38ade5351814c2cb255bcafb9ed16392aeae7383ed9c1f71d625a94055fc23ce"
    assert v4_digest == "8d97e8de00339d6797a9d0bd6fd747944ac0720eb9552757b90a14c074d47b41"
    assert v5_digest == "25f128c08a89f38da78355ffdfe3ec1334217f1cd3e894a43c55b6e8c8e4dbde"
    assert len({v3_digest, v4_digest, v5_digest}) == 3


def test_terminal_failure_receipt_v6_uses_a_distinct_stable_hash_domain() -> None:
    payload: dict[str, object] = {
        "schema_version": "inkling-smoke-terminal-v5",
        "status": "failed",
        "run_id": "run-one",
        "failure_phase": "stop_server",
    }
    v5_digest = smoke_terminal_receipt_sha256(payload)
    payload["schema_version"] = "inkling-smoke-terminal-v6"
    v6_digest = smoke_terminal_receipt_sha256(payload)

    assert v5_digest == "0e1c1f7af7ed81a4960558263d8c7b562a6d2f5bd66c4d17c166e9e46edd4e08"
    assert v6_digest == "de0f0f6163f33a68e5146cdeb575900a22819def80f922ab99deaea823c9e5a5"
    assert v6_digest != v5_digest
    payload["status"] = "passed"
    with pytest.raises(ValueError, match="v6 is valid only for failures"):
        smoke_terminal_receipt_sha256(payload)


def test_terminal_receipt_hash_preserves_legacy_v2_failure_verification() -> None:
    legacy_failure = {
        "schema_version": "inkling-smoke-terminal-v2",
        "status": "failed",
        "run_id": "legacy",
    }

    assert smoke_terminal_receipt_sha256(legacy_failure) == (
        "4bc9611832a6cb386629973cf5befa2525d43184bf5f735dda5d377058676061"
    )


def test_exact_historical_v2_failure_receipt_remains_valid() -> None:
    raw = HISTORICAL_V2_FAILURE_RECEIPT_JSON.encode("utf-8")
    payload = json.loads(raw)

    assert (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        == raw
    )
    assert smoke_terminal_receipt_sha256(payload) == (
        "9cdffb4a962626d9c60335d682276d9a9e55fed1bbbc36e48b495f6d7399f289"
    )
    observed = SmokeFailureReceiptV2.model_validate_json(raw)
    assert observed.receipt_sha256 == payload["receipt_sha256"]
    assert observed.input_id == "in-01KY5YWSRQG1AR8BEWRJ2A797D:1784759084850-0"


@pytest.mark.parametrize(
    ("schema_version", "expected_type"),
    (
        ("inkling-smoke-terminal-v2", SmokeFailureReceiptV2),
        ("inkling-smoke-terminal-v3", SmokeFailureReceiptV3),
        ("inkling-smoke-terminal-v4", SmokeFailureReceiptV4),
        ("inkling-smoke-terminal-v5", SmokeFailureReceiptV5),
        ("inkling-smoke-terminal-v6", SmokeFailureReceiptV6),
    ),
)
def test_failure_receipt_validates_exact_launch_and_outcome_path(
    schema_version: str,
    expected_type: (
        type[SmokeFailureReceiptV2]
        | type[SmokeFailureReceiptV3]
        | type[SmokeFailureReceiptV4]
        | type[SmokeFailureReceiptV5]
        | type[SmokeFailureReceiptV6]
    ),
) -> None:
    receipt, config, reference, control_plane, run_id, launch_intent_sha256 = _failure_receipt(
        schema_version
    )
    outcome_path = (
        f"runs/{run_id}/control/outcomes/smoke_test.failed.{receipt['receipt_sha256']}.json"
    )

    observed = validate_smoke_failure_receipt(
        json.dumps(receipt, sort_keys=True).encode(),
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
        launch_intent_sha256=launch_intent_sha256,
        outcome_path=outcome_path,
    )

    assert isinstance(observed, expected_type)
    assert observed.receipt_sha256 == receipt["receipt_sha256"]
    if isinstance(
        observed,
        (
            SmokeFailureReceiptV3
            | SmokeFailureReceiptV4
            | SmokeFailureReceiptV5
            | SmokeFailureReceiptV6
        ),
    ):
        assert observed.invocation.attempt_claim_sha256 == "6" * 64
        assert observed.invocation.invocation_history_sha256 == "8" * 64


@pytest.mark.parametrize(
    "schema_version",
    ("inkling-smoke-terminal-v2", "inkling-smoke-terminal-v3"),
)
def test_current_failure_receipt_rejects_a_rehashed_schema_downgrade(
    schema_version: str,
) -> None:
    receipt, config, reference, control_plane, run_id, launch_intent_sha256 = _failure_receipt(
        schema_version,
        historical_context=False,
    )

    with pytest.raises(ValueError, match="schema differs from its instrumentation patch"):
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )


def test_historical_failure_receipt_requires_the_historical_control_plane_patch() -> None:
    receipt, config, reference, control_plane, run_id, launch_intent_sha256 = _failure_receipt()
    current_control_plane = smoke_control_plane_provenance(PROJECT_ROOT)
    changed_run_id = smoke_run_id(config, current_control_plane.tree_sha256)
    receipt["run_id"] = changed_run_id
    receipt["control_plane_sha256"] = current_control_plane.tree_sha256
    receipt["invocation"]["run_id"] = changed_run_id
    receipt["invocation"]["control_plane_sha256"] = current_control_plane.tree_sha256
    receipt["invocation"]["attempt_registry_key"] = f"{changed_run_id}:smoke_test"
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="patch differs from the smoke config"):
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=current_control_plane,
            run_id=changed_run_id,
            launch_intent_sha256=launch_intent_sha256,
        )

    assert run_id != changed_run_id
    assert control_plane.tree_sha256 != current_control_plane.tree_sha256


def test_failure_receipt_requires_the_run_derived_from_config_and_control_plane() -> None:
    receipt, config, reference, control_plane, _run_id, launch_intent_sha256 = _failure_receipt(
        "inkling-smoke-terminal-v4"
    )
    changed_run_id = "different-run"
    receipt["run_id"] = changed_run_id
    receipt["invocation"]["run_id"] = changed_run_id
    receipt["invocation"]["attempt_registry_key"] = f"{changed_run_id}:smoke_test"
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="run ID is not derived"):
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=changed_run_id,
            launch_intent_sha256=launch_intent_sha256,
        )


def test_failure_receipt_v3_requires_the_persisted_invocation_binding() -> None:
    receipt, config, reference, control_plane, run_id, launch_intent_sha256 = _failure_receipt()
    receipt.pop("invocation")
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="schema is invalid"):
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )


def test_failure_receipt_v4_accepts_only_hashed_allowlisted_subprocess_evidence() -> None:
    (
        receipt,
        config,
        reference,
        control_plane,
        run_id,
        launch_intent_sha256,
    ) = _failure_receipt(
        "inkling-smoke-terminal-v4",
        called_process_error=True,
    )

    observed = validate_smoke_failure_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
        launch_intent_sha256=launch_intent_sha256,
    )

    assert isinstance(observed, SmokeFailureReceiptV4)
    assert observed.safe_subprocess_failure is not None
    assert observed.safe_subprocess_failure.command_id == "nvidia_smi_identity_v1"
    assert observed.safe_subprocess_failure.return_code == 255
    serialized = observed.model_dump(mode="json")
    subprocess_evidence = serialized["safe_subprocess_failure"]
    assert "stdout" not in subprocess_evidence
    assert "stderr" not in subprocess_evidence
    assert "argv" not in subprocess_evidence
    assert "environment" not in subprocess_evidence


def test_failure_receipt_v5_accepts_only_hashed_allowlisted_subprocess_evidence() -> None:
    (
        receipt,
        config,
        reference,
        control_plane,
        run_id,
        launch_intent_sha256,
    ) = _failure_receipt(
        "inkling-smoke-terminal-v5",
        called_process_error=True,
    )

    observed = validate_smoke_failure_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
        launch_intent_sha256=launch_intent_sha256,
    )

    assert isinstance(observed, SmokeFailureReceiptV5)
    assert observed.safe_subprocess_failure is not None
    assert observed.safe_subprocess_failure.command_id == "nvidia_smi_identity_v1"


def test_failure_receipt_v6_accepts_complete_positive_cpu_placement_evidence() -> None:
    (
        receipt,
        config,
        reference,
        control_plane,
        run_id,
        launch_intent_sha256,
    ) = _failure_receipt(
        "inkling-smoke-terminal-v6",
        backend_cpu_placement_error=True,
    )

    observed = validate_smoke_failure_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
        launch_intent_sha256=launch_intent_sha256,
    )

    assert isinstance(observed, SmokeFailureReceiptV6)
    assert observed.server_log_evidence.present is True
    assert observed.server_log_evidence.scan_integrity == "complete"
    assert observed.server_log_evidence.backend_diagnostic.cpu_model_graph_fallback_observed is True
    assert observed.server_log_evidence.backend_diagnostic.cpu_node_marker_count == 1


def test_failure_receipt_v6_accepts_cpu_evidence_after_many_benign_graphs() -> None:
    (
        receipt,
        config,
        reference,
        control_plane,
        run_id,
        launch_intent_sha256,
    ) = _failure_receipt(
        "inkling-smoke-terminal-v6",
        backend_cpu_placement_error=True,
    )
    receipt["server_log_evidence"]["backend_diagnostic"]["graph_marker_count"] = (
        MAX_BACKEND_FAILURE_RECORDS + 1
    )
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    observed = validate_smoke_failure_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
        launch_intent_sha256=launch_intent_sha256,
    )

    assert isinstance(observed, SmokeFailureReceiptV6)
    diagnostic = observed.server_log_evidence.backend_diagnostic
    assert diagnostic.graph_marker_count == MAX_BACKEND_FAILURE_RECORDS + 1
    assert diagnostic.affected_graph_marker_count == 1
    assert diagnostic.cpu_node_marker_count == 1
    assert diagnostic.records_truncated is False
    assert diagnostic.cpu_model_graph_fallback_observed is True


def test_failure_receipt_v6_matches_subprocess_evidence_to_the_exception_type() -> None:
    (
        receipt,
        config,
        reference,
        control_plane,
        run_id,
        launch_intent_sha256,
    ) = _failure_receipt(
        "inkling-smoke-terminal-v6",
        called_process_error=True,
    )

    observed = validate_smoke_failure_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
        launch_intent_sha256=launch_intent_sha256,
    )
    assert isinstance(observed, SmokeFailureReceiptV6)
    assert observed.safe_subprocess_failure is not None

    receipt["safe_subprocess_failure"] = None
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)
    with pytest.raises(ValueError, match="schema is invalid"):
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )


def test_failure_receipt_v6_rejects_invocation_identity_drift() -> None:
    receipt, config, reference, control_plane, run_id, launch_intent_sha256 = _failure_receipt(
        "inkling-smoke-terminal-v6"
    )
    receipt["invocation"]["task_id"] = "ta-DIFFERENT"
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="schema is invalid") as error:
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )

    assert "failure invocation differs from the top-level receipt identity" in str(
        error.value.__cause__
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["server_log_evidence"]["backend_diagnostic"].update(
            _backend_failure_diagnostic(cpu_fallback=False)
        ),
        lambda value: value["server_log_evidence"].update(
            {
                "present": False,
                "size_bytes": 0,
                "sha256": EMPTY_SHA256,
                "scan_integrity": "missing",
                "backend_diagnostic": _backend_failure_diagnostic(cpu_fallback=False),
            }
        ),
        lambda value: value["server_log_evidence"].update(
            {
                "scan_integrity": "malformed",
                "backend_diagnostic": _backend_failure_diagnostic(cpu_fallback=False),
            }
        ),
    ),
    ids=("negative-diagnostic", "missing-log", "malformed-scan"),
)
def test_failure_receipt_v6_requires_complete_positive_cpu_placement_evidence(
    mutate: Any,
) -> None:
    (
        receipt,
        config,
        reference,
        control_plane,
        run_id,
        launch_intent_sha256,
    ) = _failure_receipt(
        "inkling-smoke-terminal-v6",
        backend_cpu_placement_error=True,
    )
    mutate(receipt)
    receipt["server_log_sha256"] = receipt["server_log_evidence"]["sha256"]
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="schema is invalid") as error:
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )

    assert "CPU placement failure lacks complete positive backend diagnostics" in str(
        error.value.__cause__
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["server_log_evidence"].__setitem__("sha256", "a" * 64),
        lambda value: value["server_log_evidence"]["safe_failure_signals"].__setitem__(
            "out_of_memory_observed",
            True,
        ),
    ),
    ids=("log-hash", "safe-signals"),
)
def test_failure_receipt_v6_requires_nested_server_log_equality(mutate: Any) -> None:
    receipt, config, reference, control_plane, run_id, launch_intent_sha256 = _failure_receipt(
        "inkling-smoke-terminal-v6"
    )
    mutate(receipt)
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="schema is invalid") as error:
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )

    assert "top-level receipt" in str(error.value.__cause__)


def test_server_log_failure_evidence_distinguishes_missing_and_present_empty_logs() -> None:
    common = {
        "schema_version": "inkling-smoke-server-log-failure-v1",
        "size_bytes": 0,
        "sha256": EMPTY_SHA256,
        "raw_log_recorded": False,
        "safe_failure_signals": {
            "out_of_memory_observed": False,
            "no_usable_gpu_observed": False,
            "model_load_failure_observed": False,
            "projector_load_failure_observed": False,
            "unsupported_architecture_observed": False,
        },
        "backend_diagnostic": _backend_failure_diagnostic(cpu_fallback=False),
    }

    missing = SmokeServerLogFailureEvidence.model_validate(
        {**common, "present": False, "scan_integrity": "missing"}
    )
    empty = SmokeServerLogFailureEvidence.model_validate(
        {**common, "present": True, "scan_integrity": "complete"}
    )

    assert missing.present is False
    assert missing.scan_integrity == "missing"
    assert empty.present is True
    assert empty.scan_integrity == "complete"


@pytest.mark.parametrize(
    ("size_bytes", "sha256"),
    (
        (0, "a" * 64),
        (1, EMPTY_SHA256),
    ),
    ids=("empty-size-nonempty-digest", "nonempty-size-empty-digest"),
)
def test_server_log_failure_evidence_requires_exact_empty_log_identity(
    size_bytes: int,
    sha256: str,
) -> None:
    with pytest.raises(ValidationError, match="empty-log identity"):
        SmokeServerLogFailureEvidence.model_validate(
            {
                "schema_version": "inkling-smoke-server-log-failure-v1",
                "present": True,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "raw_log_recorded": False,
                "scan_integrity": "complete",
                "safe_failure_signals": {
                    "out_of_memory_observed": False,
                    "no_usable_gpu_observed": False,
                    "model_load_failure_observed": False,
                    "projector_load_failure_observed": False,
                    "unsupported_architecture_observed": False,
                },
                "backend_diagnostic": _backend_failure_diagnostic(cpu_fallback=False),
            }
        )


def test_server_log_failure_evidence_rejects_complete_truncated_scan() -> None:
    backend_diagnostic = _backend_failure_diagnostic(cpu_fallback=False)
    backend_diagnostic.update(
        {
            "graph_marker_count": MAX_BACKEND_FAILURE_RECORDS + 1,
            "affected_graph_marker_count": MAX_BACKEND_FAILURE_RECORDS + 1,
            "cpu_node_marker_count": MAX_BACKEND_FAILURE_RECORDS + 1,
            "records_truncated": True,
        }
    )

    with pytest.raises(ValidationError, match="truncated"):
        SmokeServerLogFailureEvidence.model_validate(
            {
                "schema_version": "inkling-smoke-server-log-failure-v1",
                "present": True,
                "size_bytes": 1,
                "sha256": "a" * 64,
                "raw_log_recorded": False,
                "scan_integrity": "complete",
                "safe_failure_signals": {
                    "out_of_memory_observed": False,
                    "no_usable_gpu_observed": False,
                    "model_load_failure_observed": False,
                    "projector_load_failure_observed": False,
                    "unsupported_architecture_observed": False,
                },
                "backend_diagnostic": backend_diagnostic,
            }
        )


def test_server_log_failure_evidence_rejects_unmarked_truncation() -> None:
    backend_diagnostic = _backend_failure_diagnostic(cpu_fallback=False)
    backend_diagnostic["cpu_node_marker_count"] = MAX_BACKEND_FAILURE_RECORDS + 1

    with pytest.raises(ValidationError, match="truncated"):
        SmokeServerLogFailureEvidence.model_validate(
            {
                "schema_version": "inkling-smoke-server-log-failure-v1",
                "present": True,
                "size_bytes": 1,
                "sha256": "a" * 64,
                "raw_log_recorded": False,
                "scan_integrity": "complete",
                "safe_failure_signals": {
                    "out_of_memory_observed": False,
                    "no_usable_gpu_observed": False,
                    "model_load_failure_observed": False,
                    "projector_load_failure_observed": False,
                    "unsupported_architecture_observed": False,
                },
                "backend_diagnostic": backend_diagnostic,
            }
        )


@pytest.mark.parametrize(
    "backend_diagnostic",
    (
        {
            **_backend_failure_diagnostic(cpu_fallback=False),
            "cpu_node_marker_count": 1,
        },
        _backend_diagnostic_with_graph_without_cpu_marker(),
        _backend_diagnostic_with_partial_graph_coverage(),
        {
            **_backend_failure_diagnostic(cpu_fallback=True),
            "graph_marker_count": 2,
            "affected_graph_marker_count": 2,
        },
    ),
    ids=(
        "cpu-marker-without-graph",
        "positive-cpu-graph-without-marker",
        "partial-graph-sample-coverage",
        "affected-count-differs-from-retained-graphs",
    ),
)
def test_server_log_failure_evidence_rejects_inconsistent_complete_backend_scan(
    backend_diagnostic: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError, match="complete backend scan"):
        SmokeServerLogFailureEvidence.model_validate(
            {
                "schema_version": "inkling-smoke-server-log-failure-v1",
                "present": True,
                "size_bytes": 1,
                "sha256": "a" * 64,
                "raw_log_recorded": False,
                "scan_integrity": "complete",
                "safe_failure_signals": {
                    "out_of_memory_observed": False,
                    "no_usable_gpu_observed": False,
                    "model_load_failure_observed": False,
                    "projector_load_failure_observed": False,
                    "unsupported_architecture_observed": False,
                },
                "backend_diagnostic": backend_diagnostic,
            }
        )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.__setitem__("present", True),
        lambda value: value.__setitem__("size_bytes", 1),
        lambda value: value.__setitem__("sha256", "0" * 64),
        lambda value: value.__setitem__("scan_integrity", "complete"),
        lambda value: value["safe_failure_signals"].__setitem__(
            "model_load_failure_observed",
            True,
        ),
        lambda value: value.__setitem__(
            "backend_diagnostic",
            _backend_failure_diagnostic(cpu_fallback=True),
        ),
        lambda value: value["backend_diagnostic"].__setitem__("graph_marker_count", 1),
        lambda value: value["backend_diagnostic"].__setitem__(
            "affected_graph_marker_count",
            1,
        ),
        lambda value: value["backend_diagnostic"].__setitem__("records_truncated", True),
    ),
    ids=(
        "present",
        "nonzero-size",
        "nonempty-hash",
        "complete-scan",
        "safe-signal",
        "positive-backend-diagnostic",
        "nonzero-backend-count",
        "nonzero-affected-backend-count",
        "truncated-backend-records",
    ),
)
def test_server_log_failure_evidence_requires_exact_missing_log_identity(
    mutate: Any,
) -> None:
    payload = {
        "schema_version": "inkling-smoke-server-log-failure-v1",
        "present": False,
        "size_bytes": 0,
        "sha256": EMPTY_SHA256,
        "raw_log_recorded": False,
        "scan_integrity": "missing",
        "safe_failure_signals": {
            "out_of_memory_observed": False,
            "no_usable_gpu_observed": False,
            "model_load_failure_observed": False,
            "projector_load_failure_observed": False,
            "unsupported_architecture_observed": False,
        },
        "backend_diagnostic": _backend_failure_diagnostic(cpu_fallback=False),
    }
    mutate(payload)

    with pytest.raises(ValidationError):
        SmokeServerLogFailureEvidence.model_validate(payload)


def test_server_log_failure_evidence_rejects_positive_malformed_scan() -> None:
    payload = {
        "schema_version": "inkling-smoke-server-log-failure-v1",
        "present": True,
        "size_bytes": 123,
        "sha256": "0" * 64,
        "raw_log_recorded": False,
        "scan_integrity": "malformed",
        "safe_failure_signals": {
            "out_of_memory_observed": False,
            "no_usable_gpu_observed": False,
            "model_load_failure_observed": False,
            "projector_load_failure_observed": False,
            "unsupported_architecture_observed": False,
        },
        "backend_diagnostic": _backend_failure_diagnostic(cpu_fallback=True),
    }

    with pytest.raises(ValidationError, match="requires one complete server-log scan"):
        SmokeServerLogFailureEvidence.model_validate(payload)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.__setitem__("raw_log_recorded", True),
        lambda value: value.__setitem__("raw_log", "unguarded server output"),
        lambda value: value.__setitem__("error_excerpt", "unguarded failure text"),
        lambda value: value["backend_diagnostic"].__setitem__(
            "raw_marker_lines_recorded",
            True,
        ),
        lambda value: value["backend_diagnostic"].__setitem__(
            "raw_node_names_recorded",
            True,
        ),
    ),
    ids=(
        "raw-log-recorded",
        "raw-log-field",
        "error-excerpt",
        "raw-markers-recorded",
        "raw-node-names-recorded",
    ),
)
def test_server_log_failure_evidence_rejects_raw_or_extra_fields(mutate: Any) -> None:
    payload = {
        "schema_version": "inkling-smoke-server-log-failure-v1",
        "present": True,
        "size_bytes": 123,
        "sha256": "0" * 64,
        "raw_log_recorded": False,
        "scan_integrity": "complete",
        "safe_failure_signals": {
            "out_of_memory_observed": False,
            "no_usable_gpu_observed": False,
            "model_load_failure_observed": False,
            "projector_load_failure_observed": False,
            "unsupported_architecture_observed": False,
        },
        "backend_diagnostic": _backend_failure_diagnostic(cpu_fallback=False),
    }
    mutate(payload)

    with pytest.raises(ValidationError):
        SmokeServerLogFailureEvidence.model_validate(payload)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.__setitem__("safe_subprocess_failure", None),
        lambda value: value.__setitem__(
            "exception_type",
            "pydantic_core._pydantic_core.ValidationError",
        ),
        lambda value: value["safe_subprocess_failure"].__setitem__("return_code", 0),
        lambda value: value["safe_subprocess_failure"].__setitem__(
            "command_id",
            "arbitrary_command_v1",
        ),
        lambda value: value["safe_subprocess_failure"].__setitem__(
            "stdout_recorded",
            True,
        ),
        lambda value: value["safe_subprocess_failure"].__setitem__(
            "stderr_recorded",
            True,
        ),
        lambda value: value["safe_subprocess_failure"].__setitem__(
            "stdout_sha256",
            "f" * 63,
        ),
        lambda value: value["safe_subprocess_failure"].__setitem__(
            "stderr_size_bytes",
            -1,
        ),
    ),
    ids=(
        "called-process-error-without-evidence",
        "evidence-without-called-process-error",
        "zero-return-code",
        "unallowlisted-command",
        "stdout-recorded",
        "stderr-recorded",
        "invalid-stdout-hash",
        "negative-stderr-size",
    ),
)
def test_failure_receipt_v4_rejects_inconsistent_subprocess_evidence(
    mutate: Any,
) -> None:
    (
        receipt,
        config,
        reference,
        control_plane,
        run_id,
        launch_intent_sha256,
    ) = _failure_receipt(
        "inkling-smoke-terminal-v4",
        called_process_error=True,
    )
    mutate(receipt)
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="schema is invalid"):
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("stdout", "raw command output"),
        ("stderr", "raw error output"),
        ("argv", ["nvidia-smi", "--query-gpu=uuid"]),
        ("environment", {"TOKEN": "secret"}),
        ("error_excerpt", "unguarded diagnostic text"),
    ),
)
def test_failure_receipt_v4_rejects_raw_or_extra_subprocess_fields(
    field: str,
    value: object,
) -> None:
    (
        receipt,
        config,
        reference,
        control_plane,
        run_id,
        launch_intent_sha256,
    ) = _failure_receipt(
        "inkling-smoke-terminal-v4",
        called_process_error=True,
    )
    receipt["safe_subprocess_failure"][field] = value
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="schema is invalid"):
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )


def test_failure_receipt_v4_rejects_subprocess_evidence_tampering() -> None:
    (
        receipt,
        config,
        reference,
        control_plane,
        run_id,
        launch_intent_sha256,
    ) = _failure_receipt(
        "inkling-smoke-terminal-v4",
        called_process_error=True,
    )
    receipt["safe_subprocess_failure"]["stderr_sha256"] = "a" * 64

    with pytest.raises(ValueError, match="self-hash"):
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("call_id", "fc-DIFFERENT"),
        ("input_id", "in-DIFFERENT"),
        ("task_id", "ta-DIFFERENT"),
        ("launch_intent_sha256", "9" * 64),
        ("smoke_config_hash", "a" * 64),
        ("control_plane_sha256", "b" * 64),
    ),
)
def test_failure_receipt_v3_rejects_invocation_identity_drift(
    field: str,
    replacement: str,
) -> None:
    receipt, config, reference, control_plane, run_id, launch_intent_sha256 = _failure_receipt()
    receipt["invocation"][field] = replacement
    if field == "launch_intent_sha256":
        receipt["invocation"]["post_spawn_acceptance_path"] = (
            f"control/post-spawn-acceptances/{replacement}.json"
        )
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="schema is invalid") as error:
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )

    assert "failure invocation differs from the top-level receipt identity" in str(
        error.value.__cause__
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("attempt_claim_path", "control/other.claim.json"),
        ("attempt_claim_sha256", "c" * 63),
        ("invocation_history_path", "control/history/smoke_test.attempt.2.bad.json"),
        ("invocation_history_sha256", "d" * 63),
    ),
)
def test_failure_receipt_v3_rejects_invalid_attempt_record_binding(
    field: str,
    replacement: str,
) -> None:
    receipt, config, reference, control_plane, run_id, launch_intent_sha256 = _failure_receipt()
    receipt["invocation"][field] = replacement
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="schema is invalid"):
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("run_id", "different-run"),
        ("subject_run_id", "different-subject"),
        ("smoke_config_hash", "2" * 64),
        ("verified_export_reference_sha256", "3" * 64),
        ("control_plane_sha256", "4" * 64),
        ("launch_intent_sha256", "5" * 64),
    ),
)
def test_failure_receipt_rejects_changed_launch_bindings(
    field: str,
    replacement: str,
) -> None:
    receipt, config, reference, control_plane, run_id, launch_intent_sha256 = _failure_receipt()
    receipt[field] = replacement
    if field == "run_id":
        receipt["invocation"]["run_id"] = replacement
        receipt["invocation"]["attempt_registry_key"] = f"{replacement}:smoke_test"
    if field in {
        "smoke_config_hash",
        "control_plane_sha256",
        "launch_intent_sha256",
    }:
        receipt["invocation"][field] = replacement
        if field == "launch_intent_sha256":
            receipt["invocation"]["post_spawn_acceptance_path"] = (
                f"control/post-spawn-acceptances/{replacement}.json"
            )
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="does not bind the exact launch"):
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )


def test_failure_receipt_rejects_tampering_and_extra_fields() -> None:
    receipt, config, reference, control_plane, run_id, launch_intent_sha256 = _failure_receipt()
    tampered = copy.deepcopy(receipt)
    tampered["safe_failure_signals"]["out_of_memory_observed"] = True
    with pytest.raises(ValueError, match="self-hash"):
        validate_smoke_failure_receipt(
            tampered,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )

    extra = copy.deepcopy(receipt)
    extra["error_message"] = "sensitive, ungoverned text"
    extra["receipt_sha256"] = smoke_terminal_receipt_sha256(extra)
    with pytest.raises(ValueError, match="schema is invalid"):
        validate_smoke_failure_receipt(
            extra,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )


def test_failure_receipt_rejects_duplicate_json_keys() -> None:
    receipt, config, reference, control_plane, run_id, launch_intent_sha256 = _failure_receipt()
    payload = json.dumps(receipt)
    duplicate = payload.replace(
        '"status": "failed"',
        '"status": "failed", "status": "failed"',
        1,
    )

    with pytest.raises(ValueError, match="duplicate key 'status'"):
        validate_smoke_failure_receipt(
            duplicate,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )


@pytest.mark.parametrize(
    "outcome_path",
    (
        "smoke_test.failed.{wrong}.json",
        "runs/wrong/control/outcomes/{filename}",
        "runs/{run_id}/control/outcomes/../{filename}",
        "/runs/{run_id}/control/outcomes/{filename}",
    ),
)
def test_failure_receipt_rejects_wrong_outcome_path(outcome_path: str) -> None:
    receipt, config, reference, control_plane, run_id, launch_intent_sha256 = _failure_receipt()
    filename = f"smoke_test.failed.{receipt['receipt_sha256']}.json"
    supplied_path = outcome_path.format(
        wrong="f" * 64,
        filename=filename,
        run_id=run_id,
    )

    with pytest.raises(ValueError, match="outcome"):
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
            outcome_path=supplied_path,
        )


def test_terminal_receipt_v2_can_never_be_a_success() -> None:
    receipt, config, reference, control_plane, run_id, launch_intent_sha256 = _failure_receipt(
        "inkling-smoke-terminal-v2"
    )
    receipt["status"] = "passed"
    receipt.pop("receipt_sha256")

    with pytest.raises(ValueError, match="historical failures"):
        smoke_terminal_receipt_sha256(receipt)
    receipt["receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash domain is invalid"):
        validate_smoke_failure_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
            launch_intent_sha256=launch_intent_sha256,
        )


def test_package_inventory_parsers_are_canonical_and_fail_closed() -> None:
    assert canonical_python_package_inventory([("Example_Package", "1.2.3"), ("pip", "25.1")]) == {
        "example-package": "1.2.3",
        "pip": "25.1",
    }
    assert parse_dpkg_inventory(b"libc6=2.39-0ubuntu8\npython3=3.12.3-0ubuntu2\n") == {
        "libc6": "2.39-0ubuntu8",
        "python3": "3.12.3-0ubuntu2",
    }
    assert parse_nvcc_version("Cuda compilation tools, release 13.1, V13.1.115\n") == "V13.1.115"

    with pytest.raises(ValueError, match="duplicate Python"):
        canonical_python_package_inventory([("a_b", "1"), ("a-b", "1")])
    with pytest.raises(ValueError, match="duplicate Debian"):
        parse_dpkg_inventory(b"libc6=1\nlibc6=1\n")
    with pytest.raises(ValueError, match="exactly one"):
        parse_nvcc_version("release 13.1")


def test_dpkg_inventory_preserves_multiarch_package_identities() -> None:
    assert parse_dpkg_inventory(b"libc6:amd64=2.39-0ubuntu8\nlibc6:arm64=2.39-0ubuntu8\n") == {
        "libc6:amd64": "2.39-0ubuntu8",
        "libc6:arm64": "2.39-0ubuntu8",
    }


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ("1600000 100000\n", 16_000),
        ("250000 100000", 2_500),
        ("max 100000\n", None),
        ("-1 100000\n", None),
    ),
)
def test_cgroup_cpu_quota_parser(
    payload: str,
    expected: int | None,
) -> None:
    assert parse_cgroup_cpu_quota_millicores(payload) == expected


@pytest.mark.parametrize(
    "payload",
    ("", "1600000", "0 100000", "1600000 0", "150001 100000", "invalid 100000"),
)
def test_cgroup_cpu_quota_parser_rejects_ambiguous_values(payload: str) -> None:
    with pytest.raises(ValueError, match="cgroup CPU"):
        parse_cgroup_cpu_quota_millicores(payload)


@pytest.mark.parametrize(
    ("payload", "expected"),
    (
        ("68719476736\n", 64 * 1024**3),
        ("max\n", None),
        ("-1\n", None),
        ("9223372036854771712\n", None),
    ),
)
def test_cgroup_memory_limit_parser(
    payload: str,
    expected: int | None,
) -> None:
    assert parse_cgroup_memory_limit_bytes(payload) == expected


@pytest.mark.parametrize("payload", ("", "0", "1.5", "unlimited", "68719476736 bytes"))
def test_cgroup_memory_limit_parser_rejects_ambiguous_values(payload: str) -> None:
    with pytest.raises(ValueError, match="cgroup memory"):
        parse_cgroup_memory_limit_bytes(payload)


def test_current_process_cgroup_leaf_resolver_maps_v2_through_mount_root() -> None:
    proc_self_cgroup = "0::/tenant.slice/modal/job-123/worker-456\n"
    proc_self_mountinfo = """29 23 0:25 / /proc rw,nosuid,nodev,noexec,relatime - proc proc rw
42 29 0:31 /tenant.slice /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw
"""

    assert resolve_current_process_cgroup_leaf_paths(
        proc_self_cgroup=proc_self_cgroup,
        proc_self_mountinfo=proc_self_mountinfo,
    ) == {
        "cpu": Path("/sys/fs/cgroup/modal/job-123/worker-456"),
        "memory": Path("/sys/fs/cgroup/modal/job-123/worker-456"),
    }
    assert resolve_current_process_cgroup_hierarchy_paths(
        proc_self_cgroup=proc_self_cgroup,
        proc_self_mountinfo=proc_self_mountinfo,
    ) == {
        "cpu": (
            Path("/sys/fs/cgroup/modal/job-123/worker-456"),
            Path("/sys/fs/cgroup"),
        ),
        "memory": (
            Path("/sys/fs/cgroup/modal/job-123/worker-456"),
            Path("/sys/fs/cgroup"),
        ),
    }


@pytest.mark.parametrize(
    ("cpu_record_controllers", "cpu_mount_controllers"),
    (
        ("cpu,cpuacct", "cpuacct,cpu"),
        ("cpuacct,cpu", "cpu,cpuacct"),
    ),
)
def test_current_process_cgroup_leaf_resolver_maps_v1_controller_permutations(
    cpu_record_controllers: str,
    cpu_mount_controllers: str,
) -> None:
    proc_self_cgroup = (
        f"8:{cpu_record_controllers}:/tenant.slice/cpu/job-123\n"
        "7:memory:/tenant.slice/memory/job-123\n"
        "6:cpuset:/unrelated\n"
        "5:name=systemd:/unrelated-systemd\n"
    )
    proc_self_mountinfo = (
        "41 29 0:30 /tenant.slice/cpu /sys/fs/cgroup/cpu-leaf "
        "rw,nosuid,nodev,noexec,relatime - cgroup cgroup "
        f"rw,{cpu_mount_controllers}\n"
        "42 29 0:31 /tenant.slice/memory /sys/fs/cgroup/memory-leaf "
        "rw,nosuid,nodev,noexec,relatime - cgroup cgroup rw,memory\n"
        "43 29 0:32 / /sys/fs/cgroup/cpuset "
        "rw,nosuid,nodev,noexec,relatime - cgroup cgroup rw,cpuset\n"
    )

    assert resolve_current_process_cgroup_leaf_paths(
        proc_self_cgroup=proc_self_cgroup,
        proc_self_mountinfo=proc_self_mountinfo,
    ) == {
        "cpu": Path("/sys/fs/cgroup/cpu-leaf/job-123"),
        "memory": Path("/sys/fs/cgroup/memory-leaf/job-123"),
    }


@pytest.mark.parametrize(
    ("proc_self_cgroup", "proc_self_mountinfo"),
    (
        (
            "0::/tenant.slice/../../etc\n",
            "42 29 0:31 /tenant.slice /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        ),
        (
            "0::/tenant.slice/job-123\n",
            "42 29 0:31 /tenant.slice/../escape /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        ),
        (
            "0::/tenant.slice/job-123\n",
            "42 29 0:31 /tenant.slice /sys/fs/cgroup/../escape rw - cgroup2 cgroup rw\n",
        ),
        (
            "0::/tenant.slice-other/job-123\n",
            "42 29 0:31 /tenant.slice /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        ),
    ),
    ids=(
        "cgroup-leaf-traversal",
        "mount-root-traversal",
        "mount-point-traversal",
        "leaf-outside-mount-root",
    ),
)
def test_current_process_cgroup_leaf_resolver_rejects_path_escape(
    proc_self_cgroup: str,
    proc_self_mountinfo: str,
) -> None:
    with pytest.raises(ValueError):
        resolve_current_process_cgroup_leaf_paths(
            proc_self_cgroup=proc_self_cgroup,
            proc_self_mountinfo=proc_self_mountinfo,
        )


@pytest.mark.parametrize(
    ("proc_self_cgroup", "proc_self_mountinfo"),
    (
        (
            "0::/tenant/job-123\n0::/tenant/job-456\n",
            "42 29 0:31 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        ),
        (
            "0::/tenant/job-123\n",
            "42 29 0:31 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
            "43 29 0:32 /tenant /sys/fs/cgroup-alt rw - cgroup2 cgroup rw\n",
        ),
        (
            "0::/tenant/job-123\n",
            "42 29 0:31 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
            "43 29 0:31 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n",
        ),
        (
            "0::/tenant/nested/job-123\n",
            "42 29 0:31 /tenant /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
            "43 29 0:32 /tenant/nested /sys/fs/cgroup/nested "
            "rw - cgroup2 cgroup rw\n",
        ),
        (
            "8:cpu,cpuacct:/tenant/job-123\n",
            "41 29 0:30 /tenant /sys/fs/cgroup/cpu rw - cgroup cgroup rw,cpu,cpuacct\n",
        ),
    ),
    ids=(
        "duplicate-v2-record",
        "duplicate-v2-mount",
        "duplicate-v2-mount-mapping",
        "overlapping-v2-mounts",
        "missing-v1-memory",
    ),
)
def test_current_process_cgroup_leaf_resolver_rejects_ambiguous_or_incomplete_inventory(
    proc_self_cgroup: str,
    proc_self_mountinfo: str,
) -> None:
    with pytest.raises(ValueError):
        resolve_current_process_cgroup_leaf_paths(
            proc_self_cgroup=proc_self_cgroup,
            proc_self_mountinfo=proc_self_mountinfo,
        )


def test_immutable_source_tree_identity_is_deterministic_and_binds_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "data.bin").write_bytes(b"stable data")

    first = immutable_source_tree_identity(source)
    assert immutable_source_tree_identity(source) == first
    assert first[0] == 2

    (source / "module.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert immutable_source_tree_identity(source)[1] != first[1]


def test_immutable_source_tree_identity_binds_project_relative_paths(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "alpha.py").write_bytes(b"identical bytes")
    (second / "beta.py").write_bytes(b"identical bytes")

    assert immutable_source_tree_identity(first)[0] == 1
    assert immutable_source_tree_identity(second)[0] == 1
    assert immutable_source_tree_identity(first)[1] != immutable_source_tree_identity(second)[1]


def test_immutable_source_tree_identity_excludes_runtime_caches(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    expected = immutable_source_tree_identity(source)

    cache = source / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-312.pyc").write_bytes(b"cache one")
    (source / "orphan.pyc").write_bytes(b"cache two")
    (source / "orphan.pyo").write_bytes(b"cache three")

    assert immutable_source_tree_identity(source) == expected


def test_immutable_source_tree_identity_rejects_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source / "alias.py").symlink_to(source / "module.py")

    with pytest.raises(ValueError, match="symlink or special file"):
        immutable_source_tree_identity(source)

    source_link = tmp_path / "source-link"
    source_link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        immutable_source_tree_identity(source_link)


def test_linux_host_parsers_and_topology_hash_bind_every_identity() -> None:
    cpuinfo = """processor : 0
model name : AMD EPYC 9V84

processor : 1
model name : AMD EPYC 9V84
"""
    assert parse_proc_cpu_model(cpuinfo) == "AMD EPYC 9V84"
    assert parse_proc_mem_total_bytes("MemTotal:       1048576 kB\n") == 1024**3

    host = {
        "provider": "Modal",
        "cpu_model": "AMD EPYC 9V84",
        "host_logical_cpu_count": 64,
        "host_logical_cpu_count_scope": "host_online_os_cpu_count",
        "logical_cpu_count": 16,
        "logical_cpu_count_scope": "container_cgroup_cpu_quota",
        "requested_cpu_cores": 16,
        "requested_cpu_scope": "modal_physical_cores_hard_request_and_limit",
        "cgroup_membership_sha256": "5" * 64,
        "cgroup_mountinfo_sha256": "6" * 64,
        "cgroup_visibility_scope": "process_mount_namespace_visible_hierarchy",
        "cgroup_process_pid": 1234,
        "cgroup_cpu_leaf_path_sha256": "7" * 64,
        "cgroup_cpu_leaf_pid_verified": True,
        "cgroup_cpu_leaf_cgroup_procs_sha256": "8" * 64,
        "cgroup_cpu_quota_millicores": 16_000,
        "cgroup_cpu_quota_source": "cgroup_v2_visible_hierarchy_cpu.max",
        "cgroup_cpu_limit_path_sha256s": ("9" * 64,),
        "cgroup_cpu_limit_values_millicores": (16_000,),
        "cpu_affinity_ids": tuple(range(128, 144)),
        "cpu_affinity_scope": "container_effective_sched_getaffinity",
        "host_ram_bytes": 128 * 1024**3,
        "host_ram_scope": "host_physical_proc_meminfo_memtotal",
        "ram_bytes": 64 * 1024**3,
        "ram_scope": "container_cgroup_memory_limit",
        "requested_ram_bytes": 64 * 1024**3,
        "requested_ram_scope": "modal_bytes_hard_request_and_limit",
        "cgroup_memory_leaf_path_sha256": "a" * 64,
        "cgroup_memory_leaf_pid_verified": True,
        "cgroup_memory_leaf_cgroup_procs_sha256": "b" * 64,
        "cgroup_memory_limit_bytes": 64 * 1024**3,
        "cgroup_memory_limit_source": "cgroup_v2_visible_hierarchy_memory.max",
        "cgroup_memory_limit_path_sha256s": ("c" * 64,),
        "cgroup_memory_limit_values_bytes": (64 * 1024**3,),
        "nvidia_smi_topo_m_sha256": "4" * 64,
        "topology_schema_version": "inkling-smoke-hardware-topology-v3",
    }
    gpus = [
        {
            "identity_protocol": "cuda-driver-uuid+nvidia-smi-uuid-v1",
            "cuda_driver_api_version": 13_010,
            "cuda_ordinal": 0,
            "uuid": "GPU-00000000-0000-0000-0000-000000000001",
            "cuda_driver_name": "NVIDIA B300",
            "nvidia_smi_name": "NVIDIA B300",
            "cuda_compute_capability": "10.3",
            "nvidia_smi_compute_capability": "10.3",
            "cuda_total_memory_bytes": 262144 * 1024**2,
            "nvidia_smi_memory_total_mib": 262144,
            "driver_version": "580.1",
            "pci_bus_id": "00000000:01:00.0",
            "pci_bus_id_status": "available",
            "pci_bus_id_source": "cuda_driver_api",
            "pci_bus_id_error_code": None,
        }
    ]
    first = smoke_hardware_topology_sha256(host, gpus)
    host_evidence = SmokeHostEvidence.model_validate({**host, "topology_sha256": first})
    changed = json.loads(json.dumps(gpus))
    changed[0]["uuid"] = "GPU-00000000-0000-0000-0000-000000000002"

    assert smoke_hardware_topology_sha256(host, gpus) == first
    assert smoke_hardware_topology_sha256(host_evidence, gpus) == first
    assert host_evidence.cpu_affinity_ids == tuple(range(128, 144))
    assert smoke_hardware_topology_sha256(host, changed) != first
    changed_host = dict(host)
    changed_host["cgroup_cpu_limit_values_millicores"] = (16_000, None)
    changed_host["cgroup_cpu_limit_path_sha256s"] = ("9" * 64, "d" * 64)
    assert smoke_hardware_topology_sha256(changed_host, gpus) != first


def _package_manifest_arguments() -> dict[str, object]:
    return {
        "python_implementation": "CPython",
        "python_version": "3.12.10",
        "python_executable_path": "/usr/local/bin/python3.12",
        "python_executable_sha256": "0" * 64,
        "python_inventory_scope": "image_sysconfig_purelib_v1",
        "python_purelib": "/usr/local/lib/python3.12/site-packages",
        "python_inventory_sha256": "1" * 64,
        "python_packages": {"modal": "1.5.0", "pip": "25.1"},
        "modal_runtime_version": "1.5.0",
        "modal_package_root": "/pkg/modal",
        "modal_package_tree_schema_version": "inkling-smoke-source-tree-v1",
        "modal_package_file_count": 42,
        "modal_package_tree_sha256": "2" * 64,
        "dpkg_inventory_sha256": "3" * 64,
        "dpkg_packages": {
            "libc6:amd64": "2.39-0ubuntu8",
            "zlib1g:amd64": "1:1.3.dfsg-3.1ubuntu2",
        },
        "nvcc_version": "V13.1.115",
        "nvcc_version_sha256": "4" * 64,
    }


def test_package_manifest_hash_is_independent_of_package_map_order() -> None:
    arguments = _package_manifest_arguments()
    reordered = dict(arguments)
    reordered["python_packages"] = {"pip": "25.1", "modal": "1.5.0"}
    reordered["dpkg_packages"] = {
        "zlib1g:amd64": "1:1.3.dfsg-3.1ubuntu2",
        "libc6:amd64": "2.39-0ubuntu8",
    }

    assert smoke_package_manifest_sha256(**arguments) == smoke_package_manifest_sha256(**reordered)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("python_implementation", "PyPy"),
        ("python_version", "3.12.11"),
        ("python_executable_path", "/usr/local/bin/python3"),
        ("python_executable_sha256", "9" * 64),
        ("python_inventory_scope", "other-scope"),
        ("python_purelib", "/different/purelib"),
        ("python_inventory_sha256", "8" * 64),
        ("python_packages", {"modal": "1.5.1", "pip": "25.1"}),
        ("modal_runtime_version", "1.5.1"),
        ("modal_package_root", "/different/modal"),
        ("modal_package_tree_schema_version", "inkling-smoke-source-tree-v2"),
        ("modal_package_file_count", 43),
        ("modal_package_tree_sha256", "7" * 64),
        ("dpkg_inventory_sha256", "6" * 64),
        (
            "dpkg_packages",
            {
                "libc6:amd64": "2.40-0ubuntu1",
                "zlib1g:amd64": "1:1.3.dfsg-3.1ubuntu2",
            },
        ),
        ("nvcc_version", "V13.1.116"),
        ("nvcc_version_sha256", "5" * 64),
    ),
)
def test_package_manifest_hash_binds_every_v2_identity(
    field: str,
    replacement: object,
) -> None:
    arguments = _package_manifest_arguments()
    first = smoke_package_manifest_sha256(**arguments)
    changed = dict(arguments)
    changed[field] = replacement

    assert smoke_package_manifest_sha256(**changed) != first
