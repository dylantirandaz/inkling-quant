from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from inkling_quant_lab.gguf.inkling_smoke import (
    InklingSmokeConfig,
    load_verified_export_reference,
)
from inkling_quant_lab.gguf.inkling_smoke_execution import (
    SmokeControlPlaneProvenance,
    SmokeFailureReceiptV3,
    smoke_run_id,
    smoke_terminal_receipt_sha256,
    strict_json_object,
    validate_smoke_failure_receipt,
    validate_smoke_terminal_receipt,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "tests/fixtures/inkling_smoke"
REFERENCE_PATH = PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_verified_export.json"

FIXTURE_FILE_SHA256 = {
    "historical_v3_context.synthetic.json": (
        "c54487e5a55afe2c692f4e48cee7c4f3adcda78a65a12c063a9eb153a0d95315"
    ),
    "historical_v3_failure.synthetic.json": (
        "68994333ce74baff3167e2efe79e2a50d1088aafe0872536d9a3ac82a192a383"
    ),
    "historical_v3_success.synthetic.json": (
        "9adf5eeb6a9ce838d947a3cf3aca45c464d13ed200d3d1fc7367ed94d9f57ce6"
    ),
    "historical_v4_failure_metadata.json": (
        "b6b206d7a55b856b5f9e6acad1c0ebba9e6bdc1e1d1ace90fa28f1ecc29ad7ae"
    ),
    "historical_v5_cpu_failure_metadata.json": (
        "a7f2f5a990692f976f835e51557eb6e43c3d8b6f83f65fe4818ba15156f397ac"
    ),
}
V3_SUCCESS_RECEIPT_SHA256 = "907b7d51fa384a6856bce2ba320b45d86e8e4bad80929f2ca7f552787a131d7f"
V3_FAILURE_RECEIPT_SHA256 = "d557c1fc0167d57737b4400a30e87c87b90a6d8a3a8cc0101313f4425696eb19"
KNOWN_V4_FAILURE_RECEIPT_SHA256 = "0dc186c26e973ed14b4875cf435107e7ef7c78c45e8260dab26166951ba8ae72"
KNOWN_V4_SERVER_LOG_SHA256 = "8bd6b2be2a762b504122f2dc0d4e876756d9f3a5af4dd9d7b2d2769c13b69bd2"
KNOWN_V5_CPU_FAILURE_RECEIPT_SHA256 = (
    "b0ec38d43d96f448a9e258f4ad4e32d55a59de37e85e2ea1d443f9db8715da7a"
)
KNOWN_V5_CPU_SERVER_LOG_SHA256 = "d848365973900761e90db79f86f5a081919fb3bd48e5f355f8d4677affe78773"


def _fixture(name: str) -> dict[str, Any]:
    raw = (FIXTURE_ROOT / name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_FILE_SHA256[name]
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b"\r" not in raw
    value = strict_json_object(raw.decode("utf-8"))
    assert raw == (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")
    return value


def _v3_context() -> tuple[
    InklingSmokeConfig,
    SmokeControlPlaneProvenance,
    str,
]:
    context = _fixture("historical_v3_context.synthetic.json")
    assert context["fixture_class"] == "synthetic_contract_only_not_run_evidence"
    config = InklingSmokeConfig.model_validate(context["config"])
    control_plane = SmokeControlPlaneProvenance.model_validate(context["control_plane"])
    return config, control_plane, smoke_run_id(config, control_plane.tree_sha256)


def test_fixed_synthetic_v3_success_fixture_passes_frozen_v3_rules() -> None:
    receipt = _fixture("historical_v3_success.synthetic.json")
    config, control_plane, run_id = _v3_context()
    reference = load_verified_export_reference(REFERENCE_PATH)

    assert receipt["receipt_sha256"] == V3_SUCCESS_RECEIPT_SHA256
    assert smoke_terminal_receipt_sha256(receipt) == V3_SUCCESS_RECEIPT_SHA256
    observed = validate_smoke_terminal_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
    )

    assert observed.schema_version == "inkling-smoke-terminal-v3"
    assert observed.server.command[-3:] == (
        "--log-verbosity",
        "4",
        "--no-webui",
    )
    assert tuple(
        (identity.backend_index, identity.backend_name, identity.device_name)
        for identity in observed.server.backend_audit.identities
    ) == (
        (0, "B300-0", "B300-0"),
        (1, "B300-1", "B300-1"),
    )
    assert "tools/server/server.cpp" not in observed.runtime.patched_source_paths


def test_fixed_synthetic_v3_failure_fixture_passes_frozen_v3_rules() -> None:
    receipt = _fixture("historical_v3_failure.synthetic.json")
    config, control_plane, run_id = _v3_context()
    reference = load_verified_export_reference(REFERENCE_PATH)
    outcome_path = (
        f"runs/{run_id}/control/outcomes/smoke_test.failed.{V3_FAILURE_RECEIPT_SHA256}.json"
    )

    assert receipt["receipt_sha256"] == V3_FAILURE_RECEIPT_SHA256
    assert smoke_terminal_receipt_sha256(receipt) == V3_FAILURE_RECEIPT_SHA256
    observed = validate_smoke_failure_receipt(
        receipt,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=run_id,
        launch_intent_sha256=receipt["launch_intent_sha256"],
        outcome_path=outcome_path,
    )

    assert isinstance(observed, SmokeFailureReceiptV3)
    assert observed.failure_phase == "verify_runtime_and_allocation"
    assert not any(observed.safe_failure_signals.model_dump().values())


def test_fixed_v3_success_cannot_be_relabelled_as_v4() -> None:
    receipt = copy.deepcopy(_fixture("historical_v3_success.synthetic.json"))
    config, control_plane, run_id = _v3_context()
    reference = load_verified_export_reference(REFERENCE_PATH)
    receipt["schema_version"] = "inkling-smoke-terminal-v4"
    receipt["receipt_sha256"] = smoke_terminal_receipt_sha256(receipt)

    with pytest.raises(ValueError, match="receipt schema is invalid"):
        validate_smoke_terminal_receipt(
            receipt,
            config=config,
            reference=reference,
            control_plane=control_plane,
            run_id=run_id,
        )


def test_known_v4_failure_metadata_remains_exact_without_inventing_raw_evidence() -> None:
    metadata = _fixture("historical_v4_failure_metadata.json")

    assert metadata == {
        "call_id": "fc-01KY7XBYMCRY6W86ZV45TRSF3Z",
        "failure": "loader log lacks CUDA device-count evidence",
        "failure_phase": "stop_server",
        "failure_receipt_sha256": KNOWN_V4_FAILURE_RECEIPT_SHA256,
        "http_probes_completed": True,
        "instrumentation_schema_version": ("inkling-llama-smoke-instrumentation-v1"),
        "later_terminal_parsers_ran": False,
        "model_failure": False,
        "raw_receipt_bytes_available_locally": False,
        "raw_server_log_recoverable": False,
        "record_class": "immutable_historical_metadata_only",
        "result_classification": "evidence_capture_configuration_failure",
        "run_id": "inkling-smoke-86b4d430-a015409e-5dec66fa00-b2f34d3a28",
        "safe_failure_signals": {
            "model_load_failure_observed": False,
            "no_usable_gpu_observed": False,
            "out_of_memory_observed": False,
            "projector_load_failure_observed": False,
            "unsupported_architecture_observed": False,
        },
        "schema_version": "inkling-smoke-historical-failure-metadata-v1",
        "server_log_sha256": KNOWN_V4_SERVER_LOG_SHA256,
        "smoke_config_schema_version": "inkling-smoke-config-v1",
        "smoke_pass": False,
        "terminal_schema_version": "inkling-smoke-terminal-v4",
    }


def test_known_v5_cpu_failure_metadata_does_not_invent_the_cpu_operation() -> None:
    metadata = _fixture("historical_v5_cpu_failure_metadata.json")

    assert metadata == {
        "call_id": "fc-01KY8TVQ3V0BWAWW65G788263M",
        "exact_cpu_operation_recoverable": False,
        "exception_type": "builtins.ValueError",
        "failure": "backend audit observed a CPU model graph operation",
        "failure_phase": "stop_server",
        "failure_receipt_sha256": KNOWN_V5_CPU_FAILURE_RECEIPT_SHA256,
        "http_probes_completed": True,
        "instrumentation_schema_version": "inkling-llama-smoke-instrumentation-v2",
        "model_failure": False,
        "raw_receipt_bytes_available_locally": False,
        "raw_server_log_recoverable": False,
        "record_class": "immutable_historical_metadata_only",
        "result_classification": "backend_placement_failure",
        "run_id": "inkling-smoke-86b4d430-a015409e-605f467370-cecb27bed3",
        "safe_failure_signals": {
            "model_load_failure_observed": False,
            "no_usable_gpu_observed": False,
            "out_of_memory_observed": False,
            "projector_load_failure_observed": False,
            "unsupported_architecture_observed": False,
        },
        "schema_version": "inkling-smoke-historical-failure-metadata-v1",
        "server_log_sha256": KNOWN_V5_CPU_SERVER_LOG_SHA256,
        "smoke_config_schema_version": "inkling-smoke-config-v1",
        "smoke_pass": False,
        "terminal_schema_version": "inkling-smoke-terminal-v5",
    }
