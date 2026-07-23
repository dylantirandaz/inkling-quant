from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from inkling_quant_lab.gguf.inkling_smoke import (
    InklingSmokeConfig,
    InklingVerifiedExportReference,
    load_inkling_smoke_config,
    load_verified_export_reference,
)
from inkling_quant_lab.gguf.inkling_smoke_authorization import (
    SmokeLaunchIntent,
    smoke_launch_intent_remote_path,
    validate_smoke_launch_intent,
)
from inkling_quant_lab.gguf.inkling_smoke_execution import (
    SmokeControlPlaneFile,
    SmokeControlPlaneProvenance,
    SmokeLaunchDeploymentIdentity,
    smoke_control_plane_tree_sha256,
    smoke_deployment_tag,
    smoke_run_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ATTEMPT_REGISTRY_NAME = "inkling-smoke-attempt-registry-v1"
ATTEMPT_REGISTRY_ID = "di-Attempt123"
ATTEMPT_REGISTRY_CREATED_AT_UTC = "2026-07-22T12:00:00.000000Z"


@pytest.fixture
def smoke_config() -> InklingSmokeConfig:
    return load_inkling_smoke_config(
        PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_smoke_modal.yaml"
    )


@pytest.fixture
def export_reference() -> InklingVerifiedExportReference:
    return load_verified_export_reference(
        PROJECT_ROOT / "configs/experiments/inkling_q3_k_m_verified_export.json"
    )


@pytest.fixture
def control_plane() -> SmokeControlPlaneProvenance:
    files = (
        SmokeControlPlaneFile(
            path="scripts/smoke_inkling_modal.py",
            sha256="a" * 64,
            size_bytes=123,
        ),
    )
    return SmokeControlPlaneProvenance(
        tree_sha256=smoke_control_plane_tree_sha256(files),
        file_count=1,
        files=files,
    )


def _intent(
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
) -> SmokeLaunchIntent:
    return SmokeLaunchIntent(
        authorization_nonce="b" * 64,
        created_at_utc="2026-07-22T12:34:56.123456Z",
        run_id=smoke_run_id(config, control_plane.tree_sha256),
        app_name=f"inkling-q3-smoke-{control_plane.tree_sha256[:12]}",
        deployment_version=7,
        deployment_tag=smoke_deployment_tag(control_plane.tree_sha256),
        function_id="fu-Abc123",
        attempt_registry_name=ATTEMPT_REGISTRY_NAME,
        attempt_registry_id=ATTEMPT_REGISTRY_ID,
        attempt_registry_created_at_utc=ATTEMPT_REGISTRY_CREATED_AT_UTC,
        smoke_config_hash=config.config_hash(),
        verified_export_reference_sha256=reference.reference_sha256,
        control_plane_sha256=control_plane.tree_sha256,
        workspace_budget_usd="800",
        billing_cycle_end_utc="2026-08-01T00:00:00Z",
    )


def _validate(
    payload: bytes | str | dict[str, Any],
    *,
    intent_sha256: str,
    config: InklingSmokeConfig,
    reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
) -> SmokeLaunchIntent:
    expected_intent = _intent(config, reference, control_plane)
    deployment = SmokeLaunchDeploymentIdentity(
        app_name=expected_intent.app_name,
        deployment_version=expected_intent.deployment_version,
        deployment_tag=expected_intent.deployment_tag,
        function_id=expected_intent.function_id,
        function_name=expected_intent.function_name,
        attempt_registry_name=expected_intent.attempt_registry_name,
        attempt_registry_id=expected_intent.attempt_registry_id,
        attempt_registry_created_at_utc=(expected_intent.attempt_registry_created_at_utc),
    )
    return validate_smoke_launch_intent(
        payload,
        config=config,
        reference=reference,
        control_plane=control_plane,
        run_id=smoke_run_id(config, control_plane.tree_sha256),
        launch_intent_sha256=intent_sha256,
        deployment=deployment,
    )


def test_launch_intent_validates_exact_canonical_bytes(
    smoke_config: InklingSmokeConfig,
    export_reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
) -> None:
    intent = _intent(smoke_config, export_reference, control_plane)
    payload = intent.canonical_bytes()

    assert payload.endswith(b"\n")
    assert (
        _validate(
            payload,
            intent_sha256=hashlib.sha256(payload).hexdigest(),
            config=smoke_config,
            reference=export_reference,
            control_plane=control_plane,
        )
        == intent
    )


def test_launch_intent_remote_path_is_content_addressed() -> None:
    assert smoke_launch_intent_remote_path("smoke-run", "a" * 64) == (
        "runs/smoke-run/control/launch-authorizations/" + "a" * 64 + ".json"
    )
    with pytest.raises(ValueError, match="run ID"):
        smoke_launch_intent_remote_path("../escape", "a" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        smoke_launch_intent_remote_path("smoke-run", "A" * 64)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("authorization_nonce", "b" * 63, "schema"),
        ("created_at_utc", "2026-07-22T12:34:56Z", "schema"),
        ("deployment_version", "7", "schema"),
        ("deployment_version", 999, "sealed run"),
        ("function_id", "fu-Other", "sealed run"),
        ("attempt_registry_name", "other-registry", "schema"),
        ("attempt_registry_id", "di-Other123", "sealed run"),
        (
            "attempt_registry_created_at_utc",
            "2026-07-22T12:00:01.000000Z",
            "sealed run",
        ),
        ("workspace_budget_usd", "800.0", "schema"),
        ("billing_cycle_end_utc", "2026-08-01T00:00:00+00:00", "schema"),
        ("app_name", "inkling-q3-smoke-" + "0" * 12, "schema"),
        ("deployment_tag", "iql-smoke-" + "0" * 64, "schema"),
        ("smoke_config_hash", "0" * 64, "sealed run"),
        ("verified_export_reference_sha256", "0" * 64, "sealed run"),
    ),
)
def test_launch_intent_rejects_drift(
    field: str,
    value: object,
    message: str,
    smoke_config: InklingSmokeConfig,
    export_reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
) -> None:
    intent = _intent(smoke_config, export_reference, control_plane)
    raw = intent.model_dump(mode="json")
    raw[field] = value
    payload = (
        json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()

    with pytest.raises(ValueError, match=message):
        _validate(
            payload,
            intent_sha256=hashlib.sha256(payload).hexdigest(),
            config=smoke_config,
            reference=export_reference,
            control_plane=control_plane,
        )


def test_launch_intent_rejects_noncanonical_and_duplicate_json(
    smoke_config: InklingSmokeConfig,
    export_reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
) -> None:
    intent = _intent(smoke_config, export_reference, control_plane)
    canonical = intent.canonical_bytes()
    noncanonical = json.dumps(intent.model_dump(mode="json"), indent=2).encode()
    with pytest.raises(ValueError, match="not canonical"):
        _validate(
            noncanonical,
            intent_sha256=hashlib.sha256(noncanonical).hexdigest(),
            config=smoke_config,
            reference=export_reference,
            control_plane=control_plane,
        )

    duplicate = canonical[:-2] + b',"run_id":"duplicate"}\n'
    with pytest.raises(ValueError, match="duplicate key"):
        _validate(
            duplicate,
            intent_sha256=hashlib.sha256(duplicate).hexdigest(),
            config=smoke_config,
            reference=export_reference,
            control_plane=control_plane,
        )


def test_launch_intent_rejects_wrong_digest_and_extra_field(
    smoke_config: InklingSmokeConfig,
    export_reference: InklingVerifiedExportReference,
    control_plane: SmokeControlPlaneProvenance,
) -> None:
    intent = _intent(smoke_config, export_reference, control_plane)
    with pytest.raises(ValueError, match="hash"):
        _validate(
            intent.canonical_bytes(),
            intent_sha256="0" * 64,
            config=smoke_config,
            reference=export_reference,
            control_plane=control_plane,
        )

    raw = intent.model_dump(mode="json") | {"unsealed": True}
    payload = (
        json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()
    with pytest.raises(ValueError, match="schema"):
        _validate(
            payload,
            intent_sha256=hashlib.sha256(payload).hexdigest(),
            config=smoke_config,
            reference=export_reference,
            control_plane=control_plane,
        )
