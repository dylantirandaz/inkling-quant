from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from inkling_quant_lab.exceptions import ConfigurationError
from inkling_quant_lab.gguf.inkling import (
    EXPECTED_AUDIO_TENSORS,
    EXPECTED_MODEL_BYTES,
    EXPECTED_MTP_TENSORS,
    EXPECTED_SOURCE_SHARDS,
    EXPECTED_SOURCE_TENSORS,
    EXPECTED_TEXT_TENSORS,
    EXPECTED_VISION_TENSORS,
    INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
    PINNED_INKLING_MODEL_ID,
    PINNED_INKLING_REVISION,
    InklingGGUFConfig,
    InklingSourceAdoptionReference,
    inkling_control_plane_provenance,
    inkling_source_adoption_reference_sha256,
    load_inkling_source_adoption_reference,
    validate_deployed_control_plane,
    validate_inkling_source_adoption_reference,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = PROJECT_ROOT / INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH
ORIGIN_RUN_ID = "inkling-q3km-86b4d430-a015409e-551ab8f240-bcc168525e"
ORIGIN_CONFIG_HASH = "551ab8f240269edbdc19efb61afc73e8b8b50e128e15781cf2248c674a8c4562"
ORIGIN_CONTROL_SHA256 = "bcc168525e8392944f4d19b8119fd888ab86f1cca620bbfd1c0d9e5dc5461ca3"


def _checked_mapping() -> dict[str, object]:
    value = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError("checked adoption reference must be a JSON object")
    return value


def _rehash(value: dict[str, object]) -> dict[str, object]:
    value["reference_sha256"] = inkling_source_adoption_reference_sha256(value)
    return value


def test_checked_source_adoption_reference_is_canonical_and_exact() -> None:
    reference = load_inkling_source_adoption_reference(REFERENCE_PATH)

    assert REFERENCE_PATH.read_bytes() == (reference.canonical_json() + "\n").encode("utf-8")
    assert reference.reference_sha256 == reference.computed_reference_sha256()
    assert reference.origin_run_id == ORIGIN_RUN_ID
    assert reference.origin_app_id == "ap-zhL7JVGVfoVSSeiKxrj1k3"
    assert reference.origin_app_name == "inkling-q3-k-m-bcc168525e83"
    assert reference.origin_app_required_state == "stopped"
    assert reference.origin_app_required_active_tasks == 0
    assert reference.origin_config_hash == ORIGIN_CONFIG_HASH
    assert reference.origin_control_plane_sha256 == ORIGIN_CONTROL_SHA256
    assert reference.origin_materialization_kind == "direct_huggingface_snapshot_v1"
    assert reference.origin_parent_adoption_reference_sha256 is None
    assert reference.source_volume == "inkling-source-v1"
    assert reference.source_mount_path == "/source"
    assert reference.source_run_root == f"/source/runs/{ORIGIN_RUN_ID}"
    assert reference.snapshot_path == f"/source/runs/{ORIGIN_RUN_ID}/snapshot"
    assert reference.model_id == PINNED_INKLING_MODEL_ID
    assert reference.revision == PINNED_INKLING_REVISION
    assert reference.license == "apache-2.0"
    assert reference.indexed_tensor_bytes == EXPECTED_MODEL_BYTES
    assert reference.materialized_weight_file_bytes == 1_904_755_463_940
    assert reference.source_tensor_count == EXPECTED_SOURCE_TENSORS
    assert reference.source_shard_count == EXPECTED_SOURCE_SHARDS
    assert reference.materialized_file_count == 117
    assert reference.text_tensor_count == EXPECTED_TEXT_TENSORS
    assert reference.vision_tensor_count == EXPECTED_VISION_TENSORS
    assert reference.audio_tensor_count == EXPECTED_AUDIO_TENSORS
    assert reference.mtp_tensor_count == EXPECTED_MTP_TENSORS
    assert reference.source_success_receipt.sha256 == (
        "06937bc535fb703da6adc9d11e1e804ce15f67b39ffbbe98a9a51a7dc70edbbc"
    )
    assert reference.source_inventory.sha256 == (
        "a8aa37efec2b12c5d584c8163111d3a8a22d9568ef01886343755a8af6ace571"
    )
    assert reference.origin_materialize_attempt_ledger.sha256 == (
        "3d30e78f8f2f8ee70a5fb7fe53109b17b8bb2c39bc64ecd2e9f58e9607fa51f3"
    )
    assert reference.origin_materialize_invocation_history.sha256 == (
        "42bb7cd1338485ea81c3c15346b29304f1c135167ea205877c72d1b902956456"
    )
    assert reference.local_materialize_call_receipt.sha256 == (
        "eba7fc186b0150e8e43dba50dd96c8bca865c8b6aa79587a1661e13f8b522a90"
    )


def test_adoption_reference_binds_a_distinct_target_run() -> None:
    reference = load_inkling_source_adoption_reference(REFERENCE_PATH)
    target_control_plane_sha256 = "d" * 64

    validated = validate_inkling_source_adoption_reference(
        reference,
        target_config=InklingGGUFConfig(),
        target_control_plane_sha256=target_control_plane_sha256,
    )

    assert validated is reference
    assert target_control_plane_sha256 != reference.origin_control_plane_sha256


@pytest.mark.parametrize(
    ("field_path", "bad_value", "match"),
    (
        (("origin_run_id",), "../another-run", "origin_run_id"),
        (("source_run_root",), "/source/runs/../another-run", "path"),
        (("snapshot_path",), "/work/runs/wrong/snapshot", "snapshot"),
        (
            ("source_success_receipt", "path"),
            "/source/runs/../source.success.json",
            "path",
        ),
        (("local_deployment_receipt", "path"), "../deployment.json", "path"),
    ),
)
def test_adoption_reference_rejects_unsafe_or_substituted_origin_paths(
    field_path: tuple[str, ...], bad_value: str, match: str
) -> None:
    raw = _checked_mapping()
    parent: dict[str, object] = raw
    for part in field_path[:-1]:
        child = parent[part]
        if not isinstance(child, dict):
            raise AssertionError("test field path must address a mapping")
        parent = child
    parent[field_path[-1]] = bad_value

    with pytest.raises(ValidationError, match=match):
        InklingSourceAdoptionReference.model_validate(_rehash(raw))


def test_adoption_reference_rejects_self_adoption() -> None:
    reference = load_inkling_source_adoption_reference(REFERENCE_PATH)

    with pytest.raises(ConfigurationError, match="self-adoption"):
        validate_inkling_source_adoption_reference(
            reference,
            target_config=InklingGGUFConfig(),
            target_control_plane_sha256=reference.origin_control_plane_sha256,
        )


def test_adoption_reference_rejects_transitive_adoption() -> None:
    raw = _checked_mapping()
    raw["origin_materialization_kind"] = "adopted_verified_source_v1"
    raw["origin_parent_adoption_reference_sha256"] = "f" * 64

    with pytest.raises(ValidationError, match=r"(direct_huggingface_snapshot_v1|None)"):
        InklingSourceAdoptionReference.model_validate(_rehash(raw))


def test_adoption_reference_rejects_hash_and_semantic_tampering() -> None:
    raw = _checked_mapping()
    raw["reference_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="self-hash"):
        InklingSourceAdoptionReference.model_validate(raw)

    raw = _checked_mapping()
    raw["materialized_weight_file_bytes"] = 1_904_755_463_941
    with pytest.raises(ValidationError, match="exact verified source evidence"):
        InklingSourceAdoptionReference.model_validate(_rehash(raw))


def test_adoption_reference_loader_rejects_noncanonical_json(tmp_path: Path) -> None:
    noncanonical = tmp_path / "adoption.json"
    noncanonical.write_text(json.dumps(_checked_mapping(), indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="canonical JSON"):
        load_inkling_source_adoption_reference(noncanonical)


def test_control_plane_provenance_includes_adoption_reference_as_local_only(
    tmp_path: Path,
) -> None:
    required = {
        "pyproject.toml": "[project]\nname='fixture'\n",
        "uv.lock": "version = 1\n",
        "configs/experiments/inkling_q3_k_m_modal.yaml": "schema_version: '1.1'\n",
        INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH: REFERENCE_PATH.read_text(encoding="utf-8"),
        "scripts/preflight_inkling_gguf.py": "print('preflight')\n",
        "scripts/manage_inkling_modal.py": "print('manage')\n",
        "scripts/quantize_inkling_modal.py": "print('paid')\n",
        "src/inkling_quant_lab/__init__.py": "\n",
        "src/inkling_quant_lab/gguf/inkling.py": "PIN = 1\n",
    }
    for relative, payload in required.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")

    before = inkling_control_plane_provenance(tmp_path)
    by_path = {item.path: item for item in before.files}
    assert INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH in by_path
    assert (
        validate_deployed_control_plane(
            before.canonical_json(),
            deployment_script=tmp_path / "scripts/quantize_inkling_modal.py",
            deployed_package_root=tmp_path / "src/inkling_quant_lab",
        )
        == before
    )

    reference_path = tmp_path / INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH
    reference_path.write_text(reference_path.read_text() + " ", encoding="utf-8")
    after = inkling_control_plane_provenance(tmp_path)
    assert after.tree_sha256 != before.tree_sha256
