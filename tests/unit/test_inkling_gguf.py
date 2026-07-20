from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from inkling_quant_lab.exceptions import CapabilityError, ConfigurationError
from inkling_quant_lab.gguf import inkling as inkling_module
from inkling_quant_lab.gguf.inkling import (
    EXPECTED_AUDIO_TENSORS,
    EXPECTED_MODEL_BYTES,
    EXPECTED_MTP_TENSORS,
    EXPECTED_TEXT_TENSORS,
    EXPECTED_VISION_TENSORS,
    PINNED_INKLING_REVISION,
    PINNED_LLAMA_CPP_COMMIT,
    InklingGGUFConfig,
    PaidLaunchAcknowledgement,
    WorkflowPaths,
    audit_inkling_source,
    build_conversion_plan,
    build_quantize_command,
    build_verification_plan,
    compute_cost_ceiling_usd,
    configured_deployable_wall_time_hours,
    inkling_control_plane_provenance,
    inkling_run_id,
    load_inkling_gguf_config,
    require_initial_billing_window,
    require_materialize_initial_billing_window,
    require_stage_billing_window,
    validate_deployed_control_plane,
    validate_paid_launch_acknowledgement,
    verify_execution_bindings,
)

CONFIG_PATH = Path("configs/experiments/inkling_q3_k_m_modal.yaml")


def _weight_map() -> dict[str, str]:
    shards = [f"model-{index:05d}-of-00108.safetensors" for index in range(1, 109)]
    result: dict[str, str] = {}
    for index in range(EXPECTED_TEXT_TENSORS):
        result[f"model.language_model.synthetic_text_{index}.weight"] = shards[index % len(shards)]
    for index in range(EXPECTED_VISION_TENSORS):
        result[f"model.visual.synthetic_{index}.weight"] = shards[index % len(shards)]
    for index in range(EXPECTED_AUDIO_TENSORS):
        result[f"model.audio.synthetic_{index}.weight"] = shards[index % len(shards)]
    for index in range(EXPECTED_MTP_TENSORS):
        result[f"model.mtp.synthetic_{index}.weight"] = "mtp.safetensors"
    return result


def _source_documents() -> tuple[dict[str, object], ...]:
    return (
        {
            "id": "thinkingmachines/Inkling",
            "sha": PINNED_INKLING_REVISION,
            "private": False,
            "cardData": {"license": "apache-2.0"},
        },
        {
            "architectures": ["InklingForConditionalGeneration"],
            "model_type": "inkling_mm_model",
        },
        {
            "metadata": {"total_size": EXPECTED_MODEL_BYTES},
            "weight_map": _weight_map(),
        },
    )


def _copy_weight_map(weight_index: dict[str, object]) -> dict[str, str]:
    raw = weight_index["weight_map"]
    if not isinstance(raw, dict):
        raise AssertionError("test fixture weight_map must be a dictionary")
    return {str(name): str(filename) for name, filename in raw.items()}


@pytest.fixture(autouse=True)
def _pin_synthetic_index_for_unit_fixtures(monkeypatch: pytest.MonkeyPatch) -> None:
    weight_index = _source_documents()[2]
    payload = json.dumps(
        weight_index, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    monkeypatch.setattr(
        inkling_module,
        "EXPECTED_WEIGHT_INDEX_CANONICAL_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )


def test_checked_config_pins_the_real_model_and_experimental_toolchain() -> None:
    config = load_inkling_gguf_config(CONFIG_PATH)

    assert config.source.model_id == "thinkingmachines/Inkling"
    assert config.source.revision == PINNED_INKLING_REVISION
    assert config.source.trust_remote_code is False
    assert config.source.license == "apache-2.0"
    assert config.source.checkpoint_format == "safetensors"
    assert config.toolchain.commit == PINNED_LLAMA_CPP_COMMIT
    assert config.quantization.quant_type == "Q3_K_M"
    assert config.quantization.use_importance_matrix is False
    assert config.coverage.multimodal == "separate_bf16_mmproj"
    assert config.coverage.mtp == "omitted_unsupported"
    assert config.config_hash() == config.config_hash()


def test_source_audit_accepts_only_the_exact_pinned_inkling_inventory() -> None:
    config = InklingGGUFConfig()
    model_info, model_config, weight_index = _source_documents()

    audit = audit_inkling_source(
        config,
        model_info=model_info,
        model_config=model_config,
        weight_index=weight_index,
    )

    assert audit.verified is True
    assert audit.license == "apache-2.0"
    assert audit.source_tensor_count == 1_552
    assert audit.source_shard_count == 109
    assert audit.text_tensor_count == EXPECTED_TEXT_TENSORS
    assert audit.vision_tensor_count == EXPECTED_VISION_TENSORS
    assert audit.audio_tensor_count == EXPECTED_AUDIO_TENSORS
    assert audit.mtp_tensor_count == EXPECTED_MTP_TENSORS
    assert audit.converted_source_tensor_count == 1_392
    assert audit.omitted_source_tensor_count == EXPECTED_MTP_TENSORS


@pytest.mark.parametrize(
    ("document", "key", "bad_value", "match"),
    [
        (0, "id", "another/model", "model id"),
        (0, "sha", "0" * 40, "revision"),
        (1, "architectures", ["MixtralForCausalLM"], "architecture"),
        (1, "model_type", "mixtral", "model type"),
    ],
)
def test_source_audit_rejects_identity_substitution(
    document: int, key: str, bad_value: object, match: str
) -> None:
    config = InklingGGUFConfig()
    documents = [dict(value) for value in _source_documents()]
    documents[document][key] = bad_value

    with pytest.raises(ConfigurationError, match=match):
        audit_inkling_source(
            config,
            model_info=documents[0],
            model_config=documents[1],
            weight_index=documents[2],
        )


def test_source_audit_rejects_wrong_size_and_pickle_weight() -> None:
    config = InklingGGUFConfig()
    model_info, model_config, weight_index = _source_documents()
    bad_size = dict(weight_index)
    bad_size["metadata"] = {"total_size": EXPECTED_MODEL_BYTES - 1}

    with pytest.raises(ConfigurationError, match="total_size"):
        audit_inkling_source(
            config,
            model_info=model_info,
            model_config=model_config,
            weight_index=bad_size,
        )

    bad_pickle = dict(weight_index)
    weight_map = _copy_weight_map(bad_pickle)
    first = next(iter(weight_map))
    weight_map[first] = "pytorch_model.bin"
    bad_pickle["weight_map"] = weight_map
    with pytest.raises(ConfigurationError, match="safetensors"):
        audit_inkling_source(
            config,
            model_info=model_info,
            model_config=model_config,
            weight_index=bad_pickle,
        )


def test_source_audit_rejects_wrong_tensor_count() -> None:
    config = InklingGGUFConfig()
    model_info, model_config, weight_index = _source_documents()
    bad_index = dict(weight_index)
    weight_map = _copy_weight_map(bad_index)
    weight_map.pop(next(iter(weight_map)))
    bad_index["weight_map"] = weight_map

    with pytest.raises(ConfigurationError, match="1552 tensors"):
        audit_inkling_source(
            config,
            model_info=model_info,
            model_config=model_config,
            weight_index=bad_index,
        )


def test_source_audit_rejects_wrong_component_counts() -> None:
    config = InklingGGUFConfig()
    model_info, model_config, weight_index = _source_documents()
    bad_index = dict(weight_index)
    weight_map = _copy_weight_map(bad_index)
    vision_name = next(name for name in weight_map if name.startswith("model.visual."))
    filename = weight_map.pop(vision_name)
    weight_map["model.language_model.reclassified_visual.weight"] = filename
    bad_index["weight_map"] = weight_map

    with pytest.raises(ConfigurationError, match="component tensor counts"):
        audit_inkling_source(
            config,
            model_info=model_info,
            model_config=model_config,
            weight_index=bad_index,
        )


def test_source_audit_rejects_wrong_shard_count() -> None:
    config = InklingGGUFConfig()
    model_info, model_config, weight_index = _source_documents()
    bad_index = dict(weight_index)
    weight_map = _copy_weight_map(bad_index)
    for name, filename in tuple(weight_map.items()):
        if filename == "mtp.safetensors":
            weight_map[name] = "model-00001-of-00108.safetensors"
    bad_index["weight_map"] = weight_map

    with pytest.raises(ConfigurationError, match="109 shards"):
        audit_inkling_source(
            config,
            model_info=model_info,
            model_config=model_config,
            weight_index=bad_index,
        )


@pytest.mark.parametrize(
    "unsafe_filename",
    ("/tmp/substituted.safetensors", "../substituted.safetensors"),
)
def test_source_audit_rejects_absolute_or_escaping_index_filename(
    unsafe_filename: str,
) -> None:
    config = InklingGGUFConfig()
    model_info, model_config, weight_index = _source_documents()
    bad_index = dict(weight_index)
    weight_map = _copy_weight_map(bad_index)
    weight_map[next(iter(weight_map))] = unsafe_filename
    bad_index["weight_map"] = weight_map

    with pytest.raises(ConfigurationError, match="unsafe files"):
        audit_inkling_source(
            config,
            model_info=model_info,
            model_config=model_config,
            weight_index=bad_index,
        )


def test_source_audit_rejects_canonical_index_digest_drift() -> None:
    config = InklingGGUFConfig()
    model_info, model_config, weight_index = _source_documents()
    bad_index = dict(weight_index)
    bad_index["unexpected_metadata"] = "drift"

    with pytest.raises(ConfigurationError, match="canonical SHA-256"):
        audit_inkling_source(
            config,
            model_info=model_info,
            model_config=model_config,
            weight_index=bad_index,
        )


def test_source_audit_rejects_license_drift() -> None:
    config = InklingGGUFConfig()
    model_info, model_config, weight_index = _source_documents()
    bad_model_info = dict(model_info)
    bad_model_info["cardData"] = {"license": "other"}

    with pytest.raises(ConfigurationError, match="license"):
        audit_inkling_source(
            config,
            model_info=bad_model_info,
            model_config=model_config,
            weight_index=weight_index,
        )


def test_plan_uses_pinned_local_snapshot_stock_q3_and_no_shell() -> None:
    config = InklingGGUFConfig()
    paths = WorkflowPaths(
        source_dir=Path("/vol/source/inkling"),
        work_dir=Path("/vol/work/run"),
        final_dir=Path("/vol/final/run"),
        llama_cpp_dir=Path("/opt/llama.cpp"),
    )

    plan = build_conversion_plan(config, paths)
    quantize = build_quantize_command(
        config,
        paths,
        first_bf16_split=paths.work_dir / "inkling-BF16-00001-of-00041.gguf",
    )

    assert "--remote" not in plan.text_conversion.argv
    assert str(paths.source_dir) in plan.text_conversion.argv
    assert plan.text_conversion.argv[-1] == "--no-tensor-first-split"
    assert "--mmproj" in plan.mmproj_conversion.argv
    assert quantize.argv[-2:] == ("Q3_K_M", str(config.quantization.threads))
    assert "--keep-split" in quantize.argv
    assert all("UD-Q3_K_XL" not in argument for argument in quantize.argv)
    assert all(";" not in argument and "&&" not in argument for argument in quantize.argv)

    verification = build_verification_plan(
        config,
        paths,
        first_q3_split=paths.final_dir / "q3_k_m/inkling-Q3_K_M-00001-of-00010.gguf",
        mmproj_file=paths.final_dir / "mmproj/mmproj-BF16.gguf",
    )
    assert verification.q3_split_set.argv[1:3] == ("--merge", "--dry-run")
    assert verification.mmproj.argv[1:3] == ("--split", "--dry-run")
    assert str(paths.final_dir / "mmproj/mmproj-BF16.gguf") in verification.mmproj.argv


def test_cost_ceiling_is_below_both_compute_and_user_caps() -> None:
    config = InklingGGUFConfig()
    ceiling = compute_cost_ceiling_usd(config)

    assert ceiling == Decimal("561.45")
    assert config.modal.rate_card.model_dump(mode="json") == {
        "as_of": "2026-07-17",
        "cpu_core_hour_usd": "0.04716",
        "memory_gib_hour_usd": "0.007992",
        "b300_gpu_hour_usd": "7.0992",
    }
    assert [
        (
            stage.name,
            stage.cpu_cores,
            stage.memory_gib,
            stage.gpu_type,
            stage.gpu_count,
            stage.ephemeral_disk_mib,
            stage.startup_timeout_seconds,
            stage.max_hours,
            stage.max_attempts,
            stage.max_recovery_attempts,
        )
        for stage in config.modal.stages
    ] == [
        ("materialize_source", 8, 32, None, 0, 524_288, 900, Decimal("4"), 12, 0),
        ("convert_text_bf16", 32, 192, None, 0, 3_145_728, 900, Decimal("23"), 2, 1),
        (
            "convert_multimodal_projector",
            8,
            32,
            None,
            0,
            524_288,
            900,
            Decimal("12"),
            2,
            1,
        ),
        ("quantize_text", 32, 192, None, 0, 3_145_728, 900, Decimal("23"), 2, 1),
        ("verify_export", 16, 64, None, 0, 524_288, 900, Decimal("12"), 2, 1),
        ("smoke_test", 16, 64, "B300", 2, 524_288, 900, Decimal("2"), 1, 0),
    ]
    assert all(
        Decimal(stage.ephemeral_disk_mib) / Decimal(1024 * 20) <= Decimal(stage.memory_gib)
        for stage in config.modal.stages
    )
    assert ceiling <= config.budget.planned_compute_usd
    assert config.budget.planned_compute_usd == Decimal("600")
    assert config.budget.planned_storage_usd == Decimal("150")
    assert config.budget.workspace_contingency_usd == Decimal("50")
    assert config.budget.workspace_hard_budget_usd == Decimal("800")
    assert config.budget.external_contingency_usd == Decimal("150")
    assert (
        config.budget.planned_compute_usd
        + config.budget.planned_storage_usd
        + config.budget.workspace_contingency_usd
        == config.budget.workspace_hard_budget_usd
    )
    assert (
        config.budget.workspace_hard_budget_usd + config.budget.external_contingency_usd
        == config.budget.max_total_usd
    )
    assert config.budget.max_total_usd < Decimal("1000")
    assert configured_deployable_wall_time_hours(config) == Decimal("264.00")


def test_monthly_budget_window_reserves_the_documented_storage_deletion_lag() -> None:
    config = InklingGGUFConfig()
    cycle_end = "2099-08-01T00:00:00Z"
    first_allowed_instant = datetime(2099, 7, 17, tzinfo=UTC)

    require_initial_billing_window(config, cycle_end, now=first_allowed_instant)
    with pytest.raises(ConfigurationError, match="billing-cycle window"):
        require_initial_billing_window(
            config,
            cycle_end,
            now=first_allowed_instant + timedelta(seconds=1),
        )

    quantize_last_start = datetime(2099, 7, 27, 0, 45, tzinfo=UTC)
    require_stage_billing_window(
        config,
        cycle_end,
        "quantize_text",
        now=quantize_last_start,
        include_startup=True,
    )
    with pytest.raises(ConfigurationError, match="billing-cycle window"):
        require_stage_billing_window(
            config,
            cycle_end,
            "quantize_text",
            now=quantize_last_start + timedelta(seconds=1),
            include_startup=True,
        )

    materialize_last_start = datetime(2099, 7, 25, 21, 0, tzinfo=UTC)
    require_stage_billing_window(
        config,
        cycle_end,
        "materialize_source",
        now=materialize_last_start,
        include_startup=True,
        invocations=12,
    )
    with pytest.raises(ConfigurationError, match="billing-cycle window"):
        require_stage_billing_window(
            config,
            cycle_end,
            "materialize_source",
            now=materialize_last_start + timedelta(seconds=1),
            include_startup=True,
            invocations=12,
        )

    require_materialize_initial_billing_window(
        config,
        cycle_end,
        now=materialize_last_start,
    )
    with pytest.raises(ConfigurationError, match="billing-cycle window"):
        require_materialize_initial_billing_window(
            config,
            cycle_end,
            now=materialize_last_start + timedelta(seconds=1),
        )


def test_negative_rates_and_budget_envelopes_are_rejected() -> None:
    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["modal"]["rate_card"]["cpu_core_hour_usd"] = "-1"
    with pytest.raises(ValueError, match="greater than 0"):
        InklingGGUFConfig.model_validate(raw)

    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["modal"]["rate_card"]["cpu_core_hour_usd"] = "0"
    with pytest.raises(ValueError, match="greater than 0"):
        InklingGGUFConfig.model_validate(raw)

    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["budget"]["external_contingency_usd"] = "-1"
    with pytest.raises(ValueError, match="greater than or equal to 0"):
        InklingGGUFConfig.model_validate(raw)

    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["modal"]["rate_card"]["cpu_core_hour_usd"] = "0.000000000001"
    with pytest.raises(ValueError, match="verified"):
        InklingGGUFConfig.model_validate(raw)

    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["modal"]["stages"][1]["cpu_cores"] = 1
    with pytest.raises(ValueError, match="resource/timeout/attempt matrix"):
        InklingGGUFConfig.model_validate(raw)

    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["modal"]["stages"][1]["startup_timeout_seconds"] = 899
    with pytest.raises(ValueError, match="resource/timeout/attempt matrix"):
        InklingGGUFConfig.model_validate(raw)

    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["modal"]["stages"][1]["ephemeral_disk_mib"] = 3_145_729
    with pytest.raises(ValueError, match="less than or equal to 3145728"):
        InklingGGUFConfig.model_validate(raw)

    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["modal"]["stages"][0]["ephemeral_disk_mib"] = 655_361
    with pytest.raises(ValueError, match="impute more RAM"):
        InklingGGUFConfig.model_validate(raw)

    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["modal"]["stages"][1]["ephemeral_disk_mib"] = 524_288
    with pytest.raises(ValueError, match="resource/timeout/attempt matrix"):
        InklingGGUFConfig.model_validate(raw)

    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["modal"]["source_volume"] = "different-model-source"
    with pytest.raises(ValueError, match="inkling-source-v1"):
        InklingGGUFConfig.model_validate(raw)

    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["budget"]["planned_compute_usd"] = "561"
    raw["budget"]["workspace_hard_budget_usd"] = "761"
    raw["budget"]["max_total_usd"] = "911"
    with pytest.raises(ValueError, match=r"configured compute ceiling 561\.45 exceeds 561"):
        InklingGGUFConfig.model_validate(raw)

    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["budget"]["workspace_hard_budget_usd"] = "799"
    with pytest.raises(ValueError, match="planned Modal envelopes must equal"):
        InklingGGUFConfig.model_validate(raw)

    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["budget"]["max_total_usd"] = "951"
    with pytest.raises(ValueError, match="workspace budget and external contingency must equal"):
        InklingGGUFConfig.model_validate(raw)


@pytest.mark.parametrize("label", ["another-model-Q3_K_M", "inkling-Q4_K_M", "UD-Q3_K_XL"])
def test_quantization_label_rejects_every_misleading_recipe(label: str) -> None:
    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["quantization"]["output_label"] = label

    with pytest.raises(CapabilityError, match=r"(truthfully|unpublished)"):
        InklingGGUFConfig.model_validate(raw)


def test_execution_binding_rejects_different_snapshot_or_binary() -> None:
    config = InklingGGUFConfig()
    paths = WorkflowPaths(
        source_dir=Path("/vol/source/inkling"),
        work_dir=Path("/vol/work/run"),
        final_dir=Path("/vol/final/run"),
        llama_cpp_dir=Path("/opt/llama.cpp"),
    )
    receipt = {
        "verified": True,
        "config_hash": config.config_hash(),
        "model_id": config.source.model_id,
        "revision": config.source.revision,
        "license": config.source.license,
        "source_dir": str(paths.source_dir),
        "weight_index_sha256": inkling_module.EXPECTED_WEIGHT_INDEX_CANONICAL_SHA256,
        "inventory_sha256": "b" * 64,
        "source_config_sha256": "c" * 64,
    }
    evidence = verify_execution_bindings(
        config,
        paths,
        source_receipt=receipt,
        actual_llama_cpp_commit=config.toolchain.commit,
    )
    assert evidence.model_id == "thinkingmachines/Inkling"

    with pytest.raises(ConfigurationError, match="receipt"):
        verify_execution_bindings(
            config,
            paths,
            source_receipt={**receipt, "revision": "0" * 40},
            actual_llama_cpp_commit=config.toolchain.commit,
        )
    with pytest.raises(ConfigurationError, match="checkout"):
        verify_execution_bindings(
            config,
            paths,
            source_receipt=receipt,
            actual_llama_cpp_commit="0" * 40,
        )


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    (
        ("verified", False),
        ("config_hash", "0" * 64),
        ("model_id", "another/model"),
        ("revision", "0" * 40),
        ("license", "other"),
        ("source_dir", "/vol/source/another-model"),
        ("weight_index_sha256", "0" * 64),
        ("inventory_sha256", "short"),
        ("source_config_sha256", "z" * 64),
    ),
)
def test_execution_binding_rejects_source_receipt_field_drift(
    field: str, drifted_value: object
) -> None:
    config = InklingGGUFConfig()
    paths = WorkflowPaths(
        source_dir=Path("/vol/source/inkling"),
        work_dir=Path("/vol/work/run"),
        final_dir=Path("/vol/final/run"),
        llama_cpp_dir=Path("/opt/llama.cpp"),
    )
    receipt: dict[str, object] = {
        "verified": True,
        "config_hash": config.config_hash(),
        "model_id": config.source.model_id,
        "revision": config.source.revision,
        "license": config.source.license,
        "source_dir": str(paths.source_dir),
        "weight_index_sha256": inkling_module.EXPECTED_WEIGHT_INDEX_CANONICAL_SHA256,
        "inventory_sha256": "b" * 64,
        "source_config_sha256": "c" * 64,
    }
    receipt[field] = drifted_value

    with pytest.raises(ConfigurationError, match="receipt"):
        verify_execution_bindings(
            config,
            paths,
            source_receipt=receipt,
            actual_llama_cpp_commit=config.toolchain.commit,
        )


def test_invalid_unreproducible_ud_label_fails_closed() -> None:
    raw = InklingGGUFConfig().model_dump(mode="json")
    raw["quantization"]["output_label"] = "UD-Q3_K_XL"

    with pytest.raises(CapabilityError, match="importance matrix"):
        InklingGGUFConfig.model_validate(raw)


def test_control_plane_hash_binds_run_id_and_paid_acknowledgement(tmp_path: Path) -> None:
    required = {
        "pyproject.toml": "[project]\nname='fixture'\n",
        "uv.lock": "version = 1\n",
        "configs/experiments/inkling_q3_k_m_modal.yaml": "schema_version: '1.0'\n",
        "configs/experiments/inkling_q3_k_m_source_adoption.json": "{}\n",
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

    config = InklingGGUFConfig()
    provenance = inkling_control_plane_provenance(tmp_path)
    run_id = inkling_run_id(config, provenance.tree_sha256)
    assert provenance.file_count == len(required)
    assert run_id.endswith(provenance.tree_sha256[:10])

    acknowledgement = PaidLaunchAcknowledgement(
        config_hash=config.config_hash(),
        control_plane_sha256=provenance.tree_sha256,
        launch_intent_sha256="a" * 64,
        workspace_budget_usd=config.budget.workspace_hard_budget_usd,
        billing_cycle_end_utc="2099-08-01T00:00:00Z",
    )
    validated = validate_paid_launch_acknowledgement(
        config,
        acknowledgement.canonical_json(),
        control_plane_sha256=provenance.tree_sha256,
    )
    assert validated.workspace_budget_usd == Decimal("800")
    assert validated.launch_intent_sha256 == "a" * 64
    assert validated.billing_cycle_end_utc == "2099-08-01T00:00:00Z"

    (tmp_path / "scripts/quantize_inkling_modal.py").write_text(
        "print('changed')\n", encoding="utf-8"
    )
    changed = inkling_control_plane_provenance(tmp_path)
    assert changed.tree_sha256 != provenance.tree_sha256
    with pytest.raises(ConfigurationError, match="exact run"):
        validate_paid_launch_acknowledgement(
            config,
            acknowledgement.canonical_json(),
            control_plane_sha256=changed.tree_sha256,
        )


def test_paid_acknowledgement_binds_short_window_policy_and_assumed_utc_source() -> None:
    config = InklingGGUFConfig()
    acknowledgement = PaidLaunchAcknowledgement(
        config_hash=config.config_hash(),
        control_plane_sha256="a" * 64,
        launch_intent_sha256="b" * 64,
        workspace_budget_usd=config.budget.workspace_hard_budget_usd,
        billing_cycle_end_utc="2026-08-01T00:00:00Z",
        initial_billing_window_policy="operator_accepted_short_initial_window_v1",
        billing_cycle_end_source="user_confirmed_date_assumed_utc_midnight",
    )
    assert acknowledgement.schema_version == "1.3"

    raw = acknowledgement.model_dump(mode="json")
    raw["billing_cycle_end_source"] = "dashboard_exact_utc"
    with pytest.raises(ValueError, match="source does not match"):
        PaidLaunchAcknowledgement.model_validate(raw)


def test_deployed_control_plane_rehashes_mounted_script_and_package(tmp_path: Path) -> None:
    files = {
        "pyproject.toml": "[project]\nname='fixture'\n",
        "uv.lock": "version = 1\n",
        "configs/experiments/inkling_q3_k_m_modal.yaml": "schema_version: '1.0'\n",
        "configs/experiments/inkling_q3_k_m_source_adoption.json": "{}\n",
        "scripts/preflight_inkling_gguf.py": "print('preflight')\n",
        "scripts/manage_inkling_modal.py": "print('manage')\n",
        "scripts/quantize_inkling_modal.py": "print('paid')\n",
        "src/inkling_quant_lab/__init__.py": "\n",
        "src/inkling_quant_lab/gguf/inkling.py": "PIN = 1\n",
    }
    for relative, payload in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    provenance = inkling_control_plane_provenance(tmp_path)

    validated = validate_deployed_control_plane(
        provenance.canonical_json(),
        deployment_script=tmp_path / "scripts/quantize_inkling_modal.py",
        deployed_package_root=tmp_path / "src/inkling_quant_lab",
    )
    assert validated == provenance

    (tmp_path / "src/inkling_quant_lab/gguf/inkling.py").write_text("PIN = 2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="source bytes differ"):
        validate_deployed_control_plane(
            provenance.canonical_json(),
            deployment_script=tmp_path / "scripts/quantize_inkling_modal.py",
            deployed_package_root=tmp_path / "src/inkling_quant_lab",
        )


def test_deployed_control_plane_rejects_a_forged_tree_hash(tmp_path: Path) -> None:
    required = {
        "pyproject.toml": "[project]\nname='fixture'\n",
        "uv.lock": "version = 1\n",
        "configs/experiments/inkling_q3_k_m_modal.yaml": "schema_version: '1.0'\n",
        "configs/experiments/inkling_q3_k_m_source_adoption.json": "{}\n",
        "scripts/preflight_inkling_gguf.py": "print('preflight')\n",
        "scripts/manage_inkling_modal.py": "print('manage')\n",
        "scripts/quantize_inkling_modal.py": "print('paid')\n",
        "src/inkling_quant_lab/__init__.py": "\n",
    }
    for relative, payload in required.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    provenance = inkling_control_plane_provenance(tmp_path)
    forged = provenance.model_dump(mode="json")
    forged["tree_sha256"] = "0" * 64

    with pytest.raises(ConfigurationError, match="manifest is invalid"):
        validate_deployed_control_plane(
            json.dumps(forged),
            deployment_script=tmp_path / "scripts/quantize_inkling_modal.py",
            deployed_package_root=tmp_path / "src/inkling_quant_lab",
        )
