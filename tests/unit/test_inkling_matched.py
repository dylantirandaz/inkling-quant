"""Exact subject and capacity contracts for matched Inkling work."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import inkling_quant_lab.gguf.inkling_matched as inkling_matched
from inkling_quant_lab.exceptions import CapabilityError, ConfigurationError
from inkling_quant_lab.gguf.inkling import (
    INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
    load_inkling_source_adoption_reference,
)
from inkling_quant_lab.gguf.inkling_matched import (
    BF16_SUBJECT_REFERENCE_RELATIVE_PATH,
    EXPECTED_BF16_SUBJECT_REFERENCE_SHA256,
    MATCHED_CELL_CONFIG_RELATIVE_PATH,
    InklingBF16SubjectReference,
    InklingMatchedCellConfig,
    bf16_subject_reference_sha256,
    load_bf16_subject_reference,
    load_matched_cell_bundle,
    load_matched_cell_config,
    screen_matched_capacity,
)
from inkling_quant_lab.gguf.inkling_smoke import (
    EXPECTED_PROJECTOR_BYTES,
    EXPECTED_PROJECTOR_SHA256,
    EXPECTED_Q3_TOTAL_BYTES,
    EXPECTED_VERIFIED_EXPORT_REFERENCE_SHA256,
    VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH,
    load_verified_export_reference,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BF16_REFERENCE_PATH = PROJECT_ROOT / BF16_SUBJECT_REFERENCE_RELATIVE_PATH
Q3_REFERENCE_PATH = PROJECT_ROOT / VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH
MATCHED_CONFIG_PATH = PROJECT_ROOT / MATCHED_CELL_CONFIG_RELATIVE_PATH
SOURCE_REFERENCE_PATH = PROJECT_ROOT / INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH
OBSERVED_B300_MEMORY_BYTES = 287_428_771_840


def _bf16_mapping() -> dict[str, object]:
    value = json.loads(BF16_REFERENCE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _rehash_bf16_reference(value: dict[str, object]) -> dict[str, object]:
    result = copy.deepcopy(value)
    result["reference_sha256"] = bf16_subject_reference_sha256(result)
    return result


def _matched_mapping() -> dict[str, object]:
    value = yaml.safe_load(MATCHED_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _copy_matched_bundle(project_root: Path) -> None:
    for relative_path in (
        BF16_SUBJECT_REFERENCE_RELATIVE_PATH,
        MATCHED_CELL_CONFIG_RELATIVE_PATH,
        VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH,
        INKLING_SOURCE_ADOPTION_REFERENCE_RELATIVE_PATH,
    ):
        source = PROJECT_ROOT / relative_path
        destination = project_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def test_checked_bf16_reference_is_exact_canonical_and_not_measurement_ready() -> None:
    reference = load_bf16_subject_reference(BF16_REFERENCE_PATH)

    assert BF16_REFERENCE_PATH.read_bytes() == (reference.canonical_json() + "\n").encode()
    assert reference.reference_sha256 == EXPECTED_BF16_SUBJECT_REFERENCE_SHA256
    assert reference.reference_sha256 == reference.computed_reference_sha256()
    assert reference.bf16_shard_count == 49
    assert len(reference.bf16_shards) == 49
    assert sum(item.size_bytes for item in reference.bf16_shards) == 1_894_278_547_552
    assert reference.projector.sha256 == EXPECTED_PROJECTOR_SHA256
    assert reference.q3_verified_export_reference_sha256 == (
        EXPECTED_VERIFIED_EXPORT_REFERENCE_SHA256
    )
    assert reference.storage_mutability == "mutable_modal_volume"
    assert reference.fresh_artifact_rehash_required is True
    assert reference.measurement_ready_without_rehash is False
    assert reference.quality_measured is False
    assert reference.deployment_benchmark_measured is False


def test_bf16_reference_rejects_inventory_order_and_identity_tampering() -> None:
    raw = _bf16_mapping()
    shards = raw["bf16_shards"]
    assert isinstance(shards, list)
    first = shards[0]
    assert isinstance(first, dict)
    first["sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="inventory differs"):
        InklingBF16SubjectReference.model_validate(_rehash_bf16_reference(raw))

    raw = _bf16_mapping()
    shards = raw["bf16_shards"]
    assert isinstance(shards, list)
    shards[0], shards[1] = shards[1], shards[0]
    with pytest.raises(ValidationError, match="ordered 49-file set"):
        InklingBF16SubjectReference.model_validate(_rehash_bf16_reference(raw))

    raw = _bf16_mapping()
    raw["subject_config_hash"] = "f" * 64
    with pytest.raises(ValidationError, match="subject_config_hash"):
        InklingBF16SubjectReference.model_validate(_rehash_bf16_reference(raw))

    raw = _bf16_mapping()
    raw["reference_sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="self-hash"):
        InklingBF16SubjectReference.model_validate(raw)


def test_bf16_reference_loader_rejects_noncanonical_json(tmp_path: Path) -> None:
    path = tmp_path / "bf16.json"
    path.write_text(json.dumps(_bf16_mapping(), indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="canonical JSON"):
        load_bf16_subject_reference(path)


def test_checked_matched_cell_binds_subjects_runtime_assets_and_claim_limits() -> None:
    config = load_matched_cell_config(MATCHED_CONFIG_PATH)

    assert config.bf16_subject_reference_sha256 == EXPECTED_BF16_SUBJECT_REFERENCE_SHA256
    assert config.q3_verified_export_reference_sha256 == (EXPECTED_VERIFIED_EXPORT_REFERENCE_SHA256)
    assert tuple(item.name for item in config.runtime.binaries) == (
        "llama-cli",
        "llama-server",
        "llama-bench",
        "llama-perplexity",
    )
    assert tuple(item.path for item in config.tokenizer_assets) == (
        "chat_template.jinja",
        "processor_config.json",
        "special_tokens_map.json",
        "tiktoken/tokenizer.model",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    assert config.resources.gpu_type == "B300"
    assert config.resources.gpu_count == 8
    assert config.runtime.tensor_split == (1, 1, 1, 1, 1, 1, 1, 1)
    assert config.runtime.no_cpu_fallback is True
    assert config.storage.final_mount_path == "/final"
    assert config.storage.source_volume == "inkling-source-v1"
    assert config.storage.source_mount_path == "/source"
    assert config.execution.record_status == "planning_only"
    assert config.execution.runner_implemented is False
    assert config.execution.remote_execution_allowed is False
    assert config.execution.subject_mode == "sequential_same_allocation"
    assert config.execution.subject_order == ("bf16", "q3")
    assert config.execution.measurement_execution_allowed is False
    assert config.claims.quality_retention_claim_allowed is False
    assert config.claims.speedup_claim_allowed is False
    assert config.claims.runtime_fit_proven is False
    assert len(config.config_hash()) == 64


def test_matched_cell_rejects_runtime_tampering() -> None:
    raw = _matched_mapping()
    runtime = raw["runtime"]
    assert isinstance(runtime, dict)
    binaries = runtime["binaries"]
    assert isinstance(binaries, list)
    first_binary = binaries[0]
    assert isinstance(first_binary, dict)
    first_binary["sha256"] = "f" * 64
    with pytest.raises(ValidationError, match="runtime binaries"):
        InklingMatchedCellConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("runner_implemented", True),
        ("remote_execution_allowed", True),
        ("measurement_execution_allowed", True),
    ],
)
def test_matched_cell_rejects_execution_enablement(
    field_name: str,
    value: bool,
) -> None:
    raw = _matched_mapping()
    execution = raw["execution"]
    assert isinstance(execution, dict)
    execution[field_name] = value
    with pytest.raises(ValidationError):
        InklingMatchedCellConfig.model_validate(raw)


def test_matched_bundle_loads_references_and_resolves_every_subject_path() -> None:
    bundle = load_matched_cell_bundle(PROJECT_ROOT)

    assert bundle.bf16.reference_sha256 == EXPECTED_BF16_SUBJECT_REFERENCE_SHA256
    assert bundle.q3.reference_sha256 == EXPECTED_VERIFIED_EXPORT_REFERENCE_SHA256
    assert bundle.source.reference_sha256 == bundle.config.source_adoption_reference_sha256
    assert len(bundle.paths.bf16_shards) == 49
    assert bundle.paths.bf16_shards[0] == ("/baseline/bf16/inkling-BF16-00001-of-00049.gguf")
    assert bundle.paths.bf16_shards[-1] == ("/baseline/bf16/inkling-BF16-00049-of-00049.gguf")
    assert bundle.paths.bf16_conversion_receipt == ("/baseline/convert_text_bf16.success.json")
    assert len(bundle.paths.q3_shards) == 49
    assert bundle.paths.q3_shards[0] == ("/final/q3_k_m/inkling-Q3_K_M-00001-of-00049.gguf")
    assert bundle.paths.q3_shards[-1] == ("/final/q3_k_m/inkling-Q3_K_M-00049-of-00049.gguf")
    assert bundle.paths.shared_projector == "/final/mmproj/mmproj-BF16.gguf"
    assert bundle.paths.q3_export_manifest == "/final/verification/export_manifest.json"
    assert bundle.paths.q3_verify_receipt == "/final/verify_export.success.json"
    assert bundle.paths.q3_quantize_receipt == "/final/quantize_text.success.json"
    assert bundle.paths.projector_conversion_receipt == (
        "/final/convert_multimodal_projector.success.json"
    )
    assert bundle.paths.tokenizer_assets == (
        "/source/snapshot/chat_template.jinja",
        "/source/snapshot/processor_config.json",
        "/source/snapshot/special_tokens_map.json",
        "/source/snapshot/tiktoken/tokenizer.model",
        "/source/snapshot/tokenizer.json",
        "/source/snapshot/tokenizer_config.json",
    )


def test_matched_bundle_rejects_missing_or_tampered_references(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"
    _copy_matched_bundle(missing_root)
    (missing_root / VERIFIED_EXPORT_REFERENCE_RELATIVE_PATH).unlink()
    with pytest.raises(ConfigurationError, match="Unable to load verified"):
        load_matched_cell_bundle(missing_root)

    tampered_root = tmp_path / "tampered"
    _copy_matched_bundle(tampered_root)
    bf16_path = tampered_root / BF16_SUBJECT_REFERENCE_RELATIVE_PATH
    value = json.loads(bf16_path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    value["reference_sha256"] = "0" * 64
    bf16_path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="self-hash"):
        load_matched_cell_bundle(tampered_root)


def test_matched_bundle_reports_source_root_outside_mount_as_storage_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = load_inkling_source_adoption_reference(SOURCE_REFERENCE_PATH)
    detached_root = "/detached/runs/source"
    detached_source = source.model_copy(
        update={
            "source_run_root": detached_root,
            "snapshot_path": f"{detached_root}/snapshot",
        }
    )
    monkeypatch.setattr(
        inkling_matched,
        "load_inkling_source_adoption_reference",
        lambda _path: detached_source,
    )

    with pytest.raises(ConfigurationError, match="one exact subject bundle") as captured:
        load_matched_cell_bundle(PROJECT_ROOT)

    assert captured.value.component == "inkling_matched_cell_bundle"
    assert captured.value.details["mismatches"] == ["source_storage"]


def test_capacity_screen_passes_the_observed_eight_b300_aggregate_cell() -> None:
    config = load_matched_cell_config(MATCHED_CONFIG_PATH)
    bf16 = load_bf16_subject_reference(BF16_REFERENCE_PATH)
    q3 = load_verified_export_reference(Q3_REFERENCE_PATH)

    result = screen_matched_capacity(
        config,
        bf16,
        q3,
        observed_gpu_memory_bytes=(OBSERVED_B300_MEMORY_BYTES,) * 8,
    )

    assert result.status == "pass"
    assert result.observed_total_gpu_memory_bytes == 2_299_430_174_720
    assert result.bf16_subject_bytes == 1_894_278_547_552 + EXPECTED_PROJECTOR_BYTES
    assert result.q3_subject_bytes == EXPECTED_Q3_TOTAL_BYTES + EXPECTED_PROJECTOR_BYTES
    assert result.sequential_peak_subject_bytes == result.bf16_subject_bytes
    assert result.required_headroom_bytes == 229_943_017_472
    assert result.usable_gpu_memory_bytes == 2_069_487_157_248
    assert result.aggregate_remaining_bytes == 175_025_345_408
    assert result.runtime_fit_proven is False
    assert "does not prove" in result.limitation


def test_capacity_screen_rejects_wrong_device_count_and_low_memory() -> None:
    config = load_matched_cell_config(MATCHED_CONFIG_PATH)
    bf16 = load_bf16_subject_reference(BF16_REFERENCE_PATH)
    q3 = load_verified_export_reference(Q3_REFERENCE_PATH)

    with pytest.raises(CapabilityError) as captured:
        screen_matched_capacity(
            config,
            bf16,
            q3,
            observed_gpu_memory_bytes=(OBSERVED_B300_MEMORY_BYTES,) * 2,
        )
    assert captured.value.details["expected_gpu_count"] == 8
    assert captured.value.details["observed_gpu_count"] == 2

    with pytest.raises(CapabilityError) as captured:
        screen_matched_capacity(
            config,
            bf16,
            q3,
            observed_gpu_memory_bytes=(240_000_000_000,) * 8,
        )
    assert (
        captured.value.details["sequential_peak_subject_bytes"]
        > (captured.value.details["usable_gpu_memory_bytes"])
    )

    one_small_gpu = (
        config.resources.minimum_gpu_memory_bytes - 1,
        *(OBSERVED_B300_MEMORY_BYTES,) * 7,
    )
    with pytest.raises(CapabilityError) as captured:
        screen_matched_capacity(
            config,
            bf16,
            q3,
            observed_gpu_memory_bytes=one_small_gpu,
        )
    assert captured.value.details["aggregate_remaining_bytes"] > 0
    assert captured.value.details["below_minimum_cuda_ordinals"] == [0]
