from __future__ import annotations

from pathlib import Path

import pytest

from inkling_quant_lab.artifacts import sha256_file
from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.gguf.inkling import (
    EXPECTED_WEIGHT_INDEX_CANONICAL_SHA256,
    InklingGGUFConfig,
    InklingSourceAudit,
)
from inkling_quant_lab.gguf.workflow import (
    STAGE_ORDER,
    initial_manifest,
    next_pending_stage,
    output_record,
    stage_fingerprint,
    verify_completed_stage,
)
from inkling_quant_lab.manifests import ArtifactChecksum, RunManifest


def _audit() -> InklingSourceAudit:
    return InklingSourceAudit(
        model_id="thinkingmachines/Inkling",
        revision="86b4d430ab871652a707666b89203a866888c5e5",
        architecture="InklingForConditionalGeneration",
        model_type="inkling_mm_model",
        license="apache-2.0",
        source_bytes=1_904_604_285_204,
        source_tensor_count=1_552,
        source_shard_count=109,
        text_tensor_count=1_382,
        vision_tensor_count=8,
        audio_tensor_count=2,
        mtp_tensor_count=160,
        converted_source_tensor_count=1_392,
        omitted_source_tensor_count=160,
        weight_index_sha256=EXPECTED_WEIGHT_INDEX_CANONICAL_SHA256,
        verified=True,
        warnings=("MTP is omitted because the pinned runtime does not support it.",),
    )


def _completed_preflight(
    run_directory: Path,
) -> tuple[InklingGGUFConfig, RunManifest, Path, ArtifactChecksum]:
    config = InklingGGUFConfig()
    manifest = initial_manifest(config, _audit(), run_id="inkling-test").start()
    fingerprint = stage_fingerprint("preflight", config, ())
    manifest = manifest.start_stage("preflight", fingerprint)
    evidence = run_directory / "source_audit.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text("verified\n", encoding="utf-8")
    output = output_record(run_directory, evidence)
    manifest = manifest.finish_stage("preflight", (output,))
    return config, manifest, evidence, output


def test_external_manifest_has_an_isolated_ordered_resume_dag() -> None:
    config = InklingGGUFConfig()
    manifest = initial_manifest(config, _audit(), run_id="inkling-test")

    assert tuple(manifest.stages) == STAGE_ORDER
    assert next_pending_stage(manifest) == "preflight"
    assert manifest.model.id == "thinkingmachines/Inkling"
    assert manifest.model.revision == config.source.revision


def test_completed_stage_is_skipped_only_after_checksum_verification(tmp_path: Path) -> None:
    config = InklingGGUFConfig()
    manifest = initial_manifest(config, _audit(), run_id="inkling-test").start()
    fingerprint = stage_fingerprint("preflight", config, ())
    manifest = manifest.start_stage("preflight", fingerprint)
    evidence = tmp_path / "source_audit.json"
    evidence.write_text("verified\n", encoding="utf-8")
    output = output_record(tmp_path, evidence)
    manifest = manifest.finish_stage("preflight", (output,))

    verify_completed_stage(tmp_path, manifest.stages["preflight"], fingerprint)
    assert (
        next_pending_stage(manifest, run_directory=tmp_path, config=config) == "materialize_source"
    )

    evidence.write_text("mutated\n", encoding="utf-8")
    assert sha256_file(evidence) != output.sha256
    with pytest.raises(ArtifactIntegrityError, match="Checksum mismatch"):
        verify_completed_stage(tmp_path, manifest.stages["preflight"], fingerprint)


def test_resume_rejects_stale_stage_fingerprint(tmp_path: Path) -> None:
    config = InklingGGUFConfig()
    manifest = initial_manifest(config, _audit(), run_id="inkling-test").start()
    manifest = manifest.start_stage("preflight", "0" * 64)
    evidence = tmp_path / "source_audit.json"
    evidence.write_text("verified\n", encoding="utf-8")
    manifest = manifest.finish_stage("preflight", (output_record(tmp_path, evidence),))

    with pytest.raises(ArtifactIntegrityError, match="fingerprint mismatch"):
        verify_completed_stage(
            tmp_path,
            manifest.stages["preflight"],
            stage_fingerprint("preflight", config, ()),
        )


def test_resume_rejects_missing_completed_output(tmp_path: Path) -> None:
    config, manifest, evidence, _ = _completed_preflight(tmp_path)
    evidence.unlink()

    with pytest.raises(ArtifactIntegrityError, match="Missing or unsafe output"):
        next_pending_stage(manifest, run_directory=tmp_path, config=config)


def test_resume_rejects_success_stage_with_empty_output_list(tmp_path: Path) -> None:
    config, manifest, _, _ = _completed_preflight(tmp_path)
    stage = manifest.stages["preflight"].model_copy(update={"outputs": ()})

    with pytest.raises(ArtifactIntegrityError, match="no recorded outputs"):
        verify_completed_stage(
            tmp_path,
            stage,
            stage_fingerprint("preflight", config, ()),
        )


def test_resume_rejects_output_replaced_by_symlink(tmp_path: Path) -> None:
    config, manifest, evidence, _ = _completed_preflight(tmp_path)
    target = tmp_path / "same-content-target.json"
    target.write_text("verified\n", encoding="utf-8")
    evidence.unlink()
    evidence.symlink_to(target.name)

    with pytest.raises(ArtifactIntegrityError, match="Missing or unsafe output"):
        next_pending_stage(manifest, run_directory=tmp_path, config=config)


def test_resume_rejects_escaping_output_record(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    config, manifest, _, output = _completed_preflight(run_directory)
    outside = tmp_path / "outside.json"
    outside.write_text("verified\n", encoding="utf-8")
    escaped = output.model_copy(update={"path": "../outside.json"})
    stage = manifest.stages["preflight"].model_copy(update={"outputs": (escaped,)})

    with pytest.raises(ArtifactIntegrityError, match="Missing or unsafe output"):
        verify_completed_stage(
            run_directory,
            stage,
            stage_fingerprint("preflight", config, ()),
        )


def test_resume_rejects_dependency_checksum_drift(tmp_path: Path) -> None:
    config, manifest, _, preflight_output = _completed_preflight(tmp_path)
    materialize_fingerprint = stage_fingerprint(
        "materialize_source", config, (preflight_output.sha256,)
    )
    manifest = manifest.start_stage("materialize_source", materialize_fingerprint)
    source_receipt = tmp_path / "source.success.json"
    source_receipt.write_text("source verified\n", encoding="utf-8")
    manifest = manifest.finish_stage(
        "materialize_source", (output_record(tmp_path, source_receipt),)
    )

    replacement = tmp_path / "replacement_audit.json"
    replacement.write_text("different but valid evidence\n", encoding="utf-8")
    drifted_preflight = manifest.stages["preflight"].model_copy(
        update={"outputs": (output_record(tmp_path, replacement),)}
    )
    stages = dict(manifest.stages)
    stages["preflight"] = drifted_preflight
    manifest = manifest.model_copy(update={"stages": stages})

    with pytest.raises(ArtifactIntegrityError, match="fingerprint mismatch"):
        next_pending_stage(manifest, run_directory=tmp_path, config=config)


@pytest.mark.parametrize(
    "context",
    ({}, {"run_directory": "present"}, {"config": "present"}),
)
def test_resume_rejects_skipping_completed_stage_without_full_context(
    tmp_path: Path, context: dict[str, str]
) -> None:
    config, manifest, _, _ = _completed_preflight(tmp_path)
    arguments: dict[str, object] = {}
    if "run_directory" in context:
        arguments["run_directory"] = tmp_path
    if "config" in context:
        arguments["config"] = config

    with pytest.raises(ArtifactIntegrityError, match="required to skip"):
        next_pending_stage(manifest, **arguments)  # type: ignore[arg-type]
