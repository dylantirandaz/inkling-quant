"""Strict exact-cell compatibility matrix and CLI behavior."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from inkling_quant_lab.cli import app
from inkling_quant_lab.comparison import (
    DatasetIdentity,
    EvaluationSuiteIdentity,
    MetricValue,
    NormalizedRunSummary,
    SoftwareToolchainIdentity,
)
from inkling_quant_lab.compatibility import (
    COMPATIBILITY_SCOPE_WARNING,
    CompatibilityArtifact,
    CompatibilityCell,
    CompatibilityClaim,
    CompatibilityClaims,
    CompatibilityEvidenceReference,
    CompatibilityMatrixPayload,
    CompatibilityMatrixRecord,
    CudaIdentity,
    GpuDeviceIdentity,
    HardwareIdentity,
    ModelIdentity,
    ProtocolIdentity,
    RuntimeBinaryIdentity,
    RuntimeIdentity,
    SoftwareIdentity,
    inspect_compatibility_matrix,
    load_compatibility_matrix,
    seal_compatibility_matrix,
    validate_compatibility_matrix,
)
from inkling_quant_lab.evidence import (
    EvidenceArtifactIdentity,
    ExperimentEvidencePayload,
    ExperimentEvidenceRecord,
    seal_evidence_record,
)
from inkling_quant_lab.exceptions import ArtifactIntegrityError

pytestmark = pytest.mark.unit

runner = CliRunner()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKED_MATRIX = PROJECT_ROOT / "configs/compatibility/inkling_q3_k_m.json"


def _claim(
    status: str,
    *,
    evidence_refs: tuple[str, ...] = (),
    reason: str = "Bound only to this exact matrix cell.",
) -> dict[str, object]:
    return {"status": status, "evidence_refs": evidence_refs, "reason": reason}


def _cell_dict() -> dict[str, object]:
    return {
        "cell_id": "inkling-q3-structural-modal-cpu-v1",
        "execution_role": "structural_verification",
        "model": {
            "model_id": "thinkingmachines/Inkling",
            "revision": "86b4d430ab871652a707666b89203a866888c5e5",
        },
        "artifacts": (
            {
                "artifact_id": "export-manifest",
                "kind": "export_manifest",
                "location": "verification/export_manifest.json",
                "sha256": "1" * 64,
            },
            {
                "artifact_id": "projector",
                "kind": "multimodal_projector",
                "location": "mmproj/mmproj-BF16.gguf",
                "sha256": "2" * 64,
            },
        ),
        "runtime": {
            "name": "llama.cpp-export-verifier",
            "repository": "https://github.com/danielhanchen/llama.cpp.git",
            "commit": "a015409e6c27b84f60d688823d4c0126a11571fd",
        },
        "software": {
            "container_image": (
                "debian:bookworm-slim@sha256:"
                "7b140f374b289a7c2befc338f42ebe6441b7ea838a042bbd5acbfca6ec875818"
            ),
            "identity_kind": "control_plane_tree",
            "software_identity_sha256": "3" * 64,
            "cuda": {"status": "absent", "version": None},
        },
        "hardware": {"provider": "modal", "gpu_model": "none", "gpu_count": 0},
        "evidence": (
            {
                "reference_id": "verified-export",
                "kind": "structural_verification",
                "location": "configs/experiments/verified-export.json",
                "sha256": "4" * 64,
            },
        ),
        "claims": {
            "structurally_verified": _claim("supported", evidence_refs=("verified-export",)),
            "smoke_passed": _claim("not_measured"),
            "quality_measured": _claim("not_measured"),
            "performance_measured": _claim("not_measured"),
            "mtp_supported": _claim("unsupported"),
        },
    }


def _complete_cell_dict(
    *,
    evidence_location: str = "evidence/verified-export.json",
    evidence_sha256: str = "4" * 64,
) -> dict[str, object]:
    raw = _cell_dict()
    raw["cell_id"] = "inkling-q3-deployment-modal-b300-v1"
    raw["execution_role"] = "deployment"
    raw["protocol"] = {
        "protocol_id": "inkling-structural-verification-v1",
        "protocol_sha256": "5" * 64,
    }
    raw["software"] = {
        "container_image": "cuda@test@sha256:" + "6" * 64,
        "identity_kind": "runtime_receipt",
        "software_identity_sha256": "7" * 64,
        "cuda": {"status": "present", "version": "13.1.2"},
        "driver_version": "590.48.01",
        "runtime_binaries": (
            {"name": "llama-cli", "sha256": "8" * 64},
            {"name": "llama-server", "sha256": "9" * 64},
        ),
        "build_flags": ("GGML_CUDA=ON",),
        "package_manifest_sha256": "a" * 64,
    }
    raw["hardware"] = {
        "provider": "modal",
        "gpu_model": "NVIDIA B300",
        "gpu_count": 1,
        "gpus": (
            {
                "uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                "model": "NVIDIA B300",
                "memory_bytes": 288_000_000_000,
                "compute_capability": "10.0",
            },
        ),
        "cpu_model": "AMD EPYC 9654",
        "logical_cpu_count": 192,
        "ram_bytes": 2_199_023_255_552,
        "topology_sha256": "b" * 64,
    }
    raw["evidence"] = (
        {
            "reference_id": "verified-export",
            "kind": "structural_verification",
            "location": evidence_location,
            "sha256": evidence_sha256,
        },
    )
    return raw


def _terminal_summary(
    cell: CompatibilityCell,
    *,
    failures: tuple[str, ...] = (),
) -> NormalizedRunSummary:
    suite = EvaluationSuiteIdentity(
        evaluator_name="structural_verifier",
        evaluator_version="fixture-v1",
        dataset_id="fixture://export-inventory",
        dataset_revision="fixture-v1",
        split="inventory",
        dataset_sha256="c" * 64,
        sample_ids=("manifest",),
        seed=17,
        prompt_template_hash="d" * 64,
        decode_config={"deterministic": True},
        status="success",
    )
    return NormalizedRunSummary(
        schema_version="1.2",
        run_id="verified-export-run",
        artifact_path="artifacts/verified-export-run",
        model_id=cell.model.model_id,
        model_revision=cell.model.revision,
        source_model_sha256="e" * 64,
        software_toolchain=SoftwareToolchainIdentity(
            runtime_name=cell.runtime.name,
            runtime_revision=cell.runtime.commit,
            runtime_binary_sha256="8" * 64,
            container_image_digest=cell.software.container_image.rsplit("@", maxsplit=1)[-1],
            packages={"llama.cpp": cell.runtime.commit},
            build_options=("GGML_CUDA=ON",),
        ),
        datasets=(
            DatasetIdentity(
                dataset_id="fixture://export-inventory",
                dataset_revision="fixture-v1",
                split="inventory",
                dataset_sha256="c" * 64,
            ),
        ),
        seed_set=(17,),
        sample_ids=("manifest",),
        prompt_template_hash="d" * 64,
        decode_config={"deterministic": True},
        evaluation_suites=(suite,),
        benchmark_protocol_version="structural-verification-v1",
        hardware_environment={
            "hardware": cell.hardware.model_dump(mode="json"),
            "runtime": {"backend": cell.runtime.name},
        },
        metrics={
            "structural_verification": MetricValue(
                value=1.0,
                category="other",
                direction="neutral",
                evaluation_suite=suite,
            ),
        },
        failures=failures,
    )


def _terminal_record(
    cell: CompatibilityCell,
    *,
    status: Literal["success", "partial", "failed"] = "success",
    failures: tuple[str, ...] = (),
    scope_sha256: str | None = None,
) -> ExperimentEvidenceRecord:
    artifacts = tuple(
        sorted(
            (
                EvidenceArtifactIdentity(
                    artifact_id="export-manifest",
                    role="auxiliary",
                    location="verification/export_manifest.json",
                    sha256=cell.artifacts[0].sha256,
                    size_bytes=1024,
                ),
                EvidenceArtifactIdentity(
                    artifact_id="model-export",
                    role="model_export",
                    location="q3/model.gguf",
                    sha256="f" * 64,
                    size_bytes=4096,
                ),
                EvidenceArtifactIdentity(
                    artifact_id="projector",
                    role="projector",
                    location="mmproj/mmproj.gguf",
                    sha256=cell.artifacts[1].sha256,
                    size_bytes=2048,
                ),
            ),
            key=lambda item: (item.role, item.artifact_id, item.location),
        )
    )
    payload = ExperimentEvidencePayload(
        record_id="verified-export-evidence",
        result_kind="structural_verification",
        status=status,
        comparison_role="standalone",
        compatibility_scope_sha256=scope_sha256 or cell.exact_scope_sha256(),
        subject=_terminal_summary(cell, failures=failures),
        artifacts=artifacts,
        limitations=("This result applies only to the exact compatibility cell.",),
    )
    return seal_evidence_record(payload)


def _write_valid_bundle(tmp_path: Path) -> Path:
    raw = _complete_cell_dict()
    cell = CompatibilityCell.model_validate(raw)
    evidence_path = tmp_path / "evidence" / "verified-export.json"
    evidence_path.parent.mkdir()
    evidence_path.write_text(
        _terminal_record(cell).model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    raw = _complete_cell_dict(evidence_sha256=evidence_sha256)
    final_cell = CompatibilityCell.model_validate(raw)
    assert final_cell.exact_scope_sha256() == cell.exact_scope_sha256()
    record = seal_compatibility_matrix(
        CompatibilityMatrixPayload(matrix_id="fixture-matrix-v1", cells=(final_cell,))
    )
    return _write_matrix(tmp_path / "matrix.json", record)


def _record() -> CompatibilityMatrixRecord:
    payload = CompatibilityMatrixPayload(
        matrix_id="fixture-matrix-v1",
        cells=(CompatibilityCell.model_validate(_cell_dict()),),
    )
    return seal_compatibility_matrix(payload)


def _write_matrix(path: Path, record: CompatibilityMatrixRecord) -> Path:
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def test_matrix_round_trip_preserves_five_distinct_claims(tmp_path: Path) -> None:
    matrix = load_compatibility_matrix(_write_valid_bundle(tmp_path))

    inspection = inspect_compatibility_matrix(matrix)
    validation = validate_compatibility_matrix(matrix)
    claims = inspection["cells"][0]["claims"]

    assert claims["structurally_verified"]["status"] == "supported"
    assert claims["smoke_passed"]["status"] == "not_measured"
    assert claims["quality_measured"]["status"] == "not_measured"
    assert claims["performance_measured"]["status"] == "not_measured"
    assert claims["mtp_supported"]["status"] == "unsupported"
    assert validation["status"] == "valid"
    assert validation["referenced_bytes_verified"] is True
    assert validation["reference_results"][0]["resolution"] == "verified_local"
    assert validation["warnings"] == [COMPATIBILITY_SCOPE_WARNING]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model.model_id", "*"),
        ("runtime.name", "any"),
        ("hardware.gpu_model", "unknown"),
    ],
)
def test_cell_rejects_wildcard_or_unspecified_scope(field: str, value: str) -> None:
    raw = _cell_dict()
    first, second = field.split(".")
    nested = raw[first]
    assert isinstance(nested, dict)
    nested[second] = value

    with pytest.raises(ValidationError, match="exact value"):
        CompatibilityCell.model_validate(raw)


def test_supported_smoke_requires_structural_support() -> None:
    raw = _cell_dict()
    claims = raw["claims"]
    assert isinstance(claims, dict)
    evidence = raw["evidence"]
    assert isinstance(evidence, tuple)
    raw["execution_role"] = "deployment"
    raw["software"] = {
        "container_image": "cuda@test@sha256:" + "5" * 64,
        "identity_kind": "runtime_receipt",
        "software_identity_sha256": "6" * 64,
        "cuda": {"status": "present", "version": "13.1.2"},
    }
    raw["hardware"] = {"provider": "modal", "gpu_model": "B300", "gpu_count": 2}
    raw["evidence"] = (
        {
            "reference_id": "smoke",
            "kind": "inference_smoke",
            "location": "evidence/smoke.json",
            "sha256": "7" * 64,
        },
        *evidence,
    )
    claims["structurally_verified"] = _claim("not_measured")
    claims["smoke_passed"] = _claim("supported", evidence_refs=("smoke",))

    with pytest.raises(ValidationError, match="without structural verification"):
        CompatibilityCell.model_validate(raw)


def test_supported_claim_requires_matching_evidence_kind() -> None:
    raw = _cell_dict()
    raw["execution_role"] = "deployment"
    raw["software"] = {
        "container_image": "cuda@test@sha256:" + "5" * 64,
        "identity_kind": "runtime_receipt",
        "software_identity_sha256": "6" * 64,
        "cuda": {"status": "present", "version": "13.1.2"},
    }
    raw["hardware"] = {"provider": "modal", "gpu_model": "B300", "gpu_count": 2}
    claims = raw["claims"]
    assert isinstance(claims, dict)
    claims["smoke_passed"] = _claim("supported", evidence_refs=("verified-export",))

    with pytest.raises(ValidationError, match="smoke_passed evidence"):
        CompatibilityCell.model_validate(raw)


def test_optional_unsupported_evidence_must_still_have_the_relevant_kind() -> None:
    raw = _cell_dict()
    claims = raw["claims"]
    assert isinstance(claims, dict)
    claims["mtp_supported"] = _claim(
        "unsupported",
        evidence_refs=("verified-export",),
    )

    cell = CompatibilityCell.model_validate(raw)
    assert cell.claims.mtp_supported.status == "unsupported"

    claims["smoke_passed"] = _claim(
        "unsupported",
        evidence_refs=("verified-export",),
    )
    with pytest.raises(ValidationError, match="smoke_passed evidence"):
        CompatibilityCell.model_validate(raw)


def test_unsupported_and_not_measured_claims_need_no_evidence_receipts() -> None:
    raw = _cell_dict()
    raw["evidence"] = ()
    raw["claims"] = {
        "structurally_verified": _claim("unsupported"),
        "smoke_passed": _claim("not_measured"),
        "quality_measured": _claim("not_measured"),
        "performance_measured": _claim("not_measured"),
        "mtp_supported": _claim("unsupported"),
    }

    cell = CompatibilityCell.model_validate(raw)

    assert cell.evidence == ()
    assert cell.claims.structurally_verified.status == "unsupported"
    assert cell.claims.mtp_supported.status == "unsupported"


def test_complete_matrix_with_no_supported_claims_validates_without_receipts(
    tmp_path: Path,
) -> None:
    raw = _complete_cell_dict()
    raw["evidence"] = ()
    raw["claims"] = {
        "structurally_verified": _claim("unsupported"),
        "smoke_passed": _claim("not_measured"),
        "quality_measured": _claim("not_measured"),
        "performance_measured": _claim("not_measured"),
        "mtp_supported": _claim("unsupported"),
    }
    cell = CompatibilityCell.model_validate(raw)
    path = _write_matrix(
        tmp_path / "matrix.json",
        seal_compatibility_matrix(
            CompatibilityMatrixPayload(matrix_id="unsupported-only-v1", cells=(cell,))
        ),
    )

    result = validate_compatibility_matrix(load_compatibility_matrix(path))

    assert result["status"] == "valid"
    assert result["reference_results"] == []
    assert result["referenced_bytes_verified"] is True


def test_mtp_support_requires_smoke_and_mtp_artifact() -> None:
    raw = _cell_dict()
    raw["execution_role"] = "deployment"
    raw["software"] = {
        "container_image": "cuda@test@sha256:" + "5" * 64,
        "identity_kind": "runtime_receipt",
        "software_identity_sha256": "6" * 64,
        "cuda": {"status": "present", "version": "13.1.2"},
    }
    raw["hardware"] = {"provider": "modal", "gpu_model": "B300", "gpu_count": 2}
    evidence = raw["evidence"]
    assert isinstance(evidence, tuple)
    raw["evidence"] = (
        {
            "reference_id": "mtp",
            "kind": "mtp_validation",
            "location": "evidence/mtp.json",
            "sha256": "8" * 64,
        },
        *evidence,
    )
    claims = raw["claims"]
    assert isinstance(claims, dict)
    claims["mtp_supported"] = _claim("supported", evidence_refs=("mtp",))

    with pytest.raises(ValidationError, match="before smoke_passed"):
        CompatibilityCell.model_validate(raw)


def test_loader_rejects_tampered_payload_and_duplicate_keys(tmp_path: Path) -> None:
    path = _write_matrix(tmp_path / "matrix.json", _record())
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["payload"]["cells"][0]["hardware"]["provider"] = "other"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        load_compatibility_matrix(path)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":"iql-compatibility-matrix-v1",'
        '"schema_version":"iql-compatibility-matrix-v1"}',
        encoding="utf-8",
    )
    with pytest.raises(ArtifactIntegrityError, match="duplicate JSON object key"):
        load_compatibility_matrix(duplicate)


def test_validation_rejects_incomplete_full_hardware_software_and_protocol_scope(
    tmp_path: Path,
) -> None:
    matrix = load_compatibility_matrix(_write_matrix(tmp_path / "matrix.json", _record()))

    with pytest.raises(ArtifactIntegrityError, match="lacks complete exact scope") as captured:
        validate_compatibility_matrix(matrix)

    missing = captured.value.details["missing"]
    assert "protocol identity is missing" in missing
    assert "runtime binary hashes are missing" in missing
    assert "package manifest SHA-256 is missing" in missing
    assert "CPU model is missing" in missing
    assert "RAM capacity is missing" in missing


def test_cuda_driver_scope_is_required_only_for_cuda_cells() -> None:
    with pytest.raises(ValidationError, match="cannot declare a GPU driver"):
        SoftwareIdentity(
            container_image="image@sha256:" + "e" * 64,
            identity_kind="runtime_receipt",
            software_identity_sha256="f" * 64,
            cuda=CudaIdentity(status="absent", version=None),
            driver_version="590.48.01",
        )


def test_supported_claim_cannot_use_unresolved_external_evidence(tmp_path: Path) -> None:
    cell = CompatibilityCell.model_validate(
        _complete_cell_dict(evidence_location="modal-volume://receipts/verified-export.json")
    )
    path = _write_matrix(
        tmp_path / "matrix.json",
        seal_compatibility_matrix(
            CompatibilityMatrixPayload(matrix_id="external-matrix-v1", cells=(cell,))
        ),
    )

    with pytest.raises(ArtifactIntegrityError, match="unresolved evidence"):
        validate_compatibility_matrix(load_compatibility_matrix(path))


@pytest.mark.parametrize("location", ["../outside.json", "/tmp/outside.json"])
def test_local_evidence_paths_cannot_escape_repository(
    tmp_path: Path,
    location: str,
) -> None:
    cell = CompatibilityCell.model_validate(_complete_cell_dict(evidence_location=location))
    path = _write_matrix(
        tmp_path / "matrix.json",
        seal_compatibility_matrix(
            CompatibilityMatrixPayload(matrix_id="escaping-matrix-v1", cells=(cell,))
        ),
    )

    with pytest.raises(ArtifactIntegrityError, match="escapes its repository"):
        validate_compatibility_matrix(load_compatibility_matrix(path))


def test_local_evidence_symlink_is_rejected(tmp_path: Path) -> None:
    cell = CompatibilityCell.model_validate(
        _complete_cell_dict(evidence_location="evidence/link.json")
    )
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    target = evidence_dir / "target.json"
    target.write_text(_terminal_record(cell).model_dump_json(indent=2) + "\n", encoding="utf-8")
    (evidence_dir / "link.json").symlink_to(target)
    raw = _complete_cell_dict(
        evidence_location="evidence/link.json",
        evidence_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )
    path = _write_matrix(
        tmp_path / "matrix.json",
        seal_compatibility_matrix(
            CompatibilityMatrixPayload(
                matrix_id="symlink-matrix-v1",
                cells=(CompatibilityCell.model_validate(raw),),
            )
        ),
    )

    with pytest.raises(ArtifactIntegrityError, match="symbolic link"):
        validate_compatibility_matrix(load_compatibility_matrix(path))


def test_local_evidence_bytes_and_terminal_success_are_required(tmp_path: Path) -> None:
    initial = CompatibilityCell.model_validate(_complete_cell_dict())
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    evidence_path = evidence_dir / "verified-export.json"
    evidence_path.write_text(
        _terminal_record(
            initial,
            status="failed",
            failures=("verification command failed",),
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    evidence_sha256 = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    cell = CompatibilityCell.model_validate(_complete_cell_dict(evidence_sha256=evidence_sha256))
    path = _write_matrix(
        tmp_path / "matrix.json",
        seal_compatibility_matrix(
            CompatibilityMatrixPayload(matrix_id="failed-evidence-v1", cells=(cell,))
        ),
    )

    with pytest.raises(ArtifactIntegrityError, match="does not match cell") as captured:
        validate_compatibility_matrix(load_compatibility_matrix(path))
    assert "terminal outcome" in captured.value.details["mismatches"][0]

    wrong_reference = cell.evidence[0].model_copy(update={"sha256": "0" * 64})
    wrong_cell = cell.model_copy(update={"evidence": (wrong_reference,)})
    wrong_path = _write_matrix(
        tmp_path / "wrong-sha.json",
        seal_compatibility_matrix(
            CompatibilityMatrixPayload(matrix_id="wrong-sha-v1", cells=(wrong_cell,))
        ),
    )
    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        validate_compatibility_matrix(load_compatibility_matrix(wrong_path))


def _mutate_scope(cell: CompatibilityCell, dimension: str) -> CompatibilityCell:
    if dimension == "protocol":
        assert cell.protocol is not None
        return cell.model_copy(
            update={"protocol": cell.protocol.model_copy(update={"protocol_sha256": "0" * 64})}
        )
    if dimension == "driver":
        return cell.model_copy(
            update={"software": cell.software.model_copy(update={"driver_version": "591.00"})}
        )
    if dimension == "runtime_binary":
        binary = cell.software.runtime_binaries[0].model_copy(update={"sha256": "0" * 64})
        return cell.model_copy(
            update={
                "software": cell.software.model_copy(
                    update={"runtime_binaries": (binary, *cell.software.runtime_binaries[1:])}
                )
            }
        )
    if dimension == "build_flags":
        return cell.model_copy(
            update={
                "software": cell.software.model_copy(update={"build_flags": ("GGML_CUDA=OFF",)})
            }
        )
    if dimension == "package_manifest":
        return cell.model_copy(
            update={
                "software": cell.software.model_copy(update={"package_manifest_sha256": "0" * 64})
            }
        )
    if dimension in {"gpu_uuid", "gpu_memory", "gpu_compute_capability"}:
        gpu_updates: dict[str, object] = {
            "gpu_uuid": {"uuid": "GPU-ffffffff-ffff-ffff-ffff-ffffffffffff"},
            "gpu_memory": {"memory_bytes": 287_000_000_000},
            "gpu_compute_capability": {"compute_capability": "9.0"},
        }[dimension]
        gpu = cell.hardware.gpus[0].model_copy(update=gpu_updates)
        return cell.model_copy(
            update={"hardware": cell.hardware.model_copy(update={"gpus": (gpu,)})}
        )
    hardware_field, hardware_value = {
        "cpu": ("cpu_model", "AMD EPYC 9754"),
        "ram": ("ram_bytes", 2_000_000_000_000),
        "topology": ("topology_sha256", "0" * 64),
    }[dimension]
    return cell.model_copy(
        update={"hardware": cell.hardware.model_copy(update={hardware_field: hardware_value})}
    )


@pytest.mark.parametrize(
    "dimension",
    [
        "protocol",
        "driver",
        "runtime_binary",
        "build_flags",
        "package_manifest",
        "gpu_uuid",
        "gpu_memory",
        "gpu_compute_capability",
        "cpu",
        "ram",
        "topology",
    ],
)
def test_terminal_evidence_is_bound_to_every_exact_cell_dimension(
    tmp_path: Path,
    dimension: str,
) -> None:
    original_path = _write_valid_bundle(tmp_path)
    original = load_compatibility_matrix(original_path).record.payload.cells[0]
    mutated = _mutate_scope(original, dimension)
    mutated_path = _write_matrix(
        tmp_path / f"matrix-{dimension}.json",
        seal_compatibility_matrix(
            CompatibilityMatrixPayload(matrix_id=f"mutated-{dimension}-v1", cells=(mutated,))
        ),
    )

    with pytest.raises(ArtifactIntegrityError, match="does not match cell") as captured:
        validate_compatibility_matrix(load_compatibility_matrix(mutated_path))
    assert any(
        "scope SHA-256 differs" in mismatch for mismatch in captured.value.details["mismatches"]
    )


def test_compatibility_cli_is_read_only_and_emits_one_json_document(tmp_path: Path) -> None:
    path = _write_valid_bundle(tmp_path)
    before = {item.relative_to(tmp_path) for item in tmp_path.rglob("*")}

    inspection = runner.invoke(app, ["compatibility", "inspect", str(path), "--json"])
    validation = runner.invoke(app, ["compatibility", "validate", str(path), "--json"])

    after = {item.relative_to(tmp_path) for item in tmp_path.rglob("*")}
    assert inspection.exit_code == 0, inspection.output
    assert validation.exit_code == 0, validation.output
    assert json.loads(inspection.stdout)["status"] == "inspected"
    assert json.loads(validation.stdout)["status"] == "valid"
    assert len(inspection.stdout.strip().splitlines()) == 1
    assert len(validation.stdout.strip().splitlines()) == 1
    assert after == before


def test_checked_inkling_q3_matrix_is_structural_only() -> None:
    matrix = load_compatibility_matrix(CHECKED_MATRIX)
    cell = matrix.record.payload.cells[0]

    assert cell.model.model_id == "thinkingmachines/Inkling"
    assert cell.model.revision == "86b4d430ab871652a707666b89203a866888c5e5"
    assert cell.claims.structurally_verified.status == "supported"
    assert cell.claims.smoke_passed.status == "not_measured"
    assert cell.claims.quality_measured.status == "not_measured"
    assert cell.claims.performance_measured.status == "not_measured"
    assert cell.claims.mtp_supported.status == "unsupported"
    assert cell.artifacts[0].sha256 == (
        "23db1314d521210bab5d53df20ed432f784774c59d98e8db3de9004702e1ac7a"
    )
    assert cell.artifacts[1].sha256 == (
        "8f954d089a753671321316bd4fbcffae6465748814ca6c1ec3e70f62427514f7"
    )
    assert cell.runtime.commit == "a015409e6c27b84f60d688823d4c0126a11571fd"
    assert cell.software.software_identity_sha256 == (
        "8083cf41e104b3f7164c02a1ad50ab027f630167970c4eb7e0589a6d079c1037"
    )
    assert cell.evidence[0].sha256 == (
        "1086fec05c9b6b4400caf9be21cbbbb8e6e5c4138164c5e02af010359f84ad96"
    )
    assert cell.evidence[1].sha256 == (
        "08b4928333720962e1192ef0af12672c8155c70ddc03813376cbd431c2409291"
    )
    assert cell.hardware.gpu_model == "none"
    assert cell.hardware.gpu_count == 0
    assert matrix.record.payload.scope_policy == "exact_cell_only"


def test_public_models_are_strict_and_immutable() -> None:
    artifact = CompatibilityArtifact(
        artifact_id="manifest",
        kind="export_manifest",
        location="manifest.json",
        sha256="a" * 64,
    )
    evidence = CompatibilityEvidenceReference(
        reference_id="verify",
        kind="structural_verification",
        location="verify.json",
        sha256="b" * 64,
    )
    claim = CompatibilityClaim(
        status="supported",
        evidence_refs=("verify",),
        reason="Exact evidence.",
    )
    claims = CompatibilityClaims(
        structurally_verified=claim,
        smoke_passed=CompatibilityClaim(status="not_measured", reason="Not run."),
        quality_measured=CompatibilityClaim(status="not_measured", reason="Not run."),
        performance_measured=CompatibilityClaim(status="not_measured", reason="Not run."),
        mtp_supported=CompatibilityClaim(status="unsupported", reason="Omitted."),
    )

    assert ModelIdentity(model_id="owner/model", revision="c" * 40).model_id == "owner/model"
    assert (
        RuntimeIdentity(
            name="runtime", repository="https://example.test/repo", commit="d" * 40
        ).name
        == "runtime"
    )
    assert ProtocolIdentity(protocol_id="smoke-v1", protocol_sha256="1" * 64).protocol_id == (
        "smoke-v1"
    )
    assert RuntimeBinaryIdentity(name="llama-cli", sha256="2" * 64).name == "llama-cli"
    assert (
        SoftwareIdentity(
            container_image="image@sha256:" + "e" * 64,
            identity_kind="runtime_receipt",
            software_identity_sha256="f" * 64,
            cuda=CudaIdentity(status="absent", version=None),
        ).cuda.status
        == "absent"
    )
    assert HardwareIdentity(provider="local", gpu_model="none", gpu_count=0).gpu_count == 0
    assert (
        GpuDeviceIdentity(
            uuid="GPU-1234",
            model="NVIDIA B300",
            memory_bytes=288_000_000_000,
            compute_capability="10.0",
        ).compute_capability
        == "10.0"
    )
    assert claims.structurally_verified == claim
    assert artifact.sha256 == "a" * 64
    assert evidence.sha256 == "b" * 64
