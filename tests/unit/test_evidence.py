"""Strict evidence schema, loading, comparison, and CLI behavior."""

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
    compare_summaries,
)
from inkling_quant_lab.evidence import (
    EVIDENCE_SCOPE_WARNING,
    CandidateQuantizationIdentity,
    ComparisonArtifactIdentity,
    EvidenceArtifactIdentity,
    EvidenceComparisonProtocol,
    ExperimentEvidencePayload,
    ExperimentEvidenceRecord,
    compare_evidence_records,
    evidence_comparison_payload,
    inspect_evidence_record,
    load_evidence_record,
    seal_evidence_record,
    validate_evidence_record,
)
from inkling_quant_lab.exceptions import ArtifactIntegrityError, ComparisonCompatibilityError

pytestmark = pytest.mark.unit

runner = CliRunner()


def _summary(
    run_id: str,
    *,
    quality: float = 0.8,
    source_model_sha256: str = "a" * 64,
    runtime_binary_sha256: str = "b" * 64,
    quantization_policy: dict[str, object] | None = None,
    failures: tuple[str, ...] = (),
) -> NormalizedRunSummary:
    suite = EvaluationSuiteIdentity(
        evaluator_name="fixture_quality",
        evaluator_version="fixture-v1",
        dataset_id="fixture://evaluation",
        dataset_revision="fixture-data-v1",
        split="test",
        dataset_sha256="c" * 64,
        sample_ids=("sample-1", "sample-2"),
        seed=17,
        prompt_template_hash="d" * 64,
        decode_config={"do_sample": False, "max_new_tokens": 4},
        status="success",
    )
    return NormalizedRunSummary(
        schema_version="1.2",
        run_id=run_id,
        artifact_path=f"artifacts/{run_id}",
        model_id="thinkingmachines/Inkling",
        model_revision="86b4d430ab871652a707666b89203a866888c5e5",
        source_model_sha256=source_model_sha256,
        software_toolchain=SoftwareToolchainIdentity(
            runtime_name="llama.cpp",
            runtime_revision="0123456789abcdef",
            runtime_binary_sha256=runtime_binary_sha256,
            container_image_digest="sha256:" + "e" * 64,
            packages={"cuda": "13.0", "llama.cpp": "0123456789abcdef"},
            build_options=("GGML_CUDA=ON",),
        ),
        datasets=(
            DatasetIdentity(
                dataset_id="fixture://evaluation",
                dataset_revision="fixture-data-v1",
                split="test",
                dataset_sha256="c" * 64,
            ),
        ),
        seed_set=(17,),
        sample_ids=("sample-1", "sample-2"),
        prompt_template_hash="d" * 64,
        decode_config={"do_sample": False, "max_new_tokens": 4},
        evaluation_suites=(suite,),
        benchmark_protocol_version="llama-bench-v1",
        hardware_environment={
            "hardware": {"gpu": "NVIDIA B200", "gpu_count": 8},
            "runtime": {"backend": "llama.cpp-cuda", "device_map": "tensor-split"},
        },
        quantization_policy=quantization_policy or {"format": "BF16"},
        failures=failures,
        metrics={
            "quality": MetricValue(
                value=quality,
                category="quality",
                direction="maximize",
                evaluation_suite=suite,
            ),
        },
    )


def _policy_sha256(policy: dict[str, object]) -> str:
    encoded = json.dumps(
        policy,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _protocol(
    *,
    baseline_artifact_sha256: str = "1" * 64,
    candidate_artifact_sha256: str = "2" * 64,
    candidate_policy: dict[str, object] | None = None,
) -> EvidenceComparisonProtocol:
    policy = candidate_policy or {"format": "Q3_K_M", "backend": "llama.cpp"}
    return EvidenceComparisonProtocol(
        protocol_id="inkling-bf16-vs-q3-v1",
        baseline_artifacts=(
            ComparisonArtifactIdentity(
                artifact_id="inkling-bf16.gguf",
                role="model_export",
                sha256=baseline_artifact_sha256,
            ),
        ),
        candidate_artifacts=(
            ComparisonArtifactIdentity(
                artifact_id="inkling-q3.gguf",
                role="model_export",
                sha256=candidate_artifact_sha256,
            ),
        ),
        candidate_quantization=CandidateQuantizationIdentity(
            quantization_type="Q3_K_M",
            quantizer_name="llama-quantize",
            quantizer_revision="0123456789abcdef",
            policy_sha256=_policy_sha256(policy),
            input_artifact_ids=("inkling-bf16.gguf",),
            output_artifact_ids=("inkling-q3.gguf",),
        ),
    )


def _record(
    record_id: str,
    *,
    quality: float = 0.8,
    source_model_sha256: str = "a" * 64,
    runtime_binary_sha256: str = "b" * 64,
    comparison_role: Literal["baseline", "candidate"] | None = None,
    baseline_artifact_sha256: str = "1" * 64,
    candidate_artifact_sha256: str = "2" * 64,
    status: Literal["success", "partial", "failed"] = "success",
    failures: tuple[str, ...] = (),
) -> ExperimentEvidenceRecord:
    role: Literal["baseline", "candidate"] = comparison_role or (
        "candidate" if record_id == "candidate" else "baseline"
    )
    candidate_policy: dict[str, object] = {
        "format": "Q3_K_M",
        "backend": "llama.cpp",
    }
    protocol = _protocol(
        baseline_artifact_sha256=baseline_artifact_sha256,
        candidate_artifact_sha256=candidate_artifact_sha256,
        candidate_policy=candidate_policy,
    )
    artifact_id = "inkling-q3.gguf" if role == "candidate" else "inkling-bf16.gguf"
    artifact_sha256 = candidate_artifact_sha256 if role == "candidate" else baseline_artifact_sha256
    payload = ExperimentEvidencePayload(
        record_id=record_id,
        result_kind="quality_evaluation",
        status=status,
        comparison_role=role,
        comparison_protocol=protocol,
        subject=_summary(
            record_id,
            quality=quality,
            source_model_sha256=source_model_sha256,
            runtime_binary_sha256=runtime_binary_sha256,
            quantization_policy=(candidate_policy if role == "candidate" else {"format": "BF16"}),
            failures=failures,
        ),
        artifacts=(
            EvidenceArtifactIdentity(
                artifact_id=artifact_id,
                role="model_export",
                location=f"exports/{artifact_id}",
                sha256=artifact_sha256,
                size_bytes=1024,
            ),
        ),
        claims=("The recorded quality metric was measured.",),
        limitations=("The result applies only to the recorded experiment scope.",),
    )
    return seal_evidence_record(payload)


def _write_record(path: Path, record: ExperimentEvidenceRecord) -> Path:
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def test_evidence_round_trip_inspects_scope_without_claiming_artifact_verification(
    tmp_path: Path,
) -> None:
    path = _write_record(tmp_path / "record.json", _record("baseline"))

    loaded = load_evidence_record(path)
    inspection = inspect_evidence_record(loaded)
    validation = validate_evidence_record(loaded)

    assert inspection["record_id"] == "baseline"
    assert inspection["scope"]["source_model_sha256"] == "a" * 64
    assert inspection["integrity"]["payload_digest_verified"] is True
    assert inspection["integrity"]["artifact_bytes_verified"] is False
    assert validation["status"] == "valid"
    assert validation["artifact_bytes_verified"] is False
    assert EVIDENCE_SCOPE_WARNING in validation["warnings"]


def test_summary_schema_1_2_requires_source_and_software_scope() -> None:
    raw = _summary("run").model_dump(mode="json")
    raw["source_model_sha256"] = None
    raw["software_toolchain"] = None

    with pytest.raises(ValidationError) as captured:
        NormalizedRunSummary.model_validate(raw)

    assert "source-model SHA-256" in str(captured.value)


def test_loader_rejects_unknown_schema_as_artifact_integrity_error(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    path.write_text('{"schema_version":"legacy-experiment-v9"}\n', encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError) as captured:
        load_evidence_record(path)

    assert captured.value.code == "ARTIFACT_INTEGRITY_ERROR"
    assert "schema" in captured.value.message.lower()


def test_loader_rejects_tampered_payload(tmp_path: Path) -> None:
    path = _write_record(tmp_path / "tampered.json", _record("baseline"))
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["payload"]["subject"]["model_id"] = "other/model"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="SHA-256 mismatch"):
        load_evidence_record(path)


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":"iql-evidence-v1","schema_version":"iql-evidence-v1"}',
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError, match="duplicate JSON object key"):
        load_evidence_record(path)


def test_comparison_allows_distinct_output_artifact_hashes(tmp_path: Path) -> None:
    baseline_path = _write_record(
        tmp_path / "baseline.json",
        _record("baseline", quality=0.8),
    )
    candidate_path = _write_record(
        tmp_path / "candidate.json",
        _record("candidate", quality=0.9),
    )

    comparison = compare_evidence_records(
        load_evidence_record(baseline_path),
        load_evidence_record(candidate_path),
    )
    output = evidence_comparison_payload(comparison)

    assert comparison.result.contract_compatible is True
    assert comparison.result.delta_for("quality").absolute_delta == pytest.approx(0.1)
    assert output["baseline"]["artifacts"][0]["sha256"] == "1" * 64
    assert output["candidate"]["artifacts"][0]["sha256"] == "2" * 64
    assert output["comparison_protocol"]["candidate_quantization"]["quantization_type"] == "Q3_K_M"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("comparison_protocol", None, "require a comparison protocol"),
        ("comparison_role", "standalone", "cannot declare a comparison protocol"),
    ],
)
def test_comparable_record_requires_declared_role_and_protocol(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _record("baseline").payload.model_dump(mode="json")
    payload[field] = value

    with pytest.raises(ValidationError, match=message):
        ExperimentEvidencePayload.model_validate(payload)


def test_candidate_rejects_wrong_artifact_hash_and_quantization_policy() -> None:
    raw = _record("candidate").payload.model_dump(mode="json")
    raw["artifacts"][0]["sha256"] = "9" * 64
    with pytest.raises(ValidationError, match="artifacts do not match"):
        ExperimentEvidencePayload.model_validate(raw)

    raw = _record("candidate").payload.model_dump(mode="json")
    raw["subject"]["quantization_policy"] = {"format": "Q4_K_M"}
    with pytest.raises(ValidationError, match="quantization policy"):
        ExperimentEvidencePayload.model_validate(raw)


def test_comparison_rejects_wrong_record_roles_and_protocol_hashes(tmp_path: Path) -> None:
    baseline = _write_record(tmp_path / "baseline.json", _record("baseline"))
    wrong_role = _write_record(
        tmp_path / "wrong-role.json",
        _record("other", comparison_role="baseline"),
    )

    with pytest.raises(ComparisonCompatibilityError) as captured:
        compare_evidence_records(
            load_evidence_record(baseline),
            load_evidence_record(wrong_role),
        )
    assert captured.value.details["mismatches"][0]["dimension"] == ("candidate_comparison_role")

    candidate = _write_record(
        tmp_path / "candidate.json",
        _record("candidate", candidate_artifact_sha256="8" * 64),
    )
    with pytest.raises(ComparisonCompatibilityError) as captured:
        compare_evidence_records(
            load_evidence_record(baseline),
            load_evidence_record(candidate),
        )
    assert captured.value.details["mismatches"][0]["dimension"] == "comparison_protocol"


@pytest.mark.parametrize(
    ("field", "value", "dimension"),
    [
        ("seed_set", (19,), "seed_set"),
        ("sample_ids", ("sample-2", "sample-1"), "sample_ids"),
        ("prompt_template_hash", "9" * 64, "prompt_template_hash"),
        ("decode_config", {"do_sample": False, "max_new_tokens": 8}, "decode_config"),
    ],
)
def test_schema_1_2_comparison_checks_aggregate_and_suite_scope(
    field: str,
    value: object,
    dimension: str,
) -> None:
    baseline = _summary("baseline")
    candidate = baseline.model_copy(update={"run_id": "candidate", field: value})

    with pytest.raises(ComparisonCompatibilityError) as captured:
        compare_summaries(baseline, candidate)

    assert captured.value.details["mismatches"][0]["dimension"] == dimension


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed_set", [19], "seed_set"),
        ("sample_ids", ["sample-2", "sample-1"], "sample_ids"),
        ("prompt_template_hash", "9" * 64, "prompt_template_hash"),
        ("decode_config", {"do_sample": False, "max_new_tokens": 8}, "decode_config"),
    ],
)
def test_schema_1_2_rejects_aggregate_suite_drift(
    field: str,
    value: object,
    message: str,
) -> None:
    raw = _summary("run").model_dump(mode="json")
    raw[field] = value

    with pytest.raises(ValidationError, match=message):
        NormalizedRunSummary.model_validate(raw)


def test_partial_comparison_is_partial_and_preserves_failures(tmp_path: Path) -> None:
    baseline = _write_record(tmp_path / "baseline.json", _record("baseline"))
    candidate = _write_record(
        tmp_path / "candidate.json",
        _record(
            "candidate",
            status="partial",
            failures=("vision probe timed out",),
        ),
    )

    loaded_candidate = load_evidence_record(candidate)
    inspection = inspect_evidence_record(loaded_candidate)
    output = evidence_comparison_payload(
        compare_evidence_records(load_evidence_record(baseline), loaded_candidate)
    )

    assert inspection["failures"] == ["vision probe timed out"]
    assert inspection["scope"]["quantization_policy"]["format"] == "Q3_K_M"
    assert output["status"] == "partial"
    assert output["candidate"]["experiment_status"] == "partial"
    assert output["failures"]["candidate"] == ["vision probe timed out"]


def test_success_cannot_hide_failures_and_failed_record_cannot_compare(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="successful evidence"):
        _record("candidate", failures=("hidden failure",))

    baseline = _write_record(tmp_path / "baseline.json", _record("baseline"))
    failed = _write_record(
        tmp_path / "candidate.json",
        _record("candidate", status="failed", failures=("runtime crashed",)),
    )
    with pytest.raises(ComparisonCompatibilityError, match="Failed experiment evidence"):
        compare_evidence_records(
            load_evidence_record(baseline),
            load_evidence_record(failed),
        )


@pytest.mark.parametrize("value", ["unknown", " placeholder ", "*"])
def test_evidence_rejects_placeholder_record_identity(value: str) -> None:
    raw = _record("baseline").payload.model_dump(mode="json")
    raw["record_id"] = value

    with pytest.raises(ValidationError, match=r"exact|trimmed"):
        ExperimentEvidencePayload.model_validate(raw)


def test_comparison_rejects_source_and_software_mismatches_in_stable_order(
    tmp_path: Path,
) -> None:
    baseline_path = _write_record(tmp_path / "baseline.json", _record("baseline"))
    candidate_path = _write_record(
        tmp_path / "candidate.json",
        _record(
            "candidate",
            source_model_sha256="9" * 64,
            runtime_binary_sha256="8" * 64,
        ),
    )

    with pytest.raises(ComparisonCompatibilityError) as captured:
        compare_evidence_records(
            load_evidence_record(baseline_path),
            load_evidence_record(candidate_path),
        )

    assert [item["dimension"] for item in captured.value.details["mismatches"]] == [
        "source_model_sha256",
        "software_toolchain",
    ]


def test_evidence_cli_inspect_and_validate_emit_one_json_document(tmp_path: Path) -> None:
    path = _write_record(tmp_path / "record.json", _record("baseline"))

    inspect_result = runner.invoke(app, ["evidence", "inspect", str(path), "--json"])
    validate_result = runner.invoke(app, ["evidence", "validate", str(path), "--json"])

    assert inspect_result.exit_code == 0, inspect_result.output
    assert validate_result.exit_code == 0, validate_result.output
    assert json.loads(inspect_result.stdout)["status"] == "inspected"
    assert json.loads(validate_result.stdout)["status"] == "valid"
    assert len(inspect_result.stdout.strip().splitlines()) == 1
    assert len(validate_result.stdout.strip().splitlines()) == 1


def test_evidence_cli_schema_failure_uses_exit_5(tmp_path: Path) -> None:
    path = tmp_path / "unknown.json"
    path.write_text('{"schema_version":"legacy-experiment-v9"}\n', encoding="utf-8")

    result = runner.invoke(app, ["evidence", "validate", str(path), "--json"])

    assert result.exit_code == 5
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "ARTIFACT_INTEGRITY_ERROR"


def test_evidence_cli_incompatibility_uses_exit_6_and_writes_nothing(tmp_path: Path) -> None:
    baseline_path = _write_record(tmp_path / "baseline.json", _record("baseline"))
    candidate_path = _write_record(
        tmp_path / "candidate.json",
        _record("candidate", source_model_sha256="9" * 64),
    )
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    result = runner.invoke(
        app,
        ["evidence", "compare", str(baseline_path), str(candidate_path), "--json"],
    )

    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert result.exit_code == 6
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "COMPARISON_COMPATIBILITY_ERROR"
    assert payload["error"]["details"]["mismatches"][0]["dimension"] == ("source_model_sha256")
    assert after == before


def test_evidence_cli_compare_supports_named_override_and_preserves_warning(
    tmp_path: Path,
) -> None:
    baseline_path = _write_record(tmp_path / "baseline.json", _record("baseline"))
    candidate_path = _write_record(
        tmp_path / "candidate.json",
        _record("candidate", source_model_sha256="9" * 64),
    )

    result = runner.invoke(
        app,
        [
            "evidence",
            "compare",
            str(baseline_path),
            str(candidate_path),
            "--unsafe-override",
            "source_model_sha256",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["comparison"]["unsafe_override"] is True
    assert payload["comparison"]["overridden_dimensions"] == ["source_model_sha256"]
    assert EVIDENCE_SCOPE_WARNING in payload["warnings"]
    assert any(warning.startswith("UNSAFE COMPARISON:") for warning in payload["warnings"])
