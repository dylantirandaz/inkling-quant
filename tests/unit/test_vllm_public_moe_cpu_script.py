from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.evaluate_vllm_public_moe_cpu import (
    _MODEL_FILE_FACTS,
    _WORKER_SENTINEL,
    _audit_model_tree,
    _parse_worker_stdout,
    _validate_public_record,
    atomic_write_json,
    resolve_output_path,
    select_story,
    split_stories,
    token_ids_sha256,
    validate_hardware_label,
    validate_runtime_python,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKED_EVIDENCE = PROJECT_ROOT / "docs/experiments/vllm-0.23.0-apple-cpu-stories15m-inference.json"
EVIDENCE_SHA256 = "eb3f53b508b513895dcb8cc08c84890db97f150c93b21fdbcbfae8943e27f4ee"
EVALUATOR_SHA256 = "2bc5b3aca9e843866feb02ac0be31f2d18aa9538aab2a278422fc28eabeee3ca"
PATCH_SHA256 = "3ef5849f3f2f6d63fa10c2aacf224bf755fd87392771531bba82ffeff9fc4a2e"


def test_story_selection_uses_stable_zero_based_sample_identity() -> None:
    text = "zero<|endoftext|> one <|endoftext|>two<|endoftext|>"

    assert split_stories(text) == ("zero", "one", "two")
    assert select_story(text, "story-000001") == "one"
    with pytest.raises(ValueError, match="invalid sample ID"):
        select_story(text, "story-1")
    with pytest.raises(ValueError, match="outside"):
        select_story(text, "story-000003")


def test_token_id_digest_is_canonical_without_exposing_ids() -> None:
    assert token_ids_sha256([1, 20, 300]) == (
        "789be4f68ce5c3bc0e01d0e19e3a12e82496b528eec3f76bdc798fa1842463e6"
    )


def test_model_tree_rejects_unsafe_files_and_escaping_links(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    snapshot = repository / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    inventory = _audit_model_tree(snapshot)
    assert inventory == [
        {
            "path": "config.json",
            "sha256": ("44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"),
            "size_bytes": 2,
            "symlink": False,
        }
    ]

    unsafe = snapshot / "modeling.py"
    unsafe.write_text("raise RuntimeError", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe file"):
        _audit_model_tree(snapshot)
    unsafe.unlink()

    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    escaping = snapshot / "outside.txt"
    escaping.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes repository cache"):
        _audit_model_tree(snapshot)


def test_output_must_be_new_json_below_artifacts(tmp_path: Path) -> None:
    output = resolve_output_path(tmp_path, Path("artifacts/research/record.json"))
    assert output == tmp_path / "artifacts" / "research" / "record.json"

    output.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="immutable output"):
        resolve_output_path(tmp_path, Path("artifacts/research/record.json"))
    with pytest.raises(ValueError, match="below the project artifacts"):
        resolve_output_path(tmp_path, Path("docs/record.json"))
    with pytest.raises(ValueError, match="cannot contain"):
        resolve_output_path(tmp_path, Path("artifacts/../escape.json"))
    with pytest.raises(ValueError, match=r"\.json suffix"):
        resolve_output_path(tmp_path, Path("artifacts/research/record.txt"))


def test_atomic_json_publish_is_finite_and_never_replaces(tmp_path: Path) -> None:
    output = tmp_path / "record.json"
    atomic_write_json(output, {"finite": 1.0})
    original = output.read_bytes()

    with pytest.raises(FileExistsError):
        atomic_write_json(output, {"replacement": 2.0})
    assert output.read_bytes() == original
    with pytest.raises(ValueError, match="Out of range float values"):
        atomic_write_json(tmp_path / "nan.json", {"bad": float("nan")})


def test_worker_stdout_requires_exactly_one_json_object_sentinel() -> None:
    value = {"status": "success", "count": 3}
    stdout = "ordinary log\n" + _WORKER_SENTINEL + json.dumps(value) + "\n"

    assert _parse_worker_stdout(stdout) == value
    with pytest.raises(RuntimeError, match="exactly one"):
        _parse_worker_stdout("ordinary log only")
    with pytest.raises(RuntimeError, match="exactly one"):
        _parse_worker_stdout(stdout + _WORKER_SENTINEL + "{}\n")
    with pytest.raises(RuntimeError, match="not a JSON object"):
        _parse_worker_stdout(_WORKER_SENTINEL + "[]\n")


def test_public_record_and_hardware_label_fail_closed_on_sensitive_payloads() -> None:
    _validate_public_record(
        {
            "prompt_token_ids_sha256": "a" * 64,
            "prompt_text_persisted": False,
            "generated_text_persisted": False,
        }
    )
    assert validate_hardware_label("Apple M3, 8 CPU cores, 16 GB") == (
        "Apple M3, 8 CPU cores, 16 GB"
    )

    with pytest.raises(ValueError, match="raw prompt/output"):
        _validate_public_record({"prompt_text": "private"})
    with pytest.raises(ValueError, match="credential"):
        validate_hardware_label("Apple M3 token=hf_abcdefgh1234")
    with pytest.raises(ValueError, match="whitespace"):
        validate_hardware_label(" Apple M3")


def test_runtime_python_validation_preserves_virtual_environment_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "python-target"
    target.write_text("#!/bin/sh\n", encoding="utf-8")
    target.chmod(0o700)
    virtual_environment = tmp_path / "venv" / "bin"
    virtual_environment.mkdir(parents=True)
    link = virtual_environment / "python"
    link.symlink_to(target)

    assert validate_runtime_python(link) == link
    with pytest.raises(FileNotFoundError):
        validate_runtime_python(tmp_path / "missing-python")


def test_checked_vllm_evidence_is_exact_safe_and_reference_identical() -> None:
    evidence_bytes = CHECKED_EVIDENCE.read_bytes()
    evidence = json.loads(evidence_bytes)
    evaluator_bytes = (PROJECT_ROOT / "scripts/evaluate_vllm_public_moe_cpu.py").read_bytes()
    patch_bytes = (
        PROJECT_ROOT / "scripts/patches/vllm-0.23.0-apple-cpu-stories15m.patch"
    ).read_bytes()

    assert hashlib.sha256(evidence_bytes).hexdigest() == EVIDENCE_SHA256
    assert hashlib.sha256(evaluator_bytes).hexdigest() == EVALUATOR_SHA256
    assert hashlib.sha256(patch_bytes).hexdigest() == PATCH_SHA256
    assert evidence["schema_version"] == "external-vllm-public-moe-inference-v2"
    assert evidence["status"] == "success"
    assert evidence["evaluator"]["sha256_at_start"] == EVALUATOR_SHA256
    assert evidence["evaluator"]["sha256_at_end"] == EVALUATOR_SHA256
    assert evidence["model"]["revision"] == ("b6dd737497465570b5f5e962dbc9d9454ed1e0eb")
    assert evidence["model"]["weight_sha256"] == (
        "dbfa0289f68a8dd721d10eb12d8bd82e098455682027f6f9986ba548913f9082"
    )
    observed_model_files = {
        item["path"]: (item["size_bytes"], item["sha256"])
        for item in evidence["model"]["file_inventory"]
    }
    assert observed_model_files == _MODEL_FILE_FACTS
    assert evidence["dataset"]["file_sha256"] == (
        "94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"
    )
    execution = evidence["execution"]
    expected_tokens_sha256 = "8cad579c5970c8c00d5970a222afa6ad53169fb9525acf0441546edce1f4ebdc"
    assert execution["vllm_generated_token_ids_sha256"] == [
        expected_tokens_sha256,
        expected_tokens_sha256,
        expected_tokens_sha256,
    ]
    assert execution["transformers_generated_token_ids_sha256"] == (expected_tokens_sha256)
    assert execution["all_vllm_repetitions_identical"] is True
    assert execution["vllm_matches_transformers_exactly"] is True
    assert evidence["claims"]["upstream_vllm_unmodified_supported"] is False
    assert evidence["vllm_source_and_build"]["patch_sha256"] == PATCH_SHA256

    serialized = evidence_bytes.decode("utf-8")
    assert "/Users/" not in serialized
    assert "/private/" not in serialized
    assert '"prompt_token_ids":' not in serialized
    assert '"generated_token_ids":' not in serialized
    assert '"output_token_ids":' not in serialized
