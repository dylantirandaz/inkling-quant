from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREREGISTRATION = (
    PROJECT_ROOT / "docs/experiments/stories15m-native-int8-confirmatory-256-preregistration.json"
)
ATTEMPT_ROOT = PROJECT_ROOT / "artifacts/research-slices/stories15m-native-int8-confirmatory-256"
PREREGISTRATION_SHA256 = "290dfee3ffaec8a46472f40f5b50ebc7a07f8ea34ae0caf7cec10ee8045ae0a4"
PROTOCOL_DEFINITION_SHA256 = "84dff06f85d9490f71d80a37fa4788ee6bec326a5673afef67461ecc586153dd"
ATTEMPT_START_SHA256 = {
    1: "0c8ca5830a07cc86c000c93db1efcbf1153b719ff2e07fb25936b73c0f3300ca",
    2: "c11b5ec5544a4501b5d523ab0d275bf4fd23d45d158adc4c723e89da03ac7d37",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate historical JSON key: {key}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"non-finite historical JSON number: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=reject_nonfinite,
    )
    assert isinstance(value, dict)
    return value


def _strict_preregistration() -> dict[str, Any]:
    return _strict_json(PREREGISTRATION)


def test_preregistration_is_the_pinned_outcome_blind_lock() -> None:
    preregistration = _strict_preregistration()

    assert PREREGISTRATION.stat().st_size == 18_104
    assert _sha256(PREREGISTRATION) == PREREGISTRATION_SHA256
    assert preregistration["schema_version"] == "confirmatory-quality-preregistration-v1"
    assert preregistration["status"] == "locked_before_execution"
    assert preregistration["protocol_id"] == ("stories15m-native-int8-confirmatory-quality-v1")
    assert preregistration["created_at_utc"] == "2026-07-17T04:42:19Z"
    assert preregistration["outcomes"] == {
        "holdout_model_forward_executed": False,
        "holdout_outcomes_inspected": False,
    }

    bindings = preregistration["bindings"]
    assert bindings["protocol_definition_sha256"] == PROTOCOL_DEFINITION_SHA256
    assert bindings["resolved_config_sha256"] == {
        "baseline": "e6db2959221babf9aaba1f529a20349fe3462d4309b6929caf3a7d3331d668f2",
        "candidate": "d6ee2054d6db801c56596042df7e9eca130109ab922ee46735b61953da8d7912",
    }
    assert bindings["dataset"] == {
        "file_sha256": ("94e431816c4cce81ff71e4408ff8d3bda9a42e8d2663986697c3954288cb38b4"),
        "size_bytes": 19_447_282,
        "story_count": 21_990,
    }


def test_attempt_starts_retain_the_exact_historical_source_and_environment_lock() -> None:
    preregistration = _strict_preregistration()
    bindings = preregistration["bindings"]
    files = bindings["files"]

    assert len(files) == 98
    assert all(
        isinstance(relative, str)
        and not Path(relative).is_absolute()
        and isinstance(expected_sha256, str)
        and len(expected_sha256) == 64
        for relative, expected_sha256 in files.items()
    )
    bound_source = {relative for relative in files if relative.startswith("src/inkling_quant_lab/")}
    # Completeness is a lock-time fact retained by both starts, not a requirement
    # that the evolving working tree remain byte-identical forever.
    assert len(bound_source) == 82
    assert {
        "AGENTS.md",
        "SPEC.md",
        "SDD.md",
        "TDD.md",
        "scripts/__init__.py",
        "scripts/evaluate_public_moe_native_quality.py",
        "scripts/run_confirmatory_quality_attempt.py",
        "scripts/verify_confirmatory_quality_repeats.py",
        "docs/adr/ADR-025-prospective-tinystories-int8-noninferiority.md",
        "pyproject.toml",
        "uv.lock",
    }.issubset(files)

    expected_preregistration = {
        "path": PREREGISTRATION.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": PREREGISTRATION_SHA256,
        "size_bytes": 18_104,
    }
    starts: list[dict[str, Any]] = []
    for ordinal in (1, 2):
        path = ATTEMPT_ROOT / f"attempt-{ordinal}" / "start.json"
        assert path.is_file() and not path.is_symlink()
        assert path.stat().st_size == 29_299
        assert _sha256(path) == ATTEMPT_START_SHA256[ordinal]
        start = _strict_json(path)
        starts.append(start)

        assert start["schema_version"] == "confirmatory-quality-attempt-start-v1"
        assert start["ordinal"] == ordinal
        assert start["preregistration"] == expected_preregistration
        assert start["file_sha256"] == files
        assert start["protocol_definition_sha256"] == bindings["protocol_definition_sha256"]
        assert start["resolved_config_sha256"] == bindings["resolved_config_sha256"]
        assert start["environment_contract"] == preregistration["environment_contract"]

    assert starts[0]["file_sha256"] == starts[1]["file_sha256"]
    assert starts[0]["environment_contract"] == starts[1]["environment_contract"]


def test_preregistered_analysis_and_two_attempt_promotion_are_exact() -> None:
    preregistration = _strict_preregistration()
    primary = preregistration["primary_analysis"]

    assert primary["quality_sample_count"] == 256
    assert primary["analysis_population"]["eligible_story_count"] == 21_958
    assert primary["analysis_population"]["eligible_evaluated_token_count"] == 4_441_967
    assert primary["decision"] == {
        "coverage_interpretation": (
            "nominal_approximate_coverage_under_the_predeclared_stratified_"
            "finite_population_bootstrap_design"
        ),
        "margin_nats_per_token": 0.004987541511039074,
        "margin_relative_perplexity": 0.005,
        "rule": "upper_bound_strictly_less_than_margin",
    }
    assert primary["bootstrap"]["replicates"] == 100_000
    assert primary["bootstrap"]["seed"] == 20_260_715
    assert primary["stopping_rule"] == (
        "exactly_256_stories_with_no_outcome_dependent_extension_or_substitution"
    )
    assert primary["repeat_verification"] == {
        "aggregate_path": (
            "artifacts/research-slices/stories15m-native-int8-confirmatory-256/"
            "repeat-verification.json"
        ),
        "input_records": [
            "artifacts/research-slices/stories15m-native-int8-confirmatory-256/"
            "attempt-1/record.json",
            "artifacts/research-slices/stories15m-native-int8-confirmatory-256/"
            "attempt-2/record.json",
        ],
        "promotion_rule": (
            "both_within_execution_gates_pass_and_scientific_environment_projections_match_exactly"
        ),
        "repeat_is_additional_statistical_sample": False,
        "required_clean_process_executions": 2,
        "single_execution_status": "provisional_until_2_clean_executions",
    }
    assert [attempt["ordinal"] for attempt in preregistration["planned_attempts"]] == [1, 2]


def test_preregistered_interpreter_identity_is_retained_as_historical_metadata() -> None:
    preregistration = _strict_preregistration()
    environment = preregistration["environment_contract"]
    executable = environment["python_executable"]

    assert executable == {
        "invocation_path": (
            "/Users/dylantirandaz/thinking machines/inkling-quant/.venv/bin/python"
        ),
        "resolved_path": (
            "/Users/dylantirandaz/.local/share/uv/python/"
            "cpython-3.12.13-macos-aarch64-none/bin/python3.12"
        ),
        "sha256": "36984af9f922aec72ba5b03e811ee92512268785007192e20f826cbd07cf8d04",
        "size_bytes": 49_968,
    }
    assert environment["virtual_environment"] == {
        "pyvenv_cfg_path": ".venv/pyvenv.cfg",
        "sha256": "6ac655fe109ceb2312e57bd560b481b11e982a4f57a014d426dcf6e498afd8d6",
        "size_bytes": 211,
    }
    assert environment["offline_environment"] == {
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPYCACHEPREFIX": "/dev/null",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for ordinal in (1, 2):
        start = _strict_json(ATTEMPT_ROOT / f"attempt-{ordinal}" / "start.json")
        assert start["environment_contract"] == environment
