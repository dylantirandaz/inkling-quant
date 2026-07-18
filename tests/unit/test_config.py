from __future__ import annotations

import json

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from inkling_quant_lab.config import (
    ExperimentConfig,
    SecretReference,
    dump_resolved_config,
    load_config,
)
from inkling_quant_lab.exceptions import ConfigurationError

pytestmark = pytest.mark.unit


def test_valid_minimal_config_round_trips_and_hashes_stably(tmp_path, config_factory):
    config = config_factory()
    path = tmp_path / "experiment.yaml"
    path.write_text(dump_resolved_config(config), encoding="utf-8")

    loaded = load_config(path)

    assert loaded == config
    assert loaded.model.trust_remote_code is False
    assert loaded.config_hash() == config.config_hash()
    assert json.loads(loaded.canonical_json()) == loaded.canonical_dict()


def test_component_composition_and_cli_override(tmp_path, config_factory):
    base = tmp_path / "base.yaml"
    experiment = tmp_path / "experiment.yaml"
    base.write_text(dump_resolved_config(config_factory()), encoding="utf-8")
    experiment.write_text("extends: base.yaml\nname: composed\n", encoding="utf-8")

    config = load_config(experiment, ("benchmark.repetitions=7",))

    assert config.name == "composed"
    assert config.benchmark.repetitions == 7


def test_invalid_fp8_cpu_combination_fails_before_execution(config_factory):
    raw = config_factory().canonical_dict()
    raw["quantization"]["backend"] = "fp8"
    raw["quantization"]["method"] = "fp8"

    with pytest.raises(ValidationError, match=r"FP8.*unsupported.*CPU"):
        ExperimentConfig.model_validate(raw)


def test_secret_reference_never_serializes_value(config_factory):
    raw = config_factory().canonical_dict()
    raw["security"]["secrets"] = {"hub_token": {"env": "IQL_TEST_TOKEN"}}
    config = ExperimentConfig.model_validate(raw)
    secret = "hf_this_value_must_never_persist"

    assert config.resolve_secrets({"IQL_TEST_TOKEN": secret}) == {"hub_token": secret}
    assert secret not in config.canonical_json()
    assert "IQL_TEST_TOKEN" in config.canonical_json()
    assert SecretReference(env="OPTIONAL", required=False).resolve({}) is None


def test_remote_code_requires_two_explicit_opt_ins(config_factory):
    raw = config_factory().canonical_dict()
    raw["model"]["trust_remote_code"] = True

    with pytest.raises(ValidationError, match=r"security\.allow_remote_code"):
        ExperimentConfig.model_validate(raw)


def test_calibration_and_evaluation_sample_overlap_is_rejected(config_factory):
    raw = config_factory().canonical_dict()
    raw["quantization"]["calibration"] = {
        "dataset": "local://fixtures/calibration",
        "revision": "v1",
        "split": "calibration",
        "sample_ids": ["shared"],
    }
    raw["evaluation"]["suites"][0]["sample_ids"] = ["shared"]

    with pytest.raises(ValidationError, match="must be disjoint"):
        ExperimentConfig.model_validate(raw)


def test_load_config_reports_field_specific_error(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "name: bad\nmodel:\n  model_id: x\n  dtype: definitely-not-a-dtype\n", encoding="utf-8"
    )

    with pytest.raises(ConfigurationError, match=r"model\.dtype"):
        load_config(path)


@given(st.dictionaries(st.text(min_size=1), st.integers(), max_size=20))
def test_sorted_json_is_independent_of_mapping_insertion_order(values):
    first = json.dumps(values, sort_keys=True, separators=(",", ":"))
    second_mapping = dict(reversed(list(values.items())))
    second = json.dumps(second_mapping, sort_keys=True, separators=(",", ":"))
    assert first == second


def test_unknown_fields_are_forbidden(config_factory):
    raw = yaml.safe_load(dump_resolved_config(config_factory()))
    raw["hidden_default"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ExperimentConfig.model_validate(raw)


def test_runnable_outputs_require_at_least_one_evaluation_suite(config_factory):
    raw = config_factory().canonical_dict()
    raw["evaluation"]["suites"] = []

    with pytest.raises(ValidationError, match=r"evaluation\.suites.*benchmark\.enabled"):
        ExperimentConfig.model_validate(raw)

    raw["benchmark"]["enabled"] = False
    raw["reporting"].update({"markdown": False, "html": False, "plots": False})

    inspection_only = ExperimentConfig.model_validate(raw)
    assert inspection_only.evaluation.suites == ()
    assert inspection_only.benchmark.enabled is False
