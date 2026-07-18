from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from inkling_quant_lab.config import ExperimentConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_CONFIG = PROJECT_ROOT / "configs/experiments/hf_stories15m_native_int8_isolated_peak.yaml"
MODE = "isolated_subject_artifact_peak_rss"


def _source_weight_free_mapping() -> dict[str, Any]:
    config = load_config(PUBLIC_CONFIG)
    mapping = deepcopy(config.model_dump(mode="json"))
    mapping["benchmark"]["host_memory_mode"] = MODE
    mapping["benchmark"]["protocol_version"] = "public-moe-source-weight-free-peak-v1"
    return mapping


def test_source_weight_free_peak_mode_accepts_only_the_exact_public_native_contract() -> None:
    config = ExperimentConfig.model_validate(_source_weight_free_mapping())

    assert config.benchmark.host_memory_mode == MODE
    assert config.model.adapter == "hf_causal_lm"
    assert config.model.local_files_only is True
    assert config.quantization.backend == "torch_native_dynamic_int8"
    assert config.quantization.export.enabled is True
    assert config.quantization.export.format == "safetensors"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("benchmark", "enabled"), False, "requires benchmark.enabled=true"),
        (("model", "adapter"), "local_fixture", "requires model.adapter=hf_causal_lm"),
        (("model", "local_files_only"), False, "requires offline"),
        (("model", "revision"), "0" * 40, "pinned Stories15M revision"),
        (("model", "trust_remote_code"), True, "requires security.allow_remote_code=true"),
        (("security", "allow_remote_code"), True, "remote-code opt-ins false"),
        (
            ("quantization", "backend"),
            "torch_dynamic_int8",
            "requires torch_native_dynamic_int8",
        ),
        (("quantization", "export", "enabled"), False, "requires an enabled"),
        (("quantization", "export", "format"), "recipe_json", "requires an enabled"),
        (("runtime", "backend"), "fake_cpu", "requires unsharded single-CPU"),
    ),
)
def test_source_weight_free_peak_mode_rejects_unproven_combinations(
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    mapping = _source_weight_free_mapping()
    cursor: dict[str, Any] = mapping
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value

    with pytest.raises(ValidationError, match=message):
        ExperimentConfig.model_validate(mapping)
