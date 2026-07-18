from __future__ import annotations

import os
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from inkling_quant_lab.config import ExperimentConfig
from inkling_quant_lab.exceptions import (
    ArtifactIntegrityError,
    ConfigurationError,
    ModelLoadError,
)
from inkling_quant_lab.pipeline.runner import run_experiment
from inkling_quant_lab.security import (
    Redactor,
    safe_path,
    sensitive_literal_path,
    validate_checkpoint_path,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", ["../escape", "a/../../escape", "/tmp/escape", ".."])
def test_path_traversal_is_rejected(tmp_path, value):
    with pytest.raises(ArtifactIntegrityError):
        safe_path(tmp_path / "root", value)


def test_prefix_collision_is_rejected(tmp_path):
    root = tmp_path / "artifacts"
    with pytest.raises(ArtifactIntegrityError):
        safe_path(root, root.parent / "artifacts-evil" / "file")


def test_symlink_escape_is_rejected(tmp_path):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    os.symlink(outside, root / "link")

    with pytest.raises(ArtifactIntegrityError, match="escapes"):
        safe_path(root, "link/secret.json")


def test_safe_nested_path_is_returned(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    path = safe_path(root, "metrics/result.json", create_parent=True)
    assert path.parent.is_dir()
    assert path.is_relative_to(root)


@pytest.mark.parametrize("run_id", ["../escape", "/tmp/escape", "nested/run", ".", ""])
def test_programmatic_run_id_cannot_escape_or_nest_artifact_root(
    tmp_path: Path,
    config_factory,
    run_id: str,
) -> None:
    raw = config_factory().canonical_dict()
    raw["output"]["root"] = str(tmp_path / "artifacts")
    config = ExperimentConfig.model_validate(raw)

    with pytest.raises(ArtifactIntegrityError, match="run_id must be a single safe path"):
        run_experiment(config, run_id=run_id, project_root=tmp_path)

    assert not (tmp_path / "escape").exists()


def test_missing_required_secret_does_not_create_a_partial_run(
    tmp_path: Path,
    config_factory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    variable = "IQL_TEST_REQUIRED_SECRET_IS_ABSENT"
    monkeypatch.delenv(variable, raising=False)
    raw = config_factory().canonical_dict()
    raw["output"]["root"] = str(tmp_path / "artifacts")
    raw["security"]["secrets"] = {"backend_token": {"env": variable, "required": True}}
    config = ExperimentConfig.model_validate(raw)

    with pytest.raises(ConfigurationError, match=variable):
        run_experiment(config, run_id="missing-secret", project_root=tmp_path)

    assert not (tmp_path / "artifacts/missing-secret").exists()


@given(st.lists(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1), min_size=1, max_size=5))
def test_normalized_safe_parts_stay_under_root(parts):
    root = Path("/tmp/iql-safe-path-property-root")
    path = safe_path(root, "/".join(parts))
    assert path.is_relative_to(root.resolve())


@pytest.mark.parametrize("name", ["weights.bin", "model.pt", "state.pkl", "x.ckpt"])
def test_pickle_checkpoint_rejected_by_default(tmp_path, name):
    with pytest.raises(ModelLoadError, match="Pickle-based"):
        validate_checkpoint_path(tmp_path / name, allow_pickle=False)
    validate_checkpoint_path(tmp_path / name, allow_pickle=True)


def test_safetensors_is_allowed(tmp_path):
    validate_checkpoint_path(tmp_path / "model.safetensors", allow_pickle=False)


def test_redactor_removes_explicit_and_common_tokens():
    redactor = Redactor(("super-secret",))
    output = redactor.text(
        "token=super-secret Authorization: Bearer abc123 hf_abcdefgh1234 sk-abcdefgh1234"
    )
    assert "super-secret" not in output
    assert "abc123" not in output
    assert "hf_" not in output
    assert "sk-" not in output
    assert "[REDACTED]" in output


def test_redactor_replaces_overlapping_explicit_secrets_longest_first():
    redactor = Redactor(("abc", "abcdef", "abcdef", "bcde"))

    output = redactor.text("values=abcdef,abc")

    assert output == "values=[REDACTED],[REDACTED]"
    assert "def" not in output


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("api_key", "literal-credential"),
        ("endpoint", "https://example.test?access_token=literal-credential"),
        ("note", "Authorization: Bearer literal-credential"),
        ("note", "hf_abcdefgh1234"),
    ],
)
def test_config_rejects_literal_credentials_outside_secret_references(
    config_factory,
    field: str,
    value: str,
) -> None:
    raw = config_factory().canonical_dict()
    raw["quantization"]["parameters"][field] = value

    with pytest.raises(ValidationError, match="literal credential material"):
        ExperimentConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"nested": [{"token": "literal"}]}, ("nested", "0", "token")),
        ({"values": ("safe", "sk-abcdefgh1234")}, ("values", "1")),
        ({7: {"authorization": "Bearer literal"}}, ("7", "authorization")),
        ({"token": "", "safe": [1, None, {"note": "ordinary text"}]}, None),
        ("ordinary text", None),
    ],
)
def test_sensitive_literal_scan_handles_nested_shapes(value, expected) -> None:
    assert sensitive_literal_path(value) == expected
