from __future__ import annotations

import sys

import pytest

from inkling_quant_lab.exceptions import ArtifactIntegrityError
from inkling_quant_lab.security import Redactor, run_command, safe_path, validate_checkpoint_path

pytestmark = pytest.mark.unit


def test_redactor_recurses_through_json_compatible_values():
    redactor = Redactor(("super-secret", ""))
    value = {
        "secret": "super-secret",
        7: ["hf_abcdefgh", ("safe", "sk-abcdefgh")],
        "number": 3,
    }

    redacted = redactor.value(value)

    assert redacted == {
        "secret": "[REDACTED]",
        "7": ["[REDACTED]", ("safe", "[REDACTED]")],
        "number": 3,
    }


def test_root_itself_is_not_a_valid_artifact_file_path(tmp_path):
    root = tmp_path / "root"

    with pytest.raises(ArtifactIntegrityError, match="escapes configured root"):
        safe_path(root, ".")


def test_checkpoint_suffix_check_is_case_insensitive(tmp_path):
    with pytest.raises(Exception, match="Pickle-based checkpoint"):
        validate_checkpoint_path(tmp_path / "WEIGHTS.PT", allow_pickle=False)


def test_run_command_rejects_empty_argv():
    with pytest.raises(ValueError, match="must not be empty"):
        run_command(())


def test_run_command_uses_requested_working_directory(tmp_path):
    completed = run_command(
        (sys.executable, "-c", "import os; print(os.getcwd())"),
        cwd=tmp_path,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == str(tmp_path)
    assert completed.stderr == ""
