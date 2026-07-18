"""CLI integration coverage for persisted pipeline failures."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inkling_quant_lab.cli import app
from inkling_quant_lab.config import ExperimentConfig, dump_resolved_config

pytestmark = pytest.mark.integration


def test_backend_failure_persists_evidence_and_reports_run_directory(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    raw = config_factory(backend="missing_backend").canonical_dict()
    raw["output"]["root"] = str(tmp_path / "artifacts")
    config = ExperimentConfig.model_validate(raw)
    config_path = tmp_path / "failure.yaml"
    config_path.write_text(dump_resolved_config(config), encoding="utf-8")

    result = CliRunner().invoke(app, ["run", str(config_path), "--json"])

    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    run_directory = Path(payload["run_directory"])
    assert payload["error"]["code"] == "CAPABILITY_ERROR"
    assert run_directory.is_dir()
    status = json.loads((run_directory / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["current_stage"] == "probe_runtime"
    assert status["failure"]["code"] == "CAPABILITY_ERROR"
    failure_files = tuple((run_directory / "failures").glob("probe_runtime-attempt-*.json"))
    assert len(failure_files) == 1
    assert (run_directory / "events.jsonl").is_file()
