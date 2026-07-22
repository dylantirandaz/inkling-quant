"""Focused tests for the public Typer command surface."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from inkling_quant_lab.cli import app
from inkling_quant_lab.config import (
    ExperimentConfig,
    dump_resolved_config,
    load_inspection_config,
)
from inkling_quant_lab.exceptions import CapabilityError, ConfigurationError

pytestmark = pytest.mark.unit

runner = CliRunner()


def _config_file(tmp_path: Path, config: ExperimentConfig) -> Path:
    path = tmp_path / "experiment.yaml"
    path.write_text(dump_resolved_config(config), encoding="utf-8")
    return path


def test_validate_json_applies_repeatable_dotted_overrides(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    path = _config_file(tmp_path, config_factory())

    result = runner.invoke(
        app,
        [
            "validate",
            str(path),
            "--set",
            "benchmark.repetitions=7",
            "--set",
            "routing.mode=full_trace",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "valid"
    assert payload["validation_scope"] == "configuration_only"
    assert payload["capabilities_probed"] is False
    assert payload["resolved_config"]["benchmark"]["repetitions"] == 7
    assert payload["resolved_config"]["routing"]["mode"] == "full_trace"
    assert len(payload["config_hash"]) == 64


def test_validate_failure_is_nonzero_and_field_specific(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("name: invalid\nmodel:\n  model_id: ''\n", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(path)])

    assert result.exit_code != 0
    assert "CONFIGURATION_ERROR" in result.output
    assert "model.model_id" in result.output


def test_validate_json_failure_is_one_machine_readable_document(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("name: invalid\n", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(path), "--json"])

    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "CONFIGURATION_ERROR"
    assert payload["error"]["component"] == "config"


def test_doctor_json_has_no_non_json_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "inkling_quant_lab.cli.probe_environment",
        lambda _root: {
            "python": {"version": "3.11.9", "implementation": "CPython"},
            "platform": {"system": "TestOS", "machine": "test64"},
            "hardware": {"logical_cpu_count": 8, "device": "cpu", "accelerator": None},
            "packages": {},
            "git": {"commit": None, "dirty": None},
        },
    )

    result = runner.invoke(app, ["doctor", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["backend_scope"] == "declared_registry_only"
    assert payload["backend_capabilities_probed"] is False
    assert payload["available_devices"] == ["cpu"]
    assert payload["environment"]["hardware"]["logical_cpu_count"] == 8
    assert any(item["name"] == "awq" for item in payload["backends"]["quantizers"])


def test_list_backends_reports_declared_availability_without_resolving() -> None:
    result = runner.invoke(app, ["list-backends", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    quantizers = {item["name"]: item for item in payload["backends"]["quantizers"]}
    assert quantizers["noop"]["available"] is True
    assert quantizers["awq"]["available"] is None
    assert quantizers["awq"]["optional_extra"] == "awq"


def test_run_failure_reports_planned_run_directory(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = config_factory().canonical_dict()
    raw["output"]["root"] = str(tmp_path / "artifacts")
    path = _config_file(tmp_path, ExperimentConfig.model_validate(raw))
    monkeypatch.setattr(
        "inkling_quant_lab.cli._generate_run_id",
        lambda _config: "test-run-id",
    )

    def fail_run(_config: ExperimentConfig, *, run_id: str, project_root: Path) -> Path:
        del run_id, project_root
        raise CapabilityError(
            "backend unavailable",
            stage="probe_runtime",
            component="missing_backend",
            remediation="Choose a registered CPU backend.",
        )

    monkeypatch.setattr("inkling_quant_lab.cli._run_pipeline", fail_run)

    result = runner.invoke(app, ["run", str(path), "--json"])

    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    expected = (tmp_path / "artifacts" / "test-run-id").resolve()
    assert payload["error"]["code"] == "CAPABILITY_ERROR"
    assert payload["error"]["stage"] == "probe_runtime"
    assert payload["run_directory"] == str(expected)


def test_run_requires_remote_code_cli_flag_even_when_config_opts_in(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
) -> None:
    raw = config_factory().canonical_dict()
    raw["model"]["trust_remote_code"] = True
    raw["security"]["allow_remote_code"] = True
    path = _config_file(tmp_path, ExperimentConfig.model_validate(raw))

    result = runner.invoke(app, ["run", str(path)])

    assert result.exit_code != 0
    assert "--allow-remote-code" in result.output
    assert "CONFIGURATION_ERROR" in result.output


def test_run_remote_code_flag_sets_both_required_config_fields(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = config_factory().canonical_dict()
    raw["output"]["root"] = str(tmp_path / "artifacts")
    path = _config_file(tmp_path, ExperimentConfig.model_validate(raw))
    captured: dict[str, bool] = {}
    monkeypatch.setattr("inkling_quant_lab.cli._generate_run_id", lambda _config: "remote-opt-in")

    def complete(config: ExperimentConfig, *, run_id: str, project_root: Path) -> Path:
        captured["model"] = config.model.trust_remote_code
        captured["security"] = config.security.allow_remote_code
        return project_root / config.output.root / run_id

    monkeypatch.setattr("inkling_quant_lab.cli._run_pipeline", complete)

    result = runner.invoke(app, ["run", str(path), "--allow-remote-code"])

    assert result.exit_code == 0, result.output
    assert captured == {"model": True, "security": True}


def test_run_failure_redacts_configured_secret_value(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "custom-value-that-is-not-a-token-pattern"
    monkeypatch.setenv("IQL_CLI_TEST_SECRET", secret)
    raw = config_factory().canonical_dict()
    raw["output"]["root"] = str(tmp_path / "artifacts")
    raw["security"]["secrets"] = {"backend_token": {"env": "IQL_CLI_TEST_SECRET", "required": True}}
    path = _config_file(tmp_path, ExperimentConfig.model_validate(raw))
    monkeypatch.setattr("inkling_quant_lab.cli._generate_run_id", lambda _config: "secret-run")

    def fail_run(_config: ExperimentConfig, *, run_id: str, project_root: Path) -> Path:
        del run_id, project_root
        raise RuntimeError(f"backend exposed {secret}")

    monkeypatch.setattr("inkling_quant_lab.cli._run_pipeline", fail_run)

    result = runner.invoke(app, ["run", str(path), "--json"])

    assert result.exit_code != 0
    assert secret not in result.output
    assert "[REDACTED]" in json.loads(result.stdout)["error"]["message"]


def test_resume_forwards_force_stage_and_returns_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_directory = tmp_path / "run"
    captured: dict[str, Any] = {}

    def resume(
        directory: Path,
        *,
        force_stage: str | None,
        project_root: Path,
        allow_remote_code: bool,
    ) -> Path:
        captured.update(
            directory=directory,
            force_stage=force_stage,
            project_root=project_root,
            allow_remote_code=allow_remote_code,
        )
        return directory.resolve()

    monkeypatch.setattr("inkling_quant_lab.cli._resume_pipeline", resume)

    result = runner.invoke(
        app,
        ["resume", str(run_directory), "--force-stage", "quantize", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["run_directory"] == str(run_directory.resolve())
    assert captured["directory"] == run_directory.resolve()
    assert captured["force_stage"] == "quantize"
    assert captured["allow_remote_code"] is False


def test_resume_rejects_unknown_force_stage_before_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def resume(*_args: Any, **_kwargs: Any) -> Path:
        nonlocal called
        called = True
        return tmp_path

    monkeypatch.setattr("inkling_quant_lab.cli._resume_pipeline", resume)

    result = runner.invoke(
        app,
        ["resume", str(tmp_path / "run"), "--force-stage", "not-a-stage", "--json"],
    )

    assert result.exit_code != 0
    assert json.loads(result.stdout)["error"]["code"] == "CONFIGURATION_ERROR"
    assert called is False


def test_resume_failure_redacts_secret_from_persisted_config(
    tmp_path: Path,
    config_factory: Callable[..., ExperimentConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "another-custom-sensitive-value"
    monkeypatch.setenv("IQL_CLI_RESUME_SECRET", secret)
    raw = config_factory().canonical_dict()
    raw["security"]["secrets"] = {
        "resume_token": {"env": "IQL_CLI_RESUME_SECRET", "required": True}
    }
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / "resolved_config.yaml").write_text(
        dump_resolved_config(ExperimentConfig.model_validate(raw)),
        encoding="utf-8",
    )

    def fail_resume(
        _directory: Path,
        *,
        force_stage: str | None,
        project_root: Path,
        allow_remote_code: bool,
    ) -> Path:
        del force_stage, project_root, allow_remote_code
        raise RuntimeError(f"resume exposed {secret}")

    monkeypatch.setattr("inkling_quant_lab.cli._resume_pipeline", fail_resume)

    result = runner.invoke(app, ["resume", str(run_directory)])

    assert result.exit_code != 0
    assert secret not in result.output
    assert "[REDACTED]" in result.output


def test_inspect_model_accepts_model_fragment_and_reports_moe_inventory(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """
model:
  model_id: local://fixtures/tiny-moe
  revision: fixture-v1
  adapter: local_fixture
  dtype: float32
  trust_remote_code: false
  local_files_only: true
  checkpoint_format: fixture
""".lstrip(),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["inspect-model", str(path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "ok"
    assert payload["model"]["model_id"] == "local://fixtures/tiny-moe"
    assert payload["inventory"]["module_count"] > 0
    assert payload["inventory"]["parameter_count"] > 0
    assert payload["moe"]["layers"][0]["expert_count"] == 4
    assert payload["moe"]["layers"][0]["top_k"] == 2


def test_inspect_model_accepts_direct_local_fixture_identifier() -> None:
    result = runner.invoke(app, ["inspect", "local://fixtures/tiny-moe", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["inspection_target"] == "local://fixtures/tiny-moe"
    assert payload["model"]["model_id"] == "local://fixtures/tiny-moe"
    assert payload["model"]["capabilities"]["is_moe"] is True
    assert len(payload["moe"]["layers"]) == 2


def test_inspection_config_uses_full_safe_schema_and_dotted_overrides(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        "model:\n  model_id: local://fixtures/tiny-dense\n  checkpoint_format: fixture\n",
        encoding="utf-8",
    )

    config = load_inspection_config(path, ("runtime.dtype=bfloat16",))

    assert config.name == "model-inspection"
    assert config.runtime.backend == "torch_eager_cpu"
    assert config.runtime.dtype == "bfloat16"
    assert config.benchmark.enabled is False
    assert config.reporting.markdown is False
    assert config.reporting.html is False
    assert config.reporting.plots is False
    assert config.security.allow_remote_code is False
    assert config.quantization.backend == "noop"


def test_direct_external_identifier_uses_conservative_hf_defaults() -> None:
    config = load_inspection_config("organization/model-name")

    assert config.model.model_id == "organization/model-name"
    assert config.model.adapter == "hf_causal_lm"
    assert config.model.checkpoint_format == "safetensors"
    assert config.model.local_files_only is True
    assert config.model.trust_remote_code is False
    assert config.security.allow_remote_code is False


def test_inspection_config_preserves_remote_code_security_invariant(tmp_path: Path) -> None:
    path = tmp_path / "unsafe-model.yaml"
    path.write_text(
        "model:\n"
        "  model_id: local://fixtures/tiny-dense\n"
        "  checkpoint_format: fixture\n"
        "  trust_remote_code: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="allow_remote_code"):
        load_inspection_config(path)


def test_compare_and_report_delegate_to_artifact_orchestration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = (tmp_path / "baseline").resolve()
    candidate = (tmp_path / "candidate").resolve()
    comparison = (tmp_path / "comparisons" / "comparison-id").resolve()
    report_path = comparison / "report.md"
    captured: dict[str, Any] = {}

    def compare(
        directories: tuple[Path, ...],
        *,
        output_root: Path | None,
        unsafe_overrides: set[str] | None,
    ) -> Path:
        captured["directories"] = directories
        captured["output_root"] = output_root
        captured["unsafe_overrides"] = unsafe_overrides
        return comparison

    def report(source: Path, *, destination: Path | None) -> Path:
        captured["source"] = source
        captured["destination"] = destination
        return report_path

    monkeypatch.setattr("inkling_quant_lab.cli._compare_artifacts", compare)
    monkeypatch.setattr("inkling_quant_lab.cli._report_artifact", report)

    comparison_result = runner.invoke(
        app,
        [
            "compare",
            str(baseline),
            str(candidate),
            "--unsafe-override",
            "model_revision",
            "--json",
        ],
    )
    report_result = runner.invoke(app, ["report", str(comparison), "--json"])

    assert comparison_result.exit_code == 0, comparison_result.output
    assert json.loads(comparison_result.stdout)["comparison_directory"] == str(comparison)
    assert captured["directories"] == (baseline, candidate)
    assert captured["unsafe_overrides"] == {"model_revision"}
    assert report_result.exit_code == 0, report_result.output
    assert json.loads(report_result.stdout)["report_path"] == str(report_path)
    assert captured["source"] == comparison


def test_compare_requires_at_least_two_runs(tmp_path: Path) -> None:
    result = runner.invoke(app, ["compare", str(tmp_path / "only-run"), "--json"])

    assert result.exit_code != 0
    payload = json.loads(result.stdout)
    assert payload["error"]["code"] == "CONFIGURATION_ERROR"
    assert "at least two" in payload["error"]["message"]


def test_help_lists_complete_public_command_surface() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    for command in (
        "validate",
        "run",
        "resume",
        "inspect-model",
        "compare",
        "report",
        "list-backends",
        "doctor",
    ):
        assert command in result.output


def test_help_defines_moe_and_explains_remote_code_safety() -> None:
    inspect_result = runner.invoke(app, ["inspect-model", "--help"])
    run_result = runner.invoke(app, ["run", "--help"])

    assert inspect_result.exit_code == 0, inspect_result.output
    assert "mixture-of-experts (MoE)" in inspect_result.output
    assert run_result.exit_code == 0, run_result.output
    assert "audit and pin the code" in run_result.output
