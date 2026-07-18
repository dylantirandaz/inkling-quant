"""Release coverage-gate verifier tests."""

from __future__ import annotations

import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

from scripts import check_coverage

pytestmark = pytest.mark.unit


def _file(
    *,
    covered_lines: int = 95,
    statements: int = 100,
    covered_branches: int = 10,
    branches: int = 10,
) -> dict[str, object]:
    return {
        "summary": {
            "covered_lines": covered_lines,
            "num_statements": statements,
            "covered_branches": covered_branches,
            "num_branches": branches,
            # Deliberately misleading: the gate must compute line coverage from counts.
            "percent_covered": 100.0,
        },
        "missing_lines": list(range(covered_lines + 1, statements + 1)),
        "missing_branches": [[line, line + 1] for line in range(covered_branches, branches)],
    }


def _passing_report() -> dict[str, object]:
    return {
        "meta": {"format": 3, "branch_coverage": True},
        "totals": {
            "covered_lines": 850,
            "num_statements": 1000,
            "covered_branches": 10,
            "num_branches": 10,
            "percent_covered": 100.0,
        },
        "files": {
            "src/inkling_quant_lab/quantization/policies.py": _file(),
            "src/inkling_quant_lab/manifests.py": _file(),
            "src/inkling_quant_lab/routing/metrics.py": _file(),
            "/checkout/src/inkling_quant_lab/comparison.py": _file(),
            "src\\inkling_quant_lab\\security.py": _file(
                covered_lines=20,
                statements=20,
                covered_branches=95,
                branches=100,
            ),
        },
    }


def _write_report(tmp_path: Path, report: dict[str, object]) -> Path:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _run(path: Path) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        status = check_coverage.main([str(path)])
    return status, stdout.getvalue(), stderr.getvalue()


def test_gate_accepts_exact_thresholds_and_reports_each_requirement(tmp_path: Path) -> None:
    """Exactly 85/95 percent passes and every required file is named."""

    path = _write_report(tmp_path, _passing_report())

    status, output, _ = _run(path)

    assert status == 0
    assert "PASS overall line coverage: 85.00%" in output
    assert "quantization/policies.py" in output
    assert "manifests.py" in output
    assert "routing/metrics.py" in output
    assert "comparison.py" in output
    assert "PASS security.py branch coverage: 95.00%" in output
    assert "Coverage quality gate passed" in output


def test_gate_uses_line_counts_not_combined_percent_field(tmp_path: Path) -> None:
    """A misleading Coverage.py combined percentage cannot mask low line coverage."""

    report = _passing_report()
    totals = report["totals"]
    assert isinstance(totals, dict)
    totals["covered_lines"] = 849
    path = _write_report(tmp_path, report)

    status, output, _ = _run(path)

    assert status == 1
    assert "FAIL overall line coverage: 84.90%" in output
    assert "requires >= 85.00%" in output


def test_gate_lists_low_target_file_and_missing_lines(tmp_path: Path) -> None:
    """A high-risk module failure includes the file and concrete uncovered lines."""

    report = _passing_report()
    files = report["files"]
    assert isinstance(files, dict)
    files["src/inkling_quant_lab/quantization/policies.py"] = _file(covered_lines=94)
    path = _write_report(tmp_path, report)

    status, output, _ = _run(path)

    assert status == 1
    assert "FAIL quantization/policies.py line coverage: 94.00%" in output
    assert "missing lines: 95, 96, 97, 98, 99, 100" in output


def test_gate_lists_security_branch_shortfall_and_missing_arcs(tmp_path: Path) -> None:
    """Security branch failures identify uncovered control-flow arcs."""

    report = _passing_report()
    files = report["files"]
    assert isinstance(files, dict)
    files["src\\inkling_quant_lab\\security.py"] = _file(
        covered_lines=20,
        statements=20,
        covered_branches=94,
        branches=100,
    )
    path = _write_report(tmp_path, report)

    status, output, _ = _run(path)

    assert status == 1
    assert "FAIL security.py branch coverage: 94.00%" in output
    assert "missing branches: 94->95" in output


def test_gate_fails_when_branch_instrumentation_or_target_file_is_missing(
    tmp_path: Path,
) -> None:
    """Missing evidence is a failure, never an implicit 100 percent."""

    report = _passing_report()
    meta = report["meta"]
    files = report["files"]
    assert isinstance(meta, dict)
    assert isinstance(files, dict)
    meta["branch_coverage"] = False
    del files["src/inkling_quant_lab/manifests.py"]
    path = _write_report(tmp_path, report)

    status, output, _ = _run(path)

    assert status == 1
    assert "FAIL manifests.py line coverage: file is absent" in output
    assert (
        "FAIL security.py branch coverage: coverage.json was not generated with --cov-branch"
        in output
    )


def test_gate_rejects_malformed_report_with_generation_hint(tmp_path: Path) -> None:
    """Invalid JSON/schema errors include the exact recovery command."""

    path = tmp_path / "coverage.json"
    path.write_text("[]", encoding="utf-8")

    status, _, error = _run(path)

    assert status == 2
    assert "coverage report root must be an object" in error
    assert "uv run pytest" in error
    assert "--cov-report=json:coverage.json" in error


def test_documentation_contains_coverage_json_command() -> None:
    """The script itself documents how CI and developers generate its input."""

    assert check_coverage.__doc__ is not None
    assert "--cov-report=json:coverage.json" in check_coverage.__doc__
    assert '-m "not network and not gpu and not slow and not large_model"' in (
        check_coverage.__doc__
    )
