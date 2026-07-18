"""Enforce Inkling Quant Lab's release coverage thresholds.

Generate the branch-aware Coverage.py JSON input with:

    uv run pytest -m "not network and not gpu and not slow and not large_model" \
      --cov=inkling_quant_lab --cov-branch --cov-report=term-missing \
      --cov-report=json:coverage.json --cov-fail-under=0

Then run ``uv run python scripts/check_coverage.py coverage.json``. The verifier
computes line and branch percentages from covered/total counts rather than using
Coverage.py's combined ``percent_covered`` field.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, cast

OVERALL_LINE_THRESHOLD = 85.0
CRITICAL_LINE_THRESHOLD = 95.0
SECURITY_BRANCH_THRESHOLD = 95.0

CRITICAL_LINE_FILES = (
    "inkling_quant_lab/quantization/policies.py",
    "inkling_quant_lab/manifests.py",
    "inkling_quant_lab/routing/metrics.py",
    "inkling_quant_lab/comparison.py",
)
SECURITY_FILE = "inkling_quant_lab/security.py"

GENERATION_COMMAND = (
    'uv run pytest -m "not network and not gpu and not slow and not large_model" '
    "--cov=inkling_quant_lab --cov-branch --cov-report=term-missing "
    "--cov-report=json:coverage.json --cov-fail-under=0"
)


class CoverageDataError(ValueError):
    """The supplied coverage report is absent, ambiguous, or malformed."""


@dataclass(frozen=True, slots=True)
class GateResult:
    """One threshold check and optional actionable missing-location detail."""

    label: str
    metric: str
    passed: bool
    summary: str
    detail: str | None = None

    def render(self) -> str:
        """Render a stable human-readable CI diagnostic."""

        prefix = "PASS" if self.passed else "FAIL"
        first = f"{prefix} {self.label} {self.metric} coverage: {self.summary}"
        return first if self.detail is None else f"{first}\n  {self.detail}"


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CoverageDataError(f"{context} must be an object with string keys")
    return cast(Mapping[str, object], value)


def _integer(mapping: Mapping[str, object], key: str, context: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CoverageDataError(f"{context}.{key} must be a non-negative integer")
    return value


def _percentage(covered: int, total: int) -> float:
    return covered / total * 100.0


def _metric_result(
    *,
    label: str,
    metric: str,
    covered: int,
    total: int,
    required: float,
    missing_detail: str | None = None,
) -> GateResult:
    if total == 0:
        return GateResult(
            label=label,
            metric=metric,
            passed=False,
            summary=f"no measurable {metric}s (requires >= {required:.2f}%)",
            detail="Coverage evidence is empty; ensure the target module is imported and tested.",
        )
    if covered > total:
        raise CoverageDataError(
            f"{label} has impossible {metric} counts: covered={covered}, total={total}"
        )
    percentage = _percentage(covered, total)
    return GateResult(
        label=label,
        metric=metric,
        passed=percentage >= required,
        summary=f"{percentage:.2f}% ({covered}/{total}; requires >= {required:.2f}%)",
        detail=missing_detail if percentage < required else None,
    )


def _normalized_path(value: str) -> str:
    return value.replace("\\", "/").rstrip("/")


def _find_file(files: Mapping[str, object], suffix: str) -> tuple[str, Mapping[str, object]] | None:
    normalized_suffix = _normalized_path(suffix)
    matches = [
        (path, value)
        for path, value in files.items()
        if (
            _normalized_path(path) == normalized_suffix
            or _normalized_path(path).endswith(f"/{normalized_suffix}")
        )
    ]
    if not matches:
        return None
    if len(matches) > 1:
        paths = ", ".join(sorted(path for path, _ in matches))
        raise CoverageDataError(f"multiple coverage entries match {suffix}: {paths}")
    path, value = matches[0]
    return path, _mapping(value, f"files[{path!r}]")


def _missing_lines(entry: Mapping[str, object], context: str) -> str | None:
    raw = entry.get("missing_lines", [])
    if not isinstance(raw, list) or any(
        not isinstance(line, int) or isinstance(line, bool) or line < 1 for line in raw
    ):
        raise CoverageDataError(f"{context}.missing_lines must be a list of positive integers")
    if not raw:
        return None
    displayed = ", ".join(str(line) for line in raw[:20])
    suffix = f" … and {len(raw) - 20} more" if len(raw) > 20 else ""
    return f"missing lines: {displayed}{suffix}"


def _missing_branches(entry: Mapping[str, object], context: str) -> str | None:
    raw = entry.get("missing_branches", [])
    if not isinstance(raw, list):
        raise CoverageDataError(f"{context}.missing_branches must be a list")
    arcs: list[str] = []
    for index, branch in enumerate(raw):
        if (
            not isinstance(branch, list)
            or len(branch) != 2
            or any(not isinstance(line, int) or isinstance(line, bool) for line in branch)
        ):
            raise CoverageDataError(
                f"{context}.missing_branches[{index}] must be a two-integer arc"
            )
        arcs.append(f"{branch[0]}->{branch[1]}")
    if not arcs:
        return None
    displayed = ", ".join(arcs[:20])
    suffix = f" … and {len(arcs) - 20} more" if len(arcs) > 20 else ""
    return f"missing branches: {displayed}{suffix}"


def _line_result(
    label: str,
    entry: Mapping[str, object],
    required: float,
    context: str,
) -> GateResult:
    summary = _mapping(entry.get("summary"), f"{context}.summary")
    return _metric_result(
        label=label,
        metric="line",
        covered=_integer(summary, "covered_lines", f"{context}.summary"),
        total=_integer(summary, "num_statements", f"{context}.summary"),
        required=required,
        missing_detail=_missing_lines(entry, context),
    )


def _branch_result(
    label: str,
    entry: Mapping[str, object],
    required: float,
    context: str,
) -> GateResult:
    summary = _mapping(entry.get("summary"), f"{context}.summary")
    return _metric_result(
        label=label,
        metric="branch",
        covered=_integer(summary, "covered_branches", f"{context}.summary"),
        total=_integer(summary, "num_branches", f"{context}.summary"),
        required=required,
        missing_detail=_missing_branches(entry, context),
    )


def evaluate_report(report: Mapping[str, object]) -> tuple[GateResult, ...]:
    """Evaluate all release thresholds and return stable diagnostics."""

    totals = _mapping(report.get("totals"), "totals")
    files = _mapping(report.get("files"), "files")
    meta = _mapping(report.get("meta"), "meta")
    results = [
        _metric_result(
            label="overall",
            metric="line",
            covered=_integer(totals, "covered_lines", "totals"),
            total=_integer(totals, "num_statements", "totals"),
            required=OVERALL_LINE_THRESHOLD,
        )
    ]
    for suffix in CRITICAL_LINE_FILES:
        label = suffix.removeprefix("inkling_quant_lab/")
        matched = _find_file(files, suffix)
        if matched is None:
            results.append(
                GateResult(
                    label=label,
                    metric="line",
                    passed=False,
                    summary=f"file is absent from coverage.json (expected suffix {suffix})",
                )
            )
            continue
        path, entry = matched
        results.append(_line_result(label, entry, CRITICAL_LINE_THRESHOLD, f"files[{path!r}]"))
    security = _find_file(files, SECURITY_FILE)
    if meta.get("branch_coverage") is not True:
        results.append(
            GateResult(
                label="security.py",
                metric="branch",
                passed=False,
                summary="coverage.json was not generated with --cov-branch",
                detail=f"Regenerate it with: {GENERATION_COMMAND}",
            )
        )
    elif security is None:
        results.append(
            GateResult(
                label="security.py",
                metric="branch",
                passed=False,
                summary=f"file is absent from coverage.json (expected suffix {SECURITY_FILE})",
            )
        )
    else:
        path, entry = security
        results.append(
            _branch_result(
                "security.py",
                entry,
                SECURITY_BRANCH_THRESHOLD,
                f"files[{path!r}]",
            )
        )
    return tuple(results)


def load_report(path: Path) -> Mapping[str, object]:
    """Load and minimally validate a Coverage.py JSON report."""

    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CoverageDataError(f"unable to read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise CoverageDataError(f"invalid JSON in {path}: {error}") from error
    return _mapping(raw, "coverage report root")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enforce overall, critical-module, and security branch coverage thresholds.",
        epilog=f"Generate the input with: {GENERATION_COMMAND}",
    )
    parser.add_argument(
        "coverage_json",
        nargs="?",
        default="coverage.json",
        type=Path,
        help="Coverage.py JSON report (default: coverage.json)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the gate, returning 0 for pass, 1 for thresholds, or 2 for bad evidence."""

    arguments = _parser().parse_args(argv)
    try:
        results = evaluate_report(load_report(arguments.coverage_json))
    except CoverageDataError as error:
        print(f"Coverage quality gate could not evaluate evidence: {error}", file=sys.stderr)
        print(f"Regenerate coverage JSON with: {GENERATION_COMMAND}", file=sys.stderr)
        return 2
    for result in results:
        print(result.render())
    failures = [result for result in results if not result.passed]
    if failures:
        print(f"Coverage quality gate failed: {len(failures)} requirement(s) unmet.")
        return 1
    print("Coverage quality gate passed.")
    return 0


def _exit(status: int) -> NoReturn:
    raise SystemExit(status)


if __name__ == "__main__":
    _exit(main())
