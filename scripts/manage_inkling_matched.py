"""Inspect the checked Inkling BF16/Q3 plan without starting remote work."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

# Direct CLI use must not create package bytecode in the project tree.
if __name__ == "__main__":
    sys.dont_write_bytecode = True

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
SRC_ROOT: Final = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inkling_quant_lab.exceptions import InklingQuantError  # noqa: E402
from inkling_quant_lab.gguf.inkling_matched_preflight import (  # noqa: E402
    InklingMatchedPreflightReport,
    build_matched_preflight_report,
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _add_project_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root that contains the checked local control files.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the checked Inkling BF16/Q3 plan. "
            "This command does not read remote artifacts or start compute."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    preflight = commands.add_parser(
        "preflight",
        help="Validate local controls and compile the launch-disabled plan.",
    )
    _add_project_root_argument(preflight)
    preflight.add_argument(
        "--json",
        action="store_true",
        help="Print one canonical report after argument validation.",
    )

    inspect = commands.add_parser(
        "inspect",
        help="Print a short summary of the launch-disabled plan.",
    )
    _add_project_root_argument(inspect)
    return parser


def _print_summary(report: InklingMatchedPreflightReport) -> None:
    print(f"Status: {report.status}")
    print(f"Plan SHA-256: {report.plan_sha256}")
    print(f"Model: {report.model_id}@{report.revision}")
    print(f"Execution record: {report.execution.record_status}")
    print("Local control files: verified")
    print("Remote artifact bytes: not read")
    print("Hardware: not probed")
    print("Paid compute: not started")
    print("Next required stage: rehash_bf16_subject (not_executed)")


def main(argv: Sequence[str] | None = None) -> int:
    """Run one read-only matched-plan command."""

    arguments = _build_parser().parse_args(argv)
    try:
        report = build_matched_preflight_report(arguments.project_root)
    except InklingQuantError as error:
        if arguments.command == "preflight" and arguments.json:
            print(
                _canonical_json({"error": error.as_dict(), "status": "invalid"}),
                file=sys.stderr,
            )
        else:
            print(f"Error: {error.message}", file=sys.stderr)
        return 1

    if arguments.command == "preflight" and arguments.json:
        print(report.canonical_json())
    else:
        _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
