"""Host-only checks for the BF16 diagnostic chat-template expectation.

The eight-B300 diagnostic compares the ``/props`` chat template against the
pinned official asset.  llama.cpp's jinja lexer stores the template after it
normalizes carriage returns and drops one trailing line feed, so the runner must
derive that exact runtime text instead of comparing the asset bytes directly.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = PROJECT_ROOT / "scripts/run_inkling_measurement_modal.py"


def _expected_server_chat_template(official: str) -> str:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
    derivation = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_expected_server_chat_template"
    )
    isolated = ast.Module(body=[derivation], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace: dict[str, object] = {}
    exec(compile(isolated, str(RUNNER_PATH), "exec"), namespace)
    derive = cast("Callable[[str], str]", namespace["_expected_server_chat_template"])
    return derive(official)


def test_single_trailing_line_feed_is_dropped() -> None:
    assert _expected_server_chat_template("{%- if true -%}\n{{- 'x' -}}\n{%- endif -%}\n") == (
        "{%- if true -%}\n{{- 'x' -}}\n{%- endif -%}"
    )


def test_only_one_trailing_line_feed_is_dropped() -> None:
    assert _expected_server_chat_template("{{- 'x' -}}\n\n") == "{{- 'x' -}}\n"


def test_template_without_trailing_line_feed_is_unchanged() -> None:
    assert _expected_server_chat_template("{{- 'x' -}}") == "{{- 'x' -}}"


def test_carriage_return_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="carriage return"):
        _expected_server_chat_template("{{- 'x' -}}\r\n")
