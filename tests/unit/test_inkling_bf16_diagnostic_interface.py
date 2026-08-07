"""Host-only checks for the BF16 diagnostic runtime-interface expectations.

Both checks defend a contract that an eight-B300 run already broke:

* ``/props`` reports the chat template after llama.cpp's jinja lexer normalizes
  carriage returns and drops one trailing line feed, so the runner must derive
  that exact text instead of comparing the pinned asset bytes.
* ``common_init`` in the pinned build turns on log prefixes and timestamps, so
  every llama-server line starts with ``M.SS.mmm.uuu I `` and the EOS and EOG
  patterns must match the line body.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from inkling_quant_lab.gguf.inkling_measurement_raw_evidence import (
    MeasurementResourceSampleSummary,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = PROJECT_ROOT / "scripts/run_inkling_measurement_modal.py"
SOURCE_EOS_TOKEN_ID = 200_006
COMPARISON_TOKEN_ID = 199_999


def _runner_function(name: str, namespace: dict[str, object]) -> Callable[..., object]:
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"), filename=str(RUNNER_PATH))
    definition = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    isolated = ast.Module(body=[definition], type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, str(RUNNER_PATH), "exec"), namespace)
    return cast("Callable[..., object]", namespace[name])


def _expected_server_chat_template(official: str) -> str:
    derive = _runner_function("_expected_server_chat_template", {})
    return cast("str", derive(official))


def _runtime_eog_ids(log_text: str) -> tuple[int, ...]:
    parse = _runner_function(
        "_parse_diagnostic_runtime_eog_ids",
        {"re": re, "DIAGNOSTIC_EOS_TOKEN_ID": SOURCE_EOS_TOKEN_ID},
    )
    return cast("tuple[int, ...]", parse(log_text))


def _server_log(*lines: str) -> str:
    """Render lines with the prefix that the pinned build always emits."""

    prefixed = [
        f"0.{index:02d}.{index:03d}.{index:03d} I {line}" for index, line in enumerate(lines)
    ]
    return "\n".join(prefixed) + "\n"


def _load_log(eos_ids: Sequence[int], eog_ids: Sequence[int]) -> str:
    lines = ["print_info: vocab type            = BPE"]
    lines += [f"print_info: EOS token             = {item} '<|eos|>'" for item in eos_ids]
    lines += [f"print_info: EOG token             = {item} '<|eog|>'" for item in eog_ids]
    return _server_log(*lines)


def _telemetry_window(
    telemetry: Mapping[str, object],
    *,
    started_monotonic: float,
    finished_monotonic: float,
) -> dict[str, object]:
    build = _runner_function(
        "_telemetry_window",
        {"Mapping": Mapping, "cast": cast, "RuntimeError": RuntimeError},
    )
    return cast(
        "dict[str, object]",
        build(
            telemetry,
            started_monotonic=started_monotonic,
            finished_monotonic=finished_monotonic,
        ),
    )


def _telemetry_sample(sampled_at: float) -> dict[str, object]:
    return {
        "sampled_at_monotonic_seconds": sampled_at,
        "host_rss_bytes": 4 * 1024**3,
        "gpus": [
            {"memory_used_mib": 190_000 + index, "utilization_percent": 40 + index}
            for index in range(8)
        ],
    }


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


def test_prefixed_log_yields_sorted_eog_ids() -> None:
    log_text = _load_log(
        [SOURCE_EOS_TOKEN_ID],
        [SOURCE_EOS_TOKEN_ID, COMPARISON_TOKEN_ID],
    )

    assert _runtime_eog_ids(log_text) == (COMPARISON_TOKEN_ID, SOURCE_EOS_TOKEN_ID)


def test_unexpected_eos_id_is_rejected() -> None:
    log_text = _load_log([COMPARISON_TOKEN_ID], [COMPARISON_TOKEN_ID])

    with pytest.raises(RuntimeError, match=r"EOS metadata is not exact: \(199999,\)"):
        _runtime_eog_ids(log_text)


def test_repeated_eos_line_is_rejected() -> None:
    log_text = _load_log([SOURCE_EOS_TOKEN_ID, SOURCE_EOS_TOKEN_ID], [SOURCE_EOS_TOKEN_ID])

    with pytest.raises(RuntimeError, match="EOS metadata is not exact"):
        _runtime_eog_ids(log_text)


def test_missing_eog_line_is_rejected() -> None:
    log_text = _load_log([SOURCE_EOS_TOKEN_ID], [])

    with pytest.raises(RuntimeError, match=r"EOG metadata is incomplete: \(\)"):
        _runtime_eog_ids(log_text)


def test_repeated_eog_id_is_rejected() -> None:
    log_text = _load_log([SOURCE_EOS_TOKEN_ID], [SOURCE_EOS_TOKEN_ID, SOURCE_EOS_TOKEN_ID])

    with pytest.raises(RuntimeError, match="EOG metadata is incomplete"):
        _runtime_eog_ids(log_text)


def test_match_never_crosses_a_line() -> None:
    broken = _server_log("print_info: EOS token             = 200006 '<|eos|") + _server_log(
        "print_info: EOG token             = 200006 '<|eog|>'"
    )

    with pytest.raises(RuntimeError, match=r"EOS metadata is not exact: \(\)"):
        _runtime_eog_ids(broken)


def test_telemetry_window_validates_in_python_strict_mode() -> None:
    telemetry = {"samples": [_telemetry_sample(10.0), _telemetry_sample(20.0)]}

    window = _telemetry_window(telemetry, started_monotonic=5.0, finished_monotonic=25.0)
    summary = MeasurementResourceSampleSummary.model_validate(window, strict=True)

    assert summary.sample_count == 2
    assert summary.max_sampled_per_gpu_memory_bytes == tuple(
        (190_000 + index) * 1024 * 1024 for index in range(8)
    )
    assert summary.max_sampled_per_gpu_utilization_percent == tuple(
        float(40 + index) for index in range(8)
    )
