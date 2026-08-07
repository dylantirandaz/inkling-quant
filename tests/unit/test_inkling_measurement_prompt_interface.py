"""Host-only checks for the measurement diagnostic prompt interfaces.

The published raw-interface run scored the BF16 control below its accuracy
floors. The eight-B300 interface diagnostic showed that the raw prompt, not the
weights, caused that result, so the measurement now renders the official chat
template. Both interfaces must keep an exact, reviewed prompt shape.

llama.cpp expands the media marker in place: mtmd writes ``<|content_image|>``
before an image embedding, and ``<|content_audio_input|>`` before an audio
embedding with ``<|audio_end|>`` after it. The chat render therefore leaves the
marker alone inside its own user block.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inkling_quant_lab.gguf.inkling_measurement import (
    MEASUREMENT_MEDIA_MARKER,
    MeasurementPromptInterface,
    load_measurement_bundle,
    render_measurement_diagnostic_prompt,
)

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_INTERFACE: MeasurementPromptInterface = "raw_instruction_then_lf_then_item_prompt"
CHAT_INTERFACE: MeasurementPromptInterface = (
    "chat_template_system_effort_none_then_user_then_generation_prompt"
)
CHAT_SYSTEM_TURNS = (
    "<|message_system|><|content_text|>T<|end_message|>"
    "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
)
CHAT_ITEM_TURN = "<|message_user|><|content_text|>Q<|end_message|>"


def test_raw_interface_joins_instruction_and_item_with_one_line_feed() -> None:
    render = render_measurement_diagnostic_prompt(
        prompt_template="T",
        item_prompt="Q",
        prompt_interface=RAW_INTERFACE,
        has_media=False,
    )

    assert render.prompt_text == "T\nQ"
    assert render.prompt_string == render.prompt_text


def test_raw_interface_puts_media_before_the_instruction() -> None:
    render = render_measurement_diagnostic_prompt(
        prompt_template="T",
        item_prompt="Q",
        prompt_interface=RAW_INTERFACE,
        has_media=True,
    )

    assert render.prompt_text == "T\nQ"
    assert render.prompt_string == f"{MEASUREMENT_MEDIA_MARKER}\nT\nQ"


def test_chat_interface_renders_the_official_text_turns() -> None:
    render = render_measurement_diagnostic_prompt(
        prompt_template="T",
        item_prompt="Q",
        prompt_interface=CHAT_INTERFACE,
        has_media=False,
    )

    assert render.prompt_text == f"{CHAT_SYSTEM_TURNS}{CHAT_ITEM_TURN}<|message_model|>"
    assert render.prompt_string == render.prompt_text


def test_chat_interface_gives_media_its_own_user_turn() -> None:
    render = render_measurement_diagnostic_prompt(
        prompt_template="T",
        item_prompt="Q",
        prompt_interface=CHAT_INTERFACE,
        has_media=True,
    )

    assert render.prompt_text == f"{CHAT_SYSTEM_TURNS}{CHAT_ITEM_TURN}<|message_model|>"
    assert render.prompt_string == (
        f"{CHAT_SYSTEM_TURNS}"
        f"<|message_user|>{MEASUREMENT_MEDIA_MARKER}<|end_message|>"
        f"{CHAT_ITEM_TURN}"
        "<|message_model|>"
    )


def test_chat_interface_never_wraps_the_marker_in_a_content_token() -> None:
    render = render_measurement_diagnostic_prompt(
        prompt_template="T",
        item_prompt="Q",
        prompt_interface=CHAT_INTERFACE,
        has_media=True,
    )

    assert f"<|content_image|>{MEASUREMENT_MEDIA_MARKER}" not in render.prompt_string
    assert f"<|content_audio_input|>{MEASUREMENT_MEDIA_MARKER}" not in render.prompt_string
    assert "<|unused_200054|>" not in render.prompt_string
    assert "<|unused_200053|>" not in render.prompt_string


def test_checked_config_renders_every_item_through_the_chat_interface() -> None:
    bundle = load_measurement_bundle(PROJECT_ROOT)
    quality = bundle.config.quality

    assert quality.prompt_interface == CHAT_INTERFACE

    for item in bundle.diagnostic_items:
        render = render_measurement_diagnostic_prompt(
            prompt_template=quality.prompt_template,
            item_prompt=item.prompt,
            prompt_interface=quality.prompt_interface,
            has_media=item.fixture is not None,
        )
        assert render.prompt_text.startswith("<|message_system|><|content_text|>")
        assert render.prompt_text.endswith("<|message_model|>")
        assert render.prompt_string.endswith(f"{item.prompt}<|end_message|><|message_model|>")
        assert (MEASUREMENT_MEDIA_MARKER in render.prompt_string) == (item.fixture is not None)
